"""Marginal-negative construction for neural ratio estimation.

An NRE classifier separates JOINT pairs ``(x_i, theta_i)`` -- an observation with the
hypothesis that generated it -- from MARGINAL pairs ``(x_i, theta_j)`` with ``theta_j``
drawn independently. How the marginal partner is drawn is not a detail: it *defines* the
denominator of the ratio the network learns.

The historical loop hardcoded one construction -- permute the hypothesis rows within the
batch (:class:`RowPermutation`). That is correct exactly when one row is one independent
draw of theta. It is WRONG whenever rows are GROUPED, i.e. many rows share a hypothesis:
two rows of the same group can be paired with each other, leaving the group-level
hypothesis components completely unperturbed while only the per-row components move. The
"negative" is then not a marginal draw at all, and the network is trained against a
denominator that quietly interpolates between p(x) and p(x|theta).

Concretely, for the muon program a row is one ground muon and a group is one air shower:
(log10E, cos_zen, h1) are shared by every muon of a shower, while (r, psi) vary per muon.
At batch 65536 against ~10^4 showers essentially every shower appears in every batch, so
same-shower collisions are not a tail event -- they are the common case.

Two group-aware constructions are provided, and they are NOT interchangeable:

* :func:`group_derangement_rows` -- host/numpy. Draw a derangement over the DISTINCT
  groups present in the batch, then one representative row from each row's assigned
  partner group. All rows of one source group therefore share a target group.
* :func:`group_shift_rows` -- device/jit. Draw an independent uniform ``shift in [1, K-1]``
  per row over a contiguous group-label space and take ``(label + shift) mod K``, then a
  uniform row within it. No data-dependent shapes (no ``np.unique``), so it traces.

Both guarantee the negative never comes from the row's own group. They differ in the
dependence structure ACROSS rows of one group (shared target vs independent targets), so
they do not agree seed-for-seed; validate a switch on quality (val loss / AUC), never on
bit-equality. The independent draw is the cleaner marginal; the shared draw is cheaper on
host.
"""

from typing import NamedTuple, Protocol, runtime_checkable

import jax
import jax.numpy as jnp
import numpy as np


class GroupIndex(NamedTuple):
    """Row<->group lookup tables, built ONCE per split (never per step).

    ``label`` relabels ``group_id`` into a dense contiguous ``[0, n_groups)`` space; the
    remaining arrays let a uniform draw inside any group be a pure gather:
    ``order[start[g] + u]`` with ``0 <= u < count[g]``.

    ``order`` is exactly the stable argsort of ``label``, so rows of one group occupy one
    contiguous run. Because ``np.unique`` sorts, relabelling is monotone in ``group_id``
    and the argsort is shared -- deriving the label-indexed tables from the id-indexed
    ones costs no second O(N log N) pass.
    """

    label: np.ndarray      # (N,) int32, dense group label per row
    order: np.ndarray      # (N,) int32, row positions grouped by label
    start: np.ndarray      # (K,) int32, offset of each label's run in `order`
    count: np.ndarray      # (K,) int32, rows per label
    n_groups: int

    @property
    def n_rows(self) -> int:
        return int(self.label.shape[0])

    def to_device(self) -> "GroupIndex":
        return GroupIndex(
            label=jnp.asarray(self.label), order=jnp.asarray(self.order),
            start=jnp.asarray(self.start), count=jnp.asarray(self.count),
            n_groups=self.n_groups,
        )


def build_group_index(group_id) -> GroupIndex:
    """``group_id`` (N,) of arbitrary integer labels -> :class:`GroupIndex`."""
    group_id = np.asarray(group_id)
    if group_id.ndim != 1:
        raise ValueError(f"group_id must be 1-D, got shape {group_id.shape}")
    n = int(group_id.shape[0])
    if n == 0:
        z = np.zeros(0, dtype=np.int32)
        return GroupIndex(label=z, order=z, start=np.zeros(1, np.int32),
                          count=np.zeros(1, np.int32), n_groups=0)
    uniq, label = np.unique(group_id, return_inverse=True)
    label = np.asarray(label, dtype=np.int32).reshape(-1)
    order = np.argsort(label, kind="stable").astype(np.int32)
    sorted_label = label[order]
    _u, start = np.unique(sorted_label, return_index=True)
    end = np.append(start[1:], n)
    return GroupIndex(
        label=label, order=order,
        start=np.asarray(start, dtype=np.int32),
        count=np.asarray(end - start, dtype=np.int32),
        n_groups=int(uniq.shape[0]),
    )


# ---------------------------------------------------------------------------
# the protocol
# ---------------------------------------------------------------------------
@runtime_checkable
class NegativePairing(Protocol):
    """``(key, rows, index) -> negative_rows``, same shape as ``rows``.

    ``rows`` are positions into the split's arrays; the return value is positions whose
    THETA becomes the marginal partner for each entry of ``rows``. Implementations must be
    pure and, if they are to run inside the training step, jit-traceable.
    """

    #: whether ``__call__`` traces under jit (device path) or must run on host
    jittable: bool

    def __call__(self, key, rows, index: GroupIndex): ...


# ---------------------------------------------------------------------------
# implementations
# ---------------------------------------------------------------------------
def row_permutation_rows(key, rows, index: GroupIndex):
    """Permute ``rows`` within the batch. Correct only when one row is one theta draw."""
    del index
    return jnp.take(rows, jax.random.permutation(key, rows.shape[0]), axis=0)


def group_shift_rows(key, rows, index: GroupIndex):
    """Independent uniform different-group partner per row (jit-traceable).

    ``shift ~ U{1, ..., K-1}`` is nonzero by construction, so ``(label + shift) mod K``
    is always a different group -- no rejection loop, no data-dependent shape.
    """
    k = int(index.n_groups)
    if k < 2:
        raise ValueError(
            f"group-aware negatives need >= 2 distinct groups in the split, got {k}. "
            f"With one group every 'marginal' pair would be a joint pair.")
    lbl = jnp.take(index.label, rows, axis=0)
    k_shift, k_off = jax.random.split(key)
    shift = jax.random.randint(k_shift, lbl.shape, 1, k)
    neg_lbl = jnp.mod(lbl + shift, k)
    off = jax.random.randint(k_off, lbl.shape, 0, index.count[neg_lbl])
    return jnp.take(index.order, index.start[neg_lbl] + off, axis=0)


def group_shift_rows_np(rng, rows, index: GroupIndex) -> np.ndarray:
    """Numpy twin of :func:`group_shift_rows` -- identical construction, host arrays.

    Needed because a host-resident dataset must draw its negatives on the host: calling the
    jax version there would device-put the whole ``label``/``order`` table (10^8 rows) on
    every step purely to index it. Same distribution, same guarantee, different execution
    place -- NOT bit-identical for a given seed, since the RNG streams differ.
    """
    k = int(index.n_groups)
    if k < 2:
        raise ValueError(
            f"group-aware negatives need >= 2 distinct groups in the split, got {k}. "
            f"With one group every 'marginal' pair would be a joint pair.")
    rows = np.asarray(rows)
    lbl = np.asarray(index.label)[rows]
    shift = rng.integers(1, k, size=lbl.shape)
    neg_lbl = (lbl + shift) % k
    off = rng.integers(0, np.asarray(index.count)[neg_lbl])
    return np.asarray(index.order)[np.asarray(index.start)[neg_lbl] + off]


def row_permutation_rows_np(rng, rows, index: GroupIndex) -> np.ndarray:
    """Numpy twin of :func:`row_permutation_rows`."""
    del index
    rows = np.asarray(rows)
    return rows[rng.permutation(rows.shape[0])]


def random_derangement(n: int, rng: np.random.Generator) -> np.ndarray:
    """A uniformly random permutation of ``range(n)`` with no fixed points.

    Rejection sampling: P(a random permutation is a derangement) -> 1/e, so the expected
    number of draws is ~e REGARDLESS of n. ``n <= 1`` has no derangement and returns the
    identity (a degenerate batch the caller should not be relying on anyway).
    """
    if n <= 1:
        return np.arange(n, dtype=np.int64)
    ar = np.arange(n)
    while True:
        perm = rng.permutation(n)
        if not np.any(perm == ar):
            return perm


def group_derangement_rows(rng, rows, index: GroupIndex) -> np.ndarray:
    """Shared different-group partner per source group (host/numpy).

    Derange the distinct groups PRESENT IN THIS BATCH, then draw one uniform
    representative row from each row's assigned partner group. Vectorized: ``np.unique``
    for the present groups, one batched ``rng.integers`` with a per-element ``high``, and
    three gathers -- no Python loop over groups.
    """
    rows = np.asarray(rows)
    lbl = np.asarray(index.label)[rows]
    uniq, inv = np.unique(lbl, return_inverse=True)
    inv = inv.reshape(-1)
    if uniq.shape[0] < 2:
        raise ValueError(
            f"group-aware negatives need >= 2 distinct groups in the BATCH, got "
            f"{uniq.shape[0]}. Increase the batch size or check the group ids.")
    d = random_derangement(uniq.shape[0], rng)
    target = uniq[d][inv]
    off = rng.integers(0, np.asarray(index.count)[target])
    return np.asarray(index.order)[np.asarray(index.start)[target] + off]


class RowPermutation:
    """Legacy within-batch row permutation. The pre-split ``hitman.train`` behaviour."""

    jittable = True
    group_aware = False
    name = "row_permutation"

    def __call__(self, key, rows, index: GroupIndex):
        return row_permutation_rows(key, rows, index)

    #: host twin, for a host-resident dataset (see group_shift_rows_np)
    numpy = staticmethod(row_permutation_rows_np)


class GroupShift:
    """Independent uniform different-group partner per row. The device/jit default."""

    jittable = True
    group_aware = True
    name = "group_shift"

    def __call__(self, key, rows, index: GroupIndex):
        return group_shift_rows(key, rows, index)

    numpy = staticmethod(group_shift_rows_np)


class GroupDerangement:
    """Shared different-group partner per source group. Host only (uses ``np.unique``)."""

    jittable = False
    group_aware = True
    name = "group_derangement"

    def __call__(self, rng, rows, index: GroupIndex):
        return group_derangement_rows(rng, rows, index)

    numpy = staticmethod(group_derangement_rows)


PAIRINGS = {p.name: p for p in (RowPermutation, GroupShift, GroupDerangement)}


def get_pairing(name_or_obj):
    """Resolve a pairing by name (``PAIRINGS``) or pass an instance through."""
    if isinstance(name_or_obj, str):
        try:
            return PAIRINGS[name_or_obj]()
        except KeyError as exc:
            raise ValueError(
                f"unknown pairing {name_or_obj!r}; choose from {sorted(PAIRINGS)}") from exc
    return name_or_obj
