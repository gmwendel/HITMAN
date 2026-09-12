"""Device-resident training: the whole dataset lives on the accelerator.

The GPU profile (design report, Addendum 3) showed streaming training is input-bound:
the device step takes ~6 ms but host gather + PCIe staging ~11 ms. In the compressed
representation — per-hit (pmt_id, t, event_id) plus the per-event hypothesis table and
the per-sensor geometry table — the full 1M-event dataset is ~1.2 GB, so it fits on
the device outright. Batches are then formed *inside* the jitted step by device-side
gathers (the geometry lookup costs nothing; tables live in cache), and the host input
pipeline disappears.

Use ``fit_resident`` when the dataset fits in device memory (``DeviceData.nbytes`` to
check); ``hitman.train.fit`` remains the streaming path for larger-than-VRAM stores.
"""

import os
import time
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from hitman.train.loop import FitResult, _batch_loss, _unpack_batch


class DeviceData(NamedTuple):
    """Full dataset as device arrays, hits in compressed (sensor, time) form.

    Per hit: ``t`` (n_hits,) f32, ``pmt_id`` (n_hits,) i32, ``event_id`` (n_hits,) i32.
    Per event: ``hyp`` (n_events, 7) f32, ``charge`` (n_events, 2) f32.
    Tables: ``pmt_pos`` (n_pmts, 3) f32.
    """

    t: jnp.ndarray
    pmt_id: jnp.ndarray
    event_id: jnp.ndarray
    hyp: jnp.ndarray
    charge: jnp.ndarray
    pmt_pos: jnp.ndarray
    pmt_dir: jnp.ndarray = None
    pmt_type: jnp.ndarray = None

    @property
    def n_hits(self) -> int:
        return self.t.shape[0]

    @property
    def n_events(self) -> int:
        return self.hyp.shape[0]

    @property
    def nbytes(self) -> int:
        return sum(a.nbytes for a in self)

    @classmethod
    def from_store(cls, store) -> "DeviceData":
        """Load a HitStore onto the default device (one streaming read of the memmaps)."""
        return cls(
            t=jnp.asarray(np.ascontiguousarray(store.hits[:, 3]), jnp.float32),
            pmt_id=jnp.asarray(store.pmt_id, jnp.int32),
            event_id=jnp.asarray(store.event_id, jnp.int32),
            hyp=jnp.asarray(store.hyp, jnp.float32),
            charge=jnp.asarray(store.charge, jnp.float32),
            pmt_pos=jnp.asarray(store.pmt_pos, jnp.float32),
            pmt_dir=(None if store.pmt_dir is None
                     else jnp.asarray(store.pmt_dir, jnp.float32)),
            pmt_type=(None if store.pmt_type is None
                      else jnp.asarray(store.pmt_type, jnp.int32)),
        )

    @classmethod
    def from_batch(cls, batch, pmt_pos) -> "DeviceData":
        """Build from an in-RAM EventBatch (requires batch.pmt_id)."""
        if batch.pmt_id is None:
            raise ValueError("EventBatch has no pmt_id; re-extract with the current schema")
        return cls(
            t=jnp.asarray(np.asarray(batch.hits)[:, 3], jnp.float32),
            pmt_id=jnp.asarray(batch.pmt_id, jnp.int32),
            event_id=jnp.asarray(batch.event_id, jnp.int32),
            hyp=jnp.asarray(batch.hyp, jnp.float32),
            charge=jnp.asarray(batch.charge, jnp.float32),
            pmt_pos=jnp.asarray(pmt_pos, jnp.float32),
        )


def _event_shifts(key, ev, time_sigma):
    """Per-row N(0, sigma) time shift keyed on (key, event_id), counter-style.

    O(batch) work instead of drawing normals for EVERY event in the dataset per step
    (audit finding 5: a 5M-event draw is 20 MB of randoms per make_batch call, and
    exact_polish re-evaluates make_batch ~380x per loss + again under remat). fold_in
    per row keeps the coherent-per-event contract: same (key, event) -> same shift.
    NOTE: changes the realized augmentation stream (statistics identical) — do not
    flip mid-campaign on a resumable run.
    """
    return time_sigma * jax.vmap(
        lambda e: jax.random.normal(jax.random.fold_in(key, e)))(ev)


def hit_batch(data: DeviceData, rows: jnp.ndarray, key=None, time_sigma: float = 50.0):
    """Row indices -> (hit obs (B,4), hyp (B,7)), all gathers on device.

    When ``key`` is given, applies the 1.x time-shuffle augmentation: each event's
    time origin shifts by N(0, time_sigma) ns coherently in its hit times and its
    hypothesis time. Every training hypothesis has t = 0 (the extractor convention),
    so WITHOUT this the classifier never sees theta_t variation and its theta_t
    dependence is unconstrained extrapolation — the joint pair keeps dt invariant
    while marginal pairs get mismatched shifts, which is exactly the contrast that
    teaches the dt physics. (Omitting it produced a -5 ns / 115 sigma score-identity
    violation and reconstruction collapse; see run1 validation log.)
    """
    ev = data.event_id[rows]
    t = data.t[rows]
    hyp = data.hyp[ev]
    if key is not None and time_sigma > 0:
        sh = _event_shifts(key, ev, time_sigma)
        t = t + sh
        hyp = hyp.at[:, 5].add(sh)
    obs = jnp.concatenate([data.pmt_pos[data.pmt_id[rows]], t[:, None]], axis=1)
    return obs, hyp


def frame_hit_batch(data: DeviceData, rows: jnp.ndarray, key=None, time_sigma: float = 50.0):
    """Row indices -> (obs = (pmt_id, t), hyp) for FrameHitNet-style models.

    Geometry stays inside the model (its tables); the batch carries only the sensor
    index and the (augmented) hit time. Same time-shuffle contract as hit_batch.
    """
    ev = data.event_id[rows]
    t = data.t[rows]
    hyp = data.hyp[ev]
    if key is not None and time_sigma > 0:
        sh = _event_shifts(key, ev, time_sigma)
        t = t + sh
        hyp = hyp.at[:, 5].add(sh)
    return (data.pmt_id[rows], t), hyp


def charge_batch(data: DeviceData, rows: jnp.ndarray, key=None):
    """Event indices -> (charge obs (B,2), hyp (B,7)). Charge features carry no time."""
    return data.charge[rows], data.hyp[rows]


def fit_resident(
    model,
    data: DeviceData,
    make_batch,
    n_rows: int,
    *,
    key,
    batch_size: int = 2**17,
    learning_rate: float = 1e-3,
    max_epochs: int = 1000,
    patience: int = 50,
    val_fraction: float = 0.1,
    max_val_rows: int = 2**16,
    balance_weight: float = 0.0,
    checkpoint_dir: str = None,
    checkpoint_every: int = 25,
    verbose: bool = True,
) -> FitResult:
    """Train an NRE model with every batch formed on-device (no host input path).

    Same semantics as ``hitman.train.fit`` (row split, early stopping, BNRE option);
    ``make_batch`` is ``hit_batch`` (n_rows = data.n_hits) or ``charge_batch``
    (n_rows = data.n_events), or any (data, rows) -> (obs, hyp) function.

    When ``checkpoint_dir`` is given, the best-on-val model is written to
    ``best.eqx`` on every improvement (crash safety for multi-hour runs) and a
    trajectory snapshot ``epoch_<N>.eqx`` every ``checkpoint_every`` epochs is
    KEPT (not rotated): BCE-val and the physics receipts are known to disagree
    at the margin, so final model selection can be receipt-based over the
    trajectory instead of committed to the BCE optimum. Snapshots are ~1 MB.
    """
    n_val = max(int(n_rows * val_fraction), 1)
    n_train = n_rows - n_val
    batch_size = min(batch_size, n_train)
    val_idx = np.arange(n_train, n_rows)
    if n_val > max_val_rows:
        val_idx = val_idx[:: n_val // max_val_rows + 1][:max_val_rows]
    val_rows = jnp.asarray(val_idx, jnp.int32)

    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    iota = jnp.arange(batch_size, dtype=jnp.int32)

    @eqx.filter_jit
    def train_step(model, opt_state, data, start, key):
        rows = start + iota
        k_aug, k_loss = jax.random.split(key)

        def loss_fn(model):
            obs, hyp, w = _unpack_batch(make_batch(data, rows, k_aug))
            return _batch_loss(model, obs, hyp, k_loss, balance_weight, w)

        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, opt_state = optimizer.update(grads, opt_state)
        return eqx.apply_updates(model, updates), opt_state, loss

    @eqx.filter_jit
    def val_loss_fn(model, data, rows, key):
        k_aug, k_loss = jax.random.split(key)
        obs, hyp, w = _unpack_batch(make_batch(data, rows, k_aug))
        return _batch_loss(model, obs, hyp, k_loss, balance_weight, w)

    steps_per_epoch = max(n_train // batch_size, 1)
    key, val_key = jax.random.split(key)  # fixed: val metric comparable across epochs
    best = (np.inf, 0, model)
    train_hist, val_hist = [], []

    # Batches are contiguous row windows in shuffled order (block sampling): rows
    # group by event and events are written in iid order, so a window is ~10^3 iid
    # events — statistically equivalent to a full shuffle for SGD, while the gathers
    # become coalesced DRAM reads (at 5M events, random-row gathers into ~6 GB of
    # tables were the dominant epoch cost) and the 4e8-row permutation reduces to
    # shuffling window offsets. A per-epoch random phase decorrelates window edges.
    perm_rng = np.random.default_rng(int(jax.random.randint(key, (), 0, 2**31 - 1)))
    for epoch in range(max_epochs):
        t0 = time.time()
        phase = int(perm_rng.integers(0, batch_size)) if n_train > 2 * batch_size else 0
        n_windows = (n_train - phase) // batch_size
        starts = perm_rng.permutation(n_windows).astype(np.int64) * batch_size + phase
        losses = []
        for step in range(min(steps_per_epoch, n_windows)):
            key, step_key = jax.random.split(key)
            model, opt_state, loss = train_step(
                model, opt_state, data, jnp.asarray(starts[step], jnp.int32), step_key)
            losses.append(loss)  # device scalar; no per-step sync
        v = float(val_loss_fn(model, data, val_rows, val_key))
        train_hist.append(float(jnp.mean(jnp.stack(losses))))
        val_hist.append(v)
        if v < best[0]:
            best = (v, epoch, model)
            if checkpoint_dir is not None:
                eqx.tree_serialise_leaves(os.path.join(checkpoint_dir, "best.eqx"), model)
        if checkpoint_dir is not None and epoch % checkpoint_every == 0:
            eqx.tree_serialise_leaves(
                os.path.join(checkpoint_dir, f"epoch_{epoch:04d}.eqx"), model)
        if verbose:
            print(
                f"epoch {epoch:4d}  train {train_hist[-1]:.5f}  val {v:.5f}  "
                f"({time.time() - t0:.1f}s)",
                flush=True,
            )
        if epoch - best[1] >= patience:
            break

    return FitResult(model=best[2], train_loss=train_hist, val_loss=val_hist, best_epoch=best[1])
