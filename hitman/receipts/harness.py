"""Reusable receipt harness: run the forward battery over any ``(model, data)`` (DESIGN
proposal 3).

The numeric receipt core (``chi2``, ``reweight``, ``forward``, ``score``, ``schema``) is
already detector-agnostic and synthetic-net tested; what was water-Cherenkov-coupled lived
in ``runner.py`` — the z-ring strata, the marginal-pool reweighting, and the NRE model
interface. This module lifts those into two seams so a downstream project (e.g. the
muon-station program with shower-frame ``E x r`` strata) runs the same battery:

* :class:`ReceiptModel` — the small protocol the battery needs from a model: pool log-ratios
  over a marginal pool at a fixed hypothesis, and count-grid logits. :class:`NREReceiptModel`
  is the WC hitnet/chargenet instantiation.
* ``strata_fn`` — an injectable ``(coords, n) -> (labels, n_labels)`` strata assignment; the
  WC default :func:`z_rings` (equal-occupancy z percentiles) becomes one instance,
  :func:`percentile_strata` over any coordinate (core distance r, energy, ...) another.

The WC path (``runner.run_model_dir``) is exactly ``NREReceiptModel`` + ``z_rings`` through
:func:`forward_block_from_model`; ``tests/test_receipts_harness.py`` locks it to the direct
``compute_forward_block`` numbers. The score/Bartlett receipts are already caller-stratified
(``score.score_identity`` takes ``names``; ``forward.per_ring_time_receipt`` takes precomputed
labels), so ``E x r x sign(psi)`` strata plug in by passing a ``strata_fn`` / the proposal-4
projection instrument — no core change.
"""

from typing import Callable, Optional, Protocol, runtime_checkable

import numpy as np

from hitman.receipts import forward as fwd
from hitman.receipts import schema
from hitman.receipts.reweight import self_normalized_weights


# ---------------------------------------------------------------------------
# strata plugins (generalize z_rings)
# ---------------------------------------------------------------------------
def percentile_strata(values, n: int):
    """Assign each entry of a 1-D ``values`` array to one of ``n`` equal-occupancy bins.

    Returns ``(labels, n_labels)``; ``n_labels`` may be < ``n`` if ties collapse edges. The
    detector-agnostic strata primitive: WC z-rings, muon core-distance annuli, energy bins
    all reduce to this over the appropriate coordinate.
    """
    values = np.asarray(values)
    edges = np.unique(np.percentile(values, np.linspace(0, 100, n + 1)))
    m = len(edges) - 1
    labels = np.clip(np.digitize(values, edges) - 1, 0, m - 1)
    return labels, m


def z_rings(pmt_pos, n_rings: int):
    """WC default strata: assign each PMT to a z-ring with equal geometry coverage."""
    return percentile_strata(np.asarray(pmt_pos)[:, 2], n_rings)


# ---------------------------------------------------------------------------
# forward battery core (numpy-only; the tested seam)
# ---------------------------------------------------------------------------
def compute_forward_block(
    logw,
    pool_pmt,
    pool_t,
    pmt_pos,
    mc_pmt,
    mc_t,
    mc_ntot,
    n_mc_events: int,
    chargenet_logit_c,
    n_grid,
    train_n_hist,
    n_rings: int = 8,
    time_bins=None,
    ring_time_bins=None,
    strata_fn: Optional[Callable] = None,
):
    """Assemble the serialized ``forward`` block + (self_norm, ess) from arrays.

    Everything model-specific has already been reduced to ``logw`` (pool log-ratios) and
    ``chargenet_logit_c`` (count-grid logits); this function is pure numpy and is the
    unit-tested entry point. ``strata_fn(pmt_pos, n_rings) -> (labels, n_labels)`` assigns
    the per-ring-time strata (default :func:`z_rings`, the WC z-percentile rings).
    """
    if strata_fn is None:
        strata_fn = z_rings
    pool = self_normalized_weights(logw)
    pmt_pos = np.asarray(pmt_pos)
    n_pmts = len(pmt_pos)

    pmf, mean_ntot = fwd.implied_ntot_pmf(chargenet_logit_c, train_n_hist, n_grid)

    per_pmt = fwd.per_pmt_charge_receipt(
        pool, pool_pmt, n_pmts, mc_pmt, n_mc_events, mean_ntot
    )
    toa = fwd.toa_receipt(pool, pool_t, mc_t, bins=time_bins)
    ntot = fwd.ntot_receipt(pmf, n_grid, train_n_hist, mc_ntot, n_mc_events)

    ring, nr = strata_fn(pmt_pos, n_rings)
    mc_ring = ring[np.asarray(mc_pmt)]
    pool_ring = ring[np.asarray(pool_pmt)]
    if ring_time_bins is None:
        ring_time_bins = np.linspace(
            np.floor(np.min(mc_t)), np.percentile(mc_t, 99.5), 41
        )
    ring_time = fwd.per_ring_time_receipt(
        pool, pool_t, pool_ring, mc_t, mc_ring, nr, ring_time_bins
    )

    forward_block = {
        "per_pmt_charge": schema.chi2_block(per_pmt),
        "toa": schema.chi2_block(toa),
        "n_tot": schema.ntot_block(ntot),
        "per_ring_time": schema.ring_time_block(ring_time),
    }
    return forward_block, pool.self_norm, pool.ess


# ---------------------------------------------------------------------------
# model protocol (decouple the battery from the WC hitnet/chargenet loaders)
# ---------------------------------------------------------------------------
@runtime_checkable
class ReceiptModel(Protocol):
    """What the forward battery needs from a model at one fixed hypothesis ``theta``.

    A conforming model reduces itself to two numpy arrays, so the battery never touches the
    detector's JAX/observation internals. WC's :class:`NREReceiptModel` is one instantiation;
    a density model (e.g. the splinemle spline / the muon chain-of-conditionals) supplies its
    own by returning per-pool log-densities and count-grid log-probabilities.
    """

    def pool_log_ratios(self, pool_pmt, pool_t, pmt_pos, theta) -> np.ndarray:
        """log-weight of each marginal-pool object at ``theta`` (self-normalized downstream)."""
        ...

    def count_grid_logits(self, n_grid, theta) -> np.ndarray:
        """log-ratio/log-prob of each integer count in ``n_grid`` at ``theta``."""
        ...


def hit_logits(hitnet, obs_style, pool_pmt, pool_t, pmt_pos, theta, chunk: int = 500_000):
    """log r_hit over the pool at fixed theta (chunked; obs-style aware). WC NRE."""
    import jax
    import jax.numpy as jnp

    M = len(pool_pmt)
    if obs_style == "id_t":
        fn = jax.jit(jax.vmap(lambda i, t: hitnet((i, t), theta)))
        return np.concatenate(
            [
                np.asarray(fn(jnp.asarray(pool_pmt[i : i + chunk]),
                              jnp.asarray(pool_t[i : i + chunk])))
                for i in range(0, M, chunk)
            ]
        )
    obs = np.concatenate([pmt_pos[pool_pmt], pool_t[:, None]], axis=1).astype(np.float32)
    fn = jax.jit(jax.vmap(lambda h: hitnet(h, theta)))
    return np.concatenate(
        [np.asarray(fn(jnp.asarray(obs[i : i + chunk]))) for i in range(0, M, chunk)]
    )


def chargenet_logit_grid(chargenet, n_grid, theta):
    """log r_charge over an integer N grid at fixed theta (uses (N, N) as (q, nhit)). WC NRE."""
    import jax
    import jax.numpy as jnp

    c_obs = jnp.asarray(np.stack([n_grid, n_grid], axis=1).astype(np.float32))
    return np.asarray(jax.jit(jax.vmap(lambda c: chargenet(c, theta)))(c_obs))


class NREReceiptModel:
    """Water-Cherenkov NRE instantiation of :class:`ReceiptModel` (hitnet + chargenet)."""

    def __init__(self, hitnet, chargenet, obs_style: str = "xyz", chunk: int = 500_000):
        self.hitnet = hitnet
        self.chargenet = chargenet
        self.obs_style = obs_style
        self.chunk = chunk

    def pool_log_ratios(self, pool_pmt, pool_t, pmt_pos, theta) -> np.ndarray:
        return hit_logits(self.hitnet, self.obs_style, pool_pmt, pool_t, pmt_pos, theta,
                          chunk=self.chunk)

    def count_grid_logits(self, n_grid, theta) -> np.ndarray:
        return chargenet_logit_grid(self.chargenet, n_grid, theta)


def forward_block_from_model(
    model: ReceiptModel,
    pool_pmt,
    pool_t,
    pmt_pos,
    mc_pmt,
    mc_t,
    mc_ntot,
    n_mc_events: int,
    n_grid,
    train_n_hist,
    theta,
    *,
    n_rings: int = 8,
    strata_fn: Optional[Callable] = None,
    time_bins=None,
    ring_time_bins=None,
):
    """Run the forward battery for a conforming :class:`ReceiptModel` at one test point.

    The generic entry point: reduces the model to pool log-ratios + count-grid logits, then
    calls :func:`compute_forward_block`. WC's ``run_model_dir`` is exactly this with an
    :class:`NREReceiptModel` and the default ``z_rings`` strata.
    """
    logw = model.pool_log_ratios(pool_pmt, pool_t, pmt_pos, theta)
    logit_c = model.count_grid_logits(n_grid, theta)
    return compute_forward_block(
        logw, pool_pmt, pool_t, pmt_pos, mc_pmt, mc_t, mc_ntot, n_mc_events,
        logit_c, n_grid, train_n_hist, n_rings=n_rings, time_bins=time_bins,
        ring_time_bins=ring_time_bins, strata_fn=strata_fn,
    )
