"""Ratio-preserving tail reweighting: w(x) ∝ p̂(x)^(−α), the designated near-wall fix.

The BCE risk buys accuracy where the training marginal p(x) puts mass, so the learned
ratio degrades exactly at the low-p̂(x) edge ("the wall": exact polish at 5M improved
bulk BCE while degrading near-wall KL +57%). Because the weight depends on the
observation only, applying it to BOTH class terms at the same x leaves the Bayes-optimal
logit — the log-ratio — unchanged (it cancels in the pointwise minimizer; the principled
basis is Chehab–Gramfort–Hyvärinen, arXiv:2203.01110: the optimal contrastive
distribution is not the data's). α=0 recovers the unweighted loss; α=1 equalizes
accuracy density across the marginal support (aggressive — start ~0.25–0.5).

Weights are looked up from the same (sensor, time) histogram the Z(θ) normalization
uses, are normalized to E_p̂[w]=1 (gradient scale comparable to unweighted), and are
clipped at ``w_max`` before renormalization (empty-bin/tail explosion guard).
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


class HitWeightTable(NamedTuple):
    w_grid: jnp.ndarray  # (n_sensors, n_tbins) mean-1 weights
    t_lo: float
    dt: float
    n_tbins: int
    alpha: float


def build_hit_weights(store, *, alpha: float = 0.5, n_sample: int = 30_000_000,
                      time_sigma: float = 50.0, t_lo: float = -160.0, t_hi: float = 260.0,
                      dt: float = 0.5, w_max: float = 100.0, seed: int = 0) -> HitWeightTable:
    """Histogram the (augmented) hit marginal and turn it into a weight lookup table."""
    rng = np.random.default_rng(seed)
    n = min(n_sample, store.n_hits)
    rows = np.sort(rng.choice(store.n_hits, n, replace=False))
    pmt = np.asarray(store.pmt_id[rows]).astype(np.int64)
    ev = np.asarray(store.event_id[rows])
    shift = rng.normal(0.0, time_sigma, store.n_events).astype(np.float32)
    t = np.asarray(store.hits[rows, 3]) + shift[ev]
    n_tbins = int(round((t_hi - t_lo) / dt))
    n_pmts = len(np.asarray(store.pmt_pos))
    tbin = np.clip(((t - t_lo) / dt).astype(np.int64), 0, n_tbins - 1)
    h = np.bincount(pmt * n_tbins + tbin, minlength=n_pmts * n_tbins).reshape(n_pmts, n_tbins)
    p = (h + 0.5) / (h + 0.5).sum()          # additive smoothing: empty bins get finite w
    w = p ** (-alpha)
    w = w / np.sum(p * w)                     # E_p[w] = 1
    w = np.minimum(w, w_max)
    w = w / np.sum(p * w)                     # renormalize after the clip
    return HitWeightTable(w_grid=jnp.asarray(w, jnp.float32), t_lo=float(t_lo),
                          dt=float(dt), n_tbins=n_tbins, alpha=float(alpha))


def lookup_weights(table: HitWeightTable, pmt_id: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
    tbin = jnp.clip(((t - table.t_lo) / table.dt).astype(jnp.int32), 0, table.n_tbins - 1)
    return table.w_grid[pmt_id, tbin]


def make_weighted_hit_batch(table: HitWeightTable, obs_style: str = "xyz",
                            time_sigma: float = 50.0):
    """A make_batch returning (obs, hyp, weights) — drop-in for fit_resident/train_recipe.

    Same time-shuffle augmentation contract as hit_batch/frame_hit_batch; the weight is
    looked up at the AUGMENTED hit time (the histogram was built augmented too).
    """

    from hitman.train.resident import _event_shifts

    def make(data, rows, key=None):
        ev = data.event_id[rows]
        pmt = data.pmt_id[rows]
        t = data.t[rows]
        hyp = data.hyp[ev]
        if key is not None and time_sigma > 0:
            sh = _event_shifts(key, ev, time_sigma)
            t = t + sh
            hyp = hyp.at[:, 5].add(sh)
        w = lookup_weights(table, pmt, t)
        if obs_style == "id_t":
            return (pmt, t), hyp, w
        obs = jnp.concatenate([data.pmt_pos[pmt], t[:, None]], axis=1)
        return obs, hyp, w

    return make
