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

    from hitman.wc.train.resident import _event_shifts

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


class ChargeWeightTable(NamedTuple):
    w: jnp.ndarray   # (n_max+1,) mean-1 weights indexed by nhit
    alpha: float


def build_charge_weights(store, *, alpha: float = 0.5, n_max: int = 400,
                         w_max: float = 100.0) -> ChargeWeightTable:
    """w(nhit) ∝ p̂(nhit)^(−α): the charge-axis version of the tail reweighting.

    Bright events live in the sparse upper tail of the multiplicity marginal —
    the measured energy-axis wall (E bias −0.9 MeV at 8 MeV even at the detector
    center) is the prior-weighted-accuracy tax on this axis. Same normalization
    contract as the hit table: E_p̂[w] = 1, clipped at ``w_max``.
    """
    n = np.clip(np.asarray(store.charge[:, 1]).astype(np.int64), 0, n_max)
    h = np.bincount(n, minlength=n_max + 1).astype(np.float64)
    p = (h + 0.5) / (h + 0.5).sum()
    w = p ** (-alpha)
    w = w / np.sum(p * w)
    w = np.minimum(w, w_max)
    w = w / np.sum(p * w)
    return ChargeWeightTable(w=jnp.asarray(w, jnp.float32), alpha=float(alpha))


def make_weighted_charge_batch(table: ChargeWeightTable):
    """A charge make_batch returning (obs, hyp, weights); drop-in for the loops."""

    def make(data, rows, key=None):
        c = data.charge[rows]
        n = jnp.clip(c[:, 1].astype(jnp.int32), 0, table.w.shape[0] - 1)
        return c, data.hyp[rows], table.w[n]

    return make


class BrightnessWeightTable(NamedTuple):
    w: jnp.ndarray   # (n_max+1,) weights indexed by parent-event nhit; mean-1 over HITS
    alpha: float


def build_brightness_weights(store, *, alpha: float = 0.25, n_max: int = 400,
                             w_max: float = 100.0) -> BrightnessWeightTable:
    """Per-HIT weight ``w(nhit_event) ∝ p̂(nhit)^(−α)``: the E-wall fix on the HITNET.

    The measured E-wall (paired E-ridge −0.437 MeV at 8 MeV even at the detector
    center) is localized in the hitnet: 8 MeV events (~156 hits) sit in the sparse
    upper tail of a 0–10 MeV-flat training set (mean ~98 PE), so the BCE risk under-
    trains the (sensor,t) density exactly where brightness is high. This table maps a
    hit's PARENT-EVENT multiplicity to a weight; ``make_brightness_weighted_hit_batch``
    then upweights every hit belonging to a bright event.

    Why brightness and not the (sensor,t) marginal (run11): a pure obs-only weight is
    ratio-preserving and merely redistributes accuracy WITHIN the bulk marginal — it
    cannot move mass toward the high-E tail (it "buys nothing"). Event brightness is
    monotone in energy, so for JOINT pairs (whose θ_E ≈ the parent's true energy) the
    weight acts as an importance reweighting of the training density toward the
    under-represented high-E tail — the axis the wall actually lives on.

    Normalization is over the HIT population (Σ_e n_e·w(n_e) / Σ_e n_e = 1), so the
    mean per-hit weight is 1 and the per-hit gradient scale matches the unweighted
    loss (``nre_loss`` takes an unnormalized ``jnp.mean(w··)``). Weights are clipped at
    ``w_max`` and renormalized (tail-explosion guard, as for the other tables).
    """
    n = np.clip(np.asarray(store.charge[:, 1]).astype(np.int64), 0, n_max)
    h = np.bincount(n, minlength=n_max + 1).astype(np.float64)   # events per nhit bin
    p = (h + 0.5) / (h + 0.5).sum()
    w = p ** (-alpha)
    nvals = np.arange(n_max + 1, dtype=np.float64)
    hitmass = h * nvals                       # hits contributed by each nhit bin
    hitmass = hitmass / hitmass.sum()         # per-HIT distribution over parent nhit
    w = w / np.sum(hitmass * w)               # E_hit[w] = 1
    w = np.minimum(w, w_max)
    w = w / np.sum(hitmass * w)               # renormalize after the clip
    return BrightnessWeightTable(w=jnp.asarray(w, jnp.float32), alpha=float(alpha))


def make_brightness_weighted_hit_batch(table: BrightnessWeightTable, obs_style: str = "xyz",
                                       time_sigma: float = 50.0):
    """A hit make_batch returning (obs, hyp, weights) — drop-in for train_recipe/fit_resident.

    Each hit inherits its PARENT event's brightness weight ``w(nhit_event)`` from
    ``table`` (parent nhit = ``data.charge[event_id, 1]``). Identical time-shuffle
    augmentation contract to ``hit_batch``/``make_weighted_hit_batch``: with a key, each
    event's time origin shifts by N(0, time_sigma) ns coherently in its hit times and
    its hypothesis time (the brightness weight is time-invariant, so the shift does not
    touch it).
    """

    from hitman.wc.train.resident import _event_shifts

    def make(data, rows, key=None):
        ev = data.event_id[rows]
        pmt = data.pmt_id[rows]
        t = data.t[rows]
        hyp = data.hyp[ev]
        if key is not None and time_sigma > 0:
            sh = _event_shifts(key, ev, time_sigma)
            t = t + sh
            hyp = hyp.at[:, 5].add(sh)
        nhit = jnp.clip(data.charge[ev, 1].astype(jnp.int32), 0, table.w.shape[0] - 1)
        w = table.w[nhit]
        if obs_style == "id_t":
            return (pmt, t), hyp, w
        obs = jnp.concatenate([data.pmt_pos[pmt], t[:, None]], axis=1)
        return obs, hyp, w

    return make
