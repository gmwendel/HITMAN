"""Score-identity penalty: exact moment constraints of a true likelihood ratio.

Any TRUE log likelihood-to-evidence ratio log r(x, theta) = log p(x|theta) - log p(x)
satisfies E_{x ~ p(.|theta)}[grad_theta log r(x, theta)] = 0 at every theta (the
reference p(x) is theta-free, so this is the classical score identity). A learned
ratio violates it by a coherent, theta-dependent mean tilt b(theta) — measured on
run10 as the smooth b_E(E) curve behind the MLE energy wall (see
training_runs/scoreid_baseline). This module turns the identity into a training
penalty. It is geometry- and detector-agnostic: nothing here knows about Eos.

Estimator design (receipts: training_runs/scoreid_baseline/):

* **Stratified in true E** on fixed edges — the violation is theta-dependent, so a
  global mean washes it out (b_E flips sign across the threshold stratum).
* **Split-half cross-product.** Per-event scores fluctuate at the Fisher scale
  (information, not error); the identity constrains only their mean. The penalty per
  stratum is <mean(g_A), mean(g_B)> over a random half-split — an unbiased estimator
  of ||E g||^2 in which the per-event noise cancels. Penalizing mean ||g||^2 instead
  would shrink the information itself: wrong.
* **Per-stratum whitening** by fixed scales (score std varies ~8x across E strata),
  so every stratum and component contributes O(1).
* **Winsorization** at fixed per-stratum bounds (measured P99.9 tails). Guards the
  d/dt score of stray early/late photons in the sparse time tails. Baseline receipt:
  clipping at P0.1/P99.9 moves every stratum mean by <= 0.4 SE, so the induced bias
  is negligible; keep the bounds from the same receipt that proved that.

The identity batch carries the same coherent per-event time-translation augmentation
as training (the identity holds exactly under the shifted truth, and the shifts
spread stratum coverage in theta_t for free).
"""

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


class IdentityBatchSpec(NamedTuple):
    """Device-side inputs for event-grouped identity batches.

    ``offsets`` is the store's hit-offset table; ``elig`` the event indices the
    sampler may draw (train-zone events whose hit count fits ``n_pad``).
    """

    offsets: jnp.ndarray  # (n_events + 1,) int32
    elig: jnp.ndarray     # (n_elig,) int32


class IdentityCfg(NamedTuple):
    """Score-identity penalty configuration.

    ``edges`` are FIXED strata on true E (batch-dependent edges add estimator noise);
    ``scales`` (K, 7) whitening = 1/score-std per stratum; ``bounds`` (K, 7) symmetric
    winsorize limits in raw score units (np.inf disables); both from the baseline
    receipt. ``min_half`` guards degenerate half-splits.
    """

    lam: float
    edges: jnp.ndarray    # (K + 1,)
    scales: jnp.ndarray   # (K, 7)
    bounds: jnp.ndarray   # (K, 7)
    n_events: int = 512
    n_pad: int = 320
    time_sigma: float = 50.0
    min_half: int = 8


def build_identity_spec(store, n_pad: int, lo: int = 0, hi: int = None) -> IdentityBatchSpec:
    """Eligible = events in [lo, hi) with <= n_pad hits (drop rate is E-correlated:
    report it if it matters — at pad 320 on the 5M store it is <0.6% in 9-10 MeV)."""
    hi = store.n_events if hi is None else hi
    nhit = np.diff(store.hit_offsets[lo : hi + 1])
    elig = lo + np.flatnonzero(nhit <= n_pad)
    return IdentityBatchSpec(
        offsets=jnp.asarray(store.hit_offsets, jnp.int32),
        elig=jnp.asarray(elig, jnp.int32),
    )


def make_identity_batch(data, spec: IdentityBatchSpec, key, cfg: IdentityCfg):
    """Sample an event-grouped padded batch, all gathers on device.

    Returns (hits (N, P, 4), mask (N, P), charge (N, 2), theta (N, 7)).
    """
    k_ev, k_aug = jax.random.split(key)
    ev = spec.elig[jax.random.randint(k_ev, (cfg.n_events,), 0, spec.elig.shape[0])]
    start = spec.offsets[ev]
    nh = spec.offsets[ev + 1] - start
    slot = jnp.arange(cfg.n_pad, dtype=jnp.int32)
    rows = jnp.minimum(start[:, None] + slot[None, :], data.n_hits - 1)
    mask = (slot[None, :] < nh[:, None]).astype(jnp.float32)
    t = data.t[rows]
    theta = data.hyp[ev]
    if cfg.time_sigma > 0:
        sh = cfg.time_sigma * jax.random.normal(k_aug, (cfg.n_events,))
        t = t + sh[:, None]
        theta = theta.at[:, 5].add(sh)
    pos = data.pmt_pos[data.pmt_id[rows]]
    hits = jnp.concatenate([pos, t[..., None]], axis=-1) * mask[..., None]
    return hits, mask, data.charge[ev], theta


def composed_score(hitnet, chargenet, hits, mask, charge, theta):
    """grad_theta of the composed event log-ratio at theta, one padded event."""
    if getattr(hitnet, "obs_style", "xyz") != "xyz":
        raise NotImplementedError(
            "score identity batch carries xyzt hit rows; id_t-style models need "
            "pmt_id plumbed through make_identity_batch first")

    def ll(th):
        logits = jax.vmap(lambda h: hitnet(h, th))(hits)
        return jnp.sum(logits * mask) + chargenet(charge, th)

    return jax.grad(ll)(theta)


def score_identity_penalty(hitnet, chargenet, batch, cfg: IdentityCfg, key) -> jnp.ndarray:
    """Unbiased estimate of the stratified squared mean score (whitened units).

    Sum over strata of <mean g_A, mean g_B> with a fresh random half-split; strata
    whose smaller half has < min_half events contribute 0. Expectation is
    sum_k ||E[g | stratum k]||^2_whitened, minimized exactly at the score identity.
    """
    hits, mask, charge, theta = batch
    g = jax.vmap(lambda h, m, c, th: composed_score(hitnet, chargenet, h, m, c, th))(
        hits, mask, charge, theta)                                   # (N, 7)
    k = jnp.clip(jnp.searchsorted(cfg.edges, theta[:, 6]) - 1, 0, cfg.scales.shape[0] - 1)
    g = jnp.clip(g, -cfg.bounds[k], cfg.bounds[k]) * cfg.scales[k]   # winsorize, whiten
    half = jax.random.bernoulli(key, 0.5, (g.shape[0],))
    seg = k * 2 + half.astype(jnp.int32)
    n_seg = 2 * cfg.scales.shape[0]
    sums = jax.ops.segment_sum(g, seg, num_segments=n_seg)           # (2K, 7)
    cnts = jax.ops.segment_sum(jnp.ones_like(seg, jnp.float32), seg, num_segments=n_seg)
    means = sums / jnp.maximum(cnts, 1.0)[:, None]
    m_a, m_b = means[0::2], means[1::2]
    valid = (jnp.minimum(cnts[0::2], cnts[1::2]) >= cfg.min_half).astype(jnp.float32)
    return jnp.sum(valid * jnp.sum(m_a * m_b, axis=1))


def make_score_identity_loss(
    chargenet, spec: IdentityBatchSpec, cfg: IdentityCfg
) -> Callable:
    """Closure for train_recipe's ``extra_loss`` hook: lam * penalty on a fresh
    identity batch. ``chargenet`` is frozen (closed over); the trained model is the
    hitnet passed in at call time.

    DeviceData is a CALL-TIME argument, never closed over: jit only traces function
    arguments, and a closure-captured DeviceData gets baked into the compiled step
    as ~4 GB of constants — the resulting multi-GB XLA compile hard-locked the box
    on 2026-07-19. ``spec`` (~40 MB) is the only array closure here, and drivers
    must escalate jax's captured-constants warning to an error (see
    train_run12_scoreid.py) so any regression aborts before the compile."""

    def extra_loss(hitnet, data, key):
        k_batch, k_split = jax.random.split(key)
        batch = make_identity_batch(data, spec, k_batch, cfg)
        return cfg.lam * score_identity_penalty(hitnet, chargenet, batch, cfg, k_split)

    return extra_loss
