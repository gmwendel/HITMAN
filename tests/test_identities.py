"""Score-identity penalty: exactness, detection power, and estimator unbiasedness.

Toy: per event, truth theta has t0 ~ N(0, 30) (slot 5) and E ~ U(0.5, 9.5) (slot 6);
each of 8 hits carries time t ~ N(t0, s) in h[3] and a proxy coordinate x ~ N(E, 1)
in h[0]. Both exact marginals are analytic (Gaussian; uniform-convolved Gaussian via
the normal CDF), so ExactNet's logit IS the true log ratio and the score identity
holds by construction. TiltedNet adds a coherent eps * theta_E per hit — the exact
kind of violation the penalty exists to catch (a b_E tilt).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.stats import norm

from hitman.train.identities import (IdentityBatchSpec, IdentityCfg, composed_score,
                                     make_identity_batch, make_score_identity_loss,
                                     score_identity_penalty)
from hitman.train.resident import DeviceData

S_T, SIG_T0 = 2.0, 30.0
S_MARG = float(np.sqrt(S_T**2 + SIG_T0**2))
E_LO, E_HI = 0.5, 9.5
N_HIT, N_PAD = 8, 16


class ExactNet:
    obs_style = "xyz"

    def __call__(self, h, th):
        lr_t = norm.logpdf(h[3], th[5], S_T) - norm.logpdf(h[3], 0.0, S_MARG)
        marg_x = (norm.cdf(h[0] - E_LO) - norm.cdf(h[0] - E_HI)) / (E_HI - E_LO)
        lr_x = norm.logpdf(h[0], th[6], 1.0) - jnp.log(marg_x)
        return lr_t + lr_x


class TiltedNet(ExactNet):
    def __init__(self, eps):
        self.eps = eps

    def __call__(self, h, th):
        return super().__call__(h, th) + self.eps * th[6]


def zero_chargenet(c, th):
    return 0.0 * th[0]


def gen_batch(key, n_events):
    k1, k2, k3, k4 = jax.random.split(key, 4)
    t0 = SIG_T0 * jax.random.normal(k1, (n_events,))
    e = jax.random.uniform(k2, (n_events,), minval=E_LO, maxval=E_HI)
    theta = jnp.zeros((n_events, 7)).at[:, 5].set(t0).at[:, 6].set(e)
    t = t0[:, None] + S_T * jax.random.normal(k3, (n_events, N_PAD))
    x = e[:, None] + jax.random.normal(k4, (n_events, N_PAD))
    mask = jnp.broadcast_to(
        (jnp.arange(N_PAD)[None, :] < N_HIT).astype(jnp.float32), (n_events, N_PAD))
    hits = jnp.stack([x, jnp.zeros_like(x), jnp.zeros_like(x), t], axis=-1)
    hits = hits * mask[:, :, None]
    return hits, mask, jnp.zeros((n_events, 2)), theta


CFG = IdentityCfg(
    lam=1.0,
    edges=jnp.array([E_LO, 5.0, E_HI]),
    scales=jnp.ones((2, 7)),
    bounds=jnp.full((2, 7), np.inf),
    n_events=2048,
    n_pad=N_PAD,
    time_sigma=0.0,
)


def _penalties(net, n_keys=16, n_events=2048, seed=0):
    pen = jax.jit(lambda b, k: score_identity_penalty(net, zero_chargenet, b, CFG, k))
    out = []
    for i in range(n_keys):
        kd, ks = jax.random.split(jax.random.PRNGKey(seed + 1000 * i))
        out.append(float(pen(gen_batch(kd, n_events), ks)))
    return np.array(out)


def test_exact_model_penalty_zero():
    p = _penalties(ExactNet())
    se = p.std(ddof=1) / np.sqrt(len(p))
    assert abs(p.mean()) < 4 * se, f"exact ratio violates identity: {p.mean():.4f} ± {se:.4f}"


def test_tilt_detected_and_split_half_unbiased():
    eps = 0.05
    p = _penalties(TiltedNet(eps))
    p0 = _penalties(ExactNet())
    # detection: far outside the exact model's scatter
    assert p.mean() > p0.mean() + 10 * p0.std(ddof=1)
    # unbiasedness: E[penalty] = sum_k ||E g_k||^2 = 2 strata * (eps * N_HIT)^2 exactly
    truth = 2 * (eps * N_HIT) ** 2
    se = p.std(ddof=1) / np.sqrt(len(p))
    assert abs(p.mean() - truth) < 4 * se, f"{p.mean():.4f} vs {truth:.4f} ± {se:.4f}"


def test_composed_score_shape_and_zero_components():
    hits, mask, charge, theta = gen_batch(jax.random.PRNGKey(3), 4)
    g = composed_score(ExactNet(), zero_chargenet, hits[0], mask[0], charge[0], theta[0])
    assert g.shape == (7,)
    # toy nets use only slots 5 and 6; all other score components are exactly 0
    assert np.allclose(np.asarray(g)[:5], 0.0)


def test_winsorize_bounds_clip():
    hits, mask, charge, theta = gen_batch(jax.random.PRNGKey(4), 512)
    tight = CFG._replace(bounds=jnp.full((2, 7), 1e-4))
    p = score_identity_penalty(ExactNet(), zero_chargenet,
                               (hits, mask, charge, theta), tight, jax.random.PRNGKey(0))
    # every |g| clipped to 1e-4: penalty bounded by 7 * (1e-4)^2 per stratum
    assert abs(float(p)) < 2 * 7 * 1e-8


def test_make_identity_batch_gather_and_aug():
    # 5 events with 2/3/1/4/2 hits; event 3 (4 hits) excluded by n_pad=3
    offsets = np.array([0, 2, 5, 6, 10, 12])
    n_hits = 12
    t = np.arange(n_hits, dtype=np.float32)
    data = DeviceData(
        t=jnp.asarray(t),
        pmt_id=jnp.zeros(n_hits, jnp.int32),
        event_id=jnp.asarray(np.repeat(np.arange(5), np.diff(offsets)), jnp.int32),
        hyp=jnp.asarray(np.tile(np.arange(7, dtype=np.float32), (5, 1))),
        charge=jnp.asarray(np.stack([np.arange(5), np.diff(offsets)], 1), jnp.float32),
        pmt_pos=jnp.asarray([[1.0, 2.0, 3.0]]),
    )
    elig = np.flatnonzero(np.diff(offsets) <= 3)
    spec = IdentityBatchSpec(offsets=jnp.asarray(offsets, jnp.int32),
                             elig=jnp.asarray(elig, jnp.int32))
    cfg = CFG._replace(n_events=64, n_pad=3, time_sigma=25.0)
    hits, mask, charge, theta = make_identity_batch(data, spec, jax.random.PRNGKey(7), cfg)
    assert hits.shape == (64, 3, 4) and mask.shape == (64, 3)
    ev = np.asarray(charge[:, 0]).astype(int)          # charge col 0 encodes event id
    assert set(ev) <= set(elig.tolist())               # oversize event never sampled
    assert np.array_equal(np.asarray(mask).sum(1), np.diff(offsets)[ev])
    # aug coherence: hyp slot 5 shift equals the hit-time shift, same for all hits
    sh = np.asarray(theta[:, 5]) - 5.0
    assert sh.std() > 1.0                               # aug actually applied
    m = np.asarray(mask).astype(bool)
    raw_t = np.stack([np.arange(o, o + 3).clip(max=n_hits - 1)
                      for o in offsets[ev]]).astype(np.float32)
    got_t = np.asarray(hits[..., 3])
    assert np.allclose((got_t - (raw_t + sh[:, None]))[m], 0.0, atol=1e-4)


def test_extra_loss_scales_with_lam():
    offsets = np.array([0, 8, 16, 24, 32])
    data = DeviceData(
        t=jnp.zeros(32), pmt_id=jnp.zeros(32, jnp.int32),
        event_id=jnp.asarray(np.repeat(np.arange(4), 8), jnp.int32),
        hyp=jnp.zeros((4, 7)).at[:, 6].set(jnp.array([1.0, 3.0, 6.0, 9.0])),
        charge=jnp.zeros((4, 2)), pmt_pos=jnp.asarray([[0.0, 0.0, 0.0]]),
    )
    spec = IdentityBatchSpec(offsets=jnp.asarray(offsets, jnp.int32),
                             elig=jnp.arange(4, dtype=jnp.int32))
    net = TiltedNet(0.1)
    vals = []
    for lam in (1.0, 5.0):
        cfg = CFG._replace(lam=lam, n_events=32, n_pad=8, time_sigma=0.0, min_half=2)
        loss = make_score_identity_loss(zero_chargenet, spec, cfg)
        vals.append(float(loss(net, data, jax.random.PRNGKey(0))))
    assert vals[0] != 0.0  # min_half gate must not have zeroed the penalty
    assert vals[1] == pytest.approx(5.0 * vals[0], rel=1e-5)
