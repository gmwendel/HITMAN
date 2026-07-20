"""NPE arm: flow correctness, conditioning/calibration, and batch/aug coherence.

CPU-fast (target < 90 s). Precision is pinned to float32-highest by conftest; the
numerical-Jacobian check uses autodiff (exact derivative) so float32 tolerances suffice.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from scipy import stats

from hitman.npe import (ConditionalFlow, DeepSetsEncoder, flow_nll, make_npe_batch,
                        sbc_ranks, tarp_coverage)
from hitman.npe.loss import npe_loss
from hitman.train.resident import DeviceData

TWO_PI = 2.0 * np.pi


# ---------------------------------------------------------------------------
# 1. Flow correctness: invertibility, Jacobian, periodic azimuth
# ---------------------------------------------------------------------------
def _small_flow(key, **kw):
    return ConditionalFlow(context_dim=4, key=key, n_layers=4, n_bins=6, hidden=24,
                           cond_depth=2, **kw)


def test_flow_sample_logprob_roundtrip():
    flow = _small_flow(jax.random.PRNGKey(0))
    c = jax.random.normal(jax.random.PRNGKey(1), (4,))
    # interior hypotheses (well inside the prior box, away from logit boundaries)
    thetas = jnp.array([
        [100.0, -50.0, 200.0, 1.2, 3.0, 5.0, 4.0],
        [-300.0, 400.0, -100.0, 2.0, 0.5, -30.0, 7.5],
        [0.0, 0.0, 0.0, 1.5708, 1.0, 0.0, 1.0],
    ])
    for th in thetas:
        z, ld = flow.forward_to_base(th, c)
        th2 = flow.inverse_from_base(z, c)
        assert np.max(np.abs(np.asarray(th2 - th))) < 1e-2
        assert np.isfinite(float(flow.log_prob(th, c)))
    # a drawn sample scores finite too
    s = flow.sample(jax.random.PRNGKey(2), c)
    assert np.isfinite(float(flow.log_prob(s, c)))


def test_flow_logprob_matches_numerical_jacobian():
    flow = _small_flow(jax.random.PRNGKey(3))
    c = jax.random.normal(jax.random.PRNGKey(4), (4,))
    th = jnp.array([120.0, -40.0, 150.0, 1.1, 2.5, 8.0, 3.5])
    z, ld = flow.forward_to_base(th, c)
    # exact Jacobian of theta -> z via autodiff; its log|det| must equal the accumulated
    # analytic log-det that log_prob relies on.
    J = jax.jacfwd(lambda t: flow.forward_to_base(t, c)[0])(th)
    _, logdet = jnp.linalg.slogdet(J)
    assert abs(float(ld) - float(logdet)) < 1e-2
    # and log_prob = base_log_prob(z) + logdet
    assert abs(float(flow.log_prob(th, c)) - float(flow.base_log_prob(z) + ld)) < 1e-4


def test_flow_periodic_azimuth_wraps():
    flow = _small_flow(jax.random.PRNGKey(5))
    c = jax.random.normal(jax.random.PRNGKey(6), (4,))
    base = jnp.array([50.0, 50.0, 50.0, 1.3, 0.0, 2.0, 5.0])
    lp0 = float(flow.log_prob(base.at[4].set(1e-4), c))
    lp1 = float(flow.log_prob(base.at[4].set(TWO_PI - 1e-4), c))
    assert abs(lp0 - lp1) < 1e-2
    # sampled azimuth lives on the circle
    s = flow.sample_n(jax.random.PRNGKey(7), c, 64)
    az = np.asarray(s[:, 4])
    assert az.min() >= 0.0 and az.max() < TWO_PI + 1e-4


# ---------------------------------------------------------------------------
# 2. Conditioning: 2-D linear-Gaussian toy; posterior mean, SBC, TARP.
#    theta ~ N(0, I); x = theta + N(0, sigma^2 I); posterior N(x/(1+sigma^2), .).
#    A shared module fixture trains one flow (~15 s CPU) reused by all three tests.
# ---------------------------------------------------------------------------
SIGMA = 0.6
POST_SCALE = 1.0 / (1.0 + SIGMA**2)                 # posterior mean = x * POST_SCALE
import equinox as eqx  # noqa: E402


def _toy_flow(key):
    return ConditionalFlow(context_dim=2, key=key, n_dim=2, box={0: (-6.0, 6.0),
                           1: (-6.0, 6.0)}, cos_dims=(), circular_dims=(), n_layers=5,
                           n_bins=8, tail_bound=4.0, hidden=48, cond_depth=2)


def _toy_joint(key, n):
    """Standard joint samples: theta ~ prior N(0,I), x ~ N(theta, sigma^2 I)."""
    kt, kx = jax.random.split(key)
    theta = jax.random.normal(kt, (n, 2))
    x = theta + SIGMA * jax.random.normal(kx, (n, 2))
    return x, theta


@pytest.fixture(scope="module")
def toy_flow():
    ctx, theta = _toy_joint(jax.random.PRNGKey(11), 8000)
    flow = _toy_flow(jax.random.PRNGKey(12))
    opt = optax.adam(2e-3)
    st = opt.init(eqx.filter(flow, eqx.is_inexact_array))

    @eqx.filter_jit
    def step(flow, st, c, t):
        loss, g = eqx.filter_value_and_grad(lambda m: flow_nll(m, c, t))(flow)
        upd, st = opt.update(g, st)
        return eqx.apply_updates(flow, upd), st, loss

    n = ctx.shape[0]
    for i in range(3500):
        idx = jax.random.randint(jax.random.PRNGKey(i), (1500,), 0, n)
        flow, st, _ = step(flow, st, ctx[idx], theta[idx])
    return flow


def test_conditioning_posterior_mean_matches_analytic(toy_flow):
    x_test = jnp.asarray(np.linspace(-2.0, 2.0, 20)[:, None] * np.ones((1, 2)))
    analytic = np.asarray(x_test) * POST_SCALE
    keys = jax.random.split(jax.random.PRNGKey(13), x_test.shape[0])
    means = np.asarray(jax.vmap(
        lambda c, k: jnp.mean(toy_flow.sample_n(k, c, 400), axis=0))(x_test, keys))
    rmse = float(np.sqrt(np.mean((means - analytic) ** 2)))
    assert rmse < 0.15, f"posterior-mean RMSE {rmse:.3f} too large"
    # tracks the analytic slope, not a constant
    slope = np.polyfit(analytic.ravel(), means.ravel(), 1)[0]
    assert 0.8 < slope < 1.2, f"slope {slope:.3f}"


def test_conditioning_sbc_ranks_uniform(toy_flow):
    # fresh joint draw for SBC: theta ~ prior, x ~ likelihood
    kth, kx = jax.random.split(jax.random.PRNGKey(23))
    theta_true = jax.random.normal(kth, (300, 2))
    x = theta_true + SIGMA * jax.random.normal(kx, (300, 2))
    ranks = sbc_ranks(toy_flow, x, theta_true, jax.random.PRNGKey(24), n_samples=99)
    for d in range(2):
        u = (ranks[:, d] + 0.5) / 100.0
        p = stats.kstest(u, "uniform").pvalue
        assert p > 0.01, f"dim {d} SBC ranks non-uniform (KS p={p:.4f})"


def test_tarp_returns_coverage_curve(toy_flow):
    kth, kx = jax.random.split(jax.random.PRNGKey(33))
    theta_true = jax.random.normal(kth, (300, 2))
    x = theta_true + SIGMA * jax.random.normal(kx, (300, 2))
    alpha, cov, atarp = tarp_coverage(toy_flow, x, theta_true, jax.random.PRNGKey(34),
                                      n_samples=64)
    assert alpha.shape == cov.shape
    assert cov[-1] == pytest.approx(1.0)
    assert np.all(np.diff(cov) >= -1e-9)               # ECDF monotone
    # roughly diagonal for a calibrated toy posterior
    assert np.mean(np.abs(cov - alpha)) < 0.12


# ---------------------------------------------------------------------------
# 3. Event-grouped batching + coherent time augmentation
# ---------------------------------------------------------------------------
def test_make_npe_batch_gather_and_aug():
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
    spec = _spec_from_offsets(offsets, n_pad=3)         # eligible = events with <=3 hits

    hits, mask, charge, theta = make_npe_batch(
        data, spec, jax.random.PRNGKey(7), n_events=64, n_pad=3, time_sigma=25.0)
    assert hits.shape == (64, 3, 4) and mask.shape == (64, 3)
    ev = np.asarray(charge[:, 0]).astype(int)           # charge col 0 encodes event id
    elig = np.flatnonzero(np.diff(offsets) <= 3)
    assert set(ev) <= set(elig.tolist())                # oversize event never sampled
    assert np.array_equal(np.asarray(mask).sum(1), np.diff(offsets)[ev])
    # aug coherence: theta[5] shift == hit-time shift, identical across an event's hits
    sh = np.asarray(theta[:, 5]) - 5.0                  # base hyp[5] == 5.0
    assert sh.std() > 1.0                               # aug actually applied
    m = np.asarray(mask).astype(bool)
    raw_t = np.stack([np.arange(o, o + 3).clip(max=n_hits - 1)
                      for o in offsets[ev]]).astype(np.float32)
    got_t = np.asarray(hits[..., 3])
    assert np.allclose((got_t - (raw_t + sh[:, None]))[m], 0.0, atol=1e-4)
    # padding is zeroed
    assert np.allclose(got_t[~m], 0.0)


def _spec_from_offsets(offsets, n_pad):
    from hitman.train.identities import IdentityBatchSpec
    elig = np.flatnonzero(np.diff(offsets) <= n_pad)
    return IdentityBatchSpec(offsets=jnp.asarray(offsets, jnp.int32),
                             elig=jnp.asarray(elig, jnp.int32))


def test_npe_loss_finite_on_event_batch():
    # tiny end-to-end: encoder + flow on a hand-built batch
    offsets = np.array([0, 3, 7, 9, 14, 20])
    n_hits = 20
    rng = np.random.default_rng(0)
    data = DeviceData(
        t=jnp.asarray(rng.normal(size=n_hits).astype(np.float32)),
        pmt_id=jnp.asarray(rng.integers(0, 4, n_hits), jnp.int32),
        event_id=jnp.asarray(np.repeat(np.arange(5), np.diff(offsets)), jnp.int32),
        hyp=jnp.asarray(np.column_stack([
            rng.uniform(-100, 100, (5, 3)), rng.uniform(0, np.pi, 5),
            rng.uniform(0, TWO_PI, 5), rng.uniform(-5, 5, 5),
            rng.uniform(0, 10, 5)]).astype(np.float32)),
        charge=jnp.asarray(rng.uniform(1, 50, (5, 2)).astype(np.float32)),
        pmt_pos=jnp.asarray(rng.uniform(-500, 500, (4, 3)).astype(np.float32)),
    )
    spec = _spec_from_offsets(offsets, n_pad=8)
    batch = make_npe_batch(data, spec, jax.random.PRNGKey(1), n_events=16, n_pad=8,
                           time_sigma=50.0)
    enc = DeepSetsEncoder(key=jax.random.PRNGKey(2), context_dim=16, phi_dim=16,
                          phi_hidden=16, rho_hidden=16)
    flow = ConditionalFlow(context_dim=16, key=jax.random.PRNGKey(3), n_layers=4,
                           n_bins=6, hidden=24, cond_depth=2)
    loss = float(npe_loss((enc, flow), batch))
    assert np.isfinite(loss)
