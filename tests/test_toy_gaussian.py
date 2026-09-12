"""L1 statistical closure on the analytic-ratio Gaussian source toy.

The true likelihood ratio, score, MLE and posterior are all closed form, so this is
where the *receipts themselves* are validated end-to-end against exact ground truth:

* score identity is exactly zero in expectation,
* a small NRE trained with the production loss recovers the analytic log-ratio and
  self-normalizes,
* temperature fitting recovers an injected miscalibration,
* the exact posterior passes SBC / expected-coverage,
* the analytic MLE closes bias / resolution / pull through the L2 mle_receipt.

Loose-but-meaningful tolerances; runs in well under 2 minutes on CPU.
"""

import jax
import jax.numpy as jnp
import numpy as np

from hitman.diagnostics import expected_coverage, fit_temperature, sbc_ranks
from hitman.wc.receipts.mle import mle_receipt
from hitman.receipts.score import score_identity
from hitman.wc.toys import GaussianSource, nre_logit, train_nre

SRC = GaussianSource(dim=1, s=1.0, tau=2.0)


def test_analytic_score_identity_is_zero():
    rng = np.random.default_rng(0)
    theta = np.array([0.4])
    xs = theta + SRC.s * rng.normal(size=(80_000, 1))
    res = score_identity(lambda c: (c - theta) / SRC.s**2, xs, chunk=8000)
    assert res.max_sigma < 4.0


def test_exact_posterior_passes_sbc_coverage():
    key = jax.random.PRNGKey(1)
    n_sims, n_draws = 400, 200
    kth, kx, kp = jax.random.split(key, 3)
    thetas = SRC.sample_theta(kth, n_sims)                    # (n_sims, 1)
    xs = SRC.sample_x_given_theta(kx, thetas)
    mean, std = SRC.posterior(xs)                             # vectorized over sims
    draws = mean[:, None, :] + std * jax.random.normal(kp, (n_sims, n_draws, 1))
    ranks = sbc_ranks(np.asarray(draws), np.asarray(thetas))
    levels, coverage = expected_coverage(ranks, n_draws)
    np.testing.assert_allclose(
        coverage[:, 0], levels, atol=0.09
    )  # calibrated posterior lies on the diagonal


def test_temperature_recovers_injected_miscalibration():
    key = jax.random.PRNGKey(2)
    kj, kp = jax.random.split(key)
    x, theta = SRC.sample_joint(kj, 20_000)
    theta_marg = theta[jax.random.permutation(kp, 20_000)]
    logr = jax.vmap(SRC.log_ratio)
    lj = np.asarray(logr(x, theta))
    lm = np.asarray(logr(x, theta_marg))
    # calibrated logits -> tau ~ 1
    assert abs(fit_temperature(lj, lm) - 1.0) < 0.15
    # overconfident (hot) logits scaled by c -> temperature recovers c
    for c in (1.6, 2.5):
        assert abs(fit_temperature(c * lj, c * lm) - c) < 0.2 * c


def test_mle_closure_through_receipt():
    rng = np.random.default_rng(3)
    theta0 = 0.5
    n = 4000
    x = theta0 + SRC.s * rng.normal(size=n)
    truth7 = np.array([theta0, 0, 0, np.pi / 2, 0, 0, 3.0])
    fits = np.tile(truth7, (n, 1))
    fits[:, 0] = SRC.mle(x)  # analytic MLE = x, mapped to the x-position slot
    sigmas = np.full((n, 7), 1e-6)
    sigmas[:, 0] = SRC.fisher_sigma
    res = mle_receipt(fits, truth7, sigmas=sigmas)
    xstat = next(p for p in res.params if p.name == "x")
    assert abs(xstat.bias) < 0.1                       # unbiased
    assert abs(xstat.resolution - SRC.s) < 0.1         # resolution ~ s
    assert abs(xstat.pull_sigma - 1.0) < 0.1           # Fisher pull ~ N(0,1)


def test_nre_recovers_analytic_log_ratio():
    model = train_nre(SRC, jax.random.PRNGKey(4), n_steps=2000, batch=4096)

    grid = np.linspace(-2.0, 2.0, 25)
    xs, ths = np.meshgrid(grid, grid)
    pts_x = jnp.asarray(xs.ravel()[:, None])
    pts_t = jnp.asarray(ths.ravel()[:, None])
    learned = np.asarray(jax.vmap(nre_logit, in_axes=(None, 0, 0))(model, pts_x, pts_t))
    exact = np.asarray(jax.vmap(SRC.log_ratio)(pts_x, pts_t))

    rmse = float(np.sqrt(np.mean((learned - exact) ** 2)))
    corr = float(np.corrcoef(learned, exact)[0, 1])
    assert rmse < 0.15, f"NRE did not recover analytic ratio (RMSE {rmse:.3f})"
    assert corr > 0.99

    # self-normalization on marginal draws: E_{p(x)p(theta)}[r_hat] ~ 1
    rng = np.random.default_rng(5)
    v = SRC.tau**2 + SRC.s**2
    xm = jnp.asarray(rng.normal(0, np.sqrt(v), (20_000, 1)).astype(np.float32))
    tm = jnp.asarray((SRC.tau * rng.normal(size=(20_000, 1))).astype(np.float32))
    lm = np.asarray(jax.vmap(nre_logit, in_axes=(None, 0, 0))(model, xm, tm))
    assert abs(np.mean(np.exp(lm)) - 1.0) < 0.2
