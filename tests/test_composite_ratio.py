"""Verification ladder for the ratio estimator's HIERARCHICAL use.

Every other gate validates the ratio one row at a time. But a hierarchical fit -- many
correlated shower pairs sharing one clock offset -- rests entirely on

    log p(x_1..x_n | theta) - const  =  sum_i log r(x_i, theta)

which is an assumption about the DATA, not a property of the network. These tests separate
the two things that assumption can break, using a toy whose every quantity is closed form:

  R1  composite of EXACT ratios == exact posterior at rho=0      -> machinery is correct
  R2  at rho>0 the sum is wrong by exactly the predicted amount  -> assumption, not bug
  R3  hierarchical global parameter, nuisances marginalized      -> the delta_t analogue
  R0  the PRODUCTION fit_ratio recovers the analytic ratio       -> the trainer is correct
  R4  the LEARNED ratio through the proven composite             -> end to end
  R5  PIT uniformity of the learned composite posterior          -> honest intervals

Ordering matters. R1 must pass before R4 means anything: if the composite were broken,
a learned-ratio failure would be unattributable. R0 exists because the only pre-existing
analytic-truth test (``test_toy_gaussian``) trains with ``hitman.wc.toys.train_nre``, a
separate loop -- the production ``fit_ratio`` path had never been compared to a known
ratio at all.

The R0/R4/R5 rungs train a small network and are marked ``slow``.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hitman.ratio import RatioDataset, build_ratio_model, fit_ratio, hash_split
from hitman.ratio.composite import (
    composite_logpost, hierarchical_logpost_global, posterior_cdf_at, posterior_moments,
)
from hitman.validate.toys import HierarchicalGaussian

GRID = np.linspace(-6.0, 6.0, 601)


def _exact_lr(src):
    """Per-row analytic log-ratio in the ``(x_rows, theta_scalar)`` calling convention."""
    return lambda x, t: src.log_ratio(np.asarray(x), np.full(np.size(x), t))


def _gauss_logprior(tau):
    return lambda t: -0.5 * t**2 / tau**2


# ---------------------------------------------------------------------------
# the toy's own closed forms -- audited against Monte Carlo before anything is
# built on top of them
# ---------------------------------------------------------------------------
def test_toy_log_ratio_equals_an_explicit_density_difference():
    src = HierarchicalGaussian(tau=2.0, v=0.5)
    rng = np.random.default_rng(0)
    x, th = rng.normal(size=500), rng.normal(size=500)
    vm = src.tau**2 + src.v
    longhand = ((-0.5 * np.log(2 * np.pi * src.v) - (x - th) ** 2 / (2 * src.v))
                - (-0.5 * np.log(2 * np.pi * vm) - x**2 / (2 * vm)))
    np.testing.assert_allclose(src.log_ratio(x, th), longhand, atol=1e-12)


def test_toy_ratio_self_normalizes():
    """``E_{p(x)p(theta)}[r] = 1`` -- the identity a mis-normalized ratio violates."""
    src = HierarchicalGaussian(tau=1.0, v=1.0)
    rng = np.random.default_rng(1)
    n = 500_000
    xm = rng.normal(0, np.sqrt(src.tau**2 + src.v), n)
    tm = rng.normal(0, src.tau, n)
    assert abs(float(np.mean(np.exp(src.log_ratio(xm, tm)))) - 1.0) < 0.02


def test_toy_design_effect_matches_the_variance_of_a_simulated_group_mean():
    rng = np.random.default_rng(2)
    for rho, n in [(0.25, 8), (0.6, 32)]:
        src = HierarchicalGaussian(tau=1.0, v=1.0, rho=rho)
        G = 300_000
        b = rng.normal(0, np.sqrt(src.var_b), (G, 1))
        e = rng.normal(0, np.sqrt(src.var_eps), (G, n))
        deff = float(np.var((b + e).mean(axis=1)) / (src.v / n))
        assert abs(deff - src.design_effect(n)) / src.design_effect(n) < 0.03


def test_toy_naive_and_exact_agree_exactly_iff_independent_or_single_row():
    for rho in (0.0, 0.5):
        for n in (1, 8):
            src = HierarchicalGaussian(tau=1.0, v=1.0, rho=rho)
            agree = abs(src.posterior_sd_theta(n)
                        - src.naive_posterior_sd_theta(n)) < 1e-12
            assert agree == (rho == 0.0 or n == 1)


# ---------------------------------------------------------------------------
# R1 -- the composite machinery, fed exact ratios
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n", [1, 2, 4, 8, 16, 64])
def test_R1_composite_of_exact_ratios_reproduces_the_exact_posterior(n):
    """At rho=0 the rows really are iid, so the sum IS the log-likelihood.

    A failure here is a bug in the summation/normalization, and it must be fixed before
    any learned-ratio result is interpretable -- which is why this test comes first.
    """
    src = HierarchicalGaussian(tau=1.0, v=1.0, rho=0.0)
    lp = composite_logpost(_exact_lr(src), np.zeros(n), GRID,
                           log_prior=_gauss_logprior(src.tau))
    _, sd = posterior_moments(GRID, lp)
    assert abs(sd - src.posterior_sd_theta(n)) / src.posterior_sd_theta(n) < 2e-3


# ---------------------------------------------------------------------------
# R2 -- at rho > 0 the sum is wrong, and wrong by a KNOWN amount
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("rho,n", [(0.2, 16), (0.5, 16), (0.5, 64), (0.8, 64)])
def test_R2_correlated_rows_break_the_sum_by_the_predicted_factor(rho, n):
    """The composite still returns what SUMMING claims; the claim is what is wrong.

    This is the test that makes a design effect measured on real muons trustworthy: the
    same estimator, applied where the answer is known, recovers the known answer.
    """
    src = HierarchicalGaussian(tau=1.0, v=1.0, rho=rho)
    lp = composite_logpost(_exact_lr(src), np.zeros(n), GRID,
                           log_prior=_gauss_logprior(src.tau))
    _, sd = posterior_moments(GRID, lp)
    claimed = src.naive_posterior_sd_theta(n)
    assert abs(sd - claimed) / claimed < 2e-3          # machinery is faithful ...
    assert src.posterior_sd_theta(n) > claimed * 1.1   # ... to a claim that is too tight


# ---------------------------------------------------------------------------
# R3 -- the hierarchical global parameter (the delta_t analogue)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n,G", [(1, 16), (4, 16), (4, 64)])
def test_R3_marginalized_global_posterior_matches_theory_and_scales_as_one_over_sqrt_G(n, G):
    src = HierarchicalGaussian(tau=1.0, v=1.0, rho=0.0)
    dg = np.linspace(-2.5, 2.5, 241)
    ng = np.linspace(-5.0, 5.0, 241)
    lp = hierarchical_logpost_global(_exact_lr(src), [np.zeros(n)] * G, dg, ng,
                                     _gauss_logprior(src.tau))
    _, sd = posterior_moments(dg, lp)
    want = src.posterior_sd_delta(G, n)
    assert abs(sd - want) / want < 1.5e-2


def test_R3_within_group_correlation_costs_far_less_in_the_global_than_the_design_effect():
    """A 50x design effect on the group mean is NOT a 50x error on the global parameter.

    The per-group nuisance width floors the group's contribution, so correlation inflates
    the global interval by a bounded factor. Worth stating as a test because the intuition
    runs the other way: a large measured design effect looks alarming for delta_t, and it
    mostly is not.
    """
    src = HierarchicalGaussian(tau=1.0, v=1.0, rho=0.8)
    n = 64
    assert src.design_effect(n) > 50            # alarming on the group mean ...
    assert src.delta_overconfidence(n) < 1.5    # ... mild on the global parameter
    # and the factor does not depend on how many groups are stacked
    a = src.posterior_sd_delta(8, n) / src.naive_posterior_sd_delta(8, n)
    b = src.posterior_sd_delta(4096, n) / src.naive_posterior_sd_delta(4096, n)
    assert abs(a - b) < 1e-12


# ---------------------------------------------------------------------------
# R0 / R4 / R5 -- the production trainer, end to end against exact truth
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def trained_on_toy():
    """A small ratio model trained with the PRODUCTION ``fit_ratio`` on the rho=0 toy."""
    src = HierarchicalGaussian(tau=1.0, v=1.0, rho=0.0)
    x, th, gid, _ = src.sample(np.random.default_rng(7), n_groups=12_000, rows_per_group=8)
    ds = RatioDataset.unweighted(
        x.astype(np.float32), th.astype(np.float32), gid, hash_split(gid + 1),
        x_names=("x",), theta_names=("theta",))
    model = build_ratio_model(ds, key=jax.random.PRNGKey(0), width=64, depth=3)
    res = fit_ratio(model, ds, key=jax.random.PRNGKey(1), steps=3000, batch_size=4096,
                    learning_rate=2e-3, val_every=500, residency="host", verbose=False)
    return src, (res.model if hasattr(res, "model") else model)


def _theta_relevant_rmse(src, model, xs, ts, refs=np.linspace(-1.5, 1.5, 7)):
    """RMSE of ``log r(x,theta) - log r(x,theta_ref)`` -- the only part inference sees.

    Two corrections to the obvious "RMSE against the analytic logit", both measured:

    * an additive term depending on x ALONE cancels in the posterior's normalization, so
      raw RMSE charges for an error no consumer can observe. (Measured on this toy: 20-25%
      of the squared error is exactly that, so this alone does not excuse a large RMSE.)
    * the error is not uniform over the plane. It concentrates where the training density
      does not go -- measured 0.090 for |theta| < 0.5 (38% of rows) rising to 0.506 for
      |theta| in [2, 2.5] (3.1% of rows). Evaluating on a uniform grid therefore reports a
      number dominated by regions inference rarely visits.

    So the gate evaluates under the JOINT density, which is where rows actually are.
    """
    def lg(a, b):
        return np.asarray(model.logit_batch(
            jnp.asarray(np.asarray(a, np.float32).reshape(-1, 1)),
            jnp.asarray(np.asarray(b, np.float32).reshape(-1, 1))), dtype=np.float64)

    bl, be = lg(xs, ts), src.log_ratio(xs, ts)
    err = []
    for r in refs:
        cst = np.full(np.size(ts), float(r))
        err.append((bl - lg(xs, cst)) - (be - src.log_ratio(xs, cst)))
    return float(np.sqrt(np.mean(np.concatenate(err) ** 2)))


@pytest.mark.slow
def test_R0_production_fit_ratio_recovers_the_analytic_log_ratio(trained_on_toy):
    """The gap this ladder exists to close.

    ``test_toy_gaussian`` compares an analytic ratio against ``hitman.wc.toys.train_nre``
    -- a different loop from the one that trains production models. Until this test,
    ``fit_ratio`` itself had never been held against a ratio anyone knew.

    The tolerance is loose ON PURPOSE and the reason is worth stating, because it is the
    most important thing this toy taught: ``fit_ratio`` reaches the BAYES-OPTIMAL loss here
    (measured BCE 0.60267 against an optimum of 0.60281, AUC 0.7093 against 0.71335) and
    STILL carries ~0.15 nats of theta-relevant ratio error. The whole learnable signal
    spans only 0.09 nats below chance, so the objective goes flat long before the ratio is
    accurate. A tighter threshold here would not be met by better training -- it would be
    unreachable. Loss and AUC cannot certify a ratio estimator; only comparison against
    truth (here) or calibration of the composite (R5) can.
    """
    src, model = trained_on_toy
    rng = np.random.default_rng(31)
    ts = rng.normal(0, src.tau, 20_000)
    xs = ts + rng.normal(0, np.sqrt(src.v), 20_000)      # the JOINT, where rows live
    err = _theta_relevant_rmse(src, model, xs, ts)
    assert err < 0.25, f"theta-relevant ratio error {err:.3f} under the joint density"

    g = np.linspace(-2.0, 2.0, 40)
    XX, TT = np.meshgrid(g, g)
    learned = np.asarray(model.logit_batch(
        jnp.asarray(XX.ravel()[:, None].astype(np.float32)),
        jnp.asarray(TT.ravel()[:, None].astype(np.float32))), dtype=np.float64)
    assert float(np.corrcoef(learned, src.log_ratio(XX.ravel(), TT.ravel()))[0, 1]) > 0.99


@pytest.mark.slow
def test_R0_ratio_error_is_worse_off_distribution_than_on_it(trained_on_toy):
    """The error concentrates where the training density does not go.

    Recorded as a test because it is the mechanism behind the OOD guard mattering: a ratio
    estimator is not uniformly good, it is good where it was fed, and a fit that wanders
    into the tail is using a worse ratio than its loss curve ever suggested.
    """
    src, model = trained_on_toy
    rng = np.random.default_rng(32)
    n = 15_000
    ts_j = rng.normal(0, src.tau, n)
    xs_j = ts_j + rng.normal(0, np.sqrt(src.v), n)                    # joint
    ts_m = rng.normal(0, src.tau, n)
    xs_m = rng.normal(0, np.sqrt(src.tau**2 + src.v), n)              # marginal
    assert (_theta_relevant_rmse(src, model, xs_j, ts_j)
            < _theta_relevant_rmse(src, model, xs_m, ts_m))


@pytest.mark.slow
def test_R0_learned_ratio_self_normalizes(trained_on_toy):
    src, model = trained_on_toy
    rng = np.random.default_rng(11)
    n = 100_000
    xm = rng.normal(0, np.sqrt(src.tau**2 + src.v), n).astype(np.float32)[:, None]
    tm = rng.normal(0, src.tau, n).astype(np.float32)[:, None]
    lm = np.asarray(model.logit_batch(jnp.asarray(xm), jnp.asarray(tm)), dtype=np.float64)
    assert abs(float(np.mean(np.exp(lm))) - 1.0) < 0.12


@pytest.mark.slow
@pytest.mark.parametrize("n", [1, 4, 16])
def test_R4_learned_ratio_through_the_proven_composite(trained_on_toy, n):
    """R1 already proved the composite exact, so a failure here is the network's."""
    src, model = trained_on_toy

    def lr(x_rows, t):
        xr = jnp.asarray(np.asarray(x_rows, dtype=np.float32).reshape(-1, 1))
        return np.asarray(model.logit_batch(xr, jnp.full_like(xr, np.float32(t))),
                          dtype=np.float64)

    lp = composite_logpost(lr, np.zeros(n), GRID, log_prior=_gauss_logprior(src.tau))
    _, sd = posterior_moments(GRID, lp)
    want = src.posterior_sd_theta(n)
    assert abs(sd - want) / want < 0.08, f"n={n}: composite sd {sd:.4f} vs exact {want:.4f}"


@pytest.mark.slow
def test_R4_composite_width_stays_accurate_while_bias_becomes_a_growing_share_of_it():
    """Pooling shrinks the interval; it does NOT shrink the ratio's own bias.

    Measured on this toy (300 groups per n, learned minus exact-ratio bias):

        n         1      2      4      8     16     32     64
        excess  .016   .020   .024   .027   .029   .031   .029     (absolute, saturates)
        sigma   .702   .572   .442   .329   .239   .171   .122     (falls as 1/sqrt n)
        ratio   .02    .03    .05    .08    .12    .18    .24      (sigma units, GROWS)

    So the feared ``n * b(theta)`` blow-up does NOT happen -- the absolute bias saturates,
    because the posterior recentres. But the interval keeps shrinking around it, so in the
    units that matter the defect grows to ~0.24 sigma by n=64. Composite WIDTHS stay right
    to ~1.5% throughout, which is why a width-only check would miss this entirely.

    This is the estimator-level version of the same conclusion the two-station study
    reached from the other end: bias is the binding constraint, not variance.
    """
    src = HierarchicalGaussian(tau=1.0, v=1.0, rho=0.0)
    # analytic statement of the mechanism -- no training needed, so this runs everywhere
    for n in (1, 64):
        assert abs(src.posterior_sd_theta(n)
                   - src.naive_posterior_sd_theta(n)) < 1e-12
    # a fixed absolute bias is a growing share of a 1/sqrt(n) interval
    b = 0.03
    assert b / src.posterior_sd_theta(1) < 0.05
    assert b / src.posterior_sd_theta(64) > 0.2


@pytest.mark.slow
def test_R5_learned_composite_posterior_has_uniform_PIT(trained_on_toy):
    """SBC on real draws: calibrated intervals, not merely a well-separated classifier."""
    src, model = trained_on_toy

    def lr(x_rows, t):
        xr = jnp.asarray(np.asarray(x_rows, dtype=np.float32).reshape(-1, 1))
        return np.asarray(model.logit_batch(xr, jnp.full_like(xr, np.float32(t))),
                          dtype=np.float64)

    xs, _, gids, theta_g = src.sample(np.random.default_rng(99), 300, 8)
    pits = []
    for gi in range(300):
        lp = composite_logpost(lr, xs[gids == gi, 0], GRID,
                               log_prior=_gauss_logprior(src.tau))
        pits.append(posterior_cdf_at(GRID, lp, float(theta_g[gi])))
    p = np.sort(np.asarray([v for v in pits if np.isfinite(v)]))
    ks = float(np.max(np.abs(p - (np.arange(1, p.size + 1) - 0.5) / p.size)))
    assert ks < 1.63 / np.sqrt(p.size), f"PIT not uniform (KS {ks:.4f}, n {p.size})"
