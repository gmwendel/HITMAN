"""Proposal 2: pluggable density-factor heads.

Two jobs. (1) Each extracted factor is an exactly-normalized :class:`DensityFactor`
(closed-form partition, sums/integrates to 1). (2) The EQUIVALENCE test: the monolithic
water-Cherenkov ``SplineMLE`` per-event log-likelihood is reconstructed, bit-for-bit to
float64 tolerance, from a ``SoftmaxMarkFactor`` x ``LogSplineFactor`` x ``CountFactor``
composition — proving the WC model is exactly the composition of the extracted pieces.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hitman.density import (CountFactor, DensityFactor, LogSplineFactor, SoftmaxMarkFactor,
                            marked_poisson_loglik)
from hitman.splinemle import DEFAULT_KNOTS, SplineMLE


def _toy_geometry(seed=1):
    rng = np.random.default_rng(seed)
    pmt_pos = jnp.asarray(rng.normal(size=(241, 3)) * 400.0, jnp.float32)
    nrm = rng.normal(size=(241, 3))
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    return pmt_pos, jnp.asarray(nrm, jnp.float32)


def _toy_model(width=48, key=0, **kw):
    pmt_pos, pmt_normal = _toy_geometry()
    return SplineMLE(pmt_pos, pmt_normal, key=jax.random.PRNGKey(key), width=width,
                     depth=3, **kw)


# ---------------------------------------------------------------------------
# (a) protocol conformance
# ---------------------------------------------------------------------------
def test_factors_conform_to_protocol():
    assert isinstance(SoftmaxMarkFactor(), DensityFactor)
    assert isinstance(LogSplineFactor(knots=DEFAULT_KNOTS), DensityFactor)
    assert isinstance(CountFactor(), DensityFactor)


# ---------------------------------------------------------------------------
# (b) each factor is exactly normalized
# ---------------------------------------------------------------------------
def test_logspline_factor_integrates_to_one():
    jax.config.update("jax_enable_x64", True)
    try:
        f = LogSplineFactor(knots=DEFAULT_KNOTS)
        rng = np.random.default_rng(0)
        nv = jnp.asarray(rng.normal(size=len(DEFAULT_KNOTS)) * 1.5, jnp.float64)
        ug = np.linspace(DEFAULT_KNOTS[0] + 1e-4, DEFAULT_KNOTS[-1] - 1e-4, 2_000_001)
        dens = np.asarray(f.density(jnp.asarray(ug), nv))
        assert abs(np.trapezoid(dens, ug) - 1.0) < 1e-4
        # closed-form partition matches the wrapped free function
        from hitman.density import logZ_time
        assert float(f.log_partition(nv)) == float(logZ_time(nv, f.knots_arr))
    finally:
        jax.config.update("jax_enable_x64", False)


def test_softmax_factor_sums_to_one():
    f = SoftmaxMarkFactor()
    eta = jax.random.normal(jax.random.PRNGKey(0), (241,))
    ls = jax.vmap(lambda s: f.log_prob(s, eta))(jnp.arange(241))
    assert abs(float(jnp.sum(jnp.exp(ls))) - 1.0) < 1e-5
    assert float(f.log_partition(eta)) == float(jax.scipy.special.logsumexp(eta))


def test_count_factor_normalizes_over_N():
    for count_model in ("nbinom", "poisson"):
        cf = CountFactor(count_model=count_model)
        for x, log_int in ((2.5, np.log(60.0)), (9.5, np.log(150.0))):
            Ns = jnp.arange(0.0, 5000.0)
            lp = jax.vmap(lambda N: cf.log_prob(N, jnp.asarray(log_int), jnp.asarray(x)))(Ns)
            assert abs(float(jnp.sum(jnp.exp(lp))) - 1.0) < 1e-4


def test_marked_poisson_composition_masks_pads():
    count_ll = jnp.asarray(-3.0)
    mark_lls = jnp.asarray([-1.0, -2.0, -5.0, -7.0])
    mask = jnp.asarray([1.0, 1.0, 0.0, 0.0])
    got = float(marked_poisson_loglik(count_ll, mark_lls, mask))
    assert abs(got - (-3.0 + -1.0 + -2.0)) < 1e-6


# ---------------------------------------------------------------------------
# (c) EQUIVALENCE: SplineMLE == SoftmaxMarkFactor x LogSplineFactor x CountFactor
# ---------------------------------------------------------------------------
def _count_factor_from_model(m):
    """A CountFactor whose trainable leaves are copied from a SplineMLE (same numbers)."""
    cf = CountFactor(phi_knots=m.phi_knots, count_model=m.count_model)
    return eqx.tree_at(lambda c: (c.phi_e0, c.phi_raw, c.disp), cf,
                       (m.phi_e0, m.phi_raw, m.disp))


def test_wc_model_reconstructed_from_factors():
    jax.config.update("jax_enable_x64", True)
    try:
        for count_model in ("nbinom", "poisson"):
            m = _toy_model(count_model=count_model)
            soft = SoftmaxMarkFactor()
            spline = LogSplineFactor(knots=m.knots)     # floor=-inf (exact model)
            cf = _count_factor_from_model(m)

            rng = np.random.default_rng(4)
            n_events = 5
            max_tol = 0.0
            for _ in range(n_events):
                theta = jnp.asarray(
                    np.concatenate([rng.normal(size=5) * 100.0,
                                    [rng.uniform(0, 6)],       # t
                                    [rng.uniform(0.5, 9.5)]]), jnp.float64)  # E
                eta, nodes, t_geo, _ = m.event_tables(theta)
                E = theta[6]
                sensors = rng.integers(0, 241, size=8)
                # place hit times in-support so both paths evaluate the interior density
                ts = np.asarray(t_geo)[sensors] + rng.uniform(1.0, 30.0, size=8)

                # monolithic SplineMLE per-event log-lik (sensor + time marks + NB2 count)
                mono_hits = sum(
                    float(m.log_prob_hit(jnp.asarray(t), int(s), theta))
                    for s, t in zip(sensors, ts))
                log_Lambda = m.phi(E) + jax.scipy.special.logsumexp(eta)
                mono = mono_hits + float(m.log_count(jnp.asarray(float(len(sensors))),
                                                     log_Lambda, E))

                # composed-from-factors per-event log-lik
                log_int = soft.log_partition(eta)          # == logsumexp(eta)
                mark_lls = jnp.asarray([
                    soft.log_prob(int(s), eta)
                    + spline.log_prob(jnp.asarray(t) - t_geo[int(s)], nodes[int(s)])
                    for s, t in zip(sensors, ts)])
                comp_count = cf.log_prob(jnp.asarray(float(len(sensors))), log_int, E)
                comp = float(marked_poisson_loglik(comp_count, mark_lls))

                max_tol = max(max_tol, abs(mono - comp))
            assert max_tol < 1e-9, f"{count_model}: max |mono - composed| = {max_tol:.2e}"
    finally:
        jax.config.update("jax_enable_x64", False)


def test_count_factor_matches_model_log_count():
    m = _toy_model()
    cf = _count_factor_from_model(m)
    for E in (2.5, 9.5):
        for N in (0.0, 37.0, 120.0):
            log_int = np.log(80.0)
            mono = float(m.log_count(jnp.asarray(N), jnp.asarray(m.phi(jnp.asarray(E)))
                                     + log_int, jnp.asarray(E)))
            comp = float(cf.log_prob(jnp.asarray(N), jnp.asarray(log_int), jnp.asarray(E)))
            assert abs(mono - comp) < 1e-5, f"E={E} N={N}: {mono} vs {comp}"


# ---------------------------------------------------------------------------
# (d) a downstream chain-of-conditionals composes with zero new normalization code
# ---------------------------------------------------------------------------
def test_chain_of_conditionals_normalizes():
    # NB2(N) x f(dt) x f(alpha | dt): a 1-D log-spline root factor and a conditional
    # log-spline whose node values depend on dt. Both are exactly normalized, so their
    # product integrates to 1 over (dt, alpha) at fixed N.
    jax.config.update("jax_enable_x64", True)
    try:
        root = LogSplineFactor(knots=(-5.0, -2.0, 0.0, 2.0, 5.0))
        cond = LogSplineFactor(knots=(-3.0, 0.0, 3.0))
        rng = np.random.default_rng(2)
        root_nodes = jnp.asarray(rng.normal(size=5), jnp.float64)

        def cond_nodes(dt):  # dt-dependent conditional shape (any smooth map is fine)
            return jnp.asarray([0.2 * dt, -0.1 * dt, 0.05 * dt], jnp.float64)

        dt_g = np.linspace(-5.0 + 1e-4, 5.0 - 1e-4, 4001)
        a_g = np.linspace(-3.0 + 1e-4, 3.0 - 1e-4, 4001)
        # marginal over dt of [f(dt) * integral f(alpha|dt) dalpha] must be 1 (inner = 1)
        inner = np.array([
            float(np.trapezoid(np.asarray(cond.density(jnp.asarray(a_g), cond_nodes(dt))),
                               a_g))
            for dt in dt_g[::200]])
        assert np.allclose(inner, 1.0, atol=1e-4)
        joint = np.asarray(root.density(jnp.asarray(dt_g), root_nodes))
        assert abs(np.trapezoid(joint, dt_g) - 1.0) < 1e-4
    finally:
        jax.config.update("jax_enable_x64", False)
