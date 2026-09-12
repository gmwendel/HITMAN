import jax
import jax.numpy as jnp
import numpy as np

from hitman.wc.inference.mle import PARAM_SCALE, cylinder_seeds, multistart_mle


def test_cylinder_seeds_respect_bounds():
    seeds = np.asarray(
        cylinder_seeds(
            jax.random.PRNGKey(0), 500, radius=1000.0, half_height=800.0,
            t_range=(-5.0, 5.0), e_range=(1.0, 3.0),
        )
    )
    r = np.hypot(seeds[:, 0], seeds[:, 1])
    assert r.max() <= 900.0 + 1e-3
    assert np.abs(seeds[:, 2]).max() <= 720.0 + 1e-3
    assert seeds[:, 3].min() >= 0.0 and seeds[:, 3].max() <= np.pi
    assert seeds[:, 6].min() >= 1.0 and seeds[:, 6].max() <= 3.0


def test_multistart_mle_recovers_synthetic_minimum():
    theta_star = jnp.array([200.0, -300.0, 150.0, 1.0, 2.0, 1.0, 2.0])

    def nll(theta):
        return jnp.sum(((theta - theta_star) / PARAM_SCALE) ** 2)

    seeds = cylinder_seeds(
        jax.random.PRNGKey(1), 1000, radius=1000.0, half_height=800.0,
        t_range=(-5.0, 5.0), e_range=(1.0, 3.0),
    )
    result = multistart_mle(nll, seeds, n_select=32, steps=400, learning_rate=3e-2)

    np.testing.assert_allclose(np.asarray(result.theta[:3]), np.asarray(theta_star[:3]), atol=5.0)
    np.testing.assert_allclose(np.asarray(result.theta[3:]), np.asarray(theta_star[3:]), atol=0.02)
    assert float(result.nll) < 1e-4


def test_multistart_mle_respects_bounds():
    # Minimum outside the box -> projected descent must stop at the box face.
    theta_star = jnp.array([2000.0, 0.0, 0.0, 1.0, 2.0, 0.0, 1.5])

    def nll(theta):
        return jnp.sum(((theta - theta_star) / PARAM_SCALE) ** 2)

    seeds = cylinder_seeds(
        jax.random.PRNGKey(3), 500, radius=900.0, half_height=900.0,
        t_range=(-5.0, 5.0), e_range=(1.0, 3.0),
    )
    lo = jnp.array([-900.0, -900.0, -900.0, 0.0, 0.0, -5.0, 0.5])
    hi = jnp.array([900.0, 900.0, 900.0, np.pi, 2 * np.pi, 5.0, 3.0])
    result = multistart_mle(nll, seeds, n_select=32, steps=400, bounds=(lo, hi))
    np.testing.assert_allclose(float(result.theta[0]), 900.0, atol=1.0)
    assert float(jnp.max(jnp.abs(result.theta[:3]))) <= 900.0 + 1e-3


def test_multistart_mle_is_jittable():
    theta_star = jnp.zeros(7).at[6].set(2.0)

    def nll(theta):
        return jnp.sum(((theta - theta_star) / PARAM_SCALE) ** 2)

    seeds = cylinder_seeds(
        jax.random.PRNGKey(2), 200, radius=500.0, half_height=500.0,
        t_range=(-5.0, 5.0), e_range=(1.0, 3.0),
    )
    jitted = jax.jit(lambda s: multistart_mle(nll, s, n_select=16, steps=50))
    result = jitted(seeds)
    assert np.isfinite(float(result.nll))
