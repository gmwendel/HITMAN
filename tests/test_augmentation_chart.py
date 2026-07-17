import jax
import jax.numpy as jnp
import numpy as np

from hitman.inference.mle import chart_from_theta, theta_from_chart
from hitman.train import DeviceData, hit_batch
from tests.test_resident import _toy_data


def test_time_shuffle_augmentation_coherent():
    batch, pmt_pos = _toy_data(n_events=50, seed=2)
    data = DeviceData.from_batch(batch, pmt_pos)
    rows = jnp.arange(data.n_hits)
    obs0, hyp0 = hit_batch(data, rows)  # no augmentation
    obs1, hyp1 = hit_batch(data, rows, key=jax.random.PRNGKey(0), time_sigma=50.0)

    # times really moved
    dt_shift = np.asarray(obs1[:, 3] - obs0[:, 3])
    assert np.abs(dt_shift).mean() > 5.0
    # hit time and its own event's hypothesis time moved together (dt invariant)
    np.testing.assert_allclose(
        np.asarray(obs1[:, 3] - hyp1[:, 5]),
        np.asarray(obs0[:, 3] - hyp0[:, 5]),
        atol=1e-4,
    )
    # shift is per-event coherent: same event -> same shift
    ev = np.asarray(batch.event_id)
    for e in (0, 3):
        s = dt_shift[ev == e]
        assert s.std() < 1e-4
    # positions untouched
    np.testing.assert_array_equal(np.asarray(obs1[:, :3]), np.asarray(obs0[:, :3]))


def test_direction_chart_roundtrip():
    theta = jnp.array([100.0, -50.0, 200.0, 0.7, 2.3, 1.5, 3.0])
    back = np.asarray(theta_from_chart(chart_from_theta(theta)))
    np.testing.assert_allclose(back, np.asarray(theta), rtol=1e-5, atol=1e-5)


def test_direction_chart_smooth_near_pole():
    # gradient through the chart stays finite arbitrarily close to zenith 0
    def f(u):
        th = theta_from_chart(u)
        d = jnp.array([jnp.sin(th[3]) * jnp.cos(th[4]),
                       jnp.sin(th[3]) * jnp.sin(th[4]),
                       jnp.cos(th[3])])
        return jnp.sum(d * jnp.array([0.3, 0.5, 0.8]))

    u = chart_from_theta(jnp.array([0.0, 0.0, 0.0, 1e-4, 0.3, 0.0, 3.0]))
    g = np.asarray(jax.grad(f)(u))
    assert np.isfinite(g).all()


def test_chart_unnormalized_direction_ok():
    u = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 3.0])  # |d| = 5, +z
    th = np.asarray(theta_from_chart(u))
    assert abs(th[3]) < 2e-3  # zenith ~ 0
