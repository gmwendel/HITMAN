import jax.numpy as jnp
import numpy as np

from hitman.wc.nn import features as ft


def test_hit_features_match_1x_definition():
    hit = jnp.array([500.0, -250.0, 100.0, 30.0])
    hyp = jnp.array([100.0, 200.0, -300.0, 0.7, 1.2, 5.0, 2.5])
    f = np.asarray(ft.hit_features(hit, hyp))
    assert f.shape == (ft.N_HIT_FEATURES,)
    np.testing.assert_allclose(f[0:3], [0.1, 0.2, -0.3], rtol=1e-6)
    np.testing.assert_allclose(f[3], np.sin(0.7) * np.cos(1.2), rtol=1e-6)
    np.testing.assert_allclose(f[4], np.sin(0.7) * np.sin(1.2), rtol=1e-6)
    np.testing.assert_allclose(f[5], np.cos(0.7), rtol=1e-6)
    np.testing.assert_allclose(f[6], (30.0 - 5.0) / 25.0, rtol=1e-6)
    np.testing.assert_allclose(f[7], 2.5 - 1.0, rtol=1e-6)
    np.testing.assert_allclose(f[8:11], [0.5, -0.25, 0.1], rtol=1e-6)


def test_charge_features_match_1x_definition():
    charge = jnp.array([80.0, 60.0])
    hyp = jnp.array([100.0, 200.0, -300.0, 0.7, 1.2, 5.0, 2.5])
    f = np.asarray(ft.charge_features(charge, hyp))
    assert f.shape == (ft.N_CHARGE_FEATURES,)
    np.testing.assert_allclose(f[0], 80.0 / 40.0 - 1.0, rtol=1e-6)
    np.testing.assert_allclose(f[1], 60.0 / 40.0 - 1.0, rtol=1e-6)
    np.testing.assert_allclose(f[8], 1.5, rtol=1e-6)


def test_wrap_direction_recovers_physical_angles():
    hyp = jnp.array([0.0, 0.0, 0.0, 0.4, 5.0 * np.pi / 2.0, 0.0, 1.0])  # az = 2pi + pi/2
    wrapped = np.asarray(ft.wrap_direction(hyp))
    np.testing.assert_allclose(wrapped[3], 0.4, rtol=1e-5)
    np.testing.assert_allclose(wrapped[4], np.pi / 2.0, rtol=1e-5)
    # direction vector must be unchanged
    np.testing.assert_allclose(
        np.asarray(ft.direction(jnp.asarray(wrapped))),
        np.asarray(ft.direction(hyp)),
        atol=1e-6,
    )
