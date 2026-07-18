import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hitman.nn import FrameHitNet, SensorFrame, compute_e_ref
from hitman.nn.frame import C_MM_PER_NS
from hitman.train import DeviceData, fit_resident, frame_hit_batch
from tests.test_resident import _toy_data


def _rot(axis, angle):
    """Rodrigues rotation matrix about a unit axis."""
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def _hyp(vertex, direction, t=0.0, e=3.0):
    d = np.asarray(direction, float)
    d = d / np.linalg.norm(d)
    zen = np.arccos(np.clip(d[2], -1, 1))
    az = np.mod(np.arctan2(d[1], d[0]), 2 * np.pi)
    return jnp.asarray([*vertex, zen, az, t, e], jnp.float32)


PMT_POS = np.array([300.0, -200.0, 500.0])
PMT_N = np.array([-0.3, 0.2, -0.933])
PMT_N = PMT_N / np.linalg.norm(PMT_N)


def test_invariance_under_rotation_about_pmt_axis():
    frame = SensorFrame()
    vertex = np.array([50.0, 80.0, -100.0])
    direction = np.array([0.3, -0.5, 0.8])
    f0 = frame(jnp.asarray(PMT_POS), jnp.asarray(PMT_N), 12.0, _hyp(vertex, direction))
    for angle in (0.7, 2.0, -1.3):
        R = _rot(PMT_N, angle)
        v_rot = PMT_POS + R @ (vertex - PMT_POS)  # rotate about the PMT axis THROUGH the PMT
        d_rot = R @ (direction / np.linalg.norm(direction))
        f = frame(jnp.asarray(PMT_POS), jnp.asarray(PMT_N), 12.0, _hyp(v_rot, d_rot))
        np.testing.assert_allclose(np.asarray(f), np.asarray(f0), rtol=2e-4, atol=2e-4)


def test_not_invariant_under_rotation_about_other_axis():
    frame = SensorFrame()
    vertex = np.array([50.0, 80.0, -100.0])
    direction = np.array([0.3, -0.5, 0.8])
    f0 = frame(jnp.asarray(PMT_POS), jnp.asarray(PMT_N), 12.0, _hyp(vertex, direction))
    R = _rot([0.0, 0.0, 1.0], 1.0)  # generic z rotation, NOT the PMT axis
    v_rot = PMT_POS + R @ (vertex - PMT_POS)
    d_rot = R @ (direction / np.linalg.norm(direction))
    f = frame(jnp.asarray(PMT_POS), jnp.asarray(PMT_N), 12.0, _hyp(v_rot, d_rot))
    assert not np.allclose(np.asarray(f), np.asarray(f0), atol=1e-3)


def test_chirality_sign_flips_under_reflection():
    frame = SensorFrame(include_sign=True)
    vertex = np.array([50.0, 80.0, -100.0])
    direction = np.array([0.3, -0.5, 0.8])
    # reflect through a plane containing the PMT axis: build basis, flip one transverse comp
    e1 = np.cross(PMT_N, [0, 0, 1.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(PMT_N, e1)
    refl = lambda v: (v @ PMT_N) * PMT_N + (v @ e1) * e1 - (v @ e2) * e2

    f0 = np.asarray(frame(jnp.asarray(PMT_POS), jnp.asarray(PMT_N), 12.0,
                          _hyp(vertex, direction)))
    v_ref = PMT_POS + refl(vertex - PMT_POS)
    d_ref = refl(direction / np.linalg.norm(direction))
    f1 = np.asarray(frame(jnp.asarray(PMT_POS), jnp.asarray(PMT_N), 12.0, _hyp(v_ref, d_ref)))
    np.testing.assert_allclose(f1[:6], f0[:6], rtol=2e-4, atol=2e-4)  # scalars invariant
    assert f1[6] == -f0[6]  # chirality bit flips


def test_t_res_definition_and_n_eff_gradient():
    frame = SensorFrame(n_eff_init=1.38)
    vertex = np.zeros(3)
    d = np.linalg.norm(PMT_POS)
    tof = 1.38 * d / C_MM_PER_NS
    f = frame(jnp.asarray(PMT_POS), jnp.asarray(PMT_N), tof + 2.5, _hyp(vertex, [0, 0, 1]))
    np.testing.assert_allclose(float(f[4]) * 25.0, 2.5, atol=1e-3)  # t_res = 2.5 ns

    def loss(fr):
        return fr(jnp.asarray(PMT_POS), jnp.asarray(PMT_N), 10.0, _hyp(vertex, [0, 0, 1]))[4]

    import equinox as eqx
    g = eqx.filter_grad(loss)(frame)
    assert abs(float(g.log_n_eff)) > 1e-6  # n_eff is trainable through t_res


def test_e_ref_orthonormal():
    normals = np.array([[0, 0, 1.0], [0, 0, -1.0], [1, 0, 0.0], [0.6, -0.64, 0.48]])
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    e_ref = np.asarray(compute_e_ref(jnp.asarray(normals, jnp.float32)))
    np.testing.assert_allclose(np.linalg.norm(e_ref, axis=1), 1.0, atol=1e-5)
    np.testing.assert_allclose(np.sum(e_ref * normals, axis=1), 0.0, atol=1e-5)


def test_frame_hitnet_trains_on_toy_data():
    batch, pmt_pos = _toy_data(n_events=400, seed=13)
    rng = np.random.default_rng(0)
    normals = rng.normal(size=pmt_pos.shape)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    data = DeviceData.from_batch(batch, pmt_pos)._replace(
        pmt_dir=jnp.asarray(normals, jnp.float32))

    net = FrameHitNet(pmt_pos, normals, width=32, depth=2, key=jax.random.PRNGKey(0))
    res = fit_resident(net, data, frame_hit_batch, data.n_hits,
                       key=jax.random.PRNGKey(1), batch_size=512,
                       max_epochs=3, patience=3, verbose=False)
    assert np.isfinite(res.val_loss).all()
    # geometry tables must NOT train; n_eff and MLP must
    assert np.array_equal(np.asarray(res.model.pmt_pos), pmt_pos.astype(np.float32))
    assert not np.allclose(np.asarray(res.model.mlp.layers[0].weight),
                           np.asarray(net.mlp.layers[0].weight))


def test_frame_hitnet_obs_style_and_gradients():
    batch, pmt_pos = _toy_data(n_events=50, seed=7)
    normals = np.tile([0.0, 0.0, 1.0], (len(pmt_pos), 1))
    net = FrameHitNet(pmt_pos, normals, width=32, depth=2, key=jax.random.PRNGKey(0))
    assert net.obs_style == "id_t"
    hyp = jnp.asarray(batch.hyp[0])
    logit = net((jnp.asarray(3), jnp.asarray(12.0)), hyp)
    assert logit.shape == ()
    g = jax.grad(lambda p: net((jnp.asarray(3), jnp.asarray(12.0)), p))(hyp)
    assert np.isfinite(np.asarray(g)).all() and np.any(np.asarray(g) != 0)


def test_store_roundtrip_includes_geometry():
    import os
    TEST_FILE = "/tank/playground/eos_validation/sweep5000/data/water_1MeV.root"
    if not os.path.exists(TEST_FILE):
        pytest.skip("fixture unavailable")
    from hitman.data.ratds import pmt_geometry

    pos, dirs, types = pmt_geometry([TEST_FILE])
    assert dirs is not None and dirs.shape == pos.shape
    np.testing.assert_allclose(np.linalg.norm(dirs, axis=1), 1.0, atol=1e-3)
    assert types is not None and len(types) == len(pos)
