"""Proposal 4: injectable projection instrument v(theta) in the moment vectors.

The headline requirement is a LOCK: with the default WC ray instrument, ``lean_vector`` /
``full_vector`` must reproduce the pre-refactor hardcoded ``(zen, az)`` ray formulas
bit-for-bit (the WC receipts and their frozen W depend on the exact numbers). A downstream
project injects a different instrument (e.g. a shower-axis with its own partner params).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hitman.wc.train import moments
from hitman.wc.train.moments import (FULL_NAMES, LEAN_NAMES, ProjectionInstrument,
                                  RAY_INSTRUMENT, full_vector, lean_vector, moment_names)

VECH_IDX = moments.VECH_IDX


# ---- pre-refactor reference (verbatim WC formulas) --------------------------
def _ref_direction(theta):
    zen, az = theta[3], theta[4]
    s = jnp.sin(zen)
    return jnp.stack([s * jnp.cos(az), s * jnp.sin(az), jnp.cos(zen)])


def _ref_ray_entries(B, d):
    return jnp.stack([d @ B[:3, :3] @ d, B[6, :3] @ d, B[5, :3] @ d])


def _ref_lean(g, H, theta):
    B = H + jnp.outer(g, g)
    d = _ref_direction(theta)
    return jnp.concatenate([
        g, (g[:3] @ d)[None], jnp.diag(B), B[5, 6][None], _ref_ray_entries(B, d)])


def _ref_full(g, H, theta):
    B = H + jnp.outer(g, g)
    d = _ref_direction(theta)
    vech = B[VECH_IDX[:, 0], VECH_IDX[:, 1]]
    return jnp.concatenate([g, (g[:3] @ d)[None], vech, _ref_ray_entries(B, d)])


def _random_gHtheta(rng, dtype):
    g = jnp.asarray(rng.normal(size=7), dtype)
    A = rng.normal(size=(7, 7))
    H = jnp.asarray(A + A.T, dtype)          # symmetric Hessian
    theta = jnp.asarray(rng.normal(size=7), dtype)
    return g, H, theta


# ---- (a) the LOCK: default instrument == pre-refactor, bit-for-bit ----------
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_default_lean_full_bit_identical_to_preRefactor(dtype):
    jax.config.update("jax_enable_x64", True)
    try:
        rng = np.random.default_rng(0)
        for _ in range(50):
            g, H, theta = _random_gHtheta(rng, dtype)
            assert np.array_equal(np.asarray(lean_vector(g, H, theta)),
                                  np.asarray(_ref_lean(g, H, theta)))
            assert np.array_equal(np.asarray(full_vector(g, H, theta)),
                                  np.asarray(_ref_full(g, H, theta)))
    finally:
        jax.config.update("jax_enable_x64", False)


def test_default_names_unchanged():
    assert LEAN_NAMES[-4:] == ["B_tE", "B_rayray", "B_Eray", "B_tray"]
    assert FULL_NAMES[-3:] == ["B_rayray", "B_Eray", "B_tray"]
    assert LEAN_NAMES[:8] == ["g_x", "g_y", "g_z", "g_zen", "g_az", "g_t", "g_E", "g_ray"]
    assert (moments.N_LEAN, moments.N_FULL) == (19, 39)


# ---- (b) a genuinely different instrument is honored ------------------------
def test_injected_instrument_changes_projection():
    # project onto a FIXED axis over dims (0,1,2), partners (t=5, E=6) but a constant axis
    axis = jnp.asarray([0.0, 0.0, 1.0])
    inst = ProjectionInstrument(project=lambda th: axis, spatial=(0, 1, 2),
                                partners=(5, 6), partner_names=("t", "E"))
    rng = np.random.default_rng(1)
    g, H, theta = _random_gHtheta(rng, jnp.float32)
    v = np.asarray(lean_vector(g, H, theta, inst))
    B = np.asarray(H) + np.outer(np.asarray(g), np.asarray(g))
    # g_ray component (index 7) is now g[:3] . axis = g[2]
    assert abs(v[7] - float(g[2])) < 1e-5
    # B_rayray (after g(7)+g_ray(1)+diag(7)+B_tE(1) => index 16) = axis^T B_ss axis = B[2,2]
    assert abs(v[16] - B[2, 2]) < 1e-4


def test_injected_partners_and_names():
    # a shower-frame instrument with different partner params/names
    inst = ProjectionInstrument(project=lambda th: jnp.asarray([1.0, 0.0, 0.0]),
                                spatial=(0, 1, 2), partners=(3, 4),
                                partner_names=("dt", "Xmax"))
    lean, full = moment_names(["a", "b", "c", "dt", "Xmax", "e", "f"], inst)
    assert lean[-4:] == ["B_dtXmax", "B_rayray", "B_Xmaxray", "B_dtray"]
    assert full[-3:] == ["B_rayray", "B_Xmaxray", "B_dtray"]
    rng = np.random.default_rng(2)
    g, H, theta = _random_gHtheta(rng, jnp.float32)
    v = np.asarray(lean_vector(g, H, theta, inst))
    B = np.asarray(H) + np.outer(np.asarray(g), np.asarray(g))
    # B_dtXmax is B[partners] = B[3,4], at index 7(g)+1+7(diag) = 15
    assert abs(v[15] - B[3, 4]) < 1e-4


def test_default_instrument_is_ray():
    assert RAY_INSTRUMENT.spatial == (0, 1, 2)
    assert RAY_INSTRUMENT.partners == (5, 6)
    assert RAY_INSTRUMENT.partner_names == ("t", "E")


def test_wrong_hyp_dim_raises_loudly():
    # VECH_IDX is frozen to the WC dim at import; a smaller theta must FAIL, not silently
    # clamp the out-of-bounds gather (jnp gathers clamp — wrong numbers, no error).
    rng = np.random.default_rng(3)
    g6 = jnp.asarray(rng.normal(size=6), jnp.float32)
    A = rng.normal(size=(6, 6))
    H6 = jnp.asarray(A + A.T, jnp.float32)
    theta6 = jnp.asarray(rng.normal(size=6), jnp.float32)
    with pytest.raises(ValueError, match="hypothesis dim 6"):
        lean_vector(g6, H6, theta6)
    with pytest.raises(ValueError, match="hypothesis dim 6"):
        full_vector(g6, H6, theta6)
