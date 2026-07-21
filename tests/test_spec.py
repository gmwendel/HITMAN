"""Proposal 1: the detector-agnostic Obs/Hyp spec protocol.

Two jobs. (1) LOCK the water-Cherenkov instantiation to the hardcoded WC constants it
replaces (``nn.features`` column order, ``npe.flow`` box/cos/circular, ``train.moments``
names, the receipts truth-length) so the refactor is bit-for-bit backward compatible.
(2) Prove the protocol is genuinely detector-agnostic by exercising a deliberately
non-WC instantiation (the muon-station spec: continuous angular marks, no sensor grid,
profile-summary hypothesis).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hitman.nn import features as ft
from hitman.npe.flow import ConditionalFlow, COS_DIMS, CIRCULAR_DIMS, DEFAULT_BOX
from hitman.receipts import schema
from hitman.spec import (MUON_HYP_SPEC, MUON_OBS_SPEC, WC_HYP_SPEC, WC_OBS_SPEC,
                         DimSpec, HypSpec, ObsSpec)
from hitman.train import moments


# ---------------------------------------------------------------------------
# (a) the WC spec IS the WC layout (single source of truth, no drift)
# ---------------------------------------------------------------------------
def test_wc_hypspec_matches_features_columns():
    assert WC_HYP_SPEC.dim == 7
    assert WC_HYP_SPEC.names == ("x", "y", "z", "zen", "az", "t", "E")
    assert WC_HYP_SPEC.index("zen") == ft.ZENITH
    assert WC_HYP_SPEC.index("az") == ft.AZIMUTH
    assert WC_HYP_SPEC.index("t") == ft.TIME
    assert WC_HYP_SPEC.index("E") == ft.ENERGY
    assert (WC_HYP_SPEC.zenith, WC_HYP_SPEC.azimuth) == (ft.ZENITH, ft.AZIMUTH)


def test_wc_hypspec_matches_flow_domain():
    assert WC_HYP_SPEC.box == DEFAULT_BOX
    assert WC_HYP_SPEC.cos_dims == COS_DIMS
    assert WC_HYP_SPEC.circular_dims == CIRCULAR_DIMS
    assert WC_HYP_SPEC.period == 2.0 * np.pi


def test_wc_hypspec_matches_moment_names():
    assert tuple(moments._P) == WC_HYP_SPEC.names
    assert moments.VECH_IDX.shape[0] == WC_HYP_SPEC.dim * (WC_HYP_SPEC.dim + 1) // 2


def test_wc_direction_matches_features_direction():
    rng = np.random.default_rng(0)
    for _ in range(20):
        theta = jnp.asarray(rng.normal(size=7), jnp.float32)
        a = np.asarray(WC_HYP_SPEC.direction(theta))
        b = np.asarray(ft.direction(theta))
        assert np.allclose(a, b, atol=1e-6)


def test_wc_obsspec_layout():
    assert WC_OBS_SPEC.n_marks == 4
    assert WC_OBS_SPEC.mark_names == ("sensor_x", "sensor_y", "sensor_z", "time")
    assert WC_OBS_SPEC.has_sensor_index is True
    assert WC_OBS_SPEC.count_names == ("total_charge", "n_hits")


# ---------------------------------------------------------------------------
# (b) flow built from the spec reproduces the default WC flow exactly
# ---------------------------------------------------------------------------
def test_flow_from_spec_reproduces_default():
    key = jax.random.PRNGKey(3)
    ctx = 5
    ref = ConditionalFlow(context_dim=ctx, key=key, n_layers=4, n_bins=6, hidden=16)
    got = ConditionalFlow.from_spec(WC_HYP_SPEC, context_dim=ctx, key=key,
                                    n_layers=4, n_bins=6, hidden=16)
    assert got.n_dim == ref.n_dim
    assert got.domain.kind == ref.domain.kind
    assert got.domain.lo == ref.domain.lo and got.domain.hi == ref.domain.hi
    assert got.circular == ref.circular
    # identical params (same key + same architecture) => identical log_prob
    c = jax.random.normal(jax.random.PRNGKey(4), (ctx,))
    theta = jnp.asarray([100.0, -50.0, 200.0, 1.2, 3.0, 5.0, 4.0])
    assert float(ref.log_prob(theta, c)) == float(got.log_prob(theta, c))


def test_flow_from_spec_rejects_domain_override():
    with pytest.raises(TypeError):
        ConditionalFlow.from_spec(WC_HYP_SPEC, context_dim=3, key=jax.random.PRNGKey(0),
                                  box=DEFAULT_BOX)


# ---------------------------------------------------------------------------
# (c) a genuinely different detector plugs in (muon stations)
# ---------------------------------------------------------------------------
def test_muon_spec_is_non_wc_shaped():
    assert MUON_HYP_SPEC.dim == 6
    assert MUON_HYP_SPEC.names == ("cos_zen", "Xmu_max", "width", "X1", "r", "psi")
    # psi is the signed shower-plane angle -> circular; the rest are boxed
    assert MUON_HYP_SPEC.circular_dims == (MUON_HYP_SPEC.index("psi"),)
    assert MUON_HYP_SPEC.cos_dims == ()
    assert set(MUON_HYP_SPEC.box) == {0, 1, 2, 3, 4}
    # continuous angular marks, NO discrete sensor grid
    assert MUON_OBS_SPEC.mark_names == ("dt_plane", "alpha_r", "alpha_t")
    assert MUON_OBS_SPEC.has_sensor_index is False
    assert MUON_OBS_SPEC.n_marks == 3


def test_flow_from_muon_spec_is_exact_density():
    # a 6-D flow over the muon hypothesis: sample -> log_prob round-trips, incl. the
    # circular psi coordinate. Proves the spec drives a working, exactly-normalized flow.
    flow = ConditionalFlow.from_spec(MUON_HYP_SPEC, context_dim=3,
                                     key=jax.random.PRNGKey(1),
                                     n_layers=4, n_bins=6, hidden=16)
    c = jax.random.normal(jax.random.PRNGKey(2), (3,))
    z = flow.base_sample(jax.random.PRNGKey(5))
    theta = flow.inverse_from_base(z, c)
    zz, _ = flow.forward_to_base(theta, c)
    assert np.allclose(np.asarray(z), np.asarray(zz), atol=1e-4)
    assert np.isfinite(float(flow.log_prob(theta, c)))


def test_muon_hypspec_has_no_direction():
    with pytest.raises(ValueError):
        MUON_HYP_SPEC.direction(jnp.zeros(6))


# ---------------------------------------------------------------------------
# (d) receipts validation is spec-driven, default WC-compatible
# ---------------------------------------------------------------------------
def _receipt(truth, hyp_dim=None):
    meta = {"model_dir": "m"}
    if hyp_dim is not None:
        meta["hyp_dim"] = hyp_dim
    tp = schema.testpoint_receipt(truth=truth, self_norm=1.0, ess=1.0)
    return schema.assemble(meta, {"pt": tp})


def test_schema_default_is_wc7():
    schema.validate_schema(_receipt([0.0] * 7))
    with pytest.raises(ValueError):
        schema.validate_schema(_receipt([0.0] * 6))


def test_schema_honors_meta_hyp_dim():
    schema.validate_schema(_receipt([0.0] * MUON_HYP_SPEC.dim, hyp_dim=MUON_HYP_SPEC.dim))
    with pytest.raises(ValueError):
        schema.validate_schema(_receipt([0.0] * 7, hyp_dim=MUON_HYP_SPEC.dim))


def test_schema_explicit_arg_overrides_meta():
    schema.validate_schema(_receipt([0.0] * 3), hyp_dim=3)


# ---------------------------------------------------------------------------
# (e) DimSpec/HypSpec ergonomics
# ---------------------------------------------------------------------------
def test_dimspec_kind_dispatch():
    assert DimSpec("a", -1.0, 1.0).kind == "box"
    assert DimSpec("z", cos=True).kind == "cos"
    assert DimSpec("p", circular=True).kind == "circ"


def test_hypspec_index_raises_on_unknown():
    with pytest.raises(KeyError):
        WC_HYP_SPEC.index("nope")


def test_hypspec_positive_dims():
    assert ft.ENERGY in WC_HYP_SPEC.positive_dims
    assert set(MUON_HYP_SPEC.positive_dims) == {1, 2, 3, 4}


def test_dimspec_contradictions_raise():
    with pytest.raises(ValueError, match="mutually exclusive"):
        DimSpec("bad", circular=True, cos=True).kind
    with pytest.raises(ValueError, match="must not set"):
        DimSpec("bad", 0.0, 1.0, circular=True).kind
    with pytest.raises(ValueError, match="needs both lo and hi"):
        DimSpec("bad").kind
    with pytest.raises(ValueError, match="needs both lo and hi"):
        DimSpec("bad", lo=0.0).kind
