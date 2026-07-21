"""run20 spline-MLE model: closed-form normalization, integration, recovery, invariance.

Every check hits the non-negotiable core -- that the density normalizes analytically at
every (x, theta) with no Monte Carlo and no grid in the data dimension.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from hitman.splinemle import (DEFAULT_KNOTS, SplineMLE, build_event_spec, log_prob_u,
                              logZ_time, make_event_batch, splinemle_loss,
                              step_max_intermediate_gib)
from hitman.splinemle.model import log_expm1_over_x


def _toy_geometry(seed=1):
    rng = np.random.default_rng(seed)
    pmt_pos = jnp.asarray(rng.normal(size=(241, 3)) * 400.0, jnp.float32)
    nrm = rng.normal(size=(241, 3))
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    return pmt_pos, jnp.asarray(nrm, jnp.float32)


def _toy_model(width=48, key=0):
    pmt_pos, pmt_normal = _toy_geometry()
    return SplineMLE(pmt_pos, pmt_normal, key=jax.random.PRNGKey(key), width=width, depth=3)


# ---------------------------------------------------------------------------
# (a) closed-form Z_t == numerical integration, including b ~ 0
# ---------------------------------------------------------------------------
def test_closed_form_Z_matches_numerical():
    jax.config.update("jax_enable_x64", True)
    try:
        knots = jnp.asarray(DEFAULT_KNOTS)
        knots_np = np.asarray(DEFAULT_KNOTS, np.float64)
        ug = np.linspace(knots_np[0], knots_np[-1], 4_000_001)
        rng = np.random.default_rng(0)
        for trial in range(6):
            if trial == 0:                       # exactly constant -> every b == 0
                nv = np.full(len(DEFAULT_KNOTS), 0.7)
            elif trial == 1:                     # near-constant -> b ~ 1e-6 (series branch)
                nv = 0.5 + 1e-6 * rng.normal(size=len(DEFAULT_KNOTS))
            else:
                nv = rng.normal(size=len(DEFAULT_KNOTS)) * 2.0
            nv_j = jnp.asarray(nv)
            logZ = float(logZ_time(nv_j, knots))
            ell = np.interp(ug, knots_np, nv)
            Zref = np.trapezoid(np.exp(ell), ug)
            rel = abs(np.exp(logZ) - Zref) / Zref
            assert rel < 1e-6, f"trial {trial}: rel {rel:.2e}"
    finally:
        jax.config.update("jax_enable_x64", False)


def test_log_expm1_over_x_stable_and_smooth():
    # matches log(expm1(z)/z) across scales, and its gradient is finite at z = 0 (== 1/2).
    for z in (-40.0, -1.0, -1e-4, 1e-4, 1.0, 40.0):
        ref = np.log(np.expm1(z) / z)
        assert abs(float(log_expm1_over_x(jnp.asarray(z))) - ref) < 1e-5
    g = float(jax.grad(lambda z: log_expm1_over_x(z))(jnp.asarray(0.0)))
    assert np.isfinite(g) and abs(g - 0.5) < 1e-4


# ---------------------------------------------------------------------------
# (b) log_prob integrates to 1 in t; softmax sensor factor sums to 1 exactly
# ---------------------------------------------------------------------------
def test_time_density_integrates_to_one():
    jax.config.update("jax_enable_x64", True)
    try:
        m = _toy_model()
        theta = jnp.asarray([50.0, -80.0, 120.0, 1.1, 2.0, 0.0, 3.0], jnp.float64)
        for s in (0, 73, 240):
            d = float(jnp.linalg.norm(m.pmt_pos[s].astype(jnp.float64) - theta[:3]))
            tg = float(theta[5]) + float(m.n_eff) * d / 299.792458
            tt = np.linspace(tg + DEFAULT_KNOTS[0] + 1e-4,
                             tg + DEFAULT_KNOTS[-1] - 1e-4, 2_000_001)
            dens = np.asarray(m.density_time(jnp.asarray(tt), s, theta))
            integral = np.trapezoid(dens, tt)
            assert abs(integral - 1.0) < 1e-4, f"sensor {s}: int {integral}"
    finally:
        jax.config.update("jax_enable_x64", False)


def test_sensor_softmax_sums_to_one():
    m = _toy_model()
    theta = jnp.asarray([10.0, 20.0, -30.0, 0.8, 1.0, 5.0, 4.0], jnp.float32)
    ls = jax.vmap(lambda s: m.log_prob_sensor(s, theta))(jnp.arange(241))
    total = float(jnp.sum(jnp.exp(ls)))
    assert abs(total - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# (c) recover a known piecewise-exponential density by MLE (KL small)
# ---------------------------------------------------------------------------
def _sample_piecewise_exp(node_vals, knots, n, rng):
    """Inverse-CDF sampler for the log-spline density (numpy, float64)."""
    knots = np.asarray(knots, float)
    nv = np.asarray(node_vals, float)
    dk = np.diff(knots)
    a = nv[:-1]
    b = (nv[1:] - nv[:-1]) / dk
    small = np.abs(b) < 1e-8
    Ij = np.exp(a) * np.where(small, dk, np.expm1(b * dk) / np.where(small, 1.0, b))
    cdf = np.concatenate([[0.0], np.cumsum(Ij)])
    Z = cdf[-1]
    U = rng.uniform(0.0, Z, size=n)
    j = np.clip(np.searchsorted(cdf, U) - 1, 0, len(dk) - 1)
    m = U - cdf[j]
    bj, aj = b[j], a[j]
    sb = np.abs(bj) < 1e-8
    s = np.where(sb, m * np.exp(-aj),
                 np.log1p(m * bj * np.exp(-aj)) / np.where(sb, 1.0, bj))
    return knots[:-1][j] + s


def test_recover_known_density_by_mle():
    knots = jnp.asarray(DEFAULT_KNOTS)
    rng = np.random.default_rng(7)
    # a smooth-ish truth: a bump near 0 decaying into the tail
    ku = np.asarray(DEFAULT_KNOTS)
    truth = -0.15 * np.abs(ku) + 1.2 * np.exp(-((ku - 0.5) ** 2) / 4.0)
    truth = jnp.asarray(truth, jnp.float32)
    samples = jnp.asarray(_sample_piecewise_exp(truth, knots, 40_000, rng), jnp.float32)

    nv = jnp.zeros(len(DEFAULT_KNOTS), jnp.float32)
    opt = optax.adam(5e-2)
    state = opt.init(nv)

    @jax.jit
    def step(nv, state):
        def nll(nv):
            lp = jax.vmap(lambda u: log_prob_u(u, nv, knots))(samples)
            return -jnp.mean(lp)
        loss, g = jax.value_and_grad(nll)(nv)
        upd, state = opt.update(g, state)
        return optax.apply_updates(nv, upd), state, loss

    for _ in range(600):
        nv, state, loss = step(nv, state)

    # KL(truth || fit) on a fine grid
    ug = jnp.asarray(np.linspace(DEFAULT_KNOTS[0] + 1e-3, DEFAULT_KNOTS[-1] - 1e-3, 20000),
                     jnp.float32)
    lp_t = jax.vmap(lambda u: log_prob_u(u, truth, knots))(ug)
    lp_f = jax.vmap(lambda u: log_prob_u(u, nv, knots))(ug)
    p_t = jnp.exp(lp_t)
    du = float(ug[1] - ug[0])
    kl = float(jnp.sum(p_t * (lp_t - lp_f)) * du)
    assert kl < 2e-2, f"KL(truth||fit) = {kl:.4f}"


# ---------------------------------------------------------------------------
# (d) TOF invariance: shifting hit times and theta_t together leaves the loss unchanged
# ---------------------------------------------------------------------------
def test_tof_shift_invariance():
    m = _toy_model()
    rng = np.random.default_rng(3)
    B, P = 6, 24
    pmt_ids = jnp.asarray(rng.integers(0, 241, size=(B, P)), jnp.int32)
    theta = jnp.asarray(rng.normal(size=(B, 7)).astype(np.float32))
    theta = theta.at[:, 6].set(jnp.abs(theta[:, 6]) + 1.0)  # E > 0
    # place hit times near each hit's t_geo so u is in-support
    def tgeo(s, th):
        d = jnp.linalg.norm(m.pmt_pos[s] - th[:3])
        return th[5] + m.n_eff * d / 299.792458
    tg = jax.vmap(lambda ss, th: jax.vmap(lambda s: tgeo(s, th))(ss))(pmt_ids, theta)
    t = tg + jnp.asarray(rng.uniform(0, 3, size=(B, P)), jnp.float32)
    mask = jnp.ones((B, P), jnp.float32)

    loss0, _ = splinemle_loss(m, (pmt_ids, t, mask, theta))
    delta = 17.3
    theta_s = theta.at[:, 5].add(delta)
    loss1, _ = splinemle_loss(m, (pmt_ids, t + delta, mask, theta_s))
    assert abs(float(loss0) - float(loss1)) < 1e-3


# ---------------------------------------------------------------------------
# (e) gradient flows to the learnable n_eff
# ---------------------------------------------------------------------------
def test_gradient_flows_to_n_eff():
    m = _toy_model()
    rng = np.random.default_rng(5)
    B, P = 4, 20
    pmt_ids = jnp.asarray(rng.integers(0, 241, size=(B, P)), jnp.int32)
    theta = jnp.asarray([50.0, -80.0, 120.0, 1.1, 2.0, 0.0, 3.0], jnp.float32)
    th = jnp.tile(theta, (B, 1))

    def tgeo(s):
        d = jnp.linalg.norm(m.pmt_pos[s] - theta[:3])
        return theta[5] + m.n_eff * d / 299.792458
    t = jax.vmap(lambda row: jax.vmap(tgeo)(row))(pmt_ids) + 1.0
    mask = jnp.ones((B, P), jnp.float32)

    (loss, _), g = eqx.filter_value_and_grad(
        lambda mm: splinemle_loss(mm, (pmt_ids, t, mask, th)), has_aux=True)(m)
    assert np.isfinite(float(g.log_n_eff))
    assert abs(float(g.log_n_eff)) > 0.0


# ---------------------------------------------------------------------------
# training plumbing: batch build + static preflight run on a tiny synthetic store
# ---------------------------------------------------------------------------
class _FakeStore:
    def __init__(self, n_events=200, max_hits=30, seed=2):
        rng = np.random.default_rng(seed)
        counts = rng.integers(5, max_hits, size=n_events)
        self.hit_offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        self.n_events = n_events


class _FakeData:
    def __init__(self, store, seed=2):
        rng = np.random.default_rng(seed)
        nh = int(store.hit_offsets[-1])
        self.t = jnp.asarray(rng.normal(size=nh).astype(np.float32))
        self.pmt_id = jnp.asarray(rng.integers(0, 241, size=nh).astype(np.int32))
        self.hyp = jnp.asarray(rng.normal(size=(store.n_events, 7)).astype(np.float32))
        self.n_hits = nh


# ---------------------------------------------------------------------------
# v1: NB2 count model + monotone phi(E) yield head
# ---------------------------------------------------------------------------
def test_nb2_normalizes_over_N():
    m = _toy_model()
    for E, logL in ((jnp.asarray(2.5), jnp.asarray(np.log(60.0))),
                    (jnp.asarray(9.5), jnp.asarray(np.log(150.0)))):
        Ns = jnp.arange(0.0, 4000.0)
        total = float(jnp.sum(jnp.exp(jax.vmap(lambda N: m.log_count(N, logL, E))(Ns))))
        assert abs(total - 1.0) < 1e-4, f"E={float(E)}: sum_N NB2 = {total}"


def test_nb2_matches_scipy():
    sp = pytest.importorskip("scipy.stats")
    m = _toy_model()
    mu = 80.0
    logL = jnp.asarray(np.log(mu))
    for E in (2.5, 9.5):
        r = float(np.exp(float(m.log_dispersion(jnp.asarray(E)))))
        for N in (0, 37, 120):
            mine = float(m.log_count(jnp.asarray(float(N)), logL, jnp.asarray(E)))
            ref = float(sp.nbinom.logpmf(N, r, r / (r + mu)))
            assert abs(mine - ref) < 1e-3, f"E={E} N={N}: {mine} vs {ref}"


def test_nb2_fano_matches_dispersion():
    m = _toy_model()
    rng = np.random.default_rng(11)
    for E, mu in ((2.5, 50.0), (9.5, 130.0)):
        r = float(np.exp(float(m.log_dispersion(jnp.asarray(E)))))
        p = r / (r + mu)
        draws = rng.negative_binomial(r, p, size=400_000)
        fano_emp = draws.var() / draws.mean()
        fano_theory = 1.0 + mu / r        # NB2 Fano
        assert abs(fano_emp - fano_theory) / fano_theory < 0.03, (
            f"E={E}: emp {fano_emp:.3f} vs theory {fano_theory:.3f}")


def test_phi_monotone():
    # near-flat default and a curve-initialized phi are both non-decreasing in E.
    pmt_pos, pmt_normal = _toy_geometry()
    from hitman.splinemle import DEFAULT_PHI_KNOTS
    phi_init = np.log(np.maximum(np.asarray(DEFAULT_PHI_KNOTS) * 20.0 + 1.0, 0.1))
    for kw in ({}, {"phi_init_values": phi_init, "phi_anchor": (5.0, np.log(97.7))}):
        m = SplineMLE(pmt_pos, pmt_normal, key=jax.random.PRNGKey(0), width=32, **kw)
        Eg = jnp.linspace(0.0, 10.0, 400)
        phig = jax.vmap(m.phi)(Eg)
        assert bool(jnp.all(jnp.diff(phig) >= -1e-6)), "phi not monotone"


def test_phi_cancels_in_softmax():
    # phi(E) is an E-only additive intensity scale -> softmax(eta + phi) = softmax(eta):
    # perturbing any phi parameter must leave the sensor factor exactly unchanged.
    m = _toy_model()
    theta = jnp.asarray([10.0, 20.0, -30.0, 0.8, 1.0, 5.0, 4.0], jnp.float32)
    ls0 = jax.vmap(lambda s: m.log_prob_sensor(s, theta))(jnp.arange(241))
    m2 = eqx.tree_at(lambda mm: mm.phi_e0, m, m.phi_e0 + 3.7)
    m3 = eqx.tree_at(lambda mm: mm.phi_raw, m, m.phi_raw + 1.0)
    ls2 = jax.vmap(lambda s: m2.log_prob_sensor(s, theta))(jnp.arange(241))
    ls3 = jax.vmap(lambda s: m3.log_prob_sensor(s, theta))(jnp.arange(241))
    assert float(jnp.max(jnp.abs(ls0 - ls2))) == 0.0
    assert float(jnp.max(jnp.abs(ls0 - ls3))) == 0.0


def test_gradient_flows_to_phi_and_disp():
    m = _toy_model()
    rng = np.random.default_rng(9)
    B, P = 6, 20
    pmt_ids = jnp.asarray(rng.integers(0, 241, size=(B, P)), jnp.int32)
    theta = jnp.asarray(rng.uniform(0.5, 9.5, size=(B, 7)).astype(np.float32))
    tg = jax.vmap(lambda ss, th: jax.vmap(
        lambda s: th[5] + m.n_eff * jnp.linalg.norm(m.pmt_pos[s] - th[:3]) / 299.792458
    )(ss))(pmt_ids, theta)
    t = tg + jnp.asarray(rng.uniform(0, 2, size=(B, P)), jnp.float32)
    mask = jnp.ones((B, P), jnp.float32)
    (loss, _), g = eqx.filter_value_and_grad(
        lambda mm: splinemle_loss(mm, (pmt_ids, t, mask, theta)), has_aux=True)(m)
    assert np.isfinite(float(jnp.sum(g.phi_raw))) and float(jnp.sum(jnp.abs(g.phi_raw))) > 0
    assert np.isfinite(float(jnp.sum(g.disp))) and float(jnp.sum(jnp.abs(g.disp))) > 0


# ---------------------------------------------------------------------------
# run21: v2 conditioner feature set (distance basis) + v1 back-compat
# ---------------------------------------------------------------------------
def test_v2_feature_shape_and_finiteness():
    # v2 adds 4 distance-basis features (5 -> 9); first five must equal v1 exactly, and
    # every feature is finite even at an extreme near-zero distance (epsilon-floored d).
    pmt_pos, pmt_normal = _toy_geometry()
    m1 = SplineMLE(pmt_pos, pmt_normal, key=jax.random.PRNGKey(0), width=32, feature_set="v1")
    m2 = SplineMLE(pmt_pos, pmt_normal, key=jax.random.PRNGKey(0), width=32, feature_set="v2")
    theta = jnp.asarray([50.0, -80.0, 120.0, 1.1, 2.0, 0.0, 3.0], jnp.float32)
    for pos, nrm in ((pmt_pos[0], pmt_normal[0]),
                     (jnp.asarray([50.0, -80.0, 120.0], jnp.float32),  # d ~ 0 (at vertex)
                      pmt_normal[3])):
        f1, d1 = m1._sensor_features(pos, nrm, theta)
        f2, d2 = m2._sensor_features(pos, nrm, theta)
        assert f1.shape == (5,) and f2.shape == (9,)
        assert float(jnp.max(jnp.abs(f2[:5] - f1))) == 0.0     # first five identical
        assert bool(jnp.all(jnp.isfinite(f2))), f"non-finite v2 feature at d={float(d2)}"
    # the conditioner MLP first layer consumes the wider vector under v2.
    assert m1.mlp.layers[0].weight.shape[1] == 5
    assert m2.mlp.layers[0].weight.shape[1] == 9


def test_v1_template_shapes_unchanged_and_roundtrip(tmp_path):
    # A v1 model (default feature_set) must keep IDENTICAL leaf shapes so existing receipts'
    # SplineMLE(width=192, depth=3, ...) template still deserializes v1 checkpoints. Verify
    # by a serialise/deserialise round-trip through a fresh default (v1) template.
    pmt_pos, pmt_normal = _toy_geometry()
    m = SplineMLE(pmt_pos, pmt_normal, key=jax.random.PRNGKey(3), width=192, depth=3)
    assert m.feature_set == "v1"                               # default unchanged
    shapes = [np.asarray(l).shape for l in
              jax.tree_util.tree_leaves(eqx.filter(m, eqx.is_array))]
    path = str(tmp_path / "v1.eqx")
    eqx.tree_serialise_leaves(path, m)
    tmpl = SplineMLE(pmt_pos, pmt_normal, key=jax.random.PRNGKey(999), width=192, depth=3)
    m_back = eqx.tree_deserialise_leaves(path, tmpl)
    shapes_back = [np.asarray(l).shape for l in
                   jax.tree_util.tree_leaves(eqx.filter(m_back, eqx.is_array))]
    assert shapes == shapes_back
    assert m.mlp.layers[0].weight.shape[1] == 5                # v1 first layer takes 5 features


def test_v2_forward_and_normalization():
    # v2 model still normalizes exactly: softmax sensor factor sums to 1.
    pmt_pos, pmt_normal = _toy_geometry()
    m = SplineMLE(pmt_pos, pmt_normal, key=jax.random.PRNGKey(1), width=48, feature_set="v2")
    theta = jnp.asarray([10.0, 20.0, -30.0, 0.8, 1.0, 5.0, 4.0], jnp.float32)
    ls = jax.vmap(lambda s: m.log_prob_sensor(s, theta))(jnp.arange(241))
    assert abs(float(jnp.sum(jnp.exp(ls))) - 1.0) < 1e-5


def test_batch_and_preflight():
    store = _FakeStore()
    data = _FakeData(store)
    spec = build_event_spec(store, n_pad=32)
    batch = make_event_batch(data, spec, jax.random.PRNGKey(0), n_events=8, n_pad=32)
    pmt_ids, t, mask, theta = batch
    assert pmt_ids.shape == (8, 32) and theta.shape == (8, 7)
    assert mask.max() == 1.0

    m = _toy_model(width=32)
    gib = step_max_intermediate_gib(m, data, spec, n_events=8, n_pad=32)
    assert np.isfinite(gib) and gib < 4.0


# ---------------------------------------------------------------------------
# run22: split eta/time conditioner heads (dedicated eta trunk)
# ---------------------------------------------------------------------------
def test_split_head_shapes_and_trunks():
    # split: ``mlp`` emits n_nodes time nodes (NOT n_nodes+1), ``mlp_eta`` emits 1 eta, with
    # its own width/depth. joint keeps the fused (n_nodes+1) trunk and mlp_eta is None.
    pmt_pos, pmt_normal = _toy_geometry()
    n_nodes = len(DEFAULT_KNOTS)
    mj = SplineMLE(pmt_pos, pmt_normal, key=jax.random.PRNGKey(0), width=32, depth=3)
    assert mj.head_mode == "joint" and mj.mlp_eta is None
    assert mj.mlp.layers[-1].weight.shape[0] == n_nodes + 1     # fused: eta + nodes
    ms = SplineMLE(pmt_pos, pmt_normal, key=jax.random.PRNGKey(0), width=32, depth=3,
                   head_mode="split", eta_width=24, eta_depth=2)
    assert ms.head_mode == "split" and ms.mlp_eta is not None
    assert ms.mlp.layers[-1].weight.shape[0] == n_nodes         # time trunk: nodes only
    assert ms.mlp_eta.layers[-1].weight.shape[0] == 1           # eta trunk: single scalar
    assert len(ms.mlp_eta.layers) == 2 + 1                      # eta_depth=2 -> 3 Linear
    assert ms.mlp_eta.layers[0].weight.shape == (24, 5)         # eta_width=24, v1 features
    # event_tables shapes hold in split mode: eta (241,), nodes (241, n_nodes)
    theta = jnp.asarray([10.0, 20.0, -30.0, 0.8, 1.0, 5.0, 4.0], jnp.float32)
    eta, nodes, t_geo, logZt = ms.event_tables(theta)
    assert eta.shape == (241,) and nodes.shape == (241, n_nodes)
    assert t_geo.shape == (241,) and logZt.shape == (241,)


def test_split_head_normalizes():
    # a split model must still normalize exactly: softmax sensor factor sums to 1 and the
    # time density integrates to 1 (closed-form partition unaffected by the head split).
    jax.config.update("jax_enable_x64", True)
    try:
        pmt_pos, pmt_normal = _toy_geometry()
        m = SplineMLE(pmt_pos, pmt_normal, key=jax.random.PRNGKey(1), width=32, depth=3,
                      head_mode="split", eta_width=24, eta_depth=2, feature_set="v2")
        theta = jnp.asarray([50.0, -80.0, 120.0, 1.1, 2.0, 0.0, 3.0], jnp.float64)
        ls = jax.vmap(lambda s: m.log_prob_sensor(s, theta))(jnp.arange(241))
        assert abs(float(jnp.sum(jnp.exp(ls))) - 1.0) < 1e-5
        s = 73
        d = float(jnp.linalg.norm(m.pmt_pos[s].astype(jnp.float64) - theta[:3]))
        tg = float(theta[5]) + float(m.n_eff) * d / 299.792458
        tt = np.linspace(tg + DEFAULT_KNOTS[0] + 1e-4, tg + DEFAULT_KNOTS[-1] - 1e-4, 2_000_001)
        dens = np.asarray(m.density_time(jnp.asarray(tt), s, theta))
        assert abs(float(np.trapezoid(dens, tt)) - 1.0) < 1e-4
    finally:
        jax.config.update("jax_enable_x64", False)


def test_joint_roundtrip_bit_identical_with_split_field_present(tmp_path):
    # The split-head field must NOT perturb joint-mode serialization: a joint model's array
    # leaves and their BYTES must round-trip through a fresh joint template unchanged (so the
    # real v1/v2 checkpoints still deserialize). mlp_eta=None contributes no leaves.
    pmt_pos, pmt_normal = _toy_geometry()
    m = SplineMLE(pmt_pos, pmt_normal, key=jax.random.PRNGKey(3), width=64, depth=3,
                  feature_set="v2")
    assert m.head_mode == "joint" and m.mlp_eta is None
    leaves = jax.tree_util.tree_leaves(eqx.filter(m, eqx.is_array))
    path = str(tmp_path / "joint.eqx")
    eqx.tree_serialise_leaves(path, m)
    tmpl = SplineMLE(pmt_pos, pmt_normal, key=jax.random.PRNGKey(999), width=64, depth=3,
                     feature_set="v2")
    m_back = eqx.tree_deserialise_leaves(path, tmpl)
    leaves_back = jax.tree_util.tree_leaves(eqx.filter(m_back, eqx.is_array))
    assert len(leaves) == len(leaves_back)
    for a, b in zip(leaves, leaves_back):
        assert np.array_equal(np.asarray(a), np.asarray(b))      # bit-identical


def test_split_head_serialise_roundtrip(tmp_path):
    # a split model round-trips through a split template (both trunks recovered exactly).
    pmt_pos, pmt_normal = _toy_geometry()
    m = SplineMLE(pmt_pos, pmt_normal, key=jax.random.PRNGKey(5), width=48, depth=4,
                  head_mode="split", eta_width=32, eta_depth=3, feature_set="v2")
    path = str(tmp_path / "split.eqx")
    eqx.tree_serialise_leaves(path, m)
    tmpl = SplineMLE(pmt_pos, pmt_normal, key=jax.random.PRNGKey(111), width=48, depth=4,
                     head_mode="split", eta_width=32, eta_depth=3, feature_set="v2")
    m_back = eqx.tree_deserialise_leaves(path, tmpl)
    assert m_back.mlp_eta is not None
    theta = jnp.asarray([10.0, 20.0, -30.0, 0.8, 1.0, 5.0, 4.0], jnp.float32)
    eta0, nodes0, _, _ = m.event_tables(theta)
    eta1, nodes1, _, _ = m_back.event_tables(theta)
    assert float(jnp.max(jnp.abs(eta0 - eta1))) == 0.0
    assert float(jnp.max(jnp.abs(nodes0 - nodes1))) == 0.0


def test_split_head_gradient_flows_to_both_trunks():
    # both trunks receive gradient under the event loss (eta head is no longer starved by the
    # time nodes -- it has its own parameters).
    pmt_pos, pmt_normal = _toy_geometry()
    m = SplineMLE(pmt_pos, pmt_normal, key=jax.random.PRNGKey(2), width=32, depth=3,
                  head_mode="split", eta_width=24, eta_depth=2)
    rng = np.random.default_rng(9)
    B, P = 6, 20
    pmt_ids = jnp.asarray(rng.integers(0, 241, size=(B, P)), jnp.int32)
    theta = jnp.asarray(rng.uniform(0.5, 9.5, size=(B, 7)).astype(np.float32))
    tg = jax.vmap(lambda ss, th: jax.vmap(
        lambda s: th[5] + m.n_eff * jnp.linalg.norm(m.pmt_pos[s] - th[:3]) / 299.792458
    )(ss))(pmt_ids, theta)
    t = tg + jnp.asarray(rng.uniform(0, 2, size=(B, P)), jnp.float32)
    mask = jnp.ones((B, P), jnp.float32)
    (loss, _), g = eqx.filter_value_and_grad(
        lambda mm: splinemle_loss(mm, (pmt_ids, t, mask, theta)), has_aux=True)(m)
    g_eta = sum(float(jnp.sum(jnp.abs(l.weight))) for l in g.mlp_eta.layers)
    g_time = sum(float(jnp.sum(jnp.abs(l.weight))) for l in g.mlp.layers)
    assert np.isfinite(g_eta) and g_eta > 0.0
    assert np.isfinite(g_time) and g_time > 0.0
