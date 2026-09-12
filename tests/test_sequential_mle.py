"""Tests for the sequential single-event MLE driver (hitman.wc.inference.seq).

Mirrors the compiled-solver tests: agreement is asserted in NLL space (the meaningful
invariant; raw params are degenerate for an untrained toy net). Bounds, determinism,
the exact-LM polish's monotonicity, and the Fisher output are asserted directly. The
sequential solver is a Python loop, so it is called per event (no vmap).
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hitman.wc.inference.batched import (MLEConfig, PaddedEvents, batched_multistart_mle,
                                      make_padded_nll)
from hitman.wc.inference.compiled import CompiledMLEConfig, make_compiled_mle
from hitman.wc.inference.mle import CHART_SCALE, chart_from_theta, theta_from_chart
from hitman.wc.inference.seq import _build_primitives, make_sequential_mle, SequentialMLEResult
from hitman.nn import ChargeNet, HitNet


def _toy_nets(key):
    k1, k2 = jax.random.split(key)
    return (HitNet(width=16, depth=2, key=k1), ChargeNet(width=16, depth=2, key=k2))


def _toy_events(key, n_events=8, n_pad=32, radius=600.0):
    ks = jax.random.split(key, 6)
    nhit = jax.random.randint(ks[0], (n_events,), 8, n_pad)
    slot = jnp.arange(n_pad)[None, :]
    mask = (slot < nhit[:, None]).astype(jnp.float32)
    dirs = jax.random.normal(ks[1], (n_events, n_pad, 3))
    dirs = dirs / (jnp.linalg.norm(dirs, axis=-1, keepdims=True) + 1e-9)
    pos = dirs * radius
    t = 4.0 + jax.random.uniform(ks[2], (n_events, n_pad)) * 6.0
    hits = jnp.concatenate([pos, t[..., None]], axis=-1) * mask[..., None]
    pmt_id = (jax.random.randint(ks[3], (n_events, n_pad), 0, 100) * mask).astype(jnp.int32)
    charge = jnp.stack([nhit.astype(jnp.float32) * 1.3, nhit.astype(jnp.float32)], axis=1)
    return PaddedEvents(hits=hits, pmt_id=pmt_id, t=hits[:, :, 3], mask=mask,
                        charge=charge, event_indices=np.arange(n_events),
                        dropped_fraction=0.0)


def _nll_batch(hitnet, chargenet, theta, padded):
    nll = jax.jit(jax.vmap(make_padded_nll(hitnet, chargenet)))
    ev = (padded.hits, padded.pmt_id, padded.t, padded.mask, padded.charge)
    return np.asarray(nll(jnp.asarray(theta, jnp.float32), ev))


def test_agrees_with_multistart_on_toy():
    """Sequential solver reaches NLL competitive with the 256-seed multistart teacher."""
    key = jax.random.PRNGKey(0)
    hitnet, chargenet = _toy_nets(key)
    padded = _toy_events(jax.random.PRNGKey(1), n_events=8, n_pad=32)

    gold = batched_multistart_mle(
        hitnet, chargenet, padded, key=jax.random.PRNGKey(2),
        cfg=MLEConfig(n_seeds=64, top_k=8, descent_steps=120), chunk=8)

    solve = make_sequential_mle(hitnet, chargenet, CompiledMLEConfig(), n_pad=32)
    theta_s = np.stack([solve(padded.hits[i], padded.pmt_id[i], padded.t[i],
                              padded.mask[i], padded.charge[i]) for i in range(8)])

    dnll = _nll_batch(hitnet, chargenet, theta_s, padded) - \
        _nll_batch(hitnet, chargenet, gold.theta, padded)
    assert np.median(dnll) < 0.5, f"median dNLL {np.median(dnll):.3f} too large"
    assert np.mean(dnll < 0.5) >= 0.7, f"only {np.mean(dnll < 0.5):.2f} within 0.5"
    assert np.all(np.isfinite(theta_s))


def test_agrees_with_compiled_solver():
    """Sequential and compiled solvers land at the same NLL basin (same objective)."""
    hitnet, chargenet = _toy_nets(jax.random.PRNGKey(10))
    padded = _toy_events(jax.random.PRNGKey(11), n_events=8, n_pad=32)
    cfg = CompiledMLEConfig(max_iter=25)

    fn = make_compiled_mle(hitnet, chargenet, cfg, n_pad=32)
    theta_c = np.asarray(jax.vmap(fn)(padded.hits, padded.pmt_id, padded.t,
                                      padded.mask, padded.charge))
    solve = make_sequential_mle(hitnet, chargenet, cfg, n_pad=32)
    theta_s = np.stack([solve(padded.hits[i], padded.pmt_id[i], padded.t[i],
                              padded.mask[i], padded.charge[i]) for i in range(8)])

    dnll = _nll_batch(hitnet, chargenet, theta_s, padded) - \
        _nll_batch(hitnet, chargenet, theta_c, padded)
    # sequential is at least as good as the compiled solver in NLL, up to a small margin
    assert np.median(dnll) < 0.1
    assert np.percentile(dnll, 90) < 0.5


def test_bounds_respected():
    """Every returned hypothesis lies inside the configured box."""
    hitnet, chargenet = _toy_nets(jax.random.PRNGKey(3))
    padded = _toy_events(jax.random.PRNGKey(4), n_events=8, n_pad=32)
    cfg = CompiledMLEConfig(radius=800.0, half_height=800.0, t_range=(-10.0, 10.0),
                            e_range=(0.5, 8.0))
    solve = make_sequential_mle(hitnet, chargenet, cfg, n_pad=32)
    th = np.stack([solve(padded.hits[i], padded.pmt_id[i], padded.t[i],
                         padded.mask[i], padded.charge[i]) for i in range(8)])
    assert np.all(np.abs(th[:, :3]) <= 800.0 + 1e-3)
    assert np.all((th[:, 5] >= -10.0 - 1e-3) & (th[:, 5] <= 10.0 + 1e-3))
    assert np.all((th[:, 6] >= 0.5 - 1e-3) & (th[:, 6] <= 8.0 + 1e-3))
    assert np.all((th[:, 3] >= 0.0) & (th[:, 3] <= np.pi + 1e-4))
    assert np.all((th[:, 4] >= 0.0) & (th[:, 4] <= 2 * np.pi + 1e-4))


def test_deterministic():
    """Same event -> byte-identical theta across repeated calls (fixed seed key)."""
    hitnet, chargenet = _toy_nets(jax.random.PRNGKey(5))
    padded = _toy_events(jax.random.PRNGKey(6), n_events=4, n_pad=32)
    solve = make_sequential_mle(hitnet, chargenet, CompiledMLEConfig(), n_pad=32)
    args = (padded.hits[0], padded.pmt_id[0], padded.t[0], padded.mask[0], padded.charge[0])
    a = solve(*args)
    b = solve(*args)
    assert np.array_equal(a, b)


def test_polish_is_monotone():
    """The exact-LM polish can only lower (or hold) the achieved NLL vs no polish."""
    hitnet, chargenet = _toy_nets(jax.random.PRNGKey(12))
    padded = _toy_events(jax.random.PRNGKey(13), n_events=8, n_pad=32)
    cfg = CompiledMLEConfig()
    no_polish = make_sequential_mle(hitnet, chargenet, cfg, n_pad=32, polish=0)
    with_polish = make_sequential_mle(hitnet, chargenet, cfg, n_pad=32, polish=8)
    th0 = np.stack([no_polish(padded.hits[i], padded.pmt_id[i], padded.t[i],
                              padded.mask[i], padded.charge[i]) for i in range(8)])
    th1 = np.stack([with_polish(padded.hits[i], padded.pmt_id[i], padded.t[i],
                                padded.mask[i], padded.charge[i]) for i in range(8)])
    nll0 = _nll_batch(hitnet, chargenet, th0, padded)
    nll1 = _nll_batch(hitnet, chargenet, th1, padded)
    assert np.all(nll1 <= nll0 + 1e-4), "polish increased NLL on some event"


def test_extras_and_fisher():
    """return_extras yields counts + a symmetric (7,7) Fisher when requested."""
    hitnet, chargenet = _toy_nets(jax.random.PRNGKey(7))
    padded = _toy_events(jax.random.PRNGKey(8), n_events=4, n_pad=32)
    solve = make_sequential_mle(hitnet, chargenet, CompiledMLEConfig(), n_pad=32,
                                compute_fisher=True, return_extras=True)
    res = solve(padded.hits[0], padded.pmt_id[0], padded.t[0], padded.mask[0],
                padded.charge[0])
    assert isinstance(res, SequentialMLEResult)
    assert res.fisher.shape == (7, 7)
    assert np.allclose(res.fisher, res.fisher.T, atol=1e-3)
    assert np.isfinite(res.nll)
    assert res.n_seeds >= 1 and res.n_grad >= 1


def test_degenerate_event_is_finite():
    """All-hits-on-one-PMT event still returns finite in-bounds theta."""
    hitnet, chargenet = _toy_nets(jax.random.PRNGKey(9))
    n_pad = 32
    pos = jnp.tile(jnp.array([500.0, 0.0, 0.0]), (n_pad, 1))
    t = jnp.full((n_pad,), 5.0)
    hits = jnp.concatenate([pos, t[:, None]], axis=1)
    mask = (jnp.arange(n_pad) < 10).astype(jnp.float32)
    hits = hits * mask[:, None]
    pmt_id = (jnp.zeros(n_pad) * mask).astype(jnp.int32)
    charge = jnp.array([13.0, 10.0])
    solve = make_sequential_mle(hitnet, chargenet, CompiledMLEConfig(), n_pad=n_pad)
    theta = solve(hits, pmt_id, hits[:, 3], mask, charge)
    assert np.all(np.isfinite(theta))
    assert np.all(np.abs(theta[:3]) <= 800.0 + 1e-3)


def test_t_scan_is_causal_argmin():
    """The causal t line scan returns a chart point no worse (in NLL) than the input,
    and it recovers a deliberately mis-set emission time."""
    hitnet, chargenet = _toy_nets(jax.random.PRNGKey(20))
    padded = _toy_events(jax.random.PRNGKey(21), n_events=6, n_pad=32)
    p = _build_primitives(hitnet, chargenet, CompiledMLEConfig(t_range=(-10.0, 10.0)), 32,
                          t_grid_n=16)
    t_scan, nll_true = p["t_scan"], p["nll_true"]
    for i in range(6):
        e_hit = jax.vmap(hitnet.embed_hit)(padded.hits[i])
        # start from a point with a deliberately bad emission time
        u0 = jnp.asarray(chart_from_theta(jnp.array([50.0, -30.0, 20.0, 1.0, 0.5, 9.5, 2.0]))
                         / CHART_SCALE)
        u_ts = t_scan(u0, e_hit, padded.mask[i], padded.charge[i])
        f0 = float(nll_true(u0, e_hit, padded.mask[i], padded.charge[i]))
        f1 = float(nll_true(u_ts, e_hit, padded.mask[i], padded.charge[i]))
        assert f1 <= f0 + 1e-4, f"t_scan increased NLL ({f1:.3f} > {f0:.3f})"
        # only the t slot (chart idx 6) is changed
        assert np.allclose(np.asarray(u0)[[0, 1, 2, 3, 4, 5, 7]],
                           np.asarray(u_ts)[[0, 1, 2, 3, 4, 5, 7]])


def test_subsample_matches_full_nll():
    """Coarse-to-fine subset search + full endgame lands at the same NLL as full-set
    search (the endgame/polish run on all hits, so accuracy is preserved)."""
    hitnet, chargenet = _toy_nets(jax.random.PRNGKey(22))
    padded = _toy_events(jax.random.PRNGKey(23), n_events=8, n_pad=32)
    cfg = CompiledMLEConfig()
    full = make_sequential_mle(hitnet, chargenet, cfg, n_pad=32, subsample=False)
    sub = make_sequential_mle(hitnet, chargenet, cfg, n_pad=32, subsample=True, n_sub=16)
    th_full = np.stack([full(padded.hits[i], padded.pmt_id[i], padded.t[i], padded.mask[i],
                             padded.charge[i]) for i in range(8)])
    th_sub = np.stack([sub(padded.hits[i], padded.pmt_id[i], padded.t[i], padded.mask[i],
                           padded.charge[i]) for i in range(8)])
    dnll = _nll_batch(hitnet, chargenet, th_sub, padded) - \
        _nll_batch(hitnet, chargenet, th_full, padded)
    assert np.median(dnll) < 0.1
    assert np.percentile(dnll, 90) < 0.5
    assert np.all(np.isfinite(th_sub))


def test_robust_t_still_bounded_and_deterministic():
    """robust_t + subsample keep the box and byte-for-byte determinism."""
    hitnet, chargenet = _toy_nets(jax.random.PRNGKey(24))
    padded = _toy_events(jax.random.PRNGKey(25), n_events=4, n_pad=32)
    cfg = CompiledMLEConfig(radius=800.0, half_height=800.0, t_range=(-10.0, 10.0),
                            e_range=(0.5, 8.0))
    solve = make_sequential_mle(hitnet, chargenet, cfg, n_pad=32, robust_t=True,
                                subsample=True, n_sub=16)
    args = (padded.hits[0], padded.pmt_id[0], padded.t[0], padded.mask[0], padded.charge[0])
    a, b = solve(*args), solve(*args)
    assert np.array_equal(a, b)
    th = np.stack([solve(padded.hits[i], padded.pmt_id[i], padded.t[i], padded.mask[i],
                         padded.charge[i]) for i in range(4)])
    assert np.all(np.abs(th[:, :3]) <= 800.0 + 1e-3)
    assert np.all((th[:, 5] >= -10.0 - 1e-3) & (th[:, 5] <= 10.0 + 1e-3))


def _toy_student(key, activation="hardswish", width=32, depth=3):
    """A width-32 HitNet-shaped search student with a configurable activation."""
    return HitNet(width=width, depth=depth, key=key, activation=activation)


@pytest.mark.parametrize("activation", ["mish", "swish", "softplus", "relu", "hardswish"])
def test_search_net_matches_teacher_only(activation):
    """The student-search + teacher-endgame path reaches an NLL competitive with the
    teacher-only path (the teacher endgame + polish protect accuracy) for every
    activation, stays bounded, and never returns NaN."""
    hitnet, chargenet = _toy_nets(jax.random.PRNGKey(30))
    student = _toy_student(jax.random.PRNGKey(31), activation)
    padded = _toy_events(jax.random.PRNGKey(32), n_events=8, n_pad=32)
    cfg = CompiledMLEConfig()
    base = make_sequential_mle(hitnet, chargenet, cfg, n_pad=32)
    srch = make_sequential_mle(hitnet, chargenet, cfg, n_pad=32, search_net=student)
    args = [(padded.hits[i], padded.pmt_id[i], padded.t[i], padded.mask[i],
             padded.charge[i]) for i in range(8)]
    th_b = np.stack([base(*a) for a in args])
    th_s = np.stack([srch(*a) for a in args])
    dnll = _nll_batch(hitnet, chargenet, th_s, padded) - \
        _nll_batch(hitnet, chargenet, th_b, padded)
    # teacher endgame protects the optimum: student path is no worse than a small margin
    assert np.median(dnll) < 0.1, f"{activation}: median dNLL {np.median(dnll):.3f}"
    assert np.percentile(dnll, 90) < 0.5
    assert np.all(np.isfinite(th_s))
    assert np.all(np.abs(th_s[:, :3]) <= 800.0 + 1e-3)


def test_search_net_counts_split_by_net():
    """return_extras reports student value+grads separately (n_grad_search) from the
    teacher's (n_grad); both are spent when a search net is supplied."""
    hitnet, chargenet = _toy_nets(jax.random.PRNGKey(33))
    student = _toy_student(jax.random.PRNGKey(34))
    padded = _toy_events(jax.random.PRNGKey(35), n_events=4, n_pad=32)
    solve = make_sequential_mle(hitnet, chargenet, CompiledMLEConfig(), n_pad=32,
                                search_net=student, return_extras=True)
    r = solve(padded.hits[0], padded.pmt_id[0], padded.t[0], padded.mask[0],
              padded.charge[0])
    assert isinstance(r, SequentialMLEResult)
    assert r.n_grad_search > 0 and r.n_grad > 0
    # teacher-only path spends zero student grads
    base = make_sequential_mle(hitnet, chargenet, CompiledMLEConfig(), n_pad=32,
                               return_extras=True)
    rb = base(padded.hits[0], padded.pmt_id[0], padded.t[0], padded.mask[0],
              padded.charge[0])
    assert rb.n_grad_search == 0


def test_search_net_both_and_one_seed_bounded_deterministic():
    """search_both_seeds True/False both stay in the box and are byte-deterministic."""
    hitnet, chargenet = _toy_nets(jax.random.PRNGKey(36))
    student = _toy_student(jax.random.PRNGKey(37))
    padded = _toy_events(jax.random.PRNGKey(38), n_events=4, n_pad=32)
    cfg = CompiledMLEConfig(radius=800.0, half_height=800.0)
    for both in (True, False):
        solve = make_sequential_mle(hitnet, chargenet, cfg, n_pad=32, search_net=student,
                                    search_both_seeds=both, robust_t=True)
        args = (padded.hits[0], padded.pmt_id[0], padded.t[0], padded.mask[0],
                padded.charge[0])
        a, b = solve(*args), solve(*args)
        assert np.array_equal(a, b), f"both={both} not deterministic"
        th = np.stack([solve(padded.hits[i], padded.pmt_id[i], padded.t[i],
                             padded.mask[i], padded.charge[i]) for i in range(4)])
        assert np.all(np.abs(th[:, :3]) <= 800.0 + 1e-3)
        assert np.all(np.isfinite(th))


def test_mlp_activation_static_and_backcompat():
    """The activation is a static field: a default-mish template deserialises weights
    from any width net unchanged, and each named activation round-trips through the
    separable embed path exactly (no activation on the linear first-layer embed)."""
    from hitman.nn.mlp import ACTIVATIONS, get_activation
    import equinox as eqx
    key = jax.random.PRNGKey(40)
    net = _toy_student(key, "swish", width=16, depth=2)
    blob = "/tmp/_r5_actnet.eqx"
    eqx.tree_serialise_leaves(blob, net)
    # deserialising into a template built with the SAME activation restores weights
    tmpl = _toy_student(key, "swish", width=16, depth=2)
    net2 = eqx.tree_deserialise_leaves(blob, tmpl)
    hit = jnp.array([100.0, -50.0, 20.0, 5.0]); hyp = jnp.array([1., 2., 3., 1., .5, 0., 2.])
    assert float(net(hit, hyp)) == float(net2(hit, hyp))
    # separable embed path equals the direct call for every activation
    for a in ACTIVATIONS:
        n = _toy_student(key, a, width=16, depth=2)
        direct = float(n(hit, hyp))
        sep = float(n.logit_from_embedding(n.embed_hit(hit) + n.embed_hyp(hyp)))
        assert abs(direct - sep) < 1e-4, f"{a}: embed path {sep} != direct {direct}"
    assert get_activation("mish") is not None
    with pytest.raises(ValueError):
        get_activation("not_an_activation")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))


def test_choose_buckets_matches_bruteforce():
    import itertools
    import numpy as np
    from hitman.wc.inference.seq import choose_buckets
    rng = np.random.default_rng(0)
    n = rng.integers(1, 128, 4000)
    ladder = choose_buckets(n, max_buckets=3, grid=16, b_min=32, b_max=128,
                            call_overhead_rows=8.0)
    # brute force over all 3-subsets of the candidate grid containing the top rung
    cands = [32, 48, 64, 80, 96, 112, 128]
    top = max(c for c in cands if c >= n.max()) if n.max() > 32 else 32
    def cost(lad):
        lad = sorted(lad)
        B = np.array(lad)[np.searchsorted(lad, n)]
        return np.sum(B + 8.0)
    best = None
    for k in (1, 2, 3):
        for combo in itertools.combinations([c for c in cands if c < 128], k - 1):
            lad = sorted(combo) + [128]
            if max(n) > 128:
                continue
            c = cost(lad)
            if best is None or c < best[0]:
                best = (c, lad)
    assert abs(cost(ladder) - best[0]) / best[0] < 1e-9, (ladder, best)


def test_choose_buckets_uniform_ladder_shape():
    import numpy as np
    from hitman.wc.inference.seq import choose_buckets
    n = np.random.default_rng(1).integers(1, 200, 20000)
    lad = choose_buckets(n, max_buckets=8, grid=16, b_min=32, b_max=256)
    assert lad[-1] >= 200 and len(lad) <= 8
    assert all(b2 > b1 for b1, b2 in zip(lad, lad[1:]))
    # near-even ladder expected for uniform load: max gap not absurdly larger than min
    gaps = np.diff([0] + lad)
    assert gaps.max() <= 4 * max(gaps.min(), 16)


def test_bucketed_solver_routes_and_agrees():
    import numpy as np, jax
    from hitman.wc.inference.seq import make_bucketed_mle, make_sequential_mle
    from hitman.wc.inference.compiled import CompiledMLEConfig
    from hitman.nn import ChargeNet, HitNet
    hitnet = HitNet(width=16, depth=2, key=jax.random.PRNGKey(0))
    chargenet = ChargeNet(width=16, depth=2, key=jax.random.PRNGKey(1))
    cfg = CompiledMLEConfig()
    kw = dict(polish=2, lbfgs_maxiter=15)
    solve = make_bucketed_mle(hitnet, chargenet, cfg, buckets=[48, 96], **kw)
    rng = np.random.default_rng(2)
    for n in (10, 60):
        hits = np.concatenate([rng.normal(0, 400, (n, 3)),
                               rng.normal(60, 15, (n, 1))], 1).astype(np.float32)
        pmt = rng.integers(0, 200, n).astype(np.int32)
        t = hits[:, 3].astype(np.float32)
        charge = np.array([n, n], np.float32)
        th_b = np.asarray(solve(hits, pmt, t, charge))
        assert np.isfinite(th_b).all()
        # reference: fixed-pad solver at the routed bucket must agree closely
        b = 48 if n <= 48 else 96
        hp = np.zeros((b, 4), np.float32); hp[:n] = hits
        ip = np.zeros(b, np.int32); ip[:n] = pmt
        tp = np.zeros(b, np.float32); tp[:n] = t
        mp = np.zeros(b, np.float32); mp[:n] = 1.0
        ref = make_sequential_mle(hitnet, chargenet, cfg, n_pad=b, **kw)
        th_r = np.asarray(ref(hp, ip, tp, mp, charge))
        np.testing.assert_allclose(th_b, th_r, atol=1e-4)
    # lazy compile: only the used buckets exist; oversized event raises
    import pytest
    with pytest.raises(ValueError):
        solve(np.zeros((200, 4), np.float32), np.zeros(200, np.int32),
              np.zeros(200, np.float32), np.array([200, 200], np.float32))
