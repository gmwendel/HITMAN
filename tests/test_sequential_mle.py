"""Tests for the sequential single-event MLE driver (hitman.inference.seq).

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

from hitman.inference.batched import (MLEConfig, PaddedEvents, batched_multistart_mle,
                                      make_padded_nll)
from hitman.inference.compiled import CompiledMLEConfig, make_compiled_mle
from hitman.inference.seq import make_sequential_mle, SequentialMLEResult
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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
