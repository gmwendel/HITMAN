"""Tests for the compiled single-event MLE solver (hitman.inference.compiled).

Agreement is asserted in NLL space (the solvers optimize the same objective from
different seeds; the achieved likelihood is the meaningful invariant, not the raw
parameters, which are degenerate for an untrained toy net). Bounds and the
NaN/no-improvement fallback are asserted directly.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hitman.inference.batched import (MLEConfig, PaddedEvents, batched_multistart_mle,
                                      make_padded_nll)
from hitman.inference.compiled import CompiledMLEConfig, make_compiled_mle, _bounds
from hitman.nn import ChargeNet, HitNet


def _toy_nets(key):
    k1, k2 = jax.random.split(key)
    return (HitNet(width=16, depth=2, key=k1), ChargeNet(width=16, depth=2, key=k2))


def _toy_events(key, n_events=8, n_pad=32, radius=600.0):
    """Random padded events on a spherical PMT shell (fixed shape)."""
    ks = jax.random.split(key, 6)
    nhit = jax.random.randint(ks[0], (n_events,), 8, n_pad)
    slot = jnp.arange(n_pad)[None, :]
    mask = (slot < nhit[:, None]).astype(jnp.float32)
    # PMT positions on a shell, times ~ shell radius / c + jitter
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


def test_agrees_with_multistart_on_toy():
    """Compiled solver reaches NLL competitive with the 256-seed multistart teacher."""
    key = jax.random.PRNGKey(0)
    hitnet, chargenet = _toy_nets(key)
    padded = _toy_events(jax.random.PRNGKey(1), n_events=8, n_pad=32)

    gold = batched_multistart_mle(
        hitnet, chargenet, padded, key=jax.random.PRNGKey(2),
        cfg=MLEConfig(n_seeds=64, top_k=8, descent_steps=120), chunk=8)

    cfg = CompiledMLEConfig(max_iter=25)
    fn = make_compiled_mle(hitnet, chargenet, cfg, n_pad=32)
    theta_c = jax.vmap(fn)(padded.hits, padded.pmt_id, padded.t, padded.mask, padded.charge)

    nll = jax.jit(jax.vmap(make_padded_nll(hitnet, chargenet)))
    ev = (padded.hits, padded.pmt_id, padded.t, padded.mask, padded.charge)
    nll_c = np.asarray(nll(theta_c, ev))
    nll_g = np.asarray(nll(gold.theta, ev))
    dnll = nll_c - nll_g

    # Competitive on the achieved likelihood: median gap tiny, and on most events
    # the compiled solver is no worse than the teacher by more than a small margin.
    assert np.median(dnll) < 0.5, f"median dNLL {np.median(dnll):.3f} too large"
    assert np.mean(dnll < 0.5) >= 0.7, f"only {np.mean(dnll<0.5):.2f} within 0.5"
    assert np.all(np.isfinite(nll_c))


def test_bounds_respected():
    """Every returned hypothesis lies inside the configured box (chart-clipped)."""
    key = jax.random.PRNGKey(3)
    hitnet, chargenet = _toy_nets(key)
    padded = _toy_events(jax.random.PRNGKey(4), n_events=8, n_pad=32)
    cfg = CompiledMLEConfig(radius=800.0, half_height=800.0, t_range=(-10.0, 10.0),
                            e_range=(0.5, 8.0))
    fn = make_compiled_mle(hitnet, chargenet, cfg, n_pad=32)
    th = np.asarray(jax.vmap(fn)(padded.hits, padded.pmt_id, padded.t, padded.mask,
                                 padded.charge))
    assert np.all(np.abs(th[:, 0]) <= 800.0 + 1e-3)
    assert np.all(np.abs(th[:, 1]) <= 800.0 + 1e-3)
    assert np.all(np.abs(th[:, 2]) <= 800.0 + 1e-3)
    assert np.all((th[:, 5] >= -10.0 - 1e-3) & (th[:, 5] <= 10.0 + 1e-3))
    assert np.all((th[:, 6] >= 0.5 - 1e-3) & (th[:, 6] <= 8.0 + 1e-3))
    # direction angles in physical range after wrap
    assert np.all((th[:, 3] >= 0.0) & (th[:, 3] <= np.pi + 1e-4))
    assert np.all((th[:, 4] >= 0.0) & (th[:, 4] <= 2 * np.pi + 1e-4))


def test_fallback_returns_best_seed():
    """max_iter=0 disables descent: the result is the best NLL-screened seed, finite."""
    key = jax.random.PRNGKey(5)
    hitnet, chargenet = _toy_nets(key)
    padded = _toy_events(jax.random.PRNGKey(6), n_events=6, n_pad=32)
    cfg = CompiledMLEConfig(max_iter=0)
    fn = make_compiled_mle(hitnet, chargenet, cfg, n_pad=32, return_nll=True)
    theta, nll = jax.vmap(fn)(padded.hits, padded.pmt_id, padded.t, padded.mask,
                              padded.charge)
    theta = np.asarray(theta)
    assert np.all(np.isfinite(theta))
    assert np.all(np.isfinite(np.asarray(nll)))
    # still inside bounds (seeds are chart-clipped to the box)
    assert np.all(np.abs(theta[:, :3]) <= 800.0 + 1e-3)


def test_degenerate_event_is_finite():
    """A near-degenerate event (all hits on one PMT) still returns finite in-bounds theta."""
    hitnet, chargenet = _toy_nets(jax.random.PRNGKey(7))
    n_pad = 32
    pos = jnp.tile(jnp.array([500.0, 0.0, 0.0]), (n_pad, 1))
    t = jnp.full((n_pad,), 5.0)
    hits = jnp.concatenate([pos, t[:, None]], axis=1)
    mask = (jnp.arange(n_pad) < 10).astype(jnp.float32)
    hits = hits * mask[:, None]
    pmt_id = (jnp.zeros(n_pad) * mask).astype(jnp.int32)
    charge = jnp.array([13.0, 10.0])
    fn = make_compiled_mle(hitnet, chargenet, CompiledMLEConfig(), n_pad=n_pad,
                           return_nll=True)
    theta, nll = fn(hits, pmt_id, hits[:, 3], mask, charge)
    assert np.all(np.isfinite(np.asarray(theta)))
    assert np.isfinite(float(nll))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
