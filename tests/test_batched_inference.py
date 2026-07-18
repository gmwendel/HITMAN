import warnings

import jax
import jax.numpy as jnp
import numpy as np

from hitman.inference import MLEConfig, batched_multistart_mle, batched_nuts, pad_events
from hitman.nn import ChargeNet, HitNet
from tests.test_resident import _toy_data


def test_pad_events_matches_manual_loop():
    batch, _ = _toy_data(n_events=100, seed=11)
    counts = np.asarray(batch.charge[:, 1]).astype(int)
    n_pad = int(np.percentile(counts, 90))    # force some drops
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        p = pad_events(batch, n_pad=n_pad)
        assert any("dropping" in str(x.message) for x in w)
    assert p.dropped_fraction > 0
    offsets = np.concatenate([[0], np.cumsum(counts)])
    hits = np.asarray(batch.hits)
    ids = np.asarray(batch.pmt_id)
    for k, e in enumerate(p.event_indices[:20]):
        n = counts[e]
        np.testing.assert_allclose(np.asarray(p.hits[k, :n]), hits[offsets[e]:offsets[e]+n],
                                   rtol=1e-6)
        np.testing.assert_array_equal(np.asarray(p.pmt_id[k, :n]), ids[offsets[e]:offsets[e]+n])
        assert np.asarray(p.mask[k]).sum() == n
        np.testing.assert_allclose(np.asarray(p.hits[k, n:]), 0.0)


def test_batched_mle_and_nuts_smoke():
    batch, _ = _toy_data(n_events=60, seed=12)
    p = pad_events(batch, indices=np.arange(8), n_pad=64)
    hitnet = HitNet(width=16, depth=2, key=jax.random.PRNGKey(0))
    chargenet = ChargeNet(width=16, depth=2, key=jax.random.PRNGKey(1))
    cfg = MLEConfig(n_seeds=32, top_k=4, descent_steps=20)
    mle = batched_multistart_mle(hitnet, chargenet, p, key=jax.random.PRNGKey(2),
                                 cfg=cfg, keep_minima=2, chunk=4)
    n = p.hits.shape[0]
    assert mle.theta.shape == (n, 7) and mle.chart_minima.shape == (n, 2, 8)
    assert np.isfinite(np.asarray(mle.theta)).all()
    assert np.isfinite(np.asarray(mle.sigma)).all()
    res = batched_nuts(hitnet, chargenet, p, mle, key=jax.random.PRNGKey(3), cfg=cfg,
                       n_steps=30, n_burn=10, n_chains=2, chunk=4)
    assert res.samples.shape == (n, 2 * 20, 7)
    assert np.isfinite(res.post_mean).all()
    assert (res.rhat > 0).all()
