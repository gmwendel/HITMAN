import os

import numpy as np
import pytest

TEST_FILE = "/tank/playground/eos_validation/sweep5000/data/water_1MeV.root"

pytestmark = pytest.mark.skipif(
    not os.path.exists(TEST_FILE), reason="Eos simulation test file not available"
)


def test_extract_water_1mev():
    from hitman.data import RatDSExtractor

    batch = RatDSExtractor([TEST_FILE]).load()

    assert batch.hyp.shape[1] == 7
    assert batch.charge.shape == (batch.n_events, 2)
    assert batch.hits.shape[1] == 4
    assert batch.event_id.shape == (batch.n_hits,)
    assert batch.event_id.max() < batch.n_events
    # hit counts consistent between charge column and event_id
    counts = np.bincount(batch.event_id, minlength=batch.n_events)
    np.testing.assert_array_equal(counts, batch.charge[:, 1].astype(int))
    assert np.isfinite(batch.hyp).all()
    assert np.isfinite(batch.hits).all()
    # zenith/azimuth in physical range
    assert (batch.hyp[:, 3] >= 0).all() and (batch.hyp[:, 3] <= np.pi).all()
    assert (batch.hyp[:, 4] >= 0).all() and (batch.hyp[:, 4] < 2 * np.pi).all()


def test_log_ratio_on_real_events_smoke():
    import jax

    from hitman import event_log_ratio
    from hitman.data import RatDSExtractor
    from hitman.nn import ChargeNet, HitNet

    batch = RatDSExtractor([TEST_FILE]).load().select(np.arange(50)).to_device()
    hitnet = HitNet(width=32, depth=2, key=jax.random.PRNGKey(0))
    chargenet = ChargeNet(width=32, depth=2, key=jax.random.PRNGKey(1))
    log_r = np.asarray(event_log_ratio(hitnet, chargenet, batch, batch.hyp))
    assert log_r.shape == (50,)
    assert np.isfinite(log_r).all()
