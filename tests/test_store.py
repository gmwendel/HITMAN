import os

import numpy as np
import pytest

TEST_FILE = "/tank/playground/eos_validation/sweep5000/data/water_1MeV.root"

pytestmark = pytest.mark.skipif(
    not os.path.exists(TEST_FILE), reason="Eos simulation test file not available"
)


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    from hitman.data import build_store

    return build_store([TEST_FILE], tmp_path_factory.mktemp("store"), step_size="5 MB")


def test_store_matches_in_ram_extractor(store):
    from hitman.data import RatDSExtractor

    batch = RatDSExtractor([TEST_FILE]).load()
    assert store.n_events == batch.n_events
    assert store.n_hits == batch.n_hits
    np.testing.assert_array_equal(np.asarray(store.hyp), batch.hyp)
    np.testing.assert_array_equal(np.asarray(store.charge), batch.charge)
    np.testing.assert_array_equal(np.asarray(store.hits), batch.hits)
    np.testing.assert_array_equal(np.asarray(store.event_id), batch.event_id)


def test_store_offsets_consistent(store):
    counts = np.diff(store.hit_offsets)
    np.testing.assert_array_equal(counts, np.asarray(store.charge[:, 1]).astype(np.int64))
    assert store.hit_offsets[-1] == store.n_hits
    # event_id agrees with the offset table
    e = store.n_events // 2
    lo, hi = store.hit_offsets[e], store.hit_offsets[e + 1]
    assert (np.asarray(store.event_id[lo:hi]) == e).all()


def test_event_batch_selection(store):
    idx = np.array([0, 5, store.n_events - 1])
    sub = store.event_batch(idx)
    assert sub.n_events == 3
    np.testing.assert_array_equal(sub.hyp, np.asarray(store.hyp[idx]))
    counts = np.diff(store.hit_offsets)[idx]
    assert sub.n_hits == counts.sum()
    np.testing.assert_array_equal(np.bincount(sub.event_id, minlength=3), counts)


def test_memmaps_are_lazy(store):
    # Arrays must be memmaps, not RAM copies.
    assert isinstance(store.hits, np.memmap)
    assert isinstance(store.event_id, np.memmap)
