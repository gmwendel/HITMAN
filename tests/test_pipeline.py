import numpy as np
import pytest

from hitman.data.pipeline import block_shuffled_rows, prefetch


def test_block_shuffle_covers_everything_once():
    rng = np.random.default_rng(0)
    n, bs = 10_000, 128
    rows = np.concatenate(list(block_shuffled_rows(n, bs, rng, block_size=1000, buffer_blocks=3)))
    assert len(rows) == (len(rows) // bs) * bs  # whole batches only
    assert len(np.unique(rows)) == len(rows)  # no duplicates
    assert len(rows) >= n - bs * 4  # only buffer tails dropped
    assert rows.min() >= 0 and rows.max() < n


def test_block_shuffle_batches_sorted_and_mixed():
    rng = np.random.default_rng(1)
    batches = list(block_shuffled_rows(50_000, 256, rng, block_size=4096, buffer_blocks=4))
    for b in batches[:5]:
        assert (np.diff(b) > 0).all()  # sorted -> near-sequential memmap reads
    # batches must mix rows across blocks, not return one contiguous run
    spans = [b.max() - b.min() for b in batches]
    assert np.median(spans) > 4096


def test_block_shuffle_epochs_differ():
    rng = np.random.default_rng(2)
    a = next(block_shuffled_rows(10_000, 64, rng, block_size=512))
    b = next(block_shuffled_rows(10_000, 64, rng, block_size=512))
    assert not np.array_equal(a, b)


def test_prefetch_preserves_order_and_content():
    items = list(range(50))
    assert list(prefetch(iter(items), size=4)) == items


def test_prefetch_propagates_exceptions():
    def gen():
        yield 1
        raise ValueError("boom")

    it = prefetch(gen(), size=2)
    assert next(it) == 1
    with pytest.raises(ValueError, match="boom"):
        list(it)
