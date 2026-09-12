"""Input pipeline: the useful parts of tf.data, without TensorFlow.

Two mechanisms are lifted; the rest of tf.data is unnecessary here:

- ``block_shuffled_rows`` = tf.data's shuffle *buffer*. A full uniform permutation of a
  memory-mapped dataset turns every batch into random 4 KB page reads scattered over
  the whole file — the disk seeks, not streams. Instead: shuffle the order of large
  contiguous blocks, pool several blocks in a buffer, shuffle within the buffer, and
  emit batches sorted ascending. Reads stay block-sequential (page-cache and
  ZFS-friendly) while batch composition stays well-mixed. Events are written in i.i.d.
  simulation order, so buffer shuffling is statistically equivalent to a full
  permutation for SGD.

- ``prefetch`` = tf.data's ``.prefetch(n)``. A daemon thread assembles the next
  batches (memmap gather + host->device transfer) while the device runs the current
  train step, hiding I/O latency behind compute.

Deliberately not lifted: ``.cache()`` (the OS page cache already does this for
memmaps), parallel ``.map`` (batch assembly is a trivial gather; one thread hides it),
and the graph/AUTOTUNE machinery (nothing to tune).
"""

import queue
import threading

import numpy as np


def block_shuffled_rows(
    n: int,
    batch_size: int,
    rng: np.random.Generator,
    block_size: int = 2**16,
    buffer_blocks: int = 8,
):
    """Yield sorted row-index arrays of length ``batch_size`` covering ``[0, n)``.

    Randomness: block order is shuffled globally, rows are shuffled within a
    ``buffer_blocks * block_size``-row buffer (~0.5M rows by default). The trailing
    partial batch is dropped (jit needs static shapes).
    """
    starts = np.arange(0, n, block_size)
    rng.shuffle(starts)
    for g in range(0, len(starts), buffer_blocks):
        rows = np.concatenate(
            [np.arange(s, min(s + block_size, n), dtype=np.int64) for s in starts[g : g + buffer_blocks]]
        )
        rng.shuffle(rows)
        for i in range(0, len(rows) - batch_size + 1, batch_size):
            yield np.sort(rows[i : i + batch_size])


def prefetch(iterator, size: int = 2):
    """Run ``iterator`` in a daemon thread, keeping up to ``size`` items staged.

    Exceptions in the worker are re-raised at the consuming site.
    """
    q = queue.Queue(maxsize=size)
    _end = object()

    def worker():
        try:
            for item in iterator:
                q.put(item)
        except BaseException as err:  # noqa: BLE001 - forwarded to consumer
            q.put((_end, err))
            return
        q.put((_end, None))

    threading.Thread(target=worker, daemon=True).start()
    while True:
        item = q.get()
        if isinstance(item, tuple) and len(item) == 2 and item[0] is _end:
            if item[1] is not None:
                raise item[1]
            return
        yield item
