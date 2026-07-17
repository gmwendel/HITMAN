"""Memory-mapped event store: ROOT ntuples -> flat on-disk arrays, loaded lazily.

``build_store`` streams the input files in chunks (bounded RAM regardless of dataset
size) and writes the flat EventBatch arrays as ``.npy`` files. ``HitStore`` memory-maps
them: training and reconstruction then read pages on demand through the OS cache, so a
1M-event (~70M-hit, ~1.5 GB) dataset needs no more resident memory than the working set
of the current batch.

Layout of a store directory:
    hyp.npy         (n_events, 7) float32
    charge.npy      (n_events, 2) float32
    hits.npy        (n_hits, 4)   float32
    event_id.npy    (n_hits,)     int32
    hit_offsets.npy (n_events+1,) int64   — event e owns hits[hit_offsets[e]:hit_offsets[e+1]]
    pmt_pos.npy     (n_pmts, 3)   float32
    meta.json       provenance (input files, counts)
"""

import json
import os

import numpy as np

from hitman.data.ratds import count_events_hits, iter_flat_chunks, pmt_positions
from hitman.data.structures import EventBatch

_ARRAYS = {
    "hyp": ((7,), np.float32),
    "charge": ((2,), np.float32),
    "hits": ((4,), np.float32),
    "event_id": ((), np.int32),
    "pmt_id": ((), np.int32),
}
_PER_HIT = ("hits", "event_id", "pmt_id")


def build_store(input_files, out_dir, step_size: str = "200 MB") -> "HitStore":
    """Stream ROOT files into a memmap store; peak RAM is one chunk, not the dataset."""
    os.makedirs(out_dir, exist_ok=True)

    # Pass 1 (cheap): count events and hits reading only the small per-PMT NPE branch,
    # so the output arrays can be preallocated at their exact final size.
    n_events, n_hits = count_events_hits(input_files, step_size=step_size)
    pmt_pos = pmt_positions(input_files)

    mm = {
        name: np.lib.format.open_memmap(
            os.path.join(out_dir, f"{name}.npy"),
            mode="w+",
            dtype=dtype,
            shape=(n_hits if name in _PER_HIT else n_events, *tail),
        )
        for name, (tail, dtype) in _ARRAYS.items()
    }
    offsets = np.lib.format.open_memmap(
        os.path.join(out_dir, "hit_offsets.npy"), mode="w+", dtype=np.int64,
        shape=(n_events + 1,),
    )
    offsets[0] = 0

    # Pass 2: stream and fill.
    e0, h0 = 0, 0
    for chunk in iter_flat_chunks(input_files, pmt_pos, step_size=step_size):
        ne, nh = len(chunk.hyp), len(chunk.hits)
        mm["hyp"][e0 : e0 + ne] = chunk.hyp
        mm["charge"][e0 : e0 + ne] = chunk.charge
        mm["hits"][h0 : h0 + nh] = chunk.hits
        mm["event_id"][h0 : h0 + nh] = chunk.event_id + e0
        mm["pmt_id"][h0 : h0 + nh] = chunk.pmt_id
        counts = np.bincount(chunk.event_id, minlength=ne)
        offsets[e0 + 1 : e0 + ne + 1] = h0 + np.cumsum(counts)
        e0 += ne
        h0 += nh
    assert (e0, h0) == (n_events, n_hits), f"count mismatch: {(e0, h0)} != {(n_events, n_hits)}"

    for arr in (*mm.values(), offsets):
        arr.flush()
    np.save(os.path.join(out_dir, "pmt_pos.npy"), pmt_pos)
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(
            {"input_files": list(input_files), "n_events": n_events, "n_hits": n_hits}, f
        )
    return HitStore(out_dir)


class HitStore:
    """Lazy, memory-mapped view of a built store."""

    def __init__(self, path):
        self.path = path
        with open(os.path.join(path, "meta.json")) as f:
            self.meta = json.load(f)
        load = lambda name: np.load(os.path.join(path, name), mmap_mode="r")  # noqa: E731
        self.hyp = load("hyp.npy")
        self.charge = load("charge.npy")
        self.hits = load("hits.npy")
        self.event_id = load("event_id.npy")
        self.pmt_id = load("pmt_id.npy")
        self.hit_offsets = load("hit_offsets.npy")
        self.pmt_pos = load("pmt_pos.npy")

    @property
    def n_events(self) -> int:
        return self.hyp.shape[0]

    @property
    def n_hits(self) -> int:
        return self.hits.shape[0]

    def event_batch(self, event_indices) -> EventBatch:
        """Materialize the given events (only) as an in-RAM EventBatch.

        O(selected events) via the offset table — no scan over the full hit array.
        """
        event_indices = np.asarray(event_indices)
        starts = self.hit_offsets[event_indices]
        stops = self.hit_offsets[event_indices + 1]
        hit_idx = np.concatenate(
            [np.arange(a, b, dtype=np.int64) for a, b in zip(starts, stops)]
        ) if len(event_indices) else np.empty(0, dtype=np.int64)
        event_id = np.repeat(
            np.arange(len(event_indices), dtype=np.int32), (stops - starts).astype(np.int64)
        )
        return EventBatch(
            hyp=np.asarray(self.hyp[event_indices]),
            charge=np.asarray(self.charge[event_indices]),
            hits=np.asarray(self.hits[hit_idx]),
            event_id=event_id,
            pmt_id=np.asarray(self.pmt_id[hit_idx]),
        )
