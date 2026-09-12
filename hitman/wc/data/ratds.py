"""Extraction of HITMAN training/reco data from RAT-PAC ntuple ROOT files.

All extraction is chunked and vectorized: files are streamed with ``uproot.iterate``
into awkward arrays and flattened with O(1) Python-object overhead per *chunk* (never
per event or per hit). ``RatDSExtractor.load()`` concatenates the chunks in RAM for
small datasets; for large ones, ``hitman.wc.data.store.build_store`` writes the same
chunks straight to a memory-mapped on-disk store.
"""

import awkward as ak
import numpy as np
import uproot

from hitman.data.structures import EventBatch

TRUTH_BRANCHES = ["mcx", "mcy", "mcz", "mcu", "mcv", "mcw", "mcke"]


def _resolve_files(input_files):
    """Validate inputs, returning (path, output_tree_key) pairs."""
    resolved = []
    for infile in input_files:
        try:
            with uproot.open(infile) as f:
                out_keys = [k for k in f.keys() if k.startswith("output")]
                if not out_keys:
                    print(f"Warning: no output tree in {infile}, skipping")
                    continue
                key = max(out_keys, key=lambda k: int(k.rsplit(";", 1)[-1]))
                has_pe_id = "mcPEPMTID" in f[key].keys()
            resolved.append((infile, key, has_pe_id))
        except Exception as err:  # noqa: BLE001 - any unreadable file is skipped
            print(f"Warning: opening {infile} failed ({err}), skipping")
    if not resolved:
        raise ValueError("No valid input files")
    return resolved


def pmt_geometry(input_files):
    """PMT geometry from the meta tree; must agree across files.

    Returns (positions (n,3), directions (n,3), types (n,)); directions are the
    pmtU/V/W unit vectors (x/y/z-aligned components of the PMT axis). Files without
    direction branches yield directions=None.
    """
    pos = dirs = types = None
    for infile, _, _ in _resolve_files(input_files):
        with uproot.open(infile) as f:
            meta = f["meta;1"]
            names = ["pmtX", "pmtY", "pmtZ"]
            has_dir = all(b in meta.keys() for b in ("pmtU", "pmtV", "pmtW"))
            if has_dir:
                names += ["pmtU", "pmtV", "pmtW"]
            has_type = "pmtType" in meta.keys()
            if has_type:
                names.append("pmtType")
            m = meta.arrays(names, library="np")
        p = np.stack([m["pmtX"][0], m["pmtY"][0], m["pmtZ"][0]], axis=1).astype(np.float32)
        d = (np.stack([m["pmtU"][0], m["pmtV"][0], m["pmtW"][0]], axis=1).astype(np.float32)
             if has_dir else None)
        ty = m["pmtType"][0].astype(np.int32) if has_type else None
        if d is not None:
            norms = np.linalg.norm(d, axis=1)
            if not np.allclose(norms, 1.0, atol=1e-3):
                d = d / np.clip(norms, 1e-9, None)[:, None]
        if pos is None:
            pos, dirs, types = p, d, ty
        elif p.shape != pos.shape or not np.allclose(p, pos):
            raise ValueError(f"{infile}: PMT geometry differs from {input_files[0]}")
    return pos, dirs, types


def pmt_positions(input_files) -> np.ndarray:
    """(n_pmts, 3) PMT positions from the meta tree; must agree across files."""
    return pmt_geometry(input_files)[0]


def count_events_hits(input_files, step_size: str = "200 MB"):
    """(n_events, n_hits) totals, reading only the small per-PMT NPE branch."""
    n_events, n_hits = 0, 0
    files = [(path, key) for path, key, _ in _resolve_files(input_files)]
    for arr in uproot.iterate(
        [f"{path}:{key}" for path, key in files],
        filter_name=["mcPMTNPE"],
        step_size=step_size,
    ):
        npe = arr["mcPMTNPE"]
        n_events += len(npe)
        n_hits += int(ak.sum(npe))
    return n_events, n_hits


def iter_flat_chunks(input_files, pmt_pos: np.ndarray, step_size: str = "200 MB"):
    """Yield EventBatch chunks (numpy, event_id local to the chunk) from ROOT files.

    Single pass over each file; peak memory is one decompressed chunk.
    """
    for path, key, has_pe_id in _resolve_files(input_files):
        branches = TRUTH_BRANCHES + ["mcPEFrontEndTime"] + (
            ["mcPEPMTID"] if has_pe_id else ["mcPMTID", "mcPMTNPE"]
        )
        for arr in uproot.iterate(f"{path}:{key}", filter_name=branches, step_size=step_size):
            yield _flatten_chunk(arr, pmt_pos, has_pe_id)


def _flatten_chunk(arr, pmt_pos, has_pe_id) -> EventBatch:
    azimuth = np.mod(np.arctan2(ak.to_numpy(arr["mcv"]), ak.to_numpy(arr["mcu"])), 2 * np.pi)
    zenith = np.arccos(ak.to_numpy(arr["mcw"]))
    n_events = len(zenith)
    hyp = np.stack(
        [
            ak.to_numpy(arr["mcx"]),
            ak.to_numpy(arr["mcy"]),
            ak.to_numpy(arr["mcz"]),
            zenith,
            azimuth,
            np.zeros(n_events),
            ak.to_numpy(arr["mcke"]),
        ],
        axis=1,
    ).astype(np.float32)

    times = arr["mcPEFrontEndTime"]
    n_hits_per_event = ak.to_numpy(ak.num(times)).astype(np.int64)
    # Per-PE PMT index: direct when the file has it; otherwise expand the per-PMT
    # (ID, NPE) pairs — grouped in the same order as the PE times, so a flat repeat
    # reproduces the per-PE association.
    if has_pe_id:
        pe_pmt = ak.to_numpy(ak.flatten(arr["mcPEPMTID"])).astype(np.int64)
    else:
        pe_pmt = np.repeat(
            ak.to_numpy(ak.flatten(arr["mcPMTID"])), ak.to_numpy(ak.flatten(arr["mcPMTNPE"]))
        ).astype(np.int64)

    hits = np.empty((len(pe_pmt), 4), dtype=np.float32)
    hits[:, :3] = pmt_pos[pe_pmt]
    hits[:, 3] = ak.to_numpy(ak.flatten(times))
    event_id = np.repeat(np.arange(n_events, dtype=np.int32), n_hits_per_event)

    # 1.x convention post-MC-truth switch: "charge" and hit count are both nhit.
    charge = np.stack([n_hits_per_event, n_hits_per_event], axis=1).astype(np.float32)

    return EventBatch(
        hyp=hyp, charge=charge, hits=hits, event_id=event_id,
        pmt_id=pe_pmt.astype(np.int32),
    )


class RatDSExtractor:
    """In-RAM extraction for small datasets (chunked internally; one file pass)."""

    def __init__(self, input_files):
        self.input_files = [path for path, _, _ in _resolve_files(input_files)]

    def load(self) -> EventBatch:
        pmt_pos = pmt_positions(self.input_files)
        chunks = list(iter_flat_chunks(self.input_files, pmt_pos))
        offsets = np.cumsum([0] + [c.n_events for c in chunks[:-1]])
        return EventBatch(
            hyp=np.concatenate([c.hyp for c in chunks]),
            charge=np.concatenate([c.charge for c in chunks]),
            hits=np.concatenate([c.hits for c in chunks]),
            event_id=np.concatenate(
                [c.event_id + off for c, off in zip(chunks, offsets)]
            ),
            pmt_id=np.concatenate([c.pmt_id for c in chunks]),
        )
