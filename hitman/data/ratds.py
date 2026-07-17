"""Extraction of HITMAN training/reco data from RAT-PAC ntuple ROOT files.

Port of the 1.x ``DataExtractor`` to the flat ``EventBatch`` layout: hypotheses are
stored once per event and hits carry an ``event_id`` index instead of repeated
hypothesis rows.
"""

import numpy as np
import uproot

from hitman.data.structures import EventBatch

TRUTH_BRANCHES = ["mcx", "mcy", "mcz", "mcu", "mcv", "mcw", "mcke"]
HIT_BRANCHES = ["mcPMTID", "mcPMTNPE", "mcPEFrontEndTime", "mcPEPMTID"]


class RatDSExtractor:
    """Reads RAT-PAC ntuples (``output`` tree + ``meta`` PMT geometry) into EventBatch."""

    def __init__(self, input_files):
        self.input_files = []
        self.out_keys = []
        for infile in input_files:
            key = self._valid_output_key(infile)
            if key is not None:
                self.input_files.append(infile)
                self.out_keys.append(key)
        if not self.input_files:
            raise ValueError("No valid input files")

    @staticmethod
    def _valid_output_key(infile):
        """Return the highest-cycle 'output' tree key, or None if the file is unusable."""
        try:
            with uproot.open(infile) as f:
                out_keys = [k for k in f.keys() if k.startswith("output")]
                if not out_keys:
                    print(f"Warning: no output tree in {infile}, skipping")
                    return None
                return max(out_keys, key=lambda k: int(k.rsplit(";", 1)[-1]))
        except Exception as err:  # noqa: BLE001 - any unreadable file is skipped
            print(f"Warning: opening {infile} failed ({err}), skipping")
            return None

    def _concat(self, branches):
        sources = [f"{path}:{key}" for path, key in zip(self.input_files, self.out_keys)]
        return uproot.concatenate(sources, filter_name=branches, library="np")

    def _pmt_positions(self):
        maps = uproot.concatenate(
            [f"{path}:meta;1" for path in self.input_files],
            filter_name=["pmtX", "pmtY", "pmtZ"],
            library="np",
        )
        return np.stack([maps["pmtX"][0], maps["pmtY"][0], maps["pmtZ"][0]], axis=1).astype(np.float32)

    def load(self) -> EventBatch:
        """Extract all events as one EventBatch (numpy arrays; call .to_device() for JAX)."""
        truth = self._concat(TRUTH_BRANCHES)
        obs = self._concat(HIT_BRANCHES)
        pmt_pos = self._pmt_positions()

        azimuth = np.mod(np.arctan2(truth["mcv"], truth["mcu"]), 2 * np.pi)
        zenith = np.arccos(truth["mcw"])
        n_events = len(zenith)
        hyp = np.stack(
            [
                truth["mcx"],
                truth["mcy"],
                truth["mcz"],
                zenith,
                azimuth,
                np.zeros(n_events),
                truth["mcke"],
            ],
            axis=1,
        ).astype(np.float32)

        times = obs["mcPEFrontEndTime"]
        n_hits_per_event = np.array([len(t) for t in times], dtype=np.int32)
        # Per-PE PMT index: use mcPEPMTID directly when present; otherwise expand the
        # per-PMT (ID, NPE) pairs, which are stored in the same grouped order as the PE times.
        if "mcPEPMTID" in obs and len(obs["mcPEPMTID"]) and obs["mcPEPMTID"][0] is not None:
            pe_pmt = np.concatenate(obs["mcPEPMTID"]).astype(np.int64)
        else:
            pe_pmt = np.repeat(
                np.concatenate(obs["mcPMTID"]), np.concatenate(obs["mcPMTNPE"])
            ).astype(np.int64)

        hits = np.concatenate(
            [pmt_pos[pe_pmt], np.concatenate(times).astype(np.float32)[:, None]], axis=1
        ).astype(np.float32)
        event_id = np.repeat(np.arange(n_events, dtype=np.int32), n_hits_per_event)

        # 1.x convention post-MC-truth switch: "charge" and hit count are both nhit.
        charge = np.stack([n_hits_per_event, n_hits_per_event], axis=1).astype(np.float32)

        return EventBatch(hyp=hyp, charge=charge, hits=hits, event_id=event_id)
