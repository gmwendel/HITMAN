"""Data containers and the input pipeline (detector-agnostic).

The RAT-DS ROOT ingest and the PMT-indexed memmap store moved to :mod:`hitman.wc.data`.
"""

from hitman._compat import deprecated_getattr
from hitman.data.structures import EventBatch

__getattr__ = deprecated_getattr(__name__, {
    "RatDSExtractor": "hitman.wc.data.ratds",
    "HitStore": "hitman.wc.data.store",
    "build_store": "hitman.wc.data.store",
})

__all__ = ["EventBatch", "RatDSExtractor", "HitStore", "build_store"]
