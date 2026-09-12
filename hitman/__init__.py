"""HITMAN -- simulation-based inference for particle-detector event reconstruction, in JAX.

This top-level package is the **detector-agnostic library**:

* :mod:`hitman.spec`        -- observation/hypothesis layout protocol
* :mod:`hitman.data`        -- grouped batch container + the input pipeline
* :mod:`hitman.nn`          -- the MLP building block
* :mod:`hitman.density`     -- exactly-normalized density factors
* :mod:`hitman.ratio`       -- the neural-ratio lane (pairing, training, support, bundles)
* :mod:`hitman.npe`         -- neural posterior estimation (DeepSets encoder + spline flow)
* :mod:`hitman.validate`    -- calibration / coverage / closure / OOD gates
* :mod:`hitman.diagnostics` -- the numeric calibration primitives those gates use
* :mod:`hitman.receipts`    -- detector-agnostic receipt core (chi2, IS weights, thresholds)
* :mod:`hitman.calibrate`   -- Godambe/sandwich adjustment

The water-Cherenkov application it grew out of -- hitnet/chargenet, the sensor grid, the
RAT-DS ingest, the deployment inference lanes, splinemle, the forward receipts -- lives in
:mod:`hitman.wc`. Old import paths still work with a ``DeprecationWarning``
(:mod:`hitman._compat`). The library must never import from ``hitman.wc``.

Inference lanes (all water-Cherenkov, all under ``hitman.wc.inference``):

* ``batched`` -- GPU throughput; lockstep-vmap many events, one compile total.
* ``seq``     -- single-core sequential CPU (the ratpac/Eos target).
* ``compiled``-- one exportable jitted graph (StableHLO / cppflow).
"""

__version__ = "2.1.0.dev0"

# PRECISION POLICY (2026-07-19): full f32 matmuls everywhere. On GPU, jax defaults to
# TF32 (rel err ~5e-4); the training step is bandwidth-bound so TF32 buys ~nothing,
# and this program's calibration receipts (Z gradients, sandwich Hessians, score
# identities) are exactly the small-difference quantities reduced precision pollutes
# first. Uniform full precision beats auditing TF32's harmlessness receipt by receipt.
# (BF16/TF32 opt-ins, if ever justified, must be explicit and local, never default.)
import jax as _jax

_jax.config.update("jax_default_matmul_precision", "highest")

from hitman._compat import deprecated_getattr
from hitman.data.structures import EventBatch

# Relocated to hitman.wc; served lazily so importing the library never drags in the
# water-Cherenkov application.
__getattr__ = deprecated_getattr(__name__, {
    "HitNet": "hitman.wc.nn.hitnet",
    "ChargeNet": "hitman.wc.nn.chargenet",
    "event_log_ratio": "hitman.wc.likelihood.event",
    "make_event_nll": "hitman.wc.likelihood.event",
})

__all__ = ["EventBatch", "HitNet", "ChargeNet", "event_log_ratio", "__version__"]
