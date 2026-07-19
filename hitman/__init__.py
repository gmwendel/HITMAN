"""HITMAN 2.0 — neural likelihood-ratio event reconstruction in JAX.

Networks (hitnet, chargenet) are equinox pytrees; the learned log likelihood-to-evidence
ratio composes per event via segment sums, and inference (multistart MLE, NUTS) runs as
jitted JAX programs suitable for graph export.
"""

__version__ = "2.0.0.dev0"

# PRECISION POLICY (2026-07-19): full f32 matmuls everywhere. On GPU, jax defaults to
# TF32 (rel err ~5e-4); the training step is bandwidth-bound so TF32 buys ~nothing,
# and this program's calibration receipts (Z gradients, sandwich Hessians, score
# identities) are exactly the small-difference quantities reduced precision pollutes
# first. Uniform full precision beats auditing TF32's harmlessness receipt by receipt.
# (BF16/TF32 opt-ins, if ever justified, must be explicit and local, never default.)
import jax as _jax

_jax.config.update("jax_default_matmul_precision", "highest")

from hitman.data.structures import EventBatch
from hitman.nn.hitnet import HitNet
from hitman.nn.chargenet import ChargeNet
from hitman.likelihood.event import event_log_ratio

__all__ = ["EventBatch", "HitNet", "ChargeNet", "event_log_ratio", "__version__"]
