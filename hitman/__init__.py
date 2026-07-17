"""HITMAN 2.0 — neural likelihood-ratio event reconstruction in JAX.

Networks (hitnet, chargenet) are equinox pytrees; the learned log likelihood-to-evidence
ratio composes per event via segment sums, and inference (multistart MLE, NUTS) runs as
jitted JAX programs suitable for graph export.
"""

__version__ = "2.0.0.dev0"

from hitman.data.structures import EventBatch
from hitman.nn.hitnet import HitNet
from hitman.nn.chargenet import ChargeNet
from hitman.likelihood.event import event_log_ratio

__all__ = ["EventBatch", "HitNet", "ChargeNet", "event_log_ratio", "__version__"]
