"""Detector-agnostic network building blocks.

Only :class:`~hitman.nn.mlp.MLP` is general. HitNet / ChargeNet / FrameHitNet are
water-Cherenkov networks and now live in :mod:`hitman.wc.nn`; they are still reachable
from here, lazily, with a ``DeprecationWarning``.
"""

from hitman._compat import deprecated_getattr
from hitman.nn.mlp import MLP, ACTIVATIONS, get_activation, mish

__getattr__ = deprecated_getattr(__name__, {
    "HitNet": "hitman.wc.nn.hitnet",
    "ChargeNet": "hitman.wc.nn.chargenet",
    "FrameHitNet": "hitman.wc.nn.frame",
    "SensorFrame": "hitman.wc.nn.frame",
    "compute_e_ref": "hitman.wc.nn.frame",
})

__all__ = ["MLP", "ACTIVATIONS", "get_activation", "mish",
           "HitNet", "ChargeNet", "FrameHitNet", "SensorFrame", "compute_e_ref"]
