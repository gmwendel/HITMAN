"""ChargeNet: per-event total-charge log likelihood-ratio estimator."""

import equinox as eqx
import jax.numpy as jnp

from hitman.wc.nn import features as ft
from hitman.nn.mlp import MLP


class ChargeNet(eqx.Module):
    mlp: MLP

    def __init__(self, width: int = 256, depth: int = 3, *, key):
        self.mlp = MLP(ft.N_CHARGE_FEATURES, width, depth, key=key)

    def __call__(self, charge: jnp.ndarray, hyp: jnp.ndarray) -> jnp.ndarray:
        """(charge (2,), hyp (7,)) -> scalar logit = log r_charge."""
        return self.mlp(ft.charge_features(charge, hyp))
