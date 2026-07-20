"""DeepSets context encoder: a padded event's hits -> a fixed-width context c(x).

The encoder must be HYPOTHESIS-FREE (amortized: one context per event, reused for every
posterior query), so the SensorFrame invariants in ``hitman.nn.frame`` do NOT drop in
here -- they are functions of the vertex/direction hypothesis, which is exactly what the
flow infers. The v0 per-hit feature map is therefore the raw sensor coordinate and hit
time (normalized by the shared 1.x scales), which is permutation-invariant by
construction under the sum/mean pooling.

Pooling: masked SUM and masked MEAN concatenated -- the sum carries the extensive
"how much light" signal (a nhit/energy proxy), the mean the intensive spatial/temporal
shape; a single pool conflates them. A count/charge channel (log nhit + the 2-vector
charge features) is appended before the head, giving the flow the total-charge
information the standalone ChargeNet used.

``context_dim`` is a constructor argument and is LOGGED by callers: the summary width is
a monitored sufficiency risk (too narrow a bottleneck throws away hit information the
posterior needs).
"""

import equinox as eqx
import jax
import jax.numpy as jnp

from hitman.nn import features as ft


def _hit_feature_map(hits):
    """(P, 4) absolute (x, y, z, t) -> (P, n_feat) normalized per-hit features."""
    xyz = hits[:, :3] / ft.POSITION_SCALE
    t = hits[:, 3:4] / ft.TIME_SCALE
    rho = jnp.hypot(hits[:, 0], hits[:, 1])[:, None] / ft.POSITION_SCALE
    return jnp.concatenate([xyz, t, rho], axis=-1)   # (P, 5)


N_HIT_IN = 5


class DeepSetsEncoder(eqx.Module):
    """Per-hit MLP -> masked sum+mean pool (+ count/charge channel) -> head -> c(x)."""

    phi: eqx.nn.MLP
    rho: eqx.nn.MLP
    context_dim: int = eqx.field(static=True)
    phi_dim: int = eqx.field(static=True)

    def __init__(self, *, key, context_dim=128, phi_dim=128, phi_hidden=128,
                 phi_depth=2, rho_hidden=128, rho_depth=2):
        kphi, krho = jax.random.split(key)
        self.context_dim = context_dim
        self.phi_dim = phi_dim
        self.phi = eqx.nn.MLP(N_HIT_IN, phi_dim, phi_hidden, phi_depth, key=kphi,
                              activation=jax.nn.gelu)
        # head input: [sum, mean] (2*phi_dim) + [log nhit, charge/40-1 (2)] (3)
        self.rho = eqx.nn.MLP(2 * phi_dim + 3, context_dim, rho_hidden, rho_depth,
                              key=krho, activation=jax.nn.gelu)

    def __call__(self, hits, mask, charge):
        """(hits (P,4), mask (P,), charge (2,)) -> context (context_dim,)."""
        feats = jax.vmap(self.phi)(_hit_feature_map(hits))     # (P, phi_dim)
        m = mask[:, None]
        summed = jnp.sum(feats * m, axis=0)
        cnt = jnp.sum(mask)
        mean = summed / jnp.maximum(cnt, 1.0)
        aux = jnp.concatenate([
            jnp.log(cnt + 1.0)[None],
            charge / ft.CHARGE_SCALE - 1.0,
        ])
        return self.rho(jnp.concatenate([summed, mean, aux]))
