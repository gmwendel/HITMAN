"""HitNet: per-hit log likelihood-ratio estimator.

Besides the plain forward pass, HitNet exposes an exactly-equivalent *separable* path:
every hit feature is linear in obs-only or hyp-only quantities (the mixed dt column
splits as t_obs/25 − t_hyp/25), so the first layer's pre-activation decomposes as

    z = embed_hit(hit) + embed_hyp(hyp)

During reconstruction the hits are fixed: embed all hits once, then each candidate
hypothesis (optimizer step, MCMC proposal) costs only a (width,)-sized hypothesis
embedding plus the remaining layers — the per-hit feature trafo and the hit side of
the first GEMM are never recomputed. This replaces 1.x's tiling of the hypothesis
across hits.
"""

import equinox as eqx
import jax.numpy as jnp

from hitman.nn import features as ft
from hitman.nn.mlp import MLP


class HitNet(eqx.Module):
    mlp: MLP

    def __init__(self, width: int = 256, depth: int = 3, *, key,
                 activation: str = "mish"):
        self.mlp = MLP(ft.N_HIT_FEATURES, width, depth, key=key, activation=activation)

    def __call__(self, hit: jnp.ndarray, hyp: jnp.ndarray) -> jnp.ndarray:
        """(hit (4,), hyp (7,)) -> scalar logit = log r_hit."""
        return self.mlp(ft.hit_features(hit, hyp))

    # -- separable first layer ---------------------------------------------------

    def embed_hit(self, hit: jnp.ndarray) -> jnp.ndarray:
        """Obs-only part of the first-layer pre-activation. Shape (width,)."""
        w = self.mlp.layers[0].weight
        return w[:, 8:11] @ (hit[:3] / ft.POSITION_SCALE) + w[:, ft.HIT_FEAT_DT] * (
            hit[3] / ft.TIME_SCALE
        )

    def embed_hyp(self, hyp: jnp.ndarray) -> jnp.ndarray:
        """Hyp-only part of the first-layer pre-activation (carries the bias). Shape (width,)."""
        w = self.mlp.layers[0].weight
        return (
            w[:, 0:3] @ (hyp[:3] / ft.POSITION_SCALE)
            + w[:, 3:6] @ ft.direction(hyp)
            - w[:, ft.HIT_FEAT_DT] * (hyp[ft.TIME] / ft.TIME_SCALE)
            + w[:, 7] * (hyp[ft.ENERGY] - 1.0)
            + self.mlp.layers[0].bias
        )

    def logit_from_embedding(self, pre_activation: jnp.ndarray) -> jnp.ndarray:
        """Scalar logit from embed_hit(hit) + embed_hyp(hyp)."""
        return self.mlp.head(pre_activation)
