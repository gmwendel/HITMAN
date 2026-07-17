"""Plain MLP building block with a linear scalar head.

The head is linear from birth: the network output *is* the classifier logit, i.e. the
log likelihood-to-evidence ratio. (1.x trained with a sigmoid head and re-saved the
model with a linear activation afterwards; training on logits with a
sigmoid-cross-entropy loss is equivalent and removes that step.)
"""

import equinox as eqx
import jax
import jax.numpy as jnp


def mish(x):
    """Mish activation (arXiv:1908.08681), the 1.x default; smooth for gradient inference."""
    return x * jnp.tanh(jax.nn.softplus(x))


class MLP(eqx.Module):
    layers: tuple

    def __init__(self, in_size: int, width: int = 256, depth: int = 3, *, key):
        keys = jax.random.split(key, depth + 1)
        sizes = (in_size, *([width] * depth), 1)
        self.layers = tuple(
            eqx.nn.Linear(sizes[i], sizes[i + 1], key=keys[i]) for i in range(depth + 1)
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Features (in_size,) -> scalar logit."""
        for layer in self.layers[:-1]:
            x = mish(layer(x))
        return self.layers[-1](x)[0]

    def head(self, pre_activation: jnp.ndarray) -> jnp.ndarray:
        """Forward pass given the first layer's pre-activation (see HitNet.embed_*)."""
        x = mish(pre_activation)
        for layer in self.layers[1:-1]:
            x = mish(layer(x))
        return self.layers[-1](x)[0]
