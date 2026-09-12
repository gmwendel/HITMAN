"""Plain MLP building block with a linear scalar head.

The head is linear from birth: the network output *is* the classifier logit, i.e. the
log likelihood-to-evidence ratio. (1.x trained with a sigmoid head and re-saved the
model with a linear activation afterwards; training on logits with a
sigmoid-cross-entropy loss is equivalent and removes that step.)

Activation is configurable (round-5 distilled-student study). ``mish`` is the trained
1.x default and the shipped teacher's activation; the cheaper alternatives
(``swish``/``softplus``/``relu``/``hardswish``) exist for the width-32 *search* student
whose per-element activation cost is a larger fraction of its (small) GEMM. The
activation is a STATIC field (never a trained leaf), so ``tree_deserialise_leaves``
into a default-``mish`` template is unaffected for the existing teacher checkpoints.
"""

import equinox as eqx
import jax
import jax.numpy as jnp


def mish(x):
    """Mish activation (arXiv:1908.08681), the 1.x default; smooth for gradient inference."""
    return x * jnp.tanh(jax.nn.softplus(x))


# Named activation table. All lower cleanly to StableHLO (verified in
# scripts/export_student_teacher_stablehlo.py). mish/swish/softplus are C-infinity smooth
# (safe for the final likelihood's curvature) -- as are gelu and swish (SiLU); relu is C0
# and hardswish only C1 (kinks at +/-3), adequate for the SEARCH surface (the teacher
# endgame supplies the smooth final descent) but NOT if the student surface itself must be
# curvature-clean. The round-5 receipts decide per activation; if smoothness is weighted,
# swish/gelu are the C-infinity cheap options (prefer them over hardswish).
ACTIVATIONS = {
    "mish": mish,
    "swish": jax.nn.silu,          # x * sigmoid(x); C-infinity
    "gelu": jax.nn.gelu,           # x * Phi(x) (tanh approx); C-infinity, smoother than hardswish
    "softplus": jax.nn.softplus,
    # (x + sqrt(x^2 + 4))/2: C-infinity, ALGEBRAIC (one sqrt, no
    # transcendentals) -- measured 12.5% faster than mish in the fused
    # inference kernel; monotone ReLU-like shape (no negative dip)
    "squareplus": jax.nn.squareplus,
    "relu": jax.nn.relu,
    "hardswish": jax.nn.hard_swish,
}


def get_activation(name: str):
    """Resolve an activation name to its callable (raises on an unknown name)."""
    try:
        return ACTIVATIONS[name]
    except KeyError as exc:  # pragma: no cover - guard
        raise ValueError(
            f"unknown activation {name!r}; choose from {list(ACTIVATIONS)}") from exc


class MLP(eqx.Module):
    layers: tuple
    act: str = eqx.field(static=True, default="mish")

    def __init__(self, in_size: int, width: int = 256, depth: int = 3, *, key,
                 activation: str = "mish"):
        keys = jax.random.split(key, depth + 1)
        sizes = (in_size, *([width] * depth), 1)
        self.layers = tuple(
            eqx.nn.Linear(sizes[i], sizes[i + 1], key=keys[i]) for i in range(depth + 1)
        )
        get_activation(activation)  # validate early
        self.act = activation

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Features (in_size,) -> scalar logit."""
        f = get_activation(self.act)
        for layer in self.layers[:-1]:
            x = f(layer(x))
        return self.layers[-1](x)[0]

    def head(self, pre_activation: jnp.ndarray) -> jnp.ndarray:
        """Forward pass given the first layer's pre-activation (see HitNet.embed_*)."""
        f = get_activation(self.act)
        x = f(pre_activation)
        for layer in self.layers[1:-1]:
            x = f(layer(x))
        return self.layers[-1](x)[0]
