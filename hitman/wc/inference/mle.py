"""Maximum-likelihood reconstruction as a single jitted JAX program.

Multistart strategy (port of 1.x reco, made principled):
seed broadly -> keep the best ``n_select`` by NLL -> run a fixed-length Adam descent on
every survivor in parallel (``vmap`` over a ``lax.scan``) -> return the best minimum.

1.x used a hand-tuned per-parameter learning-rate vector ("wbls best"); here the
hypothesis is optimized in a scaled space u = theta / PARAM_SCALE (matching the feature
normalizations), so one scalar learning rate serves all parameters and detectors.

Everything below is jit-compatible with static shapes — the whole reconstruction is one
compiled graph, exportable via jax2tf for the C++ (cppflow) deployment path.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax

from hitman.wc.nn.features import wrap_direction

# x, y, z [mm], zenith, azimuth [rad], t [ns], E [MeV] — matches feature normalization.
PARAM_SCALE = jnp.array([1000.0, 1000.0, 1000.0, 1.0, 1.0, 25.0, 1.0])


# Optimizer-side direction chart: (zen, az) is hostile to every optimizer — the pole
# (zen=0) makes azimuth degenerate and the polar gradient singular, and az wraps.
# Optimize an unconstrained R^3 direction instead (normalized on evaluation); convert
# to angles only at readout. Chart space: (x, y, z, dx, dy, dz, t, E), 8 components.
CHART_SCALE = jnp.array([1000.0, 1000.0, 1000.0, 1.0, 1.0, 1.0, 25.0, 1.0])


def theta_from_chart(u: jnp.ndarray) -> jnp.ndarray:
    """(8,) chart point -> (7,) hypothesis. Smooth away from the exact pole (clipped)."""
    d = u[3:6] / (jnp.linalg.norm(u[3:6]) + 1e-9)
    zen = jnp.arccos(jnp.clip(d[2], -1.0 + 1e-6, 1.0 - 1e-6))
    az = jnp.mod(jnp.arctan2(d[1], d[0]), 2.0 * jnp.pi)
    return jnp.stack([u[0], u[1], u[2], zen, az, u[6], u[7]])


def chart_from_theta(theta: jnp.ndarray) -> jnp.ndarray:
    """(7,) hypothesis -> (8,) chart point (unit direction vector)."""
    sin_zen = jnp.sin(theta[3])
    return jnp.stack([
        theta[0], theta[1], theta[2],
        sin_zen * jnp.cos(theta[4]), sin_zen * jnp.sin(theta[4]), jnp.cos(theta[3]),
        theta[5], theta[6],
    ])


class MLEResult(NamedTuple):
    theta: jnp.ndarray  # (7,) best-fit hypothesis, direction wrapped to physical range
    nll: jnp.ndarray  # scalar NLL at the minimum
    thetas: jnp.ndarray  # (n_select, 7) all local minima (diagnostics)
    nlls: jnp.ndarray  # (n_select,)


def cylinder_seeds(key, n: int, radius: float, half_height: float, t_range, e_range):
    """Uniform hypothesis seeds in a cylindrical detector (port of 1.x uniform_sample).

    Positions uniform over 90% of the cylinder volume, directions isotropic, t and E
    uniform over the given (min, max) ranges.
    """
    keys = jax.random.split(key, 6)
    u = jax.random.uniform(keys[0], (n,))
    phi = jax.random.uniform(keys[1], (n,), maxval=2 * jnp.pi)
    r = 0.9 * radius * jnp.sqrt(u)
    z = jax.random.uniform(keys[2], (n,), minval=-0.9 * half_height, maxval=0.9 * half_height)
    zenith = jnp.arccos(jax.random.uniform(keys[3], (n,), minval=-1.0, maxval=1.0))
    azimuth = jax.random.uniform(keys[4], (n,), maxval=2 * jnp.pi)
    t, e = (
        jax.random.uniform(keys[5], (2, n))
        * jnp.array([[t_range[1] - t_range[0]], [e_range[1] - e_range[0]]])
        + jnp.array([[t_range[0]], [e_range[0]]])
    )
    return jnp.stack([r * jnp.cos(phi), r * jnp.sin(phi), z, zenith, azimuth, t, e], axis=1)


def multistart_mle(
    nll,
    seeds: jnp.ndarray,
    n_select: int = 64,
    steps: int = 300,
    learning_rate: float = 3e-2,
    scale: jnp.ndarray = PARAM_SCALE,
    bounds=None,
) -> MLEResult:
    """Minimize ``nll(theta)`` from many seeds; jit/vmap/export friendly.

    ``bounds=(lo, hi)`` (each (7,)) enables projected descent: iterates are clipped to
    the box after every step. Use it — the surrogate is extrapolation outside the
    training support (the detector), and an unbounded optimizer will happily walk
    into that region and find spurious minima there.
    """
    if bounds is not None:
        lo, hi = bounds[0] / scale, bounds[1] / scale
    seed_nlls = jax.vmap(nll)(seeds)
    _, top = jax.lax.top_k(-seed_nlls, n_select)
    optimizer = optax.adam(learning_rate)

    def descend(u0):
        def step(carry, _):
            u, opt_state = carry
            value, grad = jax.value_and_grad(lambda uu: nll(uu * scale))(u)
            updates, opt_state = optimizer.update(grad, opt_state)
            u = optax.apply_updates(u, updates)
            if bounds is not None:
                u = jnp.clip(u, lo, hi)
            return (u, opt_state), value

        (u, _), _ = jax.lax.scan(step, (u0, optimizer.init(u0)), None, length=steps)
        return u * scale, nll(u * scale)

    thetas, nlls = jax.vmap(descend)(seeds[top] / scale)
    best = jnp.argmin(nlls)
    return MLEResult(
        theta=wrap_direction(thetas[best]), nll=nlls[best], thetas=thetas, nlls=nlls
    )
