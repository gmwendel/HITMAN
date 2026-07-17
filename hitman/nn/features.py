"""Physics feature transforms — pure-function port of the 1.x Keras trafo layers.

Feature definitions and normalizations are kept bit-for-bit identical to 1.x
(`hitnet_trafo`, `chargenet_trafo`) so that old and new models are comparable.
"""

import jax.numpy as jnp

# Hypothesis vector layout (7,): x, y, z [mm], zenith, azimuth [rad], t [ns], E [MeV]
X, Y, Z, ZENITH, AZIMUTH, TIME, ENERGY = range(7)

POSITION_SCALE = 1000.0  # mm
TIME_SCALE = 25.0  # ns
CHARGE_SCALE = 40.0

N_HIT_FEATURES = 11
N_CHARGE_FEATURES = 9

# Column split of the hit-feature vector (used by HitNet's separable first layer):
# hyp-only columns [0:6] + [7], obs-only columns [8:11], and the mixed dt column [6],
# which itself separates linearly: dt/25 = t_obs/25 − t_hyp/25.
HIT_FEAT_DT = 6


def direction(hyp: jnp.ndarray) -> jnp.ndarray:
    """Unit direction vector from (zenith, azimuth)."""
    sin_zen = jnp.sin(hyp[ZENITH])
    return jnp.stack(
        [
            sin_zen * jnp.cos(hyp[AZIMUTH]),
            sin_zen * jnp.sin(hyp[AZIMUTH]),
            jnp.cos(hyp[ZENITH]),
        ]
    )


def hit_features(hit: jnp.ndarray, hyp: jnp.ndarray) -> jnp.ndarray:
    """(hit (4,), hyp (7,)) -> (11,) hitnet input features.

    hit = sensor x, y, z [mm], time [ns].
    """
    return jnp.concatenate(
        [
            hyp[:3] / POSITION_SCALE,
            direction(hyp),
            (hit[3] - hyp[TIME])[None] / TIME_SCALE,
            (hyp[ENERGY] - 1.0)[None],
            hit[:3] / POSITION_SCALE,
        ]
    )


def charge_features(charge: jnp.ndarray, hyp: jnp.ndarray) -> jnp.ndarray:
    """(charge (2,), hyp (7,)) -> (9,) chargenet input features.

    charge = total charge, number of hits.
    """
    return jnp.concatenate(
        [
            charge / CHARGE_SCALE - 1.0,
            hyp[:3] / POSITION_SCALE,
            direction(hyp),
            (hyp[ENERGY] - 1.0)[None],
        ]
    )


def wrap_direction(hyp: jnp.ndarray) -> jnp.ndarray:
    """Map (zenith, azimuth) back to [0, pi] x [0, 2pi) after unconstrained optimization.

    Port of 1.x ``proper_dir``: round-trips through the unit vector, which is exact and
    handles zenith excursions outside [0, pi] (where naive modulo would flip direction).
    """
    d = direction(hyp)
    azimuth = jnp.mod(jnp.arctan2(d[1], d[0]), 2 * jnp.pi)
    zenith = jnp.arccos(jnp.clip(d[2], -1.0, 1.0))
    return hyp.at[ZENITH].set(zenith).at[AZIMUTH].set(azimuth)
