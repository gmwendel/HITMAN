"""Event-grouped padded batching for NPE, device-side gathers only.

Reuses the event-grouped batching contract validated for the score-identity penalty
(``hitman.train.identities`` -- offsets table, pad ``n_pad`` with a float mask, drop
oversize events at spec-build time). NPE needs exactly (hits, mask, charge, theta), and
the coherent time augmentation MUST shift the LABEL theta[5] together with the hit times
(a posterior model regresses theta_t; a mismatched shift teaches the wrong t posterior).

``build_npe_spec`` is re-exported from ``identities.build_identity_spec`` (identical
eligibility semantics). The E-correlated drop rate at pad 320 on the 5M store is <0.6%
in 9-10 MeV -- log it if it matters for the run.
"""

import jax
import jax.numpy as jnp

from hitman.train.identities import build_identity_spec as build_npe_spec  # noqa: F401


def make_npe_batch(data, spec, key, n_events, n_pad, time_sigma=50.0):
    """Sample an event-grouped padded batch: (hits (N,P,4), mask (N,P), charge (N,2),
    theta (N,7)). ``n_events``/``n_pad`` are static (shapes); all gathers on device.

    Coherent per-event time augmentation: each event's hit times AND its label theta[5]
    shift by the same N(0, time_sigma) draw.
    """
    k_ev, k_aug = jax.random.split(key)
    ev = spec.elig[jax.random.randint(k_ev, (n_events,), 0, spec.elig.shape[0])]
    start = spec.offsets[ev]
    nh = spec.offsets[ev + 1] - start
    slot = jnp.arange(n_pad, dtype=jnp.int32)
    rows = jnp.minimum(start[:, None] + slot[None, :], data.n_hits - 1)
    mask = (slot[None, :] < nh[:, None]).astype(jnp.float32)
    t = data.t[rows]
    theta = data.hyp[ev]
    if time_sigma > 0:
        sh = time_sigma * jax.random.normal(k_aug, (n_events,))
        t = t + sh[:, None]
        theta = theta.at[:, 5].add(sh)
    pos = data.pmt_pos[data.pmt_id[rows]]
    hits = jnp.concatenate([pos, t[..., None]], axis=-1) * mask[..., None]
    return hits, mask, data.charge[ev], theta


def make_fixed_batch(data, spec, key, n_events, n_pad, time_sigma=0.0):
    """A single held-out batch (default no augmentation) for the validation yardstick."""
    return make_npe_batch(data, spec, key, n_events, n_pad, time_sigma)
