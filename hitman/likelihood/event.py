"""Event-level log likelihood-ratio composition (all-sensor EML formulation).

log r(obs | theta) = sum_i log r_hit(hit_i, theta) + log r_charge(N_tot, theta)

Per-event sums over the flat hit array use ``segment_sum`` on ``event_id`` — no
per-hit hypothesis copies, no ragged reshaping.
"""

import jax
import jax.numpy as jnp

from hitman.data.structures import EventBatch


def event_log_ratio(hitnet, chargenet, batch: EventBatch, hyp: jnp.ndarray) -> jnp.ndarray:
    """Per-event log likelihood-ratio for a batch, one hypothesis row per event.

    Parameters
    ----------
    hyp : (n_events, 7)
        Hypothesis for each event (truth, candidate, or shuffled pairing).

    Returns
    -------
    (n_events,) log r for each event.
    """
    hit_logits = jax.vmap(hitnet)(batch.hits, hyp[batch.event_id])
    hit_term = jax.ops.segment_sum(hit_logits, batch.event_id, num_segments=hyp.shape[0])
    charge_term = jax.vmap(chargenet)(batch.charge, hyp)
    return hit_term + charge_term


def make_event_nll(hitnet, chargenet, hits: jnp.ndarray, charge: jnp.ndarray):
    """Build the negative log likelihood-ratio for ONE event as a function of theta.

    The hits are embedded through HitNet's separable first layer once, here; each call
    of the returned ``nll(theta)`` then costs one hypothesis embedding plus the MLP
    heads. This is the hot path for optimizers and samplers.
    """
    hit_emb = jax.vmap(hitnet.embed_hit)(hits)  # (n_hits, width), computed once

    def nll(theta: jnp.ndarray) -> jnp.ndarray:
        pre = hit_emb + hitnet.embed_hyp(theta)
        hit_term = jnp.sum(jax.vmap(hitnet.logit_from_embedding)(pre))
        return -(hit_term + chargenet(charge, theta))

    return nll
