"""Training-pair construction for neural ratio estimation.

The joint class pairs each observation with its own event hypothesis; the marginal
class pairs it with a hypothesis drawn from another event (a permuted index — never a
copied parameter row). The free (all-sensor) shuffle corresponds to the paper's
all-sensor EML formulation; the per-sensor ('inDOM') variant is not yet ported.
"""

import jax
import jax.numpy as jnp


def marginal_permutation(key, n: int) -> jnp.ndarray:
    """Random permutation of event indices used to build marginal (label-0) pairs."""
    return jax.random.permutation(key, n)


def time_shuffle(key, batch, sigma: float):
    """Shift each event's time origin by N(0, sigma) ns, coherently in hits and hypothesis.

    Augmentation from 1.x training: spreads absolute times so the networks learn the
    physical dependence on t_hit − t_event. Applied to both the hit times and the
    hypothesis time, so hitnet features (which depend on dt only) are unchanged, but
    the marginal-class pairings see varied absolute times.
    """
    shifts = sigma * jax.random.normal(key, (batch.n_events,))
    hits = batch.hits.at[:, 3].add(shifts[batch.event_id])
    hyp = batch.hyp.at[:, 5].add(shifts)
    return batch._replace(hits=hits, hyp=hyp)
