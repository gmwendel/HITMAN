"""Forward-KL (maximum-likelihood) NPE objective: -E_{(x,theta)} log q(theta | x).

The batch is event-grouped (hits, mask, charge, theta); the encoder produces one context
per event and the flow scores the true theta under q(.|context). Minimizing this batch
NLL over joint samples (theta ~ prior, x ~ simulator) has the true posterior as its
unique minimizer (forward KL).
"""

import jax
import jax.numpy as jnp


def encode(encoder, hits, mask, charge):
    """Vectorized context: (N,P,4),(N,P),(N,2) -> (N, context_dim)."""
    return jax.vmap(encoder)(hits, mask, charge)


def flow_nll(flow, context, theta):
    """-mean log q(theta | context) for precomputed contexts (used by the toy/SBC paths)."""
    return -jnp.mean(jax.vmap(flow.log_prob)(theta, context))


def npe_loss(model, batch):
    """Batch NLL for the (encoder, flow) pair over an event-grouped batch."""
    encoder, flow = model
    hits, mask, charge, theta = batch
    context = encode(encoder, hits, mask, charge)
    return -jnp.mean(jax.vmap(flow.log_prob)(theta, context))
