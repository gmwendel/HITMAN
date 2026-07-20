"""Exact maximum-likelihood loss for the run20 spline model.

Per event: the marked-Poisson log-likelihood
    sum_hits [ log p_hat(s_i|theta) + log p_hat(t_i|s_i,theta) ] + log Pois(N|Lambda).
The batch loss is -mean over hits of the per-hit term plus a per-event-mean count term
(its own mean, per the run20 spec). No Monte Carlo, no grid in the data dimension: every
factor is normalized in closed form inside the forward pass.
"""

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from hitman.splinemle.model import SUPPORT_FLOOR, ell_at


def event_terms(model, pmt_ids, t, mask, theta, floor=SUPPORT_FLOOR):
    """One padded event -> (hit_ll, N, count_ll).

    ``pmt_ids``/``t``/``mask`` are (P,) padded to a common slot count; masked slots
    contribute 0 to the hit sum (the finite ``floor`` keeps their -inf-density gradients
    clean before the mask multiply). N = sum(mask); the count factor is the marked-Poisson
    Poisson(Lambda), Lambda = exp(logsumexp eta).
    """
    eta, nodes, t_geo, logZt = model.event_tables(theta)
    log_Lambda = logsumexp(eta)
    knots = model.knots_arr

    def per_hit(s, tt):
        u = tt - t_geo[s]
        inside = (u >= knots[0]) & (u <= knots[-1])
        lt = jnp.where(inside, ell_at(u, nodes[s], knots) - logZt[s], floor)
        ls = eta[s] - log_Lambda
        return ls + lt

    ll = jax.vmap(per_hit)(pmt_ids, t)
    hit_ll = jnp.sum(ll * mask)
    N = jnp.sum(mask)
    count_ll = model.log_count(N, log_Lambda)
    return hit_ll, N, count_ll


def batch_terms(model, batch, floor=SUPPORT_FLOOR, n_chunk: int = 1):
    """(hit_ll (B,), N (B,), count_ll (B,)) for a batch of padded events.

    ``n_chunk`` > 1 evaluates the event vmap in sequential ``jax.checkpoint``-ed chunks via
    ``lax.map`` so the reverse pass stores one chunk's activations, not the whole batch's
    (the lax.map/scan backward otherwise STACKS every chunk's forward). B must be divisible
    by ``n_chunk``; any remainder events are dropped (document if you set it).
    """
    pmt_ids, t, mask, theta = batch
    f = lambda s, tt, m, th: event_terms(model, s, tt, m, th, floor)  # noqa: E731
    if n_chunk <= 1:
        return jax.vmap(f)(pmt_ids, t, mask, theta)
    B = theta.shape[0]
    nc = min(n_chunk, B)
    ch = B // nc
    n = nc * ch
    body = jax.checkpoint(lambda args: jax.vmap(f)(*args))
    out = jax.lax.map(body, (
        pmt_ids[:n].reshape(nc, ch, -1),
        t[:n].reshape(nc, ch, -1),
        mask[:n].reshape(nc, ch, -1),
        theta[:n].reshape(nc, ch, theta.shape[1]),
    ))
    return tuple(o.reshape(-1) for o in out)


def splinemle_loss(model, batch, w_count: float = 1.0, floor=SUPPORT_FLOOR, n_chunk: int = 1):
    """Exact-MLE loss (scalar, aux dict). -mean_hits[log p_sensor + log p_time]
    - w_count * mean_events[log Pois(N|Lambda)]."""
    hit_ll, N, count_ll = batch_terms(model, batch, floor, n_chunk)
    total_hits = jnp.sum(N)
    loss_hit = -jnp.sum(hit_ll) / jnp.maximum(total_hits, 1.0)
    loss_count = -jnp.mean(count_ll)
    loss = loss_hit + w_count * loss_count
    aux = {"loss_hit": loss_hit, "loss_count": loss_count,
           "mean_hits": total_hits / N.shape[0]}
    return loss, aux
