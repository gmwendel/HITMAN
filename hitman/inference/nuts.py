"""Bayesian posterior sampling with NUTS (blackjax) on the learned log-ratio.

The learned log likelihood-to-evidence ratio plus a log-prior is a valid unnormalized
log-posterior; the surrogate is differentiable by construction, so gradient-based
sampling comes for free (the HMC outlook of arXiv:2208.10166, realized).
"""

from typing import NamedTuple

import blackjax
import jax
import jax.numpy as jnp


class NUTSResult(NamedTuple):
    samples: jnp.ndarray  # (num_samples, dim)
    log_density: jnp.ndarray  # (num_samples,)
    acceptance_rate: jnp.ndarray  # (num_samples,)
    is_divergent: jnp.ndarray  # (num_samples,) bool — should be ~all False
    energy: jnp.ndarray = None  # (num_samples,) Hamiltonian energy per draw


def e_bfmi(energy: jnp.ndarray) -> jnp.ndarray:
    """Energy Bayesian fraction of missing information (Betancourt 2016).

    E-BFMI = mean(diff(E)^2) / var(E); values < 0.3 flag momentum-resampling
    inefficiency — for a neural surrogate logdensity this is the smoothness receipt
    (rough/high-curvature ratio surfaces show up here before divergences do).
    """
    d = jnp.diff(energy)
    return jnp.mean(d**2) / jnp.clip(jnp.var(energy), 1e-30)


def sample_nuts(
    logdensity,
    initial_position: jnp.ndarray,
    key,
    num_warmup: int = 1000,
    num_samples: int = 2000,
    target_acceptance_rate: float = 0.8,
) -> NUTSResult:
    """Window-adapted NUTS chain. Seed ``initial_position`` at the MLE for fast warmup."""
    warmup_key, sample_key = jax.random.split(key)
    adaptation = blackjax.window_adaptation(
        blackjax.nuts, logdensity, target_acceptance_rate=target_acceptance_rate
    )
    (state, parameters), _ = adaptation.run(warmup_key, initial_position, num_steps=num_warmup)
    kernel = blackjax.nuts(logdensity, **parameters)

    def step(state, key):
        state, info = kernel.step(key, state)
        return state, (state.position, state.logdensity, info.acceptance_rate,
                       info.is_divergent, info.energy)

    _, (positions, logdens, accept, divergent, energy) = jax.lax.scan(
        step, state, jax.random.split(sample_key, num_samples)
    )
    return NUTSResult(
        samples=positions,
        log_density=logdens,
        acceptance_rate=accept,
        is_divergent=divergent,
        energy=energy,
    )


def box_log_prior(lo: jnp.ndarray, hi: jnp.ndarray, stiffness: float = 1e2):
    """Smooth box prior: flat inside [lo, hi], quadratic penalty outside.

    HMC needs gradients, so a hard uniform prior (−inf outside) is unusable; this
    keeps the chain inside the detector volume — where the surrogate is trained —
    while remaining differentiable everywhere.
    """

    def log_prior(theta):
        below = jax.nn.relu(lo - theta)
        above = jax.nn.relu(theta - hi)
        return -stiffness * jnp.sum(below**2 + above**2)

    return log_prior


def make_event_logdensity(nll, log_prior=None):
    """Unnormalized log-posterior for one event from its NLL (see make_event_nll)."""

    def logdensity(theta):
        lp = -nll(theta)
        if log_prior is not None:
            lp = lp + log_prior(theta)
        return lp

    return logdensity
