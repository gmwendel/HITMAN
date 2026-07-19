"""Score-identity receipt: E_{x ~ p(x|theta)}[grad_theta log r_hat(x, theta)] = 0.

The exact ratio satisfies this at every theta; it is the FIRST-ORDER UNBIASEDNESS
condition of the downstream MLE (which solves grad log r_hat = 0). BCE constrains
logit *values*; inference consumes *gradients* — so this is the receipt that speaks
the inference layer's language (design report, Addendum 4).

Generic over the gradient callable: pass an already-vmapped ``grad_fn`` that maps a
chunk of observations to per-observation gradients ``(m, dim)`` of ``log r_hat`` w.r.t.
theta at the fixed true hypothesis. Works identically for an analytic toy ratio and a
trained surrogate; the caller owns model loading and JAX.
"""

from typing import Callable, NamedTuple, Optional, Sequence

import numpy as np


class ScoreComponent(NamedTuple):
    name: str
    mean: float
    sem: float
    sigma: float  # |mean| / sem — deviation from zero in standard errors


class ScoreIdentity(NamedTuple):
    components: Sequence[ScoreComponent]
    max_sigma: float
    n: int


def score_identity(
    grad_fn: Callable[[np.ndarray], np.ndarray],
    xs,
    chunk: int = 2000,
    names: Optional[Sequence[str]] = None,
) -> ScoreIdentity:
    """Mean per-component score over an iid ensemble at fixed truth, with error bars.

    Parameters
    ----------
    grad_fn : callable
        ``grad_fn(xs_chunk) -> (m, dim)`` array of gradients of ``log r_hat`` w.r.t.
        theta, evaluated at the fixed true hypothesis. Typically
        ``jax.jit(jax.vmap(lambda x: jax.grad(logr)(theta_true, x)))``.
    xs : (N, ...)
        Observations drawn from ``p(x | theta_true)`` (e.g. MC events at the test point).
    chunk : int
        Batch size for evaluating ``grad_fn`` (memory control).
    names : sequence of str, optional
        Component labels (default ``c0..c{dim-1}``).

    Returns
    -------
    ScoreIdentity
        Per-component mean, standard error of the mean, and ``sigma = |mean| / sem``;
        plus the worst component's sigma. A calibrated inference-grade ratio has all
        components within a few sigma of zero.
    """
    n = len(xs)
    n = n - (n % chunk) if n >= chunk else n
    if n == 0:
        raise ValueError("need at least one observation for the score identity")
    grads = np.concatenate(
        [np.asarray(grad_fn(xs[i : i + chunk])) for i in range(0, n, chunk)]
    )
    mean = grads.mean(axis=0)
    sem = grads.std(axis=0) / np.sqrt(len(grads))
    dim = grads.shape[1]
    if names is None:
        names = [f"c{i}" for i in range(dim)]
    comps = [
        ScoreComponent(
            name=names[i],
            mean=float(mean[i]),
            sem=float(sem[i]),
            sigma=float(abs(mean[i]) / max(sem[i], 1e-12)),
        )
        for i in range(dim)
    ]
    return ScoreIdentity(
        components=comps, max_sigma=max(c.sigma for c in comps), n=len(grads)
    )
