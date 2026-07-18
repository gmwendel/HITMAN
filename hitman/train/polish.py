"""Exact polish: deterministic full-dataset L-BFGS on a frozen NRE objective.

Minibatch SGD (even with lr decay) converges to a noise ball around the minimizer.
This module finishes the job: the empirical NRE risk is made DETERMINISTIC by freezing
the stochastic ingredients — a fixed row partition into chunks, and per chunk a small
set of fixed augmentation/pairing keys (averaged) — and its exact value and gradient
over the full dataset are computed as a lax.scan over chunks with rematerialization,
so device memory is bounded by one chunk's activations regardless of dataset size.
With exact gradients on a deterministic objective, optax L-BFGS (with line search)
polishes to the actual minimizer of the frozen objective.

Guard rails: validation BCE is tracked every iteration and the best-on-val model is
returned — fully minimizing empirical BCE with an expressive model risks trust-crisis
overconfidence (design report, Addendum 4), though at ~2e5 params vs ~1e8 rows the
margin is comfortable.
"""

import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from hitman.train.loop import _batch_loss


def make_exact_loss(data, make_batch, rows_all, key, chunk_size: int = 2**19,
                    n_pairings: int = 2, balance_weight: float = 0.0):
    """Deterministic full-data NRE loss over a fixed row set.

    rows_all is truncated to a multiple of chunk_size (the dropped tail is < one
    chunk out of ~200 — negligible and keeps shapes static). Every chunk uses
    n_pairings fixed (augmentation, marginal-permutation) key pairs, averaged.
    """
    n_chunks = len(rows_all) // chunk_size
    chunks = jnp.asarray(rows_all[: n_chunks * chunk_size].reshape(n_chunks, chunk_size))

    def chunk_loss(model, rows, ci):
        def one_pairing(j):
            k = jax.random.fold_in(jax.random.fold_in(key, ci), j)
            k_aug, k_perm = jax.random.split(k)
            obs, hyp = make_batch(data, rows, k_aug)
            return _batch_loss(model, obs, hyp, k_perm, balance_weight)

        return jnp.mean(jnp.stack([one_pairing(j) for j in range(n_pairings)]))

    chunk_loss = jax.remat(chunk_loss, static_argnums=())

    def exact_loss(model):
        def body(acc, ci):
            return acc + chunk_loss(model, chunks[ci], ci), None

        total, _ = jax.lax.scan(body, jnp.asarray(0.0, jnp.float32),
                                jnp.arange(n_chunks))
        return total / n_chunks

    return exact_loss


def exact_polish(
    model,
    data,
    make_batch,
    n_rows: int,
    *,
    key,
    max_iters: int = 60,
    chunk_size: int = 2**19,
    n_pairings: int = 2,
    val_fraction: float = 0.1,
    max_val_rows: int = 2**19,
    balance_weight: float = 0.0,
    memory_size: int = 10,
    verbose: bool = True,
):
    """L-BFGS on the exact frozen empirical risk; returns (best-on-val model, history)."""
    n_val = max(int(n_rows * val_fraction), 1)
    train_rows = np.arange(0, n_rows - n_val, dtype=np.int32)
    val_idx = np.arange(n_rows - n_val, n_rows, dtype=np.int32)
    if n_val > max_val_rows:
        val_idx = val_idx[:: n_val // max_val_rows + 1][:max_val_rows]

    key, loss_key, val_key = jax.random.split(key, 3)
    exact_loss_dyn = make_exact_loss(data, make_batch, train_rows, loss_key,
                                     chunk_size, n_pairings, balance_weight)

    # static/dynamic split so L-BFGS state lives over arrays only
    params, static = eqx.partition(model, eqx.is_inexact_array)

    def loss_p(p):
        return exact_loss_dyn(eqx.combine(p, static))

    @eqx.filter_jit
    def val_bce(p, rows):
        m = eqx.combine(p, static)
        k_aug, k_perm = jax.random.split(val_key)
        obs, hyp = make_batch(data, jnp.asarray(rows), k_aug)
        return _batch_loss(m, obs, hyp, k_perm, balance_weight)

    opt = optax.lbfgs(memory_size=memory_size)
    state = opt.init(params)
    value_and_grad = optax.value_and_grad_from_state(loss_p)

    @jax.jit
    def step(p, s):
        value, grad = value_and_grad(p, state=s)
        updates, s = opt.update(grad, s, p, value=value, grad=grad, value_fn=loss_p)
        return optax.apply_updates(p, updates), s, value

    best = (np.inf, model)
    hist = []
    for it in range(max_iters):
        t0 = time.time()
        params, state, value = step(params, state)
        v = float(val_bce(params, val_idx))
        hist.append((float(value), v))
        if v < best[0]:
            best = (v, eqx.combine(params, static))
        if verbose:
            print(f"lbfgs {it:3d}  exact {float(value):.6f}  val {v:.6f}  "
                  f"({time.time() - t0:.1f}s)", flush=True)

    return best[1], hist
