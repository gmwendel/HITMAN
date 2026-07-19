"""Gaussian light-source toy: an analytic-ratio problem (cf. the paper's Gull example).

Generative model (isotropic, dimension ``d``)::

    theta ~ N(0, tau^2 I)          # prior p(theta)
    x | theta ~ N(theta, s^2 I)    # likelihood
    => marginal  p(x) = N(0, (tau^2 + s^2) I)      # evidence

Everything the NRE and its receipts consume is then closed-form:

* **ratio**      log r(x, theta) = log p(x|theta) - log p(x)
                 = -||x-theta||^2 / 2s^2 + ||x||^2 / 2(tau^2+s^2)
                   - (d/2) log( s^2 / (tau^2+s^2) )
* **score**      grad_theta log r = (x - theta) / s^2,
                 and E_{x|theta}[(x-theta)/s^2] = 0  exactly  (score identity)
* **MLE**        argmax_theta log r = x, Fisher = 1/s^2 per dim
                 => bias 0, resolution s, pull ~ N(0,1)
* **posterior**  p(theta|x) = N( x * tau^2/(tau^2+s^2),  (1/s^2 + 1/tau^2)^-1 I )
                 => SBC ranks uniform / expected coverage on the diagonal

The NRE is trained on the same ``hitman`` building blocks (MLP + ``nre_loss``) so the
toy exercises the production loss/optimizer, not a bespoke one.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from hitman.nn.mlp import MLP
from hitman.train.losses import nre_loss


@dataclass(frozen=True)
class GaussianSource:
    """Isotropic Gaussian location toy. ``s`` = likelihood std, ``tau`` = prior std."""

    dim: int = 1
    s: float = 1.0
    tau: float = 2.0

    # -- generative model ------------------------------------------------------

    def sample_theta(self, key, n: int) -> jnp.ndarray:
        return self.tau * jax.random.normal(key, (n, self.dim))

    def sample_x_given_theta(self, key, theta: jnp.ndarray) -> jnp.ndarray:
        return theta + self.s * jax.random.normal(key, theta.shape)

    def sample_joint(self, key, n: int):
        kt, kx = jax.random.split(key)
        theta = self.sample_theta(kt, n)
        x = self.sample_x_given_theta(kx, theta)
        return x, theta

    # -- closed-form quantities ------------------------------------------------

    def log_ratio(self, x: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
        """Analytic log r(x, theta) for a single (x, theta) pair."""
        v_marg = self.tau**2 + self.s**2
        quad = -jnp.sum((x - theta) ** 2) / (2 * self.s**2) + jnp.sum(x**2) / (2 * v_marg)
        norm = -0.5 * self.dim * jnp.log(self.s**2 / v_marg)
        return quad + norm

    def score(self, x: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
        """grad_theta log r = (x - theta) / s^2."""
        return (x - theta) / self.s**2

    def mle(self, x: jnp.ndarray) -> jnp.ndarray:
        return x

    @property
    def fisher_sigma(self) -> float:
        """Per-component MLE sigma = s (Fisher information 1/s^2)."""
        return self.s

    def posterior(self, x: jnp.ndarray):
        """Exact posterior mean and (isotropic) std for observation ``x``."""
        v_marg = self.tau**2 + self.s**2
        mean = x * self.tau**2 / v_marg
        std = float(jnp.sqrt(1.0 / (1.0 / self.s**2 + 1.0 / self.tau**2)))
        return mean, std

    def sample_posterior(self, key, x: jnp.ndarray, n_draws: int) -> jnp.ndarray:
        mean, std = self.posterior(x)
        return mean[None, :] + std * jax.random.normal(key, (n_draws, self.dim))


def nre_logit(model: MLP, x: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
    """Learned logit = surrogate log r(x, theta). Input features are [x, theta]."""
    return model(jnp.concatenate([x, theta]))


def train_nre(
    src: GaussianSource,
    key,
    n_steps: int = 1500,
    batch: int = 4096,
    width: int = 64,
    depth: int = 2,
    lr: float = 3e-3,
) -> MLP:
    """Train a small NRE on the toy with the production ``nre_loss`` (fresh minibatches).

    Fresh joint/marginal draws every step (the toy is free to sample), so there is no
    finite-sample separability and the BCE optimum is the true log-ratio.
    """
    import optax

    kmodel, ktrain = jax.random.split(key)
    model = MLP(2 * src.dim, width=width, depth=depth, key=kmodel)
    opt = optax.adam(lr)
    state = opt.init(model)

    logit_vec = jax.vmap(nre_logit, in_axes=(None, 0, 0))

    def loss_fn(m, x, th_joint, th_marg):
        return nre_loss(logit_vec(m, x, th_joint), logit_vec(m, x, th_marg))

    @jax.jit
    def step(m, state, k):
        kj, kperm = jax.random.split(k)
        x, theta = src.sample_joint(kj, batch)
        theta_marg = theta[jax.random.permutation(kperm, batch)]
        loss, grad = jax.value_and_grad(loss_fn)(m, x, theta, theta_marg)
        updates, state = opt.update(grad, state)
        return optax.apply_updates(m, updates), state, loss

    keys = jax.random.split(ktrain, n_steps)
    for k in keys:
        model, state, _ = step(model, state, k)
    return model
