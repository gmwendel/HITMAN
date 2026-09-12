"""An analytic-truth hierarchical toy: the one place composite machinery can be PROVEN.

Every other gate in this package measures a trained model against data whose true ratio
nobody knows, so a gate that comes back clean means "no contradiction found", not
"correct". This module supplies the missing case -- a generative model whose per-row log
ratio, composite posterior, design effect and hierarchical global-parameter posterior are
ALL closed form -- so the machinery can be checked against exact numbers instead of
against itself.

The structure deliberately mirrors the two-station timing problem:

    x[g,i] = delta + theta[g] + b[g] + eps[g,i]

* ``delta``    -- ONE global parameter shared by every group        (the clock offset)
* ``theta[g]`` -- a per-group nuisance, drawn from N(0, tau^2)      (per-shower E, h1)
* ``b[g]``     -- a group-common latent that rows CANNOT resolve    (the shower-common mode)
* ``eps[g,i]`` -- per-row noise                                     (per-muon fluctuation)

``b`` is the whole point. With ``rho = 0`` the rows of a group are genuinely iid given
(delta, theta[g]), the composite log-likelihood really is the sum of per-row log ratios,
and any failure of the summation machinery is a BUG. With ``rho > 0`` the rows are
exchangeable but correlated, the sum is provably wrong, and it is wrong by an amount this
class can state exactly. That separation is what makes the toy useful: it tells you
whether a coverage failure on real muons is a code defect or the physics of a shower.

Parameterization note: total per-row variance about ``delta + theta[g]`` is held at ``v``
for every ``rho``, split as ``Var(b) = rho*v`` and ``Var(eps) = (1-rho)*v``. Holding the
row marginal fixed is what makes ``rho`` the ordinary intraclass correlation and puts the
design effect in its textbook form ``1 + (n-1)*rho``; if ``v`` moved with ``rho`` the
sweeps below would confound "more correlation" with "noisier rows".
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HierarchicalGaussian:
    """Gaussian hierarchy with closed-form ratio, composite posterior and design effect.

    Parameters
    ----------
    tau : float
        SD of the per-group nuisance ``theta[g]``. Also the prior width the per-row ratio
        is referenced to -- the learned ratio's denominator is the training marginal, so
        the proposal for ``theta`` and the prior in the analytic formulae must agree.
    v : float
        Total per-row variance about ``delta + theta[g]``, held fixed across ``rho``.
    rho : float
        Intraclass correlation in [0, 1). ``rho = 0`` is the iid case where the composite
        sum is exact.
    """

    tau: float = 1.0
    v: float = 1.0
    rho: float = 0.0

    def __post_init__(self):
        if not (0.0 <= self.rho < 1.0):
            raise ValueError(f"rho must be in [0, 1), got {self.rho}")
        if self.tau <= 0 or self.v <= 0:
            raise ValueError("tau and v must be positive")

    # -- variance bookkeeping --------------------------------------------------
    @property
    def var_b(self) -> float:
        """Variance of the group-common latent."""
        return self.rho * self.v

    @property
    def var_eps(self) -> float:
        """Variance of the per-row noise."""
        return (1.0 - self.rho) * self.v

    # -- generative model ------------------------------------------------------
    def sample(self, rng, n_groups: int, rows_per_group: int, delta: float = 0.0):
        """Draw a dataset. Returns ``(x, theta_per_row, group_id, theta_g)``.

        ``x`` is (N, 1) and ``theta_per_row`` is (N, 1) broadcast from the per-group draw,
        which is the row layout :class:`hitman.ratio.RatioDataset` expects.
        """
        rng = np.random.default_rng(rng) if not isinstance(rng, np.random.Generator) else rng
        theta_g = rng.normal(0.0, self.tau, size=n_groups)
        b_g = rng.normal(0.0, np.sqrt(self.var_b), size=n_groups) if self.rho > 0 else \
            np.zeros(n_groups)
        gid = np.repeat(np.arange(n_groups), rows_per_group).astype(np.int64)
        mu = delta + np.repeat(theta_g + b_g, rows_per_group)
        x = mu + rng.normal(0.0, np.sqrt(self.var_eps), size=gid.size)
        return (x[:, None], np.repeat(theta_g, rows_per_group)[:, None], gid, theta_g)

    # -- closed form: the PER-ROW ratio the network is trained to learn ---------
    def log_ratio(self, x, theta):
        """Analytic ``log r(x, theta) = log p(x | theta) - log p(x)`` for single rows.

        This is the target of the classifier, and it is a per-row quantity: the network
        never sees a group. Note it is derived at ``rho = 0`` in the sense that the row
        conditional is ``N(theta, v)`` either way -- ``b`` is invisible to a single row,
        which is exactly WHY a per-row classifier cannot learn about it and why the naive
        composite is overconfident. The correlation is not a defect the network could fix
        with more capacity; it is information absent from its input.
        """
        x = np.asarray(x, dtype=np.float64).ravel()
        theta = np.asarray(theta, dtype=np.float64).ravel()
        v_marg = self.tau**2 + self.v          # marginal Var(x) with theta integrated out
        return (-(x - theta) ** 2 / (2 * self.v) + x**2 / (2 * v_marg)
                - 0.5 * np.log(self.v / v_marg))

    # -- closed form: the composite posterior for ONE group's nuisance ---------
    def posterior_sd_theta(self, n: int) -> float:
        """Exact posterior SD of ``theta[g]`` from ``n`` rows of one group (delta known).

        The sufficient statistic is the group mean, whose variance about ``theta[g]`` is
        ``Var(b) + Var(eps)/n``. With the N(0, tau^2) prior the posterior precision is
        ``1/tau^2 + 1/(var_b + var_eps/n)``.
        """
        v_bar = self.var_b + self.var_eps / n
        return float(1.0 / np.sqrt(1.0 / self.tau**2 + 1.0 / v_bar))

    def naive_posterior_sd_theta(self, n: int) -> float:
        """What SUMMING ``n`` per-row log ratios claims -- correct only at ``rho = 0``.

        Summing asserts ``Var(x_bar) = v / n``; the truth is ``var_b + var_eps/n``. The two
        agree iff ``rho = 0`` or ``n = 1``.
        """
        return float(1.0 / np.sqrt(1.0 / self.tau**2 + n / self.v))

    def design_effect(self, n: int) -> float:
        """``1 + (n-1)*rho`` -- the variance inflation the naive sum fails to charge.

        Equivalently ``n_eff = n / design_effect``. Stated here in closed form so a design
        effect MEASURED on real data can be validated against a case where it is known.
        """
        return float(1.0 + (n - 1) * self.rho)

    def n_eff(self, n: int) -> float:
        return float(n / self.design_effect(n))

    # -- closed form: the HIERARCHICAL posterior for the global parameter ------
    def posterior_sd_delta(self, n_groups: int, rows_per_group: int) -> float:
        """Exact posterior SD of the GLOBAL ``delta`` from ``n_groups`` groups, flat prior.

        Marginalizing both ``theta[g]`` and ``b[g]``, each group's mean satisfies
        ``x_bar[g] | delta ~ N(delta, tau^2 + var_b + var_eps/n)``, and the groups are
        independent. So the global posterior SD is that per-group width over ``sqrt(G)``.

        This is the number the whole programme rests on: it says stacking correlated
        showers still buys ``1/sqrt(G)`` in the GLOBAL parameter, because the correlation
        is WITHIN a group and the groups are independent. What correlation costs is the
        per-group width, and it costs it once -- not once per row.
        """
        per_group = self.tau**2 + self.var_b + self.var_eps / rows_per_group
        return float(np.sqrt(per_group / n_groups))

    def naive_posterior_sd_delta(self, n_groups: int, rows_per_group: int) -> float:
        """What the naive per-row sum claims for ``delta``. Overconfident when ``rho > 0``."""
        per_group = self.tau**2 + self.v / rows_per_group
        return float(np.sqrt(per_group / n_groups))

    def delta_overconfidence(self, rows_per_group: int) -> float:
        """Ratio ``true_sd / naive_sd`` for ``delta`` -- independent of ``n_groups``.

        Group count cancels: stacking cannot repair a per-group miscalibration, it only
        shrinks both numbers together. That is the formal version of "bias beats variance".
        """
        n = rows_per_group
        true_pg = self.tau**2 + self.var_b + self.var_eps / n
        naive_pg = self.tau**2 + self.v / n
        return float(np.sqrt(true_pg / naive_pg))
