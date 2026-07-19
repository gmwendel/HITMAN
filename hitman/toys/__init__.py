"""Analytic-ratio toy problems for L1 statistical-closure tests.

A toy fixes a generative model whose true likelihood-to-evidence ratio ``r(x, theta)``
is known in closed form, so a trained NRE can be checked against exact ground truth:
ratio recovery, the score identity (exactly zero in expectation), temperature
recalibration, SBC/coverage on the exact posterior, and MLE bias/pull closure. This is
the layer where a receipt's *own* correctness is validated before it is trusted on the
real detector surrogate.
"""

from hitman.toys.gaussian_source import (
    GaussianSource,
    nre_logit,
    train_nre,
)

__all__ = ["GaussianSource", "nre_logit", "train_nre"]
