import numpy as np

from hitman.receipts.score import score_identity
from hitman.toys import GaussianSource


def test_score_identity_holds_for_exact_ratio():
    # Draw x ~ N(theta, s^2) and use the analytic score (x - theta)/s^2, whose mean
    # over x|theta is exactly zero. The receipt must not flag it.
    src = GaussianSource(dim=3, s=1.0, tau=2.0)
    rng = np.random.default_rng(0)
    theta = np.array([0.3, -0.5, 1.0])
    xs = theta + src.s * rng.normal(size=(60_000, 3))

    grad_fn = lambda chunk: (chunk - theta) / src.s**2
    res = score_identity(grad_fn, xs, chunk=5000, names=["a", "b", "c"])
    assert res.max_sigma < 4.0
    assert [c.name for c in res.components] == ["a", "b", "c"]


def test_score_identity_flags_a_biased_gradient():
    src = GaussianSource(dim=2, s=1.0, tau=2.0)
    rng = np.random.default_rng(1)
    theta = np.zeros(2)
    xs = theta + rng.normal(size=(40_000, 2))
    # inject a constant offset -> non-zero mean score -> huge sigma
    grad_fn = lambda chunk: (chunk - theta) / src.s**2 + np.array([0.2, 0.0])
    res = score_identity(grad_fn, xs, chunk=4000)
    assert res.components[0].sigma > 10.0
    assert res.components[1].sigma < 4.0
