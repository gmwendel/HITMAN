"""Equal-occupancy strata assignment -- the detector-agnostic primitive under ``z_rings``.

Kept in the library (rather than moving with the water-Cherenkov harness) because it is
pure numpy over an arbitrary 1-D coordinate and has a downstream consumer: the muon
program strata on core distance r and on energy. ``hitman.wc.receipts.harness.z_rings``
is one instantiation of it, over PMT z.
"""

import numpy as np


def percentile_strata(values, n: int):
    """Assign each entry of a 1-D ``values`` array to one of ``n`` equal-occupancy bins.

    Returns ``(labels, n_labels)``; ``n_labels`` may be < ``n`` if ties collapse edges. The
    detector-agnostic strata primitive: WC z-rings, muon core-distance annuli, energy bins
    all reduce to this over the appropriate coordinate.
    """
    values = np.asarray(values)
    edges = np.unique(np.percentile(values, np.linspace(0, 100, n + 1)))
    m = len(edges) - 1
    labels = np.clip(np.digitize(values, edges) - 1, 0, m - 1)
    return labels, m
