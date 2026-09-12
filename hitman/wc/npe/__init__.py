"""Water-Cherenkov NPE batching and training loop.

``make_npe_batch`` gathers ``data.pmt_pos[data.pmt_id[rows]]`` and shifts ``theta[:, 5]``
for the coherent time augmentation, and ``build_npe_spec`` reads a ``HitStore``'s
``hit_offsets`` -- all three are the water-Cherenkov layout, so this is application code.
The detector-agnostic half of the NPE lane (encoder, flow, loss, SBC) stayed in
:mod:`hitman.npe`.
"""

from hitman.wc.npe.batch import build_npe_spec, make_fixed_batch, make_npe_batch
from hitman.wc.npe.train import NpeFitResult, fit_npe

__all__ = ["build_npe_spec", "make_npe_batch", "make_fixed_batch", "fit_npe",
           "NpeFitResult"]
