"""Water-Cherenkov application package -- the original HITMAN detector, intact.

Everything here is specific to reconstructing one optical-neutrino event from PMT hits:
the 7-parameter hypothesis ``(x, y, z, zen, az, t, E)``, the discrete sensor grid, the
RAT-DS ingest, the deployment inference lanes (batched / sequential / compiled), the
splinemle closed-form density, and the forward receipts built on per-PMT charge and
z-ring time.

It was split out of the top-level package so that ``import hitman`` stops implying a PMT
detector. The library proper (``hitman.spec``, ``hitman.data.structures``,
``hitman.density``, ``hitman.diagnostics``, ``hitman.npe``, ``hitman.nn.mlp``,
``hitman.ratio``, ``hitman.validate``) is detector-agnostic and must never import from
here; this package may freely import from it.

Old import paths (``hitman.inference``, ``hitman.splinemle``, ``hitman.nn.hitnet``, ...)
still work through deprecation shims -- see ``hitman/_compat.py``.

Nothing in this package changed numerically in the move: it is a relocation plus import
rewrites, and the full WC test suite is the receipt for that.
"""
