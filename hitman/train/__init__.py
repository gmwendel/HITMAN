"""Generic ratio-estimation training pieces.

``nre_loss`` (BCE-on-logits + optional BNRE balance + observation weights) and the
host-path ``fit`` are detector-agnostic. The device-resident water-Cherenkov loops
(``fit_resident``, ``train_recipe``, ``fit_event_model``, ``exact_polish``), the
moment/identity receipts, NWJ and the hit/charge/brightness reweighting tables moved to
:mod:`hitman.wc.train`.

For NEW work prefer :mod:`hitman.ratio`, which is the group-aware ratio lane: injectable
negative pairing, group-level splits, and the self-describing model bundle.
"""

from hitman._compat import deprecated_getattr
from hitman.train.losses import classifier_accuracy, nre_loss
from hitman.train.loop import FitResult, fit

_WC_TRAIN = {
    "fit_resident": "hitman.wc.train.resident", "DeviceData": "hitman.wc.train.resident",
    "hit_batch": "hitman.wc.train.resident", "charge_batch": "hitman.wc.train.resident",
    "frame_hit_batch": "hitman.wc.train.resident",
    "exact_polish": "hitman.wc.train.polish",
    "train_recipe": "hitman.wc.train.recipe", "RecipeResult": "hitman.wc.train.recipe",
    "fit_event_model": "hitman.wc.train.event", "event_val_bce": "hitman.wc.train.event",
    "EventFitResult": "hitman.wc.train.event",
    "make_event_val_bce": "hitman.wc.train.event",
    "build_hit_weights": "hitman.wc.train.reweight",
    "make_weighted_hit_batch": "hitman.wc.train.reweight",
    "HitWeightTable": "hitman.wc.train.reweight",
    "build_charge_weights": "hitman.wc.train.reweight",
    "make_weighted_charge_batch": "hitman.wc.train.reweight",
    "ChargeWeightTable": "hitman.wc.train.reweight",
    "build_brightness_weights": "hitman.wc.train.reweight",
    "make_brightness_weighted_hit_batch": "hitman.wc.train.reweight",
    "BrightnessWeightTable": "hitman.wc.train.reweight",
}
__getattr__ = deprecated_getattr(__name__, _WC_TRAIN)

__all__ = ["nre_loss", "classifier_accuracy", "fit", "FitResult", *sorted(_WC_TRAIN)]
