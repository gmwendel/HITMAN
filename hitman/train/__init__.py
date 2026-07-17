from hitman.train.losses import nre_loss
from hitman.train.loop import fit
from hitman.train.resident import DeviceData, charge_batch, fit_resident, hit_batch

__all__ = ["nre_loss", "fit", "fit_resident", "DeviceData", "hit_batch", "charge_batch"]
