import numpy as np
import tensorflow as tf

class PoissonDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, charges, charge_hyp, pmt_positions, batch_size=2**15):
        self.batch_size = batch_size
        self.N_events, self.N_sensors = charges.shape
        self.total_samples = self.N_events * self.N_sensors
        
        # Flatten and pre-calculate pairs
        # charges: (N_events, N_sensors) -> (N_events * N_sensors,)
        self.flat_charges = charges.flatten()
        
        # charge_hyp: (N_events, 3) -> repeat each event N_sensors times -> (N_events * N_sensors, 3)
        self.flat_hyp = np.repeat(charge_hyp, self.N_sensors, axis=0)
        
        # pmt_positions: (N_sensors, 3) -> tile N_events times -> (N_events * N_sensors, 3)
        self.flat_pmt = np.tile(pmt_positions, (self.N_events, 1))
        
        self.indexes = np.arange(self.total_samples)
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(self.total_samples / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        
        batch_pmt = np.take(self.flat_pmt, indexes, axis=0)
        batch_hyp = np.take(self.flat_hyp, indexes, axis=0)
        batch_charges = np.take(self.flat_charges, indexes, axis=0)
        
        return (batch_pmt, batch_hyp), batch_charges

    def on_epoch_end(self):
        np.random.shuffle(self.indexes)