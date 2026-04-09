import tensorflow as tf
import numpy as np

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, x, t, batch_size=2 ** 12, shuffle='free', time_spread=50):
        assert shuffle in ['free', 'inDOM'], "Choose either 'free' or 'inDOM' shuffling."

        self.batch_size = int(batch_size / 2)  # half true labels half false labels
        self.data = np.array(x)
        self.params = np.array(t)

        self._labels = np.concatenate([
            np.ones((self.batch_size, 1), dtype=self.data.dtype),
            np.zeros((self.batch_size, 1), dtype=self.data.dtype)
        ])

        # spread absolute time values (for hitnet)
        if len(self.data[0]) > 4 and len(self.params[0]) > 5:
            time_shifts = np.random.normal(0, time_spread, len(self.data))
            self.data[:, 3] += time_shifts
            self.params[:, 5] += time_shifts

        if shuffle == 'inDOM':
            self.shuffle_params_inDOM()
        else:
            self.shuffled_params = []

        self.indexes = np.arange(len(self.data))
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def get_norms(self):

        return hyp_norm, obs_norm

    def shuffle_params_inDOM(self):
        pmt_coords = self.data[:, 0:3]
        unique_coords, inverse_indices = np.unique(pmt_coords, axis=0, return_inverse=True)
        
        sort_idx = np.argsort(inverse_indices)
        sorted_params = self.params[sort_idx]
        sorted_inv = inverse_indices[sort_idx]
        
        _, counts = np.unique(sorted_inv, return_counts=True)
        split_idx = np.cumsum(counts)[:-1]
        
        grouped_params = np.split(sorted_params, split_idx)
        shuffled_grouped_params = [np.random.permutation(g) for g in grouped_params]
        shuffled_sorted_params = np.concatenate(shuffled_grouped_params)
        
        unsort_idx = np.empty_like(sort_idx)
        unsort_idx[sort_idx] = np.arange(len(sort_idx))
        
        self.shuffled_params = shuffled_sorted_params[unsort_idx]

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        np.random.shuffle(self.indexes)  # mix between batches
        if len(self.shuffled_params) > 0:
            self.shuffle_params_inDOM()

    def __data_generation(self, indexes_temp):
        'Generates data containing batch_size samples'
        x = np.take(self.data, indexes_temp, axis=0)
        t = np.take(self.params, indexes_temp, axis=0)
        if len(self.shuffled_params) == 0:
            tr = np.random.permutation(t)
        else:
            tr = np.take(self.shuffled_params, indexes_temp, axis=0)

        d_X = np.concatenate([x, x], axis=0)
        d_T = np.concatenate([t, tr], axis=0)
        
        # Dynamically generate labels to match current batch size
        cur_batch = len(x)
        d_labels = np.concatenate([
            np.ones((cur_batch, 1), dtype=d_X.dtype),
            np.zeros((cur_batch, 1), dtype=d_X.dtype)
        ])

        d_X, d_T, d_labels = self.unison_shuffled_copies(d_X, d_T, d_labels)

        return (d_X, d_T), d_labels

    def unison_shuffled_copies(self, a, b, c):
        'Shuffles arrays in the same way'
        assert len(a) == len(b) == len(c)
        p = np.random.permutation(len(a))

        return a[p], b[p], c[p]
