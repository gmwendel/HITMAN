import tensorflow as tf
import numpy as np
import os

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, x, t, batch_size=2 ** 12, shuffle='free', time_spread=50, cache_file=None):
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
            if cache_file and os.path.exists(cache_file):
                print(f"Loading inDOM structures from {cache_file}...")
                cached = np.load(cache_file)
                self.sort_idx = cached['sort_idx']
                self.split_idx = cached['split_idx']
                self.unsort_idx = cached['unsort_idx']
                
                sorted_params = self.params[self.sort_idx]
                self.grouped_params = np.split(sorted_params, self.split_idx)
            else:
                pmt_coords = self.data[:, 0:3]
                unique_coords, inverse_indices = np.unique(pmt_coords, axis=0, return_inverse=True)
                
                self.sort_idx = np.argsort(inverse_indices)
                sorted_params = self.params[self.sort_idx]
                sorted_inv = inverse_indices[self.sort_idx]
                
                _, counts = np.unique(sorted_inv, return_counts=True)
                self.split_idx = np.cumsum(counts)[:-1]
                
                self.grouped_params = np.split(sorted_params, self.split_idx)
                
                self.unsort_idx = np.empty_like(self.sort_idx)
                self.unsort_idx[self.sort_idx] = np.arange(len(self.sort_idx))
                
                if cache_file:
                    print(f"Saving inDOM structures to {cache_file}...")
                    np.savez(cache_file, sort_idx=self.sort_idx, split_idx=self.split_idx, unsort_idx=self.unsort_idx)
            
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
        shuffled_grouped_params = [np.random.permutation(g) for g in self.grouped_params]
        shuffled_sorted_params = np.concatenate(shuffled_grouped_params)
        
        self.shuffled_params = shuffled_sorted_params[self.unsort_idx]

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
