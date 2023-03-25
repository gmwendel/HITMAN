import tensorflow as tf
import numpy as np

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, x, t, batch_size=2 ** 12, shuffle='free'):
        assert shuffle in ['free', 'inDOM'], "Choose either 'free' or 'inDOM' shuffling."

        self.batch_size = int(batch_size / 2)  # half true labels half false labels
        self.data = np.array(x)
        self.params = np.array(t)

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

        d_true_labels = np.ones((self.batch_size, 1), dtype=x.dtype)
        d_false_labels = np.zeros((self.batch_size, 1), dtype=x.dtype)

        d_X = np.append(x, x, axis=0)
        d_T = np.append(t, tr, axis=0)
        d_labels = np.append(d_true_labels, d_false_labels)

        d_X, d_T, d_labels = self.unison_shuffled_copies(d_X, d_T, d_labels)

        return (d_X, d_T), d_labels

    def unison_shuffled_copies(self, a, b, c):
        'Shuffles arrays in the same way'
        assert len(a) == len(b) == len(c)
        p = np.random.permutation(len(a))

        return a[p], b[p], c[p]
