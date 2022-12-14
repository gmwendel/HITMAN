import tensorflow as tf
import tensorflow_addons as tfa
from hitman.neural_nets.transformations import chargenet_trafo


class chargenet_trafo(tf.keras.layers.Layer):
    '''Class to transfor inputs for Charget Net
    '''

    def __init__(self, hyp_norm=None, obs_norm=None):
        '''
        Parameters:
        -----------

        labels : list
            list of labels corresponding to the data array
        '''

        super().__init__()

        assert (hyp_norm is not None) ^ (
                    obs_norm is not None), 'Error: Specify normalization for BOTH hypothesis and observation'

        self.hyp_norm = hyp_norm
        self.obs_norm = obs_norm

        self.x_idx = 0
        self.y_idx = 1
        self.z_idx = 2
        self.zenith_idx = 3
        self.azimuth_idx = 4
        self.time_idx = 5
        self.energy_idx = 6

    def get_config(self):
        return {'labels': self.labels, 'max_energy': self.min_energy, 'max_energy': self.max_energy}

    def call(self, charge, params):
        '''
        Parameters:
        -----------

        charge : tensor
            shape (N, 2), containing the event total charge and number of hit DOMs

        params : tensor
            shape (N, len(labels))

        '''

        dir_x = tf.math.sin(params[:, self.zenith_idx]) * tf.math.cos(params[:, self.azimuth_idx])
        dir_y = tf.math.sin(params[:, self.zenith_idx]) * tf.math.sin(params[:, self.azimuth_idx])
        dir_z = tf.math.cos(params[:, self.zenith_idx])

        energy = params[:, self.energy_idx] - 1

        out = tf.stack([
            charge[:, 0] / 40 - 1,
            charge[:, 1] / 40 - 1,  # n_channels
            params[:, self.x_idx] / 1000,
            params[:, self.y_idx] / 1000,
            params[:, self.z_idx] / 1000,
            dir_x,
            dir_y,
            dir_z,
            energy
        ],
            axis=1
        )

        return out


def get_chargenet(activation=tfa.activations.mish, layers=3):
    charge_input = tf.keras.Input(shape=(2,))
    params_input = tf.keras.Input(shape=(7,))

    t = chargenet_trafo()
    h = t(charge_input, params_input)
    for i in range(layers):
        h = tf.keras.layers.Dense(256, activation=activation)(h)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(h)

    chargenet = tf.keras.Model(inputs=[hit_input, params_input], outputs=outputs)

    return chargenet
