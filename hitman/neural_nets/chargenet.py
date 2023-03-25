import tensorflow as tf
import tensorflow_addons as tfa

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

        #assert (hyp_norm is not None) ^ (
        #            obs_norm is not None), 'Error: Specify normalization for BOTH hypothesis and observation'

        self.hyp_norm = hyp_norm
        self.obs_norm = obs_norm

        self.zenith_idx = 0
        self.azimuth_idx = 1
        self.energy_idx = 2

    def get_config(self):
        config = super().get_config()
        config.update({
            "hyp_norm": self.hyp_norm,
            "obs_norm": self.obs_norm,
        })
        return config

    def call(self, charge, params):
        '''
        Parameters:
        -----------

        charge : tensor
            shape (N, 1), containing the event total hits

        params : tensor
            shape (N, 3) containing zenith, azimuth, and energy

        '''

        dir_x = tf.math.sin(params[:, self.zenith_idx]) * tf.math.cos(params[:, self.azimuth_idx])
        dir_y = tf.math.sin(params[:, self.zenith_idx]) * tf.math.sin(params[:, self.azimuth_idx])
        dir_z = tf.math.cos(params[:, self.zenith_idx])

        energy = params[:, self.energy_idx] - 1

        out = tf.stack([
            charge[:, 0] / 40 - 1,
            dir_x,
            dir_y,
            dir_z,
            energy
        ],
            axis=1
        )

        return out


def get_chargenet(activation=tfa.activations.mish, layers=3):
    charge_input = tf.keras.Input(shape=(1,))
    params_input = tf.keras.Input(shape=(3,))

    t = chargenet_trafo()
    h = t(charge_input, params_input)
    for i in range(layers):
        h = tf.keras.layers.Dense(256, activation=activation)(h)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(h)

    chargenet = tf.keras.Model(inputs=[charge_input, params_input], outputs=outputs)

    return chargenet
