import tensorflow as tf
import tensorflow_addons as tfa


class hitnet_trafo(tf.keras.layers.Layer):
    '''Class to transform inputs for Hitnet
    This layer performs two main operations. The first, a transformation from spherical to cartesian coordinates for the
    directional component of the hypothesis.  The second, an optional normalization of all parameters.
    '''

    def __init__(self, hyp_norm=None, obs_norm=None):
        '''
        Parameters:
        -----------
        hyp_norm : np array
            shape (2,3); (1,) contains std dev of training dataset. (2,) contains avg of training dataset
            if no argument is specified, no normalization will be performed. Note hyp and zen norm will be ignored
        obs_norm : np array
            shape (2,2); (1,) contains std dev of training dataset. (2,) contains avg of training dataset
            if no argument is specified, no normalization will be performed
        '''

        super().__init__()

        # assert (hyp_norm is not None) ^ (obs_norm is not None), 'Error: Specify normalization for BOTH hypothesis and observation'

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
        config = super().get_config()
        config.update({
            "hyp_norm": self.hyp_norm,
            "obs_norm": self.obs_norm,
        })
        return config

    def call(self, obs, hyp):
        '''
         Parameters:
        -----------

        hit : tensor
            shape (N, 2), containing hit Photosensor ID and time

        params : tensor
            shape (N, 7) containing particle hypothesis position (x,y,z), zenith, azimuth, time, and energy

        '''

        cosphi = tf.math.cos(hyp[:, self.azimuth_idx])
        sinphi = tf.math.sin(hyp[:, self.azimuth_idx])
        sintheta = tf.math.sin(hyp[:, self.zenith_idx])
        dir_x = sintheta * cosphi
        dir_y = sintheta * sinphi
        dir_z = tf.math.cos(hyp[:, self.zenith_idx])

        energy = hyp[:, self.energy_idx] - 1
        x = hyp[:, self.x_idx]/8
        y = hyp[:, self.y_idx]/8
        z = hyp[:, self.z_idx]/8

        out = tf.stack([
            x,
            y,
            z,
            dir_x,
            dir_y,
            dir_z,
            energy,
            obs[:, 0],
            obs[:, 1] - hyp[:, self.time_idx]
        ],
            axis=1
        )

        return out


def get_hitnet(activation=tfa.activations.mish, layers=3):
    hit_input = tf.keras.Input(shape=(2,))
    params_input = tf.keras.Input(shape=(7,))

    t = hitnet_trafo()
    h = t(hit_input, params_input)
    for i in range(layers):
        h = tf.keras.layers.Dense(256, activation=activation)(h)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(h)

    hitnet = tf.keras.Model(inputs=[hit_input, params_input], outputs=outputs)

    return hitnet
