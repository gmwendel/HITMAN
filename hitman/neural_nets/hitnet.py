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
            shape (2,7); (1,) contains std dev of training dataset. (2,) contains avg of training dataset
            if no argument is specified, no normalization will be performed. Note hyp and zen norm will be ignored
        obs_norm : np array
            shape (2,5); (1,) contains std dev of training dataset. (2,) contains avg of training dataset
            if no argument is specified, no normalization will be performed
        '''

        super().__init__()

        assert (hyp_norm is not None) ^ (obs_norm is not None), 'Error: Specify normalization for BOTH hypothesis and observation'

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
        return {'hyp_labels': self.labels, 'max_energy': self.min_energy, 'max_energy': self.max_energy}

    def call(self, hit, params):
        '''
         Parameters:
        -----------

        hit : tensor
            shape (N, 5), containing hit Photosensor position (x, y, z) time, and charge

        params : tensor
            shape (N, 7) containing particle hypothesis position (x,y,z), zenith, azimuth, time, and energy

        '''

        cosphi = tf.math.cos(params[:, self.azimuth_idx])
        sinphi = tf.math.sin(params[:, self.azimuth_idx])
        sintheta = tf.math.sin(params[:, self.zenith_idx])
        dir_x = sintheta * cosphi
        dir_y = sintheta * sinphi
        dir_z = tf.math.cos(params[:, self.zenith_idx])



        dt = hit[:, 3] - params[:, self.time_idx]

        energy = params[:, self.energy_idx] - 1
        event_time = (params[:, self.time_idx]) / 25
        pmt_time = (hit[:, 3]) / 25

        out = tf.stack([
            params[:, self.x_idx] / 1000,
            params[:, self.y_idx] / 1000,
            params[:, self.z_idx] / 1000,
            dir_x,
            dir_y,
            dir_z,
            pmt_time - event_time,
            energy,
            hit[:, 0] / 1000,
            hit[:, 1] / 1000,
            hit[:, 2] / 1000,
        ],
            axis=1
        )

        return out


def get_hitnet(activation=tfa.activations.mish, layers=3):
    hit_input = tf.keras.Input(shape=(7,))
    params_input = tf.keras.Input(shape=(5,))

    t = hitnet_trafo()
    h = t(hit_input, params_input)
    for i in range(layers):
        h = tf.keras.layers.Dense(256, activation=activation)(h)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(h)

    hitnet = tf.keras.Model(inputs=[hit_input, params_input], outputs=outputs)

    return hitnet
