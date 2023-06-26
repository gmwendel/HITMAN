import tensorflow as tf


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
            shape (N, 5), containing hit Photosensor position (x, y, z) time, and charge

        params : tensor
            shape (N, 7) containing particle hypothesis position (x,y,z), zenith, azimuth, time, and energy

        '''

        cosphi = tf.math.cos(hyp[:, self.azimuth_idx])
        sinphi = tf.math.sin(hyp[:, self.azimuth_idx])
        sintheta = tf.math.sin(hyp[:, self.zenith_idx])
        dir_x = sintheta * cosphi
        dir_y = sintheta * sinphi
        dir_z = tf.math.cos(hyp[:, self.zenith_idx])

        dt = obs[:, 3] - hyp[:, self.time_idx]

        energy = hyp[:, self.energy_idx] - 1
        event_time = (hyp[:, self.time_idx]) / 25
        pmt_time = (obs[:, 3]) / 25

        out = tf.stack([
            hyp[:, self.x_idx] / 1000,
            hyp[:, self.y_idx] / 1000,
            hyp[:, self.z_idx] / 1000,
            dir_x,
            dir_y,
            dir_z,
            pmt_time - event_time,
            energy,
            obs[:, 0] / 1000,
            obs[:, 1] / 1000,
            obs[:, 2] / 1000,
        ],
            axis=1
        )

        return out


def mish(x):
    r"""Mish: A Self Regularized Non-Monotonic Neural Activation Function.

    Computes mish activation:

    $$
    \mathrm{mish}(x) = x \cdot \tanh(\mathrm{softplus}(x)).
    $$

    See [Mish: A Self Regularized Non-Monotonic Neural Activation Function](https://arxiv.org/abs/1908.08681).

    Usage:

    >>> x = tf.constant([1.0, 0.0, 1.0])
    >>> tfa.activations.mish(x)
    <tf.Tensor: shape=(3,), dtype=float32, numpy=array([0.865098..., 0.       , 0.865098...], dtype=float32)>

    Args:
        x: A `Tensor`. Must be one of the following types:
            `bfloat16`, `float16`, `float32`, `float64`.
    Returns:
        A `Tensor`. Has the same type as `x`.
    """
    x = tf.convert_to_tensor(x)
    return x * tf.math.tanh(tf.math.softplus(x))


def get_hitnet(activation=mish, layers=3):
    hit_input = tf.keras.Input(shape=(5,))
    params_input = tf.keras.Input(shape=(7,))

    t = hitnet_trafo()
    h = t(hit_input, params_input)
    for i in range(layers):
        h = tf.keras.layers.Dense(256, activation=activation, name='dense_' + str(i))(h)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid', name='dense_' + str(layers))(h)

    hitnet = tf.keras.Model(inputs=[hit_input, params_input], outputs=outputs)

    return hitnet
