import tensorflow as tf


class hitnet_trafo(tf.keras.layers.Layer):
    '''Class to transform inputs for Hitnet
    This layer performs normalization of all parameters.
    '''

    def __init__(self, hyp_norm=None, obs_norm=None):
        super().__init__()
        self.hyp_norm = hyp_norm
        self.obs_norm = obs_norm

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
            shape (N, 3) containing energy, scattering length, and absorption length
        '''

        obs_normed = obs
        if self.obs_norm is not None:
            obs_normed = (obs - self.obs_norm[1]) / self.obs_norm[0]

        hyp_normed = hyp
        if self.hyp_norm is not None:
            hyp_normed = (hyp - self.hyp_norm[1]) / self.hyp_norm[0]

        out = tf.concat([obs_normed, hyp_normed], axis=1)
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


def get_hitnet(activation=mish, layers=3, hyp_norm=None, obs_norm=None):
    hit_input = tf.keras.Input(shape=(5,))
    params_input = tf.keras.Input(shape=(3,))

    t = hitnet_trafo(hyp_norm=hyp_norm, obs_norm=obs_norm)
    h = t(hit_input, params_input)
    for i in range(layers):
        h = tf.keras.layers.Dense(256, activation=activation, name='dense_' + str(i))(h)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid', name='dense_' + str(layers))(h)

    hitnet = tf.keras.Model(inputs=[hit_input, params_input], outputs=outputs)

    return hitnet
