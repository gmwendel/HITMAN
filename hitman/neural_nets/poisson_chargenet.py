import tensorflow as tf
import numpy as np

class poisson_chargenet_trafo(tf.keras.layers.Layer):
    '''Class to transform inputs for Poisson Charge Net'''
    
    def __init__(self, hyp_norm=None, obs_norm=None, **kwargs):
        super().__init__(**kwargs)
        self.hyp_norm = hyp_norm
        self.obs_norm = obs_norm

    def get_config(self):
        config = super().get_config()
        config.update({
            "hyp_norm": self.hyp_norm,
            "obs_norm": self.obs_norm,
        })
        return config

    def call(self, pmt_pos, params):
        '''
        Parameters:
        -----------
        pmt_pos : tensor
            shape (N, 3), containing the x, y, z position of a single PMT
            
        params : tensor
            shape (N, 3), containing energy, scattering length, and absorption length
        '''
        
        # Normalization
        pmt_normed = pmt_pos
        if self.obs_norm is not None:
            pmt_normed = (pmt_pos - self.obs_norm[1]) / self.obs_norm[0]
            
        params_normed = params
        if self.hyp_norm is not None:
            params_normed = (params - self.hyp_norm[1]) / self.hyp_norm[0]
            
        out = tf.concat([pmt_normed, params_normed], axis=1)
        return out

def mish(x):
    x = tf.convert_to_tensor(x)
    return x * tf.math.tanh(tf.math.softplus(x))

def get_poisson_chargenet(activation=mish, layers=3, nodes=256, hyp_norm=None, obs_norm=None):
    pmt_input = tf.keras.Input(shape=(3,))
    params_input = tf.keras.Input(shape=(3,))

    t = poisson_chargenet_trafo(hyp_norm=hyp_norm, obs_norm=obs_norm)
    h = t(pmt_input, params_input)
    
    for i in range(layers):
        h = tf.keras.layers.Dense(nodes, activation=activation, name='dense_' + str(i))(h)
        
    # Output must be strictly positive since it represents the expected Poisson rate (lambda)
    # Adding a tiny epsilon inside the loss to prevent log(0), but softplus naturally stays positive
    outputs = tf.keras.layers.Dense(1, activation='softplus', name='dense_' + str(layers))(h)

    chargenet = tf.keras.Model(inputs=[pmt_input, params_input], outputs=outputs)

    return chargenet

def poisson_nll_loss(y_true, y_pred):
    """
    Custom Negative Log-Likelihood loss for a Poisson distribution.
    y_true: Observed charge (k)
    y_pred: Expected charge rate (lambda)
    
    NLL = lambda - k * ln(lambda)
    """
    epsilon = 1e-7 # Prevent log(0)
    y_pred = tf.clip_by_value(y_pred, epsilon, tf.float32.max)
    return tf.reduce_mean(y_pred - y_true * tf.math.log(y_pred))
