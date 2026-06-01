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

def get_poisson_chargenet(activation=mish, layers=3, nodes=256, hyp_norm=None, obs_norm=None, use_bn=False, output_bias=None):
    pmt_input = tf.keras.Input(shape=(3,))
    params_input = tf.keras.Input(shape=(3,))

    t = poisson_chargenet_trafo(hyp_norm=hyp_norm, obs_norm=obs_norm)
    h = t(pmt_input, params_input)
    
    for i in range(layers):
        h = tf.keras.layers.Dense(nodes, activation=activation, name='dense_' + str(i))(h)
        if use_bn:
            h = tf.keras.layers.BatchNormalization()(h)
            
    if output_bias is not None:
        bias_initializer = tf.keras.initializers.Constant(output_bias)
    else:
        bias_initializer = 'zeros'
        
    # Output natively predicts lambda via exp, but we will change to linear during training
    outputs = tf.keras.layers.Dense(1, activation=tf.math.exp, name='dense_' + str(layers), bias_initializer=bias_initializer)(h)

    chargenet = tf.keras.Model(inputs=[pmt_input, params_input], outputs=outputs)

    return chargenet

def poisson_nll_loss(y_true, z_pred):
    """
    Custom Negative Log-Likelihood loss for a Poisson distribution predicting log-rate.
    y_true: Observed charge (k)
    z_pred: Predicted log expected charge rate (z = ln(lambda))
    
    NLL = lambda - k * ln(lambda) = e^z - k * z
    """
    # Fix broadcasting bug: y_true is 1D (N,) and z_pred is 2D (N, 1)
    y_true = tf.cast(y_true, tf.float32)
    y_true = tf.reshape(y_true, tf.shape(z_pred))
    
    # Clip z to prevent inf/nan from e^z
    z_pred = tf.clip_by_value(z_pred, -20.0, 20.0)
    return tf.reduce_mean(tf.math.exp(z_pred) - y_true * z_pred)
