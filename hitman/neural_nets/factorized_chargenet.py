import tensorflow as tf
import numpy as np

class factorized_chargenet_trafo(tf.keras.layers.Layer):
    '''Class to transform inputs for Factorized Networks'''
    
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
        Handles both 2D and 3D inputs (e.g., (None, 3) or (None, 64, 3))
        '''
        if pmt_pos is not None:
            pmt_normed = pmt_pos
            if self.obs_norm is not None:
                pmt_normed = (pmt_pos - self.obs_norm[1]) / self.obs_norm[0]
                
        params_normed = params
        if self.hyp_norm is not None:
            params_normed = (params - self.hyp_norm[1]) / self.hyp_norm[0]
            
        if pmt_pos is not None:
            out = tf.concat([pmt_normed, params_normed], axis=-1)
            return out
        else:
            return params_normed

def mish(x):
    x = tf.convert_to_tensor(x)
    return x * tf.math.tanh(tf.math.softplus(x))

def get_shape_net(activation=mish, layers=2, nodes=128, hyp_norm=None, obs_norm=None):
    hyp_input = tf.keras.Input(shape=(2,), name="shape_hyp_in")
    obs_input = tf.keras.Input(shape=(None, 3), name="shape_obs_in") # Support arbitrary N_sensors

    # Broadcast hyp_input (None, 3) -> (None, 1, 3) -> (None, N_sensors, 3)
    num_sensors = tf.shape(obs_input)[1]
    hyp_tiled = tf.tile(tf.expand_dims(hyp_input, 1), [1, num_sensors, 1])

    t = factorized_chargenet_trafo(hyp_norm=hyp_norm, obs_norm=obs_norm)
    h = t(obs_input, hyp_tiled)
    
    for i in range(layers):
        h = tf.keras.layers.Dense(nodes, activation=activation, name='shape_dense_' + str(i))(h)

    outputs = tf.keras.layers.Dense(1, activation='linear', name='shape_dense_out')(h)
    outputs = tf.squeeze(outputs, axis=-1)
    outputs = tf.keras.layers.Softmax()(outputs)

    return tf.keras.Model(inputs=[hyp_input, obs_input], outputs=outputs, name="ShapeNet")

def get_acceptance_net(activation=mish, layers=2, nodes=128, hyp_norm=None, obs_norm=None):
    hyp_input = tf.keras.Input(shape=(2,), name="acc_hyp_in")

    t = factorized_chargenet_trafo(hyp_norm=hyp_norm, obs_norm=None)
    h = t(None, hyp_input)
    
    for i in range(layers):
        h = tf.keras.layers.Dense(nodes, activation=activation, name='acc_dense_' + str(i))(h)

    # Softplus ensures absolute detector acceptance is strictly positive
    outputs = tf.keras.layers.Dense(1, activation=tf.math.softplus, name='acc_dense_out')(h)

    return tf.keras.Model(inputs=hyp_input, outputs=outputs, name="AcceptanceNet")
