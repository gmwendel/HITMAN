import tensorflow as tf
import numpy as np

def mish(x):
    x = tf.convert_to_tensor(x)
    return x * tf.math.tanh(tf.math.softplus(x))

from hitman.neural_nets.d2h_layers import D2hSymmetrizedLayer4Param, D2hSymmetrizedLayer8Param

def get_shape_net(activation=mish, layers=2, nodes=128, hyp_norm=None, obs_norm=None, use_vertex=False):
    hyp_input = tf.keras.Input(shape=(2,), name="shape_hyp_in")
    obs_input = tf.keras.Input(shape=(None, 3), name="shape_obs_in")
    total_hits_input = tf.keras.Input(shape=(1,), name="shape_total_hits_in")
    
    # We define the detector constants based on the training dataset.
    # Currently, pitch = 10.0mm (typical for LiquidO WbLS grid), and scale = 1000.0mm
    # Note: Sensors must be offset from bounding box edges/corners.
    pitch = 10.0
    detector_scale = 1000.0

    if use_vertex:
        vertex_input = tf.keras.Input(shape=(4,), name="shape_vertex_in")
        d2h_tensor = D2hSymmetrizedLayer8Param(pitch=pitch, detector_scale=detector_scale)(
            [hyp_input, obs_input, vertex_input]
        )
        inputs_list = [hyp_input, obs_input, vertex_input, total_hits_input]
    else:
        d2h_tensor = D2hSymmetrizedLayer4Param(pitch=pitch, detector_scale=detector_scale)(
            [hyp_input, obs_input]
        )
        inputs_list = [hyp_input, obs_input, total_hits_input]

    # The transformation layer natively handles normalization of features to O(1)
    # so we can directly feed it to the dense layers.
    x = d2h_tensor
    for i in range(layers):
        x = tf.keras.layers.Dense(nodes, activation=activation)(x)

    outputs = tf.keras.layers.Dense(1, activation="linear")(x)
    outputs = tf.squeeze(outputs, axis=-1)
    pmf = tf.keras.layers.Softmax(axis=1)(outputs)
    
    # Scale PMF by total hits to predict the raw Poisson rate for each PMT
    expected_hits = pmf * total_hits_input
    expected_hits = expected_hits + 1e-9 # Floor to prevent Log(0)
    
    return tf.keras.Model(inputs=inputs_list, outputs=expected_hits, name="ShapeNet")

def get_acceptance_net(activation=mish, layers=2, nodes=128, hyp_norm=None, obs_norm=None):
    hyp_input = tf.keras.Input(shape=(2,), name="acc_hyp_in")
    
    norm_hyp = tf.keras.layers.Normalization(mean=hyp_norm[1], variance=hyp_norm[0]**2, axis=-1)(hyp_input)
    
    h = norm_hyp
    for i in range(layers):
        h = tf.keras.layers.Dense(nodes, activation=activation)(h)
        
    outputs = tf.keras.layers.Dense(1, activation=tf.math.softplus, name="acc_dense_out", 
                                    bias_initializer=tf.keras.initializers.Constant(-3.5))(h)
    return tf.keras.Model(inputs=hyp_input, outputs=outputs, name="AcceptanceNet")
