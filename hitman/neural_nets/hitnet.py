import tensorflow as tf
import tensorflow_addons as tfa
from hitman.neural_nets.transformations import hitnet_trafo

def get_hitnet(labels,activation=tfa.activations.mish):

    hit_input = tf.keras.Input(shape=(7,))
    params_input = tf.keras.Input(shape=(len(labels),))

    t = hitnet_trafo(labels=labels)

    h = t(hit_input, params_input)
    h = tf.keras.layers.Dense(32, activation=activation)(h)
    h = tf.keras.layers.Dropout(0.001)(h)
    h = tf.keras.layers.Dense(64, activation=activation)(h)
    h = tf.keras.layers.Dropout(0.001)(h)
    h = tf.keras.layers.Dense(128, activation=activation)(h)
    h = tf.keras.layers.Dropout(0.001)(h)
    h = tf.keras.layers.Dense(256, activation=activation)(h)
    h = tf.keras.layers.Dropout(0.001)(h)
#    h = tf.keras.layers.Dense(512, activation=activation)(h)
#    h = tf.keras.layers.Dropout(0.001)(h)
#    h = tf.keras.layers.Dense(512, activation=activation)(h)
#    h = tf.keras.layers.Dropout(0.001)(h)
    h = tf.keras.layers.Dense(256, activation=activation)(h)
    h = tf.keras.layers.Dropout(0.001)(h)
    h = tf.keras.layers.Dense(128, activation=activation)(h)
    h = tf.keras.layers.Dropout(0.001)(h)
    h = tf.keras.layers.Dense(64, activation=activation)(h)
    h = tf.keras.layers.Dropout(0.001)(h)
    h = tf.keras.layers.Dense(32, activation=activation)(h)
    h = tf.keras.layers.Dropout(0.001)(h)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(h)

    hitnet = tf.keras.Model(inputs=[hit_input, params_input], outputs=outputs)

    return hitnet
