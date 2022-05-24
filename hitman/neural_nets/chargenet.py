import tensorflow as tf

from hitman.neural_nets.transformations import chargenet_trafo

def get_chargenet(labels):

    charge_input = tf.keras.Input(shape=(2,))
    params_input = tf.keras.Input(shape=(len(labels),))

    t = chargenet_trafo(labels=labels)

    h = t(charge_input, params_input)
    h = tf.keras.layers.Dense(32, activation="relu")(h)
    h = tf.keras.layers.Dropout(0.001)(h)
    h = tf.keras.layers.Dense(64, activation="relu")(h)
    h = tf.keras.layers.Dropout(0.001)(h)
    h = tf.keras.layers.Dense(128, activation="relu")(h)
    h = tf.keras.layers.Dropout(0.001)(h)
    h = tf.keras.layers.Dense(256, activation="relu")(h)
    h = tf.keras.layers.Dropout(0.001)(h)
    h = tf.keras.layers.Dense(256, activation="relu")(h)
    h = tf.keras.layers.Dropout(0.001)(h)
    h = tf.keras.layers.Dense(128, activation="relu")(h)
    h = tf.keras.layers.Dropout(0.001)(h)
    h = tf.keras.layers.Dense(64, activation="relu")(h)
    h = tf.keras.layers.Dropout(0.001)(h)
    h = tf.keras.layers.Dense(32, activation="relu")(h)
    h = tf.keras.layers.Dropout(0.001)(h)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(h)

    chargenet = tf.keras.Model(inputs=[charge_input, params_input], outputs=outputs)

    return chargenet
