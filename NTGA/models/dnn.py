# +
import tensorflow as tf
from tensorflow.keras import layers

def DNN(input_shape, num_classes):
    inputs = tf.keras.Input(input_shape)
    x = layers.Dense(512)(inputs)
    x = tf.math.erf(x)
    x = layers.Dense(512)(x)
    x = tf.math.erf(x)
    x = layers.Dense(512)(x)
    x = tf.math.erf(x)
    x = layers.Dense(512)(x)
    x = tf.math.erf(x)
    outputs = layers.Dense(num_classes)(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)
