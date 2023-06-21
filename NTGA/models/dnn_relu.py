# +
import tensorflow as tf
from tensorflow.keras import layers

def DNN_ReLU(input_shape, num_classes):
    inputs = tf.keras.Input(input_shape)
    x = layers.Dense(512, activation='relu')(inputs)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dense(512, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)
