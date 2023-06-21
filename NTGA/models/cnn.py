# +
import tensorflow as tf
from tensorflow.keras import layers

def CNN(input_shape, num_classes):
    inputs = tf.keras.Input(input_shape)
    x = layers.Conv2D(64, (5, 5), padding='same', activation='relu')(inputs)
    x = layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(64, (5, 5), padding='same', activation='relu')(x)
    x = layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)

    x = layers.Flatten()(x)
    x = layers.Dense(384, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(192, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)
