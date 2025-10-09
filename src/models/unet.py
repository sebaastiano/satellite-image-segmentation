"""
U-Net model implementation for satellite image segmentation.
"""

import tensorflow as tf
from keras.layers import Input, Conv2D, Dropout, MaxPooling2D, Conv2DTranspose, concatenate, Lambda
from keras.models import Model
from tensorflow.keras.regularizers import l2


def build_model(hp):
    """
    Build U-Net model with hyperparameters for tuning.

    Args:
        hp: Keras Tuner HyperParameters object

    Returns:
        Compiled Keras Model
    """
    num_classes = 6

    # Hyperparameters
    base_filters = hp.Choice('base_filters', values=[8, 12, 16])
    filters = [base_filters * (i + 1) for i in range(5)]
    dropout_rate = hp.Float('dropout_rate', 0.1, 0.5, step=0.1)
    l2_reg = hp.Float('l2', 1e-5, 1e-2, sampling='log')
    batch_size = hp.Choice('batch_size', values=[16, 32, 64])

    # Input normalization
    inputs = Input(shape=(256, 256, 3))
    s = Lambda(lambda x: x / 255)(inputs)

    # Convolutional block
    def conv_block(inputs, num_filters, dropout_rate, l2_reg):
        x = Conv2D(num_filters, (3, 3), activation='relu', kernel_initializer='he_normal',
                   padding='same', kernel_regularizer=l2(l2_reg))(inputs)
        x = Dropout(dropout_rate)(x)
        x = Conv2D(num_filters, (3, 3), activation='relu', kernel_initializer='he_normal',
                   padding='same', kernel_regularizer=l2(l2_reg))(x)
        return x

    # Encoder block
    def encoder_block(inputs, num_filters, dropout_rate, l2_reg):
        x = conv_block(inputs, num_filters, dropout_rate, l2_reg)
        p = MaxPooling2D((2, 2))(x)
        return x, p

    # Decoder block
    def decoder_block(inputs, skip_features, num_filters, dropout_rate, l2_reg):
        x = Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='same')(inputs)
        x = concatenate([x, skip_features])
        x = conv_block(x, num_filters, dropout_rate, l2_reg)
        return x

    # Encoder
    c1, p1 = encoder_block(s, filters[0], dropout_rate, l2_reg)
    c2, p2 = encoder_block(p1, filters[1], dropout_rate, l2_reg)
    c3, p3 = encoder_block(p2, filters[2], dropout_rate, l2_reg)
    c4, p4 = encoder_block(p3, filters[3], dropout_rate, l2_reg)

    # Bottleneck
    c5 = conv_block(p4, filters[4], dropout_rate, l2_reg)

    # Decoder
    u6 = decoder_block(c5, c4, filters[3], dropout_rate, l2_reg)
    u7 = decoder_block(u6, c3, filters[2], dropout_rate, l2_reg)
    u8 = decoder_block(u7, c2, filters[1], dropout_rate, l2_reg)
    u9 = decoder_block(u8, c1, filters[0], dropout_rate, l2_reg)

    # Output layer
    outputs = Conv2D(num_classes, (1, 1), padding='same', activation='softmax')(u9)

    # Model compilation
    model = Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model
