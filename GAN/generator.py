""" Defines the generator neural network for the GAN. """

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Lambda, UpSampling2D, Conv2D

from utils import to_float


def create_generator(input_shape, output_size, resize_factor):
    """ Creates the generator for the GAN.

        Instantiates a Keras model which upsamples the input
        and performs a convolution, then resizes the image to fit output.

        Args:
            input_shape: A triple (x, y, channels) representing the input shape.
            output_size: A tuple (x, y) representing the output size.
            resize_factor: An integer factor by which to scale the input.

        Returns:
            model: The created Keras model.

    """

    model = Sequential()
    model.name = "generator"

    # Float Cast Layer
    model.add(Lambda(to_float, input_shape=input_shape))

    # Upsample Layer
    model.add(UpSampling2D(resize_factor))

    # Convolutional Layer
    model.add(Conv2D(filters=3, kernel_size=(3, 3), strides=(1, 1),
                     padding="SAME", activation="softplus"))

    # Resize Layer
    model.add(Lambda(lambda image: tf.image.resize_images(
        image, output_size,
        method=tf.image.ResizeMethod.BICUBIC,
        align_corners=True
        )))

    return model
