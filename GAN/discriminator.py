""" Defines the discriminator neural network for the GAN. """

from keras.models import Sequential
from keras.layers import Conv2D, Dense, MaxPooling2D, Flatten


def create_discriminator(input_shape):
    """ Creates the discriminator for the GAN.

        Instantiates a Keras model which performs two convolutions
        on the input, then feeds it into a dense layer to determine
        whether the given image is real or generated.

        Args:
            input_shape: A triple (x, y, channels) representing the input shape.

        Returns:
            model: The created Keras model.

    """

    model = Sequential()
    model.name = "discriminator"

    # Convolutional Layer 1
    model.add(Conv2D(filters=16, kernel_size=(2, 2), strides=(1, 1),
                     activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D())

    # Convolutional Layer 2
    model.add(Conv2D(filters=32, kernel_size=(2, 2), strides=(1, 1),
                     activation='relu'))
    model.add(Flatten())

    # Dense Layer
    model.add(Dense(32, activation='relu'))

    # Sigmoid Layer
    model.add(Dense(1, activation='sigmoid'))

    return model
