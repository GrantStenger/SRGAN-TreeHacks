import tensorflow as tf
from keras.models import Sequential
from keras.layers import Lambda, UpSampling2D, Conv2D

from utils import to_float


def create_generator(input_shape, output_shape, resize_factor):
    model = Sequential()

    # Float Cast Layer
    model.add(Lambda(to_float, input_shape=input_shape))

    # Upsample Layer
    model.add(UpSampling2D(resize_factor))

    # Convolutional Layer
    model.add(Conv2D(filters=3, kernel_size=(3, 3), strides=(1, 1),
                     padding="SAME", activation="softplus"))

    # Resize Layer
    model.add(Lambda(lambda image: tf.image.resize_images(
        image, output_shape,
        method=tf.image.ResizeMethod.BICUBIC,
        align_corners=True
        )))

    return model
