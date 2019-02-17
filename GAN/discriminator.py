# Import Dependencies
import argparse
from keras.models import load_model, Sequential
from keras.layers import Conv2D, Dense, MaxPooling2D, Dropout, Flatten

def create_discriminator(model_path, input_shape):
    # Convolutional Layer 1
    model = Sequential()
    model.name = 'discriminator'

    model.add(Conv2D(filters=8, kernel_size=(2,2), strides=(1,1), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D())

    # Convolutional Layer 2
    model.add(Conv2D(filters=16, kernel_size=(2,2), strides=(1,1), activation='relu'))
    model.add(MaxPooling2D())

    # Convolutional Layer 3
    model.add(Conv2D(filters=16, kernel_size=(2,2), strides=(1,1), activation='relu'))
    model.add(Flatten())

    # Dense Layer
    model.add(Dense(32, activation='relu'))

    # Sigmoid Layer
    model.add(Dense(1, activation='sigmoid'))

    return model
