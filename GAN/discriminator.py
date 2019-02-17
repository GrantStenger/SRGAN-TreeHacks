# Import Dependencies
import argparse
from keras.models import load_model, Sequential
from keras.layers import Conv2D, Dense, MaxPooling2D, Dropout, Flatten

def create_discriminator(model_path, input_shape):
    # Convolutional Layer 1
    model = Sequential()
    model.add(Conv2D(filters=64, kernel=(2,2), strides=(1,1), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D())

    # Convolutional Layer 2
    model.add(Conv2D(filters=32, kernel=(2,2), strides=(1,1), activation='relu'))
    model.add(Flatten())

    # Dense Layer
    model.add(Dense(1, activation='sigmoid'))

    # Compilation
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model
