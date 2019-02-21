import cv2
import os
from keras.models import load_model
import tensorflow as tf 
import keras.backend as K


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

generator = load_model('data_tiny2/gen/epoch_2.h5py', custom_objects={'root_mean_squared_error':root_mean_squared_error, 'tf': tf, 'output_shape': (480, 852)})

generator.compile(optimizer='adam', loss=root_mean_squared_error, metrics=['accuracy'])
