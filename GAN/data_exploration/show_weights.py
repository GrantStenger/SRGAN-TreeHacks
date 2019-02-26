import tensorflow as tf
from keras.models import load_model
import keras.backend as K

def root_mean_squared_error(y_true, y_pred):
    """ Defines the RMSE loss function.

        Args:
            y_true: The true label.
            y_pred: The predicted label.

        Returns:
            error: The RMSE error value

    """

    error = K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))
    return error

model3 = load_model("epoch_3.h5py", custom_objects={"tf":tf, "root_mean_squared_error":root_mean_squared_error})
model2 = load_model("epoch_2.h5py", custom_objects={"tf":tf, "root_mean_squared_error":root_mean_squared_error})

weights3 = model3.get_weights()
weights2 = model2.get_weights()

import matplotlib.pyplot as plt


weights = weights2[0]


def show_weights(weights, n_filters, n_depth):

    
    fig, axs = plt.subplots(nrows=n_filters, ncols=n_depth, figsize=(8,8))

    for i in range(n_filters):

        axi = axs[i]
        for j in range(n_depth):

            img = weights[i][j]
            axi[j].imshow(weights[i][j])
            axi[j].set_title("i: {0} j: {1}".format(i, j))
            
    return fig, axs



fig, axs = show_weights(weights2[0],3,3)
fig, axs = show_weights(weights3[0],3,3)