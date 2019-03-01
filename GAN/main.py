""" Main function for training the GAN. """

import argparse
import os
import cv2
import numpy as np
import tensorflow as tf
import keras
from keras.models import load_model, Model
from keras.layers import Input
from sklearn.utils import shuffle
from discriminator import create_discriminator
from generators import create_baseline_cnn, create_3colorsto1color_cnn, create_3colorsto1color_2layer_cnn, create_3colorsto1color_2layer_MultiFilter_cnn, create_2layer_baseline_cnn

from utils import make_trainable, root_mean_squared_error, chunks, load_img

def makedirs():
    """ Creates data directories if they don't already exist. """

    os.makedirs(FLAGS.out_dir, exist_ok=True)
    os.makedirs(FLAGS.out_dir + "/weights", exist_ok=True)
    os.makedirs(FLAGS.out_dir + "/weights/ginit", exist_ok=True)
    os.makedirs(FLAGS.out_dir + "/weights/gen", exist_ok=True)
    os.makedirs(FLAGS.out_dir + "/weights/disc", exist_ok=True)
    os.makedirs(FLAGS.out_dir + "/samples", exist_ok=True)
    os.makedirs(FLAGS.out_dir + "/samples/ginit", exist_ok=True)
    os.makedirs(FLAGS.out_dir + "/samples/gen", exist_ok=True)

def initialize_generator():
    """ Initializes the generator model.

        If a saved model is specified, that model is loaded. Otherwise,
        a new model is created.

        Returns:
            generator: The initialized Keras model.
    """

    # Creates a new model if no saved model is specified
    if FLAGS.gen_path is None:
        generator = create_baseline_cnn(input_shape=(240, 426, 3),
                                        output_size=(480, 854),
                                        resize_factor=2)
    # Otherwise loads the specified model
    else:
        params = {"root_mean_squared_error": root_mean_squared_error,
                  "tf": tf,
                  "output_shape": (480, 854)}

        generator = load_model(FLAGS.gen_path, custom_objects=params)

    # Compiles the model
    generator.compile(optimizer="adam",
                      loss=root_mean_squared_error,
                      metrics=["acc"])

    return generator

def create_batch_tensors(batch):
    """ Creates batch tensors from a batch of filepaths.

        Args:
            batch: A list of filepaths

        Returns:
            success: True if successful.
            x_train: The x_train tensor.
            y_train: The y_train tensor.
    """

    x_train = []
    y_train = []

    # Adds image filepaths to batch
    for filepath in batch:
        x_success, x_image = load_img(FLAGS.x_dir + "/" + filepath,
                                      size=(426, 240))
        y_success, y_image = load_img(FLAGS.y_dir + "/" + filepath,
                                      size=(854, 480))

        if not x_success or not y_success:
            return False, None, None

        x_train.append(x_image)
        y_train.append(y_image)

    # Removes dimensions of 1 from training set
    x_train = np.squeeze(x_train)
    y_train = np.squeeze(y_train)

    return True, x_train, y_train

def train_generator():
    """ Trains the generator by itself to achieve a baseline.

        Trains the generator against ground truth with RMSE. The point of this
        is to have a baseline generator by the time we plug it into the GAN.
        The generator will train for a total of FLAGS.ginit_epochs minus
        FLAGS.ginit_start_epoch epochs.

        Returns:
            generator: The trained Keras model.
    """

    # Initializes the model
    generator = initialize_generator()

    # Gets filepaths of training set
    files = os.listdir(FLAGS.x_dir)

    # Trains the model
    for epoch in range(FLAGS.ginit_start_epoch, FLAGS.ginit_epochs):
        # Shuffles the training set and divide it into batches
        np.random.shuffle(files)
        batches = chunks(files, FLAGS.batch_size)

        # Trains a batch
        for batch in batches:
            # Creates batch tensors
            success, x_train, y_train = create_batch_tensors(batch)
            if not success:
                continue

            # Trains the generator on a batch
            generator.fit(x_train, y_train)

        print("\n \n \n \n COMPLETED EPOCH {0} \n \n \n \n".format(epoch))

        # Gets samples for visual verification
        out = generator.predict(x_train)
        base_filepath = FLAGS.out_dir + "/samples/ginit/epoch_{0}".format(epoch)

        # Saves samples
        for i, sample in enumerate(out):
            resized_train = cv2.resize(x_train[i], (0, 0), fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(base_filepath + "_img_{0}_input.png".format(i),
                        resized_train)
            cv2.imwrite(base_filepath + "_img_{0}_pred.png".format(i),
                        sample)
            cv2.imwrite(base_filepath + "_img_{0}_true.png".format(i),
                        y_train[i])

        # Saves generator weights
        savepath = FLAGS.out_dir + "/weights/ginit/epoch_{0}.h5py".format(epoch)
        generator.save(savepath)

    return generator

def main():
    """ Creates and trains the GAN. """
    
    # Creates directories
    makedirs()

    # Initializes and trains the generator if necessary
    generator = train_generator()

    # Loads discriminator
    if FLAGS.disc_path is None:
        discriminator = create_discriminator((15, 26, 512))
    else:
        discriminator = load_model(FLAGS.disc_path)

    # Compiles discriminator
    discriminator.compile(optimizer="adam",
                          loss="binary_crossentropy",
                          metrics=["acc"])

    # Sets initial discriminator trainability to False
    make_trainable(discriminator, False)

    # Loads VGG
    vgg = keras.applications.VGG16(include_top=False)

    # VGG should never be trained
    vgg.trainable = False

    # Creates GAN Model
    input_layer = Input(shape=(240, 426, 3))
    out_gen = generator(input_layer)
    out_vgg = vgg(out_gen)
    out_disc = discriminator(out_vgg)

    # Compiles the GAN
    # Notice the loss_weights argument: this can be changed to achieve different
    # results. In general, binary cross-entropy controls resolution accuracy
    # and RMSE controls color accuracy.
    model = Model(inputs=input_layer, outputs=[out_disc, out_gen])
    model.compile(optimizer="adam",
                  loss=["binary_crossentropy", root_mean_squared_error],
                  loss_weights=[0.9, 0.1],
                  metrics=["acc"])

    # Gets filepaths of training set
    files = os.listdir(FLAGS.x_dir)

    # Set to False to train the discriminator first
    train_gen = False

    # Trains the model
    for epoch in range(FLAGS.start_epoch, FLAGS.epochs):
        # Shuffles the training set and divide it into batches
        np.random.shuffle(files)
        batches = chunks(files, FLAGS.batch_size)

        # Trains a batch
        for batch in batches:
            # Creates batch tensors
            x_train, y_train = create_batch_tensors(batch)

            # Generator's turn to train
            if train_gen:
                print("Training generator: epoch {0}".format(epoch))

                # Set discriminator trainability to False
                make_trainable(discriminator, False)

                # Trains the generator
                metrics = model.fit(x_train, [np.ones([len(x_train)]), y_train])

                # If the generator is good enough, switch to discriminator
                if metrics.history["generator_acc"][0] > .9:
                    train_gen = False

            # Discriminator's turn to train
            else:
                print("Training discriminator: epoch {0}".format(epoch))

                # Set discriminator trainability to True
                make_trainable(discriminator, True)

                # Get generated data from inputs
                gen_input = x_train
                gen_output = generator.predict(gen_input)

                # Combines x data with their corresponding y labels
                disc_input = np.concatenate([gen_output, y_train])
                ground_truth = np.concatenate([np.zeros([len(gen_output)]),
                                               np.ones([len(y_train)])])

                # Shuffles generated images and real images
                disc_input, ground_truth = shuffle(disc_input, ground_truth)

                # Run VGG embeddings on generated images
                vgg_output = vgg.predict(disc_input, batch_size=1)

                # Trains the discriminator
                metrics = discriminator.fit(vgg_output, ground_truth)

                # If the discriminator is good enough, switch to generator
                if metrics.history["acc"][0] > .9:
                    train_gen = True

        print("\n \n \n \n COMPLETED EPOCH {0} \n \n \n \n".format(epoch))

        # Get samples for visual verification
        out = generator.predict(x_train)
        base_filepath = FLAGS.out_dir + "/samples/gen/epoch_{0}".format(epoch)

        # Save samples
        for i, sample in enumerate(out):
            resized_train = cv2.resize(x_train[i], (0, 0), fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(base_filepath + "_img_{0}_input.png".format(i),
                        resized_train)
            cv2.imwrite(base_filepath + "_img_{0}_pred.png".format(i),
                        sample)
            cv2.imwrite(base_filepath + "_img_{0}_true.png".format(i),
                        y_train[i])

        # Save generator and discriminator weights
        savepath = FLAGS.out_dir + "/weights"
        generator.save(savepath + "/gen/epoch_{0}.h5py".format(epoch))
        make_trainable(discriminator, True)
        discriminator.save(savepath + "/disc/epoch_{0}.h5py".format(epoch))


if __name__ == "__main__":
    # Instantiates an arg parser
    PARSER = argparse.ArgumentParser()

    # Adds arguments
    PARSER.add_argument("--gen_path", type=str, default=None,
                        help="path to generator .h5py")
    PARSER.add_argument("--disc_path", type=str, default=None,
                        help="path to discriminator .h5py")
    PARSER.add_argument("--out_dir", type=str, default="data",
                        help="path to export directory")
    PARSER.add_argument("--x_dir", type=str, default=None,
                        help="path to x data directory")
    PARSER.add_argument("--y_dir", type=str, default=None,
                        help="path to y data directory")
    PARSER.add_argument("--batch_size", type=int, default=8,
                        help="batch size")
    PARSER.add_argument("--epochs", type=int, default=100,
                        help="number of GAN epochs")
    PARSER.add_argument("--start_epoch", type=int, default=0,
                        help="starting epoch for GAN")
    PARSER.add_argument("--ginit_epochs", type=int, default=5,
                        help="number of generator init epochs")
    PARSER.add_argument("--ginit_start_epoch", type=int, default=0,
                        help="starting epoch for generator init")

    # Parses known arguments
    FLAGS, _ = PARSER.parse_known_args()

    main()
