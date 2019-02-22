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
from generator import create_generator

from utils import make_trainable, root_mean_squared_error, chunks, load_img

def makedirs():
    os.makedirs(FLAGS.out_dir, exist_ok=True)
    os.makedirs(FLAGS.out_dir + "/weights", exist_ok=True)
    os.makedirs(FLAGS.out_dir + "/weights/ginit", exist_ok=True)
    os.makedirs(FLAGS.out_dir + "/weights/gen", exist_ok=True)
    os.makedirs(FLAGS.out_dir + "/weights/disc", exist_ok=True)
    os.makedirs(FLAGS.out_dir + "/samples", exist_ok=True)
    os.makedirs(FLAGS.out_dir + "/samples/ginit", exist_ok=True)
    os.makedirs(FLAGS.out_dir + "/samples/gen", exist_ok=True)

def initialize_generator():
    if FLAGS.gen_path is None:
        generator = create_generator((240, 426, 3), (480, 852), 2)
    else:
        params = {"root_mean_squared_error": root_mean_squared_error,
                  "tf": tf,
                  "output_shape": (480, 852)}

        generator = load_model(FLAGS.gen_path, custom_objects=params)

    generator.compile(optimizer="adam",
                      loss=root_mean_squared_error,
                      metrics=["acc"])

    return generator

def train_generator():
    generator = initialize_generator()

    files = os.listdir(FLAGS.X_dir)

    for epoch in range(FLAGS.gen_start_epoch, FLAGS.gen_epochs):
        np.random.shuffle(files)
        batches = chunks(files, FLAGS.batch_size)

        for batch in batches:
            Xtrain = []
            ytrain = []
            for filepath in batch:
                Xtrain.append(load_img(FLAGS.X_dir + "/" + filepath,
                                       size=(426, 240)))
                ytrain.append(load_img(FLAGS.y_dir + "/" + filepath,
                                       size=(852, 480)))

            Xtrain = np.squeeze(Xtrain)
            ytrain = np.squeeze(ytrain)

            generator.fit(Xtrain, ytrain)

        print("\n \n \n \n COMPLETED EPOCH {0} \n \n \n \n".format(epoch))

        # quick evaluation on train set
        out = generator.predict(Xtrain)

        for i, sample in enumerate(out):
            cv2.imwrite(FLAGS.out_dir + "/samples/ginit/epoch_{0}_img_{1}_input.png".format(epoch, i), Xtrain[i])
            cv2.imwrite(FLAGS.out_dir + "/samples/ginit/epoch_{0}_img_{1}_pred.png".format(epoch, i), sample)
            cv2.imwrite(FLAGS.out_dir + "/samples/ginit/epoch_{0}_img_{1}_true.png".format(epoch, i), ytrain[i])

        generator.save(FLAGS.out_dir + "/weights/ginit/epoch_{0}".format(epoch) + ".h5py")

    return generator

def main():
    makedirs()

    generator = train_generator()

    # Load Discriminator
    if FLAGS.disc_path is None:
        discriminator = create_discriminator((15, 26, 512))
    else:
        discriminator = load_model(FLAGS.disc_path)

    discriminator.compile(optimizer="adam",
                          loss="binary_crossentropy",
                          metrics=["acc"])
    make_trainable(discriminator, False)

    # Load VGG
    vgg = keras.applications.VGG16(include_top=False)
    vgg.trainable = False

    # Create GAN Model
    input_layer = Input(shape=(240, 426, 3))
    out_gen = generator(input_layer)
    out_vgg = vgg(out_gen)
    out_disc = discriminator(out_vgg)

    model = Model(inputs=input_layer, outputs=[out_disc, out_gen])
    model.compile(optimizer="adam",
                  loss=["binary_crossentropy", root_mean_squared_error],
                  loss_weights=[.95, 0.05],
                  metrics=["acc"])

    files = os.listdir(FLAGS.X_dir)
    train_gen = False

    # Define Training Loop
    for epoch in range(FLAGS.gan_start_epoch, FLAGS.gan_epochs):
        np.random.shuffle(files)
        batches = chunks(files, FLAGS.batch_size)

        for i, batch in enumerate(batches):
            Xtrain = []
            ytrain = []
            for filepath in batch:
                Xtrain.append(load_img(FLAGS.X_dir + "/" + filepath,
                                       size=(426, 240)))
                ytrain.append(load_img(FLAGS.y_dir + "/" + filepath,
                                       size=(852, 480)))

            Xtrain = np.squeeze(Xtrain)
            ytrain = np.squeeze(ytrain)

            if train_gen:
                print("Training generator")
                make_trainable(discriminator, False)

                metrics = model.fit(Xtrain, [np.ones([len(Xtrain)]), ytrain])
                if metrics.history["discriminator_acc"][0] > .8:
                    train_gen = False
            else:
                print("Training discriminator")
                make_trainable(discriminator, True)

                # Get generated data from inputs
                gen_input = Xtrain
                gen_output = generator.predict(gen_input)

                disc_input = np.concatenate([gen_output, ytrain])
                ground_truth = np.concatenate([np.zeros([len(gen_output)]),
                                               np.ones([len(ytrain)])])
                disc_input, ground_truth = shuffle(disc_input, ground_truth)


                vgg_output = vgg.predict(disc_input, batch_size=1)

                metrics = discriminator.fit(vgg_output, ground_truth)

                if metrics.history["acc"][0] > .8:
                    train_gen = True

        print("\n \n \n \n COMPLETED EPOCH {0} \n \n \n \n".format(epoch))
        out = generator.predict(Xtrain)

        os.makedirs(FLAGS.out_dir + "/samples", exist_ok=True)
        for i, sample in enumerate(out):
            cv2.imwrite(FLAGS.out_dir + "/samples/gen/epoch_{0}_img_{1}_input.png".format(epoch, i), Xtrain[i])
            cv2.imwrite(FLAGS.out_dir + "/samples/gen/epoch_{0}_img_{1}_pred.png".format(epoch, i), sample)
            cv2.imwrite(FLAGS.out_dir + "/samples/gen/epoch_{0}_img_{1}_true.png".format(epoch, i), ytrain[i])

        generator.save(FLAGS.out_dir + "/weights/gen/epoch_{0}.h5py".format(epoch))
        make_trainable(discriminator, True)
        discriminator.save(FLAGS.out_dir + "/weights/disc/epoch_{0}.h5py".format(epoch))


if __name__ == "__main__":
    # Instantiates Argument Parser
    parser = argparse.ArgumentParser()

    # Adds arguments
    parser.add_argument("--gen_path", type=str, default=None,
                        help="path to generator .h5py")
    parser.add_argument("--disc_path", type=str, default=None,
                        help="path to discriminator .h5py")
    parser.add_argument("--out_dir", type=str, default="data",
                        help="path to export directory")
    parser.add_argument("--X_dir", type=str, default=None,
                        help="path to X data directory")
    parser.add_argument("--y_dir", type=str, default=None,
                        help="path to y data directory")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="batch size")
    parser.add_argument("--gan_epochs", type=int, default=100,
                        help="number of GAN epochs")
    parser.add_argument("--gan_start_epoch", type=int, default=0,
                        help="starting epoch for GAN")
    parser.add_argument("--gen_epochs", type=int, default=10,
                        help="number of generator epochs")
    parser.add_argument("--gen_start_epoch", type=int, default=0,
                        help="starting epoch for generator")

    # Parses known arguments
    FLAGS, _ = parser.parse_known_args()

    main()
