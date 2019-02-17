# Import Dependencies
import argparse, os
from keras.models import load_model, Sequential, Model
from keras.layers import Conv2D, Dense, MaxPooling2D
import keras.backend as K
from discriminator import create_discriminator
import cv2
import numpy as np


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


def load_img(img, size):
    full_img = cv2.imread(img)
    full_img = cv2.resize(full_img, size)
    full_img = np.expand_dims(full_img, axis=0)
    return full_img


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def main():
    # Load Generator
    generator = load_model(FLAGS.gen_path, custom_objects={'root_mean_squared_error':root_mean_squared_error})
    generator.compile(optimizer=Adam(), loss=root_mean_squared_error, metrics=['accuracy'])

    # Load Discriminator
    if FLAGS.disc_path is None:
        discriminator = create_discriminator(FLAGS.out_dir, (480, 852, 3))
    else:
        discriminator = load_model(FLAGS.disc_path)

    # Define Loss Functions
    model = Model(inputs=generator, outputs=discriminator)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    files = os.listdir(X_dir)
    train_GAN = False

    # Define Training Loop
    for epoch in range(EPOCHS):

        np.random.shuffle(files)
        batches = chunks(files, FLAGS.batch_size)

        for batch in batches:
            Xtrain = []
            ytrain = []
            for fp in batch:
                Xtrain.append(load_img(FLAGS.X_dir+fp, size=input_shape[0:2]))
                ytrain.append(load_img(FLAGS.y_dir+fp, size=output_shape[0:2]))

            Xtrain = np.squeeze(Xtrain)
            ytrain = np.squeeze(ytrain)

            Xtrain = np.transpose(Xtrain, (0, 2, 1, 3))
            ytrain = np.transpose(ytrain, (0, 2, 1, 3))

            if train_GAN:
                discriminator.trainable = False
                generator.trainable = True

                metrics = model.fit( Xtrain, np.ones([len(Xtrain)]) )

                if metrics['accuracy'] > .8:
                    train_GAN = False
            else:
                discriminator.trainable = True
                generator.trainable = False

                inds = np.random.rand(len(Xtrain)) > .5

                # Get generated data from inputs
                gen_input = Xtrain[inds]
                gen_output = generator.predict(gen_input)

                ground_truth = ~inds
                true_output = ytrain[ground_truth]
                disc_input = np.concatenate([gen_output, true_output])

                metrics = discriminator.fit(disc_input, ground_truth)
                if metrics['accuracy'] > .8:
                    train_GAN = True

        out = generator.predict(Xtrain)

        for i in range(len(out)):
            cv2.imwrite( FLAGS.out_dir + '/samples/epoch_{0}_img_{1}_input.png'.format(epoch, i), xtrain[i])
            cv2.imwrite( FLAGS.out_dir + '/samples/epoch_{0}_img_{1}_pred.png'.format(epoch, i), out[i])
            cv2.imwrite( FLAGS.out_dir + '/samples/epoch_{0}_img_{1}_true.png'.format(epoch, i), ytrain[i])

        generator.save(FLAGS.out_dir + '/gen/epoch_{0}.h5py'.format(epoch))


if __name__ == "__main__":
    # Instantiate Argument Parser
    argparse = ArgumentParser()
    argparse.add_argument('--gen_path', type=str, default=None, help='path to generator .h5py')
    argparse.add_argument('--disc_path', type=str, default=None, help='path to discriminator .h5py')
    argparse.add_argument('--out_dir', type=str, default=None, help='path to export directory')
    argparse.add_argument('--X_dir', type=str, default=None, help='path to X data directory')
    argparse.add_argument('--y_dir', type=str, default=None, help='path to y data directory')
    argparse.add_argument('--batch_size', type=int, default=64, help='batch size')

    # Parses known arguments
    FLAGS, unparsed = parser.parse_known_args()

    main()
