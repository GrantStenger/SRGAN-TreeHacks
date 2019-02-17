# Import Dependencies
import argparse, os
import keras
from keras.models import load_model, Sequential, Model
from keras.layers import Conv2D, Dense, MaxPooling2D, Input
import keras.backend as K
from sklearn.utils import shuffle
from discriminator import create_discriminator
import cv2
import numpy as np
import tensorflow as tf


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

def preprocess_vgg(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x

def load_img(img, size):
    full_img = cv2.imread(img)
    full_img = cv2.resize(full_img, size)
    full_img = np.expand_dims(full_img, axis=0)
    return full_img

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val

def main():
    os.makedirs(FLAGS.out_dir, exist_ok=True)

    # Load Generator
    generator = load_model(FLAGS.gen_path, custom_objects={'root_mean_squared_error':root_mean_squared_error, 'tf': tf, 'output_shape': (480, 852)})
    generator.compile(optimizer='adam', loss=root_mean_squared_error, metrics=['accuracy'])

    # Load Discriminator
    if FLAGS.disc_path is None:
        discriminator = create_discriminator(FLAGS.out_dir, (15, 26, 512))
    else:
        discriminator = load_model(FLAGS.disc_path)

    generator.compile(optimizer='adam', loss=root_mean_squared_error, metrics=['accuracy'])
    discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    make_trainable(discriminator, False)

    # Load VGG
    vgg = keras.applications.VGG16(include_top=False)
    vgg.trainable = False

    # Define Loss Functions
    input_layer = Input(shape=(240, 426, 3))
    out_gen = generator(input_layer)
    out_vgg = vgg(out_gen)
    out_disc = discriminator(out_vgg)

    model = Model(inputs=input_layer, outputs=[out_disc, out_gen])
    model.compile(optimizer='adam', loss='binary_crossentropy', 
                  metrics=['accuracy'], loss_weights=[.8, .2])

    files = os.listdir(FLAGS.X_dir)
    train_gen = False

    # Define Training Loop
    for epoch in range(FLAGS.epochs):
        np.random.shuffle(files)
        batches = chunks(files, FLAGS.batch_size)

        for i, batch in enumerate(batches):
            Xtrain = []
            ytrain = []
            for fp in batch:
                Xtrain.append(load_img(FLAGS.X_dir+fp, size=(240, 426)))
                ytrain.append(load_img(FLAGS.y_dir+fp, size=(480, 852)))

            Xtrain = np.squeeze(Xtrain)
            ytrain = np.squeeze(ytrain)

            Xtrain = np.transpose(Xtrain, (0, 2, 1, 3))
            ytrain = np.transpose(ytrain, (0, 2, 1, 3))

            if train_gen:
                print("Training generator")
                make_trainable(discriminator, False)

                metrics = model.fit( Xtrain, [ np.ones([len(Xtrain)]), ytrain] )
                if metrics.history['discriminator_acc'][0] > .8:
                    train_gen = False
            else:
                print("Training discriminator")
                make_trainable(discriminator, True)

                # Get generated data from inputs
                gen_input = Xtrain
                gen_output = generator.predict(gen_input)

                disc_input = np.concatenate([gen_output, ytrain])
                ground_truth = np.concatenate( [ np.zeros([len(gen_output)]), np.ones([len(ytrain)]) ] )
                disc_input, ground_truth = shuffle(disc_input, ground_truth)


                vgg_output = vgg.predict(disc_input, batch_size=1)

                metrics = discriminator.fit(vgg_output, ground_truth)

                if metrics.history['acc'][0] > .8:
                    train_gen = True

        print("Completed epoch {0} \n \n".format(epoch))
        out = generator.predict(Xtrain)

        os.makedirs(FLAGS.out_dir + '/samples', exist_ok=True)
        for i in range(len(out)):
            cv2.imwrite( FLAGS.out_dir + '/samples/epoch_{0}_img_{1}_input.png'.format(epoch, i), Xtrain[i])
            cv2.imwrite( FLAGS.out_dir + '/samples/epoch_{0}_img_{1}_pred.png'.format(epoch, i), out[i])
            cv2.imwrite( FLAGS.out_dir + '/samples/epoch_{0}_img_{1}_true.png'.format(epoch, i), ytrain[i])

        os.makedirs(FLAGS.out_dir + '/gen', exist_ok=True)
        os.makedirs(FLAGS.out_dir + '/disc', exist_ok=True)
        generator.save(FLAGS.out_dir + '/gen/epoch_{0}.h5py'.format(epoch))
        make_trainable(discriminator, True)
        discriminator.save(FLAGS.out_dir + '/disc/epoch_{0}.h5py'.format(epoch))


if __name__ == "__main__":
    # Instantiate Argument Parser
    argparse = argparse.ArgumentParser()
    argparse.add_argument('--gen_path', type=str, default=None, help='path to generator .h5py')
    argparse.add_argument('--disc_path', type=str, default=None, help='path to discriminator .h5py')
    argparse.add_argument('--out_dir', type=str, default=None, help='path to export directory')
    argparse.add_argument('--X_dir', type=str, default=None, help='path to X data directory')
    argparse.add_argument('--y_dir', type=str, default=None, help='path to y data directory')
    argparse.add_argument('--batch_size', type=int, default=16, help='batch size')
    argparse.add_argument('--epochs', type=int, default=10, help='epochs')

    # Parses known arguments
    FLAGS, unparsed = argparse.parse_known_args()

    main()
