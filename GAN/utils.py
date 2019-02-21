import os
from subprocess import call
import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model
import keras.backend as K


def make_trainable(net, val):
    net.trainable = val
    for layer in net.layers:
        layer.trainable = val

def preprocess_input(x):
    """ Normalize an input. """
    x /= 255.
    x -= 0.5
    x *= 2.
    return x

def load_img(img, size):
    """ Load an image with CV2. """
    full_img = cv2.imread(img)
    full_img = cv2.resize(full_img, size)
    full_img = np.expand_dims(full_img, axis=0)
    return full_img

def to_float(x):
    return K.cast(x, "float32")


def chunks(l, n):
    """ Yield successive n-sized chunks from l. """
    for i in range(0, len(l), n):
        yield l[i:i + n]


def root_mean_squared_error(y_true, y_pred):
    """ Define the RMSE loss function. """
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


def process_file(input_filepath, output_dest, model):
    """ Generate smoothed video for the given input file at the given output location, updating progress """


    generator = load_model(model, custom_objects={'root_mean_squared_error':root_mean_squared_error, 'tf': tf, 'output_shape': (480, 852)})
    generator.compile(optimizer='adam', loss=root_mean_squared_error, metrics=['accuracy'])

    # openCV can only output to AVI container formats with the MP4V codec, which is not web-renderable.
    # we output to this format temporarily and then use FFMPEG to convert to the x264 codec later
    output_temp = output_dest + "_temp.avi"


    vidcap = cv2.VideoCapture(input_filepath)
    fps = float(vidcap.get(cv2.CAP_PROP_FPS))

    out = cv2.VideoWriter(output_temp, cv2.VideoWriter_fourcc(
        'm', 'p', '4', 'v'), fps, (852, 480), isColor=True)

    current_frame = None
    success, prev_frame = vidcap.read()
    count = 1

    while success and count<1500:
        success, current_frame = vidcap.read()

        if success:
            
            out_frame = model.predict([current_frame])[0]
            # write both to the final output video
            out.write(out_frame)

            prev_frame = current_frame

            count += 1


    vidcap.release()
    out.release()

    # try to convert video to x264 codec with FFMPEG
    # Ubuntu's FFMPEG installation comes with the x264 encoder, so we don't need to install separately
    if call(["ffmpeg", "-i", output_temp, "-vcodec", "libx264", "-f", "mp4", output_dest]) != 0:
        # if FFMPEG fails, raise error and leave temporary file for debugging purposes
        raise SystemError("FFMPEG command returned with nonzero exit code.")
    else:
        # if successful, delete the temporary file
        print("Deleting temporary file")
        os.remove(output_temp)
