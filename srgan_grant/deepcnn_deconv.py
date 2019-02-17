#! /usr/bin/python
# -*- coding: utf8 -*-

import os, time, pickle, random, time
from datetime import datetime
import numpy as np
from time import localtime, strftime
import logging, scipy

import tensorflow as tf
import tensorlayer as tl
from model import SRGAN_g, SRGAN_d, Vgg19_simple_api
from utils import *
from config import config, log_config
from keras.preprocessing import image
import cv2

from keras import backend as K
from keras.layers import Conv2D, Dense, Reshape, Lambda, Dropout, UpSampling2D, BatchNormalization, Deconv2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model, Sequential
from keras.optimizers import Adam


def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x

def load_img(img, size):
    full_img = image.load_img(img)
    full_img = image.img_to_array(full_img)
    full_img = cv2.resize(full_img, size)
    full_img = np.expand_dims(full_img, axis=0)
    return full_img

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]



model = Sequential()

def to_float(x):
    return K.cast(x, "float32" )

model.add(Lambda(to_float, input_shape=(240,426,3)))

model.add(Conv2D(10, (1,1), strides=(1,1),  activation=None, padding='SAME' ))
model.add(LeakyReLU())

model.add(Deconv2D(filters=20, kernel_size=(2,2), strides=(2, 2), padding='SAME', ))
model.add(LeakyReLU())
model.add(BatchNormalization())

model.add(Conv2D(3, (2,2), strides=(1,1),  activation='relu', padding='SAME' ))


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

model.compile(optimizer=Adam(), loss=root_mean_squared_error, metrics=['accuracy'])



###====================== HYPER-PARAMETERS ===========================###
## Adam
batch_size = config.TRAIN.batch_size
BATCH_SIZE = 32
lr_init = config.TRAIN.lr_init
beta1 = config.TRAIN.beta1
## initialize G
n_epoch_init = config.TRAIN.n_epoch_init
## adversarial learning (SRGAN)
n_epoch = config.TRAIN.n_epoch
lr_decay = config.TRAIN.lr_decay
decay_every = config.TRAIN.decay_every

ni = int(np.sqrt(batch_size))


def train(model):

    ## create folders to save result images and trained model
    save_dir_ginit = "samples/deepcnn_deconv"
    save_dir_gan = "samples/deepcnn_deconv"
    tl.files.exists_or_mkdir(save_dir_ginit)
    tl.files.exists_or_mkdir(save_dir_gan)
    checkpoint_dir = "checkpoint"  # checkpoint_resize_conv
    tl.files.exists_or_mkdir(checkpoint_dir)

    ###====================== PRE-LOAD DATA FILENAMES ===========================###
    train_hr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.hr_img_path, regx='.*.png', printable=False))
    train_lr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.lr_img_path, regx='.*.png', printable=False))
    valid_hr_img_list = sorted(tl.files.load_file_list(path=config.VALID.hr_img_path, regx='.*.png', printable=False))
    valid_lr_img_list = sorted(tl.files.load_file_list(path=config.VALID.lr_img_path, regx='.*.png', printable=False))

    ###============================= TRAINING ===============================###

    print(" ** fixed learning rate: %f (for init G)" % lr_init)

    for epoch in range(0, n_epoch_init):
        epoch_time = time.time()
        total_mse_loss, n_iter = 0, 0

        random.shuffle(train_hr_img_list)
        batches = chunks(train_hr_img_list, BATCH_SIZE)

        for batch in batches:
            xtrain = [] 
            ytrain = []
            for fp in batch:
                xtrain.append(load_img(config.TRAIN.lr_img_path+fp, size=(240, 426)))
                ytrain.append(load_img(config.TRAIN.hr_img_path+fp, size=(480, 852)))

            step_time = time.time()

            xtrain = np.squeeze(xtrain)
            ytrain = np.squeeze(ytrain)

            xtrain = np.transpose(xtrain, (0, 2, 1, 3))
            ytrain = np.transpose(ytrain, (0, 2, 1, 3))

            model.fit(xtrain, ytrain)

        print("\n\n\n\n DONE WITH REAL EPOCH {0} [*] save images".format(epoch))

        out = model.predict(xtrain) 
        for i in range(len(out)):
            cv2.imwrite( save_dir_ginit + '/mse_epoch_{0}_img_{1}_pred.png'.format(epoch, i), out[i])
            cv2.imwrite( save_dir_ginit + '/mse_epoch_{0}_img_{1}_true.png'.format(epoch, i), ytrain[i])

        # Save Weights
        model.save('weights/epoch_'str(+epoch)+'_deconv.h5py')

        ## quick evaluation on train set
        out = model.predict(xtrain) 
        print("[*] save images")
        for i in range(len(out)):
            cv2.imwrite( save_dir_gan + '/epoch_{0}_img_{1}_pred.png'.format(epoch, i), out[i])
            cv2.imwrite( save_dir_gan + '/epoch_{0}_img_{1}_true.png'.format(epoch, i), ytrain[i])
        
       
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='srgan', help='srgan, evaluate')

    args = parser.parse_args()

    tl.global_flag['mode'] = args.mode

    if tl.global_flag['mode'] == 'srgan':
        train(model)
    else:
        raise Exception("Unknown --mode")
