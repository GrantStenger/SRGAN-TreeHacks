#! /usr/bin/python
# -*- coding: utf8 -*-

# Import Dependencies
import os, time, pickle, random, time, argparse, cv2, logging, scipy
from datetime import datetime
import numpy as np
from time import localtime, strftime
import tensorflow as tf
import tensorlayer as tl
from model import SRGAN_g, SRGAN_d, Vgg19_simple_api
from utils import *
from config import config, log_config
from keras.preprocessing import image

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
    #full_img = preprocess_input(full_img)
    return full_img

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


###====================== HYPER-PARAMETERS ===========================###
## Adam
batch_size = config.TRAIN.batch_size
BATCH_SIZE = 8
lr_init = config.TRAIN.lr_init
beta1 = config.TRAIN.beta1
## initialize G
n_epoch_init = config.TRAIN.n_epoch_init
## adversarial learning (SRGAN)
n_epoch = config.TRAIN.n_epoch
lr_decay = config.TRAIN.lr_decay
decay_every = config.TRAIN.decay_every

ni = int(np.sqrt(batch_size))


def train():

    ## create folders to save result images and trained model
    save_dir_ginit = "samples/{}_ginit".format(tl.global_flag['mode'])
    save_dir_gan = "samples/{}_gan".format(tl.global_flag['mode'])
    tl.files.exists_or_mkdir(save_dir_ginit)
    tl.files.exists_or_mkdir(save_dir_gan)
    checkpoint_dir = "checkpoint"  # checkpoint_resize_conv
    tl.files.exists_or_mkdir(checkpoint_dir)

    ###====================== PRE-LOAD DATA FILENAMES ===========================###
    train_hr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.hr_img_path, regx='.*.png', printable=False))
    train_lr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.lr_img_path, regx='.*.png', printable=False))
    valid_hr_img_list = sorted(tl.files.load_file_list(path=config.VALID.hr_img_path, regx='.*.png', printable=False))
    valid_lr_img_list = sorted(tl.files.load_file_list(path=config.VALID.lr_img_path, regx='.*.png', printable=False))

    ###========================== DEFINE MODEL ============================###
    ## train inference
    t_image = tf.placeholder('float32', [None, 240, 426, 3], name='t_image_input_to_SRGAN_generator')
    t_target_image = tf.placeholder('float32', [None, 480, 852, 3], name='t_target_image')

    net_g = SRGAN_g(t_image, is_train=True, reuse=False)
    net_d, logits_real = SRGAN_d(t_target_image, is_train=True, reuse=False)
    _, logits_fake = SRGAN_d(net_g.outputs, is_train=True, reuse=True)

    net_g.print_params(False)
    net_g.print_layers()
    net_d.print_params(False)
    net_d.print_layers()

    ## vgg inference. 0, 1, 2, 3 BILINEAR NEAREST BICUBIC AREA
    t_target_image_224 = tf.image.resize_images(
        t_target_image, size=[224, 224], method=0,
        align_corners=False)  # resize_target_image_for_vgg # http://tensorlayer.readthedocs.io/en/latest/_modules/tensorlayer/layers.html#UpSampling2dLayer
    t_predict_image_224 = tf.image.resize_images(net_g.outputs, size=[224, 224], method=0, align_corners=False)  # resize_generate_image_for_vgg

    net_vgg, vgg_target_emb = Vgg19_simple_api((t_target_image_224 + 1) / 2, reuse=False)
    _, vgg_predict_emb = Vgg19_simple_api((t_predict_image_224 + 1) / 2, reuse=True)

    ## test inference
    net_g_test = SRGAN_g(t_image, is_train=False, reuse=True)

    # ###========================== DEFINE TRAIN OPS ==========================###
    d_loss1 = tl.cost.sigmoid_cross_entropy(logits_real, tf.ones_like(logits_real), name='d1')
    d_loss2 = tl.cost.sigmoid_cross_entropy(logits_fake, tf.zeros_like(logits_fake), name='d2')
    d_loss = d_loss1 + d_loss2

    g_gan_loss = 1e-3 * tl.cost.sigmoid_cross_entropy(logits_fake, tf.ones_like(logits_fake), name='g')
    mse_loss = tl.cost.mean_squared_error(net_g.outputs, t_target_image, is_mean=True)
    vgg_loss = 2e-6 * tl.cost.mean_squared_error(vgg_predict_emb.outputs, vgg_target_emb.outputs, is_mean=True)

    g_loss = mse_loss + vgg_loss + g_gan_loss

    g_vars = tl.layers.get_variables_with_name('SRGAN_g', True, True)
    d_vars = tl.layers.get_variables_with_name('SRGAN_d', True, True)

    with tf.variable_scope('learning_rate'):
        lr_v = tf.Variable(lr_init, trainable=False)
    ## Pretrain
    g_optim_init = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(mse_loss, var_list=g_vars)
    ## SRGAN
    g_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(g_loss, var_list=g_vars)
    d_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(d_loss, var_list=d_vars)

    ###========================== RESTORE MODEL =============================###
    sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    sess = tf.Session(config=sess_config)
    tl.layers.initialize_global_variables(sess)
    if tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/g_{}.npz'.format(tl.global_flag['mode']), network=net_g) is False:
        tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/g_{}_init.npz'.format(tl.global_flag['mode']), network=net_g)
    tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/d_{}.npz'.format(tl.global_flag['mode']), network=net_d)

    ###============================= LOAD VGG ===============================###
    vgg19_npy_path = "vgg19.npy"
    if not os.path.isfile(vgg19_npy_path):
        print("Please download vgg19.npz from : https://github.com/machrisaa/tensorflow-vgg")
        exit()
    npz = np.load(vgg19_npy_path, encoding='latin1').item()

    params = []
    for val in sorted(npz.items()):
        W = np.asarray(val[1][0])
        b = np.asarray(val[1][1])
        print("  Loading %s: %s, %s" % (val[0], W.shape, b.shape))
        params.extend([W, b])
    tl.files.assign_params(sess, params, net_vgg)
    # net_vgg.print_params(False)
    # net_vgg.print_layers()

    ###============================= TRAINING ===============================###

    ###========================= initialize G ====================###
    ## fixed learning rate
    sess.run(tf.assign(lr_v, lr_init))
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

            errM, _ = sess.run([mse_loss, g_optim_init], {t_image: xtrain, t_target_image: ytrain})
            print("Epoch [%2d/%2d] %4d time: %4.4fs, mse: %.8f " % (epoch, n_epoch_init, n_iter, time.time() - step_time, errM))
            total_mse_loss += errM
            n_iter += 1

        log = "[*] Epoch: [%2d/%2d] time: %4.4fs, mse: %.8f" % (epoch, n_epoch_init, time.time() - epoch_time, total_mse_loss / n_iter)
        print(log)

        ## quick evaluation on train set
        out = sess.run(net_g_test.outputs, {t_image: xtrain})  #; print('gen sub-image:', out.shape, out.min(), out.max())
        print("[*] save images")
        for i in range(len(out)):
            cv2.imwrite( save_dir_ginit + '/mse_epoch_{0}_img_{1}_pred.png'.format(epoch, i), out[i])
            cv2.imwrite( save_dir_ginit + '/mse_epoch_{0}_img_{1}_true.png'.format(epoch, i), ytrain[i])

        ## save model
        tl.files.save_npz(net_g.all_params, name=checkpoint_dir + '/g_{}_init.npz'.format(tl.global_flag['mode']), sess=sess)

    ###========================= train GAN (SRGAN) =========================###
    for epoch in range(0, n_epoch + 1):
        ## update learning rate
        if epoch != 0 and (epoch % decay_every == 0):
            new_lr_decay = lr_decay**(epoch // decay_every)
            sess.run(tf.assign(lr_v, lr_init * new_lr_decay))
            log = " ** new learning rate: %f (for GAN)" % (lr_init * new_lr_decay)
            print(log)
        elif epoch == 0:
            sess.run(tf.assign(lr_v, lr_init))
            log = " ** init lr: %f  decay_every_init: %d, lr_decay: %f (for GAN)" % (lr_init, decay_every, lr_decay)
            print(log)

        epoch_time = time.time()
        total_d_loss, total_g_loss, n_iter = 0, 0, 0

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

            ## update D
            errD, _ = sess.run([d_loss, d_optim], {t_image: xtrain, t_target_image: ytrain})
            ## update G
            errG, errM, errV, errA, _ = sess.run([g_loss, mse_loss, vgg_loss, g_gan_loss, g_optim], {t_image: xtrain, t_target_image: ytrain})
            print("Epoch [%2d/%2d] %4d time: %4.4fs, d_loss: %.8f g_loss: %.8f (mse: %.6f vgg: %.6f adv: %.6f)" %
                  (epoch, n_epoch, n_iter, time.time() - step_time, errD, errG, errM, errV, errA))
            total_d_loss += errD
            total_g_loss += errG
            n_iter += 1

        log = "[*] Epoch: [%2d/%2d] time: %4.4fs, d_loss: %.8f g_loss: %.8f" % (epoch, n_epoch, time.time() - epoch_time, total_d_loss / n_iter,
                                                                                total_g_loss / n_iter)
        print(log)

        ## quick evaluation on train set

        out = sess.run(net_g_test.outputs, {t_image: xtrain})  #; print('gen sub-image:', out.shape, out.min(), out.max())
        print("[*] save images")
        for i in range(len(out)):
            cv2.imwrite( save_dir_gan + '/epoch_{0}_img_{1}_pred.png'.format(epoch, i), out[i])
            cv2.imwrite( save_dir_gan + '/epoch_{0}_img_{1}_true.png'.format(epoch, i), ytrain[i])

        ## save model
        tl.files.save_npz(net_g.all_params, name=checkpoint_dir + '/g_{}.npz'.format(tl.global_flag['mode']), sess=sess)
        tl.files.save_npz(net_d.all_params, name=checkpoint_dir + '/d_{}.npz'.format(tl.global_flag['mode']), sess=sess)


def evaluate():
    ## create folders to save result images
    save_dir = "samples/{}".format(tl.global_flag['mode'])
    tl.files.exists_or_mkdir(save_dir)
    checkpoint_dir = "checkpoint"

    ###====================== PRE-LOAD DATA ===========================###
    # train_hr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.hr_img_path, regx='.*.png', printable=False))
    # train_lr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.lr_img_path, regx='.*.png', printable=False))
    valid_hr_img_list = sorted(tl.files.load_file_list(path=config.VALID.hr_img_path, regx='.*.png', printable=False))
    valid_lr_img_list = sorted(tl.files.load_file_list(path=config.VALID.lr_img_path, regx='.*.png', printable=False))

    ## If your machine have enough memory, please pre-load the whole train set.
    # train_hr_imgs = tl.vis.read_images(train_hr_img_list, path=config.TRAIN.hr_img_path, n_threads=32)
    # for im in train_hr_imgs:
    #     print(im.shape)
    valid_lr_imgs = tl.vis.read_images(valid_lr_img_list, path=config.VALID.lr_img_path, n_threads=32)
    # for im in valid_lr_imgs:
    #     print(im.shape)
    valid_hr_imgs = tl.vis.read_images(valid_hr_img_list, path=config.VALID.hr_img_path, n_threads=32)
    # for im in valid_hr_imgs:
    #     print(im.shape)
    # exit()

    ###========================== DEFINE MODEL ============================###
    imid = 64  # 0: 企鹅  81: 蝴蝶 53: 鸟  64: 古堡
    valid_lr_img = valid_lr_imgs[imid]
    valid_hr_img = valid_hr_imgs[imid]
    # valid_lr_img = get_imgs_fn('test.png', 'data2017/')  # if you want to test your own image
    valid_lr_img = (valid_lr_img / 127.5) - 1  # rescale to ［－1, 1]
    # print(valid_lr_img.min(), valid_lr_img.max())

    size = valid_lr_img.shape
    # t_image = tf.placeholder('float32', [None, size[0], size[1], size[2]], name='input_image') # the old version of TL need to specify the image size
    t_image = tf.placeholder('float32', [1, None, None, 3], name='input_image')

    net_g = SRGAN_g(t_image, is_train=False, reuse=False)

    ###========================== RESTORE G =============================###
    sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    sess = tf.Session(config=sess_config)
    tl.layers.initialize_global_variables(sess)
    tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/g_srgan.npz', network=net_g)

    ###======================= EVALUATION =============================###
    start_time = time.time()
    out = sess.run(net_g.outputs, {t_image: [valid_lr_img]})
    print("took: %4.4fs" % (time.time() - start_time))

    print("LR size: %s /  generated HR size: %s" % (size, out.shape))  # LR size: (339, 510, 3) /  gen HR size: (1, 1356, 2040, 3)
    print("[*] save images")
    tl.vis.save_image(out[0], save_dir + '/valid_gen.png')
    tl.vis.save_image(valid_lr_img, save_dir + '/valid_lr.png')
    tl.vis.save_image(valid_hr_img, save_dir + '/valid_hr.png')

    out_bicu = scipy.misc.imresize(valid_lr_img, [size[0] * 4, size[1] * 4], interp='bicubic', mode=None)
    tl.vis.save_image(out_bicu, save_dir + '/valid_bicubic.png')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='srgan', help='srgan, evaluate')

    args = parser.parse_args()

    print(args)

    # tl.global_flag['mode'] = args.mode
    #
    # if tl.global_flag['mode'] == 'srgan':
    #     train()
    # elif tl.global_flag['mode'] == 'evaluate':
    #     evaluate()
    # else:
    #     raise Exception("Unknown --mode")
