#!/usr/bin/env python3
"""
   _   _                      _ ____  _          _
  | \ | | ___ _   _ _ __ __ _| |  _ \(_)_  _____| |___
  |  \| |/ _ \ | | | '__/ _` | | |_) | \ \/ / _ \ / __|
  | |\  |  __/ |_| | | | (_| | |  __/| |>  <  __/ \__ \\
  |_|_\_|\___|\__,_|_|  \__,_|_|_|   |_/_/\_\___|_|___/

  https://neuralpixels.com/better-style-stansfer
  https://github.com/NeuralPixels/BetterStyleTransfer

  Copywrite 2018 Jaret Burkett
  All Rights Reserved
"""

import os
import argparse
import numpy as np
import tensorflow as tf
from utils import vgg_style
from collections import OrderedDict
from scipy.misc import imread, imsave

parser = argparse.ArgumentParser(
    description='Style Transfer',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
add_arg = parser.add_argument

add_arg('--content', type=str, required=True, help='Path to content image')
add_arg('--style', type=str, required=True, help='Path to style image')
add_arg('--output', type=str, required=True, help='Path to save output image')
add_arg('--max-fun', type=int, default=200, help='Maximum number of functions for L-BFGS-B optimizer (default: 100)')
add_arg('--gpu', type=int, default=0, help='Which GPU to use. (default: 0)')

args = parser.parse_args()

STYLE_LAYERS = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1']

print("""{}""".format(__doc__))


def _convert_to_gram_matrix(inputs):
    batch = tf.shape(inputs)[0]
    height = tf.shape(inputs)[1]
    width = tf.shape(inputs)[2]
    filters = tf.shape(inputs)[3]
    _size = height * width * filters
    size = tf.cast(_size, tf.float32)

    feats = tf.reshape(inputs, (batch, height * width, filters))
    feats_t = tf.transpose(feats, perm=[0, 2, 1])
    grams_raw = tf.matmul(feats_t, feats)
    gram_matrix = tf.divide(grams_raw, size)
    return gram_matrix


def pre_compute_style_features(sess, style):
    with tf.device('/cpu:0'):
        _style = tf.placeholder(tf.float32, shape=(1,) + style.shape, name='style_image')

        network = vgg_style.process(_style)
        _style_features = {}

        for layer in STYLE_LAYERS:
            _style_features[layer] = _convert_to_gram_matrix(network[layer])

        style_features = sess.run(
            _style_features,
            feed_dict={
                _style: np.array([style])
            }
        )
    return style_features


def _style_loss(target_gram_dict=None, pred_net=None):
    total_style_loss = tf.constant(0.0)

    for style_layer in STYLE_LAYERS:
        pred_grams = _convert_to_gram_matrix(pred_net[style_layer])
        target_grams = target_gram_dict[style_layer]

        gram_size = (tf.shape(target_grams)[1] * tf.shape(target_grams)[2])

        # don't sum the batch, keep separate images separate. Not used here, but done if this is used elsewhere
        def seperated_loss(y_pred, y_true):
            sum_axis = [1, 2]
            diff = tf.abs(y_pred - y_true)
            raw_loss = tf.reduce_sum(diff ** 2, axis=sum_axis)
            return raw_loss / tf.cast(gram_size, tf.float32)

        pred_itemized_loss = seperated_loss(pred_grams, target_grams)
        layer_loss = tf.reduce_mean(pred_itemized_loss * vgg_style.layer_weights[style_layer])

        # add this layer loss to the total loss
        total_style_loss += layer_loss

    # return avg layer loss
    return total_style_loss / float(len(STYLE_LAYERS))


if __name__ == '__main__':
    # make only requested GPU available to Tensorflow
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    if args.gpu >= 0:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(args.gpu)

    content_img = imread(args.content, mode='RGB').astype(np.float32)
    style_img = imread(args.style, mode='RGB').astype(np.float32)

    # start a gpu session
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True

    with tf.Session(config=tfconfig) as sess:

        # first pre-compute the style features on the cpu
        print('Pre-computing style features. One moment.')
        style_features = pre_compute_style_features(sess, style_img)

        print('Building the Graph')
        # First, build our graph.
        _output_var = tf.get_variable(
            name='output_var',
            shape=(1,) + content_img.shape,
            dtype=tf.float32
        )

        # apply tanh to the variable. Tanh takes input on -inf to inf and outputs -1 to 1
        # since we are working with 0-255, we need to scale before and after
        _output = (tf.tanh(_output_var / 127.5 - 1.) + 1.) * 127.5

        output_network = vgg_style.process(_output)
        loss = _style_loss(style_features, output_network)

        # We are going to use L-BFGS-B from the scipy package as our optimizer.
        # It runs on the cpu, but is hands down the best optimizer to use in the case.
        # In my tests, it outperforms all native Tensorflow optimizers by a huge margin.
        # It is the magic sauce that makes this work so well.
        optimizer = tf.contrib.opt.ScipyOptimizerInterface(
            loss,
            method='L-BFGS-B',
            var_list=[_output_var],
            options={
                'maxfun': args.max_fun,
                'gtol': 1e-09
            },
        )

        # initialize the variables
        sess.run(tf.global_variables_initializer())

        # Graph is built, time to run it. First, we assign our content to our variable
        sess.run(_output_var.assign(np.array([content_img])))

        print('Stylizing...')

        iteration = 0

        # the scipy interface controls the session and we want to monitor loss, we create
        # a callback function to be called after each iteration
        def loss_callback(*info):
            global iteration
            res = list(info)
            print('\r[{} of {}] loss:{:4.2e}'.format(
                str(iteration).rjust(len(str(args.max_fun)), ' '),
                args.max_fun,
                res[0]
            ), end='', flush=True)
            iteration += 1

        # hand the wheel to the scipy optimizer interface to minimize the loss for us
        optimizer.minimize(
            sess,
            fetches=[loss],
            loss_callback=loss_callback,
        )

        # get the image from the graph
        output = sess.run(_output)

        # We shouldn't need to clip because of the tanh activation, but we will for good measure
        output = np.squeeze(np.clip(output, 0, 255).astype(np.uint8), 0)

        # make sure the output difectory exists
        os.makedirs(os.path.dirname(args.output), exist_ok=True)

        # save the image
        imsave(args.output, output)
        print('\nSaved stylized photo to {}'.format(args.output))
        print('Done!')





