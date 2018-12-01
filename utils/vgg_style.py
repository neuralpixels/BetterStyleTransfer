import tensorflow as tf
import numpy as np
from collections import OrderedDict
import os
import json


# these weights have been trained over thousands of images. They are designed to be multiplied by the loss
# of each layer to normalize the loss differential from layer to layer

layer_weights = {
  "conv1_1": 0.27844879031181335,
  "conv1_2": 0.0004943962558172643,
  "conv2_1": 0.0009304438135586679,
  "conv2_2": 0.00040253016049973667,
  "conv3_1": 0.0001156232028733939,
  "conv3_2": 7.009495311649516e-05,
  "conv3_3": 7.687774996156804e-06,
  "conv3_4": 8.033587732825254e-07,
  "conv4_1": 5.199814836487349e-07
}

_weights_vgg_style = None


def _load_weights():
    weights_folder = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 'weights'
    )
    weight_dict = {}
    manifest_path = os.path.join(weights_folder, 'manifest.json')
    with open(manifest_path) as f:
        manifest = json.load(f)
        for weight_name, weight_obj in manifest.items():
            weight_file_path = os.path.join(weights_folder, weight_obj['filename'])
            with open(weight_file_path, "rb") as binary_file:
                # Read the whole file at once
                data = binary_file.read()
                weight_np = np.frombuffer(data, dtype=np.float32)
                weights_reshaped = np.reshape(weight_np, tuple(weight_obj['shape']))
                weights = tf.constant(
                    weights_reshaped, dtype=tf.float32, shape=weight_obj['shape'],
                    name='{}/{}'.format('vgg_style', weight_name)
                )
                weight_dict[weight_name] = weights
    return weight_dict


def process(input_tensor, network=None):

    layers = [
        'conv1_1', 'conv1_2', 'pool1',
        'conv2_1', 'conv2_2', 'pool2',
        'conv3_1', 'conv3_2', 'conv3_3', 'conv3_4', 'pool3',
        'conv4_1', 'relu4_1'
    ]

    if network is None:
        network = OrderedDict()

    def _conv_layer(inputs, kernel_weights, bias_weights):
        conv_out = tf.nn.conv2d(
            input=inputs,
            filter=kernel_weights,
            strides=(1, 1, 1, 1),
            padding='SAME'
        )
        bias_added = tf.nn.bias_add(conv_out, bias_weights)
        return tf.nn.relu(bias_added)

    def _pool_layer(inputs):
        return tf.nn.max_pool(
            value=inputs,
            ksize=(1, 2, 2, 1),
            strides=(1, 2, 2, 1),
            padding='SAME'
        )

    def get_weights():
        global _weights_vgg_style
        if _weights_vgg_style is None:
            _weights_vgg_style = _load_weights()
        return _weights_vgg_style

    r, g, b = tf.split(axis=-1, num_or_size_splits=3, value=input_tensor)
    mean_pixel = [103.939, 116.779, 123.68]
    inputs = tf.concat(values=[b - mean_pixel[0], g - mean_pixel[1], r - mean_pixel[2]], axis=-1)

    network['vgg_input'] = inputs
    weights = get_weights()

    current = network['vgg_input']
    for name in layers:
        kind = name[:4]
        if kind == 'conv':
            kernels = weights['{}/kernel'.format(name)]
            bias = weights['{}/bias'.format(name)]
            current = _conv_layer(current, kernels, bias)
            network['{}'.format(name)] = current
        elif kind == 'pool':
            current = _pool_layer(current)
            network['{}'.format(name)] = current

    # return the network
    return network
