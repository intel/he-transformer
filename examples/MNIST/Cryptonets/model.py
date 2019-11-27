# ******************************************************************************
# Copyright 2018-2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# *****************************************************************************

import tensorflow as tf
import numpy as np
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mnist_util import load_mnist_data, \
    conv2d_stride_2_valid, \
    avg_pool_3x3_same_size


def cryptonets_model(x):
    """Builds the graph for classifying digits based on Cryptonets

    Args:
        x: an input tensor with the dimensions (N_examples, 28, 28)

    Returns:
        A tuple (y, a scalar placeholder). y is a tensor of shape
        (N_examples, 10), with values equal to the logits of classifying the
        digit into one of 10 classes (the digits 0-9).
    """

    # Reshape to use within a conv neural net.
    # Last dimension is for "features" - there is only one here, since images
    # are grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
    # CryptoNets's output of the first conv layer has feature map size 13 x 13,
    # therefore, we manually add paddings.
    with tf.name_scope('reshape'):
        print('padding')
        paddings = [[0, 0], [0, 1], [0, 1], [0, 0]]
        x = tf.pad(x, paddings)
        print('padded')

    # First conv layer
    # Input: N x 28 x 28 x 1
    # Filter: 5 x 5 x 1 x 5
    # Output: N x 13 x 13 x 5
    with tf.name_scope('conv1'):
        W_conv1 = tf.compat.v1.get_variable("W_conv1", [5, 5, 1, 5])
        h_conv1 = tf.square(conv2d_stride_2_valid(x, W_conv1))

    # Pooling layer
    # Input: N x 13 x 13 x 5
    # Output: N x 13 x 13 x 5
    with tf.name_scope('pool1'):
        h_pool1 = avg_pool_3x3_same_size(h_conv1)

    # Second convolution
    # Input: N x 13 x 13 x 5
    # Filter: 5 x 5 x 5 x 50
    # Output: N x 5 x 5 x 50
    with tf.name_scope('conv2'):
        W_conv2 = tf.compat.v1.get_variable("W_conv2", [5, 5, 5, 50])
        h_conv2 = conv2d_stride_2_valid(h_pool1, W_conv2)

    # Second pooling layer
    # Input: N x 5 x 5 x 50
    # Output: N x 5 x 5 x 50
    with tf.name_scope('pool2'):
        h_pool2 = avg_pool_3x3_same_size(h_conv2)

    # Fully connected layer 1
    # Input: N x 5 x 5 x 50
    # Input flattened: N x 1250
    # Weight: 1250 x 100
    # Output: N x 100
    with tf.name_scope('fc1'):
        h_pool2_flat = tf.reshape(h_pool2, [-1, 5 * 5 * 50])
        W_fc1 = tf.compat.v1.get_variable("W_fc1", [5 * 5 * 50, 100])
        h_fc1 = tf.square(tf.matmul(h_pool2_flat, W_fc1))

    # Map the 100 features to 10 classes, one for each digit
    # Input: N x 100
    # Weight: 100 x 10
    # Output: N x 10
    with tf.name_scope('fc2'):
        W_fc2 = tf.compat.v1.get_variable("W_fc2", [100, 10])
        y_conv = tf.matmul(h_fc1, W_fc2)
    return y_conv


def cryptonets_squashed(x, squashed_weight):
    """Constructs test network for Cryptonets Relu using squashed weights."""
    paddings = [[0, 0], [0, 1], [0, 1], [0, 0]]
    x = tf.pad(x, paddings)

    W_conv1 = tf.compat.v1.get_default_graph().get_tensor_by_name("W_conv1:0")
    y = conv2d_stride_2_valid(x, W_conv1)
    y = tf.square(y)

    W_squash = tf.constant(squashed_weight, dtype=np.float32, shape=[5 * 13 * 13, 100])
    y = tf.reshape(y, [-1, 5 * 13 * 13])
    y = tf.matmul(y, W_squash)
    y = tf.square(y)

    W_fc2 = tf.compat.v1.get_default_graph().get_tensor_by_name("W_fc2:0")
    y = tf.matmul(y, W_fc2, name='output')

    return y

