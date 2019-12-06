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
from mnist_util import conv2d_stride_2_valid, avg_pool_3x3_same_size


def cryptonets_relu_model(x):
    W_conv1 = tf.compat.v1.get_variable('W_conv1', [5, 5, 1, 5])
    W_conv1_bias = tf.compat.v1.get_variable('W_conv1_bias', [1, 12, 12, 5])
    y = tf.layers.conv2d(x, filters=5,
                            kernel_size=(5,5),
                            use_bias=False,
                            strides=(2,2),
                            padding='same',
                            name='conv2d_1')
    y = tf.nn.relu(y)

    y = avg_pool_3x3_same_size(y)
    W_conv2 = tf.compat.v1.get_variable('W_conv2', [5, 5, 5, 50])
    y = tf.layers.conv2d(y, filters=50,
                            kernel_size=(5,5),
                            use_bias=False,
                            strides=(2,2),
                            padding='same',
                            name='conv2d_2')

    y = avg_pool_3x3_same_size(y)

    y = tf.reshape(y, [-1, 7 * 7 * 50])
    W_fc1 = tf.compat.v1.get_variable('W_fc1', [7 * 7 * 50, 100])
    W_fc1_bias = tf.compat.v1.get_variable('W_fc1_bias', [100])
    y = tf.matmul(y, W_fc1)
    y = y + W_fc1_bias
    y = tf.nn.relu(y)

    W_fc2 = tf.compat.v1.get_variable('W_fc2', [100, 10])
    W_fc2_bias = tf.compat.v1.get_variable('W_fc2_bias', [10])
    y = tf.matmul(y, W_fc2) + W_fc2_bias
    return y


def cryptonets_relu_squashed(x, squashed_weight):
    """Constructs test network for Cryptonets Relu using squashed weights."""
    W_conv1 = tf.compat.v1.get_default_graph().get_tensor_by_name("W_conv1:0")
    W_conv1_bias = tf.compat.v1.get_default_graph().get_tensor_by_name(
        "W_conv1_bias:0")
    y = tf.layers.conv2d(x, filters=5,
                            kernel_size=(5,5),
                            kernel_initializer=tf.compat.v1.constant_initializer(squashed_weight),
                            use_bias=False,
                            strides=(2,2),
                            padding='same',
                            name='conv2d_1')
    y = tf.nn.relu(y)

    W_squash = tf.constant(
        squashed_weight, dtype=np.float32, shape=[5 * 12 * 12, 100])
    y = tf.reshape(y, [-1, 5 * 12 * 12])
    y = tf.matmul(y, W_squash)
    W_fc1_bias = tf.compat.v1.get_default_graph().get_tensor_by_name(
        "W_fc1_bias:0")
    y = y + W_fc1_bias

    y = tf.nn.relu(y)
    W_fc2 = tf.compat.v1.get_default_graph().get_tensor_by_name("W_fc2:0")
    y = tf.matmul(y, W_fc2)

    W_fc2_bias = tf.compat.v1.get_default_graph().get_tensor_by_name(
        "W_fc2_bias:0")
    y = tf.add(y, W_fc2_bias, name='output')

    return y
