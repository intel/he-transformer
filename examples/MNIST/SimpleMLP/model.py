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
    get_variable, \
    conv2d_stride_2_valid, \
    avg_pool_3x3_same_size, \
    max_pool_3x3_same_size


def mnist_mlp_model(x):
    x = tf.reshape(x, [-1, 28*28])
    W = tf.compat.v1.get_variable('W',[784,10])
    y = tf.matmul(x, W)
    y = tf.nn.relu(y)
    W2 = tf.compat.v1.get_variable('W2',[10,10])
    b2 = tf.compat.v1.get_variable('b2', [1, 10])
    y = tf.add(y, b2)
    y = tf.matmul(y, W2)
    y = tf.nn.relu(y)
    W3 = tf.compat.v1.get_variable('W3',[10,10])
    b3 = tf.compat.v1.get_variable('b3', [1, 10])
    y = tf.add(y, b3)
    y = tf.matmul(y, W3, name='output')
    return y
