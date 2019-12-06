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

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Activation, AveragePooling2D, Flatten, Convolution2D, MaxPooling2D

def cryptonets_model():
    def square_activation(x):
        return x * x

    model = Sequential()
    model.add(Conv2D(filters=5,
                    kernel_size=(5, 5),
                    strides=(2, 2),
                    padding='same',
                    use_bias=True,
                    input_shape=(28, 28, 1),
                    name='conv2d_1'))

    model.add(Activation(square_activation))
    model.add(AveragePooling2D(pool_size=(3, 3),
                                strides=(1, 1),
                                padding='same'))
    model.add(Conv2D(filters=50,
                                kernel_size=(5, 5),
                                strides=(2, 2),
                                padding='same',
                                use_bias=True,
                                name='conv2d_2'))

    model.add(AveragePooling2D(pool_size=(3, 3),
                                strides=(1, 1),
                                padding='same'))
    model.add(Flatten())
    model.add(Dense(100, use_bias=True, name='fc_1'))
    model.add(Activation(square_activation))
    model.add(Dense(10, use_bias=True, name='fc_2'))

    return model


def cryptonets_model_squashed(conv1_weights, squashed_weights, fc2_weights):
    def square_activation(x):
        return x * x

    print('conv1_weights', conv1_weights[0].shape, conv1_weights[1].shape)
    print('squashed_weights', squashed_weights[0].shape, squashed_weights[1].shape)
    print('fc2_weights', fc2_weights[0].shape, fc2_weights[1].shape)

    model = Sequential()
    model.add(Conv2D(filters=5,
                    kernel_size=(5, 5),
                    strides=(2, 2),
                    padding='same',
                    use_bias=True,
                    kernel_initializer=tf.compat.v1.constant_initializer(conv1_weights[0]),
                    bias_initializer=tf.compat.v1.constant_initializer(conv1_weights[1]),
                    input_shape=(28, 28, 1),
                    trainable=False,
                    name='convd1_1'))

    model.add(Activation(square_activation))
    model.add(Flatten())
    model.add(Dense(100,
                    use_bias=True,
                    name='squash_fc_1',
                    trainable=False,
                    kernel_initializer=tf.compat.v1.constant_initializer(squashed_weights[0]),
                    bias_initializer=tf.compat.v1.constant_initializer(squashed_weights[1]))
                    )

    model.add(Activation(square_activation))
    model.add(Dense(10, use_bias=True,
                    trainable=False,
                    kernel_initializer=tf.compat.v1.constant_initializer(fc2_weights[0]),
                    bias_initializer=tf.compat.v1.constant_initializer(fc2_weights[1]),
                    name='output'))

    return model