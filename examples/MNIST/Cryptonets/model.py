# ******************************************************************************
# Copyright 2018-2020 Intel Corporation
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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Conv2D,
    Activation,
    AveragePooling2D,
    Flatten,
    Convolution2D,
    MaxPooling2D,
    Reshape,
)


def cryptonets_model(input):

    def square_activation(x):
        return x * x

    y = Conv2D(
        filters=5,
        kernel_size=(5, 5),
        strides=(2, 2),
        padding="same",
        use_bias=True,
        input_shape=(28, 28, 1),
        name="conv2d_1",
    )(input)
    y = Activation(square_activation)(y)

    y = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding="same")(y)
    y = Conv2D(
        filters=50,
        kernel_size=(5, 5),
        strides=(2, 2),
        padding="same",
        use_bias=True,
        name="conv2d_2",
    )(y)
    y = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding="same")(y)
    y = Flatten()(y)
    y = Dense(100, use_bias=True, name="fc_1")(y)
    y = Activation(square_activation)(y)
    y = Dense(10, use_bias=True, name="fc_2")(y)

    return y


def cryptonets_model_squashed(input, conv1_weights, squashed_weights,
                              fc2_weights):

    def square_activation(x):
        return x * x

    y = Conv2D(
        filters=5,
        kernel_size=(5, 5),
        strides=(2, 2),
        padding="same",
        use_bias=True,
        kernel_initializer=tf.compat.v1.constant_initializer(conv1_weights[0]),
        bias_initializer=tf.compat.v1.constant_initializer(conv1_weights[1]),
        input_shape=(28, 28, 1),
        trainable=False,
        name="convd1_1",
    )(input)
    y = Activation(square_activation)(y)

    # Using Keras model API with Flatten results in split ngraph at Flatten() or Reshape() op.
    # Use tf.reshape instead
    y = tf.reshape(y, [-1, 5 * 14 * 14])
    y = Dense(
        100,
        use_bias=True,
        name="squash_fc_1",
        trainable=False,
        kernel_initializer=tf.compat.v1.constant_initializer(
            squashed_weights[0]),
        bias_initializer=tf.compat.v1.constant_initializer(squashed_weights[1]),
    )(y)
    y = Activation(square_activation)(y)

    y = Dense(
        10,
        use_bias=True,
        trainable=False,
        kernel_initializer=tf.compat.v1.constant_initializer(fc2_weights[0]),
        bias_initializer=tf.compat.v1.constant_initializer(fc2_weights[1]),
        name="output",
    )(y)

    return y
