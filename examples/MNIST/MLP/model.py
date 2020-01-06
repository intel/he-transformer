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

import numpy as np
import tensorflow as tf
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


def mnist_mlp_model(input):
    y = Conv2D(
        filters=5,
        kernel_size=(5, 5),
        strides=(2, 2),
        padding="same",
        use_bias=True,
        input_shape=(28, 28, 1),
    )(input)
    y = Activation("relu")(y)

    y = MaxPooling2D(pool_size=(3, 3))(y)

    y = Conv2D(
        filters=50,
        kernel_size=(5, 5),
        strides=(2, 2),
        padding="same",
        use_bias=True,
    )(y)
    y = Activation("relu")(y)

    known_shape = y.get_shape()[1:]
    size = np.prod(known_shape)
    print('size', size)

    # Using Keras model API with Flatten results in split ngraph at Flatten() or Reshape() op.
    # Use tf.reshape instead
    y = tf.reshape(y, [-1, size])

    y = Dense(100, use_bias=True)(y)
    y = Activation("relu")(y)
    y = Dense(10, use_bias=True, name="output")(y)

    return y
