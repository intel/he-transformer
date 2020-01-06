# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""An MNIST classifier based on Cryptonets using convolutional layers. """

import sys
import os
import time
import numpy as np
import model

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import (SGD, RMSprop, Adam, Nadam)
from tensorflow.keras.layers import (
    Dense,
    Conv2D,
    AveragePooling2D,
    Flatten,
    Input,
    Reshape,
)

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mnist_util


def squash_layers(cryptonets_relu_model, sess):
    """Squash linear layers and return squashed weights"""

    layers = cryptonets_relu_model.layers
    layer_names = [layer.name for layer in layers]
    conv1_weights = layers[layer_names.index('conv2d_1')].get_weights()
    conv2_weights = layers[layer_names.index('conv2d_2')].get_weights()
    fc1_weights = layers[layer_names.index('fc_1')].get_weights()
    fc2_weights = layers[layer_names.index('fc_2')].get_weights()

    # Get squashed weight
    y = Input(shape=(14 * 14 * 5,), name="squashed_input")
    y = Reshape((14, 14, 5))(y)
    y = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding="same")(y)
    y = Conv2D(
        filters=50,
        kernel_size=(5, 5),
        strides=(2, 2),
        padding="same",
        use_bias=True,
        kernel_initializer=tf.compat.v1.constant_initializer(conv2_weights[0]),
        bias_initializer=tf.compat.v1.constant_initializer(conv2_weights[1]),
    )(y)
    y = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding="same")(y)
    y = Flatten()(y)
    y = Dense(
        100,
        use_bias=True,
        name="fc_1",
        kernel_initializer=tf.compat.v1.constant_initializer(fc1_weights[0]),
        bias_initializer=tf.compat.v1.constant_initializer(fc1_weights[1]),
    )(y)

    sess.run(tf.compat.v1.global_variables_initializer())

    # Pass 0 to get bias
    squashed_bias = y.eval(
        session=sess,
        feed_dict={
            "squashed_input:0": np.zeros((1, 14 * 14 * 5))
        })
    squashed_bias_plus_weights = y.eval(
        session=sess, feed_dict={
            "squashed_input:0": np.eye(14 * 14 * 5)
        })
    squashed_weights = squashed_bias_plus_weights - squashed_bias

    print("squashed layers")

    # Sanity check
    x_in = np.random.rand(100, 14 * 14 * 5)
    network_out = y.eval(session=sess, feed_dict={"squashed_input:0": x_in})
    linear_out = x_in.dot(squashed_weights) + squashed_bias
    assert np.max(np.abs(linear_out - network_out)) < 1e-3

    return (conv1_weights, (squashed_weights, squashed_bias), fc1_weights,
            fc2_weights)


def main(FLAGS):
    (x_train, y_train, x_test, y_test) = mnist_util.load_mnist_data()

    x = Input(
        shape=(
            28,
            28,
            1,
        ), name="input")
    y = model.cryptonets_relu_model(x)
    cryptonets_relu_model = Model(inputs=x, outputs=y)
    print(cryptonets_relu_model.summary())

    def loss(labels, logits):
        return keras.losses.categorical_crossentropy(
            labels, logits, from_logits=True)

    optimizer = SGD(learning_rate=0.008, momentum=0.9)
    cryptonets_relu_model.compile(
        optimizer=optimizer, loss=loss, metrics=["accuracy"])

    cryptonets_relu_model.fit(
        x_train,
        y_train,
        epochs=FLAGS.epochs,
        validation_data=(x_test, y_test),
        batch_size=FLAGS.batch_size)

    test_loss, test_acc = cryptonets_relu_model.evaluate(
        x_test, y_test, verbose=2)
    print("Test accuracy:", test_acc)

    # Squash weights and save model
    weights = squash_layers(cryptonets_relu_model,
                            tf.compat.v1.keras.backend.get_session())
    (conv1_weights, squashed_weights, fc1_weights, fc2_weights) = weights[0:4]

    tf.reset_default_graph()
    sess = tf.compat.v1.Session()

    x = Input(
        shape=(
            28,
            28,
            1,
        ), name="input")
    y = model.cryptonets_relu_model_squashed(x, conv1_weights, squashed_weights,
                                             fc2_weights)
    sess.run(tf.compat.v1.global_variables_initializer())
    mnist_util.save_model(
        sess,
        ["output/BiasAdd"],
        "./models",
        "cryptonets-relu",
    )


if __name__ == "__main__":
    FLAGS, unparsed = mnist_util.train_argument_parser().parse_known_args()
    if unparsed:
        print("Unparsed flags: ", unparsed)
        exit(1)

    main(FLAGS)
