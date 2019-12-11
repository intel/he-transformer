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
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import SGD, RMSprop, Adam, Nadam
from tensorflow.keras.losses import categorical_crossentropy

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mnist_util
import model


def main(FLAGS):
    (x_train, y_train, x_test, y_test) = mnist_util.load_mnist_data()

    x = Input(
        shape=(
            28,
            28,
            1,
        ), name="input")
    y = model.mnist_mlp_model(x)

    mlp_model = Model(inputs=x, outputs=y)
    print(mlp_model.summary())

    def loss(labels, logits):
        return categorical_crossentropy(labels, logits, from_logits=True)

    optimizer = SGD(learning_rate=0.008, momentum=0.9)
    mlp_model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

    mlp_model.fit(
        x_train,
        y_train,
        epochs=FLAGS.epochs,
        batch_size=FLAGS.batch_size,
        validation_data=(x_test, y_test),
        verbose=1)

    test_loss, test_acc = mlp_model.evaluate(x_test, y_test, verbose=1)
    print("\nTest accuracy:", test_acc)

    mnist_util.save_model(
        tf.compat.v1.keras.backend.get_session(),
        ["output/BiasAdd"],
        "./models",
        "mlp",
    )


if __name__ == "__main__":
    FLAGS, unparsed = mnist_util.train_argument_parser().parse_known_args()
    if unparsed:
        print("Unparsed flags: ", unparsed)
        exit(1)

    main(FLAGS)
