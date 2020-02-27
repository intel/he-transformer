# ==============================================================================
#  Copyright 2018-2020 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# ==============================================================================
"""An MNIST classifier using convolutional layers and relu activations. """

import time
import numpy as np
import tensorflow as tf
import ngraph_bridge

# Add parent directory to path
from mnist_util import (
    load_mnist_data,
    server_argument_parser,
    server_config_from_flags,
    load_pb_file,
    print_nodes,
)


def test_network(FLAGS):
    (x_train, y_train, x_test, y_test) = load_mnist_data(
        FLAGS.start_batch, FLAGS.batch_size)

    # Load saved model
    tf.import_graph_def(load_pb_file(FLAGS.model_file))

    print("loaded model")
    print_nodes()

    # Get input / output tensors
    x_input = tf.compat.v1.get_default_graph().get_tensor_by_name(
        FLAGS.input_node)
    y_output = tf.compat.v1.get_default_graph().get_tensor_by_name(
        FLAGS.output_node)

    # Create configuration to encrypt input
    FLAGS, unparsed = server_argument_parser().parse_known_args()
    config = server_config_from_flags(FLAGS, x_input.name)
    with tf.compat.v1.Session(config=config) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        start_time = time.time()
        y_hat = y_output.eval(feed_dict={x_input: x_test})
        elasped_time = time.time() - start_time
        print("total time(s)", np.round(elasped_time, 3))

    if not FLAGS.enable_client:
        y_test_label = np.argmax(y_test, 1)

        if FLAGS.batch_size < 60:
            print("y_hat", np.round(y_hat, 2))

        y_pred = np.argmax(y_hat, 1)
        correct_prediction = np.equal(y_pred, y_test_label)
        error_count = np.size(correct_prediction) - np.sum(correct_prediction)
        test_accuracy = np.mean(correct_prediction)

        print("Error count", error_count, "of", FLAGS.batch_size, "elements.")
        print("Accuracy: %g " % test_accuracy)


if __name__ == "__main__":
    FLAGS, unparsed = server_argument_parser().parse_known_args()

    if unparsed:
        print("Unparsed flags:", unparsed)
        exit(1)
    if FLAGS.encrypt_server_data and FLAGS.enable_client:
        raise Exception(
            "encrypt_server_data flag only valid when client is not enabled. Note: the client can specify whether or not to encrypt the data using 'encrypt' or 'plain' in the configuration map"
        )
    if FLAGS.model_file == "":
        raise Exception("FLAGS.model_file must be set")

    test_network(FLAGS)
