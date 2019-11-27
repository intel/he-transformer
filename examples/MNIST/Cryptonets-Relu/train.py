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
import time
import numpy as np
import tensorflow as tf
import model
import os
from tensorflow.python.tools import freeze_graph

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mnist_util import load_mnist_data, \
    conv2d_stride_2_valid, \
    avg_pool_3x3_same_size, \
    get_train_batch, \
    train_argument_parser


# Squash linear layers and return squashed weights
def squash_layers(sess):
    # Input from first relu layer
    x = tf.compat.v1.placeholder(tf.float32, [None, 13, 13, 5])
    y = avg_pool_3x3_same_size(x)
    W_conv2 = tf.compat.v1.get_default_graph().get_tensor_by_name("W_conv2:0")
    y = conv2d_stride_2_valid(y, W_conv2)
    y = avg_pool_3x3_same_size(y)

    W_fc1 = tf.compat.v1.get_default_graph().get_tensor_by_name("W_fc1:0")
    y = tf.reshape(y, [-1, 5 * 5 * 50])
    y = tf.matmul(y, W_fc1)

    x_in = np.eye(13 * 13 * 5)
    x_in = x_in.reshape([13 * 13 * 5, 13, 13, 5])
    squashed_weight = (sess.run([y], feed_dict={x: x_in}))[0]

    # Sanity check
    x_in = np.random.rand(100, 13, 13, 5)
    network_out = (sess.run([y], feed_dict={x: x_in}))[0]
    linear_out = x_in.reshape(100, 13 * 13 * 5).dot(squashed_weight)
    assert (np.max(np.abs(linear_out - network_out)) < 1e-5)

    print('squashed layers')

    return squashed_weight


def save_model(sess, directory, filename):
    squashed_weight = squash_layers(sess)

    x = tf.compat.v1.placeholder(tf.float32, [None, 28, 28, 1], name='input')
    y_conv = model.cryptonets_relu_squashed(x, squashed_weight)

    if not os.path.exists(directory):
        os.makedirs(directory)

    pbtxt_filename = filename + '.pbtxt'
    pbtxt_filepath = os.path.join(directory, pbtxt_filename)
    pb_filepath = os.path.join(directory, filename + '.pb')

    tf.io.write_graph(
        graph_or_graph_def=sess.graph_def,
        logdir=directory,
        name=filename + '.pb',
        as_text=False)

    tf.io.write_graph(
        graph_or_graph_def=sess.graph_def,
        logdir=directory,
        name=pbtxt_filename,
        as_text=True)

    saver = tf.compat.v1.train.Saver()
    ckpt_filepath = os.path.join(directory, filename + '.ckpt')
    saver.save(sess, ckpt_filepath)

    # Freeze graph to turn variables into constants
    freeze_graph.freeze_graph(
        input_graph=pbtxt_filepath,
        input_saver='',
        input_binary=False,
        input_checkpoint=ckpt_filepath,
        output_node_names='output',
        restore_op_name='save/restore_all',
        filename_tensor_name='save/Const:0',
        output_graph=pb_filepath,
        clear_devices=True,
        initializer_nodes='')

    print("Model saved to: %s" % pb_filepath)


def main(FLAGS):
    (x_train, y_train, x_test, y_test) = load_mnist_data()

    x = tf.compat.v1.placeholder(tf.float32, [None, 28, 28, 1])
    y_ = tf.compat.v1.placeholder(tf.float32, [None, 10])
    y_conv = model.cryptonets_relu_model(x)

    with tf.name_scope('loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=y_, logits=y_conv)
    cross_entropy = tf.reduce_mean(cross_entropy)

    with tf.name_scope('adam_optimizer'):
        train_step = tf.compat.v1.train.AdamOptimizer(1e-4).minimize(
            cross_entropy)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        for i in range(FLAGS.train_loop_count):
            x_batch, y_batch = get_train_batch(i, FLAGS.batch_size, x_train,
                                               y_train)
            if i % 100 == 0:
                t = time.time()
                train_accuracy = accuracy.eval(feed_dict={
                    x: x_batch,
                    y_: y_batch
                })
                print('step %d, training accuracy %g, %g msec to evaluate' %
                      (i, train_accuracy, 1000 * (time.time() - t)))
            t = time.time()
            sess.run([train_step, cross_entropy],
                     feed_dict={
                         x: x_batch,
                         y_: y_batch
                     })

            if i % 1000 == 999 or i == FLAGS.train_loop_count - 1:
                test_accuracy = accuracy.eval(feed_dict={
                    x: x_test,
                    y_: y_test
                })
                print('test accuracy %g' % test_accuracy)

        print("Training finished. Saving model.")
        save_model(sess, './models', 'cryptonets-relu')


if __name__ == '__main__':
    FLAGS, unparsed = train_argument_parser().parse_known_args()
    if unparsed:
        print("Unparsed flags: ", unparsed)
        exit(1)
    main(FLAGS)
