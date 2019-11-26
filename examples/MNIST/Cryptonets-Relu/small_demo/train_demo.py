import argparse
import sys
import time
import numpy as np
import itertools
import tensorflow as tf
import model
import os
from tensorflow.python.tools import freeze_graph

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mnist_util import load_mnist_data, \
    get_variable, \
    conv2d_stride_2_valid, \
    avg_pool_3x3_same_size, \
    get_train_batch

def cryptonets_relu_test_squashed(x):
    """Constructs test network for Cryptonets Relu using saved weights.
       Assumes linear layers have been squashed."""
    paddings = [[0, 0], [0, 1], [0, 1], [0, 0]]
    x = tf.pad(x, paddings)

    W_conv1 = get_variable('W_conv1', [5, 5, 1, 5], 'test')
    y = conv2d_stride_2_valid(x, W_conv1)
    W_bc1 = get_variable('W_conv1_bias', [1, 13, 13, 5], 'test')
    y = tf.nn.relu(y)

    W_squash = get_variable('W_squash', [5 * 13 * 13, 100], 'test')
    y = tf.reshape(y, [-1, 5 * 13 * 13])
    y = tf.matmul(y, W_squash)
    W_b1 = get_variable('W_fc1_bias', [100], 'test')
    y = y + W_b1

    y = tf.nn.relu(y)
    W_fc2 = get_variable('W_fc2', [100, 10], 'test')
    y = tf.matmul(y, W_fc2)

    W_b2 = get_variable('W_fc2_bias', [10], 'test')
    y = tf.add(y, W_b2, name='output')

    return y



def save_model(directory, filename):
    tf.compat.v1.reset_default_graph()

    with tf.compat.v1.Session() as sess:
        x = tf.compat.v1.placeholder(tf.float32, [None, 28, 28, 1], name='input')
        y_conv = cryptonets_relu_test_squashed(x)

        nodes = [n.name for n in tf.get_default_graph().as_graph_def().node]
        print('nodes', nodes)

        if not os.path.exists(directory):
            os.makedirs(directory)

        #saver = tf.compat.v1.train.Saver()
        #sess.run(tf.global_variables_initializer())
        #ckpt_filepath = os.path.join(directory, filename + '.ckpt')
        #saver.save(sess, ckpt_filepath)

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

        # Freeze graph to turn variables into constants
        #freeze_graph.freeze_graph(
        #    input_graph=pbtxt_filepath,
        #    input_saver='',
        #    input_binary=False,
        #    input_checkpoint=pbtxt_filepath,
        #    output_node_names='output',
        #    restore_op_name='save/restore_all',
        #    filename_tensor_name='save/Const:0',
        #    output_graph=pb_filepath,
        #    clear_devices=True,
        #    initializer_nodes='')

        print("Model saved to: %s" % pb_filepath)


# Squash weights and save as W_squash.txt
def squash_layers():
    print("Squashing layers")
    tf.compat.v1.reset_default_graph()

    # Input from h_conv1 squaring
    x = tf.compat.v1.placeholder(tf.float32, [None, 13, 13, 5])

    # Pooling layer
    h_pool1 = avg_pool_3x3_same_size(x)  # To N x 13 x 13 x 5

    # Second convolution
    W_conv2 = np.loadtxt(
        'W_conv2.txt', dtype=np.float32).reshape([5, 5, 5, 50])
    h_conv2 = conv2d_stride_2_valid(h_pool1, W_conv2)

    # Second pooling layer.
    h_pool2 = avg_pool_3x3_same_size(h_conv2)

    # Fully connected layer 1
    # Input: N x 5 x 5 x 50
    # Output: N x 100
    W_fc1 = np.loadtxt(
        'W_fc1.txt', dtype=np.float32).reshape([5 * 5 * 50, 100])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 5 * 5 * 50])
    pre_square = tf.matmul(h_pool2_flat, W_fc1)

    with tf.compat.v1.Session() as sess:
        x_in = np.eye(13 * 13 * 5)
        x_in = x_in.reshape([13 * 13 * 5, 13, 13, 5])
        W = (sess.run([pre_square], feed_dict={x: x_in}))[0]
        squashed_file_name = "W_squash.txt"
        np.savetxt(squashed_file_name, W)
        print("Saved to", squashed_file_name)

        # Sanity check
        x_in = np.random.rand(100, 13, 13, 5)
        network_out = (sess.run([pre_square], feed_dict={x: x_in}))[0]
        linear_out = x_in.reshape(100, 13 * 13 * 5).dot(W)
        assert (np.max(np.abs(linear_out - network_out)) < 1e-5)

    print("Squashed layers")


def main(FLAGS):
    (x_train, y_train, x_test, y_test) = load_mnist_data()

    x = tf.compat.v1.placeholder(tf.float32, [None, 28, 28, 1])
    y_ = tf.compat.v1.placeholder(tf.float32, [None, 10])
    y_conv = model.cryptonets_relu_model(x, 'train')

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
            _, loss = sess.run([train_step, cross_entropy],
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

        print("Training finished. Saving variables.")
        for var in tf.compat.v1.get_collection(
                tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES):
            weight = (sess.run([var]))[0].flatten().tolist()
            filename = (str(var).split())[1].replace('/', '_')
            filename = filename.replace("'", "").replace(':0', '') + '.txt'

            print("saving", filename)
            np.savetxt(str(filename), weight)

    squash_layers()
    save_model('./model', 'model')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train_loop_count',
        type=int,
        default=20000,
        help='Number of training iterations')
    parser.add_argument(
        '--batch_size', type=int, default=50, help='Batch Size')
    FLAGS, unparsed = parser.parse_known_args()

    main(FLAGS)
