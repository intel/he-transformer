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
from tensorflow import keras
from tensorflow.python.tools import freeze_graph

import model
import os
from tensorflow.keras import backend as K

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mnist_util import load_mnist_data, save_model, train_argument_parser

from tensorflow.keras.layers import Dense, Conv2D, Activation, AveragePooling2D, Flatten, Convolution2D, MaxPooling2D, Input, Reshape
from tensorflow.keras.models import load_model




# Squash linear layers and return squashed weights
def squash_layers(model, sess):

    nodes = [n.name for n in tf.get_default_graph().as_graph_def().node]
    nodes = [n for n in nodes if n[0:4] != 'Adam']
    nodes = [n for n in nodes if n[0:4] != 'loss']
    nodes = [n for n in nodes if n[0:7] != 'metrics']

    print('nodes', nodes)
    print('model.layers', model.layers)

    # Get weights of second convolution
    W_conv2 = model.layers[3].get_weights()
    conv2_weights = W_conv2[0]
    conv2_bias = W_conv2[1]
    print('weights', conv2_weights.shape)
    print('bias', conv2_bias.shape)

    # FC1 weights
    W_fc1 = model.layers[6].get_weights()
    fc1_weights = W_fc1[0]
    fc1_bias = W_fc1[1]
    print('fc1_weights', fc1_weights.shape)
    print('fc1_bias', fc1_bias.shape)

    # Get squashed weight
    y = Input(shape=(14 * 14 * 5,), name='input')
    y = Reshape((14, 14, 5))(y)
    y = AveragePooling2D(pool_size=(3, 3),
                                strides=(1, 1),
                                padding='same')(y)
    print('y', y)
    y = Conv2D(filters=50,
                kernel_size=(5, 5),
                strides=(2, 2),
                padding='same',
                use_bias=True,
                trainable=False,
                bias_initializer=tf.compat.v1.constant_initializer(conv2_bias),
                kernel_initializer=tf.compat.v1.constant_initializer(conv2_weights),
                name='conv2d_test')(y)
    y = AveragePooling2D(pool_size=(3, 3),
                          strides=(1, 1),
                          padding='same')(y)
    y = Flatten()(y)
    y = Dense(100, use_bias=True, name='fc_1',
            bias_initializer=tf.compat.v1.constant_initializer(fc1_bias),
            kernel_initializer=tf.compat.v1.constant_initializer(fc1_weights))(y)

    sess.run(tf.compat.v1.global_variables_initializer())

    # Pass 0 to get bias
    squashed_bias = y.eval(session=sess, feed_dict={'input:0': np.zeros((1, 14 * 14 * 5))})
    squashed_bias_plus_weights = y.eval(session=sess, feed_dict={'input:0': np.eye(14 * 14 * 5)})
    squashed_weights = squashed_bias_plus_weights - squashed_bias

    print('squashed layers')

    # Sanity check
    x_in = np.random.rand(100, 14 * 14 * 5)
    network_out = y.eval(session=sess, feed_dict={'input:0': x_in})
    linear_out = x_in.dot(squashed_weights) + squashed_bias
    assert (np.max(np.abs(linear_out - network_out)) < 1e-3)
    print('check passed!')

    return squashed_weights, squashed_bias


def save_model(cryptonets_model, sess, directory, filename):
    squashed_weights, squashed_bias = squash_layers(cryptonets_model, sess)

    squashed_model = model.cryptonets_model_squashed(squashed_weights, squashed_bias)
    print('squashed_model', squashed_model.summary())
    sess.run(tf.compat.v1.global_variables_initializer())

    #squashed_model.save('./tmp.h5')



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

    print('write graph to', filename + '.pb')

    tf.io.write_graph(
        graph_or_graph_def=sess.graph_def,
        logdir=directory,
        name=pbtxt_filename,
        as_text=True)

    print('write graph to', pbtxt_filename)

    nodes = [n.name for n in tf.get_default_graph().as_graph_def().node]
    print('nodes', nodes)

    saver = tf.compat.v1.train.Saver()
    ckpt_filepath = os.path.join(directory, filename + '.ckpt')
    saver.save(sess, ckpt_filepath)

    print('saved graph to ', filename + '.ckpt')

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

    cryptonets = model.cryptonets_model()

    print(cryptonets.summary())

    print('x_train', x_train.shape)
    print('y_train', y_train.shape)
    print('x_test', x_test.shape)
    print('y_test', y_test.shape)

    def my_loss(labels, logits):
        return keras.losses.categorical_crossentropy(labels, logits, from_logits=True)

    cryptonets.compile(optimizer='adam',
              loss=my_loss,
              metrics=['accuracy'])

    cryptonets.fit(x_train, y_train, epochs=FLAGS.epochs, batch_size=FLAGS.batch_size, verbose=1)


    test_loss, test_acc = cryptonets.evaluate(x_test, y_test, verbose=2)
    print('\nTest accuracy:', test_acc)

    save_model(cryptonets, tf.compat.v1.keras.backend.get_session(), './models', 'larger-cryptonets-relu')


if __name__ == '__main__':
    FLAGS, unparsed = train_argument_parser().parse_known_args()
    if unparsed:
        print("Unparsed flags: ", unparsed)
        exit(1)

    main(FLAGS)