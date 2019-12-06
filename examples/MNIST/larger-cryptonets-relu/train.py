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
from tensorflow.python.keras.utils import CustomObjectScope

import model
import os
from tensorflow.keras import backend as K

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mnist_util import load_mnist_data, save_model, train_argument_parser, print_nodes

from tensorflow.keras.layers import Dense, Conv2D, Activation, AveragePooling2D, Flatten, Convolution2D, MaxPooling2D, Input, Reshape
from tensorflow.keras.models import load_model



# Squash linear layers and return squashed weights
def squash_layers(cryptonets_model, sess):

    nodes = [n.name for n in tf.get_default_graph().as_graph_def().node]
    nodes = [n for n in nodes if n[0:4] != 'Adam']
    nodes = [n for n in nodes if n[0:4] != 'loss']
    nodes = [n for n in nodes if n[0:7] != 'metrics']

    print_nodes()
    print('cryptonets_model.layers', cryptonets_model.layers)

    conv1_weights = cryptonets_model.layers[0].get_weights()
    conv2_weights = cryptonets_model.layers[3].get_weights()
    fc1_weights = cryptonets_model.layers[6].get_weights()
    fc2_weights = cryptonets_model.layers[8].get_weights()

    # Get squashed weight
    y = Input(shape=(14 * 14 * 5,), name='input')
    y = Reshape((14, 14, 5))(y)
    y = AveragePooling2D(pool_size=(3, 3),
                                strides=(1, 1),
                                padding='same')(y)
    y = Conv2D(filters=50,
                kernel_size=(5, 5),
                strides=(2, 2),
                padding='same',
                use_bias=True,
                trainable=False,
                kernel_initializer=tf.compat.v1.constant_initializer(conv2_weights[0]),
                bias_initializer=tf.compat.v1.constant_initializer(conv2_weights[1]),
                name='conv2d_test')(y)
    y = AveragePooling2D(pool_size=(3, 3),
                          strides=(1, 1),
                          padding='same')(y)
    y = Flatten()(y)
    y = Dense(100, use_bias=True, name='fc_1',
            kernel_initializer=tf.compat.v1.constant_initializer(fc1_weights[0]),
            bias_initializer=tf.compat.v1.constant_initializer(fc1_weights[1]))(y)

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

    return (conv1_weights, (squashed_weights, squashed_bias), fc1_weights, fc2_weights)

# https://www.dlology.com/blog/how-to-convert-trained-keras-model-to-tensorflow-and-make-prediction/
# TODO: remove
def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    from tensorflow.python.framework.graph_util import convert_variables_to_constants, remove_training_nodes
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        # Graph -> GraphDef ProtoBuf
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        frozen_graph = remove_training_nodes(frozen_graph)
        return frozen_graph


def save_model(cryptonets_model, sess, directory, filename):
    weights = squash_layers(cryptonets_model, sess)
    (conv1_weights, squashed_weights, fc1_weights , fc2_weights) = weights[0:4]

    # Remove old graph
    tf.reset_default_graph()
    sess = tf.compat.v1.Session()

    squashed_model = model.cryptonets_model_squashed(conv1_weights, squashed_weights, fc2_weights)
    print('squashed_model summary: ', squashed_model.summary())

    sess.run(tf.compat.v1.global_variables_initializer())
    print('ran initializer?')

    frozen_graph = freeze_session(sess,
                              output_names=[out.op.name for out in squashed_model.outputs])
    print('froze graph')
    print_nodes(frozen_graph)

    tf.io.write_graph(frozen_graph, directory, filename + '.pb', as_text=False)
    print("Model saved to: %s" % filename + '.pb')

def main(FLAGS):
    (x_train, y_train, x_test, y_test) = load_mnist_data()

    cryptonets = model.cryptonets_model()
    print(cryptonets.summary())

    print('x_train', x_train.shape)
    print('y_train', y_train.shape)
    print('x_test', x_test.shape)
    print('y_test', y_test.shape)

    def loss(labels, logits):
        return keras.losses.categorical_crossentropy(labels, logits, from_logits=True)

    cryptonets.compile(optimizer='adam',
              loss=loss,
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