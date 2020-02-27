#!/usr/bin/python3

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
import numpy as np
import argparse
import os.path
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.tools import freeze_graph


def print_nodes(graph_def=None):
    """Prints the node names of a graph_def.
        If graph_def is not provided, use default graph_def"""

    if graph_def is None:
        nodes = [n.name for n in tf.get_default_graph().as_graph_def().node]
    else:
        nodes = [n.name for n in graph_def.node]

    print("nodes", nodes)


def load_mnist_data(start_batch=0, batch_size=10000):
    """Returns MNIST data in one-hot form"""
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    y_train = tf.compat.v1.keras.utils.to_categorical(y_train, num_classes=10)
    y_test = tf.compat.v1.keras.utils.to_categorical(y_test, num_classes=10)
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    x_train /= 255.0
    x_test /= 255.0

    x_test = x_test[start_batch:start_batch + batch_size]
    y_test = y_test[start_batch:start_batch + batch_size]

    return (x_train, y_train, x_test, y_test)


def load_pb_file(filename):
    """"Returns the graph_def from a saved protobuf file"""
    if not os.path.isfile(filename):
        raise Exception("File, " + filename + " does not exist")

    with tf.io.gfile.GFile(filename, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    print("Model restored")
    return graph_def


# https://www.dlology.com/blog/how-to-convert-trained-keras-model-to-tensorflow-and-make-prediction/
def freeze_session(session,
                   keep_var_names=None,
                   output_names=None,
                   clear_devices=True):
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
    from tensorflow.python.framework.graph_util import (
        convert_variables_to_constants,
        remove_training_nodes,
    )

    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(
            set(v.op.name for v in tf.global_variables()).difference(
                keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        # Graph -> GraphDef ProtoBuf
        input_graph_def = graph.as_graph_def()
        print_nodes(input_graph_def)
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
        frozen_graph = remove_training_nodes(frozen_graph)
        return frozen_graph


def save_model(sess, output_names, directory, filename):
    frozen_graph = freeze_session(sess, output_names=output_names)
    print_nodes(frozen_graph)
    tf.io.write_graph(frozen_graph, directory, filename + ".pb", as_text=False)
    print("Model saved to: %s" % filename + ".pb")


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("on", "yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("off", "no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def train_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch Size")

    return parser


def client_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument(
        "--hostname", type=str, default="localhost", help="Hostname of server")
    parser.add_argument(
        "--port", type=int, default=34000, help="Port of server")
    parser.add_argument(
        "--encrypt_data_str",
        type=str,
        default="encrypt",
        help='"encrypt" to encrypt client data, "plain" to not encrypt',
    )
    parser.add_argument(
        "--tensor_name",
        type=str,
        default="import/input",
        help="Input tensor name")
    parser.add_argument(
        "--start_batch", type=int, default=0, help="Test data start index")

    return parser


def server_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument(
        "--enable_client",
        type=str2bool,
        default=False,
        help="Enable the client")
    parser.add_argument(
        "--enable_gc",
        type=str2bool,
        default=False,
        help="Enable garbled circuits")
    parser.add_argument(
        "--mask_gc_inputs",
        type=str2bool,
        default=False,
        help="Mask garbled circuits inputs",
    )
    parser.add_argument(
        "--mask_gc_outputs",
        type=str2bool,
        default=False,
        help="Mask garbled circuits outputs",
    )
    parser.add_argument(
        "--num_gc_threads",
        type=int,
        default=1,
        help="Number of threads to run garbled circuits with",
    )
    parser.add_argument(
        "--backend", type=str, default="HE_SEAL", help="Name of backend to use")
    parser.add_argument(
        "--encryption_parameters",
        type=str,
        default="",
        help=
        "Filename containing json description of encryption parameters, or json description itself",
    )
    parser.add_argument(
        "--encrypt_server_data",
        type=str2bool,
        default=False,
        help=
        "Encrypt server data (should not be used when enable_client is used)",
    )
    parser.add_argument(
        "--pack_data",
        type=str2bool,
        default=True,
        help="Use plaintext packing on data")
    parser.add_argument(
        "--start_batch", type=int, default=0, help="Test data start index")
    parser.add_argument(
        "--model_file",
        type=str,
        default="",
        help="Filename of saved protobuf model")
    parser.add_argument(
        "--input_node",
        type=str,
        default="import/input:0",
        help="Tensor name of data input",
    )
    parser.add_argument(
        "--output_node",
        type=str,
        default="import/output/BiasAdd:0",
        help="Tensor name of model output",
    )

    return parser


def server_config_from_flags(FLAGS, tensor_param_name):
    rewriter_options = rewriter_config_pb2.RewriterConfig()
    rewriter_options.meta_optimizer_iterations = rewriter_config_pb2.RewriterConfig.ONE
    rewriter_options.min_graph_nodes = -1
    server_config = rewriter_options.custom_optimizers.add()
    server_config.name = "ngraph-optimizer"
    server_config.parameter_map["ngraph_backend"].s = FLAGS.backend.encode()
    server_config.parameter_map["device_id"].s = b""
    server_config.parameter_map[
        "encryption_parameters"].s = FLAGS.encryption_parameters.encode()
    server_config.parameter_map["enable_client"].s = str(
        FLAGS.enable_client).encode()
    server_config.parameter_map["enable_gc"].s = (str(FLAGS.enable_gc)).encode()
    server_config.parameter_map["mask_gc_inputs"].s = (str(
        FLAGS.mask_gc_inputs)).encode()
    server_config.parameter_map["mask_gc_outputs"].s = (str(
        FLAGS.mask_gc_outputs)).encode()
    server_config.parameter_map["num_gc_threads"].s = (str(
        FLAGS.num_gc_threads)).encode()

    if FLAGS.enable_client:
        server_config.parameter_map[tensor_param_name].s = b"client_input"
    elif FLAGS.encrypt_server_data:
        server_config.parameter_map[tensor_param_name].s = b"encrypt"

    if FLAGS.pack_data:
        server_config.parameter_map[tensor_param_name].s += b",packed"

    config = tf.compat.v1.ConfigProto()
    config.MergeFrom(
        tf.compat.v1.ConfigProto(
            graph_options=tf.compat.v1.GraphOptions(
                rewrite_options=rewriter_options)))

    return config
