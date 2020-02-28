# Import tensorflow before other imports to avoid protobuf version mismatch errors
import tensorflow as tf
import onnx
from ngraph_onnx.onnx_importer.importer import import_onnx_model
import ngraph as ng
import numpy as np
import argparse


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


def print_accuracy(y_pred, y_test):
    y_pred_label = np.argmax(y_pred, 1)
    y_test_label = np.argmax(y_test, 1)
    print('y_pred_label', y_pred_label)
    print('y_test_label', y_test_label)
    correct_prediction = np.equal(y_pred_label, y_test_label)
    error_count = np.size(correct_prediction) - np.sum(correct_prediction)
    test_accuracy = np.mean(correct_prediction)

    print("Error count", error_count, "of", y_pred_label.size, "elements.")
    print("Accuracy: %g " % test_accuracy)


def test_interpreter_backend(ng_function, x_test, y_test):
    int_runtime = ng.runtime(backend_name='INTERPRETER')
    int_cryptonets = int_runtime.computation(ng_function)
    int_pred = int_cryptonets(x_test)[0]
    print('int_pred', int_pred)
    print_accuracy(int_pred, y_test)


def test_he_backend(FLAGS, ng_function, x_test, y_test):
    print('creating runtime')
    he_runtime = ng.runtime(backend_name='HE_SEAL')
    print('created runtime')
    config = {}
    if FLAGS.encryption_parameters != '':
        config['encryption_parameters'] = FLAGS.encryption_parameters

    if FLAGS.encrypt:
        config['Parameter_8'] = 'encrypt,packed'
    else:
        config['Parameter_8'] = 'packed'

    print('config', config)

    he_runtime.set_config(config)
    he_cryptonets = he_runtime.computation(ng_function)
    he_pred = he_cryptonets(x_test)[0]
    print('he_pred', he_pred)
    print_accuracy(he_pred, y_test)


def run(FLAGS):
    print('creating runtime')
    he_runtime = ng.runtime(backend_name='HE_SEAL')
    print('created runtime')

    print('loading data')
    (x_train, y_train, x_test,
     y_test) = load_mnist_data(batch_size=FLAGS.batch_size)
    print('loaded data')
    print(x_test.shape)

    onnx_protobuf = onnx.load('cryptonets.onnx')
    ng_function = import_onnx_model(onnx_protobuf)
    print('loaded model', ng_function)

    # test_interpreter_backend(ng_function, x_test, y_test)
    test_he_backend(FLAGS, ng_function, x_test, y_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--encryption_parameters",
        type=str,
        default="",
        help=
        "Filename containing json description of encryption parameters, or json description itself",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Number of datapoints to perform inference on",
    )
    parser.add_argument(
        "--encrypt",
        type=bool,
        default=False,
        help="Whether or not to encrypt the input data",
    )

    FLAGS, unparsed = parser.parse_known_args()
    run(FLAGS)
