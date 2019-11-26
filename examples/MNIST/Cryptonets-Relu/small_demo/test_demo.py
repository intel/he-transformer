import tensorflow as tf
import ngraph_bridge
import numpy as np

from mnist_util import server_argument_parser, \
                       server_config_from_flags, \
                       load_pb_file

# Load saved model
tf.import_graph_def(load_pb_file('./model/model.pb'))

# Get input / output tensors
x_input = tf.compat.v1.get_default_graph().get_tensor_by_name("import/input:0")
y_output = tf.compat.v1.get_default_graph().get_tensor_by_name("import/output:0")

# Create configuration to encrypt input
FLAGS, unparsed = server_argument_parser().parse_known_args()
config = server_config_from_flags(FLAGS, x_input.name)
with tf.compat.v1.Session(config=config) as sess:
    # Evaluate model (random input data is discarded)
    y_output.eval(feed_dict={x_input: np.random.rand(10000, 28, 28, 1)})
