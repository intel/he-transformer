import numpy as np
from mnist_util import load_mnist_test_data, client_argument_parser
import pyhe_client

# Parse command-line arguments
FLAGS, unparsed = client_argument_parser().parse_known_args()

# Load data
(x_test, y_test) = load_mnist_test_data(FLAGS.start_batch, FLAGS.batch_size)

client = pyhe_client.HESealClient(FLAGS.hostname, FLAGS.port, FLAGS.batch_size,
                                    {FLAGS.tensor_name: (FLAGS.encrypt_data_str, x_test.flatten('C'))})
results = np.array(client.get_results())
y_pred = results.reshape(FLAGS.batch_size, 10)
accuracy = np.mean(np.argmax(y_test, 1) == np.argmax(y_pred, 1))
print('Accuracy: ', accuracy)