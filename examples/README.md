# Python example
This example demonstrates a simple example of a small matrix multiplication and addition. This example depends on the [**Intel® nGraph™ Compiler and runtime engine for TensorFlow**](https://github.com/tensorflow/ngraph-bridge). Make sure the python environment with the ngraph-tf bridge is active, i.e. run `source $HE_TRANSFORMER/build/external/venv-tf-py3/bin/activate`.

The examples rely on numpy, so first run
```bash
pip install numpy
```

To run on the CPU backend,
```bash
python $HE_TRANSFORMER/examples/ax.py --backend=CPU
```

To run on the HE backend,
```bash
python $HE_TRANSFORMER/examples/ax.py --backend=HE_SEAL
```

By default, the default encryption parameters will be used. To specify a non-default set of parameters, use the `encryption_parameters` flag. For example:
```bash
python $HE_TRANSFORMER/examples/ax.py \
  --backend=HE_SEAL \
  --encryption_parameters=$HE_TRANSFORMER/configs/he_seal_ckks_config_N11_L1.json
 ```

# Client-server model
In a proper deployment setting, the public key and secret key will not be stored in the same location. Instead, a client will store the secret key, and provide the backend with encrypted data.

The client-server model uses python bindings. See the [README.md](https://github.com/IntelAI/he-transformer/tree/master/README.md) for instructions to build he-transformer with python bindings.

For a simple demonstration of a server-client approach, run
```bash
python $HE_TRANSFORMER/examples/ax.py \
  --backend=HE_SEAL \
  --enable_client=yes
```

This will discard the Tensorflow inputs and instead wait for a client to connect and provide encrypted inputs.
To start the client, in a separate terminal on the same host (with the ngraph-tf bridge python environment active), run
```bash
python $HE_TRANSFORMER/examples/pyclient.py
```

Once the computation is complete, the output will be returned to the client and decrypted. The server will attempt decrypt the output as well; however, since it does not have the client's secret key, the output will be meaningless.

The client-server approach currently works only for functions with one result tensor.

For deep learning examples using the client-server model, see the `MNIST` folder.

## Multi-party computation with garbled circuits
One downside to the above approach is the client may deduce the deep learning model weights, since it receives the pre-activation values at each layer. One work-around is to additively mask the pre-activation values with a random number before sending them to the client. Since HE computation happens in a finite field, if the random number is chosen uniformly from the field, the client will receive a uniform random number from the field, and thereby cannot deduce anything about the model weights. Then, the server and client interactively compute the activation using multi-party computation methods, such as garbled circuits (GC). The GC approach ensures the client learns only the random additively-masked values. After the client sends the masked encrypted post-activation values to the server, the server performs the unmasking using homomorphic addition. This approach is similar to that of (Gazelle)[https://www.usenix.org/system/files/conference/usenixsecurity18/sec18-juvekar.pdf].

See the `MNIST` folder for a DL example using garbled circuits.

# List of command-line flags
  * `OMP_NUM_THREADS`. Set to 1 to enable single-threaded execution (useful for debugging). For best multi-threaded performance, this number should be tuned.
  * `NGRAPH_HE_VERBOSE_OPS`. Set to `all` to print information about every operation performed. Set to a comma-separated list to print information about those ops; for example `NGRAPH_HE_VERBOSE_OPS=add,multiply,convolution`. *Note*, `NGRAPH_HE_LOG_LEVEL` should be set to at least 3 when using `NGRAPH_HE_VERBOSE_OPS`
  * `NGRAPH_HE_LOG_LEVEL`. Defines the verbosity of the logging. Set to 0 for minimal logging, 5 for maximum logging. Roughly:
    - `NGRAPH_HE_LOG_LEVEL=0 [default]` will print minimal amount of information
    - `NGRAPH_HE_LOG_LEVEL=1` will print encryption parameters
    - `NGRAPH_HE_LOG_LEVEL=3` will print op information (when `NGRAPH_HE_VERBOSE_OPS` is enabled)
    - `NGRAPH_HE_LOG_LEVEL=4` will print communication information
    - `NGRAPH_HE_LOG_LEVEL=5` is the highest debug level

  # Creating your own DL model
  We currently only support DL models with a single `Parameter`, as is the case for most standard DL models. During training, the weights may be TensorFlow `Variable` ops, which translate to nGraph `Parameter` ops. In this case, he-transformer will be unable to tell what tensor represents the data to encrypt. So, you will need to convert the ops representing the model weights to `Constant` ops. TensorFlow, for example, has a `freeze_graph` utility to do so. See the `MNIST` folder for examples using `freeze_graph`.
