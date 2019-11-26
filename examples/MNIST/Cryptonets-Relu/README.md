This example demonstrates a simple CNN, which achieves ~98% on MNIST.
The architecture uses MaxPool and ReLU activations.

Since it is not possible to date to perform ReLU using the CKKS homomorphic encryption, this model will only run with the help of a client. The client will send encrypted data to the server. To perform the ReLU/Maxpool layer, the encrypted data is sent to the client, which decrypts, performs the ReLU/Maxpool, re-encrypts and sends the post-ReLU/Maxpool ciphertexts back to the server.

This example depends on the [**Intel® nGraph™ Compiler and runtime engine for TensorFlow**](https://github.com/tensorflow/ngraph-bridge). Make sure the python environment with ngraph-tf bridge is active, i.e. run `source $HE_TRANSFORMER/build/external/venv-tf-py3/bin/activate`. Also ensure the `pyhe_client` wheel has been installed (see `python` folder for instructions).

# Train the network
First, train the network using
```bash
python train.py
```
This trains the network briefly and stores the network weights.

# Test the network
First, make sure the python virtual environment is active:
```bash
source $HE_TRANSFORMER/build/external/venv-tf-py3/bin/activate
cd $HE_TRANSFORMER/examples/MNIST/Cryptonets-Relu
```

## CPU
To test the network using the CPU backend, run
```bash
python test.py --batch_size=512 --backend=CPU
```

## HE_SEAL plaintext
To test the network using plaintext inputs (i.e. not encrypted), run
```bash
python test.py --batch_size=512 --backend=HE_SEAL
```
This should just be used for debugging, since the data is not encrypted

## HE_SEAL encrypted
To test the network using encrypted inputs, run
```bash
python test.py --batch_size=1024 \
               --backend=HE_SEAL \
               --encrypt_server_data=yes \
               --encryption_parameters=$HE_TRANSFORMER/configs/he_seal_ckks_config_N11_L1.json
```

This runs inference on the Cryptonets network using the SEAL CKKS backend. Note, the client is *not* enabled, meaning the backend holds the secret and public keys. This should only be used for debugging, as it is *not* cryptographically secure.

See the [examples](https://github.com/NervanaSystems/he-transformer/blob/master/examples/README.md) for more details on the encryption parameters.


## Debugging
For debugging purposes, enable the `NGRAPH_HE_LOG_LEVEL` or `NGRAPH_HE_VERBOSE_OPS` flags. See [here](https://github.com/NervanaSystems/he-transformer/blob/master/examples/README.md) for more details.

## Garbled Circuits (GC)
To test the network using garbled circuits for secure computation of activations, make sure he-transformer was configured using `-DNGRAPH_HE_ABY_ENABLE=ON`. Then, install the [python client](https://github.com/NervanaSystems/he-transformer/tree/master/python). Then, in one terminal, run
```bash
source $HE_TRANSFORMER/build/external/venv-tf-py3/bin/activate
cd $HE_TRANSFORMER/examples/MNIST/Cryptonets-Relu
OMP_NUM_THREADS=24 \
NGRAPH_HE_VERBOSE_OPS=all \
NGRAPH_HE_LOG_LEVEL=3 \
python test.py \
  --backend=HE_SEAL \
  --encryption_parameters=$HE_TRANSFORMER/configs/he_seal_ckks_config_N13_L5_gc.json \
  --enable_client=yes \
  --enable_gc=yes \
  --mask_gc_inputs=yes \
  --mask_gc_outputs=yes \
  --num_gc_threads=24
```

The 'mask_gc_inputs` flag indicates pre-activation values should be additively masked.
The 'mask_gc_outputs` flag indicates post-activation values should be additively masked.
Both values should be set to `yes` to ensure privacy.

Note, `num_gc_threads` should be at most `OMP_NUM_THREADS` for optimal performance.

In another terminal, run
```bash
source $HE_TRANSFORMER/build/external/venv-tf-py3/bin/activate
cd $HE_TRANSFORMER/examples/MNIST
NGRAPH_HE_LOG_LEVEL=3 \
OMP_NUM_THREADS=24 \
python pyclient_mnist.py \
  --batch_size=50 \
  --encrypt_data=yes
```
Note, for optimal performance, `OMP_NUM_THREADS` should be set to at least `num_gc_threads` specified in the server configuration
