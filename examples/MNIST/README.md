This folder demonstrates several examples of simple CNNs on the MNIST dataset:
  * The CryptoNets folder implements the [Cryptonets](https://www.microsoft.com/en-us/research/publication/cryptonets-applying-neural-networks-to-encrypted-data-with-high-throughput-and-accuracy/) network, which achieves ~99% accuracy on MNIST.

  * The Cryptonets-Relu folder adapts the CryptoNets network to use Relu activations instead of `x^2` activations

  * The MLP folder demonstrates a model with MaxPool layers


It is impossible to perform ReLU and Maxpool using homomorphic encryption. We support these functions in three ways:

  1) A debugging interface (active by default). This runs ReLu/Maxpool locally.
  ***Warning***: This is not privacy-preserving, and should be used for debugging only.

  2) A client-server model, enabled with a command-line flag (`--enable_client=yes`) in the server `test.py` script. If activated, the client will send encrypted data to the server. To perform the ReLU/Maxpool layer, the encrypted data is sent to the client, which decrypts, performs the ReLU/Maxpool, re-encrypts and sends the post-ReLU/Maxpool ciphertexts back to the server.

  One downside of this approach is that it leaks pre- and post-activation values to the client. The client may thereby be able to deduce the weights of the DL model.

  3) An experimental client-server model using multi-party computation, specifically garbled circuits (GC). See `examples/README.md` for more information. To enabe this setting, pass
  `--enable_gc=yes  --mask_gc_inputs=yes --mask_gc_outputs=yes` to the `test.py` script.


These examples depends on the [**Intel® nGraph™ Compiler and runtime engine for TensorFlow**](https://github.com/tensorflow/ngraph-bridge). Make sure the python environment with ngraph-tf bridge is active, i.e. run `source $HE_TRANSFORMER/build/external/venv-tf-py3/bin/activate`. Also ensure the `pyhe_client` wheel has been installed (see `python` folder for instructions).


# CryptoNets
This example demonstrates the [CryptoNets](https://www.microsoft.com/en-us/research/publication/cryptonets-applying-neural-networks-to-encrypted-data-with-high-throughput-and-accuracy/) network, which achieves ~99% accuracy on MNIST.

This example depends on the [**Intel® nGraph™ Compiler and runtime engine for TensorFlow**](https://github.com/tensorflow/ngraph-bridge). Make sure the python environment with ngraph-tf bridge is active, i.e. run `source $HE_TRANSFORMER/build/external/venv-tf-py3/bin/activate`.

## Train the networks
First, train the networks using
```bash
python cryptonets/train.py
python cryptonets-relu/train.py
python mlp/train.py
```
Each `train.py` file takes a `--batch_size` and `--train_loop_count` arguments.


These commands train the network briefly and stores the network weights as protobuf files in './models/*.pb'

# Test the network

## CPU backend
To test a netowrk using the CPU backend, call
```bash
python test.py --batch_size=100 \
               --backend=CPU \
               --model_file='cryptonets/model/model.pb'
```

## HE_SEAL backend
### Plaintext
To test a netowrk using the HE_SEAL backend using unencrypted data (for debugging only), call
```bash
python test.py --batch_size=100 \
               --backend=HE_SEAL \
               --model_file='cryptonets/model/model.pb' \
               --encrypt_server_data=false
```

#### Encrypted
To test a netowrk using the HE_SEAL backend using encrypted data, call
```bash
python test.py --batch_size=100 \
               --backend=HE_SEAL \
               --model_file='cryptonets/model/model.pb' \
               --encrypt_server_data=true \
               --encryption_parameters=$HE_TRANSFORMER/configs/he_seal_ckks_config_N13_L7.json
```
This setting stores the secret key and public key on the same object, and should only be used for debugging, and estimating the runtime and memory overhead.

### Client-server model
To test the client-server model, in one terminal call
```bash
python test.py --backend=HE_SEAL \
               --model_file='cryptonets/model/model.pb' \
               --enable_client=true \
               --encryption_parameters=$HE_TRANSFORMER/configs/he_seal_ckks_config_N13_L7.json
```

In another terminal (with python environment active), call
```bash
python pyclient_mnist.py --batch_size=1024 \
                         --encrypt_data_str=encrypt
```






To test the network, with encrypted data,
```
python test.py --batch_size=4096 \
               --encryption_parameters=$HE_TRANSFORMER/configs/he_seal_ckks_config_N13_L7.json \
               --encrypt_server_data=True \
               --model_file='cryptonets/model/model.pb'
```

This runs inference on the Cryptonets network with encrypted data using the SEAL CKKS backend.
See the [Cryptonets-Relu example](https://github.com/NervanaSystems/he-transformer/blob/master/examples/MNIST/Cryptonets-Relu/README.md) for more details and possible configurations to try.


# CryptoNets-ReLU
