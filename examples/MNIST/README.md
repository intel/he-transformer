This folder demonstrates several examples of simple CNNs on the MNIST dataset:
  * The CryptoNets folder implements the [Cryptonets](https://www.microsoft.com/en-us/research/publication/cryptonets-applying-neural-networks-to-encrypted-data-with-high-throughput-and-accuracy/) network, which achieves ~99% accuracy on MNIST.

  * The Cryptonets-Relu folder adapts the CryptoNets network to use Relu activations instead of `x^2` activations

  * The MLP folder demonstrates a model with MaxPool layers


It is impossible to perform ReLU and Maxpool using the CKKS homomorphic encryption scheme directly. Instead, we support these functions in three ways:

  1) A debugging interface (active by default). This runs ReLu/Maxpool locally.
  ***Warning***: This is not privacy-preserving, and should be used for debugging only.

  2) A client-server model, enabled with a command-line flag (`--enable_client=true`) in the server `test.py` script. If activated, the client will send encrypted data to the server. To perform the ReLU/Maxpool layer, the encrypted data is sent to the client, which decrypts, performs the ReLU/Maxpool, re-encrypts and sends the post-ReLU/Maxpool ciphertexts back to the server.

  One downside of this approach is that it leaks pre- and post-activation values to the client. The client may thereby be able to deduce the weights of the DL model.

  3) An experimental client-server model using multi-party computation, specifically garbled circuits (GC). See `examples/README.md` for more information. To enabe this setting, pass
  `--enable_gc=true  --mask_gc_inputs=true --mask_gc_outputs=true` to the `test.py` script.


These examples depends on the [**Intel® nGraph™ Compiler and runtime engine for TensorFlow**](https://github.com/tensorflow/ngraph-bridge). Make sure the python environment with ngraph-tf bridge is active, i.e. run `source $HE_TRANSFORMER/build/external/venv-tf-py3/bin/activate`. Also ensure the `pyhe_client` wheel has been installed (see `python` folder for instructions).

# Train the networks
First, train the networks using
```bash
python Cryptonets/train.py --epochs=20 --batch_size=128
python Cryptonets-Relu/train.py --epochs=20 --batch_size=128
python MLP/train.py --epochs=20 --batch_size=128
```
Each `train.py` file takes a `--batch_size` and `--epochs` arguments.

These commands train the network briefly and stores the network weights as protobuf files in './models/*.pb'

# Test the network

## CPU backend
To test a network using the CPU backend, call
```bash
python test.py --batch_size=10000 \
               --backend=CPU \
               --model_file=models/cryptonets.pb
```

## HE_SEAL backend
### Plaintext
To test a network using the HE_SEAL backend using unencrypted data (for debugging only), call
```bash
python test.py --batch_size=100 \
               --backend=HE_SEAL \
               --model_file=models/cryptonets.pb \
               --encrypt_server_data=false
```

#### Encrypted
To test a network using the HE_SEAL backend using encrypted data, call
```bash
python test.py --batch_size=100 \
               --backend=HE_SEAL \
               --model_file=models/cryptonets.pb \
               --encrypt_server_data=true \
               --encryption_parameters=$HE_TRANSFORMER/configs/he_seal_ckks_config_N13_L8.json
```
This setting stores the secret key and public key on the same object, and should only be used for debugging, and estimating the runtime and memory overhead.

### Client-server model
To test the client-server model, in one terminal call
```bash
python test.py --backend=HE_SEAL \
               --model_file=models/cryptonets.pb \
               --enable_client=true \
               --encryption_parameters=$HE_TRANSFORMER/configs/he_seal_ckks_config_N13_L8.json
```

In another terminal (with the python environment active), call
```bash
python pyclient_mnist.py --batch_size=1024 \
                         --encrypt_data_str=encrypt
```

### Multi-party computation with garbled circuits (GC)
To test the network using garbled circuits for secure computation of activations, make sure he-transformer was configured using `-DNGRAPH_HE_ABY_ENABLE=ON`. Then, install the [python client](https://github.com/IntelAI/he-transformer/tree/master/python). In one terminal, run
```bash
OMP_NUM_THREADS=24 \
NGRAPH_HE_VERBOSE_OPS=all \
NGRAPH_HE_LOG_LEVEL=3 \
python test.py \
  --backend=HE_SEAL \
  --model_file=models/cryptonets-relu.pb \
  --encryption_parameters=$HE_TRANSFORMER/configs/he_seal_ckks_config_N13_L5_gc.json \
  --enable_client=true \
  --enable_gc=true \
  --mask_gc_inputs=true \
  --mask_gc_outputs=true \
  --num_gc_threads=24
```

The 'mask_gc_inputs` flag indicates pre-activation values should be additively masked.
The 'mask_gc_outputs` flag indicates post-activation values should be additively masked.
Both values should be set to `yes` to ensure privacy.

Note, `num_gc_threads` should be at most `OMP_NUM_THREADS` for optimal performance.

In another terminal, run
```bash
NGRAPH_HE_LOG_LEVEL=3 \
OMP_NUM_THREADS=24 \
python pyclient_mnist.py \
  --batch_size=50 \
  --encrypt_data_str=encrypt
```
Note, for optimal performance, `OMP_NUM_THREADS` should be set to at least `num_gc_threads` specified in the server configuration.

# Encryption parameters
The choice of encryption parameters is critical for seurity, accuracy, and performance. The `--encryption_parameters` flag is used to set the encryption parameters, to either a filename containing a configuration or the configuration itself.

If no value is passed, a small parameter choice will be used. ***Warning***: the default parameter selection does not enforce any security level, and should be used only for debugging.

Configurations should be of the following form:
```bash
{
  "scheme_name": "HE_SEAL",
  "poly_modulus_degree": 4096,
  "security_level": 128,
  "coeff_modulus": [
    30,
    22,
    22,
    30
  ],
  "scale": 4194304,
  "complex_packing": true,
}
```
where
 - `scheme_name` should always be "HE_SEAL"
 - `poly_modulus_degree` should be a power of two in `{1024, 2048, 4096, 8192, 16384}`
 - `security_level` should be in `{0, 128, 192, 256}`. Note: a security level of `0` indicates the HE backend will *not* enforce a minimum security level. This means the encryption is not secure against attacks.
 - `coeff_modulus` should be a list of integers in [1,60]. This indicates the bit-widths of the coefficient moduli used.
 - `scale` is the scale at which number are encoded; `log2(scale)` represents roughly the fixed-bit precision of the encoding. If no scale is passed a default value is selected.
 - `complex_packing` specifies whether or not to double the capacity (i.e. maximum batch size) by packing two scalars `(a,b)` in a complex number `a+bi`. Typically, the capacity is `poly_modulus_degree/2`. Enabling complex packing doubles the capacity to `poly_modulus_degree`. Note: enabling `complex_packing` will reduce the performance of ciphertext-ciphertext multiplication.

## Parameter selection
Parameter selection remains largely a hand-tuned process. We give a rough guideline below:

 1) Select the security level, `lambda`. `lambda=128` is a standard minimum selection.
 2) Compute the multiplicative depth, `L` of your computation graph. This is the largest number of multiplications between non-polynomial activations.
 3) Estimate the bit-precision `s`, required. We have found the best performance-accuracy tradeoff with ~24 bits.
 4) Choose `coeff_modulus = [s, s, s, ..., s]`, a list of `L` coeffficient moduli, each with `s` bits. Set the scale to `s`.
 5) Compute the total coefficient modulus bit width, `L * s` in the above parameter selection
 6) Set the `poly_modulus_degree` to the smallest power of two with coefficient modulus smaller than the maximum allowed, based on the Tables of Recommended Parameters from the HomomorphicEncryption.org [security whitepaper](http://homomorphicencryption.org/white_papers/security_homomorphic_encryption_white_paper.pdf). For simplicity, it may be easier to refer to SEAL's table of recommended parameters [here](https://github.com/microsoft/SEAL/blob/master/native/src/seal/util/hestdparms.h).

 For instance, with `lamabda=128`, `L=3`, and bit-precision `s=24`, we require `24*3=72` total coefficient modulus bits. For classical 128-bit security, we might have the below table:

  | N     | log2(q) |
  |-------|---------|
  | 1024  | 27      |
  | 2048  | 54      |
  | 4096  | 109     |
  | 8192  | 218     |
  | 16384 | 438     |
  | 32768 | 881     |

   If our total coefficient modulus bitwidth, `log2(q)` were less than 27, we would select `N=1024`. If `27 < log2(q) <= 54`, we would select `N=2048`. Since `54 < log2(q) = 72 <= 109` in our example, we select `N=2048`.

7. The batch size is determined by the `poly_modulus_degree`, `N`, and is at most `poly_modulus_degree/2`. Our choice of plaintext packing implies the runtime of HE operations is largely independent of `poly_modulus_degree`. So for best performance, choose batch size `poly_modulus_degree/2` during inference.

8. `complex_packing`. If the DL model inference contains no ciphertext-ciphertext multiplication, (usually the case if the model contains no polynomial activations), set to `false` to double the maximum batch size to `poly_modulus_degree`.

A few further considerations:
* The maximum bitwidth of the absolute value of an encoded value is roughly the ratio between the last two coeffcient moduli. To avoid `out of scale` errors, try increasing the last coefficient modulus bitwidth.
* If  the computation is inaccurate, try increasing the number and bitwidth of coefficient moduli.

* The runtime is roughly linear in `N` (the `poly_modulus_degree` and `log2(q)` (the total coefficient modulus bitwidth).
