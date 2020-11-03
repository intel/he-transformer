<p align="center">
  <img src="images/nGraph_mask_1-1.png" width="200">
</p>

# HE Transformer for nGraph

The **Intel® HE transformer for nGraph™** is a Homomorphic Encryption (HE) backend to the [**Intel® nGraph Compiler**](https://github.com/IntelAI/ngraph), Intel's graph compiler for Artificial Neural Networks.

Homomorphic encryption is a form of encryption that allows computation on encrypted data, and is an attractive remedy to increasing concerns about data privacy in the field of machine learning. For more information, see our [original paper](https://arxiv.org/pdf/1810.10121.pdf). Our [updated paper](https://arxiv.org/pdf/1908.04172.pdf) showcases many of the recent advances in he-transformer.

This project is meant as a proof-of-concept to demonstrate the feasibility of HE  on local machines. The goal is to measure performance of various HE schemes for deep learning. This is  **not** intended to be a production-ready product, but rather a research tool.

Currently, we support the [CKKS](https://eprint.iacr.org/2018/931.pdf) encryption scheme, implemented by the [Simple Encrypted Arithmetic Library (SEAL)](https://github.com/Microsoft/SEAL) from Microsoft Research.

To help compute non-polynomial activiations, we additionally integrate with the [ABY](https://github.com/encryptogroup/ABY) multi-party computation library. See also the [NDSS 2015 paper](https://encrypto.de/papers/DSZ15.pdf) introducing ABY. For more details about our integration with ABY, please refer to our [ARES 2020 paper](https://eprint.iacr.org/2020/721.pdf).

We also integrate with the [**Intel® nGraph™ Compiler and runtime engine for TensorFlow**](https://github.com/tensorflow/ngraph-bridge) to allow users to run inference on trained neural networks through Tensorflow.

## Examples
The [examples](https://github.com/IntelAI/he-transformer/tree/master/examples) folder contains a deep learning example which depends on the [**Intel® nGraph™ Compiler and runtime engine for TensorFlow**](https://github.com/tensorflow/ngraph-bridge).

## Building HE Transformer

### Dependencies
- Operating system: Ubuntu 16.04, Ubuntu 18.04.
- CMake >= 3.12
- Compiler: g++ version >= 6.0, clang >= 5.0 (with ABY g++ version >= 8.4)
- OpenMP is strongly suggested, though not strictly necessary. You may experience slow runtimes without OpenMP
- python3 and pip3
- virtualenv v16.1.0
- bazel v0.25.2

For a full list of dependencies, see the [docker containers](https://github.com/IntelAI/he-transformer/tree/master/contrib/docker), which build he-transformer on a reference OS.

#### The following dependencies are built automatically
- [nGraph](https://github.com/NervanaSystems/ngraph) - v0.28.0-rc.1
- [nGraph-bridge](https://github.com/tensorflow/ngraph-bridge) - v0.22.0-rc3
- [SEAL](https://github.com/Microsoft/SEAL) - v3.4.5
- [TensorFlow](https://github.com/tensorflow/tensorflow) - v1.14.0
- [Boost](https://github.com/boostorg) v1.69
- [Google protobuf](https://github.com/protocolbuffers/protobuf) v3.10.1

### To install bazel
```bash
    wget https://github.com/bazelbuild/bazel/releases/download/0.25.2/bazel-0.25.2-installer-linux-x86_64.sh
    bash bazel-0.25.2-installer-linux-x86_64.sh --user
 ```
 Add and source the bin path to your `~/.bashrc` file to call bazel
```bash
 export PATH=$PATH:~/bin
 source ~/.bashrc
```

### 1. Build HE-Transformer
Before building, make sure you deactivate any active virtual environments (i.e. run `deactivate`)
```bash
git clone https://github.com/IntelAI/he-transformer.git
cd he-transformer
export HE_TRANSFORMER=$(pwd)
mkdir build
cd $HE_TRANSFORMER/build
cmake .. -DCMAKE_CXX_COMPILER=clang++-6.0
```
Note, you may need sudo permissions to install he_seal_backend to the default location. To set a custom installation prefix, add the `-DCMAKE_INSTALL_PREFIX=~/my_install_prefix` flag to the above cmake command.

See 1a and 1b for additional configuration options. To install, run the below command (note, this may take several hours. To speed up compilation with multiple threads, call `make -j install`)
```bash
make install
```

### 1a. Multi-party computation (MPC) with garbled circuits (GC)
To enable an integration with an experimental multi-party computation backend using garbled circuits via [ABY](https://github.com/encryptogroup/ABY), call
```bash
cmake .. -DNGRAPH_HE_ABY_ENABLE=ON
```
Note: this feature is experimental, and may suffer from performance and memory issues.
To use this feature, build python bindings for the client, and see `3. python examples`.

#### 1b. To build documentation
First install the additional required dependencies:
```bash
sudo apt-get install doxygen graphviz
```
Then add the following CMake flag
```bash
cd doc
cmake .. -DNGRAPH_HE_DOC_BUILD_ENABLE=ON
```
and call
```bash
make docs
```
to create doxygen documentation in `$HE_TRANSFORMER/build/doc/doxygen`.

#### 1c. Python bindings for client
To build a client-server model with python bindings (recommended for running neural networks through TensorFlow):
```bash
cd $HE_TRANSFORMER/build
source external/venv-tf-py3/bin/activate
make install python_client
```
This will create `python/dist/pyhe_client-*.whl`. Install it using
```bash
pip install python/dist/pyhe_client-*.whl
```
To check the installation worked correctly, run
```bash
python3 -c "import pyhe_client"
```
This should run without errors.

### 2. Run C++ unit-tests
```bash
cd $HE_TRANSFORMER/build
# To run single HE_SEAL unit-test
./test/unit-test --gtest_filter="HE_SEAL.add_2_3_cipher_plain_real_unpacked_unpacked"
# To run all C++ unit-tests
./test/unit-test
```

### 3. Run python examples
See [examples/README.md](https://github.com/IntelAI/he-transformer/tree/master/examples/README.md) for examples of running he-transformer for deep learning inference on encrypted data.

## Code formatting
Please run `maint/apply-code-format.sh` before submitting a pull request.


## Papers using he-transformer
- Fabian Boemer, Yixing Lao, Rosario Cammarota, and Casimir Wierzynski. nGraph-HE: a graph compiler for deep learning on homomorphically encrypted data. In ACM International Conference on Computing Frontiers 2019. https://dl.acm.org/doi/10.1145/3310273.3323047
- Fabian Boemer, Anamaria Costache, Rosario Cammarota, and Casimir
Wierzynski. 2019. nGraph-HE2: A High-Throughput Framework for
Neural Network Inference on Encrypted Data. In WAHC’19. https://dl.acm.org/doi/pdf/10.1145/3338469.3358944
- Fabian Boemer, Rosario Cammarota, Daniel Demmler, Thomas Schneider, and Hossein Yalame. 2020. MP2ML: A Mixed-Protocol Machine
Learning Framework for Private Inference. In ARES’20. https://eprint.iacr.org/2020/721.pdf
