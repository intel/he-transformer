# Integration with ONNX

# Setup
```bash
cd ${HE_TRANSFORMER}
mkdir -p build && cd build
cmake .. -DNGRAPH_HE_ONNX_ENABLE=ON
make -j he_seal_backend

# Set up virtualenv
virtualenv -p "$(command -v python3)" venv3
source venv3/bin/activate

# Install ngraph and onnx wheels
make -j ext_ngraph_onnx
make -j ext_ngraph
pip3 install -e ext_ngraph_onnx/src/ext_ngraph_onnx/
pip3 install ext_ngraph/src/ext_ngraph-build/python/dist/ngraph_core-*.whl
```

## Dependencies
```bash
pip3 install tensorflow==1.15 tf2onnx
```

```bash
# Train CryptoNets pb
# TODO: Check why other batch sizes don't work
python ${HE_TRANSFORMER}/examples/MNIST/cryptonets/train.py --epochs=3 --batch_size=16 --export_batch_size=true

# Convert TensorFlow model to ONNX model
python -m tf2onnx.convert --input ${HE_TRANSFORMER}/examples/MNIST/models/cryptonets.pb --output ${HE_TRANSFORMER}/onnx/cryptonets.onnx --verbose --outputs=output/BiasAdd:0 --inputs=input:0

# Install he_seal_backend to virual-env used
cp ${HE_TRANSFORMER}/build/src/libhe_seal_backend.so $VIRTUAL_ENV/lib/libhe_seal_backend.so

# Evaluate with ngraph-onnx on plaintext model
cd ${HE_TRANSFORMER}/onnx
NGRAPH_HE_LOG_LEVEL=4 NGRAPH_HE_VERBOSE_OPS=all
python ${HE_TRANSFORMER}/onnx/test.py

# Evaluate on encrypted model
NGRAPH_HE_LOG_LEVEL=4 NGRAPH_HE_VERBOSE_OPS=all
python ${HE_TRANSFORMER}/onnx/test.py \
    --encryption_parameters=$HE_TRANSFORMER/configs/he_seal_ckks_config_N13_L7.json \
    --encrypt=true
```
