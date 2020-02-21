# Integration with ONNX

```bash

# Train CryptoNets pb
# TODO: Check why other batch sizes don't work
python cryptonets/train.py --epochs=3 --batch_size=16

# Convert TensorFlow model to ONNX model
python -m tf2onnx.convert --input cryptonets.pb --output cryptonets.onnx --verbose --outputs=output/BiasAdd:0 --inputs=input:0

# Install he_seal_backend to virual-env used
# Note: target directory may differ based on python environment.
# To figure out the target directory, just run test.py and look for an error message such as
# RuntimeError: Unable to find backend 'HE_SEAL' as file '/ec/pdx/disks/aipg_lab_home_pool_02/fboemer/repos/he-transformer/build/ngraph-venv/lib/python3.7/site-packages/../../libhe_seal_backend.so
cp ${HE_TRANSFORMER}/build/external/lib/libhe_seal_backend.so ${HE_TRANSFORMER}/build/ngraph-venv/lib/python3.7/site-packages/../../libhe_seal_backend.so

# Evaluate with ngraph-onnx

python test.py

```
