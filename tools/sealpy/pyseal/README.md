Python bindings for SEAL 3.3
Several functions are unimplemented, notably the util/ folder in SEAL.

# Dependencies
* `cmake` version >= 3.10
* `python3`
* `SEAL3.3` is built automatically

# Build instructions
```bash
mkdir build
cd build
cmake ..
make -j python_wheel
```
This will create a python wheel in `python/dist` folder.
To install it:

```bash
virtualenv -p "$(command -v python3)" venv3
source venv3/bin/activate
pip install python/dist/pyseal*.whl
```

# Test installation
```bash
python -c "import pyseal"
```
should work. To run unit-tests, call
```bash
cd ../test

# To run a single unit-test
python -m unittest seal.test_small_modulus

# To run all unit-tests
python -m unittest discover seal
```

# Examples
```bash
python ../examples/examples.py
```

# Differences from C++ implementation:
* Use `X.scale` instead of `X.scale()` for `X=Ciphertext(), Plaintext()`
* Use `X.parms_id` instead of `X.parms_id()` for `X=Ciphertext(), Plaintext()`
* Use `X.assign(other)` to call assignment operator, instead of `X = other`, which will call the copy constructor. This holds for `X=Ciphertext(), Plaintext(), BigUInt()`
* Use `stream = X.save()` instead of `X.save(stream)` for `X=Ciphertext(), Plaintext(), BigUInt()`
* Use `X.data(1)` instead of `BigUInt.data()[1]` for `X=BigUint()`
* Many functions unimplemented, especially in BigUInt
