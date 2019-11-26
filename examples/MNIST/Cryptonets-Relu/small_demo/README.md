# Training
python train_demo.py --train_loop_count=10000

# Testing
python test_demo.py --backend=CPU

# Client
OMP_NUM_THREADS=24 \
python pyclient_mnist_demo.py \
  --batch_size=100 \
  --encrypt_data_str='encrypt' --start_batch=0 --tensor_name=import/input

# Server
OMP_NUM_THREADS=24
python test_demo.py \
  --backend=HE_SEAL \
  --encryption_parameters=$HE_TRANSFORMER/configs/he_seal_ckks_config_N13_L5_gc.json \
  --enable_client=yes \
  --enable_gc=yes \
  --mask_gc_inputs=yes \
  --mask_gc_outputs=yes \
  --num_gc_threads=24