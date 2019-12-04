# ******************************************************************************
# Copyright 2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License")
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************

import numpy as np
import pyseal
import time
import random

from util import print_parameters, print_vector, print_example_banner, print_matrix

from pyseal import BatchEncoder, \
                  EncryptionParameters, \
                  scheme_type, \
                  SEALContext, \
                  CoeffModulus, \
                  PlainModulus, \
                  KeyGenerator, \
                  Encryptor, \
                  Evaluator, \
                  Decryptor, \
                  CKKSEncoder, \
                  Plaintext, \
                  Ciphertext, \
                  MemoryPoolHandle, \
                  DoubleVec, \
                  sec_level_type ,\
                  ComplexVec, \
                  SmallModulus, \
                  UIntVec


def example_rotation_bfv():
    print_example_banner("Example: Rotation / Rotation in BFV")

    parms = EncryptionParameters(scheme_type.BFV)

    poly_modulus_degree = 8192
    parms.set_poly_modulus_degree(poly_modulus_degree)
    parms.set_coeff_modulus(CoeffModulus.BFVDefault(poly_modulus_degree))
    parms.set_plain_modulus(PlainModulus.Batching(poly_modulus_degree, 20))

    context = SEALContext.Create(parms)
    print_parameters(context)

    keygen = KeyGenerator(context)
    public_key = keygen.public_key()
    secret_key = keygen.secret_key()
    relin_keys = keygen.relin_keys()
    encryptor = Encryptor(context, public_key)
    evaluator = Evaluator(context)
    decryptor = Decryptor(context, secret_key)

    batch_encoder = BatchEncoder(context)
    slot_count = batch_encoder.slot_count()
    row_size = slot_count // 2
    print("Plaintext matrix row size:", row_size)

    pod_matrix = UIntVec([0] * slot_count)
    pod_matrix[0] = 0
    pod_matrix[1] = 1
    pod_matrix[2] = 2
    pod_matrix[3] = 3
    pod_matrix[row_size] = 4
    pod_matrix[row_size + 1] = 5
    pod_matrix[row_size + 2] = 6
    pod_matrix[row_size + 3] = 7

    print("Input plaintext matrix:")
    print_matrix(pod_matrix, row_size)

    plain_matrix = Plaintext()
    print("Encode and encrypt.")
    batch_encoder.encode(pod_matrix, plain_matrix)
    encrypted_matrix = Ciphertext()
    encryptor.encrypt(plain_matrix, encrypted_matrix)
    print("    + Noise budget in fresh encryption:",
          decryptor.invariant_noise_budget(encrypted_matrix), "bits")

    gal_keys = keygen.galois_keys()
    print("Rotate rows 3 steps left.")
    evaluator.rotate_rows_inplace(encrypted_matrix, 3, gal_keys)
    plain_result = Plaintext()
    print("    + Noise budget after rotation:",
          decryptor.invariant_noise_budget(encrypted_matrix), "bits")
    print("    + Decrypt and decode ...... Correct.")
    decryptor.decrypt(encrypted_matrix, plain_result)
    batch_encoder.decode(plain_result, pod_matrix)
    print_matrix(pod_matrix, row_size)

    print("Rotate columns.")
    evaluator.rotate_columns_inplace(encrypted_matrix, gal_keys)
    print("    + Noise budget after rotation:",
          decryptor.invariant_noise_budget(encrypted_matrix), "bits")
    print("    + Decrypt and decode ...... Correct.")
    decryptor.decrypt(encrypted_matrix, plain_result)
    batch_encoder.decode(plain_result, pod_matrix)
    print_matrix(pod_matrix, row_size)

    print("Rotate rows 4 steps right.")
    evaluator.rotate_rows_inplace(encrypted_matrix, -4, gal_keys)
    print("    + Noise budget after rotation:",
          decryptor.invariant_noise_budget(encrypted_matrix), " its")
    print("    + Decrypt and decode ...... Correct.")
    decryptor.decrypt(encrypted_matrix, plain_result)
    batch_encoder.decode(plain_result, pod_matrix)
    print_matrix(pod_matrix, row_size)


def example_rotation_ckks():
    print_example_banner('Example 5 CKKS rotation')

    parms = EncryptionParameters(scheme_type.CKKS)
    poly_modulus_degree = 8192
    parms.set_poly_modulus_degree(poly_modulus_degree)
    parms.set_coeff_modulus(CoeffModulus.Create(poly_modulus_degree, [40] * 5))

    context = SEALContext.Create(parms)
    print_parameters(context)

    keygen = KeyGenerator(context)
    public_key = keygen.public_key()
    secret_key = keygen.secret_key()
    relin_keys = keygen.relin_keys()
    gal_keys = keygen.galois_keys()
    encryptor = Encryptor(context, public_key)
    evaluator = Evaluator(context)
    decryptor = Decryptor(context, secret_key)

    encoder = CKKSEncoder(context)

    slot_count = encoder.slot_count()
    print('Number of slots:', slot_count)

    input = DoubleVec([x / (slot_count - 1.0) for x in range(slot_count)])
    print('Input vector:')
    print_vector(input, 3, 7)
    print('Evaluating polynomial PI*x^3 + 0.4x + 1 ...')

    scale = 2**50
    print('Encode and encrypt')
    plain = Plaintext()
    encoder.encode(input, scale, plain)
    encrypted = Ciphertext()
    encryptor.encrypt(plain, encrypted)

    rotated = Ciphertext()
    print('Rotate 2 steps left.')
    evaluator.rotate_vector(encrypted, 2, gal_keys, rotated)
    print("Decrypt and decode.")
    decryptor.decrypt(rotated, plain)
    result = DoubleVec()
    encoder.decode(plain, result)
    print_vector(result, 3, 7)


def example_rotation():
    print_example_banner("Example: Rotation")

    example_rotation_bfv()
    example_rotation_ckks()
