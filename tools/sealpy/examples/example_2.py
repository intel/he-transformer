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

from util import print_parameters, print_vector, print_matrix, print_example_banner

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
                  IntegerEncoder, \
                  Plaintext, \
                  Ciphertext, \
                  MemoryPoolHandle, \
                  DoubleVec, \
                  sec_level_type ,\
                  ComplexVec, \
                  SmallModulus, \
                  UIntVec


def example_integer_encoder():
    print_example_banner('Example 2: Encoders / Integer Encoder')

    parms = EncryptionParameters(scheme_type.BFV)
    poly_modulus_degree = 4096
    parms.set_poly_modulus_degree(poly_modulus_degree)
    parms.set_coeff_modulus(CoeffModulus.BFVDefault(poly_modulus_degree))

    parms.set_plain_modulus(512)
    context = SEALContext.Create(parms)
    print_parameters(context)

    keygen = KeyGenerator(context)
    public_key = keygen.public_key()
    secret_key = keygen.secret_key()
    encryptor = Encryptor(context, public_key)
    evaluator = Evaluator(context)
    decryptor = Decryptor(context, secret_key)
    encoder = IntegerEncoder(context)

    value1 = 5
    plain1 = encoder.encode(value1)
    print("Encode", value1, "as polynomial", plain1.to_string(), "(plain1),")

    value2 = -7
    plain2 = encoder.encode(value2)
    print("encode", value2, "as polynomial", plain2.to_string(), "(plain2).")

    encrypted1 = Ciphertext()
    encrypted2 = Ciphertext()
    print("Encrypt plain1 to encrypted1 and plain2 to encrypted2.")
    encryptor.encrypt(plain1, encrypted1)
    encryptor.encrypt(plain2, encrypted2)
    print("    + Noise budget in encrypted1:",
          decryptor.invariant_noise_budget(encrypted1), "bits")
    print("    + Noise budget in encrypted2:",
          decryptor.invariant_noise_budget(encrypted2), "bits")

    encryptor.encrypt(plain2, encrypted2)
    encrypted_result = Ciphertext()
    print(
        "Compute encrypted_result = (-encrypted1 + encrypted2) * encrypted2.")
    evaluator.negate(encrypted1, encrypted_result)
    evaluator.add_inplace(encrypted_result, encrypted2)
    evaluator.multiply_inplace(encrypted_result, encrypted2)
    print("    + Noise budget in encrypted_result:",
          decryptor.invariant_noise_budget(encrypted_result), "bits")
    plain_result = Plaintext()
    print("Decrypt encrypted_result to plain_result.")
    decryptor.decrypt(encrypted_result, plain_result)
    print("    + Plaintext polynomial:", plain_result.to_string())

    print("Decode plain_result.")
    print("    + Decoded integer:", encoder.decode_int32(plain_result), end='')
    print("...... Correct.")


def example_batch_encoder():
    print_example_banner('Example 2: Encoders / Batch Encoder')

    parms = EncryptionParameters(scheme_type.BFV)
    poly_modulus_degree = 8192
    parms.set_poly_modulus_degree(poly_modulus_degree)
    parms.set_coeff_modulus(CoeffModulus.BFVDefault(poly_modulus_degree))

    parms.set_plain_modulus(PlainModulus.Batching(poly_modulus_degree, 20))

    context = SEALContext.Create(parms)
    print_parameters(context)

    qualifiers = context.first_context_data().qualifiers()
    print("Batching enabled: ", qualifiers.using_batching)

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
    print("Plaintext matrix row size: ", row_size)

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
    print("Encode plaintext matrix:")
    batch_encoder.encode(pod_matrix, plain_matrix)

    pod_result = UIntVec()
    print("    + Decode plaintext matrix ...... Correct.")
    batch_encoder.decode(plain_matrix, pod_result)
    print_matrix(pod_result, row_size)

    encrypted_matrix = Ciphertext()
    print("Encrypt plain_matrix to encrypted_matrix.")
    encryptor.encrypt(plain_matrix, encrypted_matrix)
    print("    + Noise budget in encrypted_matrix:",
          decryptor.invariant_noise_budget(encrypted_matrix), "bits")

    pod_matrix2 = UIntVec([i % 2 + 1 for i in range(slot_count)])
    plain_matrix2 = Plaintext()
    batch_encoder.encode(pod_matrix2, plain_matrix2)
    print("Second input plaintext matrix:")
    print_matrix(pod_matrix2, row_size)

    print("Sum, square, and relinearize.")
    evaluator.add_plain_inplace(encrypted_matrix, plain_matrix2)
    evaluator.square_inplace(encrypted_matrix)
    evaluator.relinearize_inplace(encrypted_matrix, relin_keys)

    print("    + Noise budget in result:",
          decryptor.invariant_noise_budget(encrypted_matrix), "bits")

    plain_result = Plaintext()
    print("Decrypt and decode result.")
    decryptor.decrypt(encrypted_matrix, plain_result)
    batch_encoder.decode(plain_result, pod_result)
    print("    + Result plaintext matrix ...... Correct.")
    print_matrix(pod_result, row_size)


def example_ckks_encoder():
    print_example_banner('Example 2 CKKS encoder')
    parms = EncryptionParameters(scheme_type.CKKS)

    poly_modulus_degree = 8192
    parms.set_poly_modulus_degree(poly_modulus_degree)
    parms.set_coeff_modulus(
        CoeffModulus.Create(poly_modulus_degree, [40, 40, 40, 40, 40]))

    context = SEALContext.Create(parms)
    print_parameters(context)

    keygen = KeyGenerator(context)
    public_key = keygen.public_key()
    secret_key = keygen.secret_key()
    relin_keys = keygen.relin_keys()

    encryptor = Encryptor(context, public_key)
    evaluator = Evaluator(context)
    decryptor = Decryptor(context, secret_key)

    encoder = CKKSEncoder(context)

    slot_count = encoder.slot_count()
    print('Number of slots:', slot_count)

    input = DoubleVec([0.0, 1.1, 2.2, 3.3])

    print('Input vector:')
    print_vector(input)

    plain = Plaintext()

    scale = 2.0**30
    print('Encode input vector')
    encoder.encode(input, scale, plain)

    output = DoubleVec()
    encoder.decode(plain, output)
    print('Decode input vector')
    print_vector(output)

    encrypted = Ciphertext()
    print('Encrypt input vector, square, and relinearize')
    encryptor.encrypt(plain, encrypted)

    evaluator.square_inplace(encrypted)
    evaluator.relinearize_inplace(encrypted, relin_keys)

    print('Scale in squared input', encrypted.scale, " (",
          np.log2(encrypted.scale), " bits)")

    print('Decrypt and decode')
    decryptor.decrypt(encrypted, plain)
    encoder.decode(plain, output)
    print('Result vector')
    print_vector(output)


def example_encoders():
    print_example_banner('Example 2: Encoders')

    example_integer_encoder()
    example_batch_encoder()
    example_ckks_encoder()
