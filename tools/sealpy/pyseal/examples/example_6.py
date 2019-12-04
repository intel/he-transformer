# ******************************************************************************
# Copyright 2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
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

from util import print_parameters, print_vector

from pyseal import EncryptionParameters, \
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
                  SmallModulus


def ckks_performance_test(context):
    microsec_per_sec = 1000000

    print_parameters(context)

    parms = context.first_context_data().parms()
    poly_modulus_degree = parms.poly_modulus_degree()

    print('Generating secret/public keys: ', end='')
    keygen = KeyGenerator(context)
    print('Done')

    secret_key = keygen.secret_key()
    public_key = keygen.public_key()

    if context.using_keyswitching:
        print('Generating relinearization keys')
        time_start = time.time()
        relin_keys = keygen.relin_keys()
        time_end = time.time()
        time_diff = time_end - time_start
        print('Done [', int(microsec_per_sec * time_diff), ' microseconds]')

        #if not context.first_context_data().qualifiers().using_batching:
        #    print('Given encryption parameters do not support batching')
        #    return

        print("Generating Galois keys: ", end='')
        time_start = time.time()
        gal_keys = keygen.galois_keys()
        time_end = time.time()
        time_diff = time_end - time_start
        print('Done [', int(microsec_per_sec * time_diff), ' microseconds]')

    encryptor = Encryptor(context, public_key)
    decryptor = Decryptor(context, secret_key)
    evaluator = Evaluator(context)
    ckks_encoder = CKKSEncoder(context)

    time_encode_sum = 0
    time_decode_sum = 0
    time_encrypt_sum = 0
    time_decrypt_sum = 0
    time_add_sum = 0
    time_multiply_sum = 0
    time_multiply_plain_sum = 0
    time_square_sum = 0
    time_relinearize_sum = 0
    time_rescale_sum = 0
    time_rotate_one_step_sum = 0
    time_rotate_random_sum = 0
    time_conjugate_sum = 0

    #  How many times to run the test?
    count = 10

    pod_vector = DoubleVec(
        [1.001 * x for x in range(ckks_encoder.slot_count())])

    print('Running tests', end='')
    for i in range(count):
        # Encoding
        plain = Plaintext(
            parms.poly_modulus_degree() * len(parms.coeff_modulus()), 0)

        scale = np.sqrt(parms.coeff_modulus()[-1].value())
        time_start = time.time()
        ckks_encoder.encode(pod_vector, scale, plain)
        time_end = time.time()
        time_encode_sum += time_end - time_start

        # Decoding
        pod_vector2 = DoubleVec([0 * ckks_encoder.slot_count()])
        time_start = time.time()
        ckks_encoder.decode(plain, pod_vector2)
        time_end = time.time()
        time_decode_sum += time_end - time_start

        # Encryption
        encrypted = Ciphertext(context)
        time_start = time.time()
        encryptor.encrypt(plain, encrypted)
        time_end = time.time()
        time_encrypt_sum += time_end - time_start

        # Decryption
        plain2 = Plaintext(poly_modulus_degree, 0)
        time_start = time.time()
        decryptor.decrypt(encrypted, plain2)
        time_end = time.time()
        time_decrypt_sum += time_end - time_start

        # Add
        encrypted1 = Ciphertext(context)
        ckks_encoder.encode(i + 1, plain)
        encryptor.encrypt(plain, encrypted1)
        encrypted2 = Ciphertext(context)
        ckks_encoder.encode(i + 1, plain)
        encryptor.encrypt(plain, encrypted2)
        time_start = time.time()
        evaluator.add_inplace(encrypted1, encrypted1)
        evaluator.add_inplace(encrypted2, encrypted2)
        evaluator.add_inplace(encrypted1, encrypted2)
        time_end = time.time()
        time_add_sum += (time_end - time_start) / 3

        # Multiply
        encrypted1.reserve(3)
        time_start = time.time()
        evaluator.multiply_inplace(encrypted1, encrypted2)
        time_end = time.time()
        time_multiply_sum += time_end - time_start

        # Multiply plain
        time_start = time.time()
        evaluator.multiply_plain_inplace(encrypted2, plain)
        time_end = time.time()
        time_multiply_plain_sum += time_end - time_start

        # Square
        time_start = time.time()
        evaluator.square_inplace(encrypted2)
        time_end = time.time()
        time_square_sum += time_end - time_start

        if context.using_keyswitching:
            # Relinearize
            time_start = time.time()
            evaluator.relinearize_inplace(encrypted1, relin_keys)
            time_end = time.time()
            time_relinearize_sum = time_end - time_start

            # Rescale
            time_start = time.time()
            evaluator.rescale_to_next_inplace(encrypted1)
            time_end = time.time()
            time_rescale_sum += time_end - time_start

            # Rotate Vector
            time_start = time.time()
            evaluator.rotate_vector_inplace(encrypted, 1, gal_keys)
            evaluator.rotate_vector_inplace(encrypted, -1, gal_keys)
            time_end = time.time()
            time_rotate_one_step_sum += (time_end - time_start) / 2.

            # Rotate Vector Random
            random_rotation = random.randint(1, ckks_encoder.slot_count() - 1)
            time_start = time.time()
            evaluator.rotate_vector_inplace(encrypted, random_rotation,
                                            gal_keys)
            time_end = time.time()
            time_rotate_random_sum += time_end - time_start

            # Complex Conjugate
            time_start = time.time()
            evaluator.complex_conjugate_inplace(encrypted, gal_keys)
            time_end = time.time()
            time_conjugate_sum += time_end - time_start

        #Print a dot to indicate progress
        print('.', end='')
    print('Done\n')

    avg_encode = int((time_encode_sum / count) * microsec_per_sec)
    avg_decode = int((time_decode_sum / count) * microsec_per_sec)
    avg_encrypt = int((time_encrypt_sum / count) * microsec_per_sec)
    avg_decrypt = int((time_decrypt_sum / count) * microsec_per_sec)
    avg_add = int((time_add_sum / count) * microsec_per_sec)
    avg_multiply = int((time_multiply_sum / count) * microsec_per_sec)
    avg_multiply_plain = int(
        (time_multiply_plain_sum / count) * microsec_per_sec)
    avg_square = int((time_square_sum / count) * microsec_per_sec)
    avg_relinearize = int((time_relinearize_sum / count) * microsec_per_sec)
    avg_rescale = int((time_rescale_sum / count) * microsec_per_sec)
    avg_rotate_one_step = int(
        (time_rotate_one_step_sum / count) * microsec_per_sec)
    avg_rotate_random = int(
        (time_rotate_random_sum / count) * microsec_per_sec)
    avg_conjugate = int((time_conjugate_sum / count) * microsec_per_sec)

    print("Average encode: ", avg_encode, " microseconds")
    print("Average decode: ", avg_decode, " microseconds")
    print("Average encrypt: ", avg_encrypt, " microseconds")
    print("Average decrypt: ", avg_decrypt, " microseconds")
    print("Average add: ", avg_add, " microseconds")
    print("Average multiply: ", avg_multiply, " microseconds")
    print("Average multiply plain: ", avg_multiply_plain, " microseconds")
    print("Average square: ", avg_square, " microseconds")
    if context.using_keyswitching():
        print("Average relinearize: ", avg_relinearize, " microseconds")
        print("Average rescale: ", avg_rescale, " microseconds")
        print("Average rotate vector one step: ", avg_rotate_one_step,
              " microseconds")
        print("Average rotate vector random: ", avg_rotate_random,
              " microseconds")
        print("Average complex conjugate: ", avg_conjugate, " microseconds\n")


def example_ckks_performance_default():
    print(
        'CKKS Performance Test with Degrees: 1024, 2048, 4096, 8192, 16384, and 32768'
    )
    parms = EncryptionParameters(scheme_type.CKKS)
    poly_modulus_degree = 1024
    parms.set_poly_modulus_degree(poly_modulus_degree)
    parms.set_coeff_modulus(CoeffModulus.BFVDefault(poly_modulus_degree))
    #ckks_performance_test(SEALContext.Create(parms))

    poly_modulus_degree = 2048
    parms.set_poly_modulus_degree(poly_modulus_degree)
    parms.set_coeff_modulus(CoeffModulus.BFVDefault(poly_modulus_degree))
    #ckks_performance_test(SEALContext.Create(parms))

    parms = EncryptionParameters(scheme_type.CKKS)
    poly_modulus_degree = 4096
    parms.set_poly_modulus_degree(poly_modulus_degree)
    parms.set_coeff_modulus(CoeffModulus.BFVDefault(poly_modulus_degree))
    ckks_performance_test(SEALContext.Create(parms))

    poly_modulus_degree = 8192
    parms.set_poly_modulus_degree(poly_modulus_degree)
    parms.set_coeff_modulus(CoeffModulus.BFVDefault(poly_modulus_degree))
    ckks_performance_test(SEALContext.Create(parms))

    poly_modulus_degree = 16384
    parms.set_poly_modulus_degree(poly_modulus_degree)
    parms.set_coeff_modulus(CoeffModulus.BFVDefault(poly_modulus_degree))
    ckks_performance_test(SEALContext.Create(parms))
    ''' Uncomment to run largest test
    poly_modulus_degree = 32768
    parms.set_poly_modulus_degree(poly_modulus_degree)
    parms.set_coeff_modulus(CoeffModulus.BFVDefault(poly_modulus_degree))
    ckks_performance_test(SEALContext.Create(parms))'''
