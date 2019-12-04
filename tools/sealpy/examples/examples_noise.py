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


def example_ckks_noise_basics():
    print('\nExamples CKKS naive noise budget estimation ')
    parms = EncryptionParameters(scheme_type.CKKS)
    parms.set_poly_modulus_degree(8192)
    parms.set_coeff_modulus(DefaultParams.coeff_modulus_128(8192))

    context = SEALContext.Create(parms)
    print_parameters(context)
    keygen = KeyGenerator(context)
    public_key = keygen.public_key()
    secret_key = keygen.secret_key()
    relin_keys = keygen.relin_keys(DefaultParams.dbc_max())
    encryptor = Encryptor(context, public_key)
    evaluator = Evaluator(context)
    decryptor = Decryptor(context, secret_key)
    encoder = CKKSEncoder(context)

    slot_count = encoder.slot_count()
    #print('Number of slots:', slot_count)

    input = DoubleVec([0.0, 1.1, 2.2, 3.3])
    #print('Input vector:')
    #print_vector(input)

    plain = Plaintext()
    scale = 2.0**60
    encoder.encode(input, scale, plain)

    context_data = context.context_data()
    q = context_data.parms().poly_modulus_degree()

    encrypted = Ciphertext()
    encryptor.encrypt(plain, encrypted)

    # print('Chain index of (encryption parameters of) encrypted:',
    #      context.context_data(encrypted.parms_id).chain_index())

    l = context.context_data(encrypted.parms_id).chain_index()
    current_noise = noise.noise_budget(q, l, scale, noise.fresh())
    print("The fresh noise budget is: " + str(current_noise) + "\n")

    evaluator.square_inplace(encrypted)
    evaluator.relinearize_inplace(encrypted, relin_keys)
    current_noise = noise.noise_budget(
        q, l, scale, noise.mult(4, 4, current_noise, current_noise))
    print("Noise budget after squaring: " + str(current_noise) + "\n")

    # print(
    #     "Current coeff_modulus size:",
    #     context.context_data(
    #         encrypted.parms_id).total_coeff_modulus_bit_count(), "bits")

    print('\nRescaling...\n')
    evaluator.rescale_to_next_inplace(encrypted)

    # print('Chain index of (encryption parameters of) encrypted:',
    #      context.context_data(encrypted.parms_id).chain_index())
    #print(
    #     "Current coeff_modulus size:",
    #    context.context_data(
    #       encrypted.parms_id).total_coeff_modulus_bit_count(), "bits")

    current_noise = noise.noise_budget(16, 16, l,
                                       noise.mult_rescale(current_noise))
    print("Noise budget after rescaling: " + str(current_noise))

    l -= 1
    print('Current level: ' + str(l))

    print('\nSquaring and rescaling ...\n')
    evaluator.square_inplace(encrypted)
    evaluator.relinearize_inplace(encrypted, relin_keys)
    evaluator.rescale_to_next_inplace(encrypted)

    #print('Chain index of (encryption parameters of) encrypted:',
    #      context.context_data(encrypted.parms_id).chain_index())
    # print(
    #     "Current coeff_modulus size:",
    #     context.context_data(
    #         encrypted.parms_id).total_coeff_modulus_bit_count(), "bits")

    current_noise = noise.noise_budget(
        2**6, 2**6, l,
        noise.mult_rescale(noise.mult(4, 4, current_noise, current_noise)))
    print("Noise budget after squaring and rescaling: " + str(current_noise) +
          "\n")

    print('\nRescaling and squaring (no relinearization) ...\n')
    evaluator.rescale_to_next_inplace(encrypted)
    evaluator.square_inplace(encrypted)

    # print('Chain index of (encryption parameters of) encrypted:',
    #      context.context_data(encrypted.parms_id).chain_index())
    print(
        "Current coeff_modulus size:",
        context.context_data(
            encrypted.parms_id).total_coeff_modulus_bit_count(), "bits")

    current_noise = noise.noise_budget(
        2**8, 2**8, l, noise.mult(4, 4, current_noise, current_noise))
    print("Noise budget after recaling and squaring (no relinearization): " +
          str(current_noise) + "\n")

    decryptor.decrypt(encrypted, plain)
    result = DoubleVec()
    encoder.decode(plain, result)
    #print('Eighth powers:')
    #print_vector(result)

    #precise_result = DoubleVec([x**8 for x in input])
    #print('Precise result:')
    #print_vector(precise_result)
