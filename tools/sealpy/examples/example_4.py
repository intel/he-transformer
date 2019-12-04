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

from util import print_parameters, print_vector, print_example_banner

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


def ckks_basics():
    print_example_banner('Example 4 CKKS basics')
    parms = EncryptionParameters(scheme_type.CKKS)
    poly_modulus_degree = 8192
    parms.set_poly_modulus_degree(poly_modulus_degree)
    parms.set_coeff_modulus(
        CoeffModulus.Create(poly_modulus_degree, [60, 40, 40, 60]))

    scale = 2.0**40

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

    input = DoubleVec([x / (slot_count - 1.0) for x in range(slot_count)])
    print('Input vector:')
    print_vector(input, 3, 7)
    print('Evaluating polynomial PI*x^3 + 0.4x + 1 ...')

    plain_coeff3 = Plaintext()
    plain_coeff1 = Plaintext()
    plain_coeff0 = Plaintext()
    encoder.encode(3.14159265, scale, plain_coeff3)
    encoder.encode(0.4, scale, plain_coeff1)
    encoder.encode(1.0, scale, plain_coeff0)

    x_plain = Plaintext()
    print('Encode input vectors')
    encoder.encode(input, scale, x_plain)
    x1_encrypted = Ciphertext()
    encryptor.encrypt(x_plain, x1_encrypted)

    x3_encrypted = Ciphertext()
    print('Compute x^2 and relinearize')
    evaluator.square(x1_encrypted, x3_encrypted)
    evaluator.relinearize_inplace(x3_encrypted, relin_keys)
    print('    + Scale of x^2 before rescale:', np.log2(x3_encrypted.scale),
          'bits')

    print('Rescale x^2')
    evaluator.rescale_to_next_inplace(x3_encrypted)
    print('    + Scale of x^2 after rescale:', np.log2(x3_encrypted.scale),
          'bits')

    print('Compute and rescale PI*x')
    x1_encrypted_coeff3 = Ciphertext()
    evaluator.multiply_plain(x1_encrypted, plain_coeff3, x1_encrypted_coeff3)
    print('    + Scale of PI*x before rescale:',
          np.log2(x1_encrypted_coeff3.scale), 'bits')
    evaluator.rescale_to_next_inplace(x1_encrypted_coeff3)
    print(
        '    + Scale of PI*x after rescale:',
        np.log2(x1_encrypted_coeff3.scale),
        'bits',
        end='\n\n')

    print('Compute, relinearize, and rescasle (PI*x)*x^2')
    evaluator.multiply_inplace(x3_encrypted, x1_encrypted_coeff3)
    evaluator.relinearize_inplace(x3_encrypted, relin_keys)
    print('    + Scale of PI*x^3 before rescale:', np.log2(x3_encrypted.scale),
          'bits')
    evaluator.rescale_to_next_inplace(x3_encrypted)
    print(
        '    + Scale of PI*x^3 after rescale:',
        np.log2(x3_encrypted.scale),
        'bits',
        end='\n\n')

    print('Compute and rescale 0.4*x')
    evaluator.multiply_plain_inplace(x1_encrypted, plain_coeff1)
    print('    + Scale of 0.4*x before rescale:', np.log2(x1_encrypted.scale),
          'bits')
    evaluator.rescale_to_next_inplace(x1_encrypted)
    print(
        '    + Scale of 0.4*x after rescale:',
        np.log2(x1_encrypted.scale),
        'bits',
        end='\n\n')

    print("Parameters used by all three terms are different:")
    print("    + Modulus chain index for x3_encrypted:", end=' ')
    print(context.get_context_data(x3_encrypted.parms_id).chain_index())
    print("    + Modulus chain index for x1_encrypted:", end=' ')
    print(context.get_context_data(x1_encrypted.parms_id).chain_index())
    print("    + Modulus chain index for plain_coeff0:", end=' ')
    print(
        context.get_context_data(plain_coeff0.parms_id).chain_index(),
        end='\n\n')

    print('The exact scales of all three terms are different')
    print("    + Exact scale in PI*x^3:", x3_encrypted.scale)
    print("    + Exact scale in  0.4*x:", x1_encrypted.scale)
    print("    + Exact scale in      1:", plain_coeff0.scale, end='\n\n')

    print('Normalize scales to 2^40')
    x3_encrypted.scale = 2**40
    x1_encrypted.scale = 2**40

    print('Normalize encryption parameters to the lowest level')
    last_parms_id = x3_encrypted.parms_id
    evaluator.mod_switch_to_inplace(x1_encrypted, last_parms_id)
    evaluator.mod_switch_to_inplace(plain_coeff0, last_parms_id)

    print('Compute PI*x^3 + 0.4*x + 1.')
    encrypted_result = Ciphertext()
    evaluator.add(x3_encrypted, x1_encrypted, encrypted_result)
    evaluator.add_plain_inplace(encrypted_result, plain_coeff0)

    print('Decrypt and decode PI*x^3 + 0.4x + 1.')
    print('    + Expected result')
    true_result = DoubleVec([np.pi * x**3 + 0.4 * x + 1 for x in input])
    print_vector(true_result, 3, 7)

    plain_result = Plaintext()
    decryptor.decrypt(encrypted_result, plain_result)
    result = DoubleVec()
    encoder.decode(plain_result, result)
    print("    + Computed result.")
    print_vector(result, 3, 7)
