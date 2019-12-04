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


def levels():
    print_example_banner('Example 3 levels')

    parms = EncryptionParameters(scheme_type.BFV)

    poly_modulus_degree = 8192
    parms.set_poly_modulus_degree(poly_modulus_degree)
    parms.set_coeff_modulus(
        CoeffModulus.Create(poly_modulus_degree, [50, 30, 30, 50, 50]))
    parms.set_plain_modulus(1 << 20)

    context = SEALContext.Create(parms)
    print_parameters(context)

    print('Print the modulus switching chain.')

    context_data = context.key_context_data()
    print(' ---> Level (chain index): ', context_data.chain_index(), end='')
    print('... key_context_data()')
    print('    parms_id: ', context_data.parms_id())
    print('    coeff_modulus_primes: ')
    for prime in context_data.parms().coeff_modulus():
        print(hex(prime.value()), end=' ')
    print('\n')

    # Iterate over remaining (data) levels
    context_data = context.first_context_data()
    while context_data:
        print('\\')
        print(' \\', end='')
        print('---> Level (chain index): ', context_data.chain_index(), end='')
        if context_data.parms_id() == context.first_parms_id():
            print('..... first_context_data()')
        elif context_data.parms_id() == context.last_parms_id():
            print('..... last_context_data()')
        else:
            print('')
        print('   parms_id', [hex(x) for x in context_data.parms_id()])
        print('   coeff_modulus primes: ', end='')
        for prime in context_data.parms().coeff_modulus():
            print(hex(prime.value()), end=' ')
        print('\n')

        # Step forward in the chain
        context_data = context_data.next_context_data()

    print('End of chain reached')

    keygen = KeyGenerator(context)
    public_key = keygen.public_key()
    secret_key = keygen.secret_key()
    relin_keys = keygen.relin_keys()
    galois_keys = keygen.galois_keys()

    print("Print the parameter IDs of generated elements.")
    print("    + public_key:  ", [hex(x) for x in public_key.parms_id()])
    print("    + secret_key:  ", [hex(x) for x in secret_key.parms_id()])
    print("    + relin_keys:  ", [hex(x) for x in relin_keys.parms_id()])
    print("    + galois_keys: ", [hex(x) for x in galois_keys.parms_id()])

    encryptor = Encryptor(context, public_key)
    evaluator = Evaluator(context)
    decryptor = Decryptor(context, secret_key)

    plain = Plaintext('1x^3 + 2x^2 + 3x^1 + 4')
    encrypted = Ciphertext()
    encryptor.encrypt(plain, encrypted)

    print("    + plain:       ", plain.parms_id, " (not set in BFV)")
    print("    + encrypted:   ", encrypted.parms_id)

    print("Perform modulus switching on encrypted and print.")
    context_data = context.first_context_data()
    print("---->")
    while context_data.next_context_data() is not None:
        print(" Level (chain index): ", context_data.chain_index())
        print("      parms_id of encrypted: ", encrypted.parms_id)
        print("      Noise budget at this level: ",
              decryptor.invariant_noise_budget(encrypted), " bits")
        print("\\")
        print(" \\-->", end='')
        evaluator.mod_switch_to_next_inplace(encrypted)
        context_data = context_data.next_context_data()
    print(" Level (chain index): ", context_data.chain_index())
    print("      parms_id of encrypted: ", encrypted.parms_id)
    print("      Noise budget at this level: ",
          decryptor.invariant_noise_budget(encrypted), " bits")
    print("\\")
    print(" \\-->", end='')
    print(" End of chain reached")

    print("Decrypt still works after modulus switching.")
    decryptor.decrypt(encrypted, plain)
    print("    + Decryption of encrypted: ", plain.to_string(), end='')
    print(" ...... Correct.")

    print("Computation is more efficient with modulus switching.")
    print("Compute the fourth power.")
    encryptor.encrypt(plain, encrypted)
    print("    + Noise budget before squaring:        ",
          decryptor.invariant_noise_budget(encrypted), "bits")
    evaluator.square_inplace(encrypted)
    evaluator.relinearize_inplace(encrypted, relin_keys)
    print("    + Noise budget after squaring:         ",
          decryptor.invariant_noise_budget(encrypted), "bits")

    evaluator.mod_switch_to_next_inplace(encrypted)
    print("    + Noise budget after modulus switching:",
          decryptor.invariant_noise_budget(encrypted), "bits")

    evaluator.square_inplace(encrypted)
    evaluator.relinearize_inplace(encrypted, relin_keys)
    print("    + Noise budget after squaring:         ",
          decryptor.invariant_noise_budget(encrypted), "bits")
    evaluator.mod_switch_to_next_inplace(encrypted)
    print("    + Noise budget after modulus switching:",
          decryptor.invariant_noise_budget(encrypted), "bits")

    decryptor.decrypt(encrypted, plain)
    print("    + Decryption of fourth power (hexadecimal) ...... Correct.")
    print("    ", plain.to_string())

    context = SEALContext.Create(parms, False)

    print("Optionally disable modulus switching chain expansion.")
    print("Print the modulus switching chain.")
    print("---->", end='')

    context_data = context.key_context_data()
    while context_data is not None:

        print(" Level (chain index): ", context_data.chain_index())
        print("      parms_id: ", context_data.parms_id())
        print("      coeff_modulus primes: ", end='')
        for prime in context_data.parms().coeff_modulus():
            print(prime.value(), end=' ')
        print("\n\\")
        print(" \\-->", end='')

        context_data = context_data.next_context_data()

    print(" End of chain reached")