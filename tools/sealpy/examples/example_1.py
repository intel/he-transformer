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


def bfv_basics():
    print_example_banner('Example 1 BFV Basics')
    parms = EncryptionParameters(scheme_type.BFV)

    poly_modulus_degree = 4096
    parms.set_poly_modulus_degree(poly_modulus_degree)
    parms.set_coeff_modulus(CoeffModulus.BFVDefault(poly_modulus_degree))
    parms.set_plain_modulus(256)

    context = SEALContext.Create(parms)
    print('Set encryption parameters and print')
    print_parameters(context)

    print('~~~~~~ A naive way to calculate 2(x^2+1)(x+1)^2. ~~~~~~')

    keygen = KeyGenerator(context)
    public_key = keygen.public_key()
    secret_key = keygen.secret_key()

    encryptor = Encryptor(context, public_key)
    evaluator = Evaluator(context)
    decryptor = Decryptor(context, secret_key)

    x = 6
    x_plain = Plaintext(str(x))
    print('Express x = ' + str(x) + ' as a plaintext polynomial 0x' +
          x_plain.to_string() + '.')

    x_encrypted = Ciphertext()
    print('Encrypt x_plain to x_encrypted.')
    encryptor.encrypt(x_plain, x_encrypted)

    print('size of freshly encrypted x:', x_encrypted.size())
    print('noise budget in freshly encrypted x:',
          decryptor.invariant_noise_budget(x_encrypted), 'bits')

    x_decrypted = Plaintext()
    print("    + decryption of x_encrypted: ", end='')
    decryptor.decrypt(x_encrypted, x_decrypted)
    print("0x" + str(x_decrypted.to_string()), " ...... Correct.")

    print("Compute x_sq_plus_one (x^2+1).")
    x_sq_plus_one = Ciphertext()
    evaluator.square(x_encrypted, x_sq_plus_one)
    plain_one = Plaintext("1")
    evaluator.add_plain_inplace(x_sq_plus_one, plain_one)

    print("    + size of x_sq_plus_one:", x_sq_plus_one.size())
    print("    + noise budget in x_sq_plus_one:",
          decryptor.invariant_noise_budget(x_sq_plus_one), "bits")

    decrypted_result = Plaintext()
    print("    + decryption of x_sq_plus_one: ", end='')
    decryptor.decrypt(x_sq_plus_one, decrypted_result)
    print("0x" + str(decrypted_result.to_string()), "...... Correct.")

    print("Compute x_plus_one_sq ((x+1)^2).")
    x_plus_one_sq = Ciphertext()
    evaluator.add_plain(x_encrypted, plain_one, x_plus_one_sq)
    evaluator.square_inplace(x_plus_one_sq)
    print("    + size of x_plus_one_sq:", x_plus_one_sq.size())
    print("    + noise budget in x_plus_one_sq:",
          decryptor.invariant_noise_budget(x_plus_one_sq), "bits")
    print("    + decryption of x_plus_one_sq: ", end='')
    decryptor.decrypt(x_plus_one_sq, decrypted_result)
    print("0x" + str(decrypted_result.to_string()), "...... Correct.")

    print("Compute encrypted_result (2(x^2+1)(x+1)^2).")
    encrypted_result = Ciphertext()
    plain_two = Plaintext("2")
    evaluator.multiply_plain_inplace(x_sq_plus_one, plain_two)
    evaluator.multiply(x_sq_plus_one, x_plus_one_sq, encrypted_result)
    print("    + size of encrypted_result:", encrypted_result.size())
    print("    + noise budget in encrypted_result:",
          decryptor.invariant_noise_budget(encrypted_result), "bits")
    print("NOTE: Decryption can be incorrect if noise budget is zero.")

    print("~~~~~~ A better way to calculate 2(x^2+1)(x+1)^2. ~~~~~~")

    print("Generate relinearization keys.")
    relin_keys = keygen.relin_keys()

    print(
        "Compute and relinearize x_squared (x^2), then compute x_sq_plus_one (x^2+1)"
    )
    x_squared = Ciphertext()
    evaluator.square(x_encrypted, x_squared)
    print("    + size of x_squared:", x_squared.size())
    evaluator.relinearize_inplace(x_squared, relin_keys)
    print("    + size of x_squared (after relinearization):", x_squared.size())
    evaluator.add_plain(x_squared, plain_one, x_sq_plus_one)
    print("    + noise budget in x_sq_plus_one:",
          decryptor.invariant_noise_budget(x_sq_plus_one), "bits")
    print("    + decryption of x_sq_plus_one: ", end='')
    decryptor.decrypt(x_sq_plus_one, decrypted_result)
    print("0x" + str(decrypted_result.to_string()), "...... Correct.")

    x_plus_one = Ciphertext()
    print(
        "Compute x_plus_one (x+1), then compute and relinearize x_plus_one_sq ((x+1)^2)."
    )
    evaluator.add_plain(x_encrypted, plain_one, x_plus_one)
    evaluator.square(x_plus_one, x_plus_one_sq)
    print("    + size of x_plus_one_sq:", x_plus_one_sq.size())
    evaluator.relinearize_inplace(x_plus_one_sq, relin_keys)
    print("    + noise budget in x_plus_one_sq:",
          decryptor.invariant_noise_budget(x_plus_one_sq), "bits")
    print("    + decryption of x_plus_one_sq: ", end='')
    decryptor.decrypt(x_plus_one_sq, decrypted_result)
    print("0x" + str(decrypted_result.to_string()), "...... Correct.")

    print("Compute and relinearize encrypted_result (2(x^2+1)(x+1)^2).")
    evaluator.multiply_plain_inplace(x_sq_plus_one, plain_two)
    evaluator.multiply(x_sq_plus_one, x_plus_one_sq, encrypted_result)
    print("    + size of encrypted_result:", encrypted_result.size())
    evaluator.relinearize_inplace(encrypted_result, relin_keys)
    print("    + size of encrypted_result (after relinearization):",
          encrypted_result.size())
    print("    + noise budget in encrypted_result:",
          decryptor.invariant_noise_budget(encrypted_result), "bits")

    print("NOTE: Notice the increase in remaining noise budget.")

    print("Decrypt encrypted_result (2(x^2+1)(x+1)^2).")
    decryptor.decrypt(encrypted_result, decrypted_result)
    print(
        "    + decryption of 2(x^2+1)(x+1)^2 = 0x" + str(
            decrypted_result.to_string()), "...... Correct.")
