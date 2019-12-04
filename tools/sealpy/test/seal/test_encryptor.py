# ******************************************************************************
# Copyright 2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License")
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http:# www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************

import numpy as np
import pyseal
import unittest
import random
from datetime import datetime

from pyseal import \
    Ciphertext, \
    CKKSEncoder, \
    CoeffModulus, \
    ComplexVec, \
    Decryptor, \
    DoubleVec, \
    EncryptionParameters, \
    Encryptor, \
    Evaluator, \
    GaloisKeys, \
    KeyGenerator, \
    MemoryPoolHandle, \
    PlainModulus, \
    Plaintext, \
    PublicKey, \
    scheme_type, \
    SEALContext, \
    SecretKey, \
    sec_level_type, \
    SmallModulus


class EncryptorTest(unittest.TestCase):
    def test_ckks_encrypt_to_zero_decrypt(self):
        parms = EncryptionParameters(scheme_type.CKKS)
        parms.set_poly_modulus_degree(64)
        parms.set_coeff_modulus(CoeffModulus.Create(64, [40, 40, 40]))

        context = SEALContext.Create(parms, True, sec_level_type.none)
        keygen = KeyGenerator(context)

        encryptor = Encryptor(context, keygen.public_key())
        decryptor = Decryptor(context, keygen.secret_key())
        encoder = CKKSEncoder(context)

        ct = Ciphertext()
        encryptor.encrypt_zero(ct)
        self.assertFalse(ct.is_transparent())
        self.assertTrue(ct.is_ntt_form())
        self.assertAlmostEqual(ct.scale, 1.0)
        ct.scale = 2.0**20
        pt = Plaintext()
        decryptor.decrypt(ct, pt)

        res = ComplexVec()
        encoder.decode(pt, res)

        for val in res:
            self.assertAlmostEqual(val.real, 0.0, delta=0.01)
            self.assertAlmostEqual(val.imag, 0.0, delta=0.01)

        next_parms = context.first_context_data().next_context_data().parms_id(
        )
        encryptor.encrypt_zero(next_parms, ct)
        self.assertFalse(ct.is_transparent())
        self.assertTrue(ct.is_ntt_form())
        self.assertAlmostEqual(ct.scale, 1.0)
        ct.scale = 2.0**20
        self.assertEqual(ct.parms_id, next_parms)
        decryptor.decrypt(ct, pt)
        self.assertEqual(pt.parms_id, next_parms)

        encoder.decode(pt, res)
        for val in res:
            self.assertAlmostEqual(val.real, 0.0, delta=0.01)
            self.assertAlmostEqual(val.imag, 0.0, delta=0.01)

    def test_ckks_encrypt_decrypt(self):
        parms = EncryptionParameters(scheme_type.CKKS)
        # input consists of ones
        slot_size = 32
        parms.set_poly_modulus_degree(2 * slot_size)
        parms.set_coeff_modulus(
            CoeffModulus.Create(2 * slot_size, [40, 40, 40, 40]))

        context = SEALContext.Create(parms, True, sec_level_type.none)
        keygen = KeyGenerator(context)

        encoder = CKKSEncoder(context)
        encryptor = Encryptor(context, keygen.public_key())
        decryptor = Decryptor(context, keygen.secret_key())

        encrypted = Ciphertext()
        plain = Plaintext()
        plainRes = Plaintext()

        input = ComplexVec([1.0] * slot_size)
        output = ComplexVec([0.0] * slot_size)
        delta = (1 << 16)

        encoder.encode(input, context.first_parms_id(), delta, plain)

        encryptor.encrypt(plain, encrypted)

        # check correctness of encryption
        self.assertTrue(encrypted.parms_id == context.first_parms_id())

        decryptor.decrypt(encrypted, plainRes)
        encoder.decode(plainRes, output)

        for i in range(slot_size):
            tmp = abs(input[i].real - output[i].real)
            self.assertTrue(tmp < 0.5)

        # input consists of zeros
        slot_size = 32
        parms.set_poly_modulus_degree(2 * slot_size)
        parms.set_coeff_modulus(
            CoeffModulus.Create(2 * slot_size, [40, 40, 40, 40]))

        context = SEALContext.Create(parms, False, sec_level_type.none)
        keygen = KeyGenerator(context)

        encoder = CKKSEncoder(context)
        encryptor = Encryptor(context, keygen.public_key())
        decryptor = Decryptor(context, keygen.secret_key())

        encrypted = Ciphertext()
        plain = Plaintext()
        plainRes = Plaintext()

        input = ComplexVec([0.0] * slot_size)
        output = ComplexVec([0.0] * slot_size)
        delta = (1 << 16)

        encoder.encode(input, context.first_parms_id(), delta, plain)
        encryptor.encrypt(plain, encrypted)

        # check correctness of encryption
        self.assertTrue(encrypted.parms_id == context.first_parms_id())

        decryptor.decrypt(encrypted, plainRes)
        encoder.decode(plainRes, output)

        for i in range(slot_size):
            tmp = abs(input[i].real - output[i].real)
            self.assertTrue(tmp < 0.5)

        #  Input is a random mix of positive and negative integers
        slot_size = 64
        parms.set_poly_modulus_degree(2 * slot_size)
        parms.set_coeff_modulus(
            CoeffModulus.Create(2 * slot_size, [60, 60, 60]))

        context = SEALContext.Create(parms, False, sec_level_type.none)
        keygen = KeyGenerator(context)

        encoder = CKKSEncoder(context)
        encryptor = Encryptor(context, keygen.public_key())
        decryptor = Decryptor(context, keygen.secret_key())

        encrypted = Ciphertext()
        plain = Plaintext()
        plainRes = Plaintext()

        input = ComplexVec([0.0] * slot_size)
        output = ComplexVec([0.0] * slot_size)

        random.seed(datetime.now())
        input_bound = 1 << 30
        delta = (1 << 50)

        for round in range(100):
            for i in range(slot_size):
                input[i] = random.randint(-input_bound, input_bound)

            encoder.encode(input, context.first_parms_id(), delta, plain)
            encryptor.encrypt(plain, encrypted)

            # check correctness of encryption
            self.assertTrue(encrypted.parms_id == context.first_parms_id())

            decryptor.decrypt(encrypted, plainRes)
            encoder.decode(plainRes, output)

            for i in range(slot_size):
                tmp = abs(input[i].real - output[i].real)
                self.assertTrue(tmp < 0.5)

        #  Input is a random mix of positive and negative integers
        slot_size = 32
        parms.set_poly_modulus_degree(128)
        parms.set_coeff_modulus(CoeffModulus.Create(128, [60, 60, 60]))

        context = SEALContext.Create(parms, False, sec_level_type.none)
        keygen = KeyGenerator(context)

        encoder = CKKSEncoder(context)
        encryptor = Encryptor(context, keygen.public_key())
        decryptor = Decryptor(context, keygen.secret_key())

        encrypted = Ciphertext()
        plain = Plaintext()
        plainRes = Plaintext()

        input = ComplexVec([0.0] * slot_size)
        output = ComplexVec([0.0] * slot_size)

        random.seed(datetime.now())
        input_bound = 1 << 30
        delta = (1 << 60)

        for round in range(100):
            for i in range(slot_size):
                input[i] = random.randint(-input_bound, input_bound)

            encoder.encode(input, context.first_parms_id(), delta, plain)
            encryptor.encrypt(plain, encrypted)

            # check correctness of encryption
            self.assertTrue(encrypted.parms_id == context.first_parms_id())

            decryptor.decrypt(encrypted, plainRes)
            encoder.decode(plain, output)

            for i in range(slot_size):
                tmp = abs(input[i].real - output[i].real)
                self.assertTrue(tmp < 0.5)

        #  Encrypt at lower level
        slot_size = 32
        parms.set_poly_modulus_degree(2 * slot_size)
        parms.set_coeff_modulus(
            CoeffModulus.Create(2 * slot_size, [40, 40, 40, 40]))

        context = SEALContext.Create(parms, True, sec_level_type.none)
        keygen = KeyGenerator(context)

        encoder = CKKSEncoder(context)
        encryptor = Encryptor(context, keygen.public_key())
        decryptor = Decryptor(context, keygen.secret_key())

        encrypted = Ciphertext()
        plain = Plaintext()
        plainRes = Plaintext()

        input = ComplexVec([1.0] * slot_size)
        output = ComplexVec([0.0] * slot_size)

        delta = (1 << 16)

        first_context_data = context.first_context_data()
        #self.assertNotEqual(nptr, first_context_data.get())
        second_context_data = first_context_data.next_context_data()
        #self.assertNotEqual(nptr, second_context_data.get())
        second_parms_id = second_context_data.parms_id()

        encoder.encode(input, second_parms_id, delta, plain)
        encryptor.encrypt(plain, encrypted)
        #  Check correctness of encryption
        self.assertTrue(encrypted.parms_id == second_parms_id)

        decryptor.decrypt(encrypted, plainRes)
        encoder.decode(plainRes, output)

        for i in range(slot_size):
            tmp = abs(input[i].real - output[i].real)
            self.assertTrue(tmp < 0.5)