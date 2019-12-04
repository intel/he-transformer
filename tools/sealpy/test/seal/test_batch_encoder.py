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
import unittest
import random

from pyseal import \
    BatchEncoder, \
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
    IntVec, \
    KeyGenerator, \
    MemoryPoolHandle, \
    PlainModulus, \
    Plaintext, \
    PublicKey, \
    scheme_type, \
    SEALContext, \
    SecretKey, \
    sec_level_type, \
    SmallModulus, \
    UIntVec


class BatchEncoderTest(unittest.TestCase):
    def test_batch_unbatch_uint_vector(self):
        parms = EncryptionParameters(scheme_type.BFV)
        parms.set_poly_modulus_degree(64)
        parms.set_coeff_modulus(CoeffModulus.Create(64, [60]))
        parms.set_plain_modulus(257)

        context = SEALContext.Create(parms, False, sec_level_type.none)
        self.assertTrue(
            context.first_context_data().qualifiers().using_batching)

        batch_encoder = BatchEncoder(context)
        self.assertEqual(64, batch_encoder.slot_count())

        plain_vec = UIntVec(range(batch_encoder.slot_count()))
        plain = Plaintext()
        batch_encoder.encode(plain_vec, plain)
        plain_vec2 = UIntVec()
        batch_encoder.decode(plain, plain_vec2)
        self.assertTrue(plain_vec == plain_vec2)

        for i in range(batch_encoder.slot_count()):
            plain_vec[i] = 5
        batch_encoder.encode(plain_vec, plain)
        self.assertTrue(plain.to_string() == "5")
        batch_encoder.decode(plain, plain_vec2)
        self.assertTrue(plain_vec == plain_vec2)

        short_plain_vec = UIntVec(range(20))
        batch_encoder.encode(short_plain_vec, plain)
        short_plain_vec2 = UIntVec()
        batch_encoder.decode(plain, short_plain_vec2)
        self.assertEqual(20, len(short_plain_vec))
        self.assertEqual(64, len(short_plain_vec2))
        for i in range(20):
            self.assertEqual(short_plain_vec[i], short_plain_vec2[i])
        for i in range(20, batch_encoder.slot_count()):
            self.assertEqual(0, short_plain_vec2[i])

    def test_batch_unbatch_int_vector(self):
        parms = EncryptionParameters(scheme_type.BFV)
        parms.set_poly_modulus_degree(64)
        parms.set_coeff_modulus(CoeffModulus.Create(64, [60]))
        parms.set_plain_modulus(257)

        context = SEALContext.Create(parms, False, sec_level_type.none)
        self.assertTrue(
            context.first_context_data().qualifiers().using_batching)

        batch_encoder = BatchEncoder(context)
        self.assertEqual(64, batch_encoder.slot_count())

        plain_vec = IntVec([0] * batch_encoder.slot_count())
        for i in range(batch_encoder.slot_count()):
            plain_vec[i] = i * (1 - 2 * (i % 2))

        plain = Plaintext()
        batch_encoder.encode(plain_vec, plain)
        plain_vec2 = IntVec()
        batch_encoder.decode(plain, plain_vec2)
        self.assertTrue(plain_vec == plain_vec2)

        for i in range(batch_encoder.slot_count()):
            plain_vec[i] = -5
        batch_encoder.encode(plain_vec, plain)
        self.assertTrue(plain.to_string() == "FC")
        batch_encoder.decode(plain, plain_vec2)
        self.assertTrue(plain_vec == plain_vec2)

        short_plain_vec = IntVec([0] * 20)
        for i in range(20):
            short_plain_vec[i] = i * (1 - 2 * (i % 2))
        batch_encoder.encode(short_plain_vec, plain)
        short_plain_vec2 = IntVec()
        batch_encoder.decode(plain, short_plain_vec2)
        self.assertEqual(20, len(short_plain_vec))
        self.assertEqual(64, len(short_plain_vec2))
        for i in range(20):
            self.assertEqual(short_plain_vec[i], short_plain_vec2[i])
        for i in range(20, batch_encoder.slot_count()):
            self.assertEqual(0, short_plain_vec2[i])

    def test_batch_unbatch_plaintext(self):
        parms = EncryptionParameters(scheme_type.BFV)
        parms.set_poly_modulus_degree(64)
        parms.set_coeff_modulus(CoeffModulus.Create(64, [60]))
        parms.set_plain_modulus(257)

        context = SEALContext.Create(parms, False, sec_level_type.none)
        self.assertTrue(
            context.first_context_data().qualifiers().using_batching)

        batch_encoder = BatchEncoder(context)
        self.assertEqual(64, batch_encoder.slot_count())
        plain = Plaintext(batch_encoder.slot_count())
        for i in range(batch_encoder.slot_count()):
            plain[i] = i

        batch_encoder.encode(plain)
        batch_encoder.decode(plain)
        for i in range(batch_encoder.slot_count()):
            self.assertTrue(plain[i] == i)

        for i in range(batch_encoder.slot_count()):
            plain[i] = 5

        batch_encoder.encode(plain)
        self.assertTrue(plain.to_string() == "5")
        batch_encoder.decode(plain)
        for i in range(batch_encoder.slot_count()):
            self.assertEqual(5, plain[i])

        short_plain = Plaintext(20)
        for i in range(20):
            short_plain[i] = i
        batch_encoder.encode(short_plain)
        batch_encoder.decode(short_plain)
        for i in range(20):
            self.assertTrue(short_plain[i] == i)
        for i in range(20, batch_encoder.slot_count()):
            self.assertTrue(short_plain[i] == 0)
