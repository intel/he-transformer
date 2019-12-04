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


class PlaintextTest(unittest.TestCase):
    def test_plaintext_basics(self):
        plain = Plaintext(2)
        self.assertEqual(2, plain.capacity())
        self.assertEqual(2, plain.coeff_count())
        self.assertEqual(0, plain.significant_coeff_count())
        self.assertFalse(plain.is_ntt_form())

        plain[0] = 1
        plain[1] = 2

        plain.reserve(10)
        self.assertEqual(10, plain.capacity())
        self.assertEqual(2, plain.coeff_count())
        self.assertEqual(2, plain.significant_coeff_count())
        self.assertEqual(1, plain[0])
        self.assertEqual(2, plain[1])
        self.assertFalse(plain.is_ntt_form())

        plain.resize(5)
        self.assertEqual(10, plain.capacity())
        self.assertEqual(5, plain.coeff_count())
        self.assertEqual(2, plain.significant_coeff_count())
        self.assertEqual(1, plain[0])
        self.assertEqual(2, plain[1])
        self.assertEqual(0, plain[2])
        self.assertEqual(0, plain[3])
        self.assertEqual(0, plain[4])
        self.assertFalse(plain.is_ntt_form())

        plain2 = Plaintext()
        plain2.resize(15)
        self.assertEqual(15, plain2.capacity())
        self.assertEqual(15, plain2.coeff_count())
        self.assertEqual(0, plain2.significant_coeff_count())
        self.assertFalse(plain.is_ntt_form())

        plain2.assign(plain)
        # Note, the relgular plain2 = plain does not call the assignemnt operator
        self.assertEqual(15, plain2.capacity())
        self.assertEqual(5, plain2.coeff_count())
        self.assertEqual(2, plain2.significant_coeff_count())
        self.assertEqual(1, plain2[0])
        self.assertEqual(2, plain2[1])
        self.assertEqual(0, plain2[2])
        self.assertEqual(0, plain2[3])
        self.assertEqual(0, plain2[4])
        self.assertFalse(plain.is_ntt_form())

        plain.parms_id = [1, 2, 3, 4]
        self.assertTrue(plain.is_ntt_form())
        plain2.assign(plain)
        self.assertTrue(plain == plain2)
        plain2.parms_id = pyseal.parms_id_zero
        self.assertFalse(plain2.is_ntt_form())
        self.assertFalse(plain == plain2)
        plain2.parms_id = [1, 2, 3, 5]
        self.assertFalse(plain == plain2)

    def test_save_load_plaintext(self):
        plain = Plaintext()
        plain2 = Plaintext()
        stream = plain.save()
        plain2.unsafe_load(stream)
        self.assertTrue(plain.data() == plain2.data())
        self.assertEqual(0, plain2.capacity())
        self.assertEqual(0, plain2.coeff_count())
        self.assertFalse(plain2.is_ntt_form())

        plain.reserve(20)
        plain.resize(5)
        plain[0] = 1
        plain[1] = 2
        plain[2] = 3
        stream = plain.save()
        plain2.unsafe_load(stream)
        #self.assertTrue(plain.data() != plain2.data())
        self.assertEqual(5, plain2.capacity())
        self.assertEqual(5, plain2.coeff_count())
        self.assertEqual(1, plain2[0])
        self.assertEqual(2, plain2[1])
        self.assertEqual(3, plain2[2])
        self.assertEqual(0, plain2[3])
        self.assertEqual(0, plain2[4])
        self.assertFalse(plain2.is_ntt_form())

        plain.parms_id = [1, 2, 3, 4]
        stream = plain.save()
        plain2.unsafe_load(stream)
        self.assertTrue(plain2.is_ntt_form())
        self.assertTrue(plain2.parms_id == plain.parms_id)

        # BFV
        parms = EncryptionParameters(scheme_type.BFV)
        parms.set_poly_modulus_degree(64)
        parms.set_coeff_modulus(CoeffModulus.Create(64, [30, 30]))

        parms.set_plain_modulus(65537)
        context = SEALContext.Create(parms, False, sec_level_type.none)

        plain.parms_id = pyseal.parms_id_zero
        plain = Plaintext("1x^63 + 2x^62 + Fx^32 + Ax^9 + 1x^1 + 1")
        stream = plain.save()
        plain2.load(context, stream)
        #self.assertTrue(plain.data() != plain2.data())
        self.assertFalse(plain2.is_ntt_form())

        evaluator = Evaluator(context)
        evaluator.transform_to_ntt_inplace(plain, context.first_parms_id())
        stream = plain.save()
        plain2.load(context, stream)
        #self.assertTrue(plain.data() != plain2.data())
        self.assertTrue(plain2.is_ntt_form())

        # CKKS
        parms = EncryptionParameters(scheme_type.CKKS)
        parms.set_poly_modulus_degree(64)
        parms.set_coeff_modulus(CoeffModulus.Create(64, [30, 30]))

        context = SEALContext.Create(parms, False, sec_level_type.none)
        encoder = CKKSEncoder(context)

        values = DoubleVec([0.1, 2.3, 3.4])
        encoder.encode(values, pow(2.0, 20), plain)
        self.assertTrue(plain.is_ntt_form())
        stream = plain.save()
        plain2.load(context, stream)
        #self.assertTrue(plain.data() != plain2.data())
        self.assertTrue(plain2.is_ntt_form())
