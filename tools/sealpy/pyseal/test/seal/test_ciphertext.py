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


class CiphertextTest(unittest.TestCase):
    def test_ciphertext_basics(self):
        parms = EncryptionParameters(scheme_type.CKKS)
        parms.set_poly_modulus_degree(2)
        parms.set_coeff_modulus(CoeffModulus.Create(2, [30]))
        context = SEALContext.Create(parms, False, sec_level_type.none)
        ctxt = Ciphertext(context)
        ctxt.reserve(10)
        self.assertEqual(0, ctxt.size())
        self.assertEqual(0, ctxt.uint64_count())
        self.assertEqual(10 * 2 * 1, ctxt.uint64_count_capacity())
        self.assertEqual(2, ctxt.poly_modulus_degree())
        self.assertTrue(ctxt.parms_id == context.first_parms_id())
        self.assertFalse(ctxt.is_ntt_form())
        ptr = ctxt.data()

        ctxt.reserve(5)
        self.assertEqual(0, ctxt.size())
        self.assertEqual(0, ctxt.uint64_count())
        self.assertEqual(5 * 2 * 1, ctxt.uint64_count_capacity())
        self.assertEqual(2, ctxt.poly_modulus_degree())
        # TODO: investigate
        #self.assertTrue(ptr != ctxt.data())
        self.assertTrue(ctxt.parms_id == context.first_parms_id())
        ptr = ctxt.data()

        ctxt.reserve(10)
        self.assertEqual(0, ctxt.size())
        self.assertEqual(0, ctxt.uint64_count())
        self.assertEqual(10 * 2 * 1, ctxt.uint64_count_capacity())
        self.assertEqual(2, ctxt.poly_modulus_degree())
        # TODO: investigate
        #self.assertTrue(ptr != ctxt.data())
        self.assertTrue(ctxt.parms_id == context.first_parms_id())
        self.assertFalse(ctxt.is_ntt_form())
        ptr = ctxt.data()

        ctxt.reserve(2)
        self.assertEqual(0, ctxt.size())
        self.assertEqual(2 * 2 * 1, ctxt.uint64_count_capacity())
        self.assertEqual(0, ctxt.uint64_count())
        self.assertEqual(2, ctxt.poly_modulus_degree())
        # TODO: investigate
        #self.assertTrue(ptr != ctxt.data())
        self.assertTrue(ctxt.parms_id == context.first_parms_id())
        self.assertFalse(ctxt.is_ntt_form())
        ptr = ctxt.data()

        ctxt.reserve(5)
        self.assertEqual(0, ctxt.size())
        self.assertEqual(5 * 2 * 1, ctxt.uint64_count_capacity())
        self.assertEqual(0, ctxt.uint64_count())
        self.assertEqual(2, ctxt.poly_modulus_degree())
        # TODO: investigate
        #self.assertTrue(ptr != ctxt.data())
        self.assertTrue(ctxt.parms_id == context.first_parms_id())
        self.assertFalse(ctxt.is_ntt_form())

        ctxt2 = Ciphertext(ctxt)
        self.assertEqual(ctxt.coeff_mod_count(), ctxt2.coeff_mod_count())
        self.assertEqual(ctxt.is_ntt_form(), ctxt2.is_ntt_form())
        self.assertEqual(ctxt.poly_modulus_degree(),
                         ctxt2.poly_modulus_degree())
        self.assertTrue(ctxt.parms_id == ctxt2.parms_id)
        self.assertEqual(ctxt.poly_modulus_degree(),
                         ctxt2.poly_modulus_degree())
        self.assertEqual(ctxt.size(), ctxt2.size())

        ctxt3 = Ciphertext()
        ctxt3 = ctxt
        self.assertEqual(ctxt.coeff_mod_count(), ctxt3.coeff_mod_count())
        self.assertEqual(ctxt.poly_modulus_degree(),
                         ctxt3.poly_modulus_degree())
        self.assertEqual(ctxt.is_ntt_form(), ctxt3.is_ntt_form())
        self.assertTrue(ctxt.parms_id == ctxt3.parms_id)
        self.assertEqual(ctxt.poly_modulus_degree(),
                         ctxt3.poly_modulus_degree())
        self.assertEqual(ctxt.size(), ctxt3.size())

    def test_save_load_ciphertext(self):
        parms = EncryptionParameters(scheme_type.BFV)
        parms.set_poly_modulus_degree(2)
        parms.set_coeff_modulus(CoeffModulus.Create(2, [30]))
        parms.set_plain_modulus(2)
        context = SEALContext.Create(parms, False, sec_level_type.none)

        ctxt = Ciphertext(context)
        ctxt2 = Ciphertext()
        stream = ctxt.save()
        ctxt2.load(context, stream)
        self.assertTrue(ctxt.parms_id == ctxt2.parms_id)
        self.assertFalse(ctxt.is_ntt_form())
        self.assertFalse(ctxt2.is_ntt_form())

        parms.set_poly_modulus_degree(1024)
        parms.set_coeff_modulus(CoeffModulus.BFVDefault(1024))
        parms.set_plain_modulus(0xF0F0)
        context = SEALContext.Create(parms, False)
        keygen = KeyGenerator(context)
        encryptor = Encryptor(context, keygen.public_key())
        encryptor.encrypt(
            Plaintext(
                "Ax^10 + 9x^9 + 8x^8 + 7x^7 + 6x^6 + 5x^5 + 4x^4 + 3x^3 + 2x^2 + 1"
            ), ctxt)
        stream = ctxt.save()
        ctxt2.load(context, stream)
        self.assertTrue(ctxt.parms_id == ctxt2.parms_id)
        self.assertFalse(ctxt.is_ntt_form())
        self.assertFalse(ctxt2.is_ntt_form())

        for i in range(
                parms.poly_modulus_degree() * len(parms.coeff_modulus()) * 2):
            self.assertEqual(ctxt[i], ctxt2[i])
        #self.assertTrue(ctxt.data() != ctxt2.data())
