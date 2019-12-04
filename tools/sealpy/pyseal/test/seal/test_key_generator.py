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
    SmallModulus, \
    UIntVec


class KeyGeneratorTest(unittest.TestCase):
    def test_ckks_key_generation(self):
        parms = EncryptionParameters(scheme_type.CKKS)
        parms.set_poly_modulus_degree(64)
        parms.set_coeff_modulus(CoeffModulus.Create(64, [60]))

        context = SEALContext.Create(parms, False, sec_level_type.none)
        keygen = KeyGenerator(context)

        evk = keygen.relin_keys()
        self.assertTrue(evk.parms_id() == context.key_parms_id())
        self.assertEqual(1, len(evk.key(2)))

        for j in range(evk.size()):
            for i in range(len(evk.key(j + 2))):
                for k in range(evk.key(j + 2)[i].data().size()):
                    all_zeros = True
                    for idx in range(
                            evk.key(j + 2)[i].data().poly_modulus_degree() *
                            evk.key(j + 2)[i].data().coeff_mod_count()):
                        if evk.key(j + 2)[i].data().data(k) != 0:
                            all_zeros = False
                            break
                    self.assertFalse(all_zeros)

        galks = keygen.galois_keys()
        self.assertTrue(galks.parms_id() == context.key_parms_id())
        self.assertEqual(1, len(galks.key(3)))
        self.assertEqual(10, galks.size())

        galks = keygen.galois_keys(galois_elts=UIntVec([1, 3, 5, 7]))
        self.assertTrue(galks.parms_id() == context.key_parms_id())
        self.assertTrue(galks.has_key(1))
        self.assertTrue(galks.has_key(3))
        self.assertTrue(galks.has_key(5))
        self.assertTrue(galks.has_key(7))
        self.assertFalse(galks.has_key(9))
        self.assertFalse(galks.has_key(127))
        self.assertEqual(1, len(galks.key(1)))
        self.assertEqual(1, len(galks.key(3)))
        self.assertEqual(1, len(galks.key(5)))
        self.assertEqual(1, len(galks.key(7)))
        self.assertEqual(4, galks.size())

        galks = keygen.galois_keys(galois_elts=UIntVec([1]))
        self.assertTrue(galks.parms_id() == context.key_parms_id())
        self.assertTrue(galks.has_key(1))
        self.assertFalse(galks.has_key(3))
        self.assertFalse(galks.has_key(127))
        self.assertEqual(1, len(galks.key(1)))
        self.assertEqual(1, galks.size())

        galks = keygen.galois_keys(galois_elts=UIntVec([127]))
        self.assertTrue(galks.parms_id() == context.key_parms_id())
        self.assertFalse(galks.has_key(1))
        self.assertTrue(galks.has_key(127))
        self.assertEqual(1, len(galks.key(127)))
        self.assertEqual(1, galks.size())

        # Larger poly modulus
        parms.set_poly_modulus_degree(256)
        parms.set_coeff_modulus(CoeffModulus.Create(256, [60, 30, 30]))

        context = SEALContext.Create(parms, False, sec_level_type.none)
        keygen = KeyGenerator(context)

        evk = keygen.relin_keys()
        self.assertTrue(evk.parms_id() == context.key_parms_id())
        self.assertEqual(2, len(evk.key(2)))
        for j in range(evk.size()):
            for i in range(len(evk.key(j + 2))):
                for k in range(evk.key(j + 2)[i].data().size()):
                    all_zeros = True
                    for idx in range(
                            evk.key(j + 2)[i].data().poly_modulus_degree() *
                            evk.key(j + 2)[i].data().coeff_mod_count()):
                        if evk.key(j + 2)[i].data().data(k) != 0:
                            all_zeros = False
                            break
                    self.assertFalse(all_zeros)

        galks = keygen.galois_keys()
        self.assertTrue(galks.parms_id() == context.key_parms_id())
        self.assertEqual(2, len(galks.key(3)))
        self.assertEqual(14, galks.size())

        galks = keygen.galois_keys(galois_elts=UIntVec([1, 3, 5, 7]))
        self.assertTrue(galks.parms_id() == context.key_parms_id())
        self.assertTrue(galks.has_key(1))
        self.assertTrue(galks.has_key(3))
        self.assertTrue(galks.has_key(5))
        self.assertTrue(galks.has_key(7))
        self.assertFalse(galks.has_key(9))
        self.assertFalse(galks.has_key(511))
        self.assertEqual(2, len(galks.key(1)))
        self.assertEqual(2, len(galks.key(3)))
        self.assertEqual(2, len(galks.key(5)))
        self.assertEqual(2, len(galks.key(7)))
        self.assertEqual(4, galks.size())

        galks = keygen.galois_keys(galois_elts=UIntVec([1]))
        self.assertTrue(galks.parms_id() == context.key_parms_id())
        self.assertTrue(galks.has_key(1))
        self.assertFalse(galks.has_key(3))
        self.assertFalse(galks.has_key(511))
        self.assertEqual(2, len(galks.key(1)))
        self.assertEqual(1, galks.size())

        galks = keygen.galois_keys(galois_elts=UIntVec([511]))
        self.assertTrue(galks.parms_id() == context.key_parms_id())
        self.assertFalse(galks.has_key(1))
        self.assertTrue(galks.has_key(511))
        self.assertEqual(2, len(galks.key(511)))
        self.assertEqual(1, galks.size())
