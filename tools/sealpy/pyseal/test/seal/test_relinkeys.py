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
    RelinKeys, \
    scheme_type, \
    SEALContext, \
    SecretKey, \
    sec_level_type, \
    SmallModulus


class RelinKeysTest(unittest.TestCase):
    def test_relinkeys_save_load(self):
        parms = EncryptionParameters(scheme_type.BFV)
        parms.set_poly_modulus_degree(64)
        parms.set_plain_modulus(1 << 6)
        parms.set_coeff_modulus(CoeffModulus.Create(64, [60]))

        context = SEALContext.Create(parms, False, sec_level_type.none)
        keygen = KeyGenerator(context)

        keys = RelinKeys()
        test_keys = RelinKeys()
        keys = keygen.relin_keys()
        stream = keys.save()
        test_keys.load(context, stream)
        self.assertEqual(keys.size(), test_keys.size())
        self.assertTrue(keys.parms_id() == test_keys.parms_id())

        for j in range(test_keys.size()):
            for i in range(len(test_keys.key(j + 2))):
                self.assertEqual(
                    keys.key(j + 2)[i].data().size(),
                    test_keys.key(j + 2)[i].data().size())
                self.assertEqual(
                    keys.key(j + 2)[i].data().uint64_count(),
                    test_keys.key(j + 2)[i].data().uint64_count())
                for idx in range(keys.key(j + 2)[i].data().uint64_count()):
                    self.assertEqual(
                        keys.key(j + 2)[i].data()[idx],
                        test_keys.key(j + 2)[i].data()[idx])

        # larger poly modulus degree
        parms = EncryptionParameters(scheme_type.BFV)
        parms.set_poly_modulus_degree(256)
        parms.set_plain_modulus(1 << 6)
        parms.set_coeff_modulus(CoeffModulus.Create(256, [60, 50]))

        context = SEALContext.Create(parms, False, sec_level_type.none)
        keygen = KeyGenerator(context)

        keys = RelinKeys()
        test_keys = RelinKeys()
        keys = keygen.relin_keys()
        stream = keys.save()
        test_keys.load(context, stream)
        self.assertEqual(keys.size(), test_keys.size())
        self.assertTrue(keys.parms_id() == test_keys.parms_id())
        for j in range(test_keys.size()):
            for i in range(len(test_keys.key(j + 2))):
                self.assertEqual(
                    keys.key(j + 2)[i].data().size(),
                    test_keys.key(j + 2)[i].data().size())
                self.assertEqual(
                    keys.key(j + 2)[i].data().uint64_count(),
                    test_keys.key(j + 2)[i].data().uint64_count())
                for idx in range(keys.key(j + 2)[i].data().uint64_count()):
                    self.assertEqual(
                        keys.key(j + 2)[i].data()[idx],
                        test_keys.key(j + 2)[i].data()[idx])