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


class GaloisKeysTest(unittest.TestCase):
    def test_galois_keys_save_load(self):
        parms = EncryptionParameters(scheme_type.BFV)
        parms.set_poly_modulus_degree(64)
        parms.set_plain_modulus(65537)
        parms.set_coeff_modulus(CoeffModulus.Create(64, [60]))
        context = SEALContext.Create(parms, False, sec_level_type.none)
        keygen = KeyGenerator(context)

        keys = GaloisKeys()
        test_keys = GaloisKeys()
        stream = keys.save()
        test_keys.unsafe_load(stream)
        self.assertEqual(len(keys.data()), len(test_keys.data()))
        self.assertTrue(keys.parms_id() == test_keys.parms_id())
        self.assertEqual(0, len(keys.data()))

        keys = keygen.galois_keys()
        stream = keys.save()
        test_keys.load(context, stream)
        self.assertEqual(len(keys.data()), len(test_keys.data()))
        self.assertTrue(keys.parms_id() == test_keys.parms_id())

        for j in range(len(test_keys.data())):
            for i in range(len(test_keys.data()[j])):
                self.assertEqual(keys.data()[j][i].data().size(),
                                 test_keys.data()[j][i].data().size())
                self.assertEqual(keys.data()[j][i].data().uint64_count(),
                                 test_keys.data()[j][i].data().uint64_count())
                for idx in range(keys.data()[j][i].data().uint64_count()):
                    self.assertEqual(keys.data()[j][i].data()[idx],
                                     test_keys.data()[j][i].data()[idx])
        self.assertEqual(64, len(keys.data()))

        parms = EncryptionParameters(scheme_type.BFV)
        parms.set_poly_modulus_degree(256)
        parms.set_plain_modulus(65537)
        parms.set_coeff_modulus(CoeffModulus.Create(256, [60, 50]))

        context = SEALContext.Create(parms, False, sec_level_type.none)
        keygen = KeyGenerator(context)

        keys = GaloisKeys()
        test_keys = GaloisKeys()
        stream = keys.save()
        test_keys.unsafe_load(stream)
        self.assertEqual(len(keys.data()), len(test_keys.data()))
        self.assertTrue(keys.parms_id() == test_keys.parms_id())
        self.assertEqual(0, len(keys.data()))

        keys = keygen.galois_keys()
        stream = keys.save()
        test_keys.load(context, stream)
        self.assertEqual(len(keys.data()), len(test_keys.data()))
        self.assertTrue(keys.parms_id() == test_keys.parms_id())
        for j in range(len(test_keys.data())):
            for i in range(len(test_keys.data()[j])):
                self.assertEqual(keys.data()[j][i].data().size(),
                                 test_keys.data()[j][i].data().size())
                self.assertEqual(keys.data()[j][i].data().uint64_count(),
                                 test_keys.data()[j][i].data().uint64_count())
                for idx in range(keys.data()[j][i].data().uint64_count()):
                    self.assertEqual(keys.data()[j][i].data()[idx],
                                     test_keys.data()[j][i].data()[idx])
        self.assertEqual(256, len(keys.data()))
