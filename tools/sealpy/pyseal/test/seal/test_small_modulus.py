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


class SmallModulusTest(unittest.TestCase):
    def test_create_small_modulus(self):
        mod = SmallModulus()
        self.assertTrue(mod.is_zero())
        self.assertEqual(mod.value(), 0)

        self.assertEqual(0, mod.value())
        self.assertEqual(0, mod.bit_count())
        self.assertEqual(1, mod.uint64_count())
        self.assertEqual(0, mod.const_ratio()[0])
        self.assertEqual(0, mod.const_ratio()[1])
        self.assertEqual(0, mod.const_ratio()[2])
        self.assertFalse(mod.is_prime())

        mod = SmallModulus(3)
        self.assertFalse(mod.is_zero())
        self.assertEqual(3, mod.value())
        self.assertEqual(2, mod.bit_count())
        self.assertEqual(1, mod.uint64_count())
        self.assertEqual(6148914691236517205, mod.const_ratio()[0])
        self.assertEqual(6148914691236517205, mod.const_ratio()[1])
        self.assertEqual(1, mod.const_ratio()[2])
        self.assertTrue(mod.is_prime())

        mod2 = SmallModulus(2)
        mod3 = SmallModulus(3)
        self.assertTrue(mod != mod2)
        self.assertTrue(mod == mod3)

        mod = SmallModulus(0)
        self.assertTrue(mod.is_zero())
        self.assertEqual(0, mod.value())
        self.assertEqual(0, mod.bit_count())
        self.assertEqual(1, mod.uint64_count())
        self.assertEqual(0, mod.const_ratio()[0])
        self.assertEqual(0, mod.const_ratio()[1])
        self.assertEqual(0, mod.const_ratio()[2])

        mod = SmallModulus(0xF00000F00000F)
        self.assertFalse(mod.is_zero())
        self.assertEqual(0xF00000F00000F, mod.value())
        self.assertEqual(52, mod.bit_count())
        self.assertEqual(1, mod.uint64_count())
        self.assertEqual(1224979098644774929, mod.const_ratio()[0])
        self.assertEqual(4369, mod.const_ratio()[1])
        self.assertEqual(281470698520321, mod.const_ratio()[2])
        self.assertFalse(mod.is_prime())

        mod = SmallModulus(0xF00000F000079)
        self.assertFalse(mod.is_zero())
        self.assertEqual(0xF00000F000079, mod.value())
        self.assertEqual(52, mod.bit_count())
        self.assertEqual(1, mod.uint64_count())
        self.assertEqual(1224979096621368355, mod.const_ratio()[0])
        self.assertEqual(4369, mod.const_ratio()[1])
        self.assertEqual(1144844808538997, mod.const_ratio()[2])
        self.assertTrue(mod.is_prime())

    def test_compare_small_modulus(self):
        sm0 = SmallModulus()
        sm2 = SmallModulus(2)
        sm5 = SmallModulus(5)
        self.assertFalse(sm0 < sm0)
        self.assertTrue(sm0 == sm0)
        self.assertTrue(sm0 <= sm0)
        self.assertTrue(sm0 >= sm0)
        self.assertFalse(sm0 > sm0)

        self.assertFalse(sm5 < sm5)
        self.assertTrue(sm5 == sm5)
        self.assertTrue(sm5 <= sm5)
        self.assertTrue(sm5 >= sm5)
        self.assertFalse(sm5 > sm5)

        self.assertFalse(sm5 < sm2)
        self.assertFalse(sm5 == sm2)
        self.assertFalse(sm5 <= sm2)
        self.assertTrue(sm5 >= sm2)
        self.assertTrue(sm5 > sm2)

        self.assertTrue(sm5 < 6)
        self.assertFalse(sm5 == 6)
        self.assertTrue(sm5 <= 6)
        self.assertFalse(sm5 >= 6)
        self.assertFalse(sm5 > 6)

    def test_save_load_small_modulus(self):
        mod = SmallModulus()
        stream = mod.save()

        mod2 = SmallModulus()
        mod2.load(stream)
        self.assertEqual(mod2.value(), mod.value())
        self.assertEqual(mod2.bit_count(), mod.bit_count())
        self.assertEqual(mod2.uint64_count(), mod.uint64_count())
        self.assertEqual(mod2.const_ratio()[0], mod.const_ratio()[0])
        self.assertEqual(mod2.const_ratio()[1], mod.const_ratio()[1])
        self.assertEqual(mod2.const_ratio()[2], mod.const_ratio()[2])
        self.assertEqual(mod2.is_prime(), mod.is_prime())

        mod = SmallModulus(3)
        stream = mod.save()
        mod2.load(stream)
        self.assertEqual(mod2.value(), mod.value())
        self.assertEqual(mod2.bit_count(), mod.bit_count())
        self.assertEqual(mod2.uint64_count(), mod.uint64_count())
        self.assertEqual(mod2.const_ratio()[0], mod.const_ratio()[0])
        self.assertEqual(mod2.const_ratio()[1], mod.const_ratio()[1])
        self.assertEqual(mod2.const_ratio()[2], mod.const_ratio()[2])
        self.assertEqual(mod2.is_prime(), mod.is_prime())

        mod = SmallModulus(0xF00000F00000F)
        stream = mod.save()
        mod2.load(stream)
        self.assertEqual(mod2.value(), mod.value())
        self.assertEqual(mod2.bit_count(), mod.bit_count())
        self.assertEqual(mod2.uint64_count(), mod.uint64_count())
        self.assertEqual(mod2.const_ratio()[0], mod.const_ratio()[0])
        self.assertEqual(mod2.const_ratio()[1], mod.const_ratio()[1])
        self.assertEqual(mod2.const_ratio()[2], mod.const_ratio()[2])
        self.assertEqual(mod2.is_prime(), mod.is_prime())

        mod = SmallModulus(0xF00000F000079)
        stream = mod.save()
        mod2.load(stream)
        self.assertEqual(mod2.value(), mod.value())
        self.assertEqual(mod2.bit_count(), mod.bit_count())
        self.assertEqual(mod2.uint64_count(), mod.uint64_count())
        self.assertEqual(mod2.const_ratio()[0], mod.const_ratio()[0])
        self.assertEqual(mod2.const_ratio()[1], mod.const_ratio()[1])
        self.assertEqual(mod2.const_ratio()[2], mod.const_ratio()[2])
        self.assertEqual(mod2.is_prime(), mod.is_prime())
