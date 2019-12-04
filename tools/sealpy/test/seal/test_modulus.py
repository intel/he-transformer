# ******************************************************************************
# Copyright 2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License")
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http:www.apache.org/licenses/LICENSE-2.0
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


class CoeffModulusTest(unittest.TestCase):
    def test_custom_exception(self):

        # Too small poly_modulus_degree
        with self.assertRaisesRegex(ValueError,
                                    'poly_modulus_degree is invalid'):
            CoeffModulus.Create(1, [2])

        # Too large poly_modulus_degree
        with self.assertRaisesRegex(ValueError,
                                    'poly_modulus_degree is invalid'):
            CoeffModulus.Create(65536, [30])

        # Invalid poly_modulus_degree
        with self.assertRaisesRegex(ValueError,
                                    'poly_modulus_degree is invalid'):
            CoeffModulus.Create(1023, [20])

        # Invalid bit-size
        with self.assertRaisesRegex(ValueError, 'bit_sizes is invalid'):
            CoeffModulus.Create(2048, [0])
        with self.assertRaisesRegex(ValueError, 'bit_sizes is invalid'):
            CoeffModulus.Create(2048, [-30])
        with self.assertRaisesRegex(ValueError, 'bit_sizes is invalid'):
            CoeffModulus.Create(2048, [30, -30])

        # Too small primes requested
        with self.assertRaisesRegex(RuntimeError,
                                    'failed to find enough qualifying primes'):
            CoeffModulus.Create(2, [2])
        with self.assertRaisesRegex(RuntimeError,
                                    'failed to find enough qualifying primes'):
            CoeffModulus.Create(2, [3, 3, 3])
        with self.assertRaisesRegex(RuntimeError,
                                    'failed to find enough qualifying primes'):
            CoeffModulus.Create(1024, [8])

    def test_custom(self):
        cm = CoeffModulus.Create(2, [])
        self.assertEqual(0, len(cm))

        cm = CoeffModulus.Create(2, [3])
        self.assertEqual(1, len(cm))
        self.assertEqual(5, cm[0].value())

        cm = CoeffModulus.Create(2, [3, 4])
        self.assertEqual(2, len(cm))
        self.assertEqual(5, cm[0].value())
        self.assertEqual(13, cm[1].value())

        cm = CoeffModulus.Create(2, [3, 5, 4, 5])
        self.assertEqual(4, len(cm))
        self.assertEqual(5, cm[0].value())
        self.assertEqual(17, cm[1].value())
        self.assertEqual(13, cm[2].value())
        self.assertEqual(29, cm[3].value())

        cm = CoeffModulus.Create(32, [30, 40, 30, 30, 40])
        self.assertEqual(5, len(cm))
        # TODO
        #self.assertEqual(30, get_significant_bit_count(cm[0].value()))
        #self.assertEqual(40, get_significant_bit_count(cm[1].value()))
        #self.assertEqual(30, get_significant_bit_count(cm[2].value()))
        #self.assertEqual(30, get_significant_bit_count(cm[3].value()))
        #self.assertEqual(40, get_significant_bit_count(cm[4].value()))
        self.assertEqual(1, cm[0].value() % 64)
        self.assertEqual(1, cm[1].value() % 64)
        self.assertEqual(1, cm[2].value() % 64)
        self.assertEqual(1, cm[3].value() % 64)
        self.assertEqual(1, cm[4].value() % 64)
