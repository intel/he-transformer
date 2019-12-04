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


class ContextTest(unittest.TestCase):
    def test_modulus_chain_expansion(self):
        parms = EncryptionParameters(scheme_type.CKKS)
        parms.set_poly_modulus_degree(4)
        parms.set_coeff_modulus([
            SmallModulus(41),
            SmallModulus(137),
            SmallModulus(193),
            SmallModulus(65537)
        ])
        context = SEALContext.Create(parms, True, sec_level_type.none)
        context_data = context.key_context_data()
        self.assertEqual(3, context_data.chain_index())
        self.assertEqual(71047416497, context_data.total_coeff_modulus())
        self.assertEqual(context_data.parms_id(), context.key_parms_id())
        prev_context_data = context_data

        context_data = context_data.next_context_data()
        self.assertEqual(2, context_data.chain_index())
        self.assertEqual(1084081, context_data.total_coeff_modulus())
        self.assertEqual(context_data.prev_context_data().parms_id(),
                         prev_context_data.parms_id())
        prev_context_data = context_data
        context_data = context_data.next_context_data()
        self.assertEqual(1, context_data.chain_index())
        self.assertEqual(5617, context_data.total_coeff_modulus())
        self.assertEqual(context_data.prev_context_data().parms_id(),
                         prev_context_data.parms_id())
        prev_context_data = context_data
        context_data = context_data.next_context_data()
        self.assertEqual(0, context_data.chain_index())
        self.assertEqual(41, context_data.total_coeff_modulus())
        self.assertEqual(context_data.prev_context_data().parms_id(),
                         prev_context_data.parms_id())
        self.assertEqual(context_data.parms_id(), context.last_parms_id())

        context = SEALContext.Create(parms, False, sec_level_type.none)
        self.assertEqual(1, context.key_context_data().chain_index())
        self.assertEqual(0, context.first_context_data().chain_index())
        self.assertEqual(71047416497,
                         context.key_context_data().total_coeff_modulus())
        self.assertEqual(1084081,
                         context.first_context_data().total_coeff_modulus())
