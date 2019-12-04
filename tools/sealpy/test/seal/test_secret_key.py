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


class SecretKeyTest(unittest.TestCase):
    def test_save_load_secret_key(self):
        parms = EncryptionParameters(scheme_type.BFV)
        parms.set_poly_modulus_degree(64)
        parms.set_plain_modulus(1 << 6)
        parms.set_coeff_modulus(CoeffModulus.Create(64, [60]))

        context = SEALContext.Create(parms, False, sec_level_type.none)
        keygen = KeyGenerator(context)

        sk = keygen.secret_key()
        self.assertTrue(sk.parms_id() == context.key_parms_id())
        stream = sk.save()

        sk2 = SecretKey()
        sk2.load(context, stream)

        self.assertTrue(sk.data() == sk2.data())
        self.assertTrue(sk.parms_id() == sk2.parms_id())

        parms = EncryptionParameters(scheme_type.BFV)
        parms.set_poly_modulus_degree(256)
        parms.set_plain_modulus(1 << 20)
        parms.set_coeff_modulus(CoeffModulus.Create(256, [30, 40]))

        context = SEALContext.Create(parms, False, sec_level_type.none)
        keygen = KeyGenerator(context)

        sk = keygen.secret_key()
        self.assertTrue(sk.parms_id() == context.key_parms_id())
        stream = sk.save()

        sk2 = SecretKey()
        sk2.load(context, stream)

        self.assertTrue(sk.data() == sk2.data())
        self.assertTrue(sk.parms_id() == sk2.parms_id())
