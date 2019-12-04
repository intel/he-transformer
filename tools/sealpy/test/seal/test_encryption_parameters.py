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
    UniformRandomGeneratorFactory


class EncryptionParametersTest(unittest.TestCase):
    def test_encryption_parameters_set(self):
        def encryption_parameters_test(scheme):
            parms = EncryptionParameters(scheme)
            parms.set_coeff_modulus([SmallModulus(2), SmallModulus(3)])
            if scheme == scheme_type.BFV:
                parms.set_plain_modulus(2)
            parms.set_poly_modulus_degree(2)
            parms.set_random_generator(
                UniformRandomGeneratorFactory.default_factory())

            self.assertTrue(parms.coeff_modulus()[0] == 2)
            self.assertTrue(parms.coeff_modulus()[1] == 3)
            if scheme == scheme_type.BFV:
                self.assertTrue(parms.plain_modulus().value() == 2)
            elif scheme == scheme_type.CKKS:
                self.assertTrue(parms.plain_modulus().value() == 0)
            self.assertTrue(parms.poly_modulus_degree() == 2)
            self.assertTrue(parms.random_generator() ==
                            UniformRandomGeneratorFactory.default_factory())

            parms.set_coeff_modulus(CoeffModulus.Create(2, [30, 40, 50]))
            if scheme == scheme_type.BFV:
                parms.set_plain_modulus(2)
            parms.set_poly_modulus_degree(128)

            #self.assertTrue(util.is_prime(parms.coeff_modulus()[0]))
            #self.assertTrue(util.is_prime(parms.coeff_modulus()[1]))
            #self.assertTrue(util.is_prime(parms.coeff_modulus()[2]))
            if scheme == scheme_type.BFV:
                self.assertTrue(parms.plain_modulus().value() == 2)
            elif scheme == scheme_type.CKKS:
                self.assertTrue(parms.plain_modulus().value() == 0)
            self.assertTrue(parms.poly_modulus_degree() == 128)
            self.assertTrue(parms.random_generator() ==
                            UniformRandomGeneratorFactory.default_factory())

        encryption_parameters_test(scheme_type.BFV)
        encryption_parameters_test(scheme_type.CKKS)

    def test_encryption_parameters_compare(self):
        scheme = scheme_type.BFV
        parms1 = EncryptionParameters(scheme)
        parms1.set_coeff_modulus(CoeffModulus.Create(64, [30]))
        if scheme == scheme_type.BFV:
            parms1.set_plain_modulus(1 << 6)
        parms1.set_poly_modulus_degree(64)

        parms2 = EncryptionParameters(parms1)
        self.assertTrue(parms1 == parms2)

        parms3 = EncryptionParameters(scheme)
        parms3 = EncryptionParameters(parms2)
        self.assertTrue(parms3 == parms2)
        parms3.set_coeff_modulus(CoeffModulus.Create(64, [32]))
        self.assertFalse(parms3 == parms2)

        parms3 = EncryptionParameters(parms2)
        self.assertTrue(parms3 == parms2)
        parms3.set_coeff_modulus(CoeffModulus.Create(64, [30, 30]))
        self.assertFalse(parms3 == parms2)

        parms3 = EncryptionParameters(parms2)
        parms3.set_poly_modulus_degree(128)
        self.assertFalse(parms3 == parms1)

        parms3 = EncryptionParameters(parms2)
        if scheme == scheme_type.BFV:
            parms3.set_plain_modulus((1 << 6) + 1)
        self.assertFalse(parms3 == parms2)

        parms3 = EncryptionParameters(parms2)
        self.assertTrue(parms3 == parms2)

        parms3 = EncryptionParameters(parms2)
        self.assertTrue(parms3 == parms2)

        parms3 = EncryptionParameters(parms2)
        parms3.set_poly_modulus_degree(128)
        parms3.set_poly_modulus_degree(64)
        self.assertTrue(parms3 == parms1)

        parms3 = EncryptionParameters(parms2)
        parms3.set_coeff_modulus([SmallModulus(2)])
        parms3.set_coeff_modulus(CoeffModulus.Create(64, [50]))
        parms3.set_coeff_modulus(parms2.coeff_modulus())
        self.assertTrue(parms3 == parms2)

    def test_encryption_parameters_save_load(self):
        scheme = scheme_type.BFV
        parms = EncryptionParameters(scheme)
        parms2 = EncryptionParameters(scheme)
        parms.set_coeff_modulus(CoeffModulus.Create(64, [30]))
        if scheme == scheme_type.BFV:
            parms.set_plain_modulus(1 << 6)
        parms.set_poly_modulus_degree(64)
        stream = EncryptionParameters.Save(parms)
        parms2 = EncryptionParameters.Load(stream)
        self.assertTrue(parms.scheme() == parms2.scheme())
        self.assertTrue(parms.coeff_modulus() == parms2.coeff_modulus())
        self.assertTrue(parms.plain_modulus() == parms2.plain_modulus())
        self.assertTrue(
            parms.poly_modulus_degree() == parms2.poly_modulus_degree())
        self.assertTrue(parms == parms2)

        parms.set_coeff_modulus(CoeffModulus.Create(64, [30, 60, 60]))

        if scheme == scheme_type.BFV:
            parms.set_plain_modulus(1 << 30)
        parms.set_poly_modulus_degree(256)

        stream = EncryptionParameters.Save(parms)
        parms2 = EncryptionParameters.Load(stream)
        self.assertTrue(parms.scheme() == parms2.scheme())
        self.assertTrue(parms.coeff_modulus() == parms2.coeff_modulus())
        self.assertTrue(parms.plain_modulus() == parms2.plain_modulus())
        self.assertTrue(
            parms.poly_modulus_degree() == parms2.poly_modulus_degree())
        self.assertTrue(parms == parms2)
