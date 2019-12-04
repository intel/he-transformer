# ******************************************************************************
# Copyright 2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License")
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http:#www.apache.org/licenses/LICENSE-2.0
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
    BatchEncoder, \
    BigUInt, \
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
    IntVec, \
    IntegerEncoder, \
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


class IntegerEncoderTest(unittest.TestCase):
    def test_encode_decode_big_uint(self):
        modulus = SmallModulus(0xFFFFFFFFFFFFFFF)
        parms = EncryptionParameters(scheme_type.BFV)
        parms.set_plain_modulus(modulus)
        context = SEALContext.Create(parms)
        encoder = IntegerEncoder(context)

        value = BigUInt(64)
        value.assign("0")
        poly = encoder.encode(value)
        self.assertEqual(0, poly.significant_coeff_count())
        self.assertTrue(poly.is_zero())
        self.assertTrue(value == encoder.decode_biguint(poly))

        value.assign("1")
        poly1 = encoder.encode(value)
        self.assertEqual(1, poly1.coeff_count())
        self.assertTrue("1" == poly1.to_string())
        self.assertTrue(value == encoder.decode_biguint(poly1))

        value.assign("2")
        poly2 = encoder.encode(value)
        self.assertEqual(2, poly2.coeff_count())
        self.assertTrue("1x^1" == poly2.to_string())
        self.assertTrue(value == encoder.decode_biguint(poly2))

        value.assign("3")
        poly3 = encoder.encode(value)
        self.assertEqual(2, poly3.coeff_count())
        self.assertTrue("1x^1 + 1" == poly3.to_string())
        self.assertTrue(value == encoder.decode_biguint(poly3))

        value.assign("FFFFFFFFFFFFFFFF")
        poly4 = encoder.encode(value)
        self.assertEqual(64, poly4.coeff_count())
        for i in range(64):
            self.assertTrue(poly4[i] == 1)
        self.assertTrue(value == encoder.decode_biguint(poly4))

        value.assign("80F02")
        poly5 = encoder.encode(value)
        self.assertEqual(20, poly5.coeff_count())
        for i in range(20):
            if i == 19 or (i >= 8 and i <= 11) or i == 1:
                self.assertTrue(poly5[i] == 1)
            else:
                self.assertTrue(poly5[i] == 0)
        self.assertTrue(value == encoder.decode_biguint(poly5))

        poly6 = Plaintext(3)
        poly6[0] = 1
        poly6[1] = 500
        poly6[2] = 1023
        value.assign(1 + 500 * 2 + 1023 * 4)
        self.assertTrue(value == encoder.decode_biguint(poly6))

        modulus = 1024
        parms.set_plain_modulus(modulus)
        context2 = SEALContext.Create(parms)
        encoder2 = IntegerEncoder(context2)
        poly7 = Plaintext(4)
        poly7[0] = 1023  # -1   (*1)
        poly7[1] = 512  # -512 (*2)
        poly7[2] = 511  # 511  (*4)
        poly7[3] = 1  # 1    (*8)
        value.assign(-1 + -512 * 2 + 511 * 4 + 1 * 8)
        self.assertTrue(value == encoder2.decode_biguint(poly7))

    def test_encode_decode_uint64(self):
        modulus = SmallModulus(0xFFFFFFFFFFFFFFF)
        parms = EncryptionParameters(scheme_type.BFV)
        parms.set_plain_modulus(modulus)
        context = SEALContext.Create(parms)
        encoder = IntegerEncoder(context)

        poly = encoder.encode(np.uint64(0))
        self.assertEqual(0, poly.significant_coeff_count())
        self.assertTrue(poly.is_zero())
        self.assertEqual(0, encoder.decode_uint64(poly))

        poly1 = encoder.encode(np.uint64(1))
        self.assertEqual(1, poly1.coeff_count())
        self.assertTrue("1" == poly1.to_string())
        self.assertEqual(1, encoder.decode_uint64(poly1))

        poly2 = encoder.encode(np.uint64(2))
        self.assertEqual(2, poly2.coeff_count())
        self.assertTrue("1x^1" == poly2.to_string())
        self.assertEqual(2, encoder.decode_uint64(poly2))

        poly3 = encoder.encode(np.uint64(3))
        self.assertEqual(2, poly3.coeff_count())
        self.assertTrue("1x^1 + 1" == poly3.to_string())
        self.assertEqual(3, encoder.decode_uint64(poly3))

        poly4 = encoder.encode(np.uint64(0xFFFFFFFFFFFFFFFF))
        self.assertEqual(64, poly4.coeff_count())
        for i in range(64):
            self.assertTrue(poly4[i] == 1)
        self.assertEqual(0xFFFFFFFFFFFFFFFF, encoder.decode_uint64(poly4))

        poly5 = encoder.encode(np.uint64(0x80F02))
        self.assertEqual(20, poly5.coeff_count())
        for i in range(20):
            if i == 19 or (i >= 8 and i <= 11) or i == 1:
                self.assertTrue(poly5[i] == 1)
            else:
                self.assertTrue(poly5[i] == 0)
        self.assertEqual(0x80F02, encoder.decode_uint64(poly5))

        poly6 = Plaintext(3)
        poly6[0] = 1
        poly6[1] = 500
        poly6[2] = 1023
        self.assertEqual((1 + 500 * 2 + 1023 * 4),
                         encoder.decode_uint64(poly6))

        modulus = 1024
        parms.set_plain_modulus(modulus)
        context2 = SEALContext.Create(parms)
        encoder2 = IntegerEncoder(context2)
        poly7 = Plaintext(4)
        poly7[0] = 1023  # -1   (*1)
        poly7[1] = 512  # -512 (*2)
        poly7[2] = 511  # 511  (*4)
        poly7[3] = 1  # 1    (*8)
        self.assertEqual((-1 + -512 * 2 + 511 * 4 + 1 * 8),
                         encoder2.decode_uint64(poly7))

    def test_encode_decode_uint32(self):
        modulus = SmallModulus(0xFFFFFFFFFFFFFFF)
        parms = EncryptionParameters(scheme_type.BFV)
        parms.set_plain_modulus(modulus)
        context = SEALContext.Create(parms)
        encoder = IntegerEncoder(context)

        poly = encoder.encode(np.uint32(0))
        self.assertEqual(0, poly.significant_coeff_count())
        self.assertTrue(poly.is_zero())
        self.assertEqual(np.uint32(0), encoder.decode_uint32(poly))

        poly1 = encoder.encode(np.uint32(1))
        self.assertEqual(1, poly1.significant_coeff_count())
        self.assertTrue("1" == poly1.to_string())
        self.assertEqual(np.uint32(1), encoder.decode_uint32(poly1))

        poly2 = encoder.encode(np.uint32(2))
        self.assertEqual(2, poly2.significant_coeff_count())
        self.assertTrue("1x^1" == poly2.to_string())
        self.assertEqual(np.uint32(2), encoder.decode_uint32(poly2))

        poly3 = encoder.encode(np.uint32(3))
        self.assertEqual(2, poly3.significant_coeff_count())
        self.assertTrue("1x^1 + 1" == poly3.to_string())
        self.assertEqual(np.uint32(3), encoder.decode_uint32(poly3))

        poly4 = encoder.encode(np.uint32(0xFFFFFFFF))
        self.assertEqual(32, poly4.significant_coeff_count())
        for i in range(32):
            self.assertTrue(1 == poly4[i])
        self.assertEqual(np.uint32(0xFFFFFFFF), encoder.decode_uint32(poly4))

        poly5 = encoder.encode(np.uint32(0x80F02))
        self.assertEqual(20, poly5.significant_coeff_count())
        for i in range(20):
            if i == 19 or (i >= 8 and i <= 11) or i == 1:
                self.assertTrue(1 == poly5[i])
            else:
                self.assertTrue(poly5[i] == 0)
        self.assertEqual(np.uint32(0x80F02), encoder.decode_uint32(poly5))

        poly6 = Plaintext(3)
        poly6[0] = 1
        poly6[1] = 500
        poly6[2] = 1023
        self.assertEqual(
            np.uint32(1 + 500 * 2 + 1023 * 4), encoder.decode_uint32(poly6))

        modulus = 1024
        parms.set_plain_modulus(modulus)
        context2 = SEALContext.Create(parms)
        encoder2 = IntegerEncoder(context2)
        poly7 = Plaintext(4)
        poly7[0] = 1023  # -1   (*1)
        poly7[1] = 512  # -512 (*2)
        poly7[2] = 511  # 511  (*4)
        poly7[3] = 1  # 1    (*8)
        self.assertEqual(
            np.uint32(-1 + -512 * 2 + 511 * 4 + 1 * 8),
            encoder2.decode_uint32(poly7))

    def test_encode_decode_int64(self):
        modulus = SmallModulus(0x7FFFFFFFFFFFF)
        parms = EncryptionParameters(scheme_type.BFV)
        parms.set_plain_modulus(modulus)
        context = SEALContext.Create(parms)
        encoder = IntegerEncoder(context)

        poly = encoder.encode(np.int64(0))
        self.assertEqual(0, poly.significant_coeff_count())
        self.assertTrue(poly.is_zero())
        self.assertEqual((0), (encoder.decode_int64(poly)))

        poly1 = encoder.encode(np.int64(1))
        self.assertEqual(1, poly1.significant_coeff_count())
        self.assertTrue("1" == poly1.to_string())
        self.assertEqual(1, (encoder.decode_int64(poly1)))

        poly2 = encoder.encode(np.int64(2))
        self.assertEqual(2, poly2.significant_coeff_count())
        self.assertTrue("1x^1" == poly2.to_string())
        self.assertEqual((2), (encoder.decode_int64(poly2)))

        poly3 = encoder.encode(np.int64(3))
        self.assertEqual(2, poly3.significant_coeff_count())
        self.assertTrue("1x^1 + 1" == poly3.to_string())
        self.assertEqual((3), (encoder.decode_int64(poly3)))

        poly4 = encoder.encode(np.int64(-1))
        self.assertEqual(1, poly4.significant_coeff_count())
        self.assertTrue("7FFFFFFFFFFFE" == poly4.to_string())
        self.assertEqual((-1), (encoder.decode_int64(poly4)))

        poly5 = encoder.encode(np.int64(-2))
        self.assertEqual(2, poly5.significant_coeff_count())
        self.assertTrue("7FFFFFFFFFFFEx^1" == poly5.to_string())
        self.assertEqual((-2), (encoder.decode_int64(poly5)))

        poly6 = encoder.encode(np.int64(-3))
        self.assertEqual(2, poly6.significant_coeff_count())
        self.assertTrue(
            "7FFFFFFFFFFFEx^1 + 7FFFFFFFFFFFE" == poly6.to_string())
        self.assertEqual((-3), (encoder.decode_int64(poly6)))

        poly7 = encoder.encode(np.int64(0x7FFFFFFFFFFFF))
        self.assertEqual(51, poly7.significant_coeff_count())
        for i in range(51):
            self.assertTrue(1 == poly7[i])
        self.assertEqual((0x7FFFFFFFFFFFF), (encoder.decode_int64(poly7)))

        poly8 = encoder.encode(np.int64(0x8000000000000))
        self.assertEqual(52, poly8.significant_coeff_count())
        self.assertTrue(poly8[51] == 1)
        for i in range(51):
            self.assertTrue(poly8[i] == 0)
        self.assertEqual((0x8000000000000), (encoder.decode_int64(poly8)))

        poly9 = encoder.encode(np.int64(0x80F02))
        self.assertEqual(20, poly9.significant_coeff_count())
        for i in range(20):
            if i == 19 or (i >= 8 and i <= 11) or i == 1:
                self.assertTrue(1 == poly9[i])
            else:
                self.assertTrue(poly9[i] == 0)
        self.assertEqual((0x80F02), (encoder.decode_int64(poly9)))

        poly10 = encoder.encode(np.int64(-1073))
        self.assertEqual(11, poly10.significant_coeff_count())
        self.assertTrue(0x7FFFFFFFFFFFE == poly10[10])
        self.assertTrue(poly10[9] == 0)
        self.assertTrue(poly10[8] == 0)
        self.assertTrue(poly10[7] == 0)
        self.assertTrue(poly10[6] == 0)
        self.assertTrue(0x7FFFFFFFFFFFE == poly10[5])
        self.assertTrue(0x7FFFFFFFFFFFE == poly10[4])
        self.assertTrue(poly10[3] == 0)
        self.assertTrue(poly10[2] == 0)
        self.assertTrue(poly10[1] == 0)
        self.assertTrue(0x7FFFFFFFFFFFE == poly10[0])
        self.assertEqual((-1073), (encoder.decode_int64(poly10)))

        modulus = SmallModulus(0xFFFF)
        parms.set_plain_modulus(modulus)
        context2 = SEALContext.Create(parms)
        encoder2 = IntegerEncoder(context2)
        poly11 = Plaintext(6)
        poly11[0] = 1
        poly11[1] = 0xFFFE  # -1
        poly11[2] = 0xFFFD  # -2
        poly11[3] = 0x8000  # -32767
        poly11[4] = 0x7FFF  # 32767
        poly11[5] = 0x7FFE  # 32766
        self.assertEqual(
            (1 + -1 * 2 + -2 * 4 + -32767 * 8 + 32767 * 16 + 32766 * 32),
            (encoder2.decode_int64(poly11)))

    def test_encode_decode_int32(self):
        modulus = SmallModulus(0x7FFFFFFFFFFFFF)
        parms = EncryptionParameters(scheme_type.BFV)
        parms.set_plain_modulus(modulus)
        context = SEALContext.Create(parms)
        encoder = IntegerEncoder(context)

        poly = encoder.encode(np.int32(0))
        self.assertEqual(0, poly.significant_coeff_count())
        self.assertTrue(poly.is_zero())
        self.assertEqual(np.int32(0), encoder.decode_int32(poly))

        poly1 = encoder.encode(np.int32(1))
        self.assertEqual(1, poly1.significant_coeff_count())
        self.assertTrue("1" == poly1.to_string())
        self.assertEqual(np.int32(1), encoder.decode_int32(poly1))

        poly2 = encoder.encode(np.int32(2))
        self.assertEqual(2, poly2.significant_coeff_count())
        self.assertTrue("1x^1" == poly2.to_string())
        self.assertEqual(np.int32(2), encoder.decode_int32(poly2))

        poly3 = encoder.encode(np.int32(3))
        self.assertEqual(2, poly3.significant_coeff_count())
        self.assertTrue("1x^1 + 1" == poly3.to_string())
        self.assertEqual(np.int32(3), encoder.decode_int32(poly3))

        poly4 = encoder.encode(np.int32(-1))
        self.assertEqual(1, poly4.significant_coeff_count())
        self.assertTrue("7FFFFFFFFFFFFE" == poly4.to_string())
        self.assertEqual(np.int32(-1), encoder.decode_int32(poly4))

        poly5 = encoder.encode(np.int32(-2))
        self.assertEqual(2, poly5.significant_coeff_count())
        self.assertTrue("7FFFFFFFFFFFFEx^1" == poly5.to_string())
        self.assertEqual(np.int32(-2), encoder.decode_int32(poly5))

        poly6 = encoder.encode(np.int32(-3))
        self.assertEqual(2, poly6.significant_coeff_count())
        self.assertTrue(
            "7FFFFFFFFFFFFEx^1 + 7FFFFFFFFFFFFE" == poly6.to_string())
        self.assertEqual(np.int32(-3), encoder.decode_int32(poly6))

        poly7 = encoder.encode(np.int32(0x7FFFFFFF))
        self.assertEqual(31, poly7.significant_coeff_count())
        for i in range(31):
            self.assertTrue(1 == poly7[i])
        self.assertEqual(np.int32(0x7FFFFFFF), encoder.decode_int32(poly7))

        poly8 = encoder.encode(np.int32(0x80000000))
        self.assertEqual(32, poly8.significant_coeff_count())
        self.assertTrue(0x7FFFFFFFFFFFFE == poly8[31])
        for i in range(31):
            self.assertTrue(poly8[i] == 0)
        self.assertEqual(np.int32(0x80000000), encoder.decode_int32(poly8))

        poly9 = encoder.encode(np.int32(0x80F02))
        self.assertEqual(20, poly9.significant_coeff_count())
        for i in range(20):
            if i == 19 or (i >= 8 and i <= 11) or i == 1:
                self.assertTrue(1 == poly9[i])
            else:
                self.assertTrue(poly9[i] == 0)
        self.assertEqual(np.int32(0x80F02), encoder.decode_int32(poly9))

        poly10 = encoder.encode(np.int32(-1073))
        self.assertEqual(11, poly10.significant_coeff_count())
        self.assertTrue(0x7FFFFFFFFFFFFE == poly10[10])
        self.assertTrue(poly10[9] == 0)
        self.assertTrue(poly10[8] == 0)
        self.assertTrue(poly10[7] == 0)
        self.assertTrue(poly10[6] == 0)
        self.assertTrue(0x7FFFFFFFFFFFFE == poly10[5])
        self.assertTrue(0x7FFFFFFFFFFFFE == poly10[4])
        self.assertTrue(poly10[3] == 0)
        self.assertTrue(poly10[2] == 0)
        self.assertTrue(poly10[1] == 0)
        self.assertTrue(0x7FFFFFFFFFFFFE == poly10[0])
        self.assertEqual(np.int32(-1073), encoder.decode_int32(poly10))

        modulus = SmallModulus(0xFFFF)
        parms.set_plain_modulus(modulus)
        context2 = SEALContext.Create(parms)
        encoder2 = IntegerEncoder(context2)
        poly11 = Plaintext(6)
        poly11[0] = 1
        poly11[1] = 0xFFFE  # -1
        poly11[2] = 0xFFFD  # -2
        poly11[3] = 0x8000  # -32767
        poly11[4] = 0x7FFF  # 32767
        poly11[5] = 0x7FFE  # 32766
        self.assertEqual(
            np.int32(1 + -1 * 2 + -2 * 4 + -32767 * 8 + 32767 * 16 +
                     32766 * 32), encoder2.decode_int32(poly11))
