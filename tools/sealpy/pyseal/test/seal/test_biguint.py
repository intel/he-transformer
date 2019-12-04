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


class BugInsignedIntTest(unittest.TestCase):
    def test_empty_big_uint(self):
        uint = BigUInt()
        self.assertEqual(0, uint.bit_count())
        self.assertTrue(None == uint.data())
        self.assertEqual(0, uint.byte_count())
        self.assertEqual(0, uint.uint64_count())
        self.assertEqual(0, uint.significant_bit_count())
        self.assertTrue("0" == uint.to_string())
        self.assertTrue(uint.is_zero())
        self.assertFalse(uint.is_alias())
        uint.set_zero()

        uint2 = BigUInt()
        self.assertTrue(uint == uint2)
        self.assertFalse(uint != uint2)

        uint.resize(1)
        self.assertEqual(1, uint.bit_count())
        self.assertTrue(None != uint.data())
        self.assertFalse(uint.is_alias())

        uint.resize(0)
        self.assertEqual(0, uint.bit_count())
        self.assertTrue(None == uint.data())
        self.assertFalse(uint.is_alias())

    def test_big_uint_64bits(self):
        uint = BigUInt(64)
        self.assertEqual(64, uint.bit_count())
        self.assertTrue(None != uint.data())
        self.assertEqual(8, uint.byte_count())
        self.assertEqual(1, uint.uint64_count())
        self.assertEqual(0, uint.significant_bit_count())
        self.assertTrue("0" == uint.to_string())
        self.assertTrue(uint.is_zero())
        self.assertEqual((0), uint.data())
        self.assertTrue(0 == uint[0])
        self.assertTrue(0 == uint[1])
        self.assertTrue(0 == uint[2])
        self.assertTrue(0 == uint[3])
        self.assertTrue(0 == uint[4])
        self.assertTrue(0 == uint[5])
        self.assertTrue(0 == uint[6])
        self.assertTrue(0 == uint[7])

        uint.assign("1")
        self.assertEqual(1, uint.significant_bit_count())
        self.assertTrue("1" == uint.to_string())
        self.assertFalse(uint.is_zero())
        self.assertEqual(1, uint.data())
        self.assertTrue(1 == uint[0])
        self.assertTrue(0 == uint[1])
        self.assertTrue(0 == uint[2])
        self.assertTrue(0 == uint[3])
        self.assertTrue(0 == uint[4])
        self.assertTrue(0 == uint[5])
        self.assertTrue(0 == uint[6])
        self.assertTrue(0 == uint[7])
        uint.set_zero()
        self.assertTrue(uint.is_zero())
        self.assertEqual((0), uint.data())

        uint.assign("7FFFFFFFFFFFFFFF")
        self.assertEqual(63, uint.significant_bit_count())
        self.assertTrue("7FFFFFFFFFFFFFFF" == uint.to_string())
        self.assertEqual(0x7FFFFFFFFFFFFFFF, uint.data())
        self.assertTrue(0xFF == uint[0])
        self.assertTrue(0xFF == uint[1])
        self.assertTrue(0xFF == uint[2])
        self.assertTrue(0xFF == uint[3])
        self.assertTrue(0xFF == uint[4])
        self.assertTrue(0xFF == uint[5])
        self.assertTrue(0xFF == uint[6])
        self.assertTrue(0x7F == uint[7])
        self.assertFalse(uint.is_zero())

        uint.assign("FFFFFFFFFFFFFFFF")
        self.assertEqual(64, uint.significant_bit_count())
        self.assertTrue("FFFFFFFFFFFFFFFF" == uint.to_string())
        self.assertEqual(0xFFFFFFFFFFFFFFFF, uint.data())
        self.assertTrue(0xFF == uint[0])
        self.assertTrue(0xFF == uint[1])
        self.assertTrue(0xFF == uint[2])
        self.assertTrue(0xFF == uint[3])
        self.assertTrue(0xFF == uint[4])
        self.assertTrue(0xFF == uint[5])
        self.assertTrue(0xFF == uint[6])
        self.assertTrue(0xFF == uint[7])
        self.assertFalse(uint.is_zero())

        uint.assign(0x8001)
        self.assertEqual(16, uint.significant_bit_count())
        self.assertTrue("8001" == uint.to_string())
        self.assertEqual(0x8001, uint.data())
        self.assertTrue(0x01 == uint[0])
        self.assertTrue(0x80 == uint[1])
        self.assertTrue(0x00 == uint[2])
        self.assertTrue(0x00 == uint[3])
        self.assertTrue(0x00 == uint[4])
        self.assertTrue(0x00 == uint[5])
        self.assertTrue(0x00 == uint[6])
        self.assertTrue(0x00 == uint[7])

    def test_big_uint_99bits(self):
        uint = BigUInt(99)
        self.assertEqual(99, uint.bit_count())
        self.assertTrue(None != uint.data())
        self.assertEqual(13, uint.byte_count())
        self.assertEqual(2, uint.uint64_count())
        self.assertEqual(0, uint.significant_bit_count())
        self.assertTrue("0" == uint.to_string())
        self.assertTrue(uint.is_zero())
        self.assertEqual(0, uint.data(0))
        self.assertEqual(0, uint.data(1))
        self.assertTrue(0 == uint[0])
        self.assertTrue(0 == uint[1])
        self.assertTrue(0 == uint[2])
        self.assertTrue(0 == uint[3])
        self.assertTrue(0 == uint[4])
        self.assertTrue(0 == uint[5])
        self.assertTrue(0 == uint[6])
        self.assertTrue(0 == uint[7])
        self.assertTrue(0 == uint[8])
        self.assertTrue(0 == uint[9])
        self.assertTrue(0 == uint[10])
        self.assertTrue(0 == uint[11])
        self.assertTrue(0 == uint[12])

        uint.assign("1")
        self.assertEqual(1, uint.significant_bit_count())
        self.assertTrue("1" == uint.to_string())
        self.assertFalse(uint.is_zero())
        self.assertEqual(1, uint.data(0))
        self.assertEqual(0, uint.data(1))
        self.assertTrue(1 == uint[0])
        self.assertTrue(0 == uint[1])
        self.assertTrue(0 == uint[2])
        self.assertTrue(0 == uint[3])
        self.assertTrue(0 == uint[4])
        self.assertTrue(0 == uint[5])
        self.assertTrue(0 == uint[6])
        self.assertTrue(0 == uint[7])
        self.assertTrue(0 == uint[8])
        self.assertTrue(0 == uint[9])
        self.assertTrue(0 == uint[10])
        self.assertTrue(0 == uint[11])
        self.assertTrue(0 == uint[12])
        uint.set_zero()
        self.assertTrue(uint.is_zero())
        self.assertEqual(0, uint.data(0))
        self.assertEqual(0, uint.data(1))

        uint.assign("7FFFFFFFFFFFFFFFFFFFFFFFF")
        self.assertEqual(99, uint.significant_bit_count())
        self.assertTrue("7FFFFFFFFFFFFFFFFFFFFFFFF" == uint.to_string())
        self.assertEqual((0xFFFFFFFFFFFFFFFF), uint.data(0))
        self.assertEqual((0x7FFFFFFFF), uint.data(1))
        self.assertTrue(0xFF == uint[0])
        self.assertTrue(0xFF == uint[1])
        self.assertTrue(0xFF == uint[2])
        self.assertTrue(0xFF == uint[3])
        self.assertTrue(0xFF == uint[4])
        self.assertTrue(0xFF == uint[5])
        self.assertTrue(0xFF == uint[6])
        self.assertTrue(0xFF == uint[7])
        self.assertTrue(0xFF == uint[8])
        self.assertTrue(0xFF == uint[9])
        self.assertTrue(0xFF == uint[10])
        self.assertTrue(0xFF == uint[11])
        self.assertTrue(0x07 == uint[12])
        self.assertFalse(uint.is_zero())
        uint.set_zero()
        self.assertTrue(uint.is_zero())
        self.assertEqual(0, uint.data(0))
        self.assertEqual(0, uint.data(1))

        uint.assign("4000000000000000000000000")
        self.assertEqual(99, uint.significant_bit_count())
        self.assertTrue("4000000000000000000000000" == uint.to_string())
        self.assertEqual((0x0000000000000000), uint.data(0))
        self.assertEqual((0x400000000), uint.data(1))
        self.assertTrue(0x00 == uint[0])
        self.assertTrue(0x00 == uint[1])
        self.assertTrue(0x00 == uint[2])
        self.assertTrue(0x00 == uint[3])
        self.assertTrue(0x00 == uint[4])
        self.assertTrue(0x00 == uint[5])
        self.assertTrue(0x00 == uint[6])
        self.assertTrue(0x00 == uint[7])
        self.assertTrue(0x00 == uint[8])
        self.assertTrue(0x00 == uint[9])
        self.assertTrue(0x00 == uint[10])
        self.assertTrue(0x00 == uint[11])
        self.assertTrue(0x04 == uint[12])
        self.assertFalse(uint.is_zero())

        uint.assign(0x8001)
        self.assertEqual(16, uint.significant_bit_count())
        self.assertTrue("8001" == uint.to_string())
        self.assertEqual((0x8001), uint.data(0))
        self.assertEqual(0, uint.data(1))
        self.assertTrue(0x01 == uint[0])
        self.assertTrue(0x80 == uint[1])
        self.assertTrue(0x00 == uint[2])
        self.assertTrue(0x00 == uint[3])
        self.assertTrue(0x00 == uint[4])
        self.assertTrue(0x00 == uint[5])
        self.assertTrue(0x00 == uint[6])
        self.assertTrue(0x00 == uint[7])
        self.assertTrue(0x00 == uint[8])
        self.assertTrue(0x00 == uint[9])
        self.assertTrue(0x00 == uint[10])
        self.assertTrue(0x00 == uint[11])
        self.assertTrue(0x00 == uint[12])

        uint2 = BigUInt("123")
        self.assertFalse(uint == uint2)
        self.assertFalse(uint2 == uint)
        self.assertTrue(uint != uint2)
        self.assertTrue(uint2 != uint)

        uint.assign(uint2)
        self.assertTrue(uint == uint2)
        self.assertFalse(uint != uint2)
        self.assertEqual(9, uint.significant_bit_count())
        self.assertTrue("123" == uint.to_string())
        self.assertEqual(0x123, uint.data(0))
        self.assertEqual(0, uint.data(1))
        self.assertTrue(0x23 == uint[0])
        self.assertTrue(0x01 == uint[1])
        self.assertTrue(0x00 == uint[2])
        self.assertTrue(0x00 == uint[3])
        self.assertTrue(0x00 == uint[4])
        self.assertTrue(0x00 == uint[5])
        self.assertTrue(0x00 == uint[6])
        self.assertTrue(0x00 == uint[7])
        self.assertTrue(0x00 == uint[8])
        self.assertTrue(0x00 == uint[9])
        self.assertTrue(0x00 == uint[10])
        self.assertTrue(0x00 == uint[11])
        self.assertTrue(0x00 == uint[12])

        uint.resize(8)
        self.assertEqual(8, uint.bit_count())
        self.assertEqual(1, uint.uint64_count())
        self.assertTrue("23" == uint.to_string())

        uint.resize(100)
        self.assertEqual(100, uint.bit_count())
        self.assertEqual(2, uint.uint64_count())
        self.assertTrue("23" == uint.to_string())

        uint.resize(0)
        self.assertEqual(0, uint.bit_count())
        self.assertEqual(0, uint.uint64_count())
        self.assertTrue(None == uint.data())

    def test_save_load_uint(self):
        value = BigUInt()
        value2 = BigUInt("100")
        stream = value.save()
        value2.load(stream)
        self.assertTrue(value == value2)

        value = BigUInt("123")
        stream = value.save()
        value2.load(stream)
        self.assertTrue(value == value2)

        value = BigUInt("FFFFFFFFFFFFFFFFFFFFFFFFFF")
        stream = value.save()
        value2.load(stream)
        self.assertTrue(value == value2)

        value = BigUInt("0")
        stream = value.save()
        value2.load(stream)
        self.assertTrue(value == value2)

    def test_duplicate_to(self):
        original = BigUInt(123)
        original.assign(56789)

        target = BigUInt()

        original.duplicate_to(target)
        self.assertEqual(target.bit_count(), original.bit_count())
        self.assertTrue(target == original)

    def test_duplicate_from(self):
        original = BigUInt(123)
        original.assign(56789)

        target = BigUInt()

        target.duplicate_from(original)
        self.assertEqual(target.bit_count(), original.bit_count())
        self.assertTrue(target == original)

    def test_big_uint_copy_move_assign(self):
        p1 = BigUInt("123")
        p2 = BigUInt("456")
        p3 = BigUInt()

        p1.assign(p2)
        p3.assign(p1)
        self.assertTrue(p1 == p2)
        self.assertTrue(p3 == p1)
        """
        p1 = BigUInt("123")
        p2 = BigUInt("456")
        p3 = BigUInt()
        p4 = BigUInt(p2)

        p1.operator = (move(p2))
        p3.operator = (move(p1))
        self.assertTrue(p3 == p4)
        self.assertTrue(p1 == p2)
        self.assertTrue(p3 == p1)

        p1_anchor = 123
        p2_anchor = 456
        p1 = BigUInt(64, p1_anchor)
        p2 = BigUInt(64, p2_anchor)
        p3 = BigUInt()

        p1.operator = (p2)
        p3.operator = (p1)
        self.assertTrue(p1 == p2)
        self.assertTrue(p3 == p1)

        p1_anchor = 123
        p2_anchor = 456
        p1 = BigUInt(64, p1_anchor)
        p2 = BigUInt(64, p2_anchor)
        p3 = BigUInt()
        p4 = BigUInt(p2)

        p1.operator = (move(p2))
        p3.operator = (move(p1))
        self.assertTrue(p3 == p4)
        self.assertTrue(p2 == 456)
        self.assertTrue(p1 == 456)
        self.assertTrue(p3 == 456)
        """
