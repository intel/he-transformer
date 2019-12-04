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


class CKKSEvaluator(unittest.TestCase):
    scale = 2**16
    small_slot_size = 8
    small_data_bound = 2**8
    parms_small = EncryptionParameters(scheme_type.CKKS)
    parms_small.set_poly_modulus_degree(128)
    parms_small.set_coeff_modulus(CoeffModulus.Create(128, [ 30, 30, 30 ]))

    large_slot_size = 32
    large_data_bound = 2**30
    parms_large = EncryptionParameters(scheme_type.CKKS)
    parms_large.set_poly_modulus_degree(64)
    parms_large.set_coeff_modulus(CoeffModulus.Create(64, [ 60, 60, 60 ]))

    num_trials = 50

    def initializer(self, slot_size, data_bound, complex_arg=False):
        if complex_arg:
            input = ComplexVec([0.0] * slot_size)
            for i in range(slot_size):
                real_part = random.uniform(0, data_bound)
                imag_part = random.uniform(0, data_bound)
                input[i] = complex(real_part, imag_part)
        else:
            input = DoubleVec([0.0] * slot_size)
            for i in range(slot_size):
                input[i] = random.uniform(0, data_bound)
        return input

    def negater(self, input1, slot_size, complex_arg=False):
        if complex_arg:
            result = ComplexVec([0.0] * slot_size)
        else:
            result = DoubleVec([0.0] * slot_size)
        for i in range(slot_size):
            result[i] = -input1[i]
        return result

    def adder(self, input1, input2, slot_size, complex_arg=False):
        if complex_arg:
            result = ComplexVec([0.0] * slot_size)
        else:
            result = DoubleVec([0.0] * slot_size)
        for i in range(slot_size):
            result[i] = input1[i] + input2[i]
        return result

    def subtracter(self, input1, input2, slot_size, complex_arg=False):
        if complex_arg:
            result = ComplexVec([0.0] * slot_size)
        else:
            result = DoubleVec([0.0] * slot_size)
        for i in range(slot_size):
            result[i] = input1[i] - input2[i]
        return result

    def multiplier(self, input1, input2, slot_size, complex_arg=False):
        if complex_arg:
            result = ComplexVec([0.0] * slot_size)
        else:
            result = DoubleVec([0.0] * slot_size)
        for i in range(slot_size):
            result[i] = input1[i] * input2[i]
        return result

    def run_test_case(self, parms, num_trials, slot_size, data_bound, scale,
                      initializer, op, op_args_str, expected_op,
                      expected_op_args_str):
        context = SEALContext.Create(parms, True, sec_level_type.none)
        keygen = KeyGenerator(context)

        encoder = CKKSEncoder(context)
        encryptor = Encryptor(context, keygen.public_key())
        decryptor = Decryptor(context, keygen.secret_key())
        evaluator = Evaluator(context)

        encrypted1 = Ciphertext()
        encrypted2 = Ciphertext()
        plain1 = Plaintext()
        plain2 = Plaintext()
        plainRes = Plaintext()

        # Get op arguments
        op_args = []
        op_arg_str_map = {
            'E1': encrypted1,
            'E2': encrypted2,
            'P1': plain1,
            'P2': plain2
        }
        for op_arg_str in op_args_str:
            if op_arg_str in op_arg_str_map:
                op_args.append(op_arg_str_map[op_arg_str])
            else:
                raise Exception('Unknown op_arg_str', op_arg_str)

        for exp_count in range(num_trials):
            for complex_case in [True, False]:
                for i in range(slot_size):
                    input1 = initializer(slot_size, data_bound, complex_case)
                    input2 = initializer(slot_size, data_bound, complex_case)

                # Get expected op arguments
                expected_op_args = []
                expectd_op_args_str_map = {'I1': input1, 'I2': input2}
                for expected_op_arg_str in expected_op_args_str:
                    if expected_op_arg_str in expectd_op_args_str_map:
                        expected_op_args.append(
                            expectd_op_args_str_map[expected_op_arg_str])
                    else:
                        raise Exception('Unknown expected op arg_str',
                                        expected_op_arg_str)

                output = DoubleVec([0.0] * slot_size)
                if complex_case:
                    output = ComplexVec([0.0 + 0.0j] * slot_size)

                encoder.encode(input1, context.first_parms_id(), scale, plain1)
                encoder.encode(input2, context.first_parms_id(), scale, plain2)
                encryptor.encrypt(plain1, encrypted1)
                encryptor.encrypt(plain2, encrypted2)

                if len(op_args) == 1:
                    op(evaluator, op_args[0])
                elif len(op_args) == 2:
                    op(evaluator, op_args[0], op_args[1])
                elif len(op_args) == 3:
                    op(evaluator, op_args[0], op_args[1], op_args[2])
                else:
                    raise Exception("Too many op_args")

                if len(expected_op_args) == 1:
                    expected = expected_op(expected_op_args[0], slot_size,
                                           complex_case)
                elif len(expected_op_args) == 2:
                    expected = expected_op(expected_op_args[0],
                                           expected_op_args[1], slot_size,
                                           complex_case)
                else:
                    raise Exception("Too many expected_op_args")

            # check correctness of encryption
            self.assertEqual(encrypted1.parms_id, context.first_parms_id())

            decryptor.decrypt(encrypted1, plainRes)
            encoder.decode(plainRes, output)

            for i in range(slot_size):
                tmp = abs(expected[i] - output[i])
                off_scale = abs(tmp) / abs(expected[i])

                if (abs(tmp) > 0.5 and off_scale > 0.01):
                    print('Test failed on trial', exp_count, ' with op', op)
                    print('Diff', tmp)
                    print('exp', expected[i])
                    print('input1', input1[i])
                    print('input2', input2[i])
                    print('output', output[i])
                self.assertTrue(tmp < 0.5 or off_scale < 0.01)

    # Negtate tests
    def test_negate_small(self):
        self.run_test_case(
            self.parms_small,
            self.num_trials,
            slot_size=self.small_slot_size,
            data_bound=self.small_data_bound,
            scale=self.scale,
            initializer=self.initializer,
            op=(lambda evaluator, x, y: evaluator.negate(x, y)),
            op_args_str=('E1', 'E1'),
            expected_op=self.negater,
            expected_op_args_str=(['I1']))

    def test_negate_large(self):
        self.run_test_case(
            self.parms_large,
            self.num_trials,
            slot_size=self.large_slot_size,
            data_bound=self.large_data_bound,
            scale=self.scale,
            initializer=self.initializer,
            op=(lambda evaluator, x, y: evaluator.negate(x, y)),
            op_args_str=('E1', 'E1'),
            expected_op=self.negater,
            expected_op_args_str=(['I1']))

    def test_negate_small_inplace(self):
        self.run_test_case(
            self.parms_large,
            self.num_trials,
            slot_size=self.large_slot_size,
            data_bound=self.large_data_bound,
            scale=self.scale,
            initializer=self.initializer,
            op=(lambda evaluator, x: evaluator.negate_inplace(x)),
            op_args_str=(['E1']),
            expected_op=self.negater,
            expected_op_args_str=(['I1']))

    # Add tests
    def test_add_small(self):
        self.run_test_case(
            self.parms_small,
            self.num_trials,
            slot_size=self.small_slot_size,
            data_bound=self.small_data_bound,
            scale=self.scale,
            initializer=self.initializer,
            op=(lambda evaluator, x, y, z: evaluator.add(x, y, z)),
            op_args_str=('E1', 'E2', 'E1'),
            expected_op=self.adder,
            expected_op_args_str=('I1', 'I2'))

    def test_add_small_inplace(self):
        self.run_test_case(
            self.parms_small,
            self.num_trials,
            slot_size=self.small_slot_size,
            data_bound=self.small_data_bound,
            scale=self.scale,
            initializer=self.initializer,
            op=(lambda evaluator, x, y: evaluator.add_inplace(x, y)),
            op_args_str=('E1', 'E2'),
            expected_op=self.adder,
            expected_op_args_str=('I1', 'I2'))

    def test_add_large(self):
        self.run_test_case(
            self.parms_large,
            self.num_trials,
            slot_size=self.large_slot_size,
            data_bound=self.large_data_bound,
            scale=self.scale,
            initializer=self.initializer,
            op=(lambda evaluator, x, y, z: evaluator.add(x, y, z)),
            op_args_str=('E1', 'E2', 'E1'),
            expected_op=self.adder,
            expected_op_args_str=('I1', 'I2'))

    def test_add_large_inplace(self):
        self.run_test_case(
            self.parms_large,
            self.num_trials,
            slot_size=self.large_slot_size,
            data_bound=self.large_data_bound,
            scale=self.scale,
            initializer=self.initializer,
            op=(lambda evaluator, x, y: evaluator.add_inplace(x, y)),
            op_args_str=('E1', 'E2'),
            expected_op=self.adder,
            expected_op_args_str=('I1', 'I2'))

    def test_add_plain_small(self):
        self.run_test_case(
            self.parms_small,
            self.num_trials,
            slot_size=self.small_slot_size,
            data_bound=self.small_data_bound,
            scale=self.scale,
            initializer=self.initializer,
            op=(lambda evaluator, x, y, z: evaluator.add_plain(x, y, z)),
            op_args_str=('E1', 'P2', 'E1'),
            expected_op=self.adder,
            expected_op_args_str=('I1', 'I2'))

    def test_add_plain_small_inplace(self):
        self.run_test_case(
            self.parms_small,
            self.num_trials,
            slot_size=self.small_slot_size,
            data_bound=self.small_data_bound,
            scale=self.scale,
            initializer=self.initializer,
            op=(lambda evaluator, x, y: evaluator.add_plain_inplace(x, y)),
            op_args_str=('E1', 'P2'),
            expected_op=self.adder,
            expected_op_args_str=('I1', 'I2'))

    def test_add_plain_large(self):
        self.run_test_case(
            self.parms_large,
            self.num_trials,
            slot_size=self.large_slot_size,
            data_bound=self.large_data_bound,
            scale=self.scale,
            initializer=self.initializer,
            op=(lambda evaluator, x, y, z: evaluator.add_plain(x, y, z)),
            op_args_str=('E1', 'P2', 'E1'),
            expected_op=self.adder,
            expected_op_args_str=('I1', 'I2'))

    def test_add_plain_large_inplace(self):
        self.run_test_case(
            self.parms_large,
            self.num_trials,
            slot_size=self.large_slot_size,
            data_bound=self.large_data_bound,
            scale=self.scale,
            initializer=self.initializer,
            op=(lambda evaluator, x, y: evaluator.add_plain_inplace(x, y)),
            op_args_str=('E1', 'P2'),
            expected_op=self.adder,
            expected_op_args_str=('I1', 'I2'))

    # Sub tests
    def test_sub_small(self):
        self.run_test_case(
            self.parms_small,
            self.num_trials,
            slot_size=self.small_slot_size,
            data_bound=self.small_data_bound,
            scale=self.scale,
            initializer=self.initializer,
            op=(lambda evaluator, x, y, z: evaluator.sub(x, y, z)),
            op_args_str=('E1', 'E2', 'E1'),
            expected_op=self.subtracter,
            expected_op_args_str=('I1', 'I2'))

    def test_sub_small_inplace(self):
        self.run_test_case(
            self.parms_small,
            self.num_trials,
            slot_size=self.small_slot_size,
            data_bound=self.small_data_bound,
            scale=self.scale,
            initializer=self.initializer,
            op=(lambda evaluator, x, y: evaluator.sub_inplace(x, y)),
            op_args_str=( 'E1', 'E2' ),
            expected_op=self.subtracter,
            expected_op_args_str=('I1', 'I2'))

    def test_sub_large(self):
        self.run_test_case(
            self.parms_large,
            self.num_trials,
            slot_size=self.large_slot_size,
            data_bound=self.large_data_bound,
            scale=self.scale,
            initializer=self.initializer,
            op=(lambda evaluator, x, y, z: evaluator.sub(x, y, z)),
            op_args_str=('E1', 'E2', 'E1'),
            expected_op=self.subtracter,
            expected_op_args_str=('I1', 'I2'))

    def test_sub_large_inplace(self):
        self.run_test_case(
            self.parms_large,
            self.num_trials,
            slot_size=self.large_slot_size,
            data_bound=self.large_data_bound,
            scale=self.scale,
            initializer=self.initializer,
            op=(lambda evaluator, x, y: evaluator.sub_inplace(x, y)),
            op_args_str=(
                'E1',
                'E2',
            ),
            expected_op=self.subtracter,
            expected_op_args_str=('I1', 'I2'))

    def test_sub_plain_small(self):
        self.run_test_case(
            self.parms_small,
            self.num_trials,
            slot_size=self.small_slot_size,
            data_bound=self.small_data_bound,
            scale=self.scale,
            initializer=self.initializer,
            op=(lambda evaluator, x, y, z: evaluator.sub_plain(x, y, z)),
            op_args_str=('E1', 'P2', 'E1'),
            expected_op=self.subtracter,
            expected_op_args_str=('I1', 'I2'))

    def test_sub_plain_small_inplace(self):
        self.run_test_case(
            self.parms_small,
            self.num_trials,
            slot_size=self.small_slot_size,
            data_bound=self.small_data_bound,
            scale=self.scale,
            initializer=self.initializer,
            op=(lambda evaluator, x, y: evaluator.sub_plain_inplace(x, y)),
            op_args_str=('E1', 'P2'),
            expected_op=self.subtracter,
            expected_op_args_str=('I1', 'I2'))

    def test_sub_plain_large(self):
        self.run_test_case(
            self.parms_large,
            self.num_trials,
            slot_size=self.large_slot_size,
            data_bound=self.large_data_bound,
            scale=self.scale,
            initializer=self.initializer,
            op=(lambda evaluator, x, y, z: evaluator.sub_plain(x, y, z)),
            op_args_str=('E1', 'P2', 'E1'),
            expected_op=self.subtracter,
            expected_op_args_str=('I1', 'I2'))

    def test_sub_plain_large_inplace(self):
        self.run_test_case(
            self.parms_large,
            self.num_trials,
            slot_size=self.large_slot_size,
            data_bound=self.large_data_bound,
            scale=self.scale,
            initializer=self.initializer,
            op=(lambda evaluator, x, y: evaluator.sub_plain_inplace(x, y)),
            op_args_str=('E1', 'P2'),
            expected_op=self.subtracter,
            expected_op_args_str=('I1', 'I2'))

class CKKSEncoderTest(unittest.TestCase):
    def test_CKKSEncoder_encode_single_decode1(self):
        parms = EncryptionParameters(scheme_type.CKKS)
        slot_size = 16
        parms.set_poly_modulus_degree(64)
        parms.set_coeff_modulus(CoeffModulus.Create(64, [ 40, 40, 40, 40 ]))
        context = SEALContext.Create(parms, False, sec_level_type.none)
        encoder = CKKSEncoder(context)

        data_bound = (1 << 30)
        delta = 2**16
        plain = Plaintext()
        result = ComplexVec([])

        for iRun in range(50):
            value = random.randint(0, data_bound)
            encoder.encode(value, context.first_parms_id(), delta, plain)
            encoder.decode(plain, result)

            for i in range(slot_size):
                tmp = abs(value - result[i].real)
                self.assertTrue(tmp < 0.5)

    def test_CKKSEncoder_encode_single_decode2(self):
        slot_size = 32
        parms = EncryptionParameters(scheme_type.CKKS)
        parms.set_poly_modulus_degree(slot_size * 2)
        parms.set_coeff_modulus(CoeffModulus.Create(2 * slot_size, [ 40, 40, 40, 40 ]))
        context = SEALContext.Create(parms, False, sec_level_type.none)
        encoder = CKKSEncoder(context)

        data_bound = (1 << 30)
        plain = Plaintext()
        result = ComplexVec([])

        for iRun in range(50):
            value = random.randint(0, data_bound)
            encoder.encode(value, context.first_parms_id(), plain)
            encoder.decode(plain, result)

            for i in range(slot_size):
                tmp = abs(value - result[i].real)
                self.assertTrue(tmp < 0.5)

    def test_CKKSEncoder_encode_single_decode3(self):
        slot_size = 32
        parms = EncryptionParameters(scheme_type.CKKS)
        parms.set_poly_modulus_degree(slot_size * 2)
        parms.set_coeff_modulus(CoeffModulus.Create(2 * slot_size, [ 40, 40, 40, 40 ]))
        context = SEALContext.Create(parms, False, sec_level_type.none)
        encoder = CKKSEncoder(context)
        #  Use a very large scale
        data_bound = (1 << 20)
        plain = Plaintext()
        result = ComplexVec([])

        for iRun in range(50):
            value = random.randint(0, data_bound)
            encoder.encode(value, context.first_parms_id(), plain)
            encoder.decode(plain, result)

            for i in range(slot_size):
                tmp = abs(value - result[i].real)
                self.assertTrue(tmp < 0.5)

    def test_CKKSEncoder_encode_single_decode4(self):
        slot_size = 32
        parms = EncryptionParameters(scheme_type.CKKS)
        parms.set_poly_modulus_degree(slot_size * 2)
        parms.set_coeff_modulus(CoeffModulus.Create(2 * slot_size, [ 40, 40, 40, 40 ]))
        context = SEALContext.Create(parms, False, sec_level_type.none)
        encoder = CKKSEncoder(context)

        #  Use a scale over 128 bits
        data_bound = (1 << 20)
        plain = Plaintext()
        result = ComplexVec([])

        for iRun in range(50):
            value = random.randint(0, data_bound)
            encoder.encode(value, context.first_parms_id(), plain)
            encoder.decode(plain, result)

            for i in range(slot_size):
                tmp = abs(value - result[i].real)
                self.assertTrue(tmp < 0.5)

    def test_CKKSEncoder_encode_vector_decode1(self):
        parms = EncryptionParameters(scheme_type.CKKS)
        slot_size = 32
        parms.set_poly_modulus_degree(2 * slot_size)
        parms.set_coeff_modulus(CoeffModulus.Create(2 * slot_size, [ 40, 40, 40, 40 ]))
        context = SEALContext.Create(parms, False, sec_level_type.none)

        values = ComplexVec([0 + 0j] * slot_size)
        encoder = CKKSEncoder(context)
        delta = (1 << 16)
        plain = Plaintext()
        encoder.encode(values, context.first_parms_id(), delta, plain)
        result = ComplexVec([])
        encoder.decode(plain, result)

        for i in range(slot_size):
            tmp = abs(values[i].real - result[i].real)
            self.assertTrue(tmp < 0.5)

    def test_CKKSEncoder_encode_vector_decode2(self):
        slot_size = 32
        parms = EncryptionParameters(scheme_type.CKKS)
        parms.set_poly_modulus_degree(2 * slot_size)
        parms.set_coeff_modulus(CoeffModulus.Create(2 * slot_size, [ 60, 60, 60, 60 ]))
        context = SEALContext.Create(parms, False, sec_level_type.none)

        data_bound = (1 << 30)

        values = ComplexVec([0 + 0j] * slot_size)
        for i in range(slot_size):
            real_part = random.randint(0, data_bound)
            values[i] = complex(real_part, 0)

        encoder = CKKSEncoder(context)
        delta = (1 << 40)
        plain = Plaintext()
        encoder.encode(values, context.first_parms_id(), delta, plain)
        result = ComplexVec([])
        encoder.decode(plain, result)

        for i in range(slot_size):
            tmp = abs(values[i].real - result[i].real)
            self.assertTrue(tmp < 0.5)

    def test_CKKSEncoder_encode_vector_decode3(self):
        slot_size = 64
        parms = EncryptionParameters(scheme_type.CKKS)
        parms.set_poly_modulus_degree(2 * slot_size)
        parms.set_coeff_modulus(CoeffModulus.Create(2 * slot_size, [ 60, 60, 60 ]))
        context = SEALContext.Create(parms, False, sec_level_type.none)

        values = ComplexVec([0 + 0j] * slot_size)
        data_bound = (1 << 30)

        for i in range(slot_size):
            real_part = random.randint(0, data_bound)
            values[i] = complex(real_part, 0)

        encoder = CKKSEncoder(context)
        delta = (1 << 40)
        plain = Plaintext()
        encoder.encode(values, context.first_parms_id(), delta, plain)
        result = ComplexVec([])
        encoder.decode(plain, result)

        for i in range(slot_size):
            tmp = abs(values[i].real - result[i].real)
            self.assertTrue(tmp < 0.5)

    def test_CKKSEncoder_encode_vector_decode4(self):
        parms = EncryptionParameters(scheme_type.CKKS)
        slot_size = 64
        parms.set_poly_modulus_degree(2 * slot_size)
        parms.set_coeff_modulus(CoeffModulus.Create(2 * slot_size, [ 30, 30, 30, 30, 30 ]))
        context = SEALContext.Create(parms, False, sec_level_type.none)

        values = ComplexVec([0. + 0.j] * slot_size)

        data_bound = (1 << 30)

        for i in range(slot_size):
            real_part = random.randint(0, data_bound)
            values[i] = complex(real_part, 0)

        encoder = CKKSEncoder(context)
        delta = (1 << 40)
        plain = Plaintext()
        encoder.encode(values, context.first_parms_id(), delta, plain)
        result = ComplexVec([])
        encoder.decode(plain, result)

        for i in range(slot_size):
            tmp = abs(values[i].real - result[i].real)
            self.assertTrue(tmp < 0.5)

    def test_CKKSEncoder_encode_vector_decode5(self):
        slot_size = 32
        parms = EncryptionParameters(scheme_type.CKKS)
        parms.set_poly_modulus_degree(128)
        parms.set_coeff_modulus(CoeffModulus.Create(2 * slot_size, [ 30, 30, 30, 30, 30 ]))
        context = SEALContext.Create(parms, False, sec_level_type.none)

        values = ComplexVec([0. + 0.j] * slot_size)

        data_bound = (1 << 30)

        for i in range(slot_size):
            real_part = random.randint(0, data_bound)
            values[i] = complex(real_part, 0)

        encoder = CKKSEncoder(context)
        delta = (1 << 40)
        plain = Plaintext()
        encoder.encode(values, context.first_parms_id(), delta, plain)
        result = ComplexVec([])
        encoder.decode(plain, result)

        for i in range(slot_size):
            tmp = abs(values[i].real - result[i].real)
            self.assertTrue(tmp < 0.5)

    def test_CKKSEncoder_encode_vector_decode5(self):
        # Many primes
        slot_size = 32
        parms = EncryptionParameters(scheme_type.CKKS)
        parms.set_poly_modulus_degree(128)
        parms.set_coeff_modulus(CoeffModulus.Create(2 * slot_size, [30]*19))
        context = SEALContext.Create(parms, False, sec_level_type.none)

        values = ComplexVec([0. + 0.j] * slot_size)

        data_bound = (1 << 30)

        for i in range(slot_size):
            real_part = random.randint(0, data_bound)
            values[i] = complex(real_part, 0)

        encoder = CKKSEncoder(context)
        delta = (1 << 40)
        plain = Plaintext()
        encoder.encode(values, context.first_parms_id(), delta, plain)
        result = ComplexVec([])
        encoder.decode(plain, result)

        for i in range(slot_size):
            tmp = abs(values[i].real - result[i].real)
            self.assertTrue(tmp < 0.5)

    def test_CKKSEncoder_encode_vector_decode5(self):
        parms = EncryptionParameters(scheme_type.CKKS)
        slot_size = 64
        parms.set_poly_modulus_degree(2 * slot_size)
        parms.set_coeff_modulus(CoeffModulus.Create(2 * slot_size, [40, 40, 40, 40, 40 ]))
        context = SEALContext.Create(parms, False, sec_level_type.none)

        values = ComplexVec([0. + 0.j] * slot_size)

        data_bound = (1 << 20)

        for i in range(slot_size):
            real_part = random.randint(0, data_bound)
            values[i] = complex(real_part, 0)

        encoder = CKKSEncoder(context)
        # Use a very large scale
        delta = pow(2.0, 110)
        plain = Plaintext()
        encoder.encode(values, context.first_parms_id(), delta, plain)
        result = ComplexVec([])
        encoder.decode(plain, result)

        for i in range(slot_size):
            tmp = abs(values[i].real - result[i].real)
            self.assertTrue(tmp < 0.5)

        # Use a scale over 128 bits
        delta = pow(2.0, 130)
        plain = Plaintext()
        encoder.encode(values, context.first_parms_id(), delta, plain)
        result = ComplexVec([])
        encoder.decode(plain, result)

        for i in range(slot_size):
            tmp = abs(values[i].real - result[i].real)
            self.assertTrue(tmp < 0.5)


class CKKSEncryptMultiplyByNumberDecrypt(unittest.TestCase):
    def test_mult_two_random_vectors_by_integer(self):
        parms = EncryptionParameters(scheme_type.CKKS)
        slot_size = 32
        parms.set_poly_modulus_degree(2 * slot_size)
        parms.set_coeff_modulus(CoeffModulus.Create(slot_size * 2, [ 60, 60, 40]))
        context = SEALContext.Create(parms, False, sec_level_type.none)
        keygen = KeyGenerator(context)
        encoder = CKKSEncoder(context)
        encryptor = Encryptor(context, keygen.public_key())
        decryptor = Decryptor(context, keygen.secret_key())
        evaluator = Evaluator(context)

        encrypted1 = Ciphertext()
        plain1 = Plaintext()
        plain2 = Plaintext()
        plainRes = Plaintext()

        input1 = ComplexVec([0]*slot_size)
        expected =ComplexVec([0]*slot_size)

        data_bound = 1 << 10

        for iExp in range(50):
            input2 = random.randint(1, data_bound)
            for i in range(slot_size):
                input1[i] = random.uniform(0, data_bound)
                expected[i] = input1[i] * input2

            output = ComplexVec([0] * slot_size)
            delta = 1 << 40
            encoder.encode(input1, context.first_parms_id(), delta, plain1)
            encoder.encode(input2, context.first_parms_id(), plain2)

            encryptor.encrypt(plain1, encrypted1)
            evaluator.multiply_plain_inplace(encrypted1, plain2)

            #check correctness of encryption
            self.assertTrue(encrypted1.parms_id == context.first_parms_id())

            decryptor.decrypt(encrypted1, plainRes)
            encoder.decode(plainRes, output)
            for i in range(slot_size):
                tmp = abs(expected[i].real - output[i].real)
                if (tmp > 0.5):
                    print('Test failed on trial', iExp)
                    print('Diff', tmp)
                    print('input1', input1[i])
                    print('input2', input2)
                    print('Expected', expected[i])
                    print('output', output[i])
                self.assertTrue(tmp < 0.5)

    def test_mult_two_random_vectors_by_integer2(self):
        parms = EncryptionParameters(scheme_type.CKKS)
        slot_size = 8
        parms.set_poly_modulus_degree(64)
        parms.set_coeff_modulus(CoeffModulus.Create(64, [60, 60]))
        context = SEALContext.Create(parms, False, sec_level_type.none)
        keygen = KeyGenerator(context)
        encoder = CKKSEncoder(context)
        encryptor = Encryptor(context, keygen.public_key())
        decryptor = Decryptor(context, keygen.secret_key())
        evaluator = Evaluator(context)

        encrypted1 = Ciphertext()
        plain1 = Plaintext()
        plain2 = Plaintext()
        plainRes = Plaintext()

        input1 = ComplexVec([0]*slot_size)
        expected =ComplexVec([0]*slot_size)

        data_bound = 1 << 10

        for iExp in range(50):
            input2 = random.randint(1, data_bound)
            for i in range(slot_size):
                input1[i] = random.uniform(0, data_bound)
                expected[i] = input1[i] * input2

            output = ComplexVec([0] * slot_size)
            delta = 1 << 40
            encoder.encode(input1, context.first_parms_id(), delta, plain1)
            encoder.encode(input2, context.first_parms_id(), plain2)

            encryptor.encrypt(plain1, encrypted1)
            evaluator.multiply_plain_inplace(encrypted1, plain2)

            #check correctness of encryption
            self.assertTrue(encrypted1.parms_id == context.first_parms_id())

            decryptor.decrypt(encrypted1, plainRes)
            encoder.decode(plainRes, output)
            for i in range(slot_size):
                tmp = abs(expected[i].real - output[i].real)
                if (tmp > 0.5):
                    print('Test failed on trial', iExp)
                    print('Diff', tmp)
                    print('input1', input1[i])
                    print('input2', input2)
                    print('Expected', expected[i])
                    print('output', output[i])
                self.assertTrue(tmp < 0.5)


    def test_mult_two_random_vectors_by_double(self):
        parms = EncryptionParameters(scheme_type.CKKS)
        slot_size = 32
        parms.set_poly_modulus_degree(2 * slot_size)
        parms.set_coeff_modulus(CoeffModulus.Create(slot_size * 2, [ 60, 60, 60]))
        context = SEALContext.Create(parms, False, sec_level_type.none)
        keygen = KeyGenerator(context)
        encoder = CKKSEncoder(context)
        encryptor = Encryptor(context, keygen.public_key())
        decryptor = Decryptor(context, keygen.secret_key())
        evaluator = Evaluator(context)

        encrypted1 = Ciphertext()
        plain1 = Plaintext()
        plain2 = Plaintext()
        plainRes = Plaintext()

        input1 = ComplexVec([0]*slot_size)
        expected =ComplexVec([0]*slot_size)

        data_bound = 1 << 10

        for iExp in range(50):
            input2 = random.uniform(0., float(data_bound))
            for i in range(slot_size):
                input1[i] = random.uniform(0, data_bound)
                expected[i] = input1[i] * input2

            output = ComplexVec([0] * slot_size)
            delta = 1 << 40
            encoder.encode(input1, context.first_parms_id(), delta, plain1)
            encoder.encode(input2, context.first_parms_id(), delta, plain2)

            encryptor.encrypt(plain1, encrypted1)
            evaluator.multiply_plain_inplace(encrypted1, plain2)

            #check correctness of encryption
            self.assertTrue(encrypted1.parms_id == context.first_parms_id())

            decryptor.decrypt(encrypted1, plainRes)
            encoder.decode(plainRes, output)
            for i in range(slot_size):
                tmp = abs(expected[i].real - output[i].real)
                if (tmp > 0.5):
                    print('Test failed on trial', iExp)
                    print('Diff', tmp)
                    print('input1', input1[i])
                    print('input2', input2)
                    print('Expected', expected[i])
                    print('output', output[i])
                self.assertTrue(tmp < 0.5)

    def test_mult_two_random_vectors_by_double2(self):
        parms = EncryptionParameters(scheme_type.CKKS)
        slot_size = 32
        parms.set_poly_modulus_degree(64)
        parms.set_coeff_modulus(CoeffModulus.Create(slot_size * 2, [ 60, 60, 60]))
        context = SEALContext.Create(parms, False, sec_level_type.none)
        keygen = KeyGenerator(context)
        encoder = CKKSEncoder(context)
        encryptor = Encryptor(context, keygen.public_key())
        decryptor = Decryptor(context, keygen.secret_key())
        evaluator = Evaluator(context)

        encrypted1 = Ciphertext()
        plain1 = Plaintext()
        plain2 = Plaintext()
        plainRes = Plaintext()

        input1 = ComplexVec([0]*slot_size)
        expected =ComplexVec([0]*slot_size)

        data_bound = 1 << 10

        for iExp in range(50):
            input2 = random.uniform(0., float(data_bound))
            for i in range(slot_size):
                input1[i] = random.uniform(0, data_bound)
                expected[i] = input1[i] * input2

            output = ComplexVec([2.1] * slot_size)
            delta = 1 << 40
            encoder.encode(input1, context.first_parms_id(), delta, plain1)
            encoder.encode(input2, context.first_parms_id(), delta, plain2)

            encryptor.encrypt(plain1, encrypted1)
            evaluator.multiply_plain_inplace(encrypted1, plain2)

            #check correctness of encryption
            self.assertTrue(encrypted1.parms_id == context.first_parms_id())

            decryptor.decrypt(encrypted1, plainRes)
            encoder.decode(plainRes, output)
            for i in range(slot_size):
                tmp = abs(expected[i].real - output[i].real)
                if (tmp > 0.5):
                    print('Test failed on trial', iExp)
                    print('Diff', tmp)
                    print('input1', input1[i])
                    print('input2', input2)
                    print('Expected', expected[i])
                    print('output', output[i])
                self.assertTrue(tmp < 0.5)

class CKKSEncryptMultiplyRelinDecrypt(unittest.TestCase):
    def test_mult_relin_two_random_vectors(self):
        parms = EncryptionParameters(scheme_type.CKKS)
        slot_size = 32
        parms.set_poly_modulus_degree(2 * slot_size)
        parms.set_coeff_modulus(CoeffModulus.Create(slot_size * 2, [ 60, 60, 60]))
        context = SEALContext.Create(parms, False, sec_level_type.none)
        keygen = KeyGenerator(context)
        encoder = CKKSEncoder(context)
        encryptor = Encryptor(context, keygen.public_key())
        decryptor = Decryptor(context, keygen.secret_key())
        evaluator = Evaluator(context)
        rlk = keygen.relin_keys()

        encrypted1 = Ciphertext()
        encrypted2 = Ciphertext()
        encryptedRes = Ciphertext()
        plain1 = Plaintext()
        plain2 = Plaintext()
        plainRes = Plaintext()

        input1 = ComplexVec([0]*slot_size)
        input2 = ComplexVec([0]*slot_size)
        expected =ComplexVec([0]*slot_size)

        data_bound = 1 << 10

        for iExp in range(50):
            for i in range(slot_size):
                input1[i] = random.uniform(0, data_bound)
                input2[i] = random.uniform(0, data_bound)
                expected[i] = input1[i] * input2[i]

            output = ComplexVec([0] * slot_size)
            delta = 1 << 40
            encoder.encode(input1, context.first_parms_id(), delta, plain1)
            encoder.encode(input2, context.first_parms_id(), delta, plain2)

            encryptor.encrypt(plain1, encrypted1)
            encryptor.encrypt(plain2, encrypted2)

            #check correctness of encryption
            self.assertTrue(encrypted1.parms_id == context.first_parms_id())
            self.assertTrue(encrypted2.parms_id == context.first_parms_id())

            evaluator.multiply_inplace(encrypted1, encrypted2)
            evaluator.relinearize_inplace(encrypted1, rlk)

            decryptor.decrypt(encrypted1, plainRes)
            encoder.decode(plainRes, output)
            for i in range(slot_size):
                tmp = abs(expected[i].real - output[i].real)
                if (tmp > 0.5):
                    print('Test failed on trial', iExp)
                    print('Diff', tmp)
                    print('input1', input1[i])
                    print('input2', input2)
                    print('Expected', expected[i])
                    print('output', output[i])
                self.assertTrue(tmp < 0.5)

class CKKSEncryptSquareRelinDecrypt(unittest.TestCase):
    def test_square_two_random_vectors(self):
        parms = EncryptionParameters(scheme_type.CKKS)
        slot_size = 32
        parms.set_poly_modulus_degree(2 * slot_size)
        parms.set_coeff_modulus(CoeffModulus.Create(slot_size * 2, [ 60, 60, 60]))
        context = SEALContext.Create(parms, False, sec_level_type.none)
        keygen = KeyGenerator(context)
        encoder = CKKSEncoder(context)
        encryptor = Encryptor(context, keygen.public_key())
        decryptor = Decryptor(context, keygen.secret_key())
        evaluator = Evaluator(context)
        rlk = keygen.relin_keys()

        encrypted = Ciphertext()
        plain = Plaintext()
        plainRes = Plaintext()

        input = ComplexVec([0]*slot_size)
        expected =ComplexVec([0]*slot_size)

        data_bound = 1 << 7

        for iExp in range(50):
            for i in range(slot_size):
                input[i] = random.uniform(0, data_bound)
                expected[i] = input[i] * input[i]

            output = ComplexVec([0] * slot_size)
            delta = 1 << 40
            encoder.encode(input, context.first_parms_id(), delta, plain)

            encryptor.encrypt(plain, encrypted)

            #check correctness of encryption
            self.assertTrue(encrypted.parms_id == context.first_parms_id())

            evaluator.square_inplace(encrypted)
            evaluator.relinearize_inplace(encrypted, rlk)

            decryptor.decrypt(encrypted, plainRes)
            encoder.decode(plainRes, output)
            for i in range(slot_size):
                tmp = abs(expected[i].real - output[i].real)
                if (tmp > 0.5):
                    print('Test failed on trial', iExp)
                    print('Diff', tmp)
                    print('input1', input[i])
                    print('Expected', expected[i])
                    print('output', output[i])
                self.assertTrue(tmp < 0.5)

class CKKSEncryptMultiplyRelinRescaleDecrypt(unittest.TestCase):
    def test_mult_relin_rescale_two_random_vectors(self):
        parms = EncryptionParameters(scheme_type.CKKS)
        slot_size = 64
        parms.set_poly_modulus_degree(2 * slot_size)
        parms.set_coeff_modulus(CoeffModulus.Create(slot_size * 2,
                [ 30, 30, 30, 30, 30, 30 ]))
        context = SEALContext.Create(parms, True, sec_level_type.none)
        keygen = KeyGenerator(context)
        encoder = CKKSEncoder(context)
        encryptor = Encryptor(context, keygen.public_key())
        decryptor = Decryptor(context, keygen.secret_key())
        evaluator = Evaluator(context)
        rlk = keygen.relin_keys()

        encrypted1 = Ciphertext()
        encrypted2 = Ciphertext()
        encryptedRes = Ciphertext()
        plain1 = Plaintext()
        plain2 = Plaintext()
        plainRes = Plaintext()

        input1 = ComplexVec([0]*slot_size)
        input2 = ComplexVec([0]*slot_size)
        expected =ComplexVec([0]*slot_size)

        data_bound = 1 << 10

        for iExp in range(50):
            for i in range(slot_size):
                input1[i] = random.uniform(0, data_bound)
                input2[i] = random.uniform(0, data_bound)
                expected[i] = input1[i] * input2[i]

            output = ComplexVec([0] * slot_size)
            delta = 1 << 40
            encoder.encode(input1, context.first_parms_id(), delta, plain1)
            encoder.encode(input2, context.first_parms_id(), delta, plain2)

            encryptor.encrypt(plain1, encrypted1)
            encryptor.encrypt(plain2, encrypted2)

            #check correctness of encryption
            self.assertTrue(encrypted1.parms_id == context.first_parms_id())
            self.assertTrue(encrypted2.parms_id == context.first_parms_id())

            evaluator.multiply_inplace(encrypted1, encrypted2)
            evaluator.relinearize_inplace(encrypted1, rlk)
            evaluator.rescale_to_next_inplace(encrypted1)

            decryptor.decrypt(encrypted1, plainRes)
            encoder.decode(plainRes, output)
            for i in range(slot_size):
                tmp = abs(expected[i].real - output[i].real)
                if (tmp > 0.5):
                    print('Test failed on trial', iExp)
                    print('Diff', tmp)
                    print('input1', input1[i])
                    print('input2', input2)
                    print('Expected', expected[i])
                    print('output', output[i])
                self.assertTrue(tmp < 0.5)

class CKKSEncryptModSwitchDecrypt(unittest.TestCase):
    def test_modulo_switch_without_rescaling(self):
        parms = EncryptionParameters(scheme_type.CKKS)
        slot_size = 64
        parms.set_poly_modulus_degree(2 * slot_size)
        parms.set_coeff_modulus(CoeffModulus.Create(128,
                [ 60, 60, 60, 60, 60 ]))
        context = SEALContext.Create(parms, True, sec_level_type.none)
        next_parms_id = context.first_context_data().next_context_data().parms_id()
        keygen = KeyGenerator(context)
        encoder = CKKSEncoder(context)
        encryptor = Encryptor(context, keygen.public_key())
        decryptor = Decryptor(context, keygen.secret_key())
        evaluator = Evaluator(context)

        encrypted = Ciphertext()
        plain = Plaintext()
        plainRes = Plaintext()

        input = ComplexVec([0] * slot_size)
        output = ComplexVec([0] * slot_size)

        data_bound = 1 << 30

        for iExp in range(50):
            for i in range(slot_size):
                input[i] = random.uniform(0, data_bound)

            output = ComplexVec([0] * slot_size)
            delta = 1 << 40
            encoder.encode(input, context.first_parms_id(), delta, plain)
            encryptor.encrypt(plain, encrypted)

            #check correctness of encryption
            self.assertTrue(encrypted.parms_id == context.first_parms_id())

            evaluator.mod_switch_to_next_inplace(encrypted)

            #check correctness of moduluo switching
            self.assertTrue(encrypted.parms_id == next_parms_id)

            decryptor.decrypt(encrypted, plainRes)
            encoder.decode(plainRes, output)
            for i in range(slot_size):
                tmp = abs(input[i].real - output[i].real)
                if (tmp > 0.5):
                    print('Test failed on trial', iExp)
                    print('Diff', tmp)
                    print('input', input[i])
                    print('output', output[i])
                self.assertTrue(tmp < 0.5)

class CKKSEncryptMultiplyRelinRescaleModSwitchAddDecrypt(unittest.TestCase):
     def test_mult_add_without_rescaling(self):
        parms = EncryptionParameters(scheme_type.CKKS)
        slot_size = 64
        parms.set_poly_modulus_degree(2 * slot_size)
        parms.set_coeff_modulus(CoeffModulus.Create(slot_size * 2, [ 50, 50, 50 ]))
        context = SEALContext.Create(parms, True, sec_level_type.none)
        next_parms_id = context.first_context_data().next_context_data().parms_id()
        keygen = KeyGenerator(context)
        encoder = CKKSEncoder(context)
        encryptor = Encryptor(context, keygen.public_key())
        decryptor = Decryptor(context, keygen.secret_key())
        evaluator = Evaluator(context)

        rlk = keygen.relin_keys()

        encrypted1 = Ciphertext()
        encrypted2 = Ciphertext()
        encrypted3 = Ciphertext()
        plain1 = Plaintext()
        plain2 = Plaintext()
        plain3 = Plaintext()
        plainRes = Plaintext()

        input1 = ComplexVec([0] * slot_size)
        input2 = ComplexVec([0] * slot_size)
        input3 = ComplexVec([0] * slot_size)
        expected = ComplexVec([0] * slot_size)

        data_bound = 1 << 8

        for iExp in range(100):
            for i in range(slot_size):
                input1[i] = random.uniform(0, data_bound)
                input2[i] = random.uniform(0, data_bound)
                input3[i] = random.uniform(0, data_bound)
                expected[i] = input1[i] * input2[i] + input3[i]

            output = ComplexVec([0] * slot_size)
            delta = 1 << 40
            encoder.encode(input1, context.first_parms_id(), delta, plain1)
            encoder.encode(input2, context.first_parms_id(), delta, plain2)
            encoder.encode(input3, context.first_parms_id(), delta * delta, plain3)

            encryptor.encrypt(plain1, encrypted1)
            encryptor.encrypt(plain2, encrypted2)
            encryptor.encrypt(plain3, encrypted3)

            #check correctness of encryption
            self.assertTrue(encrypted1.parms_id == context.first_parms_id())
            self.assertTrue(encrypted2.parms_id == context.first_parms_id())
            self.assertTrue(encrypted3.parms_id == context.first_parms_id())

            evaluator.multiply_inplace(encrypted1, encrypted2)
            evaluator.relinearize_inplace(encrypted1, rlk)
            evaluator.rescale_to_next_inplace(encrypted1)

            #check correctness of modulo switching with rescaling
            self.assertTrue(encrypted1.parms_id == next_parms_id)

            #move enc3 to the level of enc1 * enc2
            evaluator.rescale_to_inplace(encrypted3, next_parms_id)

            #enc1*enc2 + enc3
            evaluator.add_inplace(encrypted1, encrypted3)

            decryptor.decrypt(encrypted1, plainRes)
            encoder.decode(plainRes, output)
            for i in range(slot_size):
                tmp = abs(expected[i] - output[i])
                if (tmp > 0.5):
                    print('Test failed on trial', iExp)
                    print('Diff', tmp)
                    print('output', output[i])
                self.assertTrue(tmp < 0.5)


class CKKSEncryptRotateDecrypt(unittest.TestCase):
    def test_rotate_small(self):
        parms = EncryptionParameters(scheme_type.CKKS)
        slot_size = 4
        parms.set_poly_modulus_degree(2 * slot_size)
        parms.set_coeff_modulus(CoeffModulus.Create(2 * slot_size, [ 40, 40, 40, 40 ]))

        context = SEALContext.Create(parms, False, sec_level_type.none)
        keygen = KeyGenerator(context)
        encoder = CKKSEncoder(context)
        encryptor = Encryptor(context, keygen.public_key())
        decryptor = Decryptor(context, keygen.secret_key())
        evaluator = Evaluator(context)

        glk = keygen.galois_keys()

        delta = 1 << 30

        encrypted = Ciphertext()
        plain = Plaintext()

        input = ComplexVec([1.+1.j, 2.+2.j, 3.+3.j, 4.+4.j])
        output = ComplexVec([0] * slot_size)

        encoder.encode(input, context.first_parms_id(), delta, plain)
        shift = 1
        encryptor.encrypt(plain, encrypted)
        evaluator.rotate_vector_inplace(encrypted, shift, glk)

        decryptor.decrypt(encrypted, plain)
        encoder.decode(plain, output)
        for i in range(slot_size):
            self.assertEqual(input[(i + (shift)) % slot_size].real, round(output[i].real))
            self.assertEqual(input[(i + (shift)) % slot_size].imag, round(output[i].imag))

        encoder.encode(input, context.first_parms_id(), delta, plain)
        shift = 2
        encryptor.encrypt(plain, encrypted)
        evaluator.rotate_vector_inplace(encrypted, shift, glk)
        decryptor.decrypt(encrypted, plain)
        encoder.decode(plain, output)
        for i in range(slot_size):
            self.assertEqual(input[(i + (shift)) % slot_size].real, round(output[i].real))
            self.assertEqual(input[(i + (shift)) % slot_size].imag, round(output[i].imag))

        encoder.encode(input, context.first_parms_id(), delta, plain)
        shift = 3
        encryptor.encrypt(plain, encrypted)
        evaluator.rotate_vector_inplace(encrypted, shift, glk)
        decryptor.decrypt(encrypted, plain)
        encoder.decode(plain, output)
        for i in range(slot_size):
            self.assertEqual(input[(i + (shift)) % slot_size].real, round(output[i].real))
            self.assertEqual(input[(i + (shift)) % slot_size].imag, round(output[i].imag))

        encoder.encode(input, context.first_parms_id(), delta, plain)
        encryptor.encrypt(plain, encrypted)
        evaluator.complex_conjugate_inplace(encrypted, glk)
        decryptor.decrypt(encrypted, plain)
        encoder.decode(plain, output)
        for i in range(slot_size):
            self.assertEqual(input[i].real, round(output[i].real))
            self.assertEqual(-input[i].imag, round(output[i].imag))

    def test_rotate_large(self):
        slot_size = 32
        parms = EncryptionParameters(scheme_type.CKKS)
        parms.set_poly_modulus_degree(64)
        parms.set_coeff_modulus(CoeffModulus.Create(2 * slot_size, [ 40, 40, 40, 40 ]))
        context = SEALContext.Create(parms, False, sec_level_type.none)
        keygen = KeyGenerator(context)
        glk = keygen.galois_keys()

        encryptor = Encryptor(context, keygen.public_key())
        evaluator = Evaluator(context)
        decryptor = Decryptor(context, keygen.secret_key())
        encoder = CKKSEncoder(context)

        delta = 1 << 30

        encrypted = Ciphertext()
        plain = Plaintext()

        input = ComplexVec([1.+1.j, 2.+2.j, 3.+3.j, 4.+4.j] + [0]*28)
        output = ComplexVec([0] * slot_size)

        encoder.encode(input, context.first_parms_id(), delta, plain)
        shift = 1
        encryptor.encrypt(plain, encrypted)
        evaluator.rotate_vector_inplace(encrypted, shift, glk)
        decryptor.decrypt(encrypted, plain)
        encoder.decode(plain, output)

        for i in range(slot_size):
            self.assertEqual(round(input[(i + (shift)) % slot_size].real), round(output[i].real))
            self.assertEqual(round(input[(i + (shift)) % slot_size].imag), round(output[i].imag))

        encoder.encode(input, context.first_parms_id(), delta, plain)
        shift = 2
        encryptor.encrypt(plain, encrypted)
        evaluator.rotate_vector_inplace(encrypted, shift, glk)
        decryptor.decrypt(encrypted, plain)
        encoder.decode(plain, output)
        for i in range(slot_size):
            self.assertEqual(round(input[(i + (shift)) % slot_size].real), round(output[i].real))
            self.assertEqual(round(input[(i + (shift)) % slot_size].imag), round(output[i].imag))

        encoder.encode(input, context.first_parms_id(), delta, plain)
        shift = 3
        encryptor.encrypt(plain, encrypted)
        evaluator.rotate_vector_inplace(encrypted, shift, glk)
        decryptor.decrypt(encrypted, plain)
        encoder.decode(plain, output)
        for i in range(slot_size):
            self.assertEqual(round(input[(i + (shift)) % slot_size].real), round(output[i].real))
            self.assertEqual(round(input[(i + (shift)) % slot_size].imag), round(output[i].imag))

        encoder.encode(input, context.first_parms_id(), delta, plain)
        encryptor.encrypt(plain, encrypted)
        evaluator.complex_conjugate_inplace(encrypted, glk)
        decryptor.decrypt(encrypted, plain)
        encoder.decode(plain, output)
        for i in range(slot_size):
            self.assertEqual(round(input[i].real), round(output[i].real))
            self.assertEqual(round(-input[i].imag), round(output[i].imag))


class CKKSEncryptRescaleRotateDecrypt(unittest.TestCase):
    def test_small(self):
        parms = EncryptionParameters(scheme_type.CKKS)
        slot_size = 4
        parms.set_poly_modulus_degree(2 * slot_size)
        parms.set_coeff_modulus(CoeffModulus.Create(2 * slot_size, [ 40, 40, 40, 40 ]))

        context = SEALContext.Create(parms, True, sec_level_type.none)
        keygen = KeyGenerator(context)
        glk = keygen.galois_keys()

        encryptor = Encryptor(context, keygen.public_key())
        evaluator = Evaluator(context)
        decryptor = Decryptor(context, keygen.secret_key())
        encoder = CKKSEncoder(context)
        delta = 1 << 70

        encrypted = Ciphertext()
        plain = Plaintext()

        input = ComplexVec([1.+1.j, 2.+2.j, 3.+3.j, 4.+4.j])
        output = ComplexVec([0] * slot_size)

        encoder.encode(input, context.first_parms_id(), delta, plain)
        shift = 1
        encryptor.encrypt(plain, encrypted)
        evaluator.rescale_to_next_inplace(encrypted)
        evaluator.rotate_vector_inplace(encrypted, shift, glk)

        decryptor.decrypt(encrypted, plain)
        encoder.decode(plain, output)
        for i in range(slot_size):
            self.assertEqual(input[(i + (shift)) % slot_size].real, round(output[i].real))
            self.assertEqual(input[(i + (shift)) % slot_size].imag, round(output[i].imag))

        encoder.encode(input, context.first_parms_id(), delta, plain)
        shift = 2
        encryptor.encrypt(plain, encrypted)
        evaluator.rescale_to_next_inplace(encrypted)
        evaluator.rotate_vector_inplace(encrypted, shift, glk)
        decryptor.decrypt(encrypted, plain)
        encoder.decode(plain, output)
        for i in range(slot_size):
            self.assertEqual(input[(i + (shift)) % slot_size].real, round(output[i].real))
            self.assertEqual(input[(i + (shift)) % slot_size].imag, round(output[i].imag))

        encoder.encode(input, context.first_parms_id(), delta, plain)
        shift = 3
        encryptor.encrypt(plain, encrypted)
        evaluator.rescale_to_next_inplace(encrypted)
        evaluator.rotate_vector_inplace(encrypted, shift, glk)
        decryptor.decrypt(encrypted, plain)
        encoder.decode(plain, output)
        for i in range(slot_size):
            self.assertEqual(input[(i + (shift)) % slot_size].real, round(output[i].real))
            self.assertEqual(input[(i + (shift)) % slot_size].imag, round(output[i].imag))

        encoder.encode(input, context.first_parms_id(), delta, plain)
        encryptor.encrypt(plain, encrypted)
        evaluator.rescale_to_next_inplace(encrypted)
        evaluator.complex_conjugate_inplace(encrypted, glk)
        decryptor.decrypt(encrypted, plain)
        encoder.decode(plain, output)
        for i in range(slot_size):
            self.assertEqual(input[i].real, round(output[i].real))
            self.assertEqual(-input[i].imag, round(output[i].imag))

    def test_large(self):
        parms = EncryptionParameters(scheme_type.CKKS)
        slot_size = 32
        parms.set_poly_modulus_degree(64)
        parms.set_coeff_modulus(CoeffModulus.Create(2 * slot_size, [ 40, 40, 40, 40 ]))
        context = SEALContext.Create(parms, True, sec_level_type.none)
        keygen = KeyGenerator(context)
        encoder = CKKSEncoder(context)
        encryptor = Encryptor(context, keygen.public_key())
        decryptor = Decryptor(context, keygen.secret_key())
        evaluator = Evaluator(context)

        glk = keygen.galois_keys()

        delta = 1 << 70

        encrypted = Ciphertext()
        plain = Plaintext()

        input = ComplexVec([1.+1.j, 2.+2.j, 3.+3.j, 4.+4.j] + [0]*28)
        output = ComplexVec([0] * slot_size)

        encoder.encode(input, context.first_parms_id(), delta, plain)
        shift = 1
        encryptor.encrypt(plain, encrypted)
        evaluator.rescale_to_next_inplace(encrypted)
        evaluator.rotate_vector_inplace(encrypted, shift, glk)

        decryptor.decrypt(encrypted, plain)
        encoder.decode(plain, output)
        for i in range(slot_size):
            self.assertEqual(input[(i + (shift)) % slot_size].real, round(output[i].real))
            self.assertEqual(input[(i + (shift)) % slot_size].imag, round(output[i].imag))

        encoder.encode(input, context.first_parms_id(), delta, plain)
        shift = 2
        encryptor.encrypt(plain, encrypted)
        evaluator.rescale_to_next_inplace(encrypted)
        evaluator.rotate_vector_inplace(encrypted, shift, glk)
        decryptor.decrypt(encrypted, plain)
        encoder.decode(plain, output)
        for i in range(slot_size):
            self.assertEqual(input[(i + (shift)) % slot_size].real, round(output[i].real))
            self.assertEqual(input[(i + (shift)) % slot_size].imag, round(output[i].imag))

        encoder.encode(input, context.first_parms_id(), delta, plain)
        shift = 3
        encryptor.encrypt(plain, encrypted)
        evaluator.rescale_to_next_inplace(encrypted)
        evaluator.rotate_vector_inplace(encrypted, shift, glk)
        decryptor.decrypt(encrypted, plain)
        encoder.decode(plain, output)
        for i in range(slot_size):
            self.assertEqual(input[(i + (shift)) % slot_size].real, round(output[i].real))
            self.assertEqual(input[(i + (shift)) % slot_size].imag, round(output[i].imag))

        encoder.encode(input, context.first_parms_id(), delta, plain)
        encryptor.encrypt(plain, encrypted)
        evaluator.rescale_to_next_inplace(encrypted)
        evaluator.complex_conjugate_inplace(encrypted, glk)
        decryptor.decrypt(encrypted, plain)
        encoder.decode(plain, output)
        for i in range(slot_size):
            self.assertEqual(input[i].real, round(output[i].real))
            self.assertEqual(-input[i].imag, round(output[i].imag))

