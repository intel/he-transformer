# ******************************************************************************
# Copyright 2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
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
import time
import random

import noise

from pyseal import EncryptionParameters, \
                  scheme_type, \
                  SEALContext, \
                  CoeffModulus, \
                  PlainModulus, \
                  KeyGenerator, \
                  Encryptor, \
                  Evaluator, \
                  Decryptor, \
                  CKKSEncoder, \
                  Plaintext, \
                  Ciphertext, \
                  MemoryPoolHandle, \
                  DoubleVec, \
                  sec_level_type ,\
                  ComplexVec, \
                  SmallModulus

from util import print_parameters, print_vector

import example_1, example_2, example_3, example_4, example_5, example_6

import examples_noise


def main():
    input_prompt = """
+---------------------------------------------------------+
| The following examples should be executed while reading |
| comments in associated files in native/examples/.       |
+---------------------------------------------------------+
| Examples                   | Source Files               |
+----------------------------+----------------------------+
| 1. BFV Basics              | 1_bfv_basics.cpp           |
| 2. Encoders                | 2_encoders.cpp             |
| 3. Levels                  | 3_levels.cpp               |
| 4. CKKS Basics             | 4_ckks_basics.cpp          |
| 5. Rotation                | 5_rotation.cpp             |
| 6. Performance Test        | 6_performance.cpp          |
+----------------------------+----------------------------+

> Run example (1 - 6) or exit (0):"""

    while True:
        num = int(input(input_prompt))

        if num == 0:
            exit(1)
        if num == 1:
            example_1.bfv_basics()
        elif num == 2:
            example_2.example_encoders()
        elif num == 3:
            example_3.levels()
        elif num == 4:
            example_4.ckks_basics()
        elif num == 5:
            example_5.example_rotation()
        elif num == 6:
            example_6.example_ckks_performance_default()
        else:
            print('Invalid input', num)


if __name__ == "__main__":
    main()
