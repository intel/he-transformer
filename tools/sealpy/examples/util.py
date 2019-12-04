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


def print_parameters(context):

    context_data = context.key_context_data()

    scheme_name = ''
    scheme_type = context_data.parms().scheme()
    if scheme_type == scheme_type.BFV:
        scheme_name = 'BFV'
    elif scheme_type == scheme_type.CKKS:
        scheme_name = 'CKKS'
    else:
        raise Exception('Unsupported scheme')

    print('/')
    print('| Encryption parameters:')
    print('|    scheme:', scheme_name)
    print("|    poly_modulus:", context_data.parms().poly_modulus_degree())

    # Print the size of the true (product) coefficient modulus
    print(
        "|    coeff_modulus_size:",
        context_data.total_coeff_modulus_bit_count(),
        ' (',
        end='')
    coeff_modulus = context_data.parms().coeff_modulus()
    coeff_mod_count = len(coeff_modulus)
    for i in range(coeff_mod_count - 1):
        print(coeff_modulus[i].bit_count(), ' + ', end='')
    print(coeff_modulus[-1].bit_count(), end='')
    print(') bits')

    if (scheme_type == scheme_type.BFV):
        print("|     plain_modulus:",
              context_data.parms().plain_modulus().value())
    print('\\\n')


def print_vector(vec, print_size=4, prec=3):
    fmt_string = "%0." + str(prec) + "f"
    print('\n\t[', end=' ')

    slot_count = len(vec)
    if slot_count < 2 * print_size:
        for value in vec:
            print(fmt_string % value, end=' ')
    else:
        for i in range(print_size):
            print(fmt_string % vec[i], end=' ')
        print('...', end=' ')
        for i in range(print_size, -1, -1):
            print(fmt_string % vec[len(vec) - i - 1], end=' ')
    print(']\n')


def print_matrix(matrix, row_size):
    prec = 3
    fmt_string = "%0." + str(prec) + "f"
    print_size = 5
    print('\n\t[', end=' ')
    for i in range(print_size):
        print(fmt_string % matrix[i], end=', ')
    print(' ..., ', end='')
    for i in range(row_size - print_size, row_size):
        print(fmt_string % matrix[i], end='')
        if i != row_size - 1:
            print(', ', end='')
        else:
            print(' ]')

    print('\t[', end=' ')
    for i in range(row_size, row_size + print_size):
        print(fmt_string % matrix[i], end=', ')
    print(' ..., ', end='')
    for i in range(2 * row_size - print_size, 2 * row_size):
        print(fmt_string % matrix[i], end='')
        if i != 2 * row_size - 1:
            print(', ', end='')
        else:
            print(' ]\n')


def print_example_banner(title):
    banner_len = len(title) + 2 * 10
    top_row = '+' + '-' * (banner_len - 2) + '+'
    print(top_row)
    print('|' + 9 * ' ' + title + 9 * ' ' + '|')
    print(top_row)
