//*****************************************************************************
// Copyright 2018-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#pragma once

#include <memory>
#include <vector>

#include "he_type.hpp"
#include "ngraph/type/element_type.hpp"
#include "seal/he_seal_backend.hpp"
#include "seal/kernel/add_seal.hpp"
#include "seal/kernel/multiply_seal.hpp"
#include "seal/kernel/negate_seal.hpp"
#include "seal/seal.h"
#include "seal/seal_ciphertext_wrapper.hpp"
#include "seal/seal_util.hpp"

namespace ngraph::runtime::he {

/// \brief Multiplies two ciphertext/plaintext elements and adds the result to
/// the output
///  \param[in] arg0 Cipher or plaintext data to multiply
///  \param[in] arg1 Cipher or plaintext data to multiply
/// \param[in] out Stores the ciphertext or plaintext product
///  \param[in] he_seal_backend Backend used to perform multiplication
void scalar_multiply_add_seal(HEType& arg0, HEType& arg1, HEType& out,
                              size_t batch_size,
                              HESealBackend& he_seal_backend) {
  auto prod = HEType(HEPlaintext(batch_size), false);
  if (!prod.is_ciphertext()) {
    auto empty_cipher = HESealBackend::create_empty_ciphertext();
    prod.set_ciphertext(empty_cipher);
  }

  seal::Ciphertext& encrypted = arg0.get_ciphertext()->ciphertext();
  seal::Ciphertext& out_cipher = out.get_ciphertext()->ciphertext();

  NGRAPH_CHECK(arg1.is_plaintext(), "Arg1 isn't plaintext");

  double value = arg1.get_plaintext()[0];
  // Extract encryption parameters.
  auto& context_data =
      *he_seal_backend.get_context()->get_context_data(encrypted.parms_id());
  auto& parms = context_data.parms();
  auto& coeff_modulus = parms.coeff_modulus();
  size_t coeff_count = parms.poly_modulus_degree();
  size_t coeff_mod_count = coeff_modulus.size();

  // TODO: generalize?
  size_t cipher_size = 2;

  std::vector<std::uint64_t> plaintext_vals(coeff_mod_count, 0);
  double scale = encrypted.scale();
  encode(value, element::f32, scale, encrypted.parms_id(), plaintext_vals,
         he_seal_backend);
  double new_scale = scale * scale;
  // Set the scale
  out_cipher.scale() = new_scale;

  // Prepare destination
  out_cipher.resize(he_seal_backend.get_context(), context_data.parms_id(),
                    cipher_size);
  for (size_t i = 0; i < cipher_size; i++) {
    uint64_t* encrypted_ptr = encrypted.data(i);
    uint64_t* encrypted_out_ptr = out_cipher.data(i);
    for (size_t j = 0; j < coeff_mod_count; j++) {
      // TODO(fboemer): mod < (1UL << 31) if/else condition

      uint64_t* poly = encrypted_ptr + (j * coeff_count);
      uint64_t* result = encrypted_out_ptr + (j * coeff_count);

      const auto& modulus = coeff_modulus[j];
      std::uint64_t scalar = plaintext_vals[j];

      const uint64_t modulus_value = modulus.value();
      const uint64_t const_ratio_1 = modulus.const_ratio()[1];

      // NOLINTNEXTLINE
      for (size_t k = 0; k < coeff_count; ++k, poly++, result++) {
        // Multiplication
        auto z = *poly * scalar;

        // Barrett base 2^64 reduction
        // NOLINTNEXTLINE(runtime/int)
        unsigned long long carry;
        // Carry will store the result modulo 2^64
        seal::util::multiply_uint64_hw64(z, const_ratio_1, &carry);
        // Barrett subtraction
        carry = z - carry * modulus_value;
        // Possible correction term

        // new; up to 2 correction terms, since we delay mod reduction until
        // after add
        std::uint64_t sum = *result + carry;
        *result =
            sum - (modulus_value & static_cast<uint64_t>(-static_cast<int64_t>(
                                       sum >= modulus_value)));
        *result = *result -
                  (modulus_value & static_cast<uint64_t>(-static_cast<int64_t>(
                                       *result >= modulus_value)));

        /*
        std::uint64_t mult_result =
            carry - (modulus_value &
                     static_cast<uint64_t>(
                         -static_cast<int64_t>(carry >= modulus_value)));

        std::uint64_t sum = *result + mult_result;
        *result = sum - (modulus_value &
                         static_cast<std::uint64_t>(
                             -static_cast<std::int64_t>(sum >= modulus_value)));
          */
      }
    }
  }
}

}  // namespace ngraph::runtime::he
