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

#include <vector>

#include "he_type.hpp"
#include "seal/he_seal_backend.hpp"
#include "seal/seal.h"
#include "seal/seal_ciphertext_wrapper.hpp"

namespace ngraph::runtime::he {

inline void mod_reduce_seal(std::vector<HEType>& arg,
                            HESealBackend& he_seal_backend,
                            const bool verbose = false) {
  if (verbose) {
    NGRAPH_HE_LOG(3) << "Mod-reducing " << arg.size() << " elements";
  }
  auto t0 = std::chrono::system_clock::now();

#pragma omp parallel for
  for (size_t he_idx = 0; he_idx < arg.size(); ++he_idx) {
    if (!arg[he_idx].is_ciphertext()) {
      continue;
    }

    seal::Ciphertext& encrypted = arg[he_idx].get_ciphertext()->ciphertext();
    auto& context_data =
        *he_seal_backend.get_context()->get_context_data(encrypted.parms_id());
    auto& parms = context_data.parms();
    auto& coeff_modulus = parms.coeff_modulus();
    size_t coeff_count = parms.poly_modulus_degree();
    size_t coeff_mod_count = coeff_modulus.size();
    size_t encrypted_ntt_size = encrypted.size();

    for (size_t i = 0; i < encrypted_ntt_size; i++) {
      for (size_t j = 0; j < coeff_mod_count; j++) {
        auto modulus = coeff_modulus[j];
        const uint64_t modulus_value = modulus.value();
        const uint64_t const_ratio_1 = modulus.const_ratio()[1];

        for (size_t k = 0; k < coeff_count; ++k) {
          std::uint64_t* poly = encrypted.data(i) + (j * coeff_count) + k;
          //*poly = (*poly) % coeff_modulus[j].value(); via
          // Barrett base 2^64 reduction
          unsigned long long carry;
          seal::util::multiply_uint64_hw64(*poly, const_ratio_1, &carry);
          carry = *poly - carry * modulus_value;
          *poly = carry - (modulus_value &
                           static_cast<uint64_t>(
                               -static_cast<int64_t>(carry >= modulus_value)));
        }
      }
    }
  }
  auto t1 = std::chrono::system_clock::now();
  NGRAPH_HE_LOG(3)
      << "lazy mod-reduce took "
      << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count()
      << "ms";
}
}  // namespace ngraph::runtime::he
