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

  NGRAPH_HE_LOG(3) << "scalar_multiply_add_seal";
  scalar_multiply_seal(*arg0.get_ciphertext(), arg1.get_plaintext(), prod,
                       he_seal_backend, seal::MemoryManager::GetPool());

  NGRAPH_HE_LOG(3) << "scalar_add_seal";
  scalar_add_seal(*prod.get_ciphertext(), *out.get_ciphertext(),
                  out.get_ciphertext(), he_seal_backend);
}

}  // namespace ngraph::runtime::he
