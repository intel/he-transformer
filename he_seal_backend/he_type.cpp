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

#include "he_type.hpp"

#include <memory>
#include <utility>

#include "he_plaintext.hpp"
#include "ngraph/type/element_type.hpp"
#include "protos/message.pb.h"
#include "seal/he_seal_backend.hpp"
#include "seal/seal_ciphertext_wrapper.hpp"

namespace ngraph::runtime::he {

HEType::HEType(const HEPlaintext& plain, bool complex_packing)
    : HEType(complex_packing, plain.size()) {
  m_is_plain = true;
  m_plain = plain;
}

HEType::HEType(const std::shared_ptr<SealCiphertextWrapper>& cipher,
               bool complex_packing, size_t batch_size)
    : HEType(complex_packing, batch_size) {
  m_is_plain = false;
  m_cipher = cipher;
}

HEType HEType::load(const pb::HEType& pb_he_type,
                    std::shared_ptr<seal::SEALContext> context) {
  if (pb_he_type.is_plaintext()) {
    // TODO(fboemer): HEPlaintext::load function
    HEPlaintext vals{pb_he_type.plain().begin(), pb_he_type.plain().end()};

    return HEType(vals, pb_he_type.complex_packing());
  }

  auto cipher = HESealBackend::create_empty_ciphertext();
  SealCiphertextWrapper::load(*cipher, pb_he_type, std::move(context));
  return HEType(cipher, pb_he_type.complex_packing(), pb_he_type.batch_size());
}

void HEType::save(pb::HEType& pb_he_type) const {
  pb_he_type.set_is_plaintext(is_plaintext());
  pb_he_type.set_plaintext_packing(plaintext_packing());
  pb_he_type.set_complex_packing(complex_packing());
  pb_he_type.set_batch_size(batch_size());

  if (is_plaintext()) {
    // TODO(fboemer): more efficient
    for (auto& elem : get_plaintext()) {
      pb_he_type.add_plain(static_cast<float>(elem));
    }
  } else {
    get_ciphertext()->save(pb_he_type);
  }
}

void HEType::set_plaintext(HEPlaintext plain) {
  m_plain = std::move(plain);
  m_is_plain = true;
  if (m_cipher != nullptr) {
    m_cipher->ciphertext().release();
  }
  m_batch_size = plain.size();
}

}  // namespace ngraph::runtime::he
