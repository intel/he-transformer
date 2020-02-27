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

#include "seal/kernel/add_seal.hpp"

#include <algorithm>
#include <utility>

#include "ngraph/coordinate_transform.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "ngraph/shape_util.hpp"
#include "seal/he_seal_backend.hpp"
#include "seal/seal_util.hpp"

namespace ngraph::runtime::he {

void scalar_add_seal(SealCiphertextWrapper& arg0, SealCiphertextWrapper& arg1,
                     std::shared_ptr<SealCiphertextWrapper>& out,
                     HESealBackend& he_seal_backend,
                     const seal::MemoryPoolHandle& pool) {
  match_modulus_and_scale_inplace(arg0, arg1, he_seal_backend, pool);

  if (!he_seal_backend.lazy_mod()) {
    he_seal_backend.get_evaluator()->add(arg0.ciphertext(), arg1.ciphertext(),
                                         out->ciphertext());
    return;
  }

  // Inline add
  // add_inplace(out->ciphertext(), arg0.ciphertext());
  seal::Ciphertext& encrypted1 = out->ciphertext();
  seal::Ciphertext& encrypted2 = arg0.ciphertext();

  // Extract encryption parameters.
  auto& context_data =
      *he_seal_backend.get_context()->get_context_data(encrypted1.parms_id());
  auto& parms = context_data.parms();
  auto& coeff_modulus = parms.coeff_modulus();
  size_t coeff_count = parms.poly_modulus_degree();
  size_t coeff_mod_count = coeff_modulus.size();
  size_t encrypted1_size = encrypted1.size();
  size_t encrypted2_size = encrypted2.size();
  size_t max_count = std::max(encrypted1_size, encrypted2_size);
  size_t min_count = std::min(encrypted1_size, encrypted2_size);

  // Prepare destination
  encrypted1.resize(he_seal_backend.get_context(), context_data.parms_id(),
                    max_count);

  // Add ciphertexts
  for (size_t j = 0; j < min_count; j++) {
    uint64_t* encrypted1_ptr = encrypted1.data(j);
    uint64_t* encrypted2_ptr = encrypted2.data(j);
    for (size_t i = 0; i < coeff_mod_count; i++) {
      std::uint64_t* operand1 = encrypted1_ptr + (i * coeff_count);
      std::uint64_t* operand2 = encrypted2_ptr + (i * coeff_count);
#pragma omp simd
      for (size_t k = 0; k < coeff_count; k++) {
        *operand1 = *operand1 + *operand2;
        /*
        std::uint64_t sum = (*operand1 + *operand2);
        if (sum < *operand1 || sum < *operand2) {
          NGRAPH_INFO << "Overflow in add idx=" << k << "\n"
                      << "op1 = " << *operand1 << "\nop2 = " << *operand2
                      << "\nsum = " << sum;
          throw std::runtime_error("Overflow in add");
        }
        *operand1 = sum; */
        operand1++;
        operand2++;
      }
    }
  }
}

void scalar_add_seal(SealCiphertextWrapper& arg0, const HEPlaintext& arg1,
                     std::shared_ptr<SealCiphertextWrapper>& out,
                     const bool complex_packing,
                     HESealBackend& he_seal_backend) {
  // TODO(fboemer): handle case where arg1 = {0, 0, 0, 0, ...}
  bool add_zero = (arg1.size() == 0) || (arg1.size() == 1 && arg1[0] == 0.0);
  if (add_zero) {
    SealCiphertextWrapper tmp(arg0);
    out = std::make_shared<SealCiphertextWrapper>(tmp);
    return;
  }

  // TODO(fboemer): optimize for adding single complex number
  if ((arg1.size() == 1) && !complex_packing) {
    add_plain(arg0.ciphertext(), arg1[0], out->ciphertext(), he_seal_backend);
    return;
  }

  auto p = SealPlaintextWrapper(complex_packing);
  encode(p, arg1, *he_seal_backend.get_ckks_encoder(),
         arg0.ciphertext().parms_id(), element::f32, arg0.ciphertext().scale(),
         complex_packing);

  size_t chain_ind0 = he_seal_backend.get_chain_index(arg0);
  size_t chain_ind1 = he_seal_backend.get_chain_index(p);
  NGRAPH_CHECK(chain_ind0 == chain_ind1, "Chain inds ", chain_ind0, ",  ",
               chain_ind1, " don't match");

  he_seal_backend.get_evaluator()->add_plain(arg0.ciphertext(), p.plaintext(),
                                             out->ciphertext());
}

void scalar_add_seal(const HEPlaintext& arg0, const HEPlaintext& arg1,
                     HEPlaintext& out) {
  HEPlaintext out_vals;
  if (arg0.size() == 1) {
    out_vals.resize(arg1.size());
    std::transform(arg1.begin(), arg1.end(), out_vals.begin(),
                   [&](auto x) { return x + arg0[0]; });
  } else if (arg1.size() == 1) {
    out_vals.resize(arg0.size());
    std::transform(arg0.begin(), arg0.end(), out_vals.begin(),
                   [&](auto x) { return x + arg1[0]; });
  } else {
    size_t min_size = std::min(arg0.size(), arg1.size());
    out_vals.resize(min_size);
    for (size_t i = 0; i < min_size; ++i) {
      out_vals[i] = arg0[i] + arg1[i];
    }
  }
  out = std::move(out_vals);
}

void scalar_add_seal(HEType& arg0, HEType& arg1, HEType& out,
                     HESealBackend& he_seal_backend) {
  NGRAPH_CHECK(arg0.complex_packing() == arg1.complex_packing(),
               "Complex packing types don't match");
  out.complex_packing() = arg0.complex_packing();

  if (arg0.is_ciphertext() && arg1.is_ciphertext()) {
    if (!out.is_ciphertext()) {
      out.set_ciphertext(HESealBackend::create_empty_ciphertext());
    }
    scalar_add_seal(*arg0.get_ciphertext(), *arg1.get_ciphertext(),
                    out.get_ciphertext(), he_seal_backend);
  } else if (arg0.is_ciphertext() && arg1.is_plaintext()) {
    if (!out.is_ciphertext()) {
      out.set_ciphertext(HESealBackend::create_empty_ciphertext());
    }
    scalar_add_seal(*arg0.get_ciphertext(), arg1.get_plaintext(),
                    out.get_ciphertext(), arg0.complex_packing(),
                    he_seal_backend);
  } else if (arg0.is_plaintext() && arg1.is_ciphertext()) {
    if (!out.is_ciphertext()) {
      out.set_ciphertext(HESealBackend::create_empty_ciphertext());
    }
    scalar_add_seal(*arg1.get_ciphertext(), arg0.get_plaintext(),
                    out.get_ciphertext(), arg0.complex_packing(),
                    he_seal_backend);
  } else if (arg0.is_plaintext() && arg1.is_plaintext()) {
    if (!out.is_plaintext()) {
      out.set_plaintext(HEPlaintext());
    }
    scalar_add_seal(arg0.get_plaintext(), arg1.get_plaintext(),
                    out.get_plaintext());
  }
}

void add_seal(std::vector<HEType>& arg0, std::vector<HEType>& arg1,
              std::vector<HEType>& out, size_t count, const Shape& arg0_shape,
              const Shape& arg1_shape, const element::Type& element_type,
              const op::AutoBroadcastSpec& broadcast_spec,
              HESealBackend& he_seal_backend) {
  /* NGRAPH_CHECK(he_seal_backend.is_supported_type(element_type),
                "Unsupported type ", element_type);
   NGRAPH_CHECK(count <= arg0.size(), "Count ", count,
                " is too large for arg0, with size ", arg0.size());
   NGRAPH_CHECK(count <= arg1.size(), "Count ", count,
                " is too large for arg1, with size ", arg1.size()); */

  NGRAPH_INFO << "arg0 size " << arg0.size();
  NGRAPH_INFO << "arg1 size " << arg1.size();
  NGRAPH_INFO << "out size " << out.size();

  switch (broadcast_spec.m_type) {
    case op::AutoBroadcastType::NONE: {
#pragma omp parallel for
      for (size_t i = 0; i < count; ++i) {
        scalar_add_seal(arg0[i], arg1[i], out[i], he_seal_backend);
      }
      break;
    }
    case op::AutoBroadcastType::NUMPY: {
      NGRAPH_INFO << "Add NUMPY";
      // We'll be using CoordinateTransform to handle the broadcasting. The
      // general procedure is as follows:
      //
      // (1) Left pad the shorter of the two shapes with ones.
      // (2) Squeeze (remove ones from) both shapes, and record the squeezed
      // axis
      //     indices.
      // (3) Using CoordinateTransform, broadcast both args to the final
      // output
      //     shape. The "broadcasted axes" will be those that were squeezed in
      //     step 2.
      //
      // Example:
      //
      //    Input shape->Padded shape->Squeezed Shape/Squeezed Axes
      //    -----------  ------------  ----------------------------
      // a: [ 3, 2, 1]   [ 3, 2, 1]    [ 3, 2   ]     {2}
      // b: [    1, 6]   [ 1, 1, 6]    [       6]     {0,1}
      //                   |  |  |
      //                   v  v  v
      //                 Output shape
      //                 ------------
      //                 [ 3, 2, 6]
      {
        Shape arg0_padded_shape = arg0_shape;
        Shape arg1_padded_shape = arg1_shape;

        while (arg0_padded_shape.size() < arg1_padded_shape.size()) {
          arg0_padded_shape.insert(arg0_padded_shape.begin(), 1);
        }

        while (arg1_padded_shape.size() < arg0_padded_shape.size()) {
          arg1_padded_shape.insert(arg1_padded_shape.begin(), 1);
        }

        Shape arg0_squeezed_shape;
        Shape arg1_squeezed_shape;
        AxisSet arg0_squeezed_axes;
        AxisSet arg1_squeezed_axes;
        Shape output_shape;

        for (size_t i = 0; i < arg0_padded_shape.size(); i++) {
          if (arg0_padded_shape[i] == 1) {
            arg0_squeezed_axes.insert(i);
          } else {
            arg0_squeezed_shape.push_back(arg0_padded_shape[i]);
          }

          if (arg1_padded_shape[i] == 1) {
            arg1_squeezed_axes.insert(i);
          } else {
            arg1_squeezed_shape.push_back(arg1_padded_shape[i]);
          }

          output_shape.push_back(arg0_padded_shape[i] == 1
                                     ? arg1_padded_shape[i]
                                     : arg0_padded_shape[i]);
        }

        CoordinateTransform arg0_transform(arg0_squeezed_shape);
        CoordinateTransform arg1_transform(arg1_squeezed_shape);
        CoordinateTransform output_transform(output_shape);

        for (const Coordinate& output_coord : output_transform) {
          Coordinate arg0_coord = reduce(output_coord, arg0_squeezed_axes);
          Coordinate arg1_coord = reduce(output_coord, arg1_squeezed_axes);
          scalar_add_seal(arg0[arg0_transform.index(arg0_coord)],
                          arg1[arg1_transform.index(arg1_coord)],
                          out[output_transform.index(output_coord)],
                          he_seal_backend);
          // elementwise_functor(arg0[arg0_transform.index(arg0_coord)],
          //                     arg1[arg1_transform.index(arg1_coord)]);
        }
      }
      break;
    }
    case op::AutoBroadcastType::PDPD:
    default: {
      NGRAPH_ERR << "Unsupported broadcast type";
      break;
    }
  }
}

}  // namespace ngraph::runtime::he
