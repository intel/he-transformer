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

#include "aby/aby_client_executor.hpp"

#include <chrono>
#include <vector>

#include "aby/kernel/relu_aby.hpp"
#include "nlohmann/json.hpp"
#include "seal/he_seal_backend.hpp"
#include "seal/seal_util.hpp"

using json = nlohmann::json;

namespace ngraph::runtime::aby {

ABYClientExecutor::ABYClientExecutor(
    std::string mpc_protocol, const he::HESealClient& he_seal_client,
    std::string hostname, std::size_t port, uint64_t security_level,
    uint32_t bit_length, uint32_t num_threads, uint32_t num_parties,
    std::string mg_algo_str, uint32_t reserve_num_gates)
    : ABYExecutor("client", mpc_protocol, hostname, port, security_level,
                  bit_length, num_threads, num_parties, mg_algo_str,
                  reserve_num_gates),
      m_he_seal_client(he_seal_client) {
  m_lowest_coeff_modulus = m_he_seal_client.encryption_paramters()
                               .seal_encryption_parameters()
                               .coeff_modulus()[0]
                               .value();

  NGRAPH_HE_LOG(1) << "Started ABYClientExecutor";
}

void ABYClientExecutor::run_aby_circuit(const std::string& function,
                                        std::shared_ptr<he::HETensor>& tensor) {
  NGRAPH_HE_LOG(3) << "client run_aby_circuit with function " << function;
  json js = json::parse(function);
  auto name = js.at("function");

  if (name == "Relu") {
    run_aby_relu_circuit(function, tensor);
  } else {
    NGRAPH_ERR << "Unknown function name " << name;
    throw ngraph_error("Unknown function name");
  }
}

void ABYClientExecutor::run_aby_relu_circuit(
    const std::string& function, std::shared_ptr<he::HETensor>& tensor) {
  NGRAPH_HE_LOG(3) << "run_aby_relu_circuit";
  auto name = json::parse(function).at("function");
  NGRAPH_CHECK(name == "Relu", "Function name ", name, " is not Relu");

  auto& tensor_data = tensor->data();
  size_t batch_size = tensor_data[0].batch_size();

  auto tensor_size = static_cast<uint64_t>(tensor_data.size() * batch_size);
  NGRAPH_HE_LOG(3) << "Batch size " << batch_size;
  NGRAPH_HE_LOG(3) << "tensor_data.size() " << tensor_data.size();
  NGRAPH_HE_LOG(3) << "tensor_size " << tensor_size;
  NGRAPH_HE_LOG(3) << "tensor_type " << tensor->get_element_type();

  std::vector<double> relu_vals(tensor_size);
  size_t num_bytes = tensor_size * tensor->get_element_type().size();

  if (tensor->get_element_type() == element::f32) {
    std::vector<float> relu_float_vals(tensor_size);
    tensor->read(relu_float_vals.data(), num_bytes);
    relu_vals =
        std::vector<double>{relu_float_vals.begin(), relu_float_vals.end()};

  } else if (tensor->get_element_type() == element::f64) {
    tensor->read(relu_vals.data(), num_bytes);
  } else {
    throw ngraph_error("Invalid element type");
  }

  NGRAPH_HE_LOG(3) << "Converting client values to ABY integers";

  std::vector<uint64_t> client_gc_vals(tensor_size);
  for (size_t i = 0; i < tensor_size; ++i) {
    // TOOD: check
    he::HEType& he_type = tensor_data[i % tensor_data.size()];
    NGRAPH_CHECK(he_type.is_ciphertext(), "HEType is not ciphertext");
    auto scale = he_type.get_ciphertext()->scale();

    // Reduce values to range (-q/(2*scale), q/(2*scale))
    auto relu_val =
        mod_reduce_zero_centered(relu_vals[i], m_lowest_coeff_modulus / scale);

    // Turn SEAL's mapping (-q/(2*scale), q/(2*scale)) to (0,q)
    uint64_t relu_int_val;
    if (relu_val <= 0) {
      relu_int_val = std::round(relu_val * scale + m_lowest_coeff_modulus);
    } else {
      relu_int_val = std::round(relu_val * scale);
    }
    client_gc_vals[i] = relu_int_val;
  }

  NGRAPH_HE_LOG(3) << "Client creating relu circuit";

  auto party_data_start_end_idx = split_vector(tensor_size, m_num_parties);
  double scale = m_he_seal_client.scale();

  std::vector<uint64_t> relu_result(tensor_size, 0);
  std::vector<double> relu_double_result(tensor_size, 0);
#pragma omp parallel for num_threads(m_num_parties)
  for (size_t party_idx = 0; party_idx < m_num_parties; ++party_idx) {
    const auto& [start_idx, end_idx] = party_data_start_end_idx[party_idx];
    size_t party_data_size = end_idx - start_idx;
    if (party_data_size == 0) {
      continue;
    }
    BooleanCircuit* circ = get_circuit(party_idx);

    std::vector<uint64_t> zeros(party_data_size, 0);

    // TODO(fboemer): Use span?
    std::vector<uint64_t> client_party_gc_vals(party_data_size);
    for (size_t idx = start_idx; idx < end_idx; ++idx) {
      client_party_gc_vals[idx - start_idx] = client_gc_vals[idx];
    }

    auto* relu_out =
        relu_aby(*circ, party_data_size, zeros, client_party_gc_vals, zeros,
                 m_aby_bitlen, m_lowest_coeff_modulus);

    NGRAPH_HE_LOG(3) << "Client party " << party_idx
                     << " executing relu circuit with start idx " << start_idx;

    auto t1 = std::chrono::high_resolution_clock::now();
    m_ABYParties[party_idx]->ExecCircuit();
    auto t2 = std::chrono::high_resolution_clock::now();
    NGRAPH_HE_LOG(3) << "Client executing circuit took "
                     << std::chrono::duration_cast<std::chrono::microseconds>(
                            t2 - t1)
                            .count()
                     << "us";
    uint32_t out_bitlen_relu;
    uint32_t result_count;
    uint64_t* out_vals_relu;  // output of circuit this value will be encrypted
                              // and sent to server
    relu_out->get_clear_value_vec(&out_vals_relu, &out_bitlen_relu,
                                  &result_count);

    NGRAPH_CHECK(result_count == party_data_size,
                 "Wrong number of ABY result values, result_count=",
                 result_count, ", expected ", party_data_size);
    for (size_t party_result_idx = 0; party_result_idx < party_data_size;
         ++party_result_idx) {
      relu_result[start_idx + party_result_idx] =
          out_vals_relu[party_result_idx];

      relu_double_result[start_idx + party_result_idx] = uint64_to_double(
          out_vals_relu[party_result_idx], m_lowest_coeff_modulus, scale);
    }
    reset_party(party_idx);
  }

  if (tensor->get_element_type() == element::f32) {
    std::vector<float> relu_float_result{relu_double_result.begin(),
                                         relu_double_result.end()};
    tensor->write(relu_float_result.data(), num_bytes);
  } else if (tensor->get_element_type() == element::f64) {
    tensor->write(relu_double_result.data(), num_bytes);
  } else {
    throw ngraph_error("Invalid element type");
  }
}

}  // namespace ngraph::runtime::aby
