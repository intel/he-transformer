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

#include "aby/aby_server_executor.hpp"

#include <algorithm>

#include "aby/kernel/relu_aby.hpp"
#include "nlohmann/json.hpp"
#include "seal/kernel/subtract_seal.hpp"
#include "seal/seal_util.hpp"

using json = nlohmann::json;

namespace ngraph::runtime::aby {

ABYServerExecutor::ABYServerExecutor(
    he::HESealExecutable& he_seal_executable, const std::string& mpc_protocol,
    const std::string hostname, std::size_t port, uint64_t security_level,
    uint32_t bit_length, uint32_t num_threads, uint32_t num_parties,
    std::string mg_algo_str, uint32_t reserve_num_gates)
    : ABYExecutor("server", mpc_protocol, hostname, port, security_level,
                  bit_length, num_threads, num_parties, mg_algo_str,
                  reserve_num_gates),
      m_he_seal_executable{he_seal_executable} {
  m_lowest_coeff_modulus = m_he_seal_executable.he_seal_backend()
                               .get_encryption_parameters()
                               .seal_encryption_parameters()
                               .coeff_modulus()[0]
                               .value();
  m_rand_max = static_cast<int64_t>(m_lowest_coeff_modulus - 1);
  m_random_distribution = std::uniform_int_distribution<int64_t>{0, m_rand_max};
}

void ABYServerExecutor::prepare_aby_circuit(
    const std::string& function, std::shared_ptr<he::HETensor>& tensor) {
  NGRAPH_HE_LOG(4) << "server prepare_aby_circuit with function " << function;
  json js = json::parse(function);
  auto name = js.at("function");

  if (name == "Relu") {
    prepare_aby_relu_circuit(tensor->data());
  } else {
    NGRAPH_ERR << "Unknown function name " << name;
    throw ngraph_error("Unknown function name");
  }
}

void ABYServerExecutor::run_aby_circuit(const std::string& function,
                                        std::shared_ptr<he::HETensor>& tensor) {
  NGRAPH_HE_LOG(4) << "server run_aby_circuit with funciton " << function;

  json js = json::parse(function);
  auto name = js.at("function");
  if (name == "Relu") {
    run_aby_relu_circuit(tensor->data());
  } else {
    NGRAPH_ERR << "Unknown function name " << name;
    throw ngraph_error("Unknown function name");
  }
}

void ABYServerExecutor::post_process_aby_circuit(
    const std::string& function, std::shared_ptr<he::HETensor>& tensor) {
  NGRAPH_HE_LOG(4) << "server post_process_aby_circuit with funciton "
                   << function;

  json js = json::parse(function);
  auto name = js.at("function");
  if (name == "Relu") {
    post_process_aby_relu_circuit(tensor);
  } else {
    NGRAPH_ERR << "Unknown function name " << name;
    throw ngraph_error("Unknown function name");
  }
}

std::shared_ptr<he::HETensor> ABYServerExecutor::generate_gc_mask(
    const ngraph::Shape& shape, bool plaintext_packing, bool complex_packing,
    const std::string& name, bool random, uint64_t default_value) {
  auto tensor = std::make_shared<he::HETensor>(
      element::i64, shape, plaintext_packing, complex_packing, false,
      m_he_seal_executable.he_seal_backend(), name);

  std::vector<uint64_t> rand_vals(tensor->get_element_count());
  if (random) {
    auto random_gen = [this]() {
      return m_random_distribution(m_random_generator);
    };
    std::generate(rand_vals.begin(), rand_vals.end(), random_gen);
  } else {
    rand_vals = std::vector<uint64_t>(rand_vals.size(), default_value);
  }
  NGRAPH_HE_LOG(4) << "Random mask vals:";
  for (size_t i = 0; i < std::min(rand_vals.size(), 10UL); ++i) {
    NGRAPH_HE_LOG(4) << rand_vals[i];
  }

  tensor->write(rand_vals.data(), rand_vals.size() * sizeof(uint64_t));

  return tensor;
}

std::shared_ptr<he::HETensor> ABYServerExecutor::generate_gc_input_mask(
    const ngraph::Shape& shape, bool plaintext_packing, bool complex_packing,
    uint64_t default_value) {
  return generate_gc_mask(
      shape, plaintext_packing, complex_packing, "gc_input_mask",
      m_he_seal_executable.he_seal_backend().mask_gc_inputs(), default_value);
}

std::shared_ptr<he::HETensor> ABYServerExecutor::generate_gc_output_mask(
    const ngraph::Shape& shape, bool plaintext_packing, bool complex_packing,
    uint64_t default_value) {
  return generate_gc_mask(
      shape, plaintext_packing, complex_packing, "gc_output_mask",
      m_he_seal_executable.he_seal_backend().mask_gc_outputs(), default_value);
}

void ABYServerExecutor::prepare_aby_relu_circuit(
    std::vector<he::HEType>& cipher_batch) {
  NGRAPH_HE_LOG(4) << "prepare_aby_relu_circuit ";

  bool plaintext_packing = cipher_batch[0].plaintext_packing();
  bool complex_packing = cipher_batch[0].complex_packing();
  size_t batch_size = cipher_batch[0].batch_size();

  NGRAPH_HE_LOG(4) << "Generating gc input mask";
  NGRAPH_HE_LOG(4) << "complex_packing? " << complex_packing;

  m_gc_input_mask =
      generate_gc_input_mask(Shape{batch_size, cipher_batch.size()},
                             plaintext_packing, complex_packing);

  NGRAPH_HE_LOG(4) << "Generating gc output mask";

  m_gc_output_mask = generate_gc_output_mask(
      Shape{batch_size, cipher_batch.size()}, plaintext_packing,
      complex_packing, m_lowest_coeff_modulus / 2);

  std::vector<double> scales(cipher_batch.size());

  for (size_t i = 0; i < cipher_batch.size(); ++i) {
    auto& he_type = cipher_batch[i];
    auto& gc_input_mask = m_gc_input_mask->data(i);
    NGRAPH_CHECK(he_type.is_ciphertext(), "HEType is not ciphertext");

    auto cipher = he_type.get_ciphertext();

    // Switch modulus to lowest values since mask values are drawn
    // from (-q/2, q/2) for q the lowest coeff modulus
    m_he_seal_executable.he_seal_backend().mod_switch_to_lowest(*cipher);

    // Divide by scale so we can encode at the same scale as existing
    // ciphertext
    double scale = cipher->ciphertext().scale();
    scales[i] = scale;
    he::HEPlaintext scaled_gc_input_mask(gc_input_mask.get_plaintext());
    for (size_t mask_idx = 0; mask_idx < scaled_gc_input_mask.size();
         ++mask_idx) {
      scaled_gc_input_mask[mask_idx] /= scale;
    }
    NGRAPH_HE_LOG(4) << "scaled_gc_input_mask " << scaled_gc_input_mask;

    scalar_subtract_seal(*cipher, scaled_gc_input_mask, cipher,
                         he_type.complex_packing(),
                         m_he_seal_executable.he_seal_backend());
  }

  for (const auto& scale : scales) {
    NGRAPH_CHECK(std::abs(scale - scales[0]) < 1e-3f, "Scale ", scale,
                 " does not match first scale ", scales[0]);
  }
}

void ABYServerExecutor::run_aby_relu_circuit(
    std::vector<he::HEType>& cipher_batch) {
  NGRAPH_HE_LOG(4) << "run_aby_relu_circuit ";

  uint32_t num_aby_vals = cipher_batch.size() * cipher_batch[0].batch_size();

  auto party_data_start_end_idx = split_vector(num_aby_vals, m_num_parties);

  std::vector<uint64_t> gc_input_mask_vals(num_aby_vals);
  std::vector<uint64_t> gc_output_mask_vals(num_aby_vals);

  m_gc_input_mask->read(gc_input_mask_vals.data(),
                        num_aby_vals * sizeof(uint64_t));
  m_gc_output_mask->read(gc_output_mask_vals.data(),
                         num_aby_vals * sizeof(uint64_t));

#pragma omp parallel for num_threads(m_num_parties)
  for (size_t party_idx = 0; party_idx < m_num_parties; ++party_idx) {
    const auto& [start_idx, end_idx] = party_data_start_end_idx[party_idx];
    size_t party_data_size = end_idx - start_idx;
    if (party_data_size == 0) {
      continue;
    }

    NGRAPH_HE_LOG(3) << "Server creating relu circuit for party " << party_idx;
    BooleanCircuit* circ = get_circuit(party_idx);
    NGRAPH_HE_LOG(3) << "num_aby_vals " << num_aby_vals;
    NGRAPH_HE_LOG(3) << "gc_input_mask_vals " << gc_input_mask_vals.size();
    NGRAPH_HE_LOG(3) << "gc_output_mask_vals " << gc_output_mask_vals.size();

    std::vector<uint64_t> gc_input_party_mask_vals(party_data_size);
    std::vector<uint64_t> gc_output_party_mask_vals(party_data_size);

    for (size_t idx = start_idx; idx < end_idx; ++idx) {
      gc_input_party_mask_vals[idx - start_idx] = gc_input_mask_vals[idx];
      gc_output_party_mask_vals[idx - start_idx] = gc_output_mask_vals[idx];
    }

    std::vector<uint64_t> zeros(party_data_size, 0);

    ngraph::runtime::aby::relu_aby(
        *circ, party_data_size, gc_input_party_mask_vals, zeros,
        gc_output_party_mask_vals, m_aby_bitlen, m_lowest_coeff_modulus);

    NGRAPH_HE_LOG(3) << "server executing relu circuit";
    m_ABYParties[party_idx]->ExecCircuit();
    NGRAPH_HE_LOG(3) << "server done executing relu circuit";

    reset_party(party_idx);
    NGRAPH_HE_LOG(3) << "server done reset party " << party_idx;
  }
}

void ABYServerExecutor::post_process_aby_relu_circuit(
    std::shared_ptr<he::HETensor>& tensor) {
  if (m_he_seal_executable.he_seal_backend().mask_gc_outputs()) {
    NGRAPH_HE_LOG(4) << "Post-processing relu outptus";

    size_t tensor_size = tensor->data().size();
    double scale = m_he_seal_executable.he_seal_backend().get_scale();

    for (size_t tensor_idx = 0; tensor_idx < tensor_size; ++tensor_idx) {
      auto& data = tensor->data(tensor_idx);
      NGRAPH_CHECK(data.is_ciphertext(), "Data is not ciphertext");

      auto cipher = data.get_ciphertext();

      auto mask = m_gc_output_mask->data(tensor_idx).get_plaintext();
      for (auto& value : mask) {
        value = (value - m_lowest_coeff_modulus / 2.0) / scale;
      }

      // TODO(fboemer): do subtraction mod p_0 instead of p_L?
      // m_he_seal_executable.he_seal_backend().mod_switch_to_lowest(*cipher);

      scalar_subtract_seal(*cipher, mask, cipher, data.complex_packing(),
                           m_he_seal_executable.he_seal_backend());
    }
  }
}

}  // namespace ngraph::runtime::aby
