//*****************************************************************************
// Copyright 2018-2019 Intel Corporation
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

#include <chrono>
#include <memory>
#include <thread>
#include <vector>

#include "he_op_annotations.hpp"
#include "ngraph/ngraph.hpp"
#include "ngraph/op/util/op_annotations.hpp"
#include "seal/he_seal_backend.hpp"
#include "seal/he_seal_client.hpp"
#include "seal/he_seal_executable.hpp"
#include "test_util.hpp"
#include "util/all_close.hpp"
#include "util/ndarray.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

static std::string s_manifest = "${MANIFEST}";

static std::string gc_param_real_str = R"(
    {
        "scheme_name" : "HE_SEAL",
        "poly_modulus_degree" : 2048,
        "security_level" : 0,
        "coeff_modulus" : [44, 24, 24, 24, 30],
        "scale" : 16777216,
        "complex_packing": false
    })";

static std::string gc_param_complex_str = R"(
    {
        "scheme_name" : "HE_SEAL",
        "poly_modulus_degree" : 2048,
        "security_level" : 0,
        "coeff_modulus" : [44, 24, 24, 24, 30],
        "scale" : 16777216,
        "complex_packing": true
    })";

namespace ngraph::runtime::he {

NGRAPH_TEST(${BACKEND_NAME}, server_client_gc_add_3_multiple_parameters_plain) {
  auto backend = Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<HESealBackend*>(backend.get());

  size_t batch_size = 1;

  Shape shape{batch_size, 3};
  Shape shape_c{3, 1};
  auto a = op::Constant::create(element::f32, shape, {0.1, 0.2, 0.3});
  auto b = std::make_shared<op::Parameter>(element::f32, shape);

  auto c = std::make_shared<op::Parameter>(element::f32, shape_c);
  auto d = std::make_shared<op::Reshape>(c, AxisVector{0, 1}, shape);
  auto t = std::make_shared<op::Add>(a, b);
  t = std::make_shared<op::Add>(t, d);
  auto f = std::make_shared<Function>(t, ParameterVector{b, c});

  std::string error_str;
  he_backend->set_config(
      std::map<std::string, std::string>{
          {"enable_client", "true"},
          {"enable_gc", "true"},
          {"encryption_parameters", gc_param_real_str},
          {b->get_name(), "client_input"}},
      error_str);

  auto t_c = he_backend->create_plain_tensor(element::f32, shape_c);
  auto t_result = he_backend->create_cipher_tensor(element::f32, shape);
  // Server inputs which are not used
  auto t_dummy = he_backend->create_plain_tensor(element::f32, shape);

  // Used for dummy server inputs
  float DUMMY_FLOAT = 99;
  copy_data(t_dummy, std::vector<float>{DUMMY_FLOAT, DUMMY_FLOAT, DUMMY_FLOAT});
  copy_data(t_c, std::vector<float>{4, 5, 6});

  std::vector<float> results;
  auto client_thread = std::thread([&]() {
    std::vector<float> inputs{1, 2, 3};
    auto he_client = HESealClient(
        "localhost", 34000, batch_size,
        HETensorConfigMap<float>{{b->get_name(), make_pair("plain", inputs)}});

    auto double_results = he_client.get_results();
    results = std::vector<float>(double_results.begin(), double_results.end());
  });

  auto handle =
      std::static_pointer_cast<HESealExecutable>(he_backend->compile(f));

  handle->call_with_validate({t_result}, {t_dummy, t_c});
  client_thread.join();
  EXPECT_TRUE(
      test::all_close(results, std::vector<float>{5.1, 7.2, 9.3}, 1e-3f));
}

NGRAPH_TEST(${BACKEND_NAME}, server_client_gc_add_3_relu) {
  auto backend = Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<HESealBackend*>(backend.get());

  size_t batch_size = 1;

  Shape shape{batch_size, 3};
  auto a = op::Constant::create(element::f32, shape, {0.1, 0.2, 0.3});
  auto b = std::make_shared<op::Parameter>(element::f32, shape);
  auto t = std::make_shared<op::Add>(a, b);
  auto relu = std::make_shared<op::Relu>(t);
  auto f = std::make_shared<Function>(relu, ParameterVector{b});

  std::string error_str;
  he_backend->set_config(
      std::map<std::string, std::string>{
          {"enable_client", "true"},
          {"enable_gc", "true"},
          {"encryption_parameters", gc_param_real_str},
          {"mask_gc_inputs", "true"},
          {"mask_gc_outputs", "true"},
          {b->get_name(), "client_input,encrypt"}},
      error_str);

  // Server inputs which are not used
  auto t_dummy = he_backend->create_plain_tensor(element::f32, shape);
  auto t_result = he_backend->create_cipher_tensor(element::f32, shape);

  // Used for dummy server inputs
  float DUMMY_FLOAT = 99;
  copy_data(t_dummy, std::vector<float>{DUMMY_FLOAT, DUMMY_FLOAT, DUMMY_FLOAT});

  std::vector<float> results;
  auto client_thread = std::thread([&]() {
    std::vector<float> inputs{-1, -0.2, 3};
    auto he_client =
        HESealClient("localhost", 34000, batch_size,
                     HETensorConfigMap<float>{
                         {b->get_name(), make_pair("encrypt", inputs)}});

    auto double_results = he_client.get_results();
    results = std::vector<float>(double_results.begin(), double_results.end());
  });

  auto handle =
      std::static_pointer_cast<HESealExecutable>(he_backend->compile(f));

  handle->call_with_validate({t_result}, {t_dummy});

  client_thread.join();
  EXPECT_TRUE(test::all_close(results, std::vector<float>{0, 0, 3.3}, 1e-1f));
}

auto server_client_gc_relu_packed_test = [](size_t element_count,
                                            size_t batch_size,
                                            bool complex_packing,
                                            bool mask_gc_inputs,
                                            bool mask_gc_outputs) {
  auto backend = Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<HESealBackend*>(backend.get());

  Shape shape{batch_size, element_count};
  auto a = std::make_shared<op::Parameter>(element::f32, shape);
  auto relu_op = std::make_shared<op::Relu>(a);
  auto f = std::make_shared<Function>(relu_op, ParameterVector{a});

  bool packed = batch_size > 1;
  std::string tensor_config{"client_input,encrypt"};
  if (packed) {
    tensor_config.append(",packed");
  }
  std::map<std::string, std::string> config_map{
      {"enable_client", "true"},
      {"enable_gc", "true"},
      {a->get_name(), tensor_config},
      {"mask_gc_inputs", bool_to_string(mask_gc_inputs)},
      {"mask_gc_outputs", bool_to_string(mask_gc_outputs)}};

  if (complex_packing) {
    config_map["encryption_parameters"] = gc_param_real_str;
  } else {
    config_map["encryption_parameters"] = gc_param_complex_str;
  }

  std::string error_str;
  he_backend->set_config(config_map, error_str);

  NGRAPH_INFO << "complex_packing " << complex_packing;
  NGRAPH_INFO << "mask_gc_inputs " << mask_gc_inputs;
  NGRAPH_INFO << "mask_gc_outputs " << mask_gc_outputs;

  auto relu = [](double d) { return d > 0 ? d : 0.; };

  // Server inputs which are not used
  auto t_dummy = he_backend->create_packed_plain_tensor(element::f32, shape);
  auto t_result = he_backend->create_packed_cipher_tensor(element::f32, shape);

  // Used for dummy server inputs
  float DUMMY_FLOAT = 99;
  copy_data(t_dummy, std::vector<float>(shape_size(shape), DUMMY_FLOAT));

  std::vector<float> inputs(shape_size(shape));
  std::vector<float> exp_results(shape_size(shape));
  for (size_t i = 0; i < shape_size(shape); ++i) {
    inputs[i] = static_cast<int>(i) - static_cast<int>(shape_size(shape)) / 2;

    // Choose 30 instead of 32 to avoid rounding issues
    if (std::abs(inputs[i]) > 30) {
      inputs[i] = fmod(inputs[i], 32);
    }

    exp_results[i] = relu(inputs[i]);
  }

  std::vector<float> results;
  auto client_thread = std::thread([&]() {
    auto he_client =
        HESealClient("localhost", 34000, batch_size,
                     HETensorConfigMap<float>{
                         {a->get_name(), make_pair("encrypt", inputs)}});

    auto double_results = he_client.get_results();
    results = std::vector<float>(double_results.begin(), double_results.end());
  });

  auto handle =
      std::static_pointer_cast<HESealExecutable>(he_backend->compile(f));

  handle->call_with_validate({t_result}, {t_dummy});

  client_thread.join();
  EXPECT_TRUE(test::all_close(results, exp_results, 1e-2f));
};

NGRAPH_TEST(${BACKEND_NAME}, server_client_gc_10_1_relu_real) {
  server_client_gc_relu_packed_test(10, 1, false, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, server_client_gc_10_1_relu_real_mask_out) {
  server_client_gc_relu_packed_test(10, 1, false, false, true);
}

NGRAPH_TEST(${BACKEND_NAME}, server_client_gc_10_1_relu_real_mask_in) {
  server_client_gc_relu_packed_test(10, 1, false, true, false);
}

NGRAPH_TEST(${BACKEND_NAME}, server_client_gc_10_1_relu_real_mask_in_out) {
  server_client_gc_relu_packed_test(10, 1, false, true, true);
}

NGRAPH_TEST(${BACKEND_NAME}, server_client_gc_10_2_relu_real) {
  server_client_gc_relu_packed_test(10, 2, false, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, server_client_gc_10_2_relu_real_mask_out) {
  server_client_gc_relu_packed_test(10, 2, false, false, true);
}

NGRAPH_TEST(${BACKEND_NAME}, server_client_gc_10_2_relu_real_mask_in) {
  server_client_gc_relu_packed_test(10, 2, false, true, false);
}

NGRAPH_TEST(${BACKEND_NAME}, server_client_gc_10_2_relu_real_mask_in_out) {
  server_client_gc_relu_packed_test(10, 2, false, true, true);
}

NGRAPH_TEST(${BACKEND_NAME}, server_client_gc_10_1_relu_complex) {
  server_client_gc_relu_packed_test(10, 1, true, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, server_client_gc_10_1_relu_complex_mask_out) {
  server_client_gc_relu_packed_test(10, 1, true, false, true);
}

NGRAPH_TEST(${BACKEND_NAME}, server_client_gc_10_1_relu_complex_mask_in) {
  server_client_gc_relu_packed_test(10, 1, true, true, false);
}

NGRAPH_TEST(${BACKEND_NAME}, server_client_gc_10_1_relu_complex_mask_in_out) {
  server_client_gc_relu_packed_test(10, 1, true, true, true);
}

NGRAPH_TEST(${BACKEND_NAME}, server_client_gc_10_2_relu_complex) {
  server_client_gc_relu_packed_test(10, 2, true, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, server_client_gc_10_2_relu_complex_mask_out) {
  server_client_gc_relu_packed_test(10, 2, true, false, true);
}

NGRAPH_TEST(${BACKEND_NAME}, server_client_gc_10_2_relu_complex_mask_in) {
  server_client_gc_relu_packed_test(10, 2, true, true, false);
}

NGRAPH_TEST(${BACKEND_NAME}, server_client_gc_10_2_relu_complex_mask_in_out) {
  server_client_gc_relu_packed_test(10, 2, true, true, true);
}

NGRAPH_TEST(${BACKEND_NAME}, server_client_gc_100_1_relu_real) {
  server_client_gc_relu_packed_test(100, 1, false, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, server_client_gc_100_1_relu_real_mask_out) {
  server_client_gc_relu_packed_test(100, 1, false, false, true);
}

NGRAPH_TEST(${BACKEND_NAME}, server_client_gc_100_1_relu_real_mask_in) {
  server_client_gc_relu_packed_test(100, 1, false, true, false);
}

NGRAPH_TEST(${BACKEND_NAME}, server_client_gc_100_1_relu_real_mask_in_out) {
  server_client_gc_relu_packed_test(100, 1, false, true, true);
}

NGRAPH_TEST(${BACKEND_NAME}, server_client_gc_300000_1_relu_real_mask_in_out) {
  server_client_gc_relu_packed_test(300000, 1, false, true, true);
}

NGRAPH_TEST(${BACKEND_NAME}, server_client_gc_845_100_relu_real_mask_in_out) {
  server_client_gc_relu_packed_test(845, 100, false, true, true);
}

NGRAPH_TEST(${BACKEND_NAME}, server_client_gc_845_400_relu_real_mask_in_out) {
  server_client_gc_relu_packed_test(845, 400, false, true, true);
}

NGRAPH_TEST(${BACKEND_NAME}, server_client_gc_100_2_relu_real) {
  server_client_gc_relu_packed_test(100, 2, false, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, server_client_gc_100_2_relu_real_mask_out) {
  server_client_gc_relu_packed_test(100, 2, false, false, true);
}

NGRAPH_TEST(${BACKEND_NAME}, server_client_gc_100_2_relu_real_mask_in) {
  server_client_gc_relu_packed_test(100, 2, false, true, false);
}

NGRAPH_TEST(${BACKEND_NAME}, server_client_gc_100_2_relu_real_mask_in_out) {
  server_client_gc_relu_packed_test(100, 2, false, true, true);
}

NGRAPH_TEST(${BACKEND_NAME}, server_client_gc_100_1_relu_complex) {
  server_client_gc_relu_packed_test(100, 1, true, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, server_client_gc_100_1_relu_complex_mask_out) {
  server_client_gc_relu_packed_test(100, 1, true, false, true);
}

NGRAPH_TEST(${BACKEND_NAME}, server_client_gc_100_1_relu_complex_mask_in) {
  server_client_gc_relu_packed_test(100, 1, true, true, false);
}

NGRAPH_TEST(${BACKEND_NAME}, server_client_gc_100_1_relu_complex_mask_in_out) {
  server_client_gc_relu_packed_test(100, 1, true, true, true);
}

NGRAPH_TEST(${BACKEND_NAME}, server_client_gc_100_2_relu_complex) {
  server_client_gc_relu_packed_test(100, 2, true, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, server_client_gc_100_2_relu_complex_mask_out) {
  server_client_gc_relu_packed_test(100, 2, true, false, true);
}

NGRAPH_TEST(${BACKEND_NAME}, server_client_gc_100_2_relu_complex_mask_in) {
  server_client_gc_relu_packed_test(100, 2, true, true, false);
}

NGRAPH_TEST(${BACKEND_NAME}, server_client_gc_100_2_relu_complex_mask_in_out) {
  server_client_gc_relu_packed_test(100, 2, true, true, true);
}

auto server_client_gc_maxpool_test = [](const Shape& shape_a,
                                        const Shape& window_shape,
                                        const std::vector<float>& input_a,
                                        const std::vector<float>& output,
                                        const bool complex_packing,
                                        const bool packed, bool mask_gc_inputs,
                                        bool mask_gc_outputs) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<HESealBackend*>(backend.get());

  auto a = std::make_shared<op::Parameter>(element::f32, shape_a);
  auto t = std::make_shared<op::MaxPool>(a, window_shape);
  auto f = std::make_shared<Function>(t, ParameterVector{a});

  std::string tensor_config{"client_input,encrypt"};
  if (packed) {
    tensor_config.append(",packed");
  }
  std::map<std::string, std::string> config_map{
      {"enable_client", "true"},
      {"enable_gc", "true"},
      {a->get_name(), tensor_config},
      {"mask_gc_inputs", bool_to_string(mask_gc_inputs)},
      {"mask_gc_outputs", bool_to_string(mask_gc_outputs)}};

  if (complex_packing) {
    config_map["encryption_parameters"] = gc_param_real_str;
  } else {
    config_map["encryption_parameters"] = gc_param_complex_str;
  }

  std::string error_str;
  he_backend->set_config(config_map, error_str);

  NGRAPH_INFO << "complex_packing " << complex_packing;
  NGRAPH_INFO << "mask_gc_inputs " << mask_gc_inputs;
  NGRAPH_INFO << "mask_gc_outputs " << mask_gc_outputs;

  // Server inputs which are not used
  auto t_dummy = he_backend->create_packed_plain_tensor(element::f32, shape_a);
  auto t_result =
      he_backend->create_packed_cipher_tensor(element::f32, t->get_shape());

  // Used for dummy server inputs
  float DUMMY_FLOAT = 99;
  copy_data(t_dummy, std::vector<float>(shape_size(shape_a), DUMMY_FLOAT));

  size_t batch_size = packed ? shape_a[0] : 1;

  std::vector<float> results;
  auto client_thread = std::thread([&]() {
    auto he_client =
        HESealClient("localhost", 34000, batch_size,
                     HETensorConfigMap<float>{
                         {a->get_name(), make_pair("encrypt", input_a)}});

    auto double_results = he_client.get_results();
    results = std::vector<float>(double_results.begin(), double_results.end());
  });

  auto handle =
      std::static_pointer_cast<HESealExecutable>(he_backend->compile(f));

  handle->call_with_validate({t_result}, {t_dummy});

  client_thread.join();
  EXPECT_TRUE(test::all_close(results, output, 1e-2f));
};

NGRAPH_TEST(${BACKEND_NAME},
            server_client_gc_max_pool_1d_1channel_1image_plain_real_unpacked) {
  server_client_gc_maxpool_test(
      Shape{1, 1, 14}, Shape{3},
      std::vector<float>{0, 1, 0, 2, 1, 0, 3, 2, 0, 0, 2, 0, 0, 0},
      std::vector<float>{1, 2, 2, 2, 3, 3, 3, 2, 2, 2, 2, 0}, false, false,
      false, false);
}

}  // namespace ngraph::runtime::he