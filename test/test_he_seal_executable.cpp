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

#include <sstream>
#include <unordered_set>

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "seal/he_seal_executable.hpp"
#include "seal/seal.h"
#include "test_util.hpp"
#include "util/test_tools.hpp"

TEST(he_seal_executable, plaintext_with_encrypted_annotation) {
  auto backend = ngraph::runtime::Backend::create("HE_SEAL");
  auto he_backend = static_cast<ngraph::he::HESealBackend*>(backend.get());

  ngraph::Shape shape{2, 2};

  bool packed = true;
  bool arg1_encrypted = true;
  bool arg2_encrypted = false;

  auto a = std::make_shared<ngraph::op::Parameter>(ngraph::element::f32, shape);
  auto b = std::make_shared<ngraph::op::Parameter>(ngraph::element::f32, shape);
  auto t = std::make_shared<ngraph::op::Add>(a, b);
  auto f = std::make_shared<ngraph::Function>(t, ngraph::ParameterVector{a, b});

  a->set_op_annotations(
      ngraph::test::he::annotation_from_flags(false, arg1_encrypted, packed));
  b->set_op_annotations(
      ngraph::test::he::annotation_from_flags(false, arg2_encrypted, packed));

  // Create plaintext tensor for ciphertext argument
  // This behavior occurs when using ngraph-bridge
  auto t_a =
      ngraph::test::he::tensor_from_flags(*he_backend, shape, false, packed);
  auto t_b = ngraph::test::he::tensor_from_flags(*he_backend, shape,
                                                 arg2_encrypted, packed);
  auto t_result = ngraph::test::he::tensor_from_flags(
      *he_backend, shape, arg1_encrypted || arg2_encrypted, packed);

  std::vector<float> input_a{1, 2, 3, 4};
  std::vector<float> input_b{0, -1, 2, -3};
  std::vector<float> exp_result{1, 1, 5, 1};

  copy_data(t_a, input_a);
  copy_data(t_b, input_b);

  auto he_handle = std::static_pointer_cast<ngraph::he::HESealExecutable>(
      he_backend->compile(f));

  he_handle->call_with_validate({t_result}, {t_a, t_b});
  EXPECT_TRUE(ngraph::test::he::all_close(read_vector<float>(t_result),
                                          exp_result, 1e-3f));
}

TEST(he_seal_executable, performance_data) {
  auto backend = ngraph::runtime::Backend::create("HE_SEAL");
  auto he_backend = static_cast<ngraph::he::HESealBackend*>(backend.get());

  ngraph::Shape shape{2, 2};

  bool packed = true;
  bool arg1_encrypted = false;
  bool arg2_encrypted = false;

  auto a = std::make_shared<ngraph::op::Parameter>(ngraph::element::f32, shape);
  auto b = std::make_shared<ngraph::op::Parameter>(ngraph::element::f32, shape);
  auto t = std::make_shared<ngraph::op::Add>(a, b);
  auto f = std::make_shared<ngraph::Function>(t, ngraph::ParameterVector{a, b});

  a->set_op_annotations(
      ngraph::test::he::annotation_from_flags(false, arg1_encrypted, packed));
  b->set_op_annotations(
      ngraph::test::he::annotation_from_flags(false, arg2_encrypted, packed));

  auto t_a = ngraph::test::he::tensor_from_flags(*he_backend, shape,
                                                 arg1_encrypted, packed);
  auto t_b = ngraph::test::he::tensor_from_flags(*he_backend, shape,
                                                 arg2_encrypted, packed);
  auto t_result = ngraph::test::he::tensor_from_flags(
      *he_backend, shape, arg1_encrypted || arg2_encrypted, packed);

  std::vector<float> input_a{1, 2, 3, 4};
  std::vector<float> input_b{0, -1, 2, -3};
  std::vector<float> exp_result{1, 1, 5, 1};
  copy_data(t_a, input_a);
  copy_data(t_b, input_b);

  auto he_handle = std::static_pointer_cast<ngraph::he::HESealExecutable>(
      he_backend->compile(f));

  he_handle->set_verbose_all_ops(false);

  he_handle->call_with_validate({t_result}, {t_a, t_b});
  EXPECT_TRUE(ngraph::test::he::all_close(read_vector<float>(t_result),
                                          exp_result, 1e-3f));

  std::unordered_set<std::string> node_names;
  for (const auto& node : f->get_ops()) {
    node_names.insert(node->get_name());
  }

  const auto& perf_data = he_handle->get_performance_data();
  for (const auto& perf_counter : perf_data) {
    EXPECT_EQ(perf_counter.call_count(), 1);
    const std::string& node_name = perf_counter.get_node()->get_name();
    ASSERT_TRUE(node_names.find(node_name) != node_names.end());
    node_names.erase(node_name);
    NGRAPH_INFO << perf_counter.get_node()->get_name() << ": call count "
                << perf_counter.call_count() << ", microseconds "
                << perf_counter.microseconds();
  }
}

TEST(he_seal_executable, verbose_op) {
  auto backend = ngraph::runtime::Backend::create("HE_SEAL");
  auto he_backend = static_cast<ngraph::he::HESealBackend*>(backend.get());

  ngraph::Shape shape{2, 2};

  bool packed = true;
  bool arg1_encrypted = false;
  bool arg2_encrypted = false;

  auto a = std::make_shared<ngraph::op::Parameter>(ngraph::element::f32, shape);
  auto b = std::make_shared<ngraph::op::Parameter>(ngraph::element::f32, shape);
  auto t = std::make_shared<ngraph::op::Add>(a, b);
  auto f = std::make_shared<ngraph::Function>(t, ngraph::ParameterVector{a, b});

  a->set_op_annotations(
      ngraph::test::he::annotation_from_flags(false, arg1_encrypted, packed));
  b->set_op_annotations(
      ngraph::test::he::annotation_from_flags(false, arg2_encrypted, packed));

  auto t_a = ngraph::test::he::tensor_from_flags(*he_backend, shape,
                                                 arg1_encrypted, packed);
  auto t_b = ngraph::test::he::tensor_from_flags(*he_backend, shape,
                                                 arg2_encrypted, packed);
  auto t_result = ngraph::test::he::tensor_from_flags(
      *he_backend, shape, arg1_encrypted || arg2_encrypted, packed);

  std::vector<float> input_a{1, 2, 3, 4};
  std::vector<float> input_b{0, -1, 2, -3};
  std::vector<float> exp_result{1, 1, 5, 1};
  copy_data(t_a, input_a);
  copy_data(t_b, input_b);

  auto he_handle = std::static_pointer_cast<ngraph::he::HESealExecutable>(
      he_backend->compile(f));

  he_handle->set_verbose_all_ops(false);

  he_handle->call_with_validate({t_result}, {t_a, t_b});
  EXPECT_TRUE(ngraph::test::he::all_close(read_vector<float>(t_result),
                                          exp_result, 1e-3f));
}
