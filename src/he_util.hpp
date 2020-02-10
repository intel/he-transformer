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

#include <complex>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "ngraph/check.hpp"
#include "ngraph/except.hpp"
#include "ngraph/util.hpp"
#include "nlohmann/json.hpp"
#include "op/bounded_relu.hpp"
#include "protos/message.pb.h"

namespace ngraph::runtime::he {

// This expands the op list in opset_he_seal_tbl.hpp into a list of enumerations
// that look like this: Abs, Acos,
// ...
enum class OP_TYPEID {
#define NGRAPH_OP(NAME, NAMESPACE) ID_SUFFIX(NAME),
#include "seal/opset_he_seal_tbl.hpp"
#undef NGRAPH_OP
  UnknownOp
};

/// \brief Unpacks complex values to real values
/// (a+bi, c+di) => (a,b,c,d)
/// \param[in] input Vector of complex values to unpack
/// \returns Vector storing unpacked real values
std::vector<double> complex_vec_to_real_vec(
    const std::vector<std::complex<double>>& input);

/// \brief Packs elements of input into complex values
/// (a,b,c,d) => (a+bi, c+di)
/// (a,b,c) => (a+bi, c+0i)
/// \param[in] input Vector of real values to unpack
/// \returns Vector storing packed complex values
std::vector<std::complex<double>> real_vec_to_complex_vec(
    const std::vector<double>& input);

template <typename T>
inline std::unordered_map<std::string,
                          std::pair<std::string, std::vector<double>>>
map_to_double_map(
    const std::unordered_map<std::string,
                             std::pair<std::string, std::vector<T>>>& inputs) {
  std::unordered_map<std::string, std::pair<std::string, std::vector<double>>>
      outputs;

  for (const auto& elem : inputs) {
    std::vector<double> double_inputs{elem.second.second.begin(),
                                      elem.second.second.end()};
    outputs.insert(
        {elem.first, std::make_pair(elem.second.first, double_inputs)});
  }
  return outputs;
}

/// \brief Interprets a string as a boolean value
/// \param[in] string to interpret as a boolean value
/// \param[in] default_value Value to return if flag is not able to be parsed
/// \returns True if flag represents a true value, false otherwise
bool string_to_bool(const char* string, bool default_value = false);

inline bool string_to_bool(const std::string& string,
                           bool default_value = false) {
  return string_to_bool(string.c_str(), default_value);
}

inline std::string bool_to_string(const bool b) {
  std::ostringstream ss;
  ss << std::boolalpha << b;
  return ss.str();
}

int flag_to_int(const char* flag, int default_value = 0);

inline int flag_to_int(const std::string& flag, int default_value = 0) {
  return flag_to_int(flag.c_str(), default_value);
}

/// \brief Converts a type to a double using static_cast
/// Note, this means a reduction of range in int64 and uint64 values.
/// \param[in] src Source from which to read
/// \param[in] element_type Datatype to interpret source as
/// \returns double value
double type_to_double(const void* src, const element::Type& element_type);

bool param_originates_from_name(const op::Parameter& param,
                                const std::string& name);

pb::HETensor_ElementType type_to_pb_type(const element::Type& element_type);

element::Type pb_type_to_type(pb::HETensor_ElementType pb_type);

pb::Function node_to_pb_function(
    const Node& node,
    std::unordered_map<std::string, std::string> extra_configs = {});

OP_TYPEID get_typeid(const NodeTypeInfo& type_info);

}  // namespace ngraph::runtime::he
