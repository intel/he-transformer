//*****************************************************************************
// Copyright 2019 Intel Corporation
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

#include <pybind11/complex.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "pyseal/context_data.hpp"
#include "pyseal/pyseal.hpp"
#include "seal/seal.h"

using namespace seal;
using namespace pybind11::literals;
namespace py = pybind11;

void regclass_pyseal_ContextData(py::module& m) {
  py::class_<SEALContext::ContextData,
             std::shared_ptr<SEALContext::ContextData>>
      context_data(m, "ContextData");
  context_data.def("parms", &SEALContext::ContextData::parms);
  context_data.def("parms_id", &SEALContext::ContextData::parms_id);
  context_data.def("qualifiers", &SEALContext::ContextData::qualifiers);
  context_data.def("total_coeff_modulus",
                   &SEALContext::ContextData::total_coeff_modulus);
  context_data.def("total_coeff_modulus_bit_count",
                   &SEALContext::ContextData::total_coeff_modulus_bit_count);
  context_data.def("coeff_div_plain_modulus",
                   &SEALContext::ContextData::coeff_div_plain_modulus);
  context_data.def("plain_upper_half_threshold",
                   &SEALContext::ContextData::plain_upper_half_threshold);
  context_data.def("plain_upper_half_increment",
                   &SEALContext::ContextData::plain_upper_half_increment);
  context_data.def("upper_half_threshold",
                   &SEALContext::ContextData::upper_half_threshold);
  context_data.def("upper_half_increment",
                   &SEALContext::ContextData::upper_half_increment);
  context_data.def("prev_context_data",
                   &SEALContext::ContextData::prev_context_data);
  context_data.def("next_context_data",
                   &SEALContext::ContextData::next_context_data);
  context_data.def("chain_index", &SEALContext::ContextData::chain_index);
}
