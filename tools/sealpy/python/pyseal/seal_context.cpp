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

#include "pyseal/pyseal.hpp"
#include "pyseal/seal_context.hpp"
#include "seal/seal.h"

using namespace seal;
using namespace pybind11::literals;
namespace py = pybind11;

void regclass_pyseal_SEALContext(py::module& m) {
  py::class_<EncryptionParameterQualifiers> encryption_parameter_qualifiers(
      m, "EncryptionParameterQualifiers");
  encryption_parameter_qualifiers.def_readwrite(
      "parameters_set", &EncryptionParameterQualifiers::parameters_set);
  encryption_parameter_qualifiers.def_readwrite(
      "using_fft", &EncryptionParameterQualifiers::using_fft);
  encryption_parameter_qualifiers.def_readwrite(
      "using_ntt", &EncryptionParameterQualifiers::using_ntt);
  encryption_parameter_qualifiers.def_readwrite(
      "using_batching", &EncryptionParameterQualifiers::using_batching);
  encryption_parameter_qualifiers.def_readwrite(
      "using_fast_plain_lift",
      &EncryptionParameterQualifiers::using_fast_plain_lift);
  encryption_parameter_qualifiers.def_readwrite(
      "using_descending_modulus_chain",
      &EncryptionParameterQualifiers::using_descending_modulus_chain);
  encryption_parameter_qualifiers.def_readwrite(
      "sec_level", &EncryptionParameterQualifiers::sec_level);

  py::class_<SEALContext, std::shared_ptr<SEALContext>> seal_context(
      m, "SEALContext");
  seal_context.def_static("Create", &SEALContext::Create, py::arg("parms"),
                          py::arg("expand_mod_chain") = true,
                          py::arg("sec_level") = sec_level_type::tc128);
  seal_context.def("get_context_data", &SEALContext::get_context_data);
  seal_context.def("key_context_data", &SEALContext::key_context_data);
  seal_context.def("first_context_data", &SEALContext::first_context_data);
  seal_context.def("last_context_data", &SEALContext::last_context_data);
  seal_context.def("parameters_set", &SEALContext::parameters_set);
  seal_context.def("key_parms_id", &SEALContext::key_parms_id);
  seal_context.def("first_parms_id", &SEALContext::first_parms_id);
  seal_context.def("last_parms_id", &SEALContext::last_parms_id);
  seal_context.def("using_keyswitching", &SEALContext::using_keyswitching);
}
