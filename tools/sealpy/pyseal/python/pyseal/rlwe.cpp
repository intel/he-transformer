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
#include "pyseal/rlwe.hpp"
#include "seal/seal.h"
#include "seal/util/rlwe.h"

using namespace seal;
using namespace pybind11::literals;
namespace py = pybind11;

void init_rlwe(py::module& m) {
  m.def("sample_poly_ternary", &util::sample_poly_ternary);
  m.def("sample_poly_normal", &util::sample_poly_normal);
  m.def("sample_poly_uniform", &util::sample_poly_uniform);
  m.def("encrypt_zero_asymmetric", &util::encrypt_zero_asymmetric);
  m.def("encrypt_zero_symmetric", &util::encrypt_zero_symmetric);
}
