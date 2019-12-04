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

#include "pyseal/encryption_parameters.hpp"
#include "pyseal/pyseal.hpp"
#include "seal/seal.h"

using namespace seal;
using namespace pybind11::literals;
namespace py = pybind11;

void regclass_pyseal_EncryptionParameters(py::module& m) {
  py::class_<EncryptionParameters> encryption_parameters(
      m, "EncryptionParameters");
  encryption_parameters.def(py::init<scheme_type>());
  encryption_parameters.def(py::init<std::uint8_t>());
  encryption_parameters.def(py::init<const EncryptionParameters&>());
  encryption_parameters.def("set_poly_modulus_degree",
                            &EncryptionParameters::set_poly_modulus_degree);
  encryption_parameters.def("set_coeff_modulus",
                            &EncryptionParameters::set_coeff_modulus);
  encryption_parameters.def(
      "set_plain_modulus",
      (void (EncryptionParameters::*)(const SmallModulus&)) &
          EncryptionParameters::set_plain_modulus);
  encryption_parameters.def("set_plain_modulus",
                            (void (EncryptionParameters::*)(std::uint64_t)) &
                                EncryptionParameters::set_plain_modulus);
  encryption_parameters.def("set_random_generator",
                            &EncryptionParameters::set_random_generator);
  encryption_parameters.def("scheme", &EncryptionParameters::scheme);
  encryption_parameters.def("poly_modulus_degree",
                            &EncryptionParameters::poly_modulus_degree);
  encryption_parameters.def("coeff_modulus",
                            &EncryptionParameters::coeff_modulus);
  encryption_parameters.def("plain_modulus",
                            &EncryptionParameters::plain_modulus);
  encryption_parameters.def("random_generator",
                            &EncryptionParameters::random_generator);
  encryption_parameters.def("__eq__", &EncryptionParameters::operator==);
  encryption_parameters.def("__neq__", &EncryptionParameters::operator!=);
  encryption_parameters.def_static("Save", [](EncryptionParameters& parms) {
    std::stringstream ss;
    EncryptionParameters::Save(parms, ss);
    return py::bytes(ss.str());
  });
  encryption_parameters.def_static("Load", [](std::string& input) {
    std::stringstream ss(input);
    return EncryptionParameters::Load(ss);
  });
}
