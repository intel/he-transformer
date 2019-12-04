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

#include "pyseal/small_modulus.hpp"
#include "seal/seal.h"
#include "pyseal/pyseal.hpp"

using namespace seal;
using namespace pybind11::literals;
namespace py = pybind11;

void regclass_pyseal_SmallModulus(py::module& m) {
  py::class_<SmallModulus> small_modulus(m, "SmallModulus");
  small_modulus.def(py::init<>());
  small_modulus.def(py::init<std::uint64_t>());
  small_modulus.def("value",
                    (std::uint64_t(SmallModulus::*)()) & SmallModulus::value);
  small_modulus.def("bit_count", &SmallModulus::bit_count);
  small_modulus.def("uint64_count", &SmallModulus::uint64_count);
  small_modulus.def("data", &SmallModulus::data);
  small_modulus.def("value", &SmallModulus::value);
  small_modulus.def("const_ratio", &SmallModulus::const_ratio);
  small_modulus.def("is_zero", &SmallModulus::is_zero);
  small_modulus.def("is_prime", &SmallModulus::is_prime);
  small_modulus.def("__eq__", [](const SmallModulus& a, const SmallModulus& b) {
    return a == b;
  });
  small_modulus.def(
      "__eq__", [](const SmallModulus& a, std::uint64_t b) { return a == b; });
  small_modulus.def("__neq__", [](const SmallModulus& a,
                                  const SmallModulus& b) { return a != b; });
  small_modulus.def(
      "__neq__", [](const SmallModulus& a, std::uint64_t b) { return a != b; });
  small_modulus.def("__lt__", [](const SmallModulus& a, const SmallModulus& b) {
    return a < b;
  });
  small_modulus.def(
      "__lt__", [](const SmallModulus& a, std::uint64_t b) { return a < b; });
  small_modulus.def("__le__", [](const SmallModulus& a, const SmallModulus& b) {
    return a <= b;
  });
  small_modulus.def(
      "__le__", [](const SmallModulus& a, std::uint64_t b) { return a <= b; });
  small_modulus.def("__gt__", [](const SmallModulus& a, const SmallModulus& b) {
    return a > b;
  });
  small_modulus.def(
      "__gt__", [](const SmallModulus& a, std::uint64_t b) { return a > b; });
  small_modulus.def("__ge__", [](const SmallModulus& a, const SmallModulus& b) {
    return a >= b;
  });
  small_modulus.def(
      "__ge__", [](const SmallModulus& a, std::uint64_t b) { return a >= b; });
  small_modulus.def("save", [](SmallModulus& modulus) {
    std::stringstream ss;
    modulus.save(ss);
    return py::bytes(ss.str());
  });
  small_modulus.def("load", [](SmallModulus& modulus, std::string& input) {
    std::stringstream ss(input);
    modulus.load(ss);
  });
}
