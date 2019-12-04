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

#include "pyseal/plaintext.hpp"
#include "seal/seal.h"
#include "pyseal/pyseal.hpp"

using namespace seal;
using namespace pybind11::literals;
namespace py = pybind11;

void regclass_pyseal_Plaintext(py::module& m) {
  // TODO: move/copy
  py::class_<Plaintext> plaintext(m, "Plaintext");
  plaintext.def(py::init<MemoryPoolHandle>(),
                py::arg("pool") = MemoryManager::GetPool());
  plaintext.def(py::init<Plaintext::size_type, MemoryPoolHandle>(),
                py::arg("coeff_count"),
                py::arg("pool") = MemoryManager::GetPool());
  plaintext.def(
      py::init<Plaintext::size_type, Plaintext::size_type, MemoryPoolHandle>(),
      py::arg("capacity"), py::arg("coeff_count"),
      py::arg("pool") = MemoryManager::GetPool());
  plaintext.def(py::init<const std::string&, MemoryPoolHandle>(),
                py::arg("hex_poly"),
                py::arg("pool") = MemoryManager::GetPool());
  plaintext.def("reserve", &Plaintext::reserve);
  plaintext.def("shrink_to_fit", &Plaintext::shrink_to_fit);
  plaintext.def("release", &Plaintext::release);
  plaintext.def("resize", &Plaintext::resize);
  plaintext.def("assign", [](Plaintext& object, const Plaintext& other) {
    object = other;
  });
  plaintext.def("set_zero",
                py::overload_cast<Plaintext::size_type, Plaintext::size_type>(
                    &Plaintext::set_zero));
  plaintext.def("set_zero",
                py::overload_cast<Plaintext::size_type>(&Plaintext::set_zero));
  plaintext.def("set_zero", py::overload_cast<>(&Plaintext::set_zero));
  plaintext.def("data", py::overload_cast<>(&Plaintext::data, py::const_));
  plaintext.def("data", py::overload_cast<>(&Plaintext::data));
  plaintext.def("data", py::overload_cast<Plaintext::size_type>(
                            &Plaintext::data, py::const_));
  plaintext.def("data",
                py::overload_cast<Plaintext::size_type>(&Plaintext::data));
  plaintext.def("__eq__", &Plaintext::operator==);
  plaintext.def("__neq__", &Plaintext::operator!=);
  plaintext.def("is_zero", &Plaintext::is_zero);
  plaintext.def("capacity", &Plaintext::capacity);
  plaintext.def("coeff_count", &Plaintext::coeff_count);
  plaintext.def("significant_coeff_count", &Plaintext::significant_coeff_count);
  plaintext.def("nonzero_coeff_count", &Plaintext::nonzero_coeff_count);
  plaintext.def("to_string", &Plaintext::to_string);
  plaintext.def("__repr__", [](const Plaintext& p) { return p.to_string(); });
  plaintext.def("save", [](Plaintext& p) {
    std::stringstream ss;
    p.save(ss);
    return py::bytes(ss.str());
  });
  plaintext.def("unsafe_load", [](Plaintext& p, std::string& input) {
    std::stringstream ss(input);
    p.unsafe_load(ss);
  });
  plaintext.def("load", [](Plaintext& p, std::shared_ptr<SEALContext> context,
                           std::string& input) {
    std::stringstream ss(input);
    p.load(context, ss);
  });
  plaintext.def("is_ntt_form", &Plaintext::is_ntt_form);
  plaintext.def_property("parms_id",
                         py::overload_cast<>(&Plaintext::parms_id, py::const_),
                         // Setter
                         [](Plaintext& plain, parms_id_type& parms_id) {
                           plain.parms_id() = parms_id;
                         });
  plaintext.def_property(
      "scale", py::overload_cast<>(&Plaintext::scale, py::const_),
      // Setter
      [](Plaintext& plain, double& scale) { plain.scale() = scale; });
  plaintext.def("pool", &Plaintext::pool);
  plaintext.def("__getitem__",
                [](const Plaintext& plain, Plaintext::size_type coeff_index) {
                  return plain[coeff_index];
                });
  plaintext.def("__setitem__",
                [](Plaintext& plain, Plaintext::size_type coeff_index,
                   Plaintext::pt_coeff_type v) { plain[coeff_index] = v; });
}
