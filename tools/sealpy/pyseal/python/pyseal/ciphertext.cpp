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

#include "pyseal/ciphertext.hpp"
#include "pyseal/memory_pool_handle.hpp"
#include "pyseal/pyseal.hpp"
#include "seal/seal.h"

using namespace seal;
using namespace pybind11::literals;
namespace py = pybind11;

void regclass_pyseal_Ciphertext(py::module& m) {
  // TODO: copy/move operators, etc.
  py::class_<Ciphertext> ciphertext(m, "Ciphertext");
  ciphertext.def(py::init<MemoryPoolHandle>(),
                 py::arg("pool") = MemoryManager::GetPool());
  ciphertext.def(py::init<std::shared_ptr<SEALContext>, MemoryPoolHandle>(),
                 py::arg("context"),
                 py::arg("pool") = MemoryManager::GetPool());
  ciphertext.def(
      py::init<std::shared_ptr<SEALContext>, parms_id_type, MemoryPoolHandle>(),
      py::arg("context"), py::arg("parms_id"),
      py::arg("pool") = MemoryManager::GetPool());
  ciphertext.def(py::init<std::shared_ptr<SEALContext>, parms_id_type,
                          Ciphertext::size_type, MemoryPoolHandle>(),
                 py::arg("context"), py::arg("parms_id"),
                 py::arg("size_capacity"),
                 py::arg("pool") = MemoryManager::GetPool());
  ciphertext.def(py::init<const Ciphertext&>());
  // ciphertext.def(py::init<Ciphertext&&>());
  ciphertext.def(py::init<const Ciphertext&, MemoryPoolHandle>());
  ciphertext.def_property(
      "parms_id", py::overload_cast<>(&Ciphertext::parms_id, py::const_),
      // Setter
      [](Ciphertext& plain, parms_id_type& parms_id) {
        plain.parms_id() = parms_id;
      });
  ciphertext.def_property(
      "scale", py::overload_cast<>(&Ciphertext::scale, py::const_),
      // Setter
      [](Ciphertext& cipher, double& scale) { cipher.scale() = scale; });
  ciphertext.def(
      "reserve",
      py::overload_cast<std::shared_ptr<SEALContext>, parms_id_type,
                        Ciphertext::size_type>(&Ciphertext::reserve));
  ciphertext.def(
      "reserve",
      py::overload_cast<std::shared_ptr<SEALContext>, Ciphertext::size_type>(
          &Ciphertext::reserve));
  ciphertext.def("reserve", py::overload_cast<Ciphertext::size_type>(
                                &Ciphertext::reserve));
  ciphertext.def("resize",
                 py::overload_cast<std::shared_ptr<SEALContext>, parms_id_type,
                                   Ciphertext::size_type>(&Ciphertext::resize));
  ciphertext.def(
      "resize",
      py::overload_cast<std::shared_ptr<SEALContext>, Ciphertext::size_type>(
          &Ciphertext::resize));
  ciphertext.def("resize",
                 py::overload_cast<Ciphertext::size_type>(&Ciphertext::resize));
  ciphertext.def("release", &Ciphertext::release);
  ciphertext.def("assign", [](Ciphertext& object, const Ciphertext& other) {
    object = other;
  });
  ciphertext.def("data", py::overload_cast<>(&Ciphertext::data, py::const_));
  ciphertext.def("data", py::overload_cast<>(&Ciphertext::data));
  ciphertext.def("data", py::overload_cast<Ciphertext::size_type>(
                             &Ciphertext::data, py::const_));
  ciphertext.def("data",
                 py::overload_cast<Ciphertext::size_type>(&Ciphertext::data));
  ciphertext.def("__getitem__", [](const Ciphertext& cipher,
                                   Ciphertext::size_type coeff_index) {
    return cipher[coeff_index];
  });
  ciphertext.def("__setitem__",
                 [](Ciphertext& cipher, Ciphertext::size_type coeff_index,
                    Ciphertext::ct_coeff_type v) { cipher[coeff_index] = v; });
  ciphertext.def("coeff_mod_count", &Ciphertext::coeff_mod_count);
  ciphertext.def("poly_modulus_degree", &Ciphertext::poly_modulus_degree);
  ciphertext.def("size", &Ciphertext::size);
  ciphertext.def("uint64_count_capacity", &Ciphertext::uint64_count_capacity);
  ciphertext.def("size_capacity", &Ciphertext::size_capacity);
  ciphertext.def("uint64_count", &Ciphertext::uint64_count);
  ciphertext.def("is_transparent", &Ciphertext::is_transparent);
  ciphertext.def("save", [](Ciphertext& c) {
    std::stringstream ss;
    c.save(ss);
    return py::bytes(ss.str());
  });
  ciphertext.def("unsafe_load", [](Ciphertext& c, std::string& input) {
    std::stringstream ss(input);
    c.unsafe_load(ss);
  });
  ciphertext.def("load", [](Ciphertext& c, std::shared_ptr<SEALContext> context,
                            std::string& input) {
    std::stringstream ss(input);
    c.load(context, ss);
  });
  ciphertext.def("is_ntt_form",
                 py::overload_cast<>(&Ciphertext::is_ntt_form, py::const_));
  ciphertext.def("is_ntt_form", py::overload_cast<>(&Ciphertext::is_ntt_form));
  ciphertext.def("pool", &Ciphertext::pool);
}
