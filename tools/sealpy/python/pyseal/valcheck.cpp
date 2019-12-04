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
#include "pyseal/valcheck.hpp"
#include "seal/seal.h"
#include "seal/util/rlwe.h"

using namespace seal;
using namespace pybind11::literals;
namespace py = pybind11;

void init_valcheck(py::module& m) {
  m.def("is_metadata_valid_for",
        py::overload_cast<const Plaintext&, std::shared_ptr<const SEALContext>>(
            &is_metadata_valid_for));
  m.def(
      "is_metadata_valid_for",
      py::overload_cast<const Ciphertext&, std::shared_ptr<const SEALContext>>(
          &is_metadata_valid_for));
  m.def("is_metadata_valid_for",
        py::overload_cast<const SecretKey&, std::shared_ptr<const SEALContext>>(
            &is_metadata_valid_for));
  m.def("is_metadata_valid_for",
        py::overload_cast<const PublicKey&, std::shared_ptr<const SEALContext>>(
            &is_metadata_valid_for));
  m.def(
      "is_metadata_valid_for",
      py::overload_cast<const KSwitchKeys&, std::shared_ptr<const SEALContext>>(
          &is_metadata_valid_for));
  m.def("is_metadata_valid_for",
        py::overload_cast<const RelinKeys&, std::shared_ptr<const SEALContext>>(
            &is_metadata_valid_for));
  m.def(
      "is_metadata_valid_for",
      py::overload_cast<const GaloisKeys&, std::shared_ptr<const SEALContext>>(
          &is_metadata_valid_for));
  m.def("is_valid_for",
        py::overload_cast<const Plaintext&, std::shared_ptr<const SEALContext>>(
            &is_valid_for));
  m.def(
      "is_valid_for",
      py::overload_cast<const Ciphertext&, std::shared_ptr<const SEALContext>>(
          &is_valid_for));
  m.def("is_valid_for",
        py::overload_cast<const SecretKey&, std::shared_ptr<const SEALContext>>(
            &is_valid_for));
  m.def("is_valid_for",
        py::overload_cast<const PublicKey&, std::shared_ptr<const SEALContext>>(
            &is_valid_for));
  m.def(
      "is_valid_for",
      py::overload_cast<const KSwitchKeys&, std::shared_ptr<const SEALContext>>(
          &is_valid_for));
  m.def("is_valid_for",
        py::overload_cast<const RelinKeys&, std::shared_ptr<const SEALContext>>(
            &is_valid_for));
  m.def(
      "is_valid_for",
      py::overload_cast<const GaloisKeys&, std::shared_ptr<const SEALContext>>(
          &is_valid_for));
}
