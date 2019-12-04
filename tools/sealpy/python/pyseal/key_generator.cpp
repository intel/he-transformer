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

#include "pyseal/key_generator.hpp"
#include "pyseal/pyseal.hpp"
#include "seal/seal.h"

using namespace seal;
using namespace pybind11::literals;
namespace py = pybind11;

void regclass_pyseal_KeyGenerator(py::module& m) {
  py::class_<KeyGenerator> key_generator(m, "KeyGenerator");
  key_generator.def(py::init<std::shared_ptr<SEALContext>>());
  key_generator.def(py::init<std::shared_ptr<SEALContext>, const SecretKey&>());
  key_generator.def(py::init<std::shared_ptr<SEALContext>, const SecretKey&,
                             const PublicKey&>());
  key_generator.def("secret_key", &KeyGenerator::secret_key);
  key_generator.def("public_key", &KeyGenerator::public_key);
  key_generator.def("relin_keys",
                    (RelinKeys(KeyGenerator::*)()) & KeyGenerator::relin_keys);
  key_generator.def(
      "galois_keys",
      (GaloisKeys(KeyGenerator::*)(const std::vector<std::uint64_t>&)) &
          KeyGenerator::galois_keys,
      py::arg("galois_elts"));
  key_generator.def("galois_keys",
                    (GaloisKeys(KeyGenerator::*)(const std::vector<int>&)) &
                        KeyGenerator::galois_keys,
                    py::arg("steps"));
  key_generator.def("galois_keys", (GaloisKeys(KeyGenerator::*)()) &
                                       KeyGenerator::galois_keys);
}
