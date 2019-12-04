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

#include "pyseal/encryptor.hpp"
#include "pyseal/pyseal.hpp"
#include "seal/seal.h"

using namespace seal;
using namespace pybind11::literals;
namespace py = pybind11;

void regclass_pyseal_Encryptor(py::module& m) {
  py::class_<Encryptor> encryptor(m, "Encryptor");
  encryptor.def(py::init<std::shared_ptr<SEALContext>, const PublicKey&>());
  encryptor.def(
      "encrypt",
      (void (Encryptor::*)(const Plaintext&, Ciphertext&, MemoryPoolHandle)) &
          Encryptor::encrypt,
      py::arg("plain"), py::arg("destination"),
      py::arg("pool") = MemoryManager::GetPool());
  encryptor.def(
      "encrypt_zero",
      (void (Encryptor::*)(parms_id_type, Ciphertext&, MemoryPoolHandle)) &
          Encryptor::encrypt_zero,
      py::arg("parms_id"), py::arg("destination"),
      py::arg("pool") = MemoryManager::GetPool());
  encryptor.def("encrypt_zero",
                (void (Encryptor::*)(Ciphertext&, MemoryPoolHandle)) &
                    Encryptor::encrypt_zero,
                py::arg("destination"),
                py::arg("pool") = MemoryManager::GetPool());
}
