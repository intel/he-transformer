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

void regclass_pyseal_BatchEncoder(py::module& m) {
  py::class_<BatchEncoder> batch_encoder(m, "BatchEncoder");
  batch_encoder.def(py::init<std::shared_ptr<SEALContext>>());
  batch_encoder.def(
      "encode",
      py::overload_cast<const std::vector<std::uint64_t>&, Plaintext&>(
          &BatchEncoder::encode));
  batch_encoder.def(
      "encode", py::overload_cast<const std::vector<std::int64_t>&, Plaintext&>(
                    &BatchEncoder::encode));
  batch_encoder.def(
      "encode",
      py::overload_cast<Plaintext&, MemoryPoolHandle>(&BatchEncoder::encode),
      py::arg("plain"), py::arg("pool") = MemoryManager::GetPool());
  batch_encoder.def(
      "decode",
      py::overload_cast<const Plaintext&, std::vector<std::uint64_t>&,
                        MemoryPoolHandle>(&BatchEncoder::decode),
      py::arg("plain"), py::arg("destination"),
      py::arg("pool") = MemoryManager::GetPool());
  batch_encoder.def(
      "decode",
      py::overload_cast<const Plaintext&, std::vector<std::int64_t>&,
                        MemoryPoolHandle>(&BatchEncoder::decode),
      py::arg("plain"), py::arg("destination"),
      py::arg("pool") = MemoryManager::GetPool());
  batch_encoder.def(
      "decode",
      py::overload_cast<Plaintext&, MemoryPoolHandle>(&BatchEncoder::decode),
      py::arg("plain"), py::arg("pool") = MemoryManager::GetPool());
  batch_encoder.def("slot_count", &BatchEncoder::slot_count);
}
