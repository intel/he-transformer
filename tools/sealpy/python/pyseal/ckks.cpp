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

#include "pyseal/ckks.hpp"
#include "pyseal/pyseal.hpp"
#include "seal/seal.h"
#include "seal/util/rlwe.h"

using namespace seal;
using namespace pybind11::literals;
namespace py = pybind11;

void regclass_pyseal_CKKSEncoder(py::module& m) {
  py::class_<CKKSEncoder> ckks_encoder(m, "CKKSEncoder");
  ckks_encoder.def(py::init<std::shared_ptr<SEALContext>>());
  ckks_encoder.def("encode",
                   (void (CKKSEncoder::*)(
                       const std::vector<std::complex<double>>&, parms_id_type,
                       double, Plaintext&, MemoryPoolHandle)) &
                       CKKSEncoder::encode,
                   py::arg("values"), py::arg("parms_id"), py::arg("scale"),
                   py::arg("destination"),
                   py::arg("pool") = MemoryManager::GetPool());
  ckks_encoder.def(
      "encode",
      (void (CKKSEncoder::*)(const std::vector<double>&, parms_id_type, double,
                             Plaintext&, MemoryPoolHandle)) &
          CKKSEncoder::encode,
      py::arg("values"), py::arg("parms_id"), py::arg("scale"),
      py::arg("destination"), py::arg("pool") = MemoryManager::GetPool());
  ckks_encoder.def(
      "encode",
      (void (CKKSEncoder::*)(const std::vector<std::complex<double>>&, double,
                             Plaintext&, MemoryPoolHandle)) &
          CKKSEncoder::encode,
      py::arg("values"), py::arg("scale"), py::arg("destination"),
      py::arg("pool") = MemoryManager::GetPool());
  ckks_encoder.def("encode",
                   (void (CKKSEncoder::*)(const std::vector<double>&, double,
                                          Plaintext&, MemoryPoolHandle)) &
                       CKKSEncoder::encode,
                   py::arg("values"), py::arg("scale"), py::arg("destination"),
                   py::arg("pool") = MemoryManager::GetPool());
  ckks_encoder.def("encode",
                   (void (CKKSEncoder::*)(double, parms_id_type, double,
                                          Plaintext&, MemoryPoolHandle)) &
                       CKKSEncoder::encode,
                   py::arg("value"), py::arg("parms_id"), py::arg("scale"),
                   py::arg("destination"),
                   py::arg("pool") = MemoryManager::GetPool());
  ckks_encoder.def(
      "encode",
      (void (CKKSEncoder::*)(double, double, Plaintext&, MemoryPoolHandle)) &
          CKKSEncoder::encode,
      py::arg("value"), py::arg("scale"), py::arg("destination"),
      py::arg("pool") = MemoryManager::GetPool());
  ckks_encoder.def(
      "encode",
      (void (CKKSEncoder::*)(std::complex<double>, parms_id_type, double,
                             Plaintext&, MemoryPoolHandle)) &
          CKKSEncoder::encode,
      py::arg("value"), py::arg("parms_id"), py::arg("scale"),
      py::arg("destination"), py::arg("pool") = MemoryManager::GetPool());
  ckks_encoder.def("encode",
                   (void (CKKSEncoder::*)(std::complex<double>, double,
                                          Plaintext&, MemoryPoolHandle)) &
                       CKKSEncoder::encode,
                   py::arg("value"), py::arg("scale"), py::arg("destination"),
                   py::arg("pool") = MemoryManager::GetPool());
  ckks_encoder.def("encode", (void (CKKSEncoder::*)(std::int64_t, parms_id_type,
                                                    Plaintext&)) &
                                 CKKSEncoder::encode);
  ckks_encoder.def("encode", (void (CKKSEncoder::*)(std::int64_t, Plaintext&)) &
                                 CKKSEncoder::encode);
  ckks_encoder.def(
      "decode",
      (void (CKKSEncoder::*)(const Plaintext&, std::vector<double>&,
                             MemoryPoolHandle)) &
          CKKSEncoder::decode,
      py::arg("plain"), py::arg("destination"),
      py::arg("pool") = MemoryManager::GetPool());
  ckks_encoder.def("decode",
                   (void (CKKSEncoder::*)(const Plaintext&,
                                          std::vector<std::complex<double>>&,
                                          MemoryPoolHandle)) &
                       CKKSEncoder::decode,
                   py::arg("plain"), py::arg("destination"),
                   py::arg("pool") = MemoryManager::GetPool());
  ckks_encoder.def("slot_count", &CKKSEncoder::slot_count);
}
