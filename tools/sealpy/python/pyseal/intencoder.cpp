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

#include "pyseal/intencoder.hpp"
#include "pyseal/kswitchkeys.hpp"
#include "pyseal/pyseal.hpp"
#include "seal/seal.h"

using namespace seal;
using namespace pybind11::literals;
namespace py = pybind11;

void regclass_pyseal_IntegerEncoder(py::module& m) {
  py::class_<IntegerEncoder> int_encoder(m, "IntegerEncoder");
  int_encoder.def(py::init<std::shared_ptr<SEALContext>>());
  int_encoder.def("encode",
                  py::overload_cast<std::uint64_t>(&IntegerEncoder::encode));
  int_encoder.def("encode", py::overload_cast<std::uint64_t, Plaintext&>(
                                &IntegerEncoder::encode));
  // int_encoder.def("decode_uint32", IntegerEncoder::decode_uint32);
  int_encoder.def("decode_uint32",
                  (std::uint32_t(IntegerEncoder::*)(Plaintext&)) &
                      IntegerEncoder::decode_uint32);
  int_encoder.def("decode_uint64",
                  (std::uint64_t(IntegerEncoder::*)(Plaintext&)) &
                      IntegerEncoder::decode_uint64);

  // int_encoder.def("decode_uint64", IntegerEncoder::decode_uint64);
  int_encoder.def("encode",
                  py::overload_cast<std::int64_t>(&IntegerEncoder::encode));
  int_encoder.def("encode", py::overload_cast<std::int64_t, Plaintext&>(
                                &IntegerEncoder::encode));
  int_encoder.def("encode",
                  py::overload_cast<const BigUInt&>(&IntegerEncoder::encode));
  int_encoder.def("encode", py::overload_cast<const BigUInt&, Plaintext&>(
                                &IntegerEncoder::encode));
  // int_encoder.def("decode_int32", IntegerEncoder::decode_int32);
  int_encoder.def("decode_int32",
                  (std::int64_t(IntegerEncoder::*)(Plaintext&)) &
                      IntegerEncoder::decode_int32);

  // int_encoder.def("decode_int64", IntegerEncoder::decode_int64);
  int_encoder.def("decode_int64",
                  (std::int64_t(IntegerEncoder::*)(Plaintext&)) &
                      IntegerEncoder::decode_int64);

  // int_encoder.def("decode_biguint", IntegerEncoder::decode_biguint);
  int_encoder.def("decode_biguint", (BigUInt(IntegerEncoder::*)(Plaintext&)) &
                                        IntegerEncoder::decode_biguint);
  int_encoder.def("encode",
                  py::overload_cast<std::int32_t>(&IntegerEncoder::encode));
  int_encoder.def("encode",
                  py::overload_cast<std::uint32_t>(&IntegerEncoder::encode));
  int_encoder.def("encode",
                  py::overload_cast<std::int32_t>(&IntegerEncoder::encode));
  int_encoder.def("encode", py::overload_cast<std::int32_t, Plaintext&>(
                                &IntegerEncoder::encode));
  int_encoder.def("plain_modulus", &IntegerEncoder::plain_modulus);
}
