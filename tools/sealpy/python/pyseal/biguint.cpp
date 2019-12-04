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
#include <cstddef>

#include "pyseal/biguint.hpp"
#include "pyseal/pyseal.hpp"
#include "seal/seal.h"

using namespace seal;
using namespace pybind11::literals;
namespace py = pybind11;

void regclass_pyseal_BigUInt(py::module& m) {
  py::class_<BigUInt> big_uint(m, "BigUInt");
  big_uint.def(py::init<>());
  big_uint.def(py::init<int>());
  big_uint.def(py::init<const std::string&>());
  big_uint.def(py::init<int, const std::string&>());
  big_uint.def(py::init<int, std::uint64_t*>());
  big_uint.def(py::init<int, std::uint64_t>());
  big_uint.def(py::init<const BigUInt&>());
  big_uint.def("is_alias", &BigUInt::is_alias);
  big_uint.def("bit_count", &BigUInt::bit_count);
  big_uint.def("data", py::overload_cast<>(&BigUInt::data, py::const_));
  big_uint.def("data", py::overload_cast<>(&BigUInt::data));
  big_uint.def("data", [](const BigUInt& biguint, std::size_t index) {
    return biguint.data()[index];
  });
  big_uint.def("byte_count", &BigUInt::byte_count);
  big_uint.def("uint64_count", &BigUInt::uint64_count);
  big_uint.def("significant_bit_count", &BigUInt::significant_bit_count);
  big_uint.def("to_double", &BigUInt::to_double);
  big_uint.def("to_string", &BigUInt::to_string);
  big_uint.def("to_dec_string", &BigUInt::to_dec_string);
  big_uint.def("is_zero", &BigUInt::is_zero);

  big_uint.def("__getitem__", [](const BigUInt& biguint, std::size_t index) {
    return std::to_integer<int>(biguint[index]);
  });
  // TODO: operator []
  // big_uint.def("__setitem__",
  //               [](const BigUInt& biguint, std::size_t index,
  //                  SEAL_BYTE v) { biguint[index] = v; });

  big_uint.def("set_zero", &BigUInt::set_zero);
  big_uint.def("resize", &BigUInt::resize);
  big_uint.def("alias", &BigUInt::alias);
  big_uint.def("unalias", &BigUInt::unalias);
  big_uint.def("assign",
               [](BigUInt& object, const BigUInt& other) { object = other; });
  big_uint.def("assign", [](BigUInt& object, const std::string& hex_value) {
    object = hex_value;
  });
  big_uint.def("assign", [](BigUInt& object, std::uint64_t hex_value) {
    object = hex_value;
  });
  // TODO: operator +
  // TODO: operator -
  // TODO: operator ++
  // TODO: operator --
  // TODO: operator *
  // TODO: operator /
  // TODO: operator ^
  // TODO: operator &
  // TODO: operator |
  // TODO: operator compareto
  // TODO: operator <
  // TODO: operator >
  // TODO: operator <=
  // TODO: operator >=
  // TODO: operator ==
  big_uint.def("__eq__",
               [](const BigUInt& a, const BigUInt& b) { return a == b; });
  // &BigUInt::operator==);
  // TODO: operator !=
  // TODO: operator <<
  // TODO: operator >>
  // TODO: operator +=
  // TODO: operator *=
  // TODO: operator /=
  // TODO: operator ^=
  // TODO: operator &=
  // TODO: operator |=
  // TODO: operator <<=
  // TODO: operator >>=
  // TODO: divrem
  // TODO: modinv
  // TODO: trymodinv
  // TODO: trymodinv
  big_uint.def("save", [](BigUInt& biguint) {
    std::stringstream ss;
    biguint.save(ss);
    return py::bytes(ss.str());
  });
  big_uint.def("load", [](BigUInt& biguint, std::string& input) {
    std::stringstream ss(input);
    biguint.load(ss);
  });
  big_uint.def("of", &BigUInt::of);
  big_uint.def("duplicate_to", &BigUInt::duplicate_to);
  big_uint.def("duplicate_from", &BigUInt::duplicate_from);
}
