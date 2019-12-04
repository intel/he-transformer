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

#include "pyseal/public_key.hpp"
#include "pyseal/pyseal.hpp"
#include "seal/seal.h"

using namespace seal;
using namespace pybind11::literals;
namespace py = pybind11;

void regclass_pyseal_PublicKey(py::module& m) {
  py::class_<PublicKey> public_key(m, "PublicKey");
  public_key.def(py::init<>());
  public_key.def(py::init<const PublicKey&>());
  public_key.def("data", py::overload_cast<>(&PublicKey::data));
  public_key.def("data", py::overload_cast<>(&PublicKey::data, py::const_));
  public_key.def("save", [](PublicKey& pk) {
    std::stringstream ss;
    pk.save(ss);
    return py::bytes(ss.str());
  });
  public_key.def("unsafe_load", [](PublicKey& pk, std::string& input) {
    std::stringstream ss(input);
    pk.unsafe_load(ss);
  });
  public_key.def("load", [](PublicKey& pk, std::shared_ptr<SEALContext> context,
                            std::string& input) {
    std::stringstream ss(input);
    pk.load(context, ss);
  });
  public_key.def("parms_id",
                 py::overload_cast<>(&PublicKey::parms_id, py::const_));
  public_key.def("parms_id", py::overload_cast<>(&PublicKey::parms_id));
  public_key.def("pool", &PublicKey::pool);
}
