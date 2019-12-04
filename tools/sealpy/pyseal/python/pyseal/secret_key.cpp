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
#include "pyseal/secret_key.hpp"
#include "seal/seal.h"

using namespace seal;
using namespace pybind11::literals;
namespace py = pybind11;

void regclass_pyseal_SecretKey(py::module& m) {
  py::class_<SecretKey> secret_key(m, "SecretKey");
  secret_key.def(py::init<>());
  secret_key.def(py::init<const SecretKey&>());
  secret_key.def("data", py::overload_cast<>(&SecretKey::data));
  secret_key.def("data", py::overload_cast<>(&SecretKey::data, py::const_));
  secret_key.def("save", [](SecretKey& sk) {
    std::stringstream ss;
    sk.save(ss);
    return py::bytes(ss.str());
  });
  secret_key.def("unsafe_load", [](SecretKey& sk, std::string& input) {
    std::stringstream ss(input);
    sk.unsafe_load(ss);
  });
  secret_key.def("load", [](SecretKey& sk, std::shared_ptr<SEALContext> context,
                            std::string& input) {
    std::stringstream ss(input);
    sk.load(context, ss);
  });
  secret_key.def("parms_id",
                 py::overload_cast<>(&SecretKey::parms_id, py::const_));
  secret_key.def("parms_id", py::overload_cast<>(&SecretKey::parms_id));
  secret_key.def("pool", &SecretKey::pool);
}
