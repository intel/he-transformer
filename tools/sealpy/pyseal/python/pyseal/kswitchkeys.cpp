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

#include "pyseal/kswitchkeys.hpp"
#include "pyseal/pyseal.hpp"
#include "seal/seal.h"

using namespace seal;
using namespace pybind11::literals;
namespace py = pybind11;

void regclass_pyseal_KSwitchKeys(py::module& m) {
  py::class_<KSwitchKeys> kswitchkeys(m, "KSwitchKeys");
  kswitchkeys.def(py::init<>());
  kswitchkeys.def(py::init<const KSwitchKeys&>());
  kswitchkeys.def("size", &KSwitchKeys::size);
  kswitchkeys.def("parms_id",
                  py::overload_cast<>(&KSwitchKeys::parms_id, py::const_));
  kswitchkeys.def("parms_id", py::overload_cast<>(&KSwitchKeys::parms_id));
  kswitchkeys.def("data", py::overload_cast<>(&KSwitchKeys::data, py::const_));
  kswitchkeys.def("data", py::overload_cast<>(&KSwitchKeys::data));
  kswitchkeys.def("data",
                  py::overload_cast<size_t>(&KSwitchKeys::data, py::const_));
  kswitchkeys.def("data", py::overload_cast<size_t>(&KSwitchKeys::data));
  kswitchkeys.def("save", [](KSwitchKeys& ksk) {
    std::stringstream ss;
    ksk.save(ss);
    return py::bytes(ss.str());
  });
  kswitchkeys.def("unsafe_load", [](KSwitchKeys& ksk, std::string& input) {
    std::stringstream ss(input);
    ksk.unsafe_load(ss);
  });
  kswitchkeys.def("load",
                  [](KSwitchKeys& ksk, std::shared_ptr<SEALContext> context,
                     std::string& input) {
                    std::stringstream ss(input);
                    ksk.load(context, ss);
                  });
  kswitchkeys.def("pool", &KSwitchKeys::pool);
}
