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

#include "pyseal/memory_pool_handle.hpp"
#include "pyseal/pyseal.hpp"
#include "seal/seal.h"

using namespace seal;
using namespace pybind11::literals;
namespace py = pybind11;

void regclass_pyseal_MemoryPoolHandle(py::module& m) {
  py::class_<MemoryPoolHandle> memory_pool_handle(m, "MemoryPoolHandle");
  memory_pool_handle.def(py::init<>());
  memory_pool_handle.def(py::init<std::shared_ptr<util::MemoryPool>>());
  memory_pool_handle.def(py::init<const MemoryPoolHandle&>());
  memory_pool_handle.def_static("Global", &MemoryPoolHandle::Global);
  memory_pool_handle.def_static("ThreadLocal", &MemoryPoolHandle::ThreadLocal);
  memory_pool_handle.def_static("New",
                                py::overload_cast<bool>(&MemoryPoolHandle::New),
                                py::arg("clear_on_destruction") = false);
}
