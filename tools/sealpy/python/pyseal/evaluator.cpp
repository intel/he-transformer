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

#include "pyseal/evaluator.hpp"
#include "pyseal/pyseal.hpp"
#include "seal/seal.h"

using namespace seal;
using namespace pybind11::literals;
namespace py = pybind11;

void regclass_pyseal_Evaluator(py::module& m) {
  // TODO: finish and test default pool
  py::class_<Evaluator> evaluator(m, "Evaluator");
  evaluator.def(py::init<std::shared_ptr<SEALContext>>());
  evaluator.def("negate_inplace", &Evaluator::negate_inplace);
  evaluator.def("negate", &Evaluator::negate);
  evaluator.def("add_inplace", &Evaluator::add_inplace);
  evaluator.def("add", &Evaluator::add);
  evaluator.def("add_many", &Evaluator::add_many);
  evaluator.def("sub_inplace", &Evaluator::sub_inplace);
  evaluator.def("sub", &Evaluator::sub);
  evaluator.def(
      "multiply_inplace",
      (void (Evaluator::*)(Ciphertext&, const Ciphertext&, MemoryPoolHandle)) &
          Evaluator::multiply_inplace,
      py::arg("encrypted1"), py::arg("encrypted2"),
      py::arg("pool") = MemoryManager::GetPool());
  evaluator.def("multiply",
                (void (Evaluator::*)(const Ciphertext&, const Ciphertext&,
                                     Ciphertext&, MemoryPoolHandle)) &
                    Evaluator::multiply,
                py::arg("encrypted1"), py::arg("encrypted2"),
                py::arg("destination"),
                py::arg("pool") = MemoryManager::GetPool());
  evaluator.def("square_inplace",
                (void (Evaluator::*)(Ciphertext&, MemoryPoolHandle)) &
                    Evaluator::square_inplace,
                py::arg("encrypted"),
                py::arg("pool") = MemoryManager::GetPool());
  evaluator.def(
      "square",
      (void (Evaluator::*)(const Ciphertext&, Ciphertext&, MemoryPoolHandle)) &
          Evaluator::square,
      py::arg("encrypted"), py::arg("destination"),
      py::arg("pool") = MemoryManager::GetPool());
  evaluator.def(
      "relinearize_inplace",
      (void (Evaluator::*)(Ciphertext&, const RelinKeys&, MemoryPoolHandle)) &
          Evaluator::relinearize_inplace,
      py::arg("encrypted"), py::arg("relin_keys"),
      py::arg("pool") = MemoryManager::GetPool());
  evaluator.def("relinearize",
                (void (Evaluator::*)(const Ciphertext&, const RelinKeys&,
                                     Ciphertext&, MemoryPoolHandle)) &
                    Evaluator::relinearize,
                py::arg("encrypted"), py::arg("relin_keys"),
                py::arg("destination"),
                py::arg("pool") = MemoryManager::GetPool());
  evaluator.def(
      "mod_switch_to_next",
      (void (Evaluator::*)(const Ciphertext&, Ciphertext&, MemoryPoolHandle)) &
          Evaluator::mod_switch_to_next,
      py::arg("encrypted"), py::arg("destination"),
      py::arg("pool") = MemoryManager::GetPool());
  evaluator.def("mod_switch_to_next_inplace",
                (void (Evaluator::*)(Ciphertext&, MemoryPoolHandle)) &
                    Evaluator::mod_switch_to_next_inplace,
                py::arg("encrypted"),
                py::arg("pool") = MemoryManager::GetPool());
  evaluator.def("mod_switch_to_next_inplace",
                (void (Evaluator::*)(Plaintext&)) &
                    Evaluator::mod_switch_to_next_inplace);
  evaluator.def("mod_switch_to_next",
                (void (Evaluator::*)(const Plaintext&, Plaintext&)) &
                    Evaluator::mod_switch_to_next);
  evaluator.def(
      "mod_switch_to_inplace",
      (void (Evaluator::*)(Ciphertext&, parms_id_type, MemoryPoolHandle)) &
          Evaluator::mod_switch_to_inplace,
      py::arg("encrypted"), py::arg("parms_id"),
      py::arg("pool") = MemoryManager::GetPool());
  evaluator.def("mod_switch_to",
                (void (Evaluator::*)(const Ciphertext&, parms_id_type,
                                     Ciphertext&, MemoryPoolHandle)) &
                    Evaluator::mod_switch_to,
                py::arg("encrypted"), py::arg("parms_id"),
                py::arg("destination"),
                py::arg("pool") = MemoryManager::GetPool());
  evaluator.def("mod_switch_to_inplace",
                (void (Evaluator::*)(Plaintext&, parms_id_type)) &
                    Evaluator::mod_switch_to_inplace);
  evaluator.def(
      "mod_switch_to",
      (void (Evaluator::*)(const Plaintext&, parms_id_type, Plaintext&)) &
          Evaluator::mod_switch_to);
  evaluator.def(
      "rescale_to_next",
      (void (Evaluator::*)(const Ciphertext&, Ciphertext&, MemoryPoolHandle)) &
          Evaluator::rescale_to_next,
      py::arg("encrypted"), py::arg("destination"),
      py::arg("pool") = MemoryManager::GetPool());
  evaluator.def("rescale_to_next_inplace",
                (void (Evaluator::*)(Ciphertext&, MemoryPoolHandle)) &
                    Evaluator::rescale_to_next_inplace,
                py::arg("encrypted"),
                py::arg("pool") = MemoryManager::GetPool());
  evaluator.def(
      "rescale_to_inplace",
      (void (Evaluator::*)(Ciphertext&, parms_id_type, MemoryPoolHandle)) &
          Evaluator::rescale_to_inplace,
      py::arg("encrypted"), py::arg("parms_id"),
      py::arg("pool") = MemoryManager::GetPool());
  evaluator.def("rescale_to",
                (void (Evaluator::*)(const Ciphertext&, parms_id_type,
                                     Ciphertext&, MemoryPoolHandle)) &
                    Evaluator::rescale_to,
                py::arg("encrypted"), py::arg("parms_id"),
                py::arg("destination"),
                py::arg("pool") = MemoryManager::GetPool());
  evaluator.def("multiply_many",
                (void (Evaluator::*)(std::vector<Ciphertext>&, const RelinKeys&,
                                     Ciphertext&, MemoryPoolHandle)) &
                    Evaluator::multiply_many,
                py::arg("encrypteds"), py::arg("relin_keys"),
                py::arg("destination"),
                py::arg("pool") = MemoryManager::GetPool());
  evaluator.def("exponentiate_inplace",
                (void (Evaluator::*)(Ciphertext&, std::uint64_t,
                                     const RelinKeys&, MemoryPoolHandle)) &
                    Evaluator::exponentiate_inplace,
                py::arg("encrypted"), py::arg("exponent"),
                py::arg("relin_keys"),
                py::arg("pool") = MemoryManager::GetPool());
  evaluator.def(
      "exponentiate",
      (void (Evaluator::*)(const Ciphertext&, std::uint64_t, const RelinKeys&,
                           Ciphertext&, MemoryPoolHandle)) &
          Evaluator::exponentiate,
      py::arg("encrypted"), py::arg("exponent"), py::arg("relin_keys"),
      py::arg("destination"), py::arg("pool") = MemoryManager::GetPool());
  evaluator.def("add_plain_inplace", &Evaluator::add_plain_inplace);
  evaluator.def("add_plain", &Evaluator::add_plain);
  evaluator.def("sub_plain_inplace", &Evaluator::sub_plain_inplace);
  evaluator.def("sub_plain", &Evaluator::sub_plain);
  evaluator.def(
      "multiply_plain_inplace",
      (void (Evaluator::*)(Ciphertext&, const Plaintext&, MemoryPoolHandle)) &
          Evaluator::multiply_plain_inplace,
      py::arg("encrypted"), py::arg("plain"),
      py::arg("pool") = MemoryManager::GetPool());
  evaluator.def("multiply_plain",
                (void (Evaluator::*)(const Ciphertext&, const Plaintext&,
                                     Ciphertext&, MemoryPoolHandle)) &
                    Evaluator::multiply_plain,
                py::arg("encrypted"), py::arg("plain"), py::arg("destination"),
                py::arg("pool") = MemoryManager::GetPool());
  evaluator.def(
      "transform_to_ntt_inplace",
      (void (Evaluator::*)(Plaintext&, parms_id_type, MemoryPoolHandle)) &
          Evaluator::transform_to_ntt_inplace,
      py::arg("plain"), py::arg("parms_id"),
      py::arg("pool") = MemoryManager::GetPool());
  evaluator.def("transform_to_ntt",
                (void (Evaluator::*)(const Plaintext&, parms_id_type,
                                     Plaintext&, MemoryPoolHandle)) &
                    Evaluator::transform_to_ntt,
                py::arg("plain"), py::arg("parms_id"), py::arg("destination"),
                py::arg("pool") = MemoryManager::GetPool());
  evaluator.def(
      "transform_to_ntt_inplace",
      (void (Evaluator::*)(Ciphertext&)) & Evaluator::transform_to_ntt_inplace);
  evaluator.def("transform_to_ntt",
                (void (Evaluator::*)(const Ciphertext&, Ciphertext&)) &
                    Evaluator::transform_to_ntt);
  evaluator.def("transform_from_ntt_inplace",
                &Evaluator::transform_from_ntt_inplace);
  evaluator.def("transform_from_ntt", &Evaluator::transform_from_ntt);
  evaluator.def("apply_galois_inplace",
                (void (Evaluator::*)(Ciphertext&, std::uint64_t,
                                     const GaloisKeys&, MemoryPoolHandle)) &
                    Evaluator::apply_galois_inplace,
                py::arg("encrypted"), py::arg("galois_elt"),
                py::arg("galois_keys"),
                py::arg("pool") = MemoryManager::GetPool());
  evaluator.def(
      "apply_galois",
      (void (Evaluator::*)(const Ciphertext&, std::uint64_t, const GaloisKeys&,
                           Ciphertext&, MemoryPoolHandle)) &
          Evaluator::apply_galois,
      py::arg("encrypted"), py::arg("galois_elt"), py::arg("galois_keys"),
      py::arg("destination"), py::arg("pool") = MemoryManager::GetPool());
  evaluator.def("rotate_rows_inplace",
                (void (Evaluator::*)(Ciphertext&, int, const GaloisKeys&,
                                     MemoryPoolHandle)) &
                    Evaluator::rotate_rows_inplace,
                py::arg("encrypted"), py::arg("steps"), py::arg("galois_keys"),
                py::arg("pool") = MemoryManager::GetPool());
  evaluator.def("rotate_rows",
                (void (Evaluator::*)(const Ciphertext&, int, const GaloisKeys&,
                                     Ciphertext&, MemoryPoolHandle)) &
                    Evaluator::rotate_rows,
                py::arg("encrypted"), py::arg("steps"), py::arg("galois_keys"),
                py::arg("destination"),
                py::arg("pool") = MemoryManager::GetPool());
  evaluator.def(
      "rotate_columns_inplace",
      (void (Evaluator::*)(Ciphertext&, const GaloisKeys&, MemoryPoolHandle)) &
          Evaluator::rotate_columns_inplace,
      py::arg("encrypted"), py::arg("galois_keys"),
      py::arg("pool") = MemoryManager::GetPool());
  evaluator.def("rotate_columns",
                (void (Evaluator::*)(const Ciphertext&, const GaloisKeys&,
                                     Ciphertext&, MemoryPoolHandle)) &
                    Evaluator::rotate_columns,
                py::arg("encrypted"), py::arg("galois_keys"),
                py::arg("destination"),
                py::arg("pool") = MemoryManager::GetPool());
  evaluator.def("rotate_vector_inplace",
                (void (Evaluator::*)(Ciphertext&, int, const GaloisKeys&,
                                     MemoryPoolHandle)) &
                    Evaluator::rotate_vector_inplace,
                py::arg("encrypted"), py::arg("steps"), py::arg("galois_keys"),
                py::arg("pool") = MemoryManager::GetPool());
  evaluator.def("rotate_vector",
                (void (Evaluator::*)(const Ciphertext&, int, const GaloisKeys&,
                                     Ciphertext&, MemoryPoolHandle)) &
                    Evaluator::rotate_vector,
                py::arg("encrypted"), py::arg("steps"), py::arg("galois_keys"),
                py::arg("destination"),
                py::arg("pool") = MemoryManager::GetPool());
  evaluator.def(
      "complex_conjugate_inplace",
      (void (Evaluator::*)(Ciphertext&, const GaloisKeys&, MemoryPoolHandle)) &
          Evaluator::complex_conjugate_inplace,
      py::arg("encrypted"), py::arg("galois_keys"),
      py::arg("pool") = MemoryManager::GetPool());
  evaluator.def("complex_conjugate",
                (void (Evaluator::*)(const Ciphertext&, const GaloisKeys&,
                                     Ciphertext&, MemoryPoolHandle)) &
                    Evaluator::complex_conjugate,
                py::arg("encrypted"), py::arg("galois_keys"),
                py::arg("destination"),
                py::arg("pool") = MemoryManager::GetPool());
}
