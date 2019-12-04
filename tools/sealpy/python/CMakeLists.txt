# ******************************************************************************
# Copyright 2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************

set(BUILD_SH_IN "${CMAKE_SOURCE_DIR}/python/build_wheel.sh.in")
set(BUILD_SH    "${CMAKE_BINARY_DIR}/python/build_wheel.sh")
configure_file(${BUILD_SH_IN} ${BUILD_SH} @ONLY)

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -fsanitize=address -fno-omit-frame-pointer")


#-----------------------------------------------------------------------------------------------
# Print Global Options
#-----------------------------------------------------------------------------------------------
message(STATUS "Compile Flags: ${CMAKE_CXX_FLAGS}")
message(STATUS "Shared Link Flags: ${CMAKE_SHARED_LINKER_FLAGS}")
message(STATUS "CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE}")
message(STATUS "CMAKE_CXX_FLAGS_DEBUG ${CMAKE_CXX_FLAGS_DEBUG}")

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/pyseal/pyseal.cpp.in
               ${CMAKE_CURRENT_SOURCE_DIR}/pyseal/pyseal.cpp @ONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in
               ${CMAKE_CURRENT_SOURCE_DIR}/setup.py @ONLY)


set(PYSEAL_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/pyseal/biguint.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/pyseal/ciphertext.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/pyseal/ckks.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/pyseal/coeff_modulus.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/pyseal/context_data.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/pyseal/decryptor.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/pyseal/encryption_parameters.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/pyseal/encryptor.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/pyseal/evaluator.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/pyseal/galoiskeys.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/pyseal/intencoder.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/pyseal/key_generator.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/pyseal/kswitchkeys.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/pyseal/memory_pool_handle.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/pyseal/plain_modulus.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/pyseal/plaintext.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/pyseal/public_key.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/pyseal/pyseal.cpp.in
        ${CMAKE_CURRENT_SOURCE_DIR}/pyseal/randomgen.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/pyseal/relinkeys.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/pyseal/rlwe.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/pyseal/seal_context.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/pyseal/secret_key.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/pyseal/small_modulus.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/pyseal/valcheck.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in)

add_custom_command(
        OUTPUT ${CMAKE_BINARY_DIR}/python/dist/
        POST_BUILD
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${PYSEAL_SRC}
        COMMAND make -C ../ DESTDIR=python/_install && bash build_wheel.sh
)

# TODO: add to ALL
add_custom_target(python_wheel DEPENDS ext_seal pybind11 ${CMAKE_BINARY_DIR}/python/dist/)
