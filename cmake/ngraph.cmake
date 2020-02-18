# ******************************************************************************
# Copyright 2018-2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.
# ******************************************************************************

include(ExternalProject)

set(EXTERNAL_NGRAPH_INSTALL_DIR ${EXTERNAL_INSTALL_DIR})
set(NGRAPH_CMAKE_PREFIX ext_ngraph)

set(NGRAPH_REPO_URL https://github.com/NervanaSystems/ngraph.git)
set(NGRAPH_GIT_LABEL v0.28.0-rc.1)

set(NGRAPH_SRC_DIR
    ${CMAKE_BINARY_DIR}/${NGRAPH_CMAKE_PREFIX}/src/${NGRAPH_CMAKE_PREFIX})
set(NGRAPH_BUILD_DIR ${NGRAPH_SRC_DIR}/build)

ExternalProject_Add(ext_ngraph
                    GIT_REPOSITORY ${NGRAPH_REPO_URL}
                    GIT_TAG ${NGRAPH_GIT_LABEL}
                    PREFIX ${NGRAPH_CMAKE_PREFIX}
                    CMAKE_ARGS {NGRAPH_HE_FORWARD_CMAKE_ARGS}
                      -DNGRAPH_UNIT_TEST_ENABLE=OFF
                      -DNGRAPH_GENERIC_CPU_ENABLE=OFF
                      -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_DIR}
                    UPDATE_COMMAND "")

ExternalProject_Get_Property(ext_ngraph SOURCE_DIR)
add_library(libngraph INTERFACE)
add_dependencies(libngraph ext_ngraph)
