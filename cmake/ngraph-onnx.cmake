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

set(EXTERNAL_NGRAPH_ONNX_INSTALL_DIR ${EXTERNAL_INSTALL_DIR})
set(NGRAPH_ONNX_CMAKE_PREFIX ext_ngraph_onnx)

set(NGRAPH_ONNX_REPO_URL https://github.com/NervanaSystems/ngraph-onnx.git)
set(NGRAPH_ONNX_GIT_LABEL master)

set(NGRAPH_ONNX_SRC_DIR
    ${CMAKE_BINARY_DIR}/${NGRAPH_ONNX_CMAKE_PREFIX}/src/${NGRAPH_ONNX_CMAKE_PREFIX}
)

ExternalProject_Add(ext_ngraph_onnx
      GIT_REPOSITORY ${NGRAPH_ONNX_REPO_URL}
      GIT_TAG ${NGRAPH_ONNX_GIT_LABEL}
      PREFIX ${NGRAPH_ONNX_CMAKE_PREFIX}
      CONFIGURE_COMMAND ""
      BUILD_COMMAND pip install -r ${NGRAPH_ONNX_SRC_DIR}/requirements.txt && pip install -e ${NGRAPH_ONNX_SRC_DIR}
      UPDATE_COMMAND ""
      INSTALL_COMMAND "")
