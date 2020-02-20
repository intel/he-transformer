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


set(ABSEIL_REPO_URL https://github.com/abseil/abseil-cpp.git)
set(ABSEIL_GIT_TAG master)
set(ABSEIL_CXX_FLAGS "-O2 -Wformat -Wformat-security -D_FORTIFY_SOURCE=2 -fstack-protector-all -march=native")

# Using header-only, so no need to build
ExternalProject_Add(ext_abseil
                    PREFIX ext_abseil
                    GIT_REPOSITORY ${ABSEIL_REPO_URL}
                    GIT_TAG ${ABY_GIT_TAG}
                    CONFIGURE_COMMAND ""
                    BUILD_COMMAND ""
                    INSTALL_COMMAND ""
                    UPDATE_COMMAND ""
                    EXCLUDE_FROM_ALL TRUE)

 ExternalProject_Get_Property(ext_abseil SOURCE_DIR)
add_library(libabseil INTERFACE)
target_include_directories(libabseil SYSTEM INTERFACE ${SOURCE_DIR})
add_dependencies(libabseil ext_abseil)

set(abseil_INCLUDE_DIR ${SOURCE_DIR})
