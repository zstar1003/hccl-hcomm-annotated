# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

include_guard(GLOBAL)

unset(yaml_cpp_FOUND CACHE)

set(YAMLCPP_FILE "yaml-cpp-0.8.0.tar.gz")
set(YAMLCPP_URL "https://raw.gitcode.com/src-openeuler/yaml-cpp/blobs/d1ead4fff417073b9cdbf98b8b55eb0efc00b0ba/yaml-cpp-0.8.0.tar.gz")
set(YAMLCPP_PKG_PATH ${CMAKE_SOURCE_DIR}/third_party/${YAMLCPP_FILE})
set(YAMLCPP_INSTALL_PATH ${CMAKE_SOURCE_DIR}/third_party/yaml-cpp)
set(YAMLCPP_INCLUDE_DIR ${YAMLCPP_INSTALL_PATH}/include)

set(YAMLCPP_LIB_DIR "")
if(EXISTS "${YAMLCPP_INSTALL_PATH}/lib64/libyaml-cpp.a")
    set(YAMLCPP_LIB_DIR "${YAMLCPP_INSTALL_PATH}/lib64")
elseif(EXISTS "${YAMLCPP_INSTALL_PATH}/lib/libyaml-cpp.a")
    set(YAMLCPP_LIB_DIR "${YAMLCPP_INSTALL_PATH}/lib")
endif()

message(STATUS "[ThirdParty] YAMLCPP_INSTALL_PATH=${YAMLCPP_INSTALL_PATH}")
message(STATUS "[ThirdParty] YAMLCPP_LIB_DIR=${YAMLCPP_LIB_DIR}")

if(YAMLCPP_LIB_DIR AND EXISTS "${YAMLCPP_INCLUDE_DIR}/yaml-cpp/yaml.h")
    set(yaml_cpp_FOUND TRUE)
    message(STATUS "[ThirdParty] yaml-cpp found in ${YAMLCPP_INSTALL_PATH}")
else()
    set(yaml_cpp_FOUND FALSE)
    message(STATUS "[ThirdParty] yaml-cpp not found, will build from source")
endif()

if(NOT yaml_cpp_FOUND)
    if(EXISTS ${YAMLCPP_PKG_PATH})
        message(STATUS "[ThirdParty] Found local yaml-cpp package: ${YAMLCPP_PKG_PATH}")
        set(YAMLCPP_PROJECT_URL ${YAMLCPP_PKG_PATH})
    else()
        message(STATUS "[ThirdParty] Downloading yaml-cpp from ${YAMLCPP_URL}")
        set(YAMLCPP_PROJECT_URL ${YAMLCPP_URL})
    endif()

    set(YAMLCPP_CXXFLAGS "-D_GLIBCXX_USE_CXX11_ABI=0 -O2 -D_FORTIFY_SOURCE=2 -fPIC -fstack-protector-all -Wl,-z,relro,-z,now,-z,noexecstack")
    set(YAMLCPP_CFLAGS   "-D_GLIBCXX_USE_CXX11_ABI=0 -O2 -D_FORTIFY_SOURCE=2 -fPIC -fstack-protector-all -Wl,-z,relro,-z,now,-z,noexecstack")

    set(YAMLCPP_OPTS
        -DCMAKE_CXX_FLAGS=${YAMLCPP_CXXFLAGS}
        -DCMAKE_C_FLAGS=${YAMLCPP_CFLAGS}
        -DCMAKE_INSTALL_PREFIX=${YAMLCPP_INSTALL_PATH}
        -DCMAKE_INSTALL_LIBDIR=lib64
        -DCMAKE_CXX_STANDARD=17
        -DYAML_CPP_BUILD_TESTS=OFF
        -DYAML_CPP_BUILD_TOOLS=OFF
        -DYAML_CPP_BUILD_CONTRIB=OFF
        -DBUILD_SHARED_LIBS=OFF
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5
    )

    include(ExternalProject)
    ExternalProject_Add(third_party_yaml_cpp
        URL ${YAMLCPP_PROJECT_URL}
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        DOWNLOAD_DIR ${CMAKE_SOURCE_DIR}/third_party
        DOWNLOAD_NO_PROGRESS TRUE
        CONFIGURE_COMMAND ${CMAKE_COMMAND} ${YAMLCPP_OPTS} <SOURCE_DIR>
        BUILD_COMMAND ${CMAKE_MAKE_PROGRAM}
        INSTALL_COMMAND ${CMAKE_MAKE_PROGRAM} install
        EXCLUDE_FROM_ALL TRUE
    )
    set(YAMLCPP_LIB_DIR "${YAMLCPP_INSTALL_PATH}/lib64")
endif()

if(NOT EXISTS ${YAMLCPP_INCLUDE_DIR})
    file(MAKE_DIRECTORY "${YAMLCPP_INCLUDE_DIR}")
endif()

add_library(yaml-cpp::yaml-cpp STATIC IMPORTED GLOBAL)
if(TARGET third_party_yaml_cpp)
    add_dependencies(yaml-cpp::yaml-cpp third_party_yaml_cpp)
endif()

add_library(yaml-cpp STATIC IMPORTED GLOBAL)
if(TARGET third_party_yaml_cpp)
    add_dependencies(yaml-cpp third_party_yaml_cpp)
endif()

set_target_properties(yaml-cpp::yaml-cpp PROPERTIES
    IMPORTED_LOCATION ${YAMLCPP_LIB_DIR}/libyaml-cpp.a
    INTERFACE_INCLUDE_DIRECTORIES ${YAMLCPP_INCLUDE_DIR}
)

set_target_properties(yaml-cpp PROPERTIES
    IMPORTED_LOCATION ${YAMLCPP_LIB_DIR}/libyaml-cpp.a
    INTERFACE_INCLUDE_DIRECTORIES ${YAMLCPP_INCLUDE_DIR}
)
