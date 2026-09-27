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

unset(gtest_FOUND CACHE)

set(GTEST_FILE "googletest-1.14.0.tar.gz")
set(GTEST_URL "https://gitcode.com/cann-src-third-party/googletest/releases/download/v1.14.0/${GTEST_FILE}")
set(GTEST_PKG_PATH ${CMAKE_SOURCE_DIR}/third_party/${GTEST_FILE})
set(GTEST_INSTALL_PATH ${CMAKE_SOURCE_DIR}/third_party/gtest)
set(GTEST_INCLUDE_DIR ${GTEST_INSTALL_PATH}/include)

set(GTEST_LIB_DIR "")
if(EXISTS "${GTEST_INSTALL_PATH}/lib64/libgtest.a")
    set(GTEST_LIB_DIR "${GTEST_INSTALL_PATH}/lib64")
elseif(EXISTS "${GTEST_INSTALL_PATH}/lib/libgtest.a")
    set(GTEST_LIB_DIR "${GTEST_INSTALL_PATH}/lib")
endif()

message(STATUS "[ThirdParty] GTEST_INSTALL_PATH=${GTEST_INSTALL_PATH}")
message(STATUS "[ThirdParty] GTEST_LIB_DIR=${GTEST_LIB_DIR}")

if(GTEST_LIB_DIR AND EXISTS "${GTEST_INCLUDE_DIR}/gtest/gtest.h")
    set(gtest_FOUND TRUE)
    message(STATUS "[ThirdParty] GTest found in ${GTEST_INSTALL_PATH}")
else()
    set(gtest_FOUND FALSE)
    message(STATUS "[ThirdParty] GTest not found, will build from source")
endif()

if(NOT gtest_FOUND)
    if(EXISTS ${GTEST_PKG_PATH})
        message(STATUS "[ThirdParty] Found local gtest package: ${GTEST_PKG_PATH}")
        set(GTEST_PROJECT_URL ${GTEST_PKG_PATH})
    else()
        message(STATUS "[ThirdParty] Downloading GTest from ${GTEST_URL}")
        set(GTEST_PROJECT_URL ${GTEST_URL})
    endif()

    set(GTEST_CXXFLAGS "-D_GLIBCXX_USE_CXX11_ABI=0 -O2 -D_FORTIFY_SOURCE=2 -fPIC -fstack-protector-all -Wl,-z,relro,-z,now,-z,noexecstack")
    set(GTEST_CFLAGS   "-D_GLIBCXX_USE_CXX11_ABI=0 -O2 -D_FORTIFY_SOURCE=2 -fPIC -fstack-protector-all -Wl,-z,relro,-z,now,-z,noexecstack")

    set(GTEST_OPTS
        -DCMAKE_CXX_FLAGS=${GTEST_CXXFLAGS}
        -DCMAKE_C_FLAGS=${GTEST_CFLAGS}
        -DCMAKE_INSTALL_PREFIX=${GTEST_INSTALL_PATH}
        -DCMAKE_INSTALL_LIBDIR=lib64
        -DBUILD_TESTING=OFF
        -DBUILD_SHARED_LIBS=OFF
    )

    include(ExternalProject)
    ExternalProject_Add(third_party_gtest
        URL ${GTEST_PROJECT_URL}
        URL_HASH SHA256=8ad598c73ad796e0d8280b082cebd82a630d73e73cd3c70057938a6501bba5d7
        DOWNLOAD_DIR ${CMAKE_SOURCE_DIR}/third_party
        DOWNLOAD_NO_PROGRESS TRUE
        CONFIGURE_COMMAND ${CMAKE_COMMAND} ${GTEST_OPTS} <SOURCE_DIR>
        BUILD_COMMAND ${CMAKE_MAKE_PROGRAM}
        INSTALL_COMMAND ${CMAKE_MAKE_PROGRAM} install
        EXCLUDE_FROM_ALL TRUE
    )
    set(GTEST_LIB_DIR "${GTEST_INSTALL_PATH}/lib64")
endif()

if(NOT EXISTS ${GTEST_INCLUDE_DIR})
    file(MAKE_DIRECTORY "${GTEST_INCLUDE_DIR}")
endif()

add_library(gtest STATIC IMPORTED GLOBAL)
if(TARGET third_party_gtest)
    add_dependencies(gtest third_party_gtest)
endif()

add_library(gmock STATIC IMPORTED GLOBAL)
if(TARGET third_party_gtest)
    add_dependencies(gmock third_party_gtest)
endif()

add_library(gtest_main STATIC IMPORTED GLOBAL)
if(TARGET third_party_gtest)
    add_dependencies(gtest_main third_party_gtest)
endif()

add_library(gmock_main STATIC IMPORTED GLOBAL)
if(TARGET third_party_gtest)
    add_dependencies(gmock_main third_party_gtest)
endif()

set_target_properties(gtest PROPERTIES
    IMPORTED_LOCATION ${GTEST_LIB_DIR}/libgtest.a
    INTERFACE_INCLUDE_DIRECTORIES ${GTEST_INCLUDE_DIR}
)

set_target_properties(gmock PROPERTIES
    IMPORTED_LOCATION ${GTEST_LIB_DIR}/libgmock.a
    INTERFACE_INCLUDE_DIRECTORIES ${GTEST_INCLUDE_DIR}
)

set_target_properties(gtest_main PROPERTIES
    IMPORTED_LOCATION ${GTEST_LIB_DIR}/libgtest_main.a
    INTERFACE_INCLUDE_DIRECTORIES ${GTEST_INCLUDE_DIR}
)

set_target_properties(gmock_main PROPERTIES
    IMPORTED_LOCATION ${GTEST_LIB_DIR}/libgmock_main.a
    INTERFACE_INCLUDE_DIRECTORIES ${GTEST_INCLUDE_DIR}
)
