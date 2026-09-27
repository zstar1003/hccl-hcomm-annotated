# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(DISP_PROBE_THIRDLIB_INCLUDE_DIR "${THIRDLIB_ROOT}/include")
set(DISP_PROBE_THIRDLIB_LIBRARY_DIR "${THIRDLIB_ROOT}/lib")

include_directories(SYSTEM
    "${DISP_PROBE_THIRDLIB_INCLUDE_DIR}"
    "${DISP_PROBE_THIRDLIB_INCLUDE_DIR}/eigen3")

function(disp_probe_require_third_party_path required_path description)
    if(NOT EXISTS "${required_path}")
        message(FATAL_ERROR
            "Missing ${description}: ${required_path}\n"
            "Run ./install.sh --prefix ${THIRDLIB_ROOT} before configuring this project.")
    endif()
endfunction()

function(disp_probe_add_third_party_include target include_dir)
    target_include_directories(${target} INTERFACE "${include_dir}")
    target_compile_options(${target} INTERFACE "$<$<COMPILE_LANGUAGE:CXX>:-I${include_dir}>")
endfunction()

disp_probe_require_third_party_path(
    "${DISP_PROBE_THIRDLIB_INCLUDE_DIR}/CLI/CLI.hpp"
    "CLI11 header")
disp_probe_require_third_party_path(
    "${DISP_PROBE_THIRDLIB_INCLUDE_DIR}/nlohmann/json.hpp"
    "nlohmann_json header")
disp_probe_require_third_party_path(
    "${DISP_PROBE_THIRDLIB_INCLUDE_DIR}/eigen3/Eigen/Dense"
    "Eigen headers")
disp_probe_require_third_party_path(
    "${DISP_PROBE_THIRDLIB_INCLUDE_DIR}/rpc/client.h"
    "rpclib headers")
disp_probe_require_third_party_path(
    "${DISP_PROBE_THIRDLIB_LIBRARY_DIR}/librpc.a"
    "rpclib static library")

add_library(CLI INTERFACE)
disp_probe_add_third_party_include(CLI "${DISP_PROBE_THIRDLIB_INCLUDE_DIR}")
target_compile_features(CLI INTERFACE cxx_std_14)

add_library(nlohmann_json INTERFACE)
disp_probe_add_third_party_include(nlohmann_json "${DISP_PROBE_THIRDLIB_INCLUDE_DIR}")
target_compile_features(nlohmann_json INTERFACE cxx_std_17)

add_library(Eigen INTERFACE)
disp_probe_add_third_party_include(Eigen "${DISP_PROBE_THIRDLIB_INCLUDE_DIR}/eigen3")
target_compile_features(Eigen INTERFACE cxx_std_17)

add_library(rpc STATIC IMPORTED GLOBAL)
set_target_properties(rpc PROPERTIES
    IMPORTED_LOCATION "${DISP_PROBE_THIRDLIB_LIBRARY_DIR}/librpc.a")
disp_probe_add_third_party_include(rpc "${DISP_PROBE_THIRDLIB_INCLUDE_DIR}")
target_compile_features(rpc INTERFACE cxx_std_17)

if(EXISTS "${DISP_PROBE_THIRDLIB_INCLUDE_DIR}/fmt/format.h")
    add_library(fmt INTERFACE)
    disp_probe_add_third_party_include(fmt "${DISP_PROBE_THIRDLIB_INCLUDE_DIR}")
    target_compile_features(fmt INTERFACE cxx_std_11)
endif()

if(EXISTS "${DISP_PROBE_THIRDLIB_INCLUDE_DIR}/spdlog/spdlog.h")
    add_library(spdlog INTERFACE)
    disp_probe_add_third_party_include(spdlog "${DISP_PROBE_THIRDLIB_INCLUDE_DIR}")
    target_compile_features(spdlog INTERFACE cxx_std_11)
    if(TARGET fmt)
        target_link_libraries(spdlog INTERFACE fmt)
    endif()
endif()
