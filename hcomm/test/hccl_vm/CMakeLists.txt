# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.14)
project(hccl_vm VERSION 1.0.0)
#set(CMAKE_VERBOSE_MAKEFILE true)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

function(hcclvm_apply_common_flags target)
    target_compile_options(${target} PRIVATE
        -Werror
        -fno-common
        -fno-strict-aliasing
        -pipe
        -O0
        -g
    )
    target_compile_definitions(${target} PRIVATE
        _GLIBCXX_USE_CXX11_ABI=0
    )
endfunction()

set(OUTPUT_DIR ${CMAKE_BINARY_DIR}/output)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${OUTPUT_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${OUTPUT_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${OUTPUT_DIR})

option(ENABLE_COVERAGE "Enable gcov/lcov coverage instrumentation for all targets" OFF)
if(ENABLE_COVERAGE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage -fprofile-arcs -ftest-coverage")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --coverage")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} --coverage")
    message(STATUS "Coverage instrumentation enabled: --coverage (all targets)")
endif()

option(BUILD_DEVICE_ARM "Build device_arm (ARM cross-compile)" OFF)
option(BUILD_A6_CCU_INSTR "Build A6 ccu instruction (ARM cross-compile)" ON)

if(BUILD_A6_CCU_INSTR)
    add_definitions(-DBUILD_A6_CCU_INSTR)
endif()

set(HCOMM_VM_ROOT_PATH ${CMAKE_CURRENT_SOURCE_DIR})

if(NOT DEFINED ASCEND_CANN_PACKAGE_PATH)
    if(NOT DEFINED ENV{ASCEND_TOOLKIT_HOME})
    message(FATAL_ERROR "Error: ASCEND_TOOLKIT_HOME environment variable is not set. Please set it to the Ascend Toolkit installation path.")
    endif()
    set(ASCEND_CANN_PACKAGE_PATH  $ENV{ASCEND_TOOLKIT_HOME})
endif()
message(STATUS "ASCEND_CANN_PACKAGE_PATH: ${ASCEND_CANN_PACKAGE_PATH}")

if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)")
    set(CANN_ARCH_DIR "aarch64-linux")
else()
    set(CANN_ARCH_DIR "x86_64-linux")
endif()
message(STATUS "CANN_ARCH_DIR: ${CANN_ARCH_DIR} (processor: ${CMAKE_SYSTEM_PROCESSOR})")

if(BUILD_DEVICE_ARM)
    if(NOT DEFINED HCCL_CODE_ROOT)
        if(NOT DEFINED ENV{HCCL_CODE_HOME})
            message(FATAL_ERROR "Error: BUILD_DEVICE_ARM is ON, but HCCL_CODE_HOME environment variable is not set. Please set it to the HCCL source path.")
        endif()
        set(HCCL_CODE_ROOT $ENV{HCCL_CODE_HOME})
    endif()
    message(STATUS "HCCL_CODE_ROOT: ${HCCL_CODE_ROOT}")
endif()

if(NOT DEFINED HCOMM_CODE_ROOT)
    if(NOT DEFINED ENV{HCOMM_CODE_HOME})
    message(FATAL_ERROR "Error: HCOMM_CODE_HOME environment variable is not set. Please set it to the Ascend Toolkit installation path.")
    endif()
    set(HCOMM_CODE_ROOT  $ENV{HCOMM_CODE_HOME})
endif()
message(STATUS "HCOMM_CODE_ROOT: ${HCOMM_CODE_ROOT}")

set(HCCL_VM_INCLUDE      ${CMAKE_SOURCE_DIR}/include)
set(HCCL_VM_SRC          ${CMAKE_SOURCE_DIR}/src)
set(HCCL_VM_INSTALL      ${CMAKE_SOURCE_DIR}/hccl_vm_install)
set(THRID_PARTY_DIR      ${CMAKE_SOURCE_DIR}/third_party)
message(STATUS "HCOMM_CODE_ROOT: ${HCOMM_CODE_ROOT}")

set(HCOMM_CODE_PLATFORM  ${HCOMM_CODE_ROOT}/src/platform)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/hccl_vm_install" CACHE PATH "Default install prefix" FORCE)
    message(STATUS "Install prefix not set by user, defaulting to: ${CMAKE_INSTALL_PREFIX}")
endif()

set(CMAKE_BUILD_RPATH "$ORIGIN")
set(CMAKE_INSTALL_RPATH "$ORIGIN")
set(CMAKE_BUILD_RPATH_USE_ORIGIN TRUE)

include(cmake/third_party/yaml-cpp.cmake)
include(cmake/third_party/nlohmann_json.cmake)
include(cmake/third_party/cli11.cmake)
include(cmake/third_party/spdlog.cmake)
include(cmake/third_party/sqlite3.cmake)
include(cmake/third_party/cann_cmake.cmake)

include_directories(
    ${CMAKE_SOURCE_DIR}/third_party/nlohmann_json
    ${CMAKE_SOURCE_DIR}/third_party
    ${CMAKE_SOURCE_DIR}/third_party/yaml-cpp/include
)

function(config_hcclvm_install)
    install(DIRECTORY DESTINATION "data")
    install(DIRECTORY DESTINATION "archive")
    install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/config/"
        DESTINATION "config")
    install(FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/asset/allocate_eid.sh"
        "${CMAKE_CURRENT_SOURCE_DIR}/asset/generate_cluster_topo.sh"
        "${CMAKE_CURRENT_SOURCE_DIR}/asset/generate_server_topo.sh"
        "${CMAKE_CURRENT_SOURCE_DIR}/asset/hccl_config.sh"
        "${CMAKE_CURRENT_SOURCE_DIR}/asset/archive.sh"
        DESTINATION "script"
        PERMISSIONS
            OWNER_READ OWNER_WRITE OWNER_EXECUTE
            GROUP_READ GROUP_EXECUTE
            WORLD_READ WORLD_EXECUTE
    )
    install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/asset/cluster_model/config/cluster"
        DESTINATION "config"
        FILE_PERMISSIONS
            OWNER_READ OWNER_WRITE
            GROUP_READ
            WORLD_READ
        DIRECTORY_PERMISSIONS
            OWNER_READ OWNER_WRITE OWNER_EXECUTE
            GROUP_READ GROUP_EXECUTE
            WORLD_READ WORLD_EXECUTE
    )
    install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/asset/cluster_model/config/server_or_pod"
        DESTINATION "config"
        FILE_PERMISSIONS
            OWNER_READ OWNER_WRITE
            GROUP_READ
            WORLD_READ
        DIRECTORY_PERMISSIONS
            OWNER_READ OWNER_WRITE OWNER_EXECUTE
            GROUP_READ GROUP_EXECUTE
            WORLD_READ WORLD_EXECUTE
    )
    install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/asset/cluster_model/topo_meta"
        DESTINATION "config"
        FILE_PERMISSIONS
            OWNER_READ OWNER_WRITE
            GROUP_READ
            WORLD_READ
        DIRECTORY_PERMISSIONS
            OWNER_READ OWNER_WRITE OWNER_EXECUTE
            GROUP_READ GROUP_EXECUTE
            WORLD_READ WORLD_EXECUTE
    )
endfunction()

# 编译HCCL-VM
if (NOT BUILD_DEVICE_ARM)
    config_hcclvm_install()
    add_subdirectory(src)
endif()

set(CPACK_PACKAGE_NAME "hccl_vm")
set(CPACK_GENERATOR "TGZ")
include(CPack)

# 编译DEVICE-ARM-交叉编译版本
if(BUILD_DEVICE_ARM)
    message(STATUS "BUILD_DEVICE_ARM start")
    set(DEVICE_ARM_BUILD_DIR ${CMAKE_SOURCE_DIR}/build_device)
    set(DEVICE_ARM_INSTALL_DIR ${CMAKE_SOURCE_DIR}/hccl_vm_install)
    include(ExternalProject)
    ExternalProject_Add(device_arm_external
        SOURCE_DIR ${CMAKE_SOURCE_DIR}/src/device_arm
        CMAKE_ARGS
            -DCMAKE_INSTALL_PREFIX=${DEVICE_ARM_INSTALL_DIR}
            -DHCCL_VM_SRC=${HCCL_VM_SRC}
            -DTHRID_PARTY_DIR=${THRID_PARTY_DIR}
            -DHCCL_VM_INCLUDE=${HCCL_VM_INCLUDE}
            -DASCEND_CANN_PACKAGE_PATH=${ASCEND_CANN_PACKAGE_PATH}
            -DCANN_ARCH_DIR=${CANN_ARCH_DIR}
            -DHCOMM_CODE_ROOT=${HCOMM_CODE_ROOT}
            -DHCCL_CODE_ROOT=${HCCL_CODE_ROOT}

        BINARY_DIR ${DEVICE_ARM_BUILD_DIR}
        INSTALL_DIR ${DEVICE_ARM_INSTALL_DIR}
        BUILD_ALWAYS ON
    )

    message(STATUS "device_arm cross-compilation enabled")
    message(STATUS "  Build dir: ${DEVICE_ARM_BUILD_DIR}")
    message(STATUS "  Install dir: ${DEVICE_ARM_INSTALL_DIR}")
endif()

option(BUILD_TESTS "Build unit tests" OFF)
if(BUILD_TESTS)
    include(cmake/third_party/gtest.cmake)
    add_subdirectory(test)
endif()
