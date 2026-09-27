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

add_subdirectory(proxy)

set(COMMON_INCLUDE_DIRS
    ${CMAKE_SOURCE_DIR}/src/device_arm
    ${CMAKE_SOURCE_DIR}/src/device_arm/proxy
    ${CMAKE_SOURCE_DIR}/src/common
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/store
    ${CMAKE_SOURCE_DIR}/include/runnerdb
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src/utils
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/header/external
    ${CMAKE_SOURCE_DIR}/third_party/CLI11
    ${YAMLCPP_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/third_party
    ${ASCEND_CANN_PACKAGE_PATH}/${CANN_ARCH_DIR}/include/driver
    ${ASCEND_CANN_PACKAGE_PATH}/${CANN_ARCH_DIR}/pkg_inc/driver
    ${ASCEND_CANN_PACKAGE_PATH}/${CANN_ARCH_DIR}/pkg_inc/runtime
    ${ASCEND_CANN_PACKAGE_PATH}/${CANN_ARCH_DIR}/pkg_inc/toolchain
    ${ASCEND_CANN_PACKAGE_PATH}/${CANN_ARCH_DIR}/pkg_inc/profiling
    ${ASCEND_CANN_PACKAGE_PATH}/${CANN_ARCH_DIR}/pkg_inc
    ${ASCEND_CANN_PACKAGE_PATH}/${CANN_ARCH_DIR}/pkg_inc/hccl
    ${ASCEND_CANN_PACKAGE_PATH}/${CANN_ARCH_DIR}/pkg_inc/base
    ${ASCEND_CANN_PACKAGE_PATH}/${CANN_ARCH_DIR}/include
    ${ASCEND_CANN_PACKAGE_PATH}/${CANN_ARCH_DIR}/include/hccl

    ${HCOMM_CODE_ROOT}/src/base_comm/resources/ccu/ccu_microcode
)

set(COMMON_SOURCES
    ${CMAKE_SOURCE_DIR}/src/utils/sim_log.cc
    ${CMAKE_SOURCE_DIR}/src/proxy/log_stub.cc
)

set(COMMON_LIBS
    gtest_main
    gtest
    log
    dl
    rt
    store
    runnerdb
    sqlite3
)

# 1. test_hccl_task_collection (10 TEST_F)
add_executable(test_hccl_task_collection hccl_task_collection_test.cc)

target_sources(test_hccl_task_collection PRIVATE
    ${CMAKE_SOURCE_DIR}/src/device_arm/hccl_task_collection.cc
    ${COMMON_SOURCES}
)

target_include_directories(test_hccl_task_collection PRIVATE ${COMMON_INCLUDE_DIRS})

target_link_libraries(test_hccl_task_collection PRIVATE ${COMMON_LIBS})

set_target_properties(test_hccl_task_collection PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)

# 2. test_hccl_device_pub (12 TEST_F)
add_executable(test_hccl_device_pub hccl_device_pub_test.cc)

target_sources(test_hccl_device_pub PRIVATE
    ${CMAKE_SOURCE_DIR}/src/device_arm/hccl_device_pub.cc
    ${CMAKE_SOURCE_DIR}/src/common/sim_pipe_io.cc
    ${CMAKE_SOURCE_DIR}/src/store/store_sim_memory_manager.cc
    ${CMAKE_SOURCE_DIR}/src/runnerdb/tracedb/db_sim_sqlite_db.cc
    ${COMMON_SOURCES}
)

target_include_directories(test_hccl_device_pub PRIVATE ${COMMON_INCLUDE_DIRS})

target_link_libraries(test_hccl_device_pub PRIVATE ${COMMON_LIBS})

set_target_properties(test_hccl_device_pub PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)

target_compile_options(test_hccl_device_pub PRIVATE
    -DCANN_VERSION_NUM=90000000
)
