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

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -D_GLIBCXX_USE_CXX11_ABI=0")

# HCOMM_CODE_ROOT and ASCEND_CANN_PACKAGE_PATH are set by root CMakeLists.txt

include_directories(
    ${HCOMM_CODE_ROOT}/src/base_comm/resources/ccu/ccu_microcode

    ${CMAKE_SOURCE_DIR}/src/store
    ${CMAKE_SOURCE_DIR}/include/store
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/runnerdb
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/third_party/nlohmann_json
    ${YAMLCPP_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/third_party/nlohmann_json
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/common
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/header/external
)

if(DEFINED HCOMM_CODE_ROOT)
    include_directories(
        ${HCOMM_CODE_ROOT}/src/base_comm/resources/ccu/ccu_microcode

        ${HCOMM_CODE_ROOT}/include
        ${HCOMM_CODE_ROOT}/include/hccl
    )
endif()
include_directories(
    ${ASCEND_CANN_PACKAGE_PATH}/${CANN_ARCH_DIR}/include
    ${ASCEND_CANN_PACKAGE_PATH}/${CANN_ARCH_DIR}/include/hccl
)

link_libraries(
    gtest_main
    gtest
)

add_executable(test_shm_ops sim_shm_ops_test.cc)
target_link_libraries(test_shm_ops PRIVATE store sqlite3 log)
set_target_properties(test_shm_ops PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)

add_executable(test_shm_head shm_head_test.cc)
target_link_libraries(test_shm_head PRIVATE store sqlite3 log)
set_target_properties(test_shm_head PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)

add_executable(test_sim_memory_manager sim_memory_manager_test.cc)
target_link_libraries(test_sim_memory_manager PRIVATE pthread store sqlite3 log)
set_target_properties(test_sim_memory_manager PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)

add_executable(test_binary_data_operator binary_data_operator_test.cc)
target_sources(test_binary_data_operator PRIVATE
    ${CMAKE_SOURCE_DIR}/src/store/store_binary_data_operator.cc
)
target_link_libraries(test_binary_data_operator PRIVATE log)
set_target_properties(test_binary_data_operator PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)

add_executable(test_dump_shm_data dump_shm_data_test.cc)
target_link_libraries(test_dump_shm_data PRIVATE store runnerdb sqlite3 log)
set_target_properties(test_dump_shm_data PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)

add_executable(test_sim_memory_manager_full sim_memory_manager_full_test.cc)
target_link_libraries(test_sim_memory_manager_full PRIVATE pthread rt store runnerdb sqlite3 log)
set_target_properties(test_sim_memory_manager_full PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)

add_executable(test_sim_device_memory_manager sim_device_memory_manager_test.cc)
target_link_libraries(test_sim_device_memory_manager PRIVATE pthread rt store runnerdb sqlite3 log)
set_target_properties(test_sim_device_memory_manager PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)

add_executable(test_sim_comm_memory_manager sim_comm_memory_manager_test.cc)
target_link_libraries(test_sim_comm_memory_manager PRIVATE pthread rt store runnerdb sqlite3 log)
set_target_properties(test_sim_comm_memory_manager PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)

add_executable(test_sim_shm_memory_common sim_shm_memory_common_test.cc)
target_link_libraries(test_sim_shm_memory_common PRIVATE pthread rt store runnerdb sqlite3 log)
set_target_properties(test_sim_shm_memory_common PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)

add_executable(test_sim_data_dump sim_data_dump_test.cc
    ${CMAKE_SOURCE_DIR}/src/common/sim_data_dump.cc
    ${CMAKE_SOURCE_DIR}/test/hccl_plugin_manager/hccl_plugin_manager_stub.cc
    ${CMAKE_SOURCE_DIR}/test/hccl_plugin_manager/hccl_plugin_stub.cc
)
target_include_directories(test_sim_data_dump PRIVATE
    ${CMAKE_SOURCE_DIR}/src/cmd
    ${CMAKE_SOURCE_DIR}/test/hccl_plugin_manager
)
target_link_libraries(test_sim_data_dump PRIVATE pthread rt store runnerdb sqlite3 log z)
set_target_properties(test_sim_data_dump PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)

add_executable(test_sim_shm_ops_hccl sim_shm_ops_hccl_test.cc)
target_link_libraries(test_sim_shm_ops_hccl PRIVATE pthread rt store runnerdb sqlite3 log)
set_target_properties(test_sim_shm_ops_hccl PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)

add_executable(test_hccl_shm_pub hccl_shm_pub_test.cc
    ${CMAKE_SOURCE_DIR}/src/store/store_sim_store_pub.cc
)
target_include_directories(test_hccl_shm_pub PRIVATE
    ${CMAKE_SOURCE_DIR}/src/store
)
target_link_libraries(test_hccl_shm_pub PRIVATE pthread rt store runnerdb sqlite3 log)
set_target_properties(test_hccl_shm_pub PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)

add_executable(test_sim_shm_ops_static sim_shm_ops_static_test.cc)
target_link_libraries(test_sim_shm_ops_static PRIVATE pthread rt log)
set_target_properties(test_sim_shm_ops_static PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)

add_executable(test_sim_run_mode sim_run_mode_test.cc)
target_link_libraries(test_sim_run_mode PRIVATE pthread rt store runnerdb sqlite3 log)
set_target_properties(test_sim_run_mode PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)

add_executable(test_sim_check_only_off sim_check_only_off_test.cc)
target_link_libraries(test_sim_check_only_off PRIVATE pthread rt store runnerdb sqlite3 log)
set_target_properties(test_sim_check_only_off PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)
