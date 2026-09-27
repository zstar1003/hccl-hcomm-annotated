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

include_directories(
    ${CMAKE_SOURCE_DIR}/src/plugin/runner/ccu_executor
    ${CMAKE_SOURCE_DIR}/src/plugin/runner/ccu_executor/trans_executor
    ${CMAKE_SOURCE_DIR}/src/plugin/runner/ccu_executor/reduce_executor
    ${CMAKE_SOURCE_DIR}/src/plugin/runner/runner_utils
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/header/external
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/store
    ${CMAKE_SOURCE_DIR}/include/runnerdb
    ${YAMLCPP_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/third_party/nlohmann_json
)

link_libraries(

    gtest
)

add_executable(test_sync_cke_executor sync_cke_executor_test.cc main.cc)
target_sources(test_sync_cke_executor PRIVATE
    ${CMAKE_SOURCE_DIR}/src/plugin/runner/ccu_executor/trans_executor/sync_cke_executor.cc
)
set_target_properties(test_sync_cke_executor PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)
target_link_libraries(test_sync_cke_executor ccu_executor_common_objs ${CCU_EXECUTOR_COMMON_LIBS})
ccu_executor_register_test(test_sync_cke_executor)

add_executable(test_sync_gsa_executor sync_gsa_executor_test.cc main.cc)
target_sources(test_sync_gsa_executor PRIVATE
    ${CMAKE_SOURCE_DIR}/src/plugin/runner/ccu_executor/trans_executor/sync_gsa_executor.cc
)
set_target_properties(test_sync_gsa_executor PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)
target_link_libraries(test_sync_gsa_executor ccu_executor_common_objs ${CCU_EXECUTOR_COMMON_LIBS})
ccu_executor_register_test(test_sync_gsa_executor)

add_executable(test_sync_xn_executor sync_xn_executor_test.cc main.cc)
target_sources(test_sync_xn_executor PRIVATE
    ${CMAKE_SOURCE_DIR}/src/plugin/runner/ccu_executor/trans_executor/sync_xn_executor.cc
)
set_target_properties(test_sync_xn_executor PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)
target_link_libraries(test_sync_xn_executor ccu_executor_common_objs ${CCU_EXECUTOR_COMMON_LIBS})
ccu_executor_register_test(test_sync_xn_executor)

add_executable(test_trans_loc_ms_to_loc_mem_executor trans_loc_ms_to_loc_mem_executor_test.cc main.cc)
target_sources(test_trans_loc_ms_to_loc_mem_executor PRIVATE
    ${CMAKE_SOURCE_DIR}/src/plugin/runner/ccu_executor/trans_executor/trans_loc_ms_to_loc_mem_executor.cc
)
set_target_properties(test_trans_loc_ms_to_loc_mem_executor PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)
target_link_libraries(test_trans_loc_ms_to_loc_mem_executor ccu_executor_common_objs ${CCU_EXECUTOR_COMMON_LIBS})
ccu_executor_register_test(test_trans_loc_ms_to_loc_mem_executor)

add_executable(test_trans_loc_mem_to_loc_ms_executor trans_loc_mem_to_loc_ms_executor_test.cc main.cc)
target_sources(test_trans_loc_mem_to_loc_ms_executor PRIVATE
    ${CMAKE_SOURCE_DIR}/src/plugin/runner/ccu_executor/trans_executor/trans_loc_mem_to_loc_ms_executor.cc
)
set_target_properties(test_trans_loc_mem_to_loc_ms_executor PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)
target_link_libraries(test_trans_loc_mem_to_loc_ms_executor ccu_executor_common_objs ${CCU_EXECUTOR_COMMON_LIBS})
ccu_executor_register_test(test_trans_loc_mem_to_loc_ms_executor)

add_executable(test_trans_loc_ms_to_loc_ms_executor trans_loc_ms_to_loc_ms_executor_test.cc main.cc)
target_sources(test_trans_loc_ms_to_loc_ms_executor PRIVATE
    ${CMAKE_SOURCE_DIR}/src/plugin/runner/ccu_executor/trans_executor/trans_loc_ms_to_loc_ms_executor.cc
)
set_target_properties(test_trans_loc_ms_to_loc_ms_executor PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)
target_link_libraries(test_trans_loc_ms_to_loc_ms_executor ccu_executor_common_objs ${CCU_EXECUTOR_COMMON_LIBS})
ccu_executor_register_test(test_trans_loc_ms_to_loc_ms_executor)

add_executable(test_trans_loc_ms_to_rmt_ms_executor trans_loc_ms_to_rmt_ms_executor_test.cc main.cc)
target_sources(test_trans_loc_ms_to_rmt_ms_executor PRIVATE
    ${CMAKE_SOURCE_DIR}/src/plugin/runner/ccu_executor/trans_executor/trans_loc_ms_to_rmt_ms_executor.cc
)
set_target_properties(test_trans_loc_ms_to_rmt_ms_executor PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)
target_link_libraries(test_trans_loc_ms_to_rmt_ms_executor ccu_executor_common_objs ${CCU_EXECUTOR_COMMON_LIBS})
ccu_executor_register_test(test_trans_loc_ms_to_rmt_ms_executor)

add_executable(test_trans_loc_ms_to_rmt_mem_executor trans_loc_ms_to_rmt_mem_executor_test.cc main.cc)
target_sources(test_trans_loc_ms_to_rmt_mem_executor PRIVATE
    ${CMAKE_SOURCE_DIR}/src/plugin/runner/ccu_executor/trans_executor/trans_loc_ms_to_rmt_mem_executor.cc
)
set_target_properties(test_trans_loc_ms_to_rmt_mem_executor PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)
target_link_libraries(test_trans_loc_ms_to_rmt_mem_executor ccu_executor_common_objs ${CCU_EXECUTOR_COMMON_LIBS})
ccu_executor_register_test(test_trans_loc_ms_to_rmt_mem_executor)

add_executable(test_trans_loc_mem_to_loc_mem_executor trans_loc_mem_to_loc_mem_executor_test.cc main.cc)
target_sources(test_trans_loc_mem_to_loc_mem_executor PRIVATE
    ${CMAKE_SOURCE_DIR}/src/plugin/runner/ccu_executor/trans_executor/trans_loc_mem_to_loc_mem_executor.cc
)
set_target_properties(test_trans_loc_mem_to_loc_mem_executor PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)
target_link_libraries(test_trans_loc_mem_to_loc_mem_executor ccu_executor_common_objs ${CCU_EXECUTOR_COMMON_LIBS})
ccu_executor_register_test(test_trans_loc_mem_to_loc_mem_executor)

add_executable(test_trans_loc_mem_to_rmt_mem_executor trans_loc_mem_to_rmt_mem_executor_test.cc main.cc)
target_sources(test_trans_loc_mem_to_rmt_mem_executor PRIVATE
    ${CMAKE_SOURCE_DIR}/src/plugin/runner/ccu_executor/trans_executor/trans_loc_mem_to_rmt_mem_executor.cc
)
set_target_properties(test_trans_loc_mem_to_rmt_mem_executor PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)
target_link_libraries(test_trans_loc_mem_to_rmt_mem_executor ccu_executor_common_objs ${CCU_EXECUTOR_COMMON_LIBS})
ccu_executor_register_test(test_trans_loc_mem_to_rmt_mem_executor)

add_executable(test_trans_rmt_executors trans_rmt_executors_test.cc main.cc)
target_sources(test_trans_rmt_executors PRIVATE
    ${CMAKE_SOURCE_DIR}/src/plugin/runner/ccu_executor/trans_executor/trans_rmt_ms_to_loc_ms_executor.cc
    ${CMAKE_SOURCE_DIR}/src/plugin/runner/ccu_executor/trans_executor/trans_rmt_ms_to_loc_mem_executor.cc
    ${CMAKE_SOURCE_DIR}/src/plugin/runner/ccu_executor/trans_executor/trans_rmt_mem_to_loc_ms_executor.cc
    ${CMAKE_SOURCE_DIR}/src/plugin/runner/ccu_executor/trans_executor/trans_rmt_mem_to_loc_mem_executor.cc
)
set_target_properties(test_trans_rmt_executors PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)
target_link_libraries(test_trans_rmt_executors ccu_executor_common_objs ${CCU_EXECUTOR_COMMON_LIBS})
ccu_executor_register_test(test_trans_rmt_executors)
