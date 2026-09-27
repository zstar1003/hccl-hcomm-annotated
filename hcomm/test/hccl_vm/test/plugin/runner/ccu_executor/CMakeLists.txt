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

add_library(ccu_executor_common_objs OBJECT
    ${CMAKE_SOURCE_DIR}/src/plugin/runner/ccu_executor/ccu_string_util.cc
    ${CMAKE_SOURCE_DIR}/src/plugin/runner/ccu_executor/ccu_executor_base.cc
    ${CMAKE_SOURCE_DIR}/src/plugin/runner/ccu_executor/ccu_simulator.cc
    ${CMAKE_SOURCE_DIR}/src/plugin/runner/ccu_executor/ccu_resource_manager.cc
    ${CMAKE_SOURCE_DIR}/src/plugin/runner/ccu_executor/ccu_reduce_operator.cc
    ${CMAKE_SOURCE_DIR}/src/plugin/runner/ccu_executor/ccu_executor_manager.cc
    ${CMAKE_SOURCE_DIR}/src/plugin/runner/ccu_executor/ccu_resource_v1.cc
    ${CMAKE_SOURCE_DIR}/src/plugin/runner/ccu_executor/ccu_resource_v2.cc
    ${CMAKE_SOURCE_DIR}/src/plugin/runner/ccu_executor/trace/ccu_trace_collector.cc
    ${CMAKE_SOURCE_DIR}/src/plugin/runner/ccu_executor/trace/ccu_trace_serializer.cc
    ${CMAKE_SOURCE_DIR}/src/utils/sim_log.cc
    ${CMAKE_SOURCE_DIR}/src/store/store_sim_shm_memory_common.cc
)

set(CCU_EXECUTOR_COMMON_LIBS
    runnerdb
    store
    log
    sqlite3
)

function(ccu_executor_register_test TEST_NAME)
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
    set_property(TEST ${TEST_NAME} PROPERTY ENVIRONMENT
        "GCOV_PREFIX=${CMAKE_BINARY_DIR}/coverage/${TEST_NAME}"
    )
endfunction()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -D_GLIBCXX_USE_CXX11_ABI=0")

include_directories(
    ${HCOMM_CODE_ROOT}/src/base_comm/resources/ccu/ccu_microcode

    ${CMAKE_SOURCE_DIR}/src/plugin/runner/ccu_executor
    ${CMAKE_SOURCE_DIR}/src/plugin/runner/ccu_executor/control_executor
    ${CMAKE_SOURCE_DIR}/src/plugin/runner/ccu_executor/load_executor
    ${CMAKE_SOURCE_DIR}/src/plugin/runner/ccu_executor/reduce_executor
    ${CMAKE_SOURCE_DIR}/src/plugin/runner/ccu_executor/trans_executor
    ${CMAKE_SOURCE_DIR}/src/plugin/runner/ccu_executor/trace
    ${CMAKE_SOURCE_DIR}/src/plugin/runner/runner_utils
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/header/external
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/store
    ${CMAKE_SOURCE_DIR}/include/runnerdb
    ${YAMLCPP_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/third_party/yaml-cpp/include
    ${CMAKE_SOURCE_DIR}/third_party/nlohmann_json
)

link_libraries(gtest)

# ==================== Control Type Tests ====================
add_executable(test_ccu_executor_control ccu_executor_control_test.cc main.cc)
target_sources(test_ccu_executor_control PRIVATE
    ${CMAKE_SOURCE_DIR}/src/plugin/runner/ccu_executor/control_executor/clear_cke_executor.cc
    ${CMAKE_SOURCE_DIR}/src/plugin/runner/ccu_executor/control_executor/jump_executor.cc
    ${CMAKE_SOURCE_DIR}/src/plugin/runner/ccu_executor/control_executor/loop_executor.cc
    ${CMAKE_SOURCE_DIR}/src/plugin/runner/ccu_executor/control_executor/loop_group_executor.cc
    ${CMAKE_SOURCE_DIR}/src/plugin/runner/ccu_executor/control_executor/set_cke_executor.cc
)
set_target_properties(test_ccu_executor_control PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)
target_link_libraries(test_ccu_executor_control ccu_executor_common_objs ${CCU_EXECUTOR_COMMON_LIBS})
ccu_executor_register_test(test_ccu_executor_control)

# ==================== Load Type Tests ====================
add_executable(test_ccu_executor_load ccu_executor_load_test.cc main.cc)
target_sources(test_ccu_executor_load PRIVATE
    ${CMAKE_SOURCE_DIR}/src/plugin/runner/ccu_executor/load_executor/load_gsa_gsa_executor.cc
    ${CMAKE_SOURCE_DIR}/src/plugin/runner/ccu_executor/load_executor/load_gsa_xn_executor.cc
    ${CMAKE_SOURCE_DIR}/src/plugin/runner/ccu_executor/load_executor/load_imd_to_gsa_executor.cc
    ${CMAKE_SOURCE_DIR}/src/plugin/runner/ccu_executor/load_executor/load_imd_to_xn_executor.cc
    ${CMAKE_SOURCE_DIR}/src/plugin/runner/ccu_executor/load_executor/load_sqe_args_to_gsa_executor.cc
    ${CMAKE_SOURCE_DIR}/src/plugin/runner/ccu_executor/load_executor/load_sqe_args_to_xn_executor.cc
    ${CMAKE_SOURCE_DIR}/src/plugin/runner/ccu_executor/load_executor/load_xn_xn_executor.cc
)
set_target_properties(test_ccu_executor_load PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)
target_link_libraries(test_ccu_executor_load ccu_executor_common_objs ${CCU_EXECUTOR_COMMON_LIBS})
ccu_executor_register_test(test_ccu_executor_load)

# ==================== Reduce Type Tests ====================
add_executable(test_ccu_executor_reduce ccu_executor_reduce_test.cc main.cc)
target_sources(test_ccu_executor_reduce PRIVATE
    ${CMAKE_SOURCE_DIR}/src/plugin/runner/ccu_executor/reduce_executor/reduce_add_executor.cc
    ${CMAKE_SOURCE_DIR}/src/plugin/runner/ccu_executor/reduce_executor/reduce_max_executor.cc
    ${CMAKE_SOURCE_DIR}/src/plugin/runner/ccu_executor/reduce_executor/reduce_min_executor.cc
)
set_target_properties(test_ccu_executor_reduce PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)
target_link_libraries(test_ccu_executor_reduce ccu_executor_common_objs ${CCU_EXECUTOR_COMMON_LIBS})
ccu_executor_register_test(test_ccu_executor_reduce)

# ==================== Trans Type Tests ====================
add_executable(test_ccu_executor_trans ccu_executor_trans_test.cc main.cc)
target_sources(test_ccu_executor_trans PRIVATE
    ${CMAKE_SOURCE_DIR}/src/plugin/runner/ccu_executor/trans_executor/sync_cke_executor.cc
    ${CMAKE_SOURCE_DIR}/src/plugin/runner/ccu_executor/trans_executor/sync_gsa_executor.cc
    ${CMAKE_SOURCE_DIR}/src/plugin/runner/ccu_executor/trans_executor/sync_xn_executor.cc
    ${CMAKE_SOURCE_DIR}/src/plugin/runner/ccu_executor/trans_executor/trans_loc_mem_to_loc_mem_executor.cc
    ${CMAKE_SOURCE_DIR}/src/plugin/runner/ccu_executor/trans_executor/trans_loc_mem_to_loc_ms_executor.cc
    ${CMAKE_SOURCE_DIR}/src/plugin/runner/ccu_executor/trans_executor/trans_loc_mem_to_rmt_mem_executor.cc
    ${CMAKE_SOURCE_DIR}/src/plugin/runner/ccu_executor/trans_executor/trans_loc_ms_to_loc_mem_executor.cc
    ${CMAKE_SOURCE_DIR}/src/plugin/runner/ccu_executor/trans_executor/trans_loc_ms_to_loc_ms_executor.cc
    ${CMAKE_SOURCE_DIR}/src/plugin/runner/ccu_executor/trans_executor/trans_loc_ms_to_rmt_mem_executor.cc
    ${CMAKE_SOURCE_DIR}/src/plugin/runner/ccu_executor/trans_executor/trans_loc_ms_to_rmt_ms_executor.cc
    ${CMAKE_SOURCE_DIR}/src/plugin/runner/ccu_executor/trans_executor/trans_rmt_mem_to_loc_mem_executor.cc
    ${CMAKE_SOURCE_DIR}/src/plugin/runner/ccu_executor/trans_executor/trans_rmt_mem_to_loc_ms_executor.cc
    ${CMAKE_SOURCE_DIR}/src/plugin/runner/ccu_executor/trans_executor/trans_rmt_ms_to_loc_mem_executor.cc
    ${CMAKE_SOURCE_DIR}/src/plugin/runner/ccu_executor/trans_executor/trans_rmt_ms_to_loc_ms_executor.cc
)
set_target_properties(test_ccu_executor_trans PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)
target_link_libraries(test_ccu_executor_trans ccu_executor_common_objs ${CCU_EXECUTOR_COMMON_LIBS})
ccu_executor_register_test(test_ccu_executor_trans)

add_subdirectory(reduce_executor)

# ==================== FWK Types Tests ====================
add_executable(test_ccu_fp16 ccu_fp16_test.cc main.cc)
set_target_properties(test_ccu_fp16 PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)
ccu_executor_register_test(test_ccu_fp16)
