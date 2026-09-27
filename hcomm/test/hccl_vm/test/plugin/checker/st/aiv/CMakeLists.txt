# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(CHECKER_SRC ${CMAKE_SOURCE_DIR}/src/plugin/checker/src)
set(CHECKER_HDR ${CMAKE_SOURCE_DIR}/src/plugin/checker/header)

add_executable(test_aiv_st
    aiv_st_test.cc
    aiv_test_stub.cc
    ../../main.cc
)

target_sources(test_aiv_st PRIVATE
    ${CHECKER_SRC}/framework/task_graph_generator_v3/task_graph_generator_v3.cc
    ${CHECKER_SRC}/framework/task_graph_generator_v3/task_graph_mem_conflict_v3.cc
    ${CHECKER_SRC}/framework/task_graph_generator_v3/task_graph_reachability_v3.cc
    ${CHECKER_SRC}/framework/task_graph_generator_v3/task_graph_single_task_check_v3.cc
    ${CHECKER_SRC}/framework/task_graph_generator_v3/task_def_v3.cc
    ${CHECKER_SRC}/framework/task_graph_generator_v3/aiv_graph_generator_v3/aiv_snapshot_json_loader_v3.cc
    ${CHECKER_SRC}/framework/task_graph_generator_v3/aiv_graph_generator_v3/aiv_task_transform_v3.cc
    ${CHECKER_SRC}/utils/storage_manager.cc
    ${CHECKER_SRC}/utils/binary_data_operator.cc
    ${CHECKER_SRC}/framework/task_graph_generator/ccu_all_rank_param_recorder.cc
    ${CHECKER_SRC}/log/log_stub.cc
)

target_include_directories(test_aiv_st BEFORE PRIVATE
    ${CHECKER_SRC}/utils
    ${CHECKER_SRC}/checker
    ${CHECKER_SRC}/log
    ${CHECKER_SRC}/framework/task_graph_generator
    ${CHECKER_SRC}/framework/task_graph_generator_v3
    ${CHECKER_SRC}/framework/task_graph_generator_v3/aiv_graph_generator_v3
    ${CHECKER_SRC}/framework/task_graph_generator_v3/ccu_graph_generator_v3
    ${CHECKER_HDR}/internal
    ${CHECKER_HDR}/external
    ${CHECKER_HDR}/external/nlohmann
    ${CMAKE_SOURCE_DIR}/src/plugin/runner/ccu_executor
    ${ASCEND_CANN_PACKAGE_PATH}/${CANN_ARCH_DIR}/include
    ${ASCEND_CANN_PACKAGE_PATH}/${CANN_ARCH_DIR}/pkg_inc
    ${ASCEND_CANN_PACKAGE_PATH}/${CANN_ARCH_DIR}/pkg_inc/hccl

    ${HCOMM_CODE_ROOT}/src/base_comm/resources/ccu/ccu_microcode
)

target_link_libraries(test_aiv_st PRIVATE
    log
    ${ASCEND_CANN_PACKAGE_PATH}/${CANN_ARCH_DIR}/lib64/libc_sec.a
)

set_target_properties(test_aiv_st PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)
