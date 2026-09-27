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

include_directories(BEFORE
    ${CHECKER_SRC}
    ${CHECKER_SRC}/utils/dump
    ${CHECKER_SRC}/utils
    ${CHECKER_SRC}/checker
    ${CHECKER_SRC}/log
    ${CHECKER_SRC}/framework/semantics_check
    ${CHECKER_SRC}/framework/singletask_check
    ${CHECKER_SRC}/framework/mem_conflict_check
    ${CHECKER_SRC}/framework/task_graph_generator
    ${CHECKER_SRC}/framework/task_graph_generator_v3
    ${CHECKER_SRC}/dfx
    ${CHECKER_HDR}/internal
    ${CHECKER_HDR}/external
    ${CHECKER_HDR}/external/nlohmann
    ${CMAKE_SOURCE_DIR}/src/plugin/runner/ccu_executor
    ${CMAKE_SOURCE_DIR}/src/plugin/runner/runner_utils
    ${ASCEND_CANN_PACKAGE_PATH}/${CANN_ARCH_DIR}/include
    ${ASCEND_CANN_PACKAGE_PATH}/${CANN_ARCH_DIR}/pkg_inc
    ${ASCEND_CANN_PACKAGE_PATH}/${CANN_ARCH_DIR}/pkg_inc/hccl

    ${HCOMM_CODE_ROOT}/src/base_comm/resources/ccu/ccu_microcode
)

add_executable(test_checker checker_test.cc main.cc)
target_sources(test_checker PRIVATE
    ${CHECKER_SRC}/checker/checker.cc
    ${CHECKER_SRC}/dfx/dag_graphviz_dump.cc
    ${CHECKER_SRC}/log/log_stub.cc
    ${CHECKER_SRC}/utils/check_utils.cc
    ${CHECKER_SRC}/utils/sim_task.cc
    ${CHECKER_SRC}/utils/task_utils.cc
    ${CHECKER_SRC}/utils/task_ccu.cc
    ${CHECKER_SRC}/utils/storage_manager.cc
    ${CHECKER_SRC}/utils/binary_data_operator.cc
    ${CHECKER_SRC}/framework/singletask_check/singletask_check.cc
    ${CHECKER_SRC}/framework/semantics_check/task_check_op_semantics.cc
    ${CHECKER_SRC}/framework/semantics_check/allreduce_semantics_checker.cc
    ${CHECKER_SRC}/framework/semantics_check/allgather_semantics_checker.cc
    ${CHECKER_SRC}/framework/semantics_check/allgather_v_semantics_checker.cc
    ${CHECKER_SRC}/framework/semantics_check/reduce_scatter_semantics_checker.cc
    ${CHECKER_SRC}/framework/semantics_check/reduce_scatter_v_semantics_checker.cc
    ${CHECKER_SRC}/framework/semantics_check/reduce_semantics_checker.cc
    ${CHECKER_SRC}/framework/semantics_check/broadcast_semantics_checker.cc
    ${CHECKER_SRC}/framework/semantics_check/scatter_semantics_checker.cc
    ${CHECKER_SRC}/framework/semantics_check/send_recv_semantics_checker.cc
    ${CHECKER_SRC}/framework/semantics_check/batchsendrecv_semantics_checker.cc
    ${CHECKER_SRC}/framework/semantics_check/all2all_semantics_checker.cc
    ${CHECKER_SRC}/framework/task_graph_generator/ccu_task_transform_instruct_a5.cc
    ${CHECKER_SRC}/framework/task_graph_generator/ccu_task_transform_instruct_common.cc
    ${CHECKER_SRC}/framework/task_graph_generator/task_graph_generator.cc
    ${CHECKER_SRC}/framework/task_graph_generator/ccu_task_common.cc
    ${CHECKER_SRC}/framework/task_graph_generator/ccu_task_transform.cc
    ${CHECKER_SRC}/framework/task_graph_generator/ccu_all_rank_param_recorder.cc
    ${CHECKER_SRC}/framework/task_graph_generator/type_conversion.cc
    ${CHECKER_SRC}/framework/task_graph_generator_v3/task_graph_generator_v3.cc
    ${CHECKER_SRC}/framework/task_graph_generator_v3/task_graph_mem_conflict_v3.cc
    ${CHECKER_SRC}/framework/task_graph_generator_v3/task_graph_sync_conflict_v3.cc
    ${CHECKER_SRC}/framework/task_graph_generator_v3/task_graph_reachability_v3.cc
    ${CHECKER_SRC}/framework/task_graph_generator_v3/task_graph_semantic_check_v3.cc
    ${CHECKER_SRC}/framework/task_graph_generator_v3/task_graph_single_task_check_v3.cc
    ${CHECKER_SRC}/framework/task_graph_generator_v3/task_meta_translator_v3.cc
    ${CHECKER_SRC}/framework/task_graph_generator_v3/task_def_v3.cc
    ${CHECKER_SRC}/framework/task_graph_generator_v3/aiv_graph_generator_v3/aiv_snapshot_json_loader_v3.cc
    ${CHECKER_SRC}/framework/task_graph_generator_v3/aiv_graph_generator_v3/aiv_task_transform_v3.cc
    ${CHECKER_SRC}/framework/task_graph_generator_v3/ccu_graph_generator_v3/ccu_all_rank_param_recorder_v3.cc
    ${CHECKER_SRC}/framework/task_graph_generator_v3/ccu_graph_generator_v3/ccu_loop_merge_v3.cc
    ${CHECKER_SRC}/framework/task_graph_generator_v3/ccu_graph_generator_v3/ccu_task_common_v3.cc
    ${CHECKER_SRC}/framework/task_graph_generator_v3/ccu_graph_generator_v3/ccu_task_transform_v3.cc
    ${CHECKER_SRC}/framework/task_graph_generator_v3/ccu_graph_generator_v3/ccu_task_transform_instruct_a5_v3.cc
    ${CHECKER_SRC}/framework/task_graph_generator_v3/ccu_graph_generator_v3/ccu_task_transform_instruct_common_v3.cc
)

if(BUILD_A6_CCU_INSTR)
    target_sources(test_checker PRIVATE
        ${CHECKER_SRC}/framework/task_graph_generator/ccu_task_transform_instruct_a6.cc
        ${CHECKER_SRC}/framework/task_graph_generator_v3/ccu_graph_generator_v3/ccu_task_transform_instruct_a6_v3.cc
    )
endif()

target_sources(test_checker PRIVATE
    ${CHECKER_SRC}/framework/mem_conflict_check/mem_conflict_check_utils.cc
    ${CHECKER_SRC}/framework/mem_conflict_check/check_rank_mem.cc
    ${CHECKER_SRC}/framework/mem_conflict_check/task_graph_revamp.cc
    ${CHECKER_SRC}/framework/mem_conflict_check/task_graph_revamp_base.cc
    ${CHECKER_SRC}/framework/mem_conflict_check/task_graph_revamp_bilateral_ccu.cc
    ${CHECKER_SRC}/framework/mem_conflict_check/task_graph_revamp_parallel.cc
    ${CHECKER_SRC}/utils/file_utils.cc
    ${CHECKER_SRC}/utils/setting_manager.cc
    ${CHECKER_SRC}/utils/dump/dump_json_utils.cc
    ${CHECKER_SRC}/utils/dump/dump_manager.cc
    ${CHECKER_SRC}/utils/dump/dump_memory_timeline.cc
    ${CHECKER_SRC}/utils/dump/dump_graph.cc
    ${CHECKER_SRC}/utils/dump/dump_run_manifest.cc
    ${CHECKER_SRC}/utils/dump/validation_issue_recorder.cc
    ${CHECKER_SRC}/utils/dump/msgpack_writer.cc
    ${CHECKER_SRC}/utils/dump_v3/dump_v3_manager.cc
    ${CHECKER_SRC}/utils/dump_v3/dump_v3_dag.cc
)
target_include_directories(test_checker BEFORE PRIVATE
    ${CHECKER_SRC}/utils
    ${CHECKER_SRC}/checker
    ${CHECKER_SRC}/log
    ${CHECKER_SRC}/framework/semantics_check
    ${CHECKER_SRC}/framework/singletask_check
    ${CHECKER_SRC}/framework/mem_conflict_check
    ${CHECKER_SRC}/framework/task_graph_generator
    ${CHECKER_SRC}/dfx
    ${CHECKER_HDR}/internal
    ${CHECKER_HDR}/external
    ${CHECKER_HDR}/external/nlohmann
    ${THRID_PARTY_DIR}/nlohmann_json
    ${THRID_PARTY_DIR}
    ${THRID_PARTY_DIR}/yaml-cpp/include
    ${CMAKE_SOURCE_DIR}/src/plugin/runner/ccu_executor
    ${ASCEND_CANN_PACKAGE_PATH}/${CANN_ARCH_DIR}/include
    ${ASCEND_CANN_PACKAGE_PATH}/${CANN_ARCH_DIR}/pkg_inc
    ${ASCEND_CANN_PACKAGE_PATH}/${CANN_ARCH_DIR}/pkg_inc/hccl

    ${HCOMM_CODE_ROOT}/src/base_comm/resources/ccu/ccu_microcode
)
get_target_property(_checker_includes test_checker INCLUDE_DIRECTORIES)
list(FILTER _checker_includes EXCLUDE REGEX "^${CMAKE_SOURCE_DIR}/include$")
set_target_properties(test_checker PROPERTIES INCLUDE_DIRECTORIES "${_checker_includes}")
target_link_libraries(test_checker PRIVATE
    log
    ${ASCEND_CANN_PACKAGE_PATH}/${CANN_ARCH_DIR}/lib64/libc_sec.a
)
set_target_properties(test_checker PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)

add_executable(test_check_utils check_utils_test.cc main.cc)
target_sources(test_check_utils PRIVATE
    ${CHECKER_SRC}/utils/check_utils.cc
    ${CHECKER_SRC}/log/log_stub.cc
)
target_link_libraries(test_check_utils PRIVATE
    log
    ${ASCEND_CANN_PACKAGE_PATH}/${CANN_ARCH_DIR}/lib64/libc_sec.a
)
set_target_properties(test_check_utils PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)

add_executable(test_sim_task sim_task_test.cc main.cc)
target_sources(test_sim_task PRIVATE
    ${CHECKER_SRC}/utils/sim_task.cc
    ${CHECKER_SRC}/log/log_stub.cc
)
target_link_libraries(test_sim_task PRIVATE
    log
    ${ASCEND_CANN_PACKAGE_PATH}/${CANN_ARCH_DIR}/lib64/libc_sec.a
)
set_target_properties(test_sim_task PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)

add_executable(test_binary_data_operator_checker binary_data_operator_checker_test.cc main.cc)
target_sources(test_binary_data_operator_checker PRIVATE
    ${CHECKER_SRC}/utils/binary_data_operator.cc
    ${CHECKER_SRC}/log/log_stub.cc
)
target_link_libraries(test_binary_data_operator_checker PRIVATE
    log
    ${ASCEND_CANN_PACKAGE_PATH}/${CANN_ARCH_DIR}/lib64/libc_sec.a
)
set_target_properties(test_binary_data_operator_checker PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)

add_executable(test_storage_manager_checker storage_manager_checker_test.cc main.cc)
target_sources(test_storage_manager_checker PRIVATE
    ${CHECKER_SRC}/utils/storage_manager.cc
    ${CHECKER_SRC}/utils/binary_data_operator.cc
    ${CHECKER_SRC}/framework/task_graph_generator/ccu_all_rank_param_recorder.cc
    ${CHECKER_SRC}/log/log_stub.cc
)
target_include_directories(test_storage_manager_checker BEFORE PRIVATE
    ${CHECKER_SRC}/framework/task_graph_generator
    ${CHECKER_SRC}/utils
    ${CHECKER_HDR}/internal
    ${CHECKER_HDR}/external
    ${CHECKER_HDR}/external/nlohmann
    ${THRID_PARTY_DIR}/nlohmann_json
    ${THRID_PARTY_DIR}
    ${THRID_PARTY_DIR}/yaml-cpp/include
    ${CMAKE_SOURCE_DIR}/src/plugin/runner/ccu_executor
    ${ASCEND_CANN_PACKAGE_PATH}/${CANN_ARCH_DIR}/include
    ${ASCEND_CANN_PACKAGE_PATH}/${CANN_ARCH_DIR}/pkg_inc
    ${ASCEND_CANN_PACKAGE_PATH}/${CANN_ARCH_DIR}/pkg_inc/hccl

    ${HCOMM_CODE_ROOT}/src/base_comm/resources/ccu/ccu_microcode
)
get_target_property(_storage_mgr_includes test_storage_manager_checker INCLUDE_DIRECTORIES)
list(FILTER _storage_mgr_includes EXCLUDE REGEX "^${CMAKE_SOURCE_DIR}/include$")
set_target_properties(test_storage_manager_checker PROPERTIES INCLUDE_DIRECTORIES "${_storage_mgr_includes}")
target_link_libraries(test_storage_manager_checker PRIVATE
    log
    ${ASCEND_CANN_PACKAGE_PATH}/${CANN_ARCH_DIR}/lib64/libc_sec.a
)
set_target_properties(test_storage_manager_checker PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)

add_executable(test_task_ccu task_ccu_test.cc main.cc)
target_sources(test_task_ccu PRIVATE
    ${CHECKER_SRC}/utils/task_ccu.cc
    ${CHECKER_SRC}/utils/sim_task.cc
    ${CMAKE_SOURCE_DIR}/src/plugin/runner/ccu_executor/ccu_string_util.cc
    ${CHECKER_SRC}/log/log_stub.cc
)
target_link_libraries(test_task_ccu PRIVATE
    log
    ${ASCEND_CANN_PACKAGE_PATH}/${CANN_ARCH_DIR}/lib64/libc_sec.a
)
set_target_properties(test_task_ccu PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)

add_executable(test_task_utils task_utils_test.cc main.cc)
target_sources(test_task_utils PRIVATE
    ${CHECKER_SRC}/utils/task_utils.cc
    ${CHECKER_SRC}/utils/sim_task.cc
    ${CHECKER_SRC}/utils/task_ccu.cc
    ${CHECKER_SRC}/utils/storage_manager.cc
    ${CHECKER_SRC}/utils/binary_data_operator.cc
    ${CHECKER_SRC}/framework/task_graph_generator/ccu_all_rank_param_recorder.cc
    ${CMAKE_SOURCE_DIR}/src/plugin/runner/ccu_executor/ccu_string_util.cc
    ${CHECKER_SRC}/log/log_stub.cc
)
target_include_directories(test_task_utils BEFORE PRIVATE
    ${CHECKER_SRC}/framework/task_graph_generator
    ${CHECKER_SRC}/utils
    ${CHECKER_SRC}/checker
    ${CHECKER_SRC}/log
    ${CHECKER_HDR}/internal
    ${CHECKER_HDR}/external
    ${CHECKER_HDR}/external/nlohmann
    ${THRID_PARTY_DIR}/nlohmann_json
    ${THRID_PARTY_DIR}
    ${THRID_PARTY_DIR}/yaml-cpp/include
    ${CMAKE_SOURCE_DIR}/src/plugin/runner/ccu_executor
    ${ASCEND_CANN_PACKAGE_PATH}/${CANN_ARCH_DIR}/pkg_inc
    ${ASCEND_CANN_PACKAGE_PATH}/${CANN_ARCH_DIR}/pkg_inc/hccl

    ${HCOMM_CODE_ROOT}/src/base_comm/resources/ccu/ccu_microcode
)
get_target_property(_task_utils_includes test_task_utils INCLUDE_DIRECTORIES)
list(FILTER _task_utils_includes EXCLUDE REGEX "^${CMAKE_SOURCE_DIR}/include$")
set_target_properties(test_task_utils PROPERTIES INCLUDE_DIRECTORIES "${_task_utils_includes}")
target_link_libraries(test_task_utils PRIVATE
    log
    ${ASCEND_CANN_PACKAGE_PATH}/${CANN_ARCH_DIR}/lib64/libc_sec.a
)
set_target_properties(test_task_utils PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)

add_subdirectory(framework)
add_subdirectory(st)
