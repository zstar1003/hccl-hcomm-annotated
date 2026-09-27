# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(CCU_EXECUTOR_COMMON_LIBS
    runnerdb
    store
    log
    sqlite3
)

add_subdirectory(ccu_executor)

add_executable(test_ccu_string_util ccu_string_util_test.cc ../main.cc)
target_sources(test_ccu_string_util PRIVATE
    ${CMAKE_SOURCE_DIR}/src/plugin/runner/ccu_executor/ccu_string_util.cc
)
set_target_properties(test_ccu_string_util PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)

add_executable(test_device_resource device_resource_test.cc ../main.cc)
target_sources(test_device_resource PRIVATE
    ${CMAKE_SOURCE_DIR}/src/plugin/runner/runner_utils/device_resource.cc
)
set_target_properties(test_device_resource PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)

add_executable(test_device_resource_manager device_resource_manager_test.cc ../main.cc)
target_sources(test_device_resource_manager PRIVATE
    ${CMAKE_SOURCE_DIR}/src/plugin/runner/runner_utils/device_resource_manager.cc
    ${CMAKE_SOURCE_DIR}/src/plugin/runner/runner_utils/device_resource.cc
)
set_target_properties(test_device_resource_manager PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)

add_executable(test_ccu_executor_base ccu_executor_base_test.cc ../main.cc)
set_target_properties(test_ccu_executor_base PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)
target_link_libraries(test_ccu_executor_base ccu_executor_common_objs ${CCU_EXECUTOR_COMMON_LIBS})

add_executable(test_ccu_executor_manager ccu_executor_manager_test.cc ../main.cc)
set_target_properties(test_ccu_executor_manager PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)
target_link_libraries(test_ccu_executor_manager ccu_executor_common_objs ${CCU_EXECUTOR_COMMON_LIBS})

add_executable(test_ccu_resource_manager ccu_resource_manager_test.cc ../main.cc)
set_target_properties(test_ccu_resource_manager PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)
target_link_libraries(test_ccu_resource_manager ccu_executor_common_objs ${CCU_EXECUTOR_COMMON_LIBS})

add_executable(test_ccu_resource_v1 ccu_resource_v1_test.cc ../main.cc)
set_target_properties(test_ccu_resource_v1 PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)
target_link_libraries(test_ccu_resource_v1 ccu_executor_common_objs ${CCU_EXECUTOR_COMMON_LIBS})

add_executable(test_hccl_task_reduce_process hccl_task_reduce_process_test.cc ../main.cc)
target_sources(test_hccl_task_reduce_process PRIVATE
    ${CMAKE_SOURCE_DIR}/src/plugin/runner/runner_utils/hccl_task_reduce_process.cc
)
set_target_properties(test_hccl_task_reduce_process PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)

add_executable(test_hccl_task_thread hccl_task_thread_test.cc ../main.cc)
target_sources(test_hccl_task_thread PRIVATE
    ${CMAKE_SOURCE_DIR}/src/plugin/runner/runner_utils/hccl_task_thread.cc
    ${CMAKE_SOURCE_DIR}/src/plugin/runner/runner_utils/hccl_task_reduce_process.cc
    ${CMAKE_SOURCE_DIR}/src/plugin/runner/runner_utils/device_resource.cc
    ${CMAKE_SOURCE_DIR}/src/plugin/runner/runner_utils/device_resource_manager.cc
    ${CMAKE_SOURCE_DIR}/src/plugin/runner/runner_utils/storage_manager.cc
    ${CMAKE_SOURCE_DIR}/src/plugin/runner/aiv_executor/aiv_resource_manager.cc
    ${CMAKE_SOURCE_DIR}/src/plugin/runner/aiv_executor/aiv_graph_executor.cc
    ${CMAKE_SOURCE_DIR}/src/plugin/runner/aiv_executor/aiv_graph_executor_mgr.cc
    ${CMAKE_SOURCE_DIR}/src/plugin/runner/aiv_executor/aiv_task_snapshot_loader.cc
    ${CMAKE_SOURCE_DIR}/src/proxy/aiv_kernel/aiv_mode_stub/ai_core_stub.cc
    ${CMAKE_SOURCE_DIR}/src/proxy/aiv_kernel/aiv_mode_stub/aiv_task.cc
    ${CMAKE_SOURCE_DIR}/src/proxy/aiv_kernel/aiv_mode_stub/aiv_task_json.cc
    ${CMAKE_SOURCE_DIR}/src/proxy/aiv_kernel/ascendc_stub/ascendc_base_stub.cc
    ${CMAKE_SOURCE_DIR}/src/store/store_sim_store_pub.cc
    ${CMAKE_SOURCE_DIR}/src/store/store_binary_data_operator.cc
    ${CMAKE_SOURCE_DIR}/src/runnerdb/tracedb/db_sim_sqlite_db.cc
)
target_link_libraries(test_hccl_task_thread ccu_executor_common_objs ${CCU_EXECUTOR_COMMON_LIBS}
    ${ASCEND_CANN_PACKAGE_PATH}/${CANN_ARCH_DIR}/lib64/libhcomm.so)
set_target_properties(test_hccl_task_thread PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)

add_executable(test_storage_manager storage_manager_test.cc ../main.cc)
target_sources(test_storage_manager PRIVATE
    ${CMAKE_SOURCE_DIR}/src/plugin/runner/runner_utils/storage_manager.cc
    ${CMAKE_SOURCE_DIR}/src/plugin/runner/aiv_executor/aiv_resource_manager.cc
    ${CMAKE_SOURCE_DIR}/src/plugin/runner/aiv_executor/aiv_graph_executor.cc
    ${CMAKE_SOURCE_DIR}/src/plugin/runner/aiv_executor/aiv_graph_executor_mgr.cc
    ${CMAKE_SOURCE_DIR}/src/plugin/runner/aiv_executor/aiv_task_snapshot_loader.cc
    ${CMAKE_SOURCE_DIR}/src/proxy/aiv_kernel/aiv_mode_stub/ai_core_stub.cc
    ${CMAKE_SOURCE_DIR}/src/proxy/aiv_kernel/aiv_mode_stub/aiv_task.cc
    ${CMAKE_SOURCE_DIR}/src/proxy/aiv_kernel/aiv_mode_stub/aiv_task_json.cc
    ${CMAKE_SOURCE_DIR}/src/proxy/aiv_kernel/ascendc_stub/ascendc_base_stub.cc
    ${CMAKE_SOURCE_DIR}/src/store/store_sim_store_pub.cc
    ${CMAKE_SOURCE_DIR}/src/store/store_binary_data_operator.cc
    ${CMAKE_SOURCE_DIR}/src/runnerdb/tracedb/db_sim_sqlite_db.cc
)
target_link_libraries(test_storage_manager ccu_executor_common_objs ${CCU_EXECUTOR_COMMON_LIBS})
set_target_properties(test_storage_manager PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)

add_executable(test_ccu_simulator ccu_simulator_test.cc ../main.cc)
set_target_properties(test_ccu_simulator PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)
target_link_libraries(test_ccu_simulator ccu_executor_common_objs ${CCU_EXECUTOR_COMMON_LIBS})

add_executable(test_ccu_simulator_base ccu_simulator_base_test.cc ../main.cc)
target_sources(test_ccu_simulator_base PRIVATE
    ${CMAKE_SOURCE_DIR}/src/plugin/runner/ccu_executor/ccu_simulator_base.cc
)
set_target_properties(test_ccu_simulator_base PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)

add_executable(test_aiv_task_json aiv_task_json_test.cc ../main.cc)
target_sources(test_aiv_task_json PRIVATE
    ${CMAKE_SOURCE_DIR}/src/proxy/aiv_kernel/aiv_mode_stub/aiv_task_json.cc
    ${CMAKE_SOURCE_DIR}/src/proxy/aiv_kernel/aiv_mode_stub/aiv_task.cc
    ${CMAKE_SOURCE_DIR}/src/proxy/aiv_kernel/aiv_mode_stub/ai_core_stub.cc
    ${CMAKE_SOURCE_DIR}/src/proxy/aiv_kernel/ascendc_stub/ascendc_base_stub.cc
)
set_target_properties(test_aiv_task_json PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)
target_include_directories(test_aiv_task_json PRIVATE
    ${THRID_PARTY_DIR}/nlohmann_json
    ${THRID_PARTY_DIR}
)
target_link_libraries(test_aiv_task_json log pthread)

add_executable(test_aiv_task_snapshot_loader aiv_task_snapshot_loader_test.cc ../main.cc)
target_sources(test_aiv_task_snapshot_loader PRIVATE
    ${CMAKE_SOURCE_DIR}/src/plugin/runner/aiv_executor/aiv_task_snapshot_loader.cc
    ${CMAKE_SOURCE_DIR}/src/proxy/aiv_kernel/aiv_mode_stub/aiv_task.cc
    ${CMAKE_SOURCE_DIR}/src/proxy/aiv_kernel/aiv_mode_stub/ai_core_stub.cc
    ${CMAKE_SOURCE_DIR}/src/proxy/aiv_kernel/ascendc_stub/ascendc_base_stub.cc
)
set_target_properties(test_aiv_task_snapshot_loader PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)
target_include_directories(test_aiv_task_snapshot_loader PRIVATE
    ${THRID_PARTY_DIR}/nlohmann_json
    ${THRID_PARTY_DIR}
    ${CMAKE_SOURCE_DIR}/src/plugin/runner/aiv_executor
)
target_link_libraries(test_aiv_task_snapshot_loader log pthread)

add_executable(test_aiv_resource_manager aiv_resource_manager_test.cc ../main.cc)
target_sources(test_aiv_resource_manager PRIVATE
    ${CMAKE_SOURCE_DIR}/src/plugin/runner/aiv_executor/aiv_resource_manager.cc
)
target_link_libraries(test_aiv_resource_manager ccu_executor_common_objs ${CCU_EXECUTOR_COMMON_LIBS})
set_target_properties(test_aiv_resource_manager PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)

add_executable(test_aiv_graph_executor_mgr aiv_graph_executor_mgr_test.cc aiv_block_stub.cc ../main.cc)
target_sources(test_aiv_graph_executor_mgr PRIVATE
    ${CMAKE_SOURCE_DIR}/src/plugin/runner/aiv_executor/aiv_graph_executor_mgr.cc
    ${CMAKE_SOURCE_DIR}/src/proxy/aiv_kernel/aiv_mode_stub/aiv_task.cc
    ${CMAKE_SOURCE_DIR}/src/proxy/aiv_kernel/aiv_mode_stub/ai_core_stub.cc
    ${CMAKE_SOURCE_DIR}/src/proxy/aiv_kernel/ascendc_stub/ascendc_base_stub.cc
)
set_target_properties(test_aiv_graph_executor_mgr PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)
target_link_libraries(test_aiv_graph_executor_mgr log pthread)

add_executable(test_aiv_graph_executor aiv_graph_executor_test.cc ../main.cc)
target_sources(test_aiv_graph_executor PRIVATE
    ${CMAKE_SOURCE_DIR}/src/plugin/runner/aiv_executor/aiv_graph_executor.cc
    ${CMAKE_SOURCE_DIR}/src/plugin/runner/aiv_executor/aiv_resource_manager.cc
    ${CMAKE_SOURCE_DIR}/src/plugin/runner/aiv_executor/aiv_task_snapshot_loader.cc
    ${CMAKE_SOURCE_DIR}/src/proxy/aiv_kernel/aiv_mode_stub/aiv_task.cc
    ${CMAKE_SOURCE_DIR}/src/proxy/aiv_kernel/aiv_mode_stub/aiv_task_json.cc
    ${CMAKE_SOURCE_DIR}/src/proxy/aiv_kernel/aiv_mode_stub/ai_core_stub.cc
    ${CMAKE_SOURCE_DIR}/src/proxy/aiv_kernel/ascendc_stub/ascendc_base_stub.cc
    ${CMAKE_SOURCE_DIR}/src/store/store_sim_store_pub.cc
    ${CMAKE_SOURCE_DIR}/src/store/store_binary_data_operator.cc
    ${CMAKE_SOURCE_DIR}/src/runnerdb/tracedb/db_sim_sqlite_db.cc
)
target_link_libraries(test_aiv_graph_executor ccu_executor_common_objs ${CCU_EXECUTOR_COMMON_LIBS})
set_target_properties(test_aiv_graph_executor PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)
