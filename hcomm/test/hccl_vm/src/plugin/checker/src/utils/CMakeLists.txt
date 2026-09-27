# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

target_sources(checker PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/check_utils.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/file_utils.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/setting_manager.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/task_utils.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/sim_task.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/task_ccu.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/storage_manager.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/binary_data_operator.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/dump/dump_json_utils.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/dump/dump_manager.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/dump/dump_memory_timeline.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/dump/dump_graph.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/dump/dump_run_manifest.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/dump/validation_issue_recorder.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/dump/msgpack_writer.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/dump_v3/dump_v3_manager.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/dump_v3/dump_v3_dag.cc
)
