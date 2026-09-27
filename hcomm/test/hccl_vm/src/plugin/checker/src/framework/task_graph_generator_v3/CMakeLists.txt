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
    ${CMAKE_CURRENT_SOURCE_DIR}/aiv_graph_generator_v3/aiv_snapshot_json_loader_v3.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/aiv_graph_generator_v3/aiv_task_transform_v3.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/task_graph_generator_v3.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/task_graph_mem_conflict_v3.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/task_graph_sync_conflict_v3.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/task_graph_reachability_v3.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/task_graph_semantic_check_v3.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/task_graph_single_task_check_v3.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/task_meta_translator_v3.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/task_def_v3.cc
)

add_subdirectory(ccu_graph_generator_v3)
