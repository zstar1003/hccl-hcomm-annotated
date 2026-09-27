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
    ${CMAKE_CURRENT_SOURCE_DIR}/task_graph_generator.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ccu_task_transform.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ccu_task_transform_instruct_a5.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ccu_task_transform_instruct_common.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ccu_all_rank_param_recorder.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ccu_task_common.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/type_conversion.cc
    # ${CMAKE_CURRENT_SOURCE_DIR}/task_graph_generator_aiv.cc
)

if(BUILD_A6_CCU_INSTR)
    add_definitions(-DBUILD_A6_CCU_INSTR)
endif()

if(BUILD_A6_CCU_INSTR)
    target_sources(checker PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/ccu_task_transform_instruct_a6.cc
    )
endif()