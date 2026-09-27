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
    ${CMAKE_CURRENT_SOURCE_DIR}/ccu_all_rank_param_recorder_v3.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ccu_loop_merge_v3.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ccu_task_common_v3.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ccu_task_transform_v3.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ccu_task_transform_instruct_a5_v3.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ccu_task_transform_instruct_common_v3.cc
)

if(BUILD_A6_CCU_INSTR)
    target_sources(checker PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/ccu_task_transform_instruct_a6_v3.cc
    )
endif()
