# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

file(GLOB_RECURSE CCU_REP_SOURCES "*.cc")

if(TARGET hcomm_base_obj)
    target_sources(hcomm_base_obj
        PRIVATE
        ${CCU_REP_SOURCES}
    )
    target_include_directories(hcomm_base_obj PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/interface
        ${CMAKE_CURRENT_SOURCE_DIR}/context
        ${CMAKE_CURRENT_SOURCE_DIR}/reps
        ${CMAKE_CURRENT_SOURCE_DIR}/reps/arithmetic
        ${CMAKE_CURRENT_SOURCE_DIR}/reps/common
        ${CMAKE_CURRENT_SOURCE_DIR}/reps/control
        ${CMAKE_CURRENT_SOURCE_DIR}/reps/data
        ${CMAKE_CURRENT_SOURCE_DIR}/reps/logical
        ${CMAKE_CURRENT_SOURCE_DIR}/reps/loop
        ${CMAKE_CURRENT_SOURCE_DIR}/reps/shift
        ${CMAKE_CURRENT_SOURCE_DIR}/reps/sync
        ${CMAKE_CURRENT_SOURCE_DIR}/reps/translator
        ${CMAKE_CURRENT_SOURCE_DIR}/../ccu_microcode
        ${CMAKE_CURRENT_SOURCE_DIR}/../ccu_microcode_opt
        ${CMAKE_CURRENT_SOURCE_DIR}/../pub_inc
    )
endif()
