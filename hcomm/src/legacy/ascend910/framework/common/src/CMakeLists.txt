# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(src_list
    ${CMAKE_CURRENT_SOURCE_DIR}/config.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/param_check.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/json_utils.cc
)

if(TARGET hcomm)
    target_sources(hcomm PRIVATE
        ${src_list}
        ${CMAKE_CURRENT_SOURCE_DIR}/hccl_hash_utils.cc
    )
    target_include_directories(hcomm PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/host
        ${CMAKE_CURRENT_SOURCE_DIR}/h2d_dto
        ${CMAKE_CURRENT_SOURCE_DIR}/mgr
    )
endif()

if(TARGET ccl_kernel)
    target_sources(ccl_kernel PRIVATE
        ${src_list}
    )
    target_include_directories(ccl_kernel PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/aicpu
        ${CMAKE_CURRENT_SOURCE_DIR}/mgr
        ${CMAKE_CURRENT_SOURCE_DIR}/config
    )
endif()

add_subdirectory(opexecounter)
add_subdirectory(topo)
add_subdirectory(task)
add_subdirectory(exception)
add_subdirectory(h2d_tlv)
add_subdirectory(thread)
add_subdirectory(config)
add_subdirectory(onesided_memory_management)
add_subdirectory(order_launch)
