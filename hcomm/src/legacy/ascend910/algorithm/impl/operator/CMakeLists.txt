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
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_alg_operator.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/scatter_operator.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/reduce_operator.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/all_reduce_operator.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/reduce_scatter_operator.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/reduce_scatter_v_operator.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/all_gather_operator.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/all_gather_v_operator.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/broadcast_operator.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/alltoall_operator.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/batchsendrecv_operator.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/send_operator.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/receive_operator.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/batch_write_operator.cc
)

if(TARGET hccl_alg)
    target_sources(hccl_alg PRIVATE
        ${src_list}
    )
endif()

add_subdirectory(registry)