# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(src_list_pub
    ${CMAKE_CURRENT_SOURCE_DIR}/alltoallv_pairwise.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/alltoallv_staged_base.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/alltoallv_staged_calculator.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/alltoallv_staged_mesh.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/alltoallv_staged_pairwise.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/alltoallv_for_310p.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/alltoallv_direct_fullmesh.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/alltoallv_continuous_pipeline.cc
)

if(TARGET hccl_alg)
    target_sources(hccl_alg PRIVATE
        ${src_list_pub}
    )
endif()

if(TARGET ccl_kernel)
    target_sources(ccl_kernel PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/alltoallv_pairwise.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/alltoallv_staged_base.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/alltoallv_staged_pairwise.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/alltoallv_staged_mesh.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/alltoallv_staged_calculator.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/alltoallv_for_310p.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/alltoallv_direct_fullmesh.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/alltoallv_continuous_pipeline.cc
    )
    target_include_directories(ccl_kernel PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
endif()
