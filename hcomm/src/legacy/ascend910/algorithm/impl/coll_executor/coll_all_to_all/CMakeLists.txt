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
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_all_to_all_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_all_to_all_executor_host.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_all_to_all_v_2level_pipeline_excecutor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_all_to_all_v_fullmesh_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_all_to_all_v_staged_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_all_to_all_single_rank_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_all_to_all_mesh_aiv_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_all_to_all_mesh_aiv_smallcount_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_all_to_all_v_direct_fullmesh_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_all_to_all_staged_aiv_rdma_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_all_to_all_mesh_aiv_for_910_93_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_all_to_all_v_continuous_pipeline_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_all_to_all_direct_fullmesh_aiv_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_all_to_all_v_aiv_direct_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_all_to_all_symmetric_memory_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_all_to_all_v_pipeline_for_910_93_executor.cc
)

if(TARGET hccl_alg)
    target_sources(hccl_alg PRIVATE
        ${src_list}
    )
endif()

if(TARGET ccl_kernel)
    target_sources(ccl_kernel PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/coll_all_to_all_executor.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/coll_all_to_all_executor_aicpu.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/coll_all_to_all_single_rank_executor.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/coll_all_to_all_v_fullmesh_executor.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/coll_all_to_all_v_direct_fullmesh_executor.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/coll_all_to_all_symmetric_memory_executor.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/coll_all_to_all_v_2level_pipeline_excecutor.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/coll_all_to_all_v_continuous_pipeline_executor.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/coll_all_to_all_v_staged_executor.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/coll_all_to_all_v_pipeline_for_910_93_executor.cc
    )
    target_include_directories(ccl_kernel PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
endif()

add_subdirectory(310P)
