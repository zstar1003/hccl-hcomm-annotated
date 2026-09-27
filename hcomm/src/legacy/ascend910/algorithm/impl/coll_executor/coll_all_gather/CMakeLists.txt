# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(src_list)
file(GLOB src_list "${CMAKE_CURRENT_SOURCE_DIR}/*.cc")

if(TARGET hccl_alg)
    target_sources(hccl_alg PRIVATE
      ${src_list}
    )
endif()

if(TARGET ccl_kernel)
    target_sources(ccl_kernel PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/coll_all_gather_executor.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/coll_all_gather_comm_executor.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/coll_all_gather_ring_executor.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/coll_all_gather_ring_for_910_93_executor.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/coll_aligned_all_gather_double_ring_for_910_93_executor.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/coll_all_gather_mesh_executor.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/coll_all_gather_mesh_opbase_executor.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/coll_all_gather_mesh_opbase_pipeline_executor.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/coll_all_gather_single_rank_executor.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/coll_all_gather_small_count_executor.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/coll_all_gather_semi_ring_executor.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/coll_all_gather_mix_executor.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/coll_all_gather_hccs_sio_executor.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/coll_all_gather_ring_zerocopy_executor.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/coll_all_gather_ring_zerocopy_exchange_executor.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/coll_all_gather_ring_zerocopy_pipeline_executor.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/coll_all_gather_ars_for_910_93_executor.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/coll_all_gather_mid_count_for_910_93_executor.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/coll_all_gather_mesh_graph_executor.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/coll_all_gather_mesh_graph_pipeline_executor.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/coll_all_gather_pipeline_for_910_93_executor.cc
    )
    target_include_directories(ccl_kernel PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
endif()

add_subdirectory(310P)
