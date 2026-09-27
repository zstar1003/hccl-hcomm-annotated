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
    ${CMAKE_CURRENT_SOURCE_DIR}/all_reduce_recursive_hd.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/all_reduce_ring.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/all_reduce_nhr_oneshot.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/all_reduce_nhr.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/all_reduce_nhr_v1.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/all_reduce_ahc.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/all_reduce_ahc_broke.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/all_reduce_nb.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/all_reduce_reduce_broadcast.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/all_reduce_mesh_opbase.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/all_reduce_mesh_oneshot.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/all_reduce_doubling.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/all_reduce_doubling_direct.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/all_reduce_local_reduce_bcast.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/all_reduce_local_reduce.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/all_reduce_chunk_mesh.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/all_reduce_opbase_pipeline.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/all_reduce_hd_optim.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/all_reduce_graph_pipeline.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/all_reduce_doubling_local_reduce.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/all_reduce_multi_deter_pipeline.cc
)

if(TARGET hccl_alg)
    target_sources(hccl_alg PRIVATE
        ${src_list_pub}
    )
endif()

if(TARGET ccl_kernel)
    target_sources(ccl_kernel PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/all_reduce_recursive_hd.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/all_reduce_ring.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/all_reduce_nhr_oneshot.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/all_reduce_nhr.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/all_reduce_nhr_v1.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/all_reduce_ahc.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/all_reduce_ahc_broke.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/all_reduce_nb.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/all_reduce_reduce_broadcast.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/all_reduce_mesh_opbase.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/all_reduce_mesh_oneshot.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/all_reduce_doubling.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/all_reduce_doubling_direct.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/all_reduce_local_reduce_bcast.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/all_reduce_local_reduce.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/all_reduce_chunk_mesh.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/all_reduce_opbase_pipeline.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/all_reduce_hd_optim.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/all_reduce_graph_pipeline.cc
    )
    target_include_directories(ccl_kernel PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
endif()
