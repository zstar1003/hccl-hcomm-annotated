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
    ${CMAKE_CURRENT_SOURCE_DIR}/aligned_reduce_scatter_double_ring_with_serial_local_copy.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/aligned_reduce_scatter_double_ring.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/reduce_scatter_halving_doubling.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/reduce_scatter_mesh_atomic_opbase.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/reduce_scatter_mesh_atomic.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/reduce_scatter_mesh.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/reduce_scatter_recursive_hd.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/reduce_scatter_ring_concurrent_direct.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/reduce_scatter_ring.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/reduce_scatter_mesh_mix.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/reduce_scatter_mesh_mix_single_stream.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/reduce_scatter_graph_pipeline.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/reduce_scatter_nhr.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/reduce_scatter_nhr_v1.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/reduce_scatter_nb.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/reduce_scatter_ahc.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/reduce_scatter_ahc_broke.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/reduce_scatter_hd_stage.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/reduce_scatter_local_reduce.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/reduce_scatter_pipeline.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/reduce_scatter_plant_local_reduce.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/reduce_scatter_plant_local_reduce_combine.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/reduce_scatter_unified_march.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/reduce_scatter_hccs_sio.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/reduce_scatter_slim_ring.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/reduce_scatter_v_pipeline.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/reduce_scatter_multi_deter_pipeline.cc

)

if(TARGET hccl_alg)
    target_sources(hccl_alg PRIVATE
        ${src_list_pub}
    )
endif()

if(TARGET ccl_kernel)
    target_sources(ccl_kernel PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/aligned_reduce_scatter_double_ring_with_serial_local_copy.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/aligned_reduce_scatter_double_ring.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/reduce_scatter_ring_concurrent_direct.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/reduce_scatter_halving_doubling.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/reduce_scatter_recursive_hd.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/reduce_scatter_mesh.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/reduce_scatter_ring.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/reduce_scatter_mesh_atomic.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/reduce_scatter_mesh_mix.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/reduce_scatter_mesh_mix_single_stream.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/reduce_scatter_mesh_atomic_opbase.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/reduce_scatter_nhr.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/reduce_scatter_nhr_v1.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/reduce_scatter_nb.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/reduce_scatter_ahc.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/reduce_scatter_ahc_broke.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/reduce_scatter_hd_stage.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/reduce_scatter_local_reduce.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/reduce_scatter_pipeline.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/reduce_scatter_unified_march.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/reduce_scatter_hccs_sio.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/reduce_scatter_plant_local_reduce.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/reduce_scatter_plant_local_reduce_combine.cc
    )
    target_include_directories(ccl_kernel PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
endif()
