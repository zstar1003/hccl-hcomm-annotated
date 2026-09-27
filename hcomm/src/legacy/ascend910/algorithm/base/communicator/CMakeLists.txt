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
    ${CMAKE_CURRENT_SOURCE_DIR}/topo_info_extractor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/calc_hd_transport_req.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/calc_mesh_transport_req.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/calc_partial_mesh_transport_req.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/calc_ring_transport_req.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/calc_transport_req_base.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/calc_p2p_transport_req.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/calc_hccs_plus_sio_transport_req.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/calc_nb_transport_req.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/calc_nhr_transport_req.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/calc_nhr_v1_transport_req.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/calc_ahc_transport_req_base.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/calc_ahc_transport_req.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/calc_ahc_broke_transport_req.cc
)

if(TARGET hccl_alg)
    target_sources(hccl_alg PRIVATE
        ${src_list}
    )
endif()

if(TARGET ccl_kernel)
    target_sources(ccl_kernel PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/calc_hd_transport_req.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/calc_mesh_transport_req.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/calc_partial_mesh_transport_req.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/calc_ring_transport_req.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/calc_transport_req_base.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/calc_p2p_transport_req.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/calc_hccs_plus_sio_transport_req.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/calc_ahc_broke_transport_req.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/calc_ahc_transport_req.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/calc_ahc_transport_req_base.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/calc_nb_transport_req.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/calc_nhr_transport_req.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/calc_nhr_v1_transport_req.cc
    )
    target_include_directories(ccl_kernel PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
endif()

add_subdirectory(legacy)
