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
    ${CMAKE_CURRENT_SOURCE_DIR}/recursive_halvingdoubling_base.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/nonuniform_hierarchical_ring_base.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/nonuniform_hierarchical_ring_v1_base.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/asymmetric_hierarchical_concatenate_base.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/asymmetric_hierarchical_concatenate_alg_template_base.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/asymmetric_hierarchical_concatenate_base.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/asymmetric_hierarchical_concatenate_nslb.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/nonuniform_bruck_base.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/alg_template_register.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/alg_template_base.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/alg_template_multi_deter_pipeline.cc
)

if(TARGET hccl_alg)
    target_sources(hccl_alg PRIVATE
        ${src_list_pub}
    )
endif()

if(TARGET ccl_kernel)
    target_sources(ccl_kernel PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/alg_template_register.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/alg_template_base.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/nonuniform_hierarchical_ring_base.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/nonuniform_hierarchical_ring_v1_base.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/nonuniform_bruck_base.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/asymmetric_hierarchical_concatenate_base.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/asymmetric_hierarchical_concatenate_alg_template_base.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/asymmetric_hierarchical_concatenate_nslb.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/recursive_halvingdoubling_base.cc
    )
    target_include_directories(ccl_kernel PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
endif()

add_subdirectory(component)
add_subdirectory(temp_alltoall)
add_subdirectory(temp_all_gather)
add_subdirectory(temp_all_reduce)
add_subdirectory(temp_alltoallv)
add_subdirectory(temp_broadcast)
add_subdirectory(temp_gather)
add_subdirectory(temp_reduce)
add_subdirectory(temp_reduce_scatter)
add_subdirectory(temp_scatter)
add_subdirectory(temp_send_recv)
