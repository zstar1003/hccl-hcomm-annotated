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
    ${CMAKE_CURRENT_SOURCE_DIR}/comm_base.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/comm_factory.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/comm_ring.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/comm_p2p.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/comm_mesh.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/comm_halving_doubling.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/comm_star.cc
)

if(TARGET hccl_alg)
    target_sources(hccl_alg PRIVATE
        ${src_list}
    )
endif()

if(TARGET ccl_kernel)
    target_sources(ccl_kernel PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/comm_base.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/comm_ring.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/comm_p2p.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/comm_mesh.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/comm_halving_doubling.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/comm_star.cc
    )
    target_include_directories(ccl_kernel PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
endif()
