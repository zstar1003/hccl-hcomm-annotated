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
    coll_comm_aicpu_mgr.cc
    coll_comm_aicpu.cc
    hccl_aicpu_hdc_handler.cc
    coll_comm_aicpu_destroy_func.cc
)

add_subdirectory(c_adpt)

if(TARGET ccl_kernel)
    target_sources(ccl_kernel PRIVATE
        ${src_list}
    )
    target_include_directories(ccl_kernel PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/c_adpt
        ${CMAKE_CURRENT_SOURCE_DIR}/resource_mgr/local/my_rank/comm_engine_reses
        ${CMAKE_CURRENT_SOURCE_DIR}/resource_mgr/remote
        ${HCOMM_DIR}/src/coll_communicator_mgr/dfx/ns_recovery/aicpu
    )
endif()

if(TARGET hcomm)
    target_include_directories(hcomm PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
endif()

add_subdirectory(resource_mgr)
