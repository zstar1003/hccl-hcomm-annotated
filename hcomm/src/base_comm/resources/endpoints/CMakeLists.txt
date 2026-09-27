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
    cpu_roce_endpoint.cc
    cpu_urma_endpoint.cc
    urma_endpoint.cc
    ub_mem_endpoint.cc
    aicpu_ts_roce_endpoint.cc
    aicputs_hccs_endpoint.cc
    endpoint.cc
    uboe_endpoint.cc
    ub_rtp_endpoint.cc
)

if(TARGET hcomm_base_obj)
    target_sources(hcomm_base_obj PRIVATE
        ${src_list}
    )
    target_include_directories(hcomm_base_obj PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/src/base_comm/resources/hccp/inc/private/network
    )
endif()

add_subdirectory(mgr)
add_subdirectory(server_socket)
add_subdirectory(server_socket_context)
add_subdirectory(comm_queue_context)
add_subdirectory(net_dev)
add_subdirectory(dfx)