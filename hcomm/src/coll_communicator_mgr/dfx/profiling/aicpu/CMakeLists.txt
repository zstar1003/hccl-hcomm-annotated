# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(device_src_list
    ${CMAKE_CURRENT_SOURCE_DIR}/hcclCommProfilingLite.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/dfx_profiling_handler_lite.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/dfx_profiling_reporter_lite.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/dfx_profiling_command_handle_lite.cc
)

if(TARGET ccl_kernel)
    target_sources(ccl_kernel PRIVATE
        ${device_src_list}
    )
    target_include_directories(ccl_kernel PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${HCOMM_DIR}/src/base_comm/resources/comm_engine_res/threads
        ${CMAKE_CURRENT_SOURCE_DIR}/../../..
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../communicator/device
    )
endif()

if(TARGET hccl_v2)
    target_sources(hccl_v2 PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/dfx_profiling_handler_lite.cc
    )
endif()
