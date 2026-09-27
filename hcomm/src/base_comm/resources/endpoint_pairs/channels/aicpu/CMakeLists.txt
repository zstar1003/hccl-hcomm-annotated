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
    aicpu_ts_urma_channel.cc
    aicpu_ts_p2p_channel.cc
    aicpu_ts_uboe_ub_rtp_channel_helper.cc
    aicpu_ts_uboe_channel.cc
    aicpu_ts_ub_rtp_channel.cc
    aicpu_ts_roce_channel.cc
    aicpu_ts_hccs_channel.cc
    aicpu_ts_roce_channel_v2.cc
    aicpu_ts_channel_helper.cc
    dev_rdma_connection_v2.cc
)

if(TARGET hcomm_base_obj)
    target_sources(hcomm_base_obj PRIVATE
        ${src_list}
    )
    target_include_directories(hcomm_base_obj PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/device
    )
endif()

add_subdirectory(device)