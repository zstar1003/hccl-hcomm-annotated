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
    orion_adpt_utils.cc
    hccp_tlv_hdc_mgr.cc
    eid_info_mgr.cc
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/common/tp_qos.cc
    tp_mgr.cc
    dlurma_function.cc
    hccl_dl.cc
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

add_subdirectory(debug)
add_subdirectory(device)
