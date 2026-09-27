# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

add_subdirectory(system)

target_sources(hccl_v2_utest 
    PUBLIC
    ut_adapter_hccp.cc
    ut_adapter_rts.cc
    ut_adapter_trace.cc
    ${HCCL_V2_CODE_ROOT}/unified_platform/external_system/orion_adapter_rts.cc
    ${HCCL_V2_CODE_ROOT}/unified_platform/external_system/orion_adapter_hccp.cc
    ${HCCL_V2_CODE_ROOT}/unified_platform/external_system/orion_adapter_tsd.cc
    ${HCCL_V2_CODE_ROOT}/unified_platform/external_system/orion_adapter_trace.cc
)
