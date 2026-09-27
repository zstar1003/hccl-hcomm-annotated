# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

file(GLOB_RECURSE CCU_REP_SOURCES "${HCCL_V2_CODE_ROOT}/unified_platform/ccu/*.cpp")

target_sources(hccl_v2_utest 
    PUBLIC
    ut_ccu_rep_context.cpp
    ut_ccu_rep_translator.cpp
    ut_ccu_context.cpp
    ut_ccu_context_tracker.cpp
    ut_ccu_rep_load.cpp
    ut_ccu_rep_store.cpp
    ut_ccu_dfx.cpp

    ${CCU_REP_SOURCES}
)