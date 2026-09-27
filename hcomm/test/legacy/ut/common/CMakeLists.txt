# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

add_subdirectory(utils)
add_subdirectory(exception)

target_sources(hccl_v2_utest 
    PUBLIC
    slog_stub_ut.cc
    driver_stub.cc
    ut_log.cc
    ut_rma_buffer_mgr_log.cc
    driver_stub.cc
    ut_binary_stream.cc
    ut_checkcrc.cc
    ut_stream_utils.cc
    ut_ip_address.cc
    ${HCCL_V2_CODE_ROOT}/common/log.cc
    ${HCCL_V2_CODE_ROOT}/common/sal.cc
    ${HCCL_V2_CODE_ROOT}/common/checkcrc.cc
    ${HCCL_LLT_ROOT}/../ut/stub/llt_stub_mm.c
    ${HCCL_LLT_ROOT}/../ut/stub/llt_stub_sec.c
)

