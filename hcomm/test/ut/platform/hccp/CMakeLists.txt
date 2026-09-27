# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../..)

include(${HCCL_CODE_ROOT}/cmake/hcomm_utils.cmake)
add_cann_third_party(openssl)

# --- UT Time-constant acceleration values (single source of truth) ---
# To add a new override: add a set() here + add #undef/#define in the .h.in template.
set(UT_RA_THREAD_TRY_TIME       5000)
set(UT_RA_THREAD_SLEEP_TIME     100)
set(UT_HDC_USLEEP_TIME          1000)
set(UT_RS_USLEEP_TIME           1000)
set(UT_RS_TRY_TIME              5000)
set(UT_RS_PING_PERIOD_TIME_USEC 1000)

# --- Generate shadow headers to build directory ---
set(SHADOW_OUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/shadow)
file(MAKE_DIRECTORY ${SHADOW_OUT_DIR})

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/shadow/ra_rs_comm.h.in
               ${SHADOW_OUT_DIR}/ra_rs_comm.h @ONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/shadow/rs_inner.h.in
               ${SHADOW_OUT_DIR}/rs_inner.h @ONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/shadow/rs_ping_inner.h.in
               ${SHADOW_OUT_DIR}/rs_ping_inner.h @ONLY)

add_subdirectory(hccp_service)
add_subdirectory(rdma_agent)
add_subdirectory(rdma_service_abnormal)
add_subdirectory(rdma_service_normal)