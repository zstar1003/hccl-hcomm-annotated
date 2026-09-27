# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(aicpu_kfc_utest_aicpu_list
    main.cc
    ut_aicpu_kfc_interface_utest.cc
    ut_aicpu_kfc_trace_utest.cc
    ut_aicpu_kfc_unfold_utest.cc
)
add_executable(aicpu_kfc_utest_aicpu  ${aicpu_kfc_utest_aicpu_list})

target_include_directories(aicpu_kfc_utest_aicpu PRIVATE
    ${UT_COMMON_INCLUDE_LIST}
    ${CMAKE_CURRENT_SOURCE_DIR}/../stub
)

target_compile_definitions(aicpu_kfc_utest_aicpu PRIVATE
    HALF_T=float
    google=ascend_private
    CCL_KERNEL
    -DCCL_LLT
)

target_compile_options(aicpu_kfc_utest_aicpu PRIVATE
    -U_FORTIFY_SOURCE
    -Os
    -O1 -g
    -fno-var-tracking-assignments
)

target_link_libraries(aicpu_kfc_utest_aicpu  PRIVATE
    $<BUILD_INTERFACE:intf_llt_pub>
    $<BUILD_INTERFACE:rdma_core_headers>
    $<$<STREQUAL:${ENABLE_ASAN},true>:-fsanitize=address>
    -pthread
    hccl_llt
    -Wl,--whole-archive
    -Wl,--no-whole-archi
    -lgcov
    -lrt
    -ldl
)

enable_testing()
add_test(NAME aicpu_kfc_utest_aicpu COMMAND aicpu_kfc_utest_aicpu)
