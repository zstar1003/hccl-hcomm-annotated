# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(hccl_ut_aicpu_task_cache_src_list
    main.cc
    ut_aicpu_task_cache_entry.cc
    ut_aicpu_task_cache.cc
    ut_aicpu_task_cache_manager.cc
    ut_aicpu_task_cache_c_adpt.cc
    ut_aicpu_task_utils.cc
    ${HCCL_CODE_ROOT}/test/ut/stub/legacy/llt_stub_ub_transport_lite_impl.cc
)

add_executable(hccl_ut_aicpu_task_cache ${hccl_ut_aicpu_task_cache_src_list})

target_include_directories(hccl_ut_aicpu_task_cache PRIVATE
    ${UT_COMMON_INCLUDE_LIST}
    ${HCCL_CODE_ROOT}/src/base_comm/primitives/aicpu/task_cache
)

target_compile_definitions(hccl_ut_aicpu_task_cache PRIVATE
    HALF_T=float
    google=ascend_private
    CCL_KERNEL
    -DCCL_LLT
    OPEN_BUILD_PROJECT
)

target_compile_options(hccl_ut_aicpu_task_cache PRIVATE
    -U_FORTIFY_SOURCE
    -fno-var-tracking-assignments
    -Os
    -O0 -g --coverage -fprofile-arcs -ftest-coverage -fno-access-control
)

target_link_libraries(hccl_ut_aicpu_task_cache PRIVATE
    $<BUILD_INTERFACE:intf_llt_pub>
    $<BUILD_INTERFACE:rdma_core_headers>
    -pthread
    hccl_llt
    -Wl,--whole-archive
    -Wl,--no-whole-archi
    -lgcov
    -lrt
    -ldl
)
add_dependencies(hccl_ut_aicpu_task_cache hccl_llt)

enable_testing()
add_test(NAME hccl_ut_aicpu_task_cache COMMAND hccl_ut_aicpu_task_cache)
