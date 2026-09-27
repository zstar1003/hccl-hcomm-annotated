# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(hccl_utest_platform_dispatcher_list
    main.cc
    ut_aicpu_dispatcher_utest.cc
    ut_dispatcher_graph.cc
)
add_executable(hccl_utest_platform_dispatcher  ${hccl_utest_platform_dispatcher_list})

target_include_directories(hccl_utest_platform_dispatcher  PRIVATE
    ${UT_COMMON_INCLUDE_LIST}
)

target_compile_definitions(hccl_utest_platform_dispatcher    PRIVATE
    HALF_T=float
    google=ascend_private
    CCL_KERNEL
    -DCCL_LLT
)

target_compile_options(hccl_utest_platform_dispatcher   PRIVATE 
    -U_FORTIFY_SOURCE
    -O0 -g --coverage -fprofile-arcs -ftest-coverage -fno-access-control
    $<$<COMPILE_LANGUAGE:CXX>:-std=c++17>
)


target_link_libraries(hccl_utest_platform_dispatcher  PRIVATE
    $<BUILD_INTERFACE:intf_llt_pub>
    $<BUILD_INTERFACE:rdma_core_headers>
    -pthread
    hccl_llt
    -Wl,--whole-archive
    -Wl,--no-whole-archi
    -lgcov
    -lrt
    -pthread
)

add_dependencies(hccl_utest_platform_dispatcher hccl_llt)

enable_testing()
add_test(NAME hccl_utest_platform_dispatcher COMMAND hccl_utest_platform_dispatcher)
