# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(src_to_test_list
    main.cc
    hccl_api_base_test.cc
    ut_HcclAllGatherV_API_test.cc
    ut_HcclBatchSendRecv_API_test.cc
    ut_HcclReduceScatterV_API_test.cc
    ut_HcclAllReduce_API_test.cc
    ut_HcclBroadcast_API_test.cc
    ut_HcclScatter_API_test.cc
    ut_HcclAlltoAll_API_test.cc
    ut_HcclRecv_API_test.cc
    ut_HcclSend_API_test.cc
    ut_HcclAlltoAllV_API_test.cc
    ut_HcclAlltoAllVC_API_test.cc
    ut_HcclReduce_API_test.cc
    ut_HcclAllGather_API_test.cc
    ut_HcclBarrier_API_test.cc
    ut_HcclReduceScatter_API_test.cc
    ut_HcomGetCommHandleByGroup_API_test.cc
    ut_op_base_misc_test.cc
    # ut_op_base_v2_test.cc
)

add_executable(hccl_utest_framework_op_base_api ${src_to_test_list})

#头文件搜索路径
target_include_directories(hccl_utest_framework_op_base_api PRIVATE
    ${UT_COMMON_INCLUDE_LIST}
)

target_compile_definitions(hccl_utest_framework_op_base_api PRIVATE
    _GLIBCXX_USE_CXX11_ABI=0
    google=ascend_private
    -DOPEN_HCCL_TEST
    OPEN_BUILD_PROJECT
)

target_compile_options(hccl_utest_framework_op_base_api PRIVATE
    -U_FORTIFY_SOURCE
    -fno-var-tracking-assignments
    -Os
    -O0 -g --coverage -fprofile-arcs -ftest-coverage -fno-access-control
    $<$<STREQUAL:${ENABLE_ASAN},true>:-fsanitize=address -fsanitize-recover=address,all -fno-omit-frame-pointer>
)

target_link_libraries(hccl_utest_framework_op_base_api PRIVATE
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
add_dependencies(hccl_utest_framework_op_base_api hccl_llt)

enable_testing()
add_test(NAME hccl_utest_framework_op_base_api COMMAND hccl_utest_framework_op_base_api)
