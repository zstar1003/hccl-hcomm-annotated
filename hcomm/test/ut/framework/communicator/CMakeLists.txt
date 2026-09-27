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
    ut_HcclCommActivateCommMemory_API_test.cc
    ut_HcclCommChannel_API_test.cc
    ut_HcclCommConfigInit_API_test.cc
    ut_HcclCommDeactivateCommMemory_API_test.cc
    ut_HcclCommDestroy_API_test.cc
    ut_HcclCommEngineCtx_API_test.cc
    ut_HcclCommGetHandleWithName_API_test.cc
    ut_HcclCommGetStatus_API_test.cc
    ut_HcclCommRegCommStateCallback_API_test.cc
    ut_HcclCommInitAll_API_test.cc
    ut_HcclCommInitClusterInfoConfig_API_test.cc
    ut_HcclCommInitClusterInfo_API_test.cc
    ut_HcclCommInitRootInfoConfig_API_test.cc
    ut_HcclCommInitRootInfo_API_test.cc
    ut_HcclCommInitRootInfoScalable_API_test.cc
    ut_HcclCommResume_API_test.cc
    ut_HcclCommSetMemoryRange_API_test.cc
    ut_HcclCommSuspend_API_test.cc
    ut_HcclCommUnsetMemoryRange_API_test.cc
    ut_HcclCommWorkingDevNicSet_API_test.cc
    ut_HcclCreateSubCommConfig_API_test.cc
    ut_HcclGetCommAsyncError_API_test.cc
    ut_HcclGetCommConfigCapability_API_test.cc
    ut_HcclGetCommName_API_test.cc
    ut_HcclGetConfig_API_test.cc
    ut_HcclGetErrorString_API_test.cc
    ut_HcclGetRankId_API_test.cc
    ut_HcclGetRankSize_API_test.cc
    ut_HcclGetRootInfo_API_test.cc
    ut_HcclGetStreamNumforAivMode_API_test.cc
    ut_HcclChannelGetHcclBuffer_API_test.cc
    ut_HcclGroupLeaderListen_test.cc
    # ut_HcclSetConfig_API_test.cc
    # ut_HcclTopoInfoDetect_API_test.cc
    ut_hccl_comm_host.cc
    ut_ind_op_transport_alloc.cc
    ut_aclgraph_communicator.cc
    ut_coll_comm_mgr.cc
)

add_executable(hccl_utest_framework_communicator ${src_to_test_list})

#头文件搜索路径
target_include_directories(hccl_utest_framework_communicator PRIVATE
    ${UT_COMMON_INCLUDE_LIST}
)

target_compile_definitions(hccl_utest_framework_communicator PRIVATE
    _GLIBCXX_USE_CXX11_ABI=0
    google=ascend_private
    -DOPEN_HCCL_TEST
    OPEN_BUILD_PROJECT
)

target_compile_options(hccl_utest_framework_communicator PRIVATE
    -U_FORTIFY_SOURCE
    -fno-var-tracking-assignments
    -Os
    -O0 -g --coverage -fprofile-arcs -ftest-coverage -fno-access-control
    $<$<STREQUAL:${ENABLE_ASAN},true>:-fsanitize=address -fsanitize-recover=address,all -fno-omit-frame-pointer>
)

target_link_libraries(hccl_utest_framework_communicator PRIVATE
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
add_dependencies(hccl_utest_framework_communicator hccl_llt)

enable_testing()
add_test(NAME hccl_utest_framework_communicator COMMAND hccl_utest_framework_communicator)
