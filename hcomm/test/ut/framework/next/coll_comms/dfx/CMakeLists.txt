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
    # ut_HcommCcuProfiling_test.cc
    ut_HcommAivProfiling_test.cc
    ut_ccu_task_exception_decode_test.cc
    ut_hcclCommTaskExceptionLite.cc
    ut_cluster_monitor.cc
    ut_ccuTaskexception_test.cc
    ut_ccuTaskException_InitChannelMap_test.cc
    ut_ccuTaskException_GetChannelByHandle_test.cc
    ut_ccuTaskException_halfRtt_test.cc
    ut_GenerateTaskErrMsg_GetTaskParam_test.cc
    ut_hcclCommDfx.cc
    ut_hcclCommProfiling_test.cc
    ut_dfx_profiling_handler_lite.cc
    ut_dfx_profiling_reporter_lite.cc
    ut_dfx_profiling_command_handle_lite.cc
    ut_dfx_profiling_handler_host.cc
)

add_executable(hccl_utest_test_dfx_resource ${src_to_test_list})

#头文件搜索路径
target_include_directories(hccl_utest_test_dfx_resource PRIVATE
    ${UT_COMMON_INCLUDE_LIST}
    ${HCCL_CODE_ROOT}/external_depends/tsch
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/dfx/profiling/aicpu/common
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/communicator/device
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/communicator/device/resource_mgr/local/my_rank/comm_engine_reses
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/communicator/device/resource_mgr/remote
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/communicator/device/c_adpt
)
 
target_compile_definitions(hccl_utest_test_dfx_resource PRIVATE
    _GLIBCXX_USE_CXX11_ABI=0
    google=ascend_private
    -DOPEN_HCCL_TEST
    ORION_MODE
    OPEN_BUILD_PROJECT
)
 
target_compile_options(hccl_utest_test_dfx_resource PRIVATE
    -U_FORTIFY_SOURCE
    -fno-var-tracking-assignments
    -Os
    -O0 -g --coverage -fprofile-arcs -ftest-coverage -fno-access-control
    $<$<STREQUAL:${ENABLE_ASAN},true>:-fsanitize=address -fsanitize-recover=address,all -fno-omit-frame-pointer>
)

target_link_libraries(hccl_utest_test_dfx_resource PRIVATE
    $<BUILD_INTERFACE:intf_llt_pub>
    $<BUILD_INTERFACE:rdma_core_headers>
    $<$<STREQUAL:${ENABLE_ASAN},true>:-fsanitize=address>
    -pthread
    hccl_llt
    -Wl,--whole-archive
    -Wl,--no-whole-archive
    -lgcov
    -lrt
    -ldl
)
add_dependencies(hccl_utest_test_dfx_resource hccl_llt)

enable_testing()
add_test(NAME hccl_utest_test_dfx_resource COMMAND hccl_utest_test_dfx_resource)
