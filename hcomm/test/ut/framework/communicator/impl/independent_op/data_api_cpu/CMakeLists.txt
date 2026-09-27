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
    ut_cpu_HcommChannelNotifyRecordOnThread.cc
    ut_cpu_HcommChannelNotifyWaitOnThread.cc
    ut_cpu_HcommWriteNbiOnThread.cc
    ut_cpu_HcommWriteWithNotifyNbiOnThread.cc
    ut_cpu_HcommReadNbiOnThread.cc
    ut_cpu_HcommFenceOnThread.cc
    ut_cpu_HcommChannelFenceOnThread.cc
    ut_cpu_HcommDpuChannelRegisterDfx.cc
    ut_cpu_HcommPluginChannelOps.cc
    ${HCCL_CODE_ROOT}/src/base_comm/primitives/api_c_adpt/cpu_primitives_c_adpt.cc
)

add_executable(hcomm_utest_api_data_cpu ${src_to_test_list})

#头文件搜索路径
target_include_directories(hcomm_utest_api_data_cpu PRIVATE
    ${UT_COMMON_INCLUDE_LIST}
)

target_compile_definitions(hcomm_utest_api_data_cpu PRIVATE
    _GLIBCXX_USE_CXX11_ABI=0
    google=ascend_private
    -DOPEN_HCCL_TEST
    OPEN_BUILD_PROJECT
    ENABLE_EXPERIMENTAL
)

target_compile_options(hcomm_utest_api_data_cpu PRIVATE
    -U_FORTIFY_SOURCE
    -fno-var-tracking-assignments
    -Os
    -O0 -g --coverage -fprofile-arcs -ftest-coverage -fno-access-control
    $<$<STREQUAL:${ENABLE_ASAN},true>:-fsanitize=address -fsanitize-recover=address,all -fno-omit-frame-pointer>
)

target_link_libraries(hcomm_utest_api_data_cpu PRIVATE
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
add_dependencies(hcomm_utest_api_data_cpu hccl_llt)

enable_testing()
add_test(NAME hcomm_utest_api_data_cpu COMMAND hcomm_utest_api_data_cpu)
