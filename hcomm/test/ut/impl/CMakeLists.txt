# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(hccl_utest_impl_list
    main.cc
    ut_hccl_impl_common_new.cc
    # ut_hccl_impl_common.cc
    # ut_hccl_impl.cc
    ut_hccl_impl_new.cc
    ut_coll_batch_send_recv_group_executor.cc
    # ut_zero_copy_memory_agent.cc
    ut_preempt_port_manager.cc
    ut_template_base.cc
    ut_alltoallv_continuous_pipeline.cc
    # ut_netdevice.cc
    ut_global_mem_mgr.cc
    ut_global_mem_mgr_multhread.cc
    ut_local_dispatcher_ctx.cc
)
add_executable(hccl_utest_impl ${hccl_utest_impl_list})

target_include_directories(hccl_utest_impl PRIVATE
    ${UT_COMMON_INCLUDE_LIST}
)

target_compile_definitions(hccl_utest_impl PRIVATE
    _GLIBCXX_USE_CXX11_ABI=0
    HALF_T=float
    google=ascend_private
    CCL_KERNEL
    CCL_LLT
    OPEN_BUILD_PROJECT
)

target_compile_options(hccl_utest_impl PRIVATE
    -U_FORTIFY_SOURCE
    -fno-var-tracking-assignments
    -Os
    -O0 -g --coverage -fprofile-arcs -ftest-coverage -fno-access-control
    $<$<STREQUAL:${ENABLE_ASAN},true>:-fsanitize=address -fsanitize-recover=address,all -fno-omit-frame-pointer>
)

target_link_libraries(hccl_utest_impl PRIVATE
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
add_dependencies(hccl_utest_impl hccl_llt)

enable_testing()
add_test(NAME hccl_utest_impl COMMAND hccl_utest_impl)
