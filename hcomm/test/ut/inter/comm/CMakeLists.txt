# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(hccl_utest_inter_comm_list
    main.cc
    # ut_adapter.cc
    # ut_comm_halving_doubling.cc
    # ut_comm_p2p.cc
    # ut_hccl_comm_910B.cc
    # ut_hccl_impl_alg_double_ring.cc
    # ut_allgather_mix_op_base.cc
    # ut_comm_init.cc

    ut_coll_alg_operator.cc
    ut_allreduce_mesh_opbase.cc
    ut_alltoall_aiv.cc
    ut_hccl_impl_alg_nhr.cc
    ut_hccl_impl_alg_nb.cc
    ut_hccl_impl_alg_ahc.cc
    ut_hccl_impl_alg.cc
    ut_hccl_impl_alg_double_ring_concurrent.cc
    ut_hccl_impl_alg_double_ring_concurrent_allreduce.cc
    ut_alg_registry.cc
    ut_alltoallv_staged.cc
    ut_callback_thread_manager.cc
    ut_comm_base.cc
    ut_command_handle.cc
    ut_component_reducer.cc
    ut_comm_factory.cc
    ut_comm_ring.cc
    ut_comm_inner.cc
    ut_comm_mesh.cc
    ut_socket.cc
)
add_executable(hccl_utest_inter_comm  ${hccl_utest_inter_comm_list})

target_include_directories(hccl_utest_inter_comm PRIVATE
    ${UT_COMMON_INCLUDE_LIST}
)

target_compile_definitions(hccl_utest_inter_comm PRIVATE
    _GLIBCXX_USE_CXX11_ABI=0
    HALF_T=float
    google=ascend_private
    CCL_KERNEL
    CCL_LLT
    OPEN_BUILD_PROJECT
)

target_compile_options(hccl_utest_inter_comm PRIVATE
    -U_FORTIFY_SOURCE
    -fno-var-tracking-assignments
    -Os
    -O0 -g --coverage -fprofile-arcs -ftest-coverage -fno-access-control
    $<$<STREQUAL:${ENABLE_ASAN},true>:-fsanitize=address -fsanitize-recover=address,all -fno-omit-frame-pointer>
)

target_link_libraries(hccl_utest_inter_comm PRIVATE
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
add_dependencies(hccl_utest_inter_comm hccl_llt)

enable_testing()
add_test(NAME hccl_utest_inter_comm COMMAND hccl_utest_inter_comm)

