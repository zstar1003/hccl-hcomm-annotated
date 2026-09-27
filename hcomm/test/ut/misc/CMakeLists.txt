# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(hccl_utest_misc_list
    main.cc
    ut_sal.cc
    # ut_send_receive.cc
    ut_wrapper.cc
    # ut_external_input.cc
    ut_prepare_slice.cc
    # ut_reducescatter_common.cc
    ut_opretry_conn.cc
    # ut_op_base.cc
    # ut_op_base_multithread.cc
    ut_rankconsistent.cc
    ut_task_exception.cc
    # ut_mpi_tcp_sendrecv.cc
    # ut_tbe_vector_reduce.cc
    ut_heartbeat.cc
    ut_p2p_mgmt.cc
    ut_stream_active_manager.cc
    ut_socket_manager.cc
    ut_op_base_stream_manager.cc
    ut_offload_stream_manager.cc
    ut_stream.cc
    ut_topoinfo_exchange_dispatcher.cc
    ut_op_counter.cc
    ut_hccl_hdc.cc
    ut_opretry.cc
    ut_peterson_lock.cc
    ut_task_abort_handler.cc
    # ut_hccl_comm_aicpuunfold.cc
    # ut_hccl_transport_p2p.cc
    # ut_onesided_service.cc
    ut_transport_ibverbs.cc
    ut_load_backup_ip.cc
    # ut_backup_link.cc
    ut_adapter.cc
    # ut_switch_nic.cc
    ut_switch_nic_fsm.cc
    ut_interface_multithread.cc
    ut_interface_hccl_verbs.cc
    ut_topoinfo_exchange_agent.cc
    ut_exchange_info.cc
    ut_rankconsistent_v2.cc
)
add_executable(hccl_utest_misc ${hccl_utest_misc_list})

target_include_directories(hccl_utest_misc PRIVATE
    ${UT_COMMON_INCLUDE_LIST}
)

target_compile_definitions(hccl_utest_misc PRIVATE
    _GLIBCXX_USE_CXX11_ABI=0
    HALF_T=float
    google=ascend_private
    CCL_KERNEL
    CCL_LLT
    OPEN_HCCL_TEST
    OPEN_BUILD_PROJECT
)

target_compile_options(hccl_utest_misc PRIVATE
    -U_FORTIFY_SOURCE
    -fno-var-tracking-assignments
    -Os
    -O0 -g --coverage -fprofile-arcs -ftest-coverage -fno-access-control
    $<$<STREQUAL:${ENABLE_ASAN},true>:-fsanitize=address -fsanitize-recover=address,all -fno-omit-frame-pointer>
)

target_link_libraries(hccl_utest_misc PRIVATE
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
add_dependencies(hccl_utest_misc hccl_llt)

enable_testing()
add_test(NAME hccl_utest_misc COMMAND hccl_utest_misc)
