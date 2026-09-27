# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(hccl_utest_inter_misc_list
    main.cc
    # ut_hcom_graph_mc2.cc
    # ut_mem_host.cc
    # ut_mem_device.cc
    # ut_stream.cc
    # ut_op_base_stream_manager.cc
    # ut_offload_stream_manager.cc
    # ut_workspace_mem.cc
    # ut_notify_pool.cc
    # ut_transport_virtural.cc
    # ut_hccl_transport_ibv_exp.cc
    # ut_hccl_transport_heterog_p2p.cc
    # ut_hccl_transport_p2p.cc 
    # ut_hccl_inter_new.cc
    # ut_nonuniform_hierarchical_ring_v1_base.cc
    # ut_reducescatter_common.cc
    # ut_hvd_adapter.cc
    # ut_threadManage.cc
    # ut_hccl_kernel_builder.cc
    # ut_hvd_kernel_builder.cc
    # ut_heterog_send_recv.cc
    # ut_stream_active_manager.cc
    # ut_rank_table.cc 
    # ut_hvd_graph_optimizer.cc
    # ut_hvd_ops_kernel_info_store.cc
    # ut_prepare_slice.cc
    # ut_coll_executor_broadcast.cc
    # ut_coll_executor_reduce.cc
    # ut_op_counter.cc
    # ut_hccl_comm_aicpuunfold.cc
    # ut_coll_executor_muli_qp.cc
)
add_executable(hccl_utest_inter_misc  ${hccl_utest_inter_misc_list})

set_legacy_protobuf()

#头文件搜索路径
if(BUILD_OPEN_PROJECT) # 蓝区构建
    target_include_directories(hccl_utest_inter_misc  PRIVATE
        ${package_include_list}
        ${include_list}
        ${algorithm_include_list}
        ${framework_include_list}
        ${common_include_list}
        ${platform_include_list}
        ${other_include_list}
        ${JSON_INCLUDE_DIR}
        ${mockcpp_INCLUDE_DIRS}
        ${HCCL_STUB_CODE_ROOT}/framework_stub
    )
    
    target_compile_definitions(hccl_utest_inter_misc    PRIVATE
        _GLIBCXX_USE_CXX11_ABI=0
        HALF_T=float
        google=ascend_private
        CCL_KERNEL
        -DCCL_LLT
        -DOPEN_HCCL_TEST
        OPEN_BUILD_PROJECT
    )
else()
    target_include_directories(hccl_utest_inter_misc  PRIVATE
        ${include_list}
    )
    
    target_compile_definitions(hccl_utest_inter_misc    PRIVATE
        google=ascend_private
        HALF_T=float
        google=ascend_private
        CCL_KERNEL
        -DCCL_LLT
    )
endif()

target_compile_options(hccl_utest_inter_misc   PRIVATE
    -Os
    -g
    -U_FORTIFY_SOURCE
)

if(BUILD_OPEN_PROJECT) # 蓝区构建
    target_link_libraries(hccl_utest_inter_misc PRIVATE
        $<BUILD_INTERFACE:intf_llt_pub>
        framework_stub
        mmpa
        $<$<STREQUAL:${ENABLE_ASAN},true>:-fsanitize=address>
        ${LIBRARY_MOCKCPP}
    )
    add_dependencies(hccl_utest_inter_misc hccl hccl_alg hccl_plf framework_stub)
else() 
    target_link_libraries(hccl_utest_inter_misc  PRIVATE
        $<BUILD_INTERFACE:intf_llt_pub>
        $<BUILD_INTERFACE:hccl_headers>
        $<BUILD_INTERFACE:mmpa_headers>
        $<BUILD_INTERFACE:msprof_headers>
        $<BUILD_INTERFACE:slog_headers>
        $<BUILD_INTERFACE:npu_runtime_headers>
        $<BUILD_INTERFACE:cce_headers>
        $<BUILD_INTERFACE:atrace_headers>
        $<BUILD_INTERFACE:kernel_tiling_headers> 
        c_sec_static
        ascend_protobuf
        hccl_llt
        hccl_llt_open
        -lrt
        -ldl
    )

    add_dependencies(hccl_utest_inter_misc hccl_llt hccl_llt_open)
    enable_testing()
    add_test(NAME hccl_utest_inter_misc COMMAND hccl_utest_inter_misc)
endif()
