# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# 注意a3与a5测试用例拆分不同可执行文件
set(src_to_test_list
    main.cc
    # 设备状态刷新回调UT：依赖进程级注册标志位初始为false，须作为首个测试单元链接，
    # 请勿将其移动到其他触发 HcommEndpoint* 接口的测试文件之后
    host/ut_device_refresh_callback.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/hccl_api_base_test.cc
    ut_HcclRankGraph_API.cc
    ut_HcomInitCollComm.cc
    comms/aiv/ut_AivUbMemTransport_API_test.cc
    comms/common/ut_orion_adpt_utils.cc
    comms/common/ut_tp_mgr.cc
    comms/common/ut_config_mgr.cc
    comms/common/ut_rma_buffer_mgr.cc
    comms/comm_engine_res/threads/test_aicpu_ts_thread_class.cc
    comms/comm_engine_res/threads/test_cpu_ts_thread_class.cc
    comms/comm_engine_res/threads/test_iaicpu_ts_thread_interface.cc
    comms/comm_engine_res/threads/test_thread_api.cc
    comms/comm_engine_res/threads/test_thread_acquire_with_config.cc
    comms/comm_engine_res/kernel_launch/test_kernel_launch_aicpu.cc
    comms/endpoint_pairs/socket/ut_socket_process.cc
    comms/endpoint_pairs/ut_endpoint_pair.cc
    comms/api_c_adpt/ut_hcomm_c_adpt.cc
    comms/api_c_adpt/ut_hcomm_c_adpt_ubattr.cc
    comms/api_c_adpt/ut_hcomm_c_adpt_ubmemattr.cc
    coll_comms/rank/ut_my_rank.cc
    coll_comms/rank/ut_ccu_ms_per_device_exclusive.cc
    coll_comms/resource_mgr/endpoints/ut_endpoint_mgr.cc
    coll_comms/rank/comm_mems/test_getHcclBuffer_api.cc
    coll_comms/rank/ut_my_rank_tls.cc
    coll_comms/rank/comm_mems/test_comm_reg_mem_api.cc
    comms/api_c_adpt/ut_hcomm_res_mgr_test.cc
    comms/api_c_adpt/ut_hcomm_endpoint_test.cc
    comms/api_c_adpt/ut_hcomm_endpoint_uboe_test.cc
    comms/api_c_adpt/ut_hcomm_endpoint_ub_rtp_test.cc
    comms/api_c_adpt/ut_hcomm_mem_test.cc
    comms/api_c_adpt/ut_hcomm_channel_test.cc
    comms/api_c_adpt/ut_hcomm_channel_config_test.cc
    comms/api_c_adpt/ut_hcomm_engine_ctx_test.cc
    comms/api_c_adpt/ut_hcomm_dfx_test.cc
    # endpoint/RegedMemMgr 公共流程抽取归一 UT
    comms/api_c_adpt/ut_endpoint_ctx.cc
    comms/api_c_adpt/ut_hcomm_endpoint_mgr.cc
    comms/endpoints/ut_server_socket_context.cc
    comms/endpoints/reged_mems/ut_plugin_reged_mem_mgr.cc
    coll_comms/communicator/ut_coll_comm_test.cc
    coll_comms/communicator/group_schedule_mgr/ut_group_schedule_mgr.cc
    comms/endpoint_pairs/channels/ut_channel_process_test.cc
    comms/endpoint_pairs/channels/ut_channel_roce_factory.cc
    coll_comms/communicator/aicpu/ut_aicpu_indop_process_test.cc
    host/ut_cpu_roce_endpoint.cc
    host/ut_host_cpu_roce_channel.cc
    host/ut_host_rdma_connection.cc
    host/ut_server_socket_manager.cc
    host/ut_cpu_urma_endpoint.cc
    host/ut_host_cpu_urma_channel.cc
    host/ut_host_ub_connection.cc
    host/ut_device_reset_callback.cc
    comms/endpoint_pairs/channels/aicpu/ut_aicpu_ts_hccs_channel.cc
    comms/endpoint_pairs/channels/aicpu/ut_aicpu_ts_roce_channel_v2.cc
    comms/endpoint_pairs/channels/aicpu/ut_dev_rdma_connection.cc
    comms/endpoints/ut_ub_mem_endpoint.cc
    comms/endpoints/reged_mems/ut_roce_mem_test.cc
    comms/endpoints/reged_mems/ut_urma_mem_test.cc
    comms/endpoints/reged_mems/ut_ub_mem_test.cc
    comms/endpoints/reged_mems/ut_hccs_reged_mem_mgr_test.cc
    coll_comms/api_c_adpt/ut_coll_comm_res_c_adpt.cc
    coll_comms/api_c_adpt/ut_coll_comm_ccu_c_adpt.cc
    coll_comms/api_c_adpt/ut_hccl_channel_config_test.cc
    coll_comms/team/hccl/ut_hccl_team_c_adpt.cc
    coll_comms/team/hccl/ut_hccl_team_mgr.cc
    coll_comms/team/hccl/ut_ubmem_symmetric_memory.cc
    comms/endpoints/reged_mems/ut_aicpu_ts_roce_mem.cc
    # Ns recovery相关测试用例
    comms/endpoint_pairs/channels/aicpu/ut_aicpu_ts_urma_channel.cc
    comms/endpoint_pairs/channels/aicpu/ut_aicpu_ts_p2p_channel.cc
    comms/endpoint_pairs/channels/aicpu/ut_aicpu_ts_uboe_channel.cc
    comms/endpoint_pairs/channels/aicpu/ut_aicpu_ts_ub_rtp_channel.cc
    comms/endpoint_pairs/channels/aicpu/ut_aicpu_ts_roce_channel.cc
    comms/endpoint_pairs/channels/aicpu/device/ut_aicpu_channel_process.cc
    comms/endpoints/ut_urma_endpoint.cc
    comms/endpoints/ut_aicpu_ts_roce_endpoint.cc
    comms/endpoint_pairs/channels/ccu/ut_ccu_urma_channel.cc
    comms/endpoint_pairs/channels/aiv/ut_aiv_ub_mem_channel.cc
    comms/endpoint_pairs/channels/aiv/ut_aiv_ub_mem_transport.cc
    comms/endpoint_pairs/channels/aiv/ut_aiv_urma_channel.cc
    comms/endpoint_pairs/ut_endpoint_pair_mgr.cc
    comms/ccu/ccu_device/ccu_comp/ut_ccu_comp_pub.cc
    coll_comms/communicator/ns_recovery/aicpu/ut_ns_recovery_func_lite.cc
    coll_comms/communicator/ns_recovery/aicpu/ut_ns_recovery_lite.cc
    coll_comms/rank_pairs/ut_rank_pair.cc
    coll_comms/resource_mgr/rank_pairs/ut_channel_manager.cc
    coll_comms/resource_mgr/local/my_rank/comm_mems/ut_comm_mem_manager.cc
    coll_comms/resource_mgr/local/my_rank/comm_engine/engine_ctxs/ut_engine_ctxs.cc
    coll_comms/communicator/ns_recovery/ut_hccl_task_abort_handler.cc
    coll_comms/communicator/aicpu/ut_coll_comm_aicpu.cc
    coll_comms/communicator/aicpu/ut_coll_comm_aicpu_destroy_func.cc
    coll_comms/communicator/aicpu/ut_thread_aicpu_mgr.cc
    coll_comms/communicator/aicpu/ut_notify_aicpu_mgr.cc
    coll_comms/communicator/aicpu/ut_comm_engine_res_aicpu_mgr.cc
    coll_comms/communicator/aicpu/ut_channel_aicpu_mgr.cc
    coll_comms/communicator/ns_recovery/ut_ns_recovery.cc
    comms/adpt/ut_hcomm_adapter_hccp_test.cc
    comms/endpoints/dfx/ut_endpoint_monitor.cc
    coll_comms/rank/ut_exchange_info_mgr.cc
    coll_comms/rank/comm_mems/test_hcclCommMemReg_api.cc
    # Shared jetty UT
    comms/endpoint_pairs/channels/ut_shared_jetty_mgr.cc
    comms/endpoint_pairs/channels/ut_shared_jetty_connection_adapter.cc
    comms/endpoint_pairs/channels/ut_shared_jetty_channel_helper.cc
    comms/endpoints/reged_mems/ut_jetty_context.cc
    coll_comms/rank/ut_shared_jetty_channel_pool.cc
    coll_comms/rank_graphs/common/ut_conn_interface.cc
    coll_comms/rank_graphs/common/ut_graph.cc
    coll_comms/rank_graphs/phy_topo/ut_phy_topo.cc
    coll_comms/rank_graphs/phy_topo_builder/ut_phy_topo_builder.cc
    coll_comms/rank_graphs/rank_graph/ut_net_instance.cc
    coll_comms/rank_graphs/rank_graph/ut_rank_graph.cc
    coll_comms/rank_graphs/rank_graph_builder/ut_detour_service.cc
    coll_comms/rank_graphs/rank_graph_builder/ut_rank_graph_builder.cc
    coll_comms/rank_graphs/rank_graph_builder/ut_updater_for_64_plus_1.cc
    coll_comms/rank_graphs/rank_table_info/ut_address_info.cc
    coll_comms/rank_graphs/rank_table_info/ut_control_plane.cc
    coll_comms/rank_graphs/rank_table_info/ut_new_rank_info.cc
    coll_comms/rank_graphs/rank_table_info/ut_rank_level_info.cc
    coll_comms/rank_graphs/rank_table_info/ut_rank_table_info.cc
    coll_comms/rank_graphs/topo_info/ut_edge_info.cc
    coll_comms/rank_graphs/topo_info/ut_peer_info.cc
    coll_comms/rank_graphs/topo_info/ut_topo_info.cc
    coll_comms/rank_info_detect/ut_bootstrap_ip.cc
    coll_comms/rank_info_detect/ut_rank_info_detect.cc
    coll_comms/rank_info_detect/ut_rank_info_detect_client.cc
    coll_comms/rank_info_detect/ut_rank_info_detect_service.cc
    coll_comms/rank_info_detect/ut_rankinfo_dispatcher.cc
    coll_comms/rank_info_detect/ut_whitelist.cc
    coll_comms/rank_info_detect/whitelist_test.cc
    ${HCCL_CODE_ROOT}/test/legacy/ut/framework/env_config/env_config_stub.cc
    coll_comms/team/hcomm/ut_hcomm_team_c_adpt.cc
    coll_comms/team/hcomm/ut_hcomm_team_mgr.cc
)

add_executable(hccl_utest_test_next_resource ${src_to_test_list})
target_include_directories(hccl_utest_test_next_resource PRIVATE
    ${HCCL_CODE_ROOT}/src/base_comm/resources/endpoint_pairs/channels
    ${HCCL_CODE_ROOT}/src/base_comm/config_mgr
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/team/hcomm
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/team/hccl
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/resource_mgr/local/my_rank/
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/resource_mgr/local/my_rank/comm_mems
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/communicator/group_schedule_mgr/

    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/resource_mgr/local/my_rank/endpoints
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/resource_mgr/remote/rank_pairs
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/resource_mgr/local/my_rank/comm_engine_reses/notify
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/rank_graph/common
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/rank_graph/phy_topo
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/rank_graph/phy_topo_builder
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/rank_graph/rank_graph
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/rank_graph/rank_graph_builder
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/rank_graph/rank_table_info
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/rank_graph/topo_info
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/rank_info_detect
    ${UT_COMMON_INCLUDE_LIST}
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/framework/topo/rank_info_detect
    ${HCCL_CODE_ROOT}/test/legacy/ut/framework/env_config
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_comms/rank_graphs/common
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_comms/rank_graphs/rank_graph_64_plus_1
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_comms/rank_graphs/rank_graph_builder
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/resource_mgr/local/my_rank/comm_engine/notify
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/dfx/cluster_monitor
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/communicator/device
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/communicator/device/resource_mgr/local/my_rank/comm_engine_reses
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/communicator/device/resource_mgr/remote
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/communicator/device/c_adpt
)

target_compile_definitions(hccl_utest_test_next_resource PRIVATE
    _GLIBCXX_USE_CXX11_ABI=0
    google=ascend_private
    -DOPEN_HCCL_TEST
    ORION_MODE
    OPEN_BUILD_PROJECT
)
 
target_compile_options(hccl_utest_test_next_resource PRIVATE
    -U_FORTIFY_SOURCE
    -fno-var-tracking-assignments
    -Os
    -O0 -g --coverage -fprofile-arcs -ftest-coverage -fno-access-control
    $<$<STREQUAL:${ENABLE_ASAN},true>:-fsanitize=address -fsanitize-recover=address,all -fno-omit-frame-pointer>
)

target_link_libraries(hccl_utest_test_next_resource PRIVATE
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
add_dependencies(hccl_utest_test_next_resource hccl_llt)

enable_testing()
add_test(NAME hccl_utest_test_next_resource COMMAND hccl_utest_test_next_resource)

set(src_to_a3_test_list
    main.cc
    hccl_api_base_test.cc
    coll_comms/rank/comm_mems/test_getHcclBuffer_a3_api.cc
    comms/endpoints/ut_hccs_mem_endpoint.cc
    comms/api_c_adpt/ut_hcomm_hccs_channel_test.cc
)

add_executable(hccl_utest_test_next_resource_a3 ${src_to_a3_test_list})

#头文件搜索路径
target_include_directories(hccl_utest_test_next_resource_a3 PRIVATE
    ${UT_COMMON_INCLUDE_LIST}
)
 
target_compile_definitions(hccl_utest_test_next_resource_a3 PRIVATE
    _GLIBCXX_USE_CXX11_ABI=0
    google=ascend_private
    -DOPEN_HCCL_TEST
    ORION_MODE
    OPEN_BUILD_PROJECT
)
 
target_compile_options(hccl_utest_test_next_resource_a3 PRIVATE
    -U_FORTIFY_SOURCE
    -fno-var-tracking-assignments
    -Os
    -O0 -g --coverage -fprofile-arcs -ftest-coverage -fno-access-control
    $<$<STREQUAL:${ENABLE_ASAN},true>:-fsanitize=address -fsanitize-recover=address,all -fno-omit-frame-pointer>
)

target_link_libraries(hccl_utest_test_next_resource_a3 PRIVATE
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

add_dependencies(hccl_utest_test_next_resource_a3 hccl_llt)

enable_testing()
add_test(NAME hccl_utest_test_next_resource_a3 COMMAND hccl_utest_test_next_resource_a3)

add_subdirectory(coll_comms/common/loggers)
