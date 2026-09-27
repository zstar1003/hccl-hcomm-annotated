# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

## 源代码路径加入到该文件中
target_sources(hccl_v2_utest
        PUBLIC
        ${HCCL_V2_CODE_ROOT}/unified_platform/external_system/orion_adapter_rts.cc
        ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/rank_graph/rank_graph_builder/rank_graph_builder_bridge.cc
        ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/rank_graph/rank_graph_builder/rank_graph_builder_provider.cc
        ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/rank_info_detect/root_info_detect_bridge.cc

        ${HCCL_V2_CODE_ROOT}/framework/communicator/communicator_impl.cc
        ${HCCL_V2_CODE_ROOT}/framework/communicator/hccl_communicator.cc
        ${HCCL_V2_CODE_ROOT}/framework/communicator/hccl_params_pub.cc
        ${HCCL_V2_CODE_ROOT}/framework/communicator/hdc.cc
        ${HCCL_V2_CODE_ROOT}/framework/communicator/task_abort_handler.cc
        ${HCCL_V2_CODE_ROOT}/framework/communicator/op_params_checker.cc
        

        ${HCCL_V2_CODE_ROOT}/framework/communicator/aicpu/kernel_entrance.cc
        ${HCCL_V2_CODE_ROOT}/framework/communicator/aicpu/communicator_impl_lite_manager.cc
        ${HCCL_V2_CODE_ROOT}/framework/communicator/aicpu/communicator_impl_lite.cc
        ${HCCL_V2_CODE_ROOT}/framework/communicator/aicpu/aicpu_res_package_helper.cc
        ${HCCL_V2_CODE_ROOT}/framework/communicator/aicpu/hdc_lite.cc
        ${HCCL_V2_CODE_ROOT}/framework/communicator/aicpu/aicpu_hdc_handler.cc
        ${HCCL_V2_CODE_ROOT}/framework/communicator/aicpu/aicpu_comm_destroy_func.cc
        ${HCCL_V2_CODE_ROOT}/framework/communicator/aicpu/ns_recovery_handler_func.cc
        ${HCCL_V2_CODE_ROOT}/framework/communicator/aicpu/aicpu_mc2_handler.cc
        ${HCCL_V2_CODE_ROOT}/framework/communicator/aicpu/hccl_mc2_ex.cc
        ${HCCL_V2_CODE_ROOT}/framework/communicator/aicpu/aicpu_utils.cc
        ${HCCL_V2_CODE_ROOT}/framework/communicator/aicpu/launch_device.cc


        ${HCCL_V2_CODE_ROOT}/framework/service/coll_service_base.cc
        ${HCCL_V2_CODE_ROOT}/framework/service/coll_service_ai_cpu_impl.cc
        ${HCCL_V2_CODE_ROOT}/framework/service/coll_service_default_impl.cc
        ${HCCL_V2_CODE_ROOT}/framework/service/connections_builder.cc
        ${HCCL_V2_CODE_ROOT}/framework/service/interpreter.cc
        ${HCCL_V2_CODE_ROOT}/framework/service/ins_rules.cc

        ${HCCL_V2_CODE_ROOT}/framework/service/aicpu/alg_topo_package_helper.cc
        ${HCCL_V2_CODE_ROOT}/framework/service/aicpu/ins_executor.cc
        ${HCCL_V2_CODE_ROOT}/framework/service/aicpu/ins_to_sqe_rule.cc
        ${HCCL_V2_CODE_ROOT}/framework/service/aicpu/ins_to_sqe_rule_v82.cc
        ${HCCL_V2_CODE_ROOT}/framework/resource_manager/stream/aicpu/stream_lite_mgr.cc

        ${HCCL_V2_CODE_ROOT}/framework/resource_manager/notify/queue_notify_manager.cc
        ${HCCL_V2_CODE_ROOT}/framework/resource_manager/notify/conn_local_notify_manager.cc
        ${HCCL_V2_CODE_ROOT}/framework/resource_manager/notify/queue_wait_group_cnt_notify_manager.cc
        ${HCCL_V2_CODE_ROOT}/framework/resource_manager/notify/queue_bcast_post_cnt_notify_manager.cc
        ${HCCL_V2_CODE_ROOT}/framework/resource_manager/notify/conn_local_cnt_notify_manager.cc
        ${HCCL_V2_CODE_ROOT}/framework/resource_manager/notify/aicpu/queue_notify_lite_mgr.cc
        ${HCCL_V2_CODE_ROOT}/framework/resource_manager/notify/aicpu/cnt_1ton_notify_lite_mgr.cc
        ${HCCL_V2_CODE_ROOT}/framework/resource_manager/notify/aicpu/cnt_nto1_notify_lite_mgr.cc
        ${HCCL_V2_CODE_ROOT}/framework/resource_manager/notify/aicpu/host_device_sync_notify_lite_mgr.cc
        ${HCCL_V2_CODE_ROOT}/framework/resource_manager/buffer/aicpu/rmt_data_buffer_mgr.cc

        ${HCCL_V2_CODE_ROOT}/framework/resource_manager/buffer/data_buf_manager.cc
        ${HCCL_V2_CODE_ROOT}/framework/resource_manager/buffer/local_rma_buf_manager.cc
        ${HCCL_V2_CODE_ROOT}/framework/resource_manager/buffer/remote_rma_buf_manager.cc

        ${HCCL_V2_CODE_ROOT}/framework/resource_manager/socket/socket_handle_manager.cc
        ${HCCL_V2_CODE_ROOT}/framework/resource_manager/socket/socket_manager.cc
        ${HCCL_V2_CODE_ROOT}/framework/resource_manager/socket/host_socket_handle_manager.cc

        ${HCCL_V2_CODE_ROOT}/framework/resource_manager/transport/mem_transport_manager.cc
        ${HCCL_V2_CODE_ROOT}/framework/resource_manager/transport/cnt_notify_res_helper.cc
        ${HCCL_V2_CODE_ROOT}/framework/resource_manager/transport/aicpu/mem_transport_lite_mgr.cc
        ${HCCL_V2_CODE_ROOT}/framework/resource_manager/transport/aicpu/connected_link_mgr.cc

        ${HCCL_V2_CODE_ROOT}/framework/resource_manager/connection/rma_conn_manager.cc
        ${HCCL_V2_CODE_ROOT}/framework/resource_manager/connection/ub_ci_updater/ub_ci_updater_manager.cc

        ${HCCL_V2_CODE_ROOT}/framework/resource_manager/notify/queue_notify_manager.cc
        ${HCCL_V2_CODE_ROOT}/framework/resource_manager/notify/conn_local_notify_manager.cc
        ${HCCL_V2_CODE_ROOT}/framework/resource_manager/notify/queue_wait_group_cnt_notify_manager.cc
        ${HCCL_V2_CODE_ROOT}/framework/resource_manager/notify/queue_bcast_post_cnt_notify_manager.cc
        ${HCCL_V2_CODE_ROOT}/framework/resource_manager/notify/conn_local_cnt_notify_manager.cc
        ${HCCL_V2_CODE_ROOT}/framework/resource_manager/notify/host_device_sync_notify_manager.cc

        ${HCCL_V2_CODE_ROOT}/framework/env_config/env_func.cc
        ${HCCL_V2_CODE_ROOT}/framework/env_config/base_config_legacy.cc
        ${HCCL_V2_CODE_ROOT}/framework/env_config/env_config.cc

        ${HCOMM_DIR}/src/base_comm/common/debug/plf_debug_config.cc

        ${HCCL_V2_CODE_ROOT}/framework/misc/json_parser/json_parser.cc
        ${HCOMM_DIR}/src/coll_communicator_mgr/rank_info_detect/whitelist.cc
        ${HCCL_V2_CODE_ROOT}/framework/misc/mask_event/mask_event.cc

        ${HCCL_V2_CODE_ROOT}/framework/resource_manager/transport/ub_memory_transport_mgr.cc

        ${HCCL_V2_CODE_ROOT}/unified_platform/aiv/ub_memory_transport.cc
        ${HCCL_V2_CODE_ROOT}/unified_platform/aiv/urma_direct_transport.cc

        ${HCCL_V2_CODE_ROOT}/unified_platform/common/rdma_handle_manager.cc
        ${HCCL_V2_CODE_ROOT}/unified_platform/common/hccp_hdc_manager.cc
        ${HCCL_V2_CODE_ROOT}/unified_platform/common/hccp_tlv_hdc_manager.cc
        ${HCCL_V2_CODE_ROOT}/unified_platform/ccu/ccu_device/ccu_driver_handle.cc
        ${HCCL_V2_CODE_ROOT}/unified_platform/common/p2p_enable_manager.cc
        ${HCCL_V2_CODE_ROOT}/unified_platform/common/hccp_peer_manager.cc
        ${HCCL_V2_CODE_ROOT}/unified_platform/common/dev_capability.cc
        ${HCCL_V2_CODE_ROOT}/unified_platform/common/dlprof_func.cc
        ${HCCL_CODE_ROOT}/src/base_comm/common/hccl_dl.cc
        ${HCCL_V2_CODE_ROOT}/unified_platform/common/tp_manager.cc
        ${HCCL_V2_CODE_ROOT}/unified_platform/common/tp_qos.cc
        
        ${HCCL_V2_CODE_ROOT}/unified_platform/common/inner_net_dev_manager.cc
        ${HCCL_V2_CODE_ROOT}/unified_platform/common/inner_net_dev.cc
        ${HCCL_V2_CODE_ROOT}/unified_platform/common/net_device.cc
        ${HCCL_V2_CODE_ROOT}/unified_platform/resource/net_dev/hccl_net_dev_v2.cc
        ${HCCL_V2_CODE_ROOT}/unified_platform/resource/net_dev/hccl_net_dev.cc

        ${HCCL_V2_CODE_ROOT}/unified_platform/resource/stream/aicpu/stream_lite.cc
        ${HCCL_V2_CODE_ROOT}/unified_platform/resource/stream/aicpu/rtsq_base.cc
        ${HCCL_V2_CODE_ROOT}/unified_platform/resource/stream/aicpu/rtsq_a5.cc
        ${HCCL_V2_CODE_ROOT}/unified_platform/resource/stream/aicpu/sqe_build_a5.cc
        ${HCCL_V2_CODE_ROOT}/unified_platform/resource/connection/rma_connection.cc
        ${HCCL_V2_CODE_ROOT}/unified_platform/resource/connection/dev_rdma_connection.cc
        ${HCCL_V2_CODE_ROOT}/unified_platform/resource/connection/p2p_connection.cc
        ${HCCL_V2_CODE_ROOT}/unified_platform/resource/connection/host_ub_connection.cc
        ${HCCL_V2_CODE_ROOT}/unified_platform/resource/connection/dev_ub_connection.cc
        ${HCCL_V2_CODE_ROOT}/unified_platform/resource/connection/rma_connection.cc

        ${HCCL_V2_CODE_ROOT}/unified_platform/resource/connection/aicpu/rma_conn_lite.cc
        ${HCCL_V2_CODE_ROOT}/unified_platform/resource/connection/aicpu/ub_conn_lite.cc
        ${HCCL_V2_CODE_ROOT}/unified_platform/resource/connection/aicpu/ub_conn_lite_mgr.cc
        ${HCCL_V2_CODE_ROOT}/unified_platform/resource/connection/aicpu/rdma_conn_lite_v2.cc

        ${HCCL_V2_CODE_ROOT}/unified_platform/resource/connection/ub_ci_updater/ub_ci_updater.cc

        ${HCCL_V2_CODE_ROOT}/unified_platform/resource/socket/socket.cc

        ${HCCL_V2_CODE_ROOT}/unified_platform/resource/task/task.cc

        ${HCCL_V2_CODE_ROOT}/unified_platform/resource/buffer/buffer.cc
        ${HCCL_V2_CODE_ROOT}/unified_platform/resource/buffer/local_ipc_rma_buffer.cc
        ${HCCL_V2_CODE_ROOT}/unified_platform/resource/buffer/local_rdma_rma_buffer.cc
        ${HCCL_V2_CODE_ROOT}/unified_platform/resource/buffer/local_ub_rma_buffer.cc
        ${HCCL_V2_CODE_ROOT}/unified_platform/resource/buffer/remote_rma_buffer.cc
        ${HCCL_V2_CODE_ROOT}/unified_platform/resource/buffer/host_buffer.cc
        ${HCCL_V2_CODE_ROOT}/unified_platform/resource/buffer/dev_buffer.cc

        ${HCCL_V2_CODE_ROOT}/unified_platform/resource/buffer/aicpu/rma_buf_slice_lite.cc
        ${HCCL_V2_CODE_ROOT}/unified_platform/resource/buffer/aicpu/rma_buffer_lite.cc
        ${HCCL_V2_CODE_ROOT}/unified_platform/resource/buffer/aicpu/rmt_rma_buf_slice_lite.cc
        ${HCCL_V2_CODE_ROOT}/unified_platform/resource/buffer/aicpu/rmt_rma_buffer_lite.cc

        ${HCCL_V2_CODE_ROOT}/unified_platform/resource/notify/ipc_local_notify.cc
        ${HCCL_V2_CODE_ROOT}/unified_platform/resource/notify/rdma_local_notify.cc
        ${HCCL_V2_CODE_ROOT}/unified_platform/resource/notify/ub_local_notify.cc
        ${HCCL_V2_CODE_ROOT}/unified_platform/resource/notify/ipc_remote_notify.cc
        ${HCCL_V2_CODE_ROOT}/unified_platform/resource/notify/local_cnt_notify.cc
        ${HCCL_V2_CODE_ROOT}/unified_platform/resource/notify/rts_notify.cc
        ${HCCL_V2_CODE_ROOT}/unified_platform/resource/notify/rts_cnt_notify.cc
        ${HCCL_V2_CODE_ROOT}/unified_platform/resource/notify/rts_1ton_cnt_notify.cc

        ${HCCL_V2_CODE_ROOT}/unified_platform/resource/notify/aicpu/notify_lite.cc
        ${HCCL_V2_CODE_ROOT}/unified_platform/resource/notify/aicpu/cnt_1ton_notify_lite.cc
        ${HCCL_V2_CODE_ROOT}/unified_platform/resource/notify/aicpu/cnt_nto1_notify_lite.cc

        ${HCCL_V2_CODE_ROOT}/unified_platform/resource/stream/stream.cc
        ${HCCL_V2_CODE_ROOT}/unified_platform/resource/stream/aicpu/stream_lite.cc
        ${HCCL_V2_CODE_ROOT}/unified_platform/resource/stream/aicpu/hccl_sqe.cc
        ${HCCL_V2_CODE_ROOT}/unified_platform/resource/stream/aicpu/sqe_mgr.cc
        ${HCCL_V2_CODE_ROOT}/unified_platform/resource/stream/aicpu/hccl_sqe_v82.cc

        ${HCCL_V2_CODE_ROOT}/unified_platform/resource/transport/base_mem_transport.cc
        ${HCCL_V2_CODE_ROOT}/unified_platform/resource/transport/p2p_transport.cc
        ${HCCL_V2_CODE_ROOT}/unified_platform/resource/transport/ub_mem_transport.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/exception_handler_stub.cc

        ${HCCL_V2_CODE_ROOT}/unified_platform/resource/transport/aicpu/mem_transport_lite.cc
        ${HCCL_V2_CODE_ROOT}/unified_platform/resource/transport/aicpu/ub_transport_lite_impl.cc
        ${HCCL_V2_CODE_ROOT}/unified_platform/resource/transport/aicpu/p2p_transport_lite_impl.cc
        ${HCCL_V2_CODE_ROOT}/unified_platform/resource/transport/aicpu/roce_transport_lite_impl.cc

        ${HCCL_V2_CODE_ROOT}/unified_platform/ccu/ccu_device/ccu_device_manager.cpp
        ${HCCL_V2_CODE_ROOT}/unified_platform/ccu/ccu_device/ccu_res_specs.cpp
        ${HCCL_V2_CODE_ROOT}/unified_platform/ccu/ccu_device/ccu_res_batch_allocator.cpp

        ${HCCL_V2_CODE_ROOT}/unified_platform/ccu/ccu_device/ccu_component/ccu_component.cpp
        ${HCCL_V2_CODE_ROOT}/unified_platform/ccu/ccu_device/ccu_component/ccu_res_allocator.cpp
        ${HCCL_V2_CODE_ROOT}/unified_platform/ccu/ccu_device/ccu_component/ccu_eid_info.cpp

        ${HCCL_V2_CODE_ROOT}/unified_platform/ccu/ccu_device/ccu_component/ccu_channel/ccu_channel_mgr.cpp
        ${HCCL_V2_CODE_ROOT}/unified_platform/ccu/ccu_device/ccu_component/ccu_channel/ccu_jetty_ctx_mgr.cpp
        ${HCCL_V2_CODE_ROOT}/unified_platform/ccu/ccu_device/ccu_component/ccu_channel/ccu_wqebb_mgr.cpp
        ${HCCL_V2_CODE_ROOT}/unified_platform/ccu/ccu_device/ccu_component/ccu_channel/ccu_channel_v1/ccu_channel_mgr_v1.cpp
        ${HCCL_V2_CODE_ROOT}/unified_platform/ccu/ccu_device/ccu_component/ccu_channel/ccu_channel_v1/ccu_jetty_ctx_mgr_v1.cpp

        ${HCCL_V2_CODE_ROOT}/unified_platform/ccu/ccu_device/ccu_component/ccu_pfe/ccu_pfe_mgr.cpp
        ${HCCL_V2_CODE_ROOT}/unified_platform/ccu/ccu_device/ccu_component/ccu_pfe/ccu_pfe_cfg_generator.cpp

        ${HCCL_V2_CODE_ROOT}/unified_platform/ccu/ccu_context/ccu_context.cpp
        ${HCCL_V2_CODE_ROOT}/unified_platform/ccu/ccu_context/ccu_context_mgr.cpp
        ${HCCL_V2_CODE_ROOT}/unified_platform/ccu/ccu_context/ccu_context_mgr_imp.cpp

        ${HCCL_V2_CODE_ROOT}/unified_platform/resource/ccu_transport/ccu_transport_group.cpp

        ${HCCL_V2_CODE_ROOT}/unified_platform/resource/ccu_transport/ccu_connection.cpp
        ${HCCL_V2_CODE_ROOT}/unified_platform/resource/ccu_transport/ccu_transport.cpp
        ${HCCL_V2_CODE_ROOT}/unified_platform/resource/ccu_transport/ccu_transport_group.cpp

        ${HCCL_V2_CODE_ROOT}/framework/ccu/ccu_manager/ccu_transport_group_manager.cpp
        ${HCCL_V2_CODE_ROOT}/framework/ccu/ccu_manager/ccu_transport_manager.cpp
        ${HCCL_V2_CODE_ROOT}/framework/ccu/ccu_manager/ccu_jetty_mgr.cpp
        ${HCCL_V2_CODE_ROOT}/unified_platform/trace/trace.cc

        ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/rank_graph/common/conn_interface.cc
        ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/rank_graph/phy_topo/phy_topo.cc
        ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/rank_graph/phy_topo_builder/phy_topo_builder.cc
        ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/rank_graph/rank_table_info/changed_rank_info.cc
        ${HCCL_V2_CODE_ROOT}/framework/communicator/diff_rank_updater.cc

        ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/rank_graph/rank_table_info/new_rank_info.cc
        ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/rank_graph/rank_table_info/rank_level_info.cc
        ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/rank_graph/rank_table_info/rank_table_info.cc
        ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/rank_graph/rank_table_info/control_plane.cc
        ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/rank_graph/rank_table_info/address_info.cc

        ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/rank_graph/topo_info/edge_info.cc
        ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/rank_graph/topo_info/peer_info.cc
        ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/rank_graph/topo_info/topo_info.cc
        
        ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/rank_graph/rank_graph/rank_graph.cc
        ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/rank_graph/rank_graph/net_instance.cc
        ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/rank_graph/rank_graph_builder/rank_graph_builder.cc
        ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/rank_graph/rank_graph_builder/detour_service.cc
        ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/rank_graph/rank_graph_builder/updater_for_64_plus_1.cc
        ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/rank_graph/rank_graph_builder/detour_rules.cc

        ${HCCL_V2_CODE_ROOT}/framework/topo/virtual_topo.cc

        ${HCCL_V2_CODE_ROOT}/unified_platform/external_system/orion_adapter_trace.cc

        ${HCCL_V2_CODE_ROOT}/framework/ccu/ccu_ins/ccu_ins_group.cpp
        ${HCCL_V2_CODE_ROOT}/framework/ccu/ccu_ins/ccu_ins.cc
        ${HCCL_V2_CODE_ROOT}/framework/ccu/ccu_communicator.cpp
        ${HCCL_V2_CODE_ROOT}/framework/ccu/ccu_ins_preprocessor.cpp
        ${HCCL_V2_CODE_ROOT}/framework/ccu/ccu_manager/ccu_respack_mgr.cpp
        ${HCCL_V2_CODE_ROOT}/framework/ccu/ccu_manager/ccu_registered_ctx_mgr.cpp
        ${HCCL_V2_CODE_ROOT}/framework/ccu/ccu_mc2/mc2_compont.cpp
        ${HCCL_V2_CODE_ROOT}/framework/ccu/ccu_mc2/mc2_context.cpp

        ${HCCL_V2_CODE_ROOT}/framework/fault_recovery/recover_info.cc
        ${HCCL_V2_CODE_ROOT}/framework/fault_recovery/snap_shot_parse.cc
        ${HCCL_V2_CODE_ROOT}/common/checkcrc.cc

        ${HCCL_V2_CODE_ROOT}/common/stream_utils.cc
        ${HCCL_V2_CODE_ROOT}/framework/communicator/aicpu/daemon/aicpu_daemon_service.cpp

        ${HCCL_V2_CODE_ROOT}/framework/dfx/common/task_info.cc
        ${HCCL_V2_CODE_ROOT}/framework/dfx/common/global_mirror_tasks.cc
        ${HCCL_V2_CODE_ROOT}/framework/dfx/common/mirror_task_manager.cc
        ${HCCL_V2_CODE_ROOT}/framework/dfx/aicpu/common/mirror_task_manager_lite.cc
        ${HCCL_V2_CODE_ROOT}/framework/dfx/common/mc2_global_mirror_tasks.cc

        ${HCCL_V2_CODE_ROOT}/framework/resource_manager/transport/mem_transport_callback.cc
        ${HCCL_V2_CODE_ROOT}/framework/communicator/aicpu/daemon/aicpu_daemon_service.cpp

        ${HCCL_V2_CODE_ROOT}/framework/device_mode/coll_service_device_mode.cpp
        ${HCCL_V2_CODE_ROOT}/framework/device_mode/aicpu_kernel_launcher.cpp
        ${HCCL_V2_CODE_ROOT}/framework/device_mode/aicpu_ins_preprocessor.cpp
        ${HCCL_V2_CODE_ROOT}/framework/resource_manager/stream/aicpu_stream_manager.cpp
        ${HCCL_V2_CODE_ROOT}/framework/resource_manager/transport/aicpu/mem_transport_callback_lite.cc
        ${HCCL_V2_CODE_ROOT}/framework/dfx/aicpu/task_exception/task_exception_func.cpp
        ${HCCL_V2_CODE_ROOT}/framework/dfx/aicpu/task_exception/task_exception_handler_lite.cpp
        ${HCCL_V2_CODE_ROOT}/framework/dfx/aicpu/profiling/profiling_handler_lite.cc
        ${HCCL_V2_CODE_ROOT}/framework/dfx/aicpu/profiling/profiling_command_handle_lite.cc
        ${HCCL_V2_CODE_ROOT}/framework/dfx/aicpu/profiling/profiling_reporter_lite.cc
        ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/dfx/profiling/aicpu/dfx_profiling_handler_lite.cc
        ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/dfx/profiling/aicpu/dfx_profiling_command_handle_lite.cc
        ${HCCL_V2_CODE_ROOT}/framework/dfx/profiling/profiling_handler.cc
        ${HCCL_V2_CODE_ROOT}/framework/dfx/profiling/profiling_reporter.cc
        ${HCCL_V2_CODE_ROOT}/framework/dfx/profiling/dlprof_function.cc

        ${HCCL_V2_CODE_ROOT}/framework/service/one_sided_service/hccl_one_sided_service.cc
        ${HCCL_V2_CODE_ROOT}/framework/service/one_sided_service/hccl_one_sided_conn.cc
        ${HCCL_V2_CODE_ROOT}/unified_platform/resource/transport/transport_urma_mem.cc

        ${HCCL_V2_CODE_ROOT}/framework/resource_manager/stream/stream_manager.cpp
        ${HCCL_V2_CODE_ROOT}/framework/resource_manager/stream/opbase_stream_manager.cpp
        ${HCCL_V2_CODE_ROOT}/framework/resource_manager/stream/offload_stream_manager.cpp

        ${HCCL_V2_CODE_ROOT}/framework/ccu/ccu_manager/ccu_ctx_creator_registry.cpp
        ${HCCL_V2_CODE_ROOT}/framework/communicator/aicpu/one_sided_component/one_sided_component_lite.cpp
        ${HCCL_V2_CODE_ROOT}/framework/service/one_sided_service/hccl_one_sided_service.cc
        ${HCCL_V2_CODE_ROOT}/framework/service/one_sided_service/hccl_one_sided_conn.cc
        ${HCCL_V2_CODE_ROOT}/unified_platform/resource/transport/transport_urma_mem.cc

        ${HCCL_V2_CODE_ROOT}/unified_platform/common/tokenInfo_manager.cc

        ${HCCL_V2_CODE_ROOT}/unified_platform/resource/mem/hccl_mem.cc
        ${HCCL_V2_CODE_ROOT}/unified_platform/resource/mem/hccl_mem_v2.cc
        ${HCCL_V2_CODE_ROOT}/unified_platform/resource/mem/local_ub_rma_buffer_manager.cc

        ${HCCL_V2_CODE_ROOT}/framework/topo/rank_info_detect/preempt_port_manager.cc

        ${HCCL_V2_CODE_ROOT}/framework/topo/topo_addr_info/src/eid_util.c
        ${HCCL_V2_CODE_ROOT}/framework/topo/topo_addr_info/src/hal.c
        ${HCCL_V2_CODE_ROOT}/framework/topo/topo_addr_info/src/product_card.c
        ${HCCL_V2_CODE_ROOT}/framework/topo/topo_addr_info/src/product_server.c
        ${HCCL_V2_CODE_ROOT}/framework/topo/topo_addr_info/src/product_pod.c
        ${HCCL_V2_CODE_ROOT}/framework/topo/topo_addr_info/src/rank_info_types.c
        ${HCCL_V2_CODE_ROOT}/framework/topo/topo_addr_info/src/topo_addr_info.c
        ${HCCL_V2_CODE_ROOT}/framework/topo/topo_addr_info/src/topo.c

        ${HCCL_V2_CODE_ROOT}/framework/aiv/aiv_ins_preprocessor.cpp
        ${HCCL_V2_CODE_ROOT}/framework/aiv/aiv_ins/aiv_ins.cc
        ${HCCL_V2_CODE_ROOT}/framework/aiv/aiv_ins/hccl_aiv_utils.cc
        ${HCCL_V2_CODE_ROOT}/framework/aiv/aiv_mc2/aiv_mc2_compont.cpp
        ${HCCL_V2_CODE_ROOT}/framework/resource_manager/notify/ccu_stream_sync_notify_manager.cc
        ${HCCL_V2_CODE_ROOT}/unified_platform/common/orion_adapter_hal.cc
        ${HCCL_V2_CODE_ROOT}/unified_platform/common/dlhal_function_v2.cc
        ${HCCL_CODE_ROOT}/src/base_comm/common/debug/aicpu_task_utils.cc
)
