# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

target_sources(hccl_v2_stest
        PUBLIC
        st_commuincator_impl.cc
        st_hccl_communicator_pub.cc
        st_host_device_sync_notify_manager.cc
        st_mem_transport_manager.cc
        st_mem_transport_lite_mgr.cc
        st_connected_link_mgr.cc
        st_remote_rma_buffer_manager.cc
        st_stream_manager.cc
        ##st_stream_lite_manager.cc
        st_ub_ci_updater_manager.cc
        st_queue_notify_lite_mgr.cc
        st_cnt_1ton_notify_lite_mgr.cc
        st_cnt_nto1_notify_lite_mgr.cc
        ##st_host_device_sync_notify_lite_mgr.cc
        st_rma_conn_lite_mgr.cc
        #st_ins_to_sqe_rule_v82.cc
        st_aicpu_daemon_service.cpp
        st_aicpu_res_package_helper.cc
        st_socket_manager.cc
        st_rmt_data_buffer_mgr.cc
        st_hdc.cc
        ##st_aicpu_comm_destroy_func.cc
        st_ns_recovery_handler_func.cc
        st_task_abort_handler.cc
        st_hccl_one_sided_conn.cc
        st_hccl_one_sided_service.cc
        st_one_sided_component_lite.cpp
        st_coll_operator_check.cc
        st_ins_executor_new.cc
        st_comm_topo_desc.cc
)

add_subdirectory(ccu)
add_subdirectory(fault_recovery)
add_subdirectory(dfx)
add_subdirectory(interface)
add_subdirectory(aiv)