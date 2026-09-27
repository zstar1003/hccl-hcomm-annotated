# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

target_sources(hccl_v2_utest
        PUBLIC
        ut_communicator_impl.cc
        ut_hccl_communicator_pub.cc
        ut_kernel_entrance.cc
        ut_communicator_impl_lite.cc
        ut_aicpu_daemon_service.cpp
        ut_aicpu_res_package_helper.cc
        ut_task_abort_handler.cc
        ut_hdc.cc
        #ut_aicpu_comm_destroy_func.cc
        ut_ns_recovery_handler_func.cc
        ut_one_sided_component_lite.cpp
        ut_op_params_checker.cc
        ut_ins_executor.cc
        ut_aicpu_mc2_handler.cc
        ut_hccl_mc2_ex.cc
        ut_task_service.cc
        ut_flush_handle.cc
        ut_flush_manager.cc

        ${HCCL_V2_CODE_ROOT}/framework/fault_recovery/snap_shot_parse.cc
        ${HCCL_V2_CODE_ROOT}/common/checkcrc.cc
        ${HCCL_V2_CODE_ROOT}/framework/communicator/aicpu/one_sided_component/one_sided_component_lite.cpp
        ${HCCL_V2_CODE_ROOT}/framework/entrance/hcom_comm/communicator_callback.cc
        ${HCCL_V2_CODE_ROOT}/framework/communicator/ccu_super_fast_load.cc
        ${HCCL_V2_CODE_ROOT}/framework/communicator/aicpu/aicpu_mc2_handler.cc
        ${HCCL_V2_CODE_ROOT}/framework/communicator/aicpu/hccl_mc2_ex.cc
        ${HCCL_V2_CODE_ROOT}/framework/communicator/hostdpu/flush_manager.cc
        ${HCCL_V2_CODE_ROOT}/framework/communicator/hostdpu/flush_handle.cc
        ${HCCL_V2_CODE_ROOT}/framework/communicator/hostdpu/dpu_kernel_entrance.cc
        ${HCCL_V2_CODE_ROOT}/framework/communicator/hostdpu/task_service.cc
        ${HCCL_V2_CODE_ROOT}/framework/communicator/comm_topo_desc.cc
)