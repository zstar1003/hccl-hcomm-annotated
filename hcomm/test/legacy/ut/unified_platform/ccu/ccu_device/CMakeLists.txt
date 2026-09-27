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
    ut_ccu_res_specs.cpp
    ut_ccu_component.cpp
    ut_ccu_res_batch_allocator.cpp
    ut_ccu_pfe.cc

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

    ${HCCL_V2_CODE_ROOT}/unified_platform/common/rdma_handle_manager.cc
    ${HCCL_V2_CODE_ROOT}/unified_platform/common/tp_manager.cc
    ${HCCL_V2_CODE_ROOT}/unified_platform/common/tp_qos.cc
    ${HCCL_V2_CODE_ROOT}/unified_platform/external_system/orion_adapter_hccp.cc
    ${HCCL_V2_CODE_ROOT}/unified_platform/external_system/orion_adapter_rts.cc
    
    ${HCCL_V2_CODE_ROOT}/unified_platform/trace/trace.cc
    
    ${HCCL_V2_CODE_ROOT}/framework/env_config/env_func.cc
    ${HCCL_V2_CODE_ROOT}/framework/env_config/base_config_legacy.cc
    ${HCCL_V2_CODE_ROOT}/framework/env_config/env_config.cc

    ${HCCL_V2_CODE_ROOT}/framework/misc/json_parser/json_parser.cc
    ${HCOMM_DIR}/src/coll_communicator_mgr/rank_info_detect/whitelist.cc
    ${HCCL_V2_CODE_ROOT}/framework/misc/mask_event/mask_event.cc

    ${HCCL_V2_CODE_ROOT}/unified_platform/resource/stream/stream.cc
    ${HCCL_V2_CODE_ROOT}/unified_platform/resource/buffer/buffer.cc
    ${HCCL_V2_CODE_ROOT}/unified_platform/resource/buffer/local_ub_rma_buffer.cc

    ${HCCL_V2_CODE_ROOT}/unified_platform/common/inner_net_dev_manager.cc
    ${HCCL_V2_CODE_ROOT}/unified_platform/common/inner_net_dev.cc
    ${HCCL_V2_CODE_ROOT}/unified_platform/common/net_device.cc
    ${HCCL_V2_CODE_ROOT}/unified_platform/resource/net_dev/hccl_net_dev_v2.cc
    ${HCCL_V2_CODE_ROOT}/unified_platform/resource/net_dev/hccl_net_dev.cc

    ${HCCL_V2_CODE_ROOT}/common/checkcrc.cc
)
