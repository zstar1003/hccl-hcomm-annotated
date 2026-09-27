# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if(TARGET hccl_v2)
    target_sources(hccl_v2 PRIVATE
            hccp_hdc_manager.cc
            hccp_tlv_hdc_manager.cc
            p2p_enable_manager.cc
            rdma_handle_manager.cc
            hccp_peer_manager.cc
            dev_capability.cc
            dlprof_func.cc
            tp_manager.cc
            tp_qos.cc
            inner_net_dev_manager.cc
            inner_net_dev.cc
            net_device.cc
            tokenInfo_manager.cc
            dlhal_function_v2.cc
            ${HCOMM_DIR}/src/base_comm/common/hccl_dl.cc
            orion_adapter_hal.cc
    )
endif()

if(TARGET ccl_kernel)
    target_sources(ccl_kernel PRIVATE
            dev_capability.cc
            dlhal_function_v2.cc
            orion_adapter_hal.cc
    )
endif()
