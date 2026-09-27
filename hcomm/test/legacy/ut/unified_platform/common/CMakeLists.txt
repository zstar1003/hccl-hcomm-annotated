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
        ut_hccl_dl_v2.cc
        ut_hccp_hdc_manager.cc
        ut_p2p_enable_manager.cc
        ut_rdma_handle_manager.cc
        ut_hccp_peer_manager.cc
        ut_dev_capability.cc
        ut_tp_manager.cc
        ut_inner_net_dev.cc
        ut_inner_net_dev_manager.cc
        ut_net_device.cc
        ut_hccl_net_dev_v2.cc
        ut_tokenInfo_manager.cc
        ut_ccu_driver_handle.cc
        ut_hccp_tlv_hdc_manager.cc
)