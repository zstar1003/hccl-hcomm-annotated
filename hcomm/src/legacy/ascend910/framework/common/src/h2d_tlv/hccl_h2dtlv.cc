/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <unistd.h>
#include <hccl/hccl_types.h>
#include "hccl_communicator.h"
#include "adapter_hccp_common.h"
#include "../nslbdp/hccl_nslbdp_pub.h"
#include "hccl_h2dtlv.h"

namespace hccl {

hcclH2dTlv& hcclH2dTlv::GetInstance()
{
    static hcclH2dTlv phcclH2dTlv;
    return phcclH2dTlv;
}

HcclResult hcclH2dTlv::InitHccpChannel(u32 devicePhyId)
{
    /* 避免重复初始化  */
    if (hcclH2dTlvInitFlag_ == true) {
        HCCL_INFO("hcclNslbDp getHccpInitFlag is init");
        return HCCL_SUCCESS;
    }

    u32 phyID = (static_cast<s32>(devicePhyId) == HOST_DEVICE_ID) ? 0 : devicePhyId;
    nslb_inithccp_info nslbHccp;
    nslbHccp.version = NSLBDP_HCCP_VERSION;
    nslbHccp.phyId = phyID;
    nslbHccp.nic_posion = NSLBDP_HCCP_NICPOSION;

    u32 tlvBuffersize = 0;
    void* tlvHandle;
    HCCL_INFO(
        "Entry InitHccpChannel version:[%u]-phy_id:[%u]-nic_position:[%u] .", nslbHccp.version, nslbHccp.phyId,
        nslbHccp.nic_posion);

    HcclResult ret = H2DTlvInit(reinterpret_cast<TlvInitInfo*>(&nslbHccp), &tlvBuffersize, &tlvHandle);
    if (ret != HCCL_SUCCESS) {
        return ret;
    }

    hcclH2dTlvBuffsize_ = tlvBuffersize;
    hcclH2dTlvHandle_ = tlvHandle;
    hcclH2dTlvInitFlag_ = true;
    return HCCL_SUCCESS;
}

void hcclH2dTlv::DeinitHccpChannel()
{
    if (hcclH2dTlvInitFlag_ == false) {
        HCCL_INFO("Hccp channel is already deinit.");
        return;
    }
    if (hcclH2dTlvHandle_ == nullptr) {
        HCCL_INFO("Try to deinit hccp channel while hcclH2dTlvHandle_ is null.");
        return;
    }
    HcclResult ret = H2DTlvDeinit(hcclH2dTlvHandle_);
    if (ret != HCCL_SUCCESS) {
        HCCL_WARNING("DeInit hccp channel failed ret[%u].", ret);
        return;
    }

    hcclH2dTlvBuffsize_ = H2D_TLVBUFFERSIZE;
    hcclH2dTlvHandle_ = nullptr;
    hcclH2dTlvInitFlag_ = false;
    return;
}

bool hcclH2dTlv::GetH2dTlvInitFlag() { return hcclH2dTlvInitFlag_; }

unsigned int hcclH2dTlv::GetH2dTlvBufferSize() const { return hcclH2dTlvBuffsize_; }

void* hcclH2dTlv::GetH2dTlvHandle() { return hcclH2dTlvHandle_; }

} // namespace hccl
