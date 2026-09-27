/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl_mem_defs.h"
#include "aicputs_hccs_endpoint.h"
#include "log.h"
#include "net_dev/global_net_dev_manager.h"
#include "hccs_reged_mem_mgr.h"

using namespace hccl;

namespace hcomm {
AicpuTsHccsEndpoint::AicpuTsHccsEndpoint(const EndpointDesc& endpointDesc) : Endpoint(endpointDesc) {}

AicpuTsHccsEndpoint::~AicpuTsHccsEndpoint()
{
    try {
        // 析构顺序保持：先停止 GlobalNetDevMgr Server（子类 context 承载），再 UnRefNetDevCtx
        if (serverSocketContext_.has_value()) {
            (void)serverSocketContext_->ServerSocketStopListen(serverPort_);
        }
    } catch (...) {
    }

    regedMemMgr_ = nullptr;

    try {
        if (netDevCtx_ != nullptr) {
            (void)hccl::GlobalNetDevMgr::GetInstance(endpointDesc_.loc.device.devPhyId)
                .UnRefNetDevCtx(NicType::VNIC_TYPE, devIpAddr_, serverPort_);
            netDevCtx_ = nullptr;
        }
    } catch (...) {
    }
}

HcclResult AicpuTsHccsEndpoint::Init()
{
    HCCL_INFO(
        "[%s]localEndpoint protocol[%d], type[%d], id[%u] locType[%d], devPhyId[%u], serverIdx[%u], "
        "superDevId[%u], superPodIdx[%u]",
        __func__, endpointDesc_.protocol, endpointDesc_.commAddr.type, endpointDesc_.commAddr.id,
        endpointDesc_.loc.locType, endpointDesc_.loc.device.devPhyId, endpointDesc_.loc.device.serverIdx,
        endpointDesc_.loc.device.superDevId, endpointDesc_.loc.device.superPodIdx);

    if (endpointDesc_.loc.locType != ENDPOINT_LOC_TYPE_DEVICE) {
        HCCL_INFO("[AicpuTsHccsEndpoint][%s] AicpuTsHccsEndpoint not support host", __func__);
        return HCCL_E_NOT_SUPPORT;
    }

    serverSocketContext_.emplace(endpointDesc_.loc.device.devPhyId, serverPort_);

    u32 devPhyId = endpointDesc_.loc.device.devPhyId;
    uint32_t superDevId = endpointDesc_.loc.device.superDevId;
    CHK_RET(GlobalNetDevMgr::GetDeviceVnicIP(devPhyId, superDevId, devIpAddr_));
    HCCL_INFO(
        "[AicpuTsHccsEndpoint]devPhyId[%u] superDevId[%u] devIpAddr_[%s] ", devPhyId, superDevId,
        devIpAddr_.GetReadableAddress());

    CHK_RET(hccl::GlobalNetDevMgr::GetInstance(endpointDesc_.loc.device.devPhyId)
                .RefNetDevCtx(NicType::VNIC_TYPE, devIpAddr_, serverPort_, netDevCtx_));
    EXCEPTION_CATCH(regedMemMgr_ = std::make_shared<HccsRegedMemMgr>(netDevCtx_), return HCCL_E_PARA);
    return HCCL_SUCCESS;
}
} // namespace hcomm
