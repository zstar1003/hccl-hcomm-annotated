/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl_comm_pub.h"
#include "hccl_communicator.h"

namespace hccl {
HcclResult hcclComm::RegistTaskAbortHandler() const { return HCCL_SUCCESS; }

HcclResult hcclComm::UnRegistTaskAbortHandler() const { return HCCL_SUCCESS; }

HcclResult hcclComm::GetOneSidedService([[maybe_unused]] IHcclOneSidedService** service) { return HCCL_SUCCESS; }
HcclResult hcclComm::InitOneSidedServiceNetDevCtx([[maybe_unused]] u32 remoteRankId) { return HCCL_SUCCESS; }
HcclResult
hcclComm::OneSidedServiceStartListen([[maybe_unused]] NicType nicType, [[maybe_unused]] HcclNetDevCtx netDevCtx)
{
    return HCCL_SUCCESS;
}
HcclResult hcclComm::GetOneSidedServiceDevIpAndPort(
    [[maybe_unused]] NicType nicType, [[maybe_unused]] HcclIpAddress& ipAddress, [[maybe_unused]] u32& port)
{
    return HCCL_SUCCESS;
}
HcclResult hcclComm::DeinitOneSidedService() { return HCCL_SUCCESS; }

HcclResult
hcclComm::RegisterCommUserMem([[maybe_unused]] void* addr, [[maybe_unused]] u64 size, [[maybe_unused]] void** handle)
{
    return HCCL_SUCCESS;
}

HcclResult hcclComm::DeregisterCommUserMem([[maybe_unused]] void* handle) { return HCCL_SUCCESS; }

HcclResult hcclComm::ExchangeCommUserMem([[maybe_unused]] void* handle, [[maybe_unused]] std::vector<u32>& peerRanks)
{
    return HCCL_SUCCESS;
}

HcclResult hcclComm::SetIndependentOpConfig(
    [[maybe_unused]] const CommConfig& commConfig, [[maybe_unused]] const RankTable_t& rankTable)
{
    return HCCL_SUCCESS;
}

HcclResult hcclComm::InitIndependentOp() { return HCCL_SUCCESS; }

HcclResult hcclComm::PrepareChannelMem(
    [[maybe_unused]] const std::string& tag, [[maybe_unused]] TransportIOMem& transMem,
    [[maybe_unused]] const HcclMemHandle* memHandles, [[maybe_unused]] uint32_t memHandleNum)
{
    return HCCL_SUCCESS;
}
HcclResult hcclComm::IndOpTransportAlloc(
    [[maybe_unused]] const std::string& tag, [[maybe_unused]] OpCommTransport& opCommTransport,
    [[maybe_unused]] bool isAicpuModeEn, [[maybe_unused]] const HcclMemHandle* memHandles,
    [[maybe_unused]] uint32_t memHandleNum)
{
    return HCCL_SUCCESS;
}
HcclResult hcclComm::CommGetNetLayers([[maybe_unused]] uint32_t** netLayers, [[maybe_unused]] uint32_t* netLayerNum)
{
    return HCCL_SUCCESS;
}

HcclResult hcclComm::CommGetInstSizeByNetLayer([[maybe_unused]] uint32_t netLayer, [[maybe_unused]] uint32_t* rankNum)
{
    return HCCL_SUCCESS;
}

HcclResult hcclComm::CommGetInstTopoTypeByNetLayer([[maybe_unused]] uint32_t netLayer, [[maybe_unused]] u32* topoType)
{
    return HCCL_SUCCESS;
}
HcclResult hcclComm::GetNetLayers([[maybe_unused]] uint32_t** netLayers, [[maybe_unused]] uint32_t* netLayerNum)
{
    return HCCL_SUCCESS;
}

HcclResult hcclComm::GetInstSizeByNetLayer([[maybe_unused]] uint32_t netLayer, [[maybe_unused]] uint32_t* rankNum)
{
    return HCCL_SUCCESS;
}

HcclResult hcclComm::GetInstTopoTypeByNetLayer([[maybe_unused]] uint32_t netLayer, [[maybe_unused]] CommTopo* topoType)
{
    return HCCL_SUCCESS;
}

HcclResult hcclComm::GetInstRanksByNetLayer(
    [[maybe_unused]] uint32_t netLayer, [[maybe_unused]] uint32_t** rankList, [[maybe_unused]] uint32_t* rankNum)
{
    return HCCL_SUCCESS;
}

HcclResult hcclComm::GetInstSizeListByNetLayer(
    [[maybe_unused]] uint32_t netLayer, [[maybe_unused]] uint32_t** instSizeList, [[maybe_unused]] uint32_t* listSize)
{
    return HCCL_SUCCESS;
}

HcclResult
hcclComm::GetRankGraph([[maybe_unused]] GraphType type, [[maybe_unused]] void** graph, [[maybe_unused]] uint32_t* len)
{
    return HCCL_SUCCESS;
}

HcclResult hcclComm::GetLinks(
    [[maybe_unused]] uint32_t netLayer, [[maybe_unused]] uint32_t srcRank, [[maybe_unused]] uint32_t dstRank,
    [[maybe_unused]] CommLink** linkList, [[maybe_unused]] uint32_t* listSize)
{
    return HCCL_SUCCESS;
}

HcclResult hcclComm::GetHeterogMode([[maybe_unused]] HcclHeterogMode* mode) { return HCCL_SUCCESS; }

HcclComm hcclComm::GetCommunicatorV2()
{
    HCCL_ERROR("[HcclComm][GetCommunicatorV2]collComm_ is nullptr");
    return nullptr;
}

void hcclComm::BinaryUnLoad() { binHandle_ = nullptr; }

HcclResult hcclComm::Resume()
{
    if (!IsCommunicatorV2()) {
        CHK_RET(communicator_->Resume());
    }
    return HCCL_SUCCESS;
}

} // namespace hccl
