/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "rank_graph_v2.h"

namespace hccl {

RankGraphV2::RankGraphV2() {}

RankGraphV2::~RankGraphV2() {}

RankGraphV2::RankGraphV2(void* rankGraphPtr) { pImpl = std::make_unique<Hccl::IRankGraph>(rankGraphPtr); }

HcclResult RankGraphV2::GetRankSize(uint32_t* rankSize) { return pImpl->GetRankSize(rankSize); }

HcclResult RankGraphV2::GetDevicePort(const uint32_t rank, uint32_t* devPort)
{
    CHK_RET(pImpl->GetDevicePort(rank, devPort));
    return HCCL_SUCCESS;
}

HcclResult RankGraphV2::GetListenPort(const uint32_t rank, uint32_t* listenPort, EndpointLocType locType)
{
    if (locType == EndpointLocType::ENDPOINT_LOC_TYPE_DEVICE) {
        CHK_RET(pImpl->GetDevicePort(rank, listenPort));
    } else if (locType == EndpointLocType::ENDPOINT_LOC_TYPE_HOST) {
        CHK_RET(pImpl->GetHostPort(rank, listenPort));
    } else {
        HCCL_ERROR("[%s] Invalid locType[%d] for rank[%u]", __func__, locType, rank);
        return HcclResult::HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

HcclResult RankGraphV2::GetRankId(uint32_t* rank) { return pImpl->GetRankId(rank); }

HcclResult
RankGraphV2::GetLinks(uint32_t netLayer, uint32_t srcRank, uint32_t dstRank, CommLink** linkList, uint32_t* listSize)
{
    return pImpl->GetLinks(netLayer, srcRank, dstRank, linkList, listSize);
}

HcclResult RankGraphV2::GetRankGraphInfo([[maybe_unused]] GraphType type, void** graph, uint32_t* len)
{
    return pImpl->GetRankGraphInfo(graph, len);
}

HcclResult RankGraphV2::GetDeviceId(uint32_t rankId, uint32_t* deviceId)
{
    return pImpl->GetDeviceId(rankId, deviceId);
}

HcclResult RankGraphV2::GetNetLayers(uint32_t** netLayers, uint32_t* netLayerNum)
{
    return pImpl->GetNetLayers(netLayers, netLayerNum);
}

HcclResult RankGraphV2::GetInstTopoTypeByNetLayer(uint32_t netLayer, CommTopo* topoType)
{
    return pImpl->GetInstTopoTypeByNetLayer(netLayer, topoType);
}

HcclResult RankGraphV2::GetInstSizeByNetLayer(uint32_t netLayer, uint32_t* rankNum)
{
    return pImpl->GetInstSizeByNetLayer(netLayer, rankNum);
}

HcclResult RankGraphV2::GetInstRanksByNetLayer(uint32_t netLayer, uint32_t** rankList, uint32_t* rankNum)
{
    return pImpl->GetInstRanksByNetLayer(netLayer, rankList, rankNum);
}

HcclResult RankGraphV2::GetInstSizeListByNetLayer(uint32_t netLayer, uint32_t** instSizeList, uint32_t* listSize)
{
    return pImpl->GetInstSizeListByNetLayer(netLayer, instSizeList, listSize);
}

HcclResult RankGraphV2::GetTopoInstsByLayer(uint32_t netLayer, uint32_t** topoInsts, uint32_t* topoInstNum)
{
    return pImpl->GetTopoInstsByLayer(netLayer, topoInsts, topoInstNum);
}

HcclResult RankGraphV2::GetTopoType(const uint32_t netLayer, const uint32_t topoInstId, CommTopo* topoType)
{
    return pImpl->GetTopoType(netLayer, topoInstId, topoType);
}

HcclResult
RankGraphV2::GetRanksByTopoInst(const uint32_t netLayer, const uint32_t topoInstId, uint32_t** ranks, uint32_t* rankNum)
{
    return pImpl->GetRanksByTopoInst(netLayer, topoInstId, ranks, rankNum);
}

HcclResult RankGraphV2::GetEndpointNum(uint32_t netLayer, uint32_t topoInstId, uint32_t* num)
{
    return pImpl->GetEndpointNum(netLayer, topoInstId, num);
}

HcclResult
RankGraphV2::GetEndpointDesc(uint32_t netLayer, uint32_t topoInstId, uint32_t* descNum, EndpointDesc* endpointDesc)
{
    return pImpl->GetEndpointDesc(netLayer, topoInstId, descNum, endpointDesc);
}

HcclResult RankGraphV2::GetEndpointInfo(
    uint32_t rankId, const EndpointDesc* endPointDesc, EndpointAttr endpointAttr, uint32_t infoLen, void* info)
{
    return pImpl->GetEndpointInfo(rankId, endPointDesc, endpointAttr, infoLen, info);
}

}; // namespace hccl
