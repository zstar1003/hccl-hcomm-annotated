/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RANK_GRAPH_V1_H
#define RANK_GRAPH_V1_H

#include "topoinfo_struct.h"
#include "hccl/hccl_res.h"
#include "hccl_independent_common.h"
#include "hccl_common.h"
#include "hccl_impl_pub.h"
#include "hccl_rank_graph.h"
#include "hccl_rankgraph.h"
#include "rank_graph_base.h"
namespace hccl {

class RankGraphV1 : public RankGraph {
    struct RankGraphInfo {
        RankInfo_t rankInfo;
        std::vector<EndpointDesc> endPoints;
    };

    enum class HcclNetLayerlevel {
        HCCL_NetLayer_L0 = 0,
        HCCL_NetLayer_L1,
        HCCL_NetLayer_L2,
        HCCL_NetLayer_MAX,
    };

public:
    HcclResult Init(const RankTable_t& rankTable, const HcclTopoAttr& topoAttr) override;
    HcclResult Init(const HcclTopoAttr& topoAttr);
    HcclResult
    GetLinks(uint32_t netLayer, uint32_t srcRank, uint32_t dstRank, CommLink** linkList, uint32_t* listSize) override;
    HcclResult GetHeterogMode(HcclHeterogMode* mode) const override;
    const RankInfo_t* FindRank(uint32_t rankId) const;
    HcclResult GetRankGraphInfo(GraphType type, void** graph, uint32_t* len) override;
    HcclResult GetNetLayers(uint32_t** netLayers, uint32_t* netLayerNum) override;
    HcclResult GetDeviceId(uint32_t rankId, uint32_t* deviceId) override;
    HcclResult GetInstTopoTypeByNetLayer(uint32_t netLayer, CommTopo* topoType) override;
    HcclResult GetInstSizeByNetLayer(uint32_t netLayer, uint32_t* rankNum) override;
    HcclResult GetInstRanksByNetLayer(uint32_t netLayer, uint32_t** rankList, uint32_t* rankNum) override;
    HcclResult GetInstSizeListByNetLayer(uint32_t netLayer, uint32_t** instSizeList, uint32_t* listSize) override;

    HcclResult GetTopoInstsByLayer(uint32_t netLayer, uint32_t** topoInsts, uint32_t* topoInstNum);
    HcclResult GetTopoType(uint32_t netLayer, CommTopo* topoType);
    HcclResult GetRanksByTopoInst(uint32_t netLayer, uint32_t topoInstId, uint32_t** ranks, uint32_t* rankNum);
    std::vector<const RankInfo_t*> GetRanksInTopoInst(uint32_t netLayer, uint32_t topoInstId);
    std::set<CommProtocol>
    GetProtocolsByConnections(uint32_t netLayer, const std::vector<const RankInfo_t*>& topoInstRanks);
    HcclResult GetEndpointNum(uint32_t netLayer, uint32_t topoInstId, uint32_t* num);
    HcclResult GetEndpointDesc(uint32_t netLayer, uint32_t topoInstId, uint32_t* descNum, EndpointDesc* endpointDesc);
    HcclResult GetEndpointInfo(
        uint32_t rankId, const EndpointDesc* endPointDesc, EndpointAttr endpointAttr, uint32_t infoLen,
        void* info) override;
    HcclResult GetRankSize(uint32_t* rankSize) override;
    HcclResult GetDevicePort(const uint32_t rank, uint32_t* devPort) override;

private:
    const EndpointDesc* MatchEndpointByAddr(const RankGraphInfo& rankGraphInfo, const EndpointDesc* endPointDesc) const;
    HcclResult
    FillAttr(EndpointAttr endpointAttr, const EndpointDesc* foundEndpoint, uint32_t infoLen, void* info) const;
    HcclResult DevTypeToCommProtocol(DevType& type, CommProtocol& protocol) const;
    HcclResult
    BuildRankGraphInfo(const RankInfo_t& rankItem, const CommProtocol& protocol, RankGraphInfo& outInfo) const;
    CommProtocol GetCommProtocolFromRankInfo(const RankInfo_t& srcInfo, const RankInfo_t& dstInfo, uint32_t netLayer);
    HcclResult InitRankInfo();
    HcclResult InitServerRankInfo();
    HcclResult InitSuperPodRankInfo();
    HcclResult InitNetLayer();
    HcclResult InitGraphRankInfo();
    CommProtocol GetCommProtocolInSameServer(const RankInfo_t& srcInfo, const RankInfo_t& dstInfo);
    CommProtocol GetCommProtocolBetweenServers(const RankInfo_t& srcInfo, const RankInfo_t& dstInfo) const;
    bool NeedIgnoreEndPoints(CommProtocol srcProtocol, CommProtocol dstProtocol, CommProtocol linkProtocol) const;
    void PrintLinksInfo(CommLink& link) const;
    bool IsRoceInSameServer(uint32_t netLayer, const RankInfo_t& srcInfo, const RankInfo_t& dstInfo);
    HcclResult InitHeterogMode();
    RankTable_t rankTable_;
    std::unordered_map<uint32_t, RankGraphInfo> rankIndex_;
    std::map<std::tuple<uint32_t, uint32_t, uint32_t>, std::vector<CommLink>> rankPairInfo_;
    std::vector<uint32_t> netLayer_;
    std::unordered_map<uint32_t, std::vector<u32>> rankList_;     // level->rankList
    std::unordered_map<uint32_t, std::vector<u32>> rankSizeList_; // level->rankSizeList
    std::vector<RankInfo_t> rankGraph_;
    std::vector<struct GraphRankInfo> graphRankInfo_;
    std::vector<uint32_t> topoInstsVec_; // 供 GetTopoInstsByLayer 输出 topoInsts
    HcclTopoAttr topoAttr_;
    RankInfo rankData_;
    DevType devType_ = DevType::DEV_TYPE_NOSOC;
    HcclHeterogMode heterogMode_{HcclHeterogMode::HCCL_HETEROG_MODE_INVALID};

    std::map<u32, std::vector<RankInfo>> serverToRank_;
    std::map<u32, std::vector<RankInfo>> superPodToRank_;
};
} // namespace hccl
#endif
