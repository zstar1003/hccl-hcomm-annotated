/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_RANK_GRAPH_V2_H
#define HCOMM_RANK_GRAPH_V2_H
#include "rank_graph_base.h"
#include "rank_graph_interface.h"
namespace hccl {

class RankGraphV2 : public RankGraph {
public:
    RankGraphV2();
    RankGraphV2(void* rankGraphPtr);
    ~RankGraphV2() override;

    HcclResult GetRankId(uint32_t* rank) override;
    HcclResult GetRankSize(uint32_t* rankSize) override;
    HcclResult
    GetLinks(uint32_t netLayer, uint32_t srcRank, uint32_t dstRank, CommLink** linkList, uint32_t* listSize) override;
    HcclResult GetRankGraphInfo(GraphType type, void** graph, uint32_t* len) override;
    HcclResult GetNetLayers(uint32_t** netLayers, uint32_t* netLayerNum) override;
    HcclResult GetInstTopoTypeByNetLayer(uint32_t netLayer, CommTopo* topoType) override;
    HcclResult GetInstSizeByNetLayer(uint32_t netLayer, uint32_t* rankNum) override;
    HcclResult GetInstRanksByNetLayer(uint32_t netLayer, uint32_t** rankList, uint32_t* rankNum) override;
    HcclResult GetInstSizeListByNetLayer(uint32_t netLayer, uint32_t** instSizeList, uint32_t* listSize) override;
    HcclResult GetDevicePort(const uint32_t rank, uint32_t* devPort) override;
    HcclResult GetListenPort(const uint32_t rank, uint32_t* listenPort, EndpointLocType locType) override;
    HcclResult GetTopoInstsByLayer(uint32_t netLayer, uint32_t** topoInsts, uint32_t* topoInstNum);
    HcclResult GetTopoType(const uint32_t netLayer, const uint32_t topoInstId, CommTopo* topoType);
    HcclResult
    GetRanksByTopoInst(const uint32_t netLayer, const uint32_t topoInstId, uint32_t** ranks, uint32_t* rankNum);
    HcclResult GetEndpointNum(uint32_t netLayer, uint32_t topoInstId, uint32_t* num);
    HcclResult GetEndpointDesc(uint32_t netLayer, uint32_t topoInstId, uint32_t* descNum, EndpointDesc* endpointDesc);
    HcclResult GetEndpointInfo(
        uint32_t rankId, const EndpointDesc* endPointDesc, EndpointAttr endpointAttr, uint32_t infoLen,
        void* info) override;
    HcclResult GetDeviceId(uint32_t rankId, uint32_t* deviceId) override;

private:
    std::unique_ptr<Hccl::IRankGraph> pImpl;
};
} // namespace hccl
#endif
