/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RANK_HCOMM_GRAPH_H
#define RANK_HCOMM_GRAPH_H

#include "topoinfo_struct.h"
#include "hccl/hccl_res.h"
#include "hccl_independent_common.h"
#include "hccl_common.h"
#include "hccl_impl_pub.h"
#include "hccl_rank_graph.h"
#include "hccl_rankgraph.h"

namespace hccl {
class RankGraph {
public:
    static constexpr uint32_t HCCL_NETLAYER_0 = 0;
    static constexpr uint32_t HCCL_NETLAYER_1 = 1;
    static constexpr uint32_t HCCL_NETLAYER_2 = 2;

    RankGraph() = default;
    virtual ~RankGraph() = default;
    virtual HcclResult
    Init([[maybe_unused]] const RankTable_t& rankTable, [[maybe_unused]] const HcclTopoAttr& topoAttr)
    {
        return HCCL_SUCCESS;
    };
    virtual HcclResult GetRankId([[maybe_unused]] uint32_t* rank) { return HCCL_E_NOT_SUPPORT; };
    virtual HcclResult GetRankSize([[maybe_unused]] uint32_t* rankSize) { return HCCL_E_NOT_SUPPORT; };
    virtual HcclResult GetDevicePort([[maybe_unused]] const uint32_t rank, [[maybe_unused]] uint32_t* devPort)
    {
        return HCCL_E_NOT_SUPPORT;
    };
    virtual HcclResult GetListenPort(
        [[maybe_unused]] const uint32_t rank, [[maybe_unused]] uint32_t* listenPort,
        [[maybe_unused]] EndpointLocType locType)
    {
        return HCCL_E_NOT_SUPPORT;
    };
    virtual HcclResult GetRankGraphInfo(GraphType type, void** graph, uint32_t* len) = 0;
    virtual HcclResult
    GetLinks(uint32_t netLayer, uint32_t srcRank, uint32_t dstRank, CommLink** linkList, uint32_t* listSize)
        = 0;
    virtual HcclResult GetNetLayers(uint32_t** netLayers, uint32_t* netLayerNum) = 0;
    virtual HcclResult GetHeterogMode([[maybe_unused]] HcclHeterogMode* mode) const { return HCCL_SUCCESS; }
    virtual HcclResult GetInstTopoTypeByNetLayer(uint32_t netLayer, CommTopo* topoType) = 0;
    virtual HcclResult GetInstSizeByNetLayer(uint32_t netLayer, uint32_t* rankNum) = 0;
    virtual HcclResult GetInstRanksByNetLayer(uint32_t netLayer, uint32_t** rankList, uint32_t* rankNum) = 0;
    virtual HcclResult GetInstSizeListByNetLayer(uint32_t netLayer, uint32_t** instSizeList, uint32_t* listSize) = 0;
    virtual HcclResult GetDeviceId(uint32_t rankId, uint32_t* deviceId) = 0;
    virtual HcclResult GetEndpointInfo(
        uint32_t rankId, const EndpointDesc* endPointDesc, EndpointAttr endpointAttr, uint32_t infoLen, void* info)
        = 0;
};
} // namespace hccl
#endif
