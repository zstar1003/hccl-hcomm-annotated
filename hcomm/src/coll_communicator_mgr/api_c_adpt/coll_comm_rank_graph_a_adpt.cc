/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl/hccl_res.h"
#include "log.h"
#include "hccl_comm_pub.h"
#include "independent_op.h"
#include <string>
#include "param_check_pub.h"
#include "hccl_comm.h"
#include "hccl_inner.h"
#include "rank_graph.h"
#include "rank_graph_v2.h"
#include "op_base.h"
#include "hccl_independent_common.h"

using namespace hccl;

#ifndef CCL_KERNEL_AICPU
HcclResult HcclGetRankGraph(HcclComm comm, GraphType type, void** graph, uint32_t* len)
{
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(graph);
    CHK_PTR_NULL(len);
    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
    HcclResult ret = HCCL_SUCCESS;
    if (hcclComm->IsCommunicatorV2()) {
        CollComm* collComm = hcclComm->GetCollComm();
        CHK_PTR_NULL(collComm);
        RankGraph* rankGraph = collComm->GetRankGraph();
        CHK_PTR_NULL(rankGraph);
        ret = rankGraph->GetRankGraphInfo(type, graph, len);
    } else {
        ret = hcclComm->GetRankGraph(type, graph, len);
    }
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[%s] Failed to HcclGetRankGraph ret[%d]", __func__, ret);
        return ret;
    }
    HCCL_RUN_INFO("[%s] success, group[%s], len[%u]", __func__, hcclComm->GetIdentifier().c_str(), *len);
    return HCCL_SUCCESS;
}

static inline HcclResult GetRankGraphFromComm(HcclComm comm, RankGraph** rankGraph)
{
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(rankGraph);
    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
    CollComm* collComm = hcclComm->GetCollComm();
    CHK_PTR_NULL(collComm);
    *rankGraph = collComm->GetRankGraph();
    CHK_PTR_NULL(*rankGraph);
    return HCCL_SUCCESS;
}

HcclResult HcclRankGraphGetLinks(
    HcclComm comm, uint32_t netLayer, uint32_t srcRank, uint32_t dstRank, CommLink** links, uint32_t* linkNum)
{
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(links);
    CHK_PTR_NULL(linkNum);
    HcclResult ret = HCCL_SUCCESS;
    HCCLV2_FUNC_RUN([&]() -> HcclResult {
        if (srcRank == dstRank) {
            HCCL_ERROR("[%s] srcRank[%u] and dstRank[%u] is same", __func__, srcRank, dstRank);
            return HCCL_E_PARA;
        }
        RankGraph* rankGraph = nullptr;
        CHK_RET(GetRankGraphFromComm(comm, &rankGraph));
        CHK_RET(rankGraph->GetLinks(netLayer, srcRank, dstRank, links, linkNum));
        HCCL_INFO(
            "HcclRankGraphGetLinks success with netLayer[%u], srcRank[%u], dstRank[%u], output linkNum[%u]", netLayer,
            srcRank, dstRank, *linkNum);
        return HCCL_SUCCESS;
    }());
    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
    HCCL_RUN_INFO(
        "Entry-%s: comm[%s], netLayer[%u], srcRank[%u], dstRank[%u]", __func__, hcclComm->GetIdentifier().c_str(),
        netLayer, srcRank, dstRank);
    ret = hcclComm->GetLinks(netLayer, srcRank, dstRank, links, linkNum);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[%s] Failed to get links for netLayer[%u], srcRank[%u], dstRank[%u] ret[%d]", __func__, netLayer, srcRank,
            dstRank, ret);
        return ret;
    }
    HCCL_RUN_INFO("[%s] success: comm[%s] linkNum[%u]", __func__, hcclComm->GetIdentifier().c_str(), *linkNum);
    return HCCL_SUCCESS;
}

HcclResult HcclRankGraphGetLayers(HcclComm comm, uint32_t** netLayers, uint32_t* netLayerNum)
{
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(netLayers);
    CHK_PTR_NULL(netLayerNum);
    HcclResult ret = HCCL_SUCCESS;
    HCCLV2_FUNC_RUN([&]() -> HcclResult {
        RankGraph* rankGraph = nullptr;
        CHK_RET(GetRankGraphFromComm(comm, &rankGraph));
        CHK_RET(rankGraph->GetNetLayers(netLayers, netLayerNum));
        HCCL_INFO("HcclRankGraphGetLayers success, netLayerNum [%u]", *netLayerNum);
        return HCCL_SUCCESS;
    }());
    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
    ret = hcclComm->GetNetLayers(netLayers, netLayerNum);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[%s] Failed to GetCommNetLayers ret[%d]", __func__, ret);
        return ret;
    }
    HCCL_RUN_INFO(
        "[%s] success, group[%s], netLayerNum size[%u]", __func__, hcclComm->GetIdentifier().c_str(), *netLayerNum);
    return HCCL_SUCCESS;
}

HcclResult HcclRankGraphGetTopoTypeByLayer(HcclComm comm, uint32_t netLayer, CommTopo* topoType)
{
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(topoType);
    HcclResult ret = HCCL_SUCCESS;
    HCCLV2_FUNC_RUN([&]() -> HcclResult {
        RankGraph* rankGraph = nullptr;
        CHK_RET(GetRankGraphFromComm(comm, &rankGraph));
        CHK_RET(rankGraph->GetInstTopoTypeByNetLayer(netLayer, topoType));
        HCCL_INFO("HcclRankGraphGetTopoTypeByLayer success, topoType [%d]", *topoType);
        return HCCL_SUCCESS;
    }());
    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
    ret = hcclComm->GetInstTopoTypeByNetLayer(netLayer, topoType);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[%s] Failed, ret[%d]", __func__, ret);
        return ret;
    }
    HCCL_RUN_INFO("[%s] success, group[%s], [%d]", __func__, hcclComm->GetIdentifier().c_str(), *topoType);
    return HCCL_SUCCESS;
}

HcclResult HcclRankGraphGetRankSizeByLayer(HcclComm comm, uint32_t netLayer, uint32_t* rankNum)
{
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(rankNum);

    HcclResult ret = HCCL_SUCCESS;
    HCCLV2_FUNC_RUN([&]() -> HcclResult {
        RankGraph* rankGraph = nullptr;
        CHK_RET(GetRankGraphFromComm(comm, &rankGraph));
        CHK_RET(rankGraph->GetInstSizeByNetLayer(netLayer, rankNum));
        HCCL_INFO("HcclRankGraphGetRankSizeByLayer success, rankNum [%u]", *rankNum);
        return HCCL_SUCCESS;
    }());
    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
    ret = hcclComm->GetInstSizeByNetLayer(netLayer, rankNum);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[%s] Failed, ret[%d]", __func__, ret);
        return ret;
    }
    HCCL_RUN_INFO("[%s] success, group[%s], rankNum[%u]", __func__, hcclComm->GetIdentifier().c_str(), *rankNum);
    return HCCL_SUCCESS;
}

HcclResult HcclRankGraphGetRanksByLayer(HcclComm comm, uint32_t netLayer, uint32_t** ranks, uint32_t* rankNum)
{
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(rankNum);
    CHK_PTR_NULL(ranks);
    HcclResult ret = HCCL_SUCCESS;
    HCCLV2_FUNC_RUN([&]() -> HcclResult {
        RankGraph* rankGraph = nullptr;
        CHK_RET(GetRankGraphFromComm(comm, &rankGraph));
        CHK_RET(rankGraph->GetInstRanksByNetLayer(netLayer, ranks, rankNum));
        HCCL_INFO("HcclRankGraphGetRanksByLayer success, rankNum [%u]", *rankNum);
        return HCCL_SUCCESS;
    }());
    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
    ret = hcclComm->GetInstRanksByNetLayer(netLayer, ranks, rankNum);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[%s] Failed, ret[%d]", __func__, ret);
        return ret;
    }
    HCCL_RUN_INFO("[%s] success, group[%s], rankNum[%u]", __func__, hcclComm->GetIdentifier().c_str(), *rankNum);
    return HCCL_SUCCESS;
}

HcclResult
HcclRankGraphGetInstSizeListByLayer(HcclComm comm, uint32_t netLayer, uint32_t** instSizeList, uint32_t* listSize)
{
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(instSizeList);
    CHK_PTR_NULL(listSize);
    HcclResult ret = HCCL_SUCCESS;
    HCCLV2_FUNC_RUN([&]() -> HcclResult {
        RankGraph* rankGraph = nullptr;
        CHK_RET(GetRankGraphFromComm(comm, &rankGraph));
        CHK_RET(rankGraph->GetInstSizeListByNetLayer(netLayer, instSizeList, listSize));
        HCCL_INFO("HcclRankGraphGetInstSizeListByLayer success, listSize [%u]", *listSize);
        return HCCL_SUCCESS;
    }());
    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
    ret = hcclComm->GetInstSizeListByNetLayer(netLayer, instSizeList, listSize);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[%s] Failed, ret[%d]", __func__, ret);
        return ret;
    }
    HCCL_RUN_INFO("[%s] success, group[%s], listSize[%u]", __func__, hcclComm->GetIdentifier().c_str(), *listSize);
    return HCCL_SUCCESS;
}

HcclResult
HcclRankGraphGetTopoInstsByLayer(HcclComm comm, uint32_t netLayer, uint32_t** topoInsts, uint32_t* topoInstNum)
{
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(topoInsts);
    CHK_PTR_NULL(topoInstNum);
    HCCLV2_FUNC_RUN([&]() -> HcclResult {
        RankGraph* rankGraph = nullptr;
        CHK_RET(GetRankGraphFromComm(comm, &rankGraph));
        RankGraphV2* rankGraphV2 = static_cast<RankGraphV2*>(rankGraph);
        CHK_RET(rankGraphV2->GetTopoInstsByLayer(netLayer, topoInsts, topoInstNum));
        HCCL_INFO("HcclRankGraphGetTopoInstsByLayer success, topoInstNum [%u]", *topoInstNum);
        return HCCL_SUCCESS;
    }());

    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
    HcclResult ret = hcclComm->GetTopoInstsByLayer(netLayer, topoInsts, topoInstNum);

    return ret;
}

HcclResult HcclRankGraphGetTopoType(HcclComm comm, uint32_t netLayer, uint32_t topoInstId, CommTopo* topoType)
{
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(topoType);
    HCCLV2_FUNC_RUN([&]() -> HcclResult {
        RankGraph* rankGraph = nullptr;
        CHK_RET(GetRankGraphFromComm(comm, &rankGraph));
        RankGraphV2* rankGraphV2 = static_cast<RankGraphV2*>(rankGraph);
        CHK_RET(rankGraphV2->GetTopoType(netLayer, topoInstId, topoType));
        HCCL_INFO("HcclRankGraphGetTopoType success, topoType [%d]", *topoType);
        return HCCL_SUCCESS;
    }());
    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
    HcclResult ret = hcclComm->GetTopoType(netLayer, topoInstId, topoType);
    return ret;
}

HcclResult HcclRankGraphGetRanksByTopoInst(
    HcclComm comm, uint32_t netLayer, uint32_t topoInstId, uint32_t** ranks, uint32_t* rankNum)
{
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(ranks);
    CHK_PTR_NULL(rankNum);
    HCCLV2_FUNC_RUN([&]() -> HcclResult {
        RankGraph* rankGraph = nullptr;
        CHK_RET(GetRankGraphFromComm(comm, &rankGraph));
        RankGraphV2* rankGraphV2 = static_cast<RankGraphV2*>(rankGraph);
        CHK_RET(rankGraphV2->GetRanksByTopoInst(netLayer, topoInstId, ranks, rankNum));
        HCCL_INFO("HcclRankGraphGetRanksByTopoInst success, rankNum [%u]", *rankNum);
        return HCCL_SUCCESS;
    }());
    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
    HcclResult ret = hcclComm->GetRanksByTopoInst(netLayer, topoInstId, ranks, rankNum);
    return ret;
}

HcclResult HcclRankGraphGetEndpointNum(HcclComm comm, uint32_t layer, uint32_t topoInstId, uint32_t* num)
{
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(num);
    HCCLV2_FUNC_RUN([&]() -> HcclResult {
        RankGraph* rankGraph = nullptr;
        CHK_RET(GetRankGraphFromComm(comm, &rankGraph));
        RankGraphV2* rankGraphV2 = static_cast<RankGraphV2*>(rankGraph);
        CHK_RET(rankGraphV2->GetEndpointNum(layer, topoInstId, num));
        HCCL_INFO("HcclRankGraphGetEndpointNum success, num [%u]", *num);
        return HCCL_SUCCESS;
    }());
    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
    HcclResult ret = hcclComm->GetEndpointNum(layer, topoInstId, num);
    return ret;
}

HcclResult HcclRankGraphGetEndpointDesc(
    HcclComm comm, uint32_t layer, uint32_t topoInstId, uint32_t* descNum, EndpointDesc* endpointDesc)
{
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(descNum);
    CHK_PTR_NULL(endpointDesc);
    HCCLV2_FUNC_RUN([&]() -> HcclResult {
        RankGraph* rankGraph = nullptr;
        CHK_RET(GetRankGraphFromComm(comm, &rankGraph));
        RankGraphV2* rankGraphV2 = static_cast<RankGraphV2*>(rankGraph);
        CHK_RET(rankGraphV2->GetEndpointDesc(layer, topoInstId, descNum, endpointDesc));
        HCCL_INFO("HcclRankGraphGetEndpointDesc success");
        return HCCL_SUCCESS;
    }());
    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
    HcclResult ret = hcclComm->GetEndpointDesc(layer, topoInstId, descNum, endpointDesc);

    return ret;
}

HcclResult HcclRankGraphGetEndpointInfo(
    HcclComm comm, uint32_t rankId, const EndpointDesc* endpointDesc, EndpointAttr endpointAttr, uint32_t infoLen,
    void* info)
{
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(endpointDesc);
    CHK_PTR_NULL(info);
    HCCLV2_FUNC_RUN([&]() -> HcclResult {
        RankGraph* rankGraph = nullptr;
        CHK_RET(GetRankGraphFromComm(comm, &rankGraph));
        RankGraphV2* rankGraphV2 = static_cast<RankGraphV2*>(rankGraph);
        CHK_RET(rankGraphV2->GetEndpointInfo(rankId, endpointDesc, endpointAttr, infoLen, info));
        HCCL_INFO("HcclRankGraphGetEndpointInfo success");
        return HCCL_SUCCESS;
    }());
    RankGraph* rankGraph = nullptr;
    CHK_RET(GetRankGraphFromComm(comm, &rankGraph));
    HcclResult ret = rankGraph->GetEndpointInfo(rankId, endpointDesc, endpointAttr, infoLen, info);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[%s] Failed to get endpoint info, ret[%d]", __func__, ret);
        return ret;
    }
    HCCL_INFO("HcclRankGraphGetEndpointInfo success");
    return HCCL_SUCCESS;
}

HcclResult HcclGetHeterogMode(HcclComm comm, HcclHeterogMode* mode)
{
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(mode);
    HCCLV2_FUNC_RUN(HcclGetHeterogModeV2(comm, mode));
    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
    HcclResult ret = hcclComm->GetHeterogMode(mode);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[%s] Failed, ret[%d]", __func__, ret);
        return ret;
    }
    HCCL_RUN_INFO("[%s] success, group[%s], mode[%u]", __func__, hcclComm->GetIdentifier().c_str(), *mode);
    return HCCL_SUCCESS;
}

HcclResult HcclGetRankSize(HcclComm comm, uint32_t* rankSize)
{
    // 入参合法性校验
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(rankSize);
    HCCLV2_FUNC_RUN([&]() -> HcclResult {
        RankGraph* rankGraph = nullptr;
        CHK_RET(GetRankGraphFromComm(comm, &rankGraph));
        CHK_RET(rankGraph->GetRankSize(rankSize));
        return HCCL_SUCCESS;
    }());
    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
    u32 tmpRankSize = INVALID_VALUE_RANKSIZE;
    CHK_RET(hcclComm->GetRankSize(tmpRankSize));
    *rankSize = tmpRankSize;
    /* 关键状态记录 */
    HCCL_INFO("HcclGetRankSize success, rankSizePtr[%p], rankSize[%u]", rankSize, tmpRankSize);
    return HCCL_SUCCESS;
}

HcclResult HcclGetRankId(HcclComm comm, uint32_t* rank)
{
    // 入参合法性校验
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(rank);
    HCCLV2_FUNC_RUN([&]() -> HcclResult {
        RankGraph* rankGraph = nullptr;
        CHK_RET(GetRankGraphFromComm(comm, &rankGraph));
        CHK_RET(rankGraph->GetRankId(rank));
        return HCCL_SUCCESS;
    }());
    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
    u32 tmpRankId = Hccl::DFX_INVALID_RANKID;
    CHK_RET(hcclComm->GetUserRank(tmpRankId));
    *rank = tmpRankId;
    /* 关键状态记录 */
    HCCL_INFO("HcclGetRankId success, rankIdPtr[%p], rankId[%u]", rank, tmpRankId);
    return HCCL_SUCCESS;
}
#endif

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus
HcclResult CommGetNetLayers(HcclComm comm, uint32_t** netLayers, uint32_t* netLayerNum)
{
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(netLayers);
    CHK_PTR_NULL(netLayerNum);
    HCCLV2_FUNC_RUN(HcclGetNetLayersV2(comm, netLayers, netLayerNum));
    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
    HcclResult ret = hcclComm->CommGetNetLayers(netLayers, netLayerNum);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[%s] Failed to GetCommNetLayers ret[%d]", __func__, ret);
        return ret;
    }
    HCCL_RUN_INFO(
        "[%s] success, group[%s], netLayerNum size[%u]", __func__, hcclComm->GetIdentifier().c_str(), *netLayerNum);
    return HCCL_SUCCESS;
}

HcclResult CommGetInstTopoTypeByNetLayer(HcclComm comm, uint32_t netLayer, uint32_t* topoType)
{
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(topoType);
    HCCLV2_FUNC_RUN(HcclGetInstTopoTypeByNetLayerV2(comm, netLayer, topoType));

    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
    HcclResult ret = hcclComm->CommGetInstTopoTypeByNetLayer(netLayer, topoType);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[%s] Failed, ret[%d]", __func__, ret);
        return ret;
    }
    HCCL_RUN_INFO("[%s] success, group[%s], [%d]", __func__, hcclComm->GetIdentifier().c_str(), *topoType);
    return HCCL_SUCCESS;
}

HcclResult CommGetInstSizeByNetLayer(HcclComm comm, uint32_t netLayer, uint32_t* rankNum)
{
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(rankNum);
    HCCLV2_FUNC_RUN(HcclGetInstSizeByNetLayerV2(comm, netLayer, rankNum));

    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
    HcclResult ret = hcclComm->CommGetInstSizeByNetLayer(netLayer, rankNum);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[%s] Failed, ret[%d]", __func__, ret);
        return ret;
    }
    HCCL_RUN_INFO("[%s] success, group[%s], rankNum[%u]", __func__, hcclComm->GetIdentifier().c_str(), *rankNum);
    return HCCL_SUCCESS;
}
#ifdef __cplusplus
}
#endif // __cplusplus
