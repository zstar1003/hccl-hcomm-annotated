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

// [中文导读] [AllReduce逐行 S349] HcclGetRankSize的接口声明：通信域句柄、域内Rank总数；这些参数属于本函数调用边界。
HcclResult HcclGetRankSize(HcclComm comm, uint32_t* rankSize)
// [中文导读] [AllReduce逐行 S350] 进入HcclGetRankSize函数体：取得域内 Rank 总数，输出的是 Rank 条数而非字节容量。
{
    // 入参合法性校验
    // [中文导读] [AllReduce逐行 S352] 检查`comm`（通信域句柄）不是空对象；宏命中失败条件时立即返回对应指针错误。
    CHK_PTR_NULL(comm);
    // [中文导读] [AllReduce逐行 S353] 检查`rankSize`（域内Rank总数）不是空对象；宏命中失败条件时立即返回对应指针错误。
    CHK_PTR_NULL(rankSize);
    // [中文导读] [AllReduce逐行 S354] 定义逐片处理回调，捕获当前连接/配置上下文；回调参数描述本地与远端同一分片及其首尾位置。
    HCCLV2_FUNC_RUN([&]() -> HcclResult {
        // [中文导读] [AllReduce逐行 S355] 为Host编译时检查设备是否支持V2，支持则直接返回新流程表达式的状态补入`RankGraph* rankGraph = nullptr`；本行是参数/结构化初始化续行。
        RankGraph* rankGraph = nullptr;
        // [中文导读] [AllReduce逐行 S356] 调用GetRankGraphFromComm，使用通信域句柄；返回非成功时由检查宏立即向上传递。
        CHK_RET(GetRankGraphFromComm(comm, &rankGraph));
        // [中文导读] [AllReduce逐行 S357] 取得域Rank总数；返回非成功时由检查宏立即向上传递。
        CHK_RET(rankGraph->GetRankSize(rankSize));
        // [中文导读] [AllReduce逐行 S358] 当前路径返回成功状态；仅说明本函数处理/任务组织成功，完成语义由其具体调用职责决定。
        return HCCL_SUCCESS;
    // [中文导读] [AllReduce逐行 S359] 关闭并立即调用前述lambda；其新域查询结果交由外层异常/返回宏处理。
    }());
    // [中文导读] [AllReduce逐行 S360] 设置域的兼容外层对象为/按`static_cast<hccl::hcclComm*>(comm)`（域的兼容外层对象、通信域句柄）。
    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
    // [中文导读] [AllReduce逐行 S361] 设置临时Rank总数为/按`INVALID_VALUE_RANKSIZE`。
    u32 tmpRankSize = INVALID_VALUE_RANKSIZE;
    // [中文导读] [AllReduce逐行 S362] 取得域Rank总数；返回非成功时由检查宏立即向上传递。
    CHK_RET(hcclComm->GetRankSize(tmpRankSize));
    // [中文导读] [AllReduce逐行 S363] 设置域内Rank总数为/按`tmpRankSize`（临时Rank总数）。
    *rankSize = tmpRankSize;
    /* 关键状态记录 */
    // [中文导读] [AllReduce逐行 S365] 记录HcclGetRankSize的状态/性能诊断，字段包含域内Rank总数、临时Rank总数；日志本身不执行传输。
    HCCL_INFO("HcclGetRankSize success, rankSizePtr[%p], rankSize[%u]", rankSize, tmpRankSize);
    // [中文导读] [AllReduce逐行 S366] 当前路径返回成功状态；仅说明本函数处理/任务组织成功，完成语义由其具体调用职责决定。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S367] 结束HcclGetRankSize函数体；控制流返回外层。
}

// [中文导读] [AllReduce逐行 S369] HcclGetRankId的接口声明：通信域句柄、Rank编号；这些参数属于本函数调用边界。
HcclResult HcclGetRankId(HcclComm comm, uint32_t* rank)
// [中文导读] [AllReduce逐行 S370] 进入HcclGetRankId函数体：取得通信域中的本端 Rank ID，按编译路径与新旧域对象适配。
{
    // 入参合法性校验
    // [中文导读] [AllReduce逐行 S372] 检查`comm`（通信域句柄）不是空对象；宏命中失败条件时立即返回对应指针错误。
    CHK_PTR_NULL(comm);
    // [中文导读] [AllReduce逐行 S373] 检查`rank`（Rank编号）不是空对象；宏命中失败条件时立即返回对应指针错误。
    CHK_PTR_NULL(rank);
    // [中文导读] [AllReduce逐行 S374] 定义逐片处理回调，捕获当前连接/配置上下文；回调参数描述本地与远端同一分片及其首尾位置。
    HCCLV2_FUNC_RUN([&]() -> HcclResult {
        // [中文导读] [AllReduce逐行 S375] 为Host编译时检查设备是否支持V2，支持则直接返回新流程表达式的状态补入`RankGraph* rankGraph = nullptr`；本行是参数/结构化初始化续行。
        RankGraph* rankGraph = nullptr;
        // [中文导读] [AllReduce逐行 S376] 调用GetRankGraphFromComm，使用通信域句柄；返回非成功时由检查宏立即向上传递。
        CHK_RET(GetRankGraphFromComm(comm, &rankGraph));
        // [中文导读] [AllReduce逐行 S377] 调用GetRankId，使用Rank编号；返回非成功时由检查宏立即向上传递。
        CHK_RET(rankGraph->GetRankId(rank));
        // [中文导读] [AllReduce逐行 S378] 当前路径返回成功状态；仅说明本函数处理/任务组织成功，完成语义由其具体调用职责决定。
        return HCCL_SUCCESS;
    // [中文导读] [AllReduce逐行 S379] 关闭并立即调用前述lambda；其新域查询结果交由外层异常/返回宏处理。
    }());
    // [中文导读] [AllReduce逐行 S380] 设置域的兼容外层对象为/按`static_cast<hccl::hcclComm*>(comm)`（域的兼容外层对象、通信域句柄）。
    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
    // [中文导读] [AllReduce逐行 S381] 设置临时Rank编号为/按`Hccl::DFX_INVALID_RANKID`。
    u32 tmpRankId = Hccl::DFX_INVALID_RANKID;
    // [中文导读] [AllReduce逐行 S382] 取得当前域中本端Rank编号；返回非成功时由检查宏立即向上传递。
    CHK_RET(hcclComm->GetUserRank(tmpRankId));
    // [中文导读] [AllReduce逐行 S383] 设置Rank编号为/按`tmpRankId`（临时Rank编号）。
    *rank = tmpRankId;
    /* 关键状态记录 */
    // [中文导读] [AllReduce逐行 S385] 记录HcclGetRankId的状态/性能诊断，字段包含Rank编号、临时Rank编号；日志本身不执行传输。
    HCCL_INFO("HcclGetRankId success, rankIdPtr[%p], rankId[%u]", rank, tmpRankId);
    // [中文导读] [AllReduce逐行 S386] 当前路径返回成功状态；仅说明本函数处理/任务组织成功，完成语义由其具体调用职责决定。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S387] 结束HcclGetRankId函数体；控制流返回外层。
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
