/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_jetty_ctx_mgr_v1_legacy.h"

namespace Hccl {

CcuJettyCtxMgrV1::CcuJettyCtxMgrV1(const int32_t devLogicId, const uint8_t dieId, const uint32_t devPhyId)
    : CcuJettyCtxMgr(devLogicId, dieId, devPhyId)
{}

HcclResult CcuJettyCtxMgrV1::Alloc(
    const uint32_t feId, const uint32_t jettyNum, const uint32_t sqSize, std::vector<JettyInfo>& jettyInfos)
{
    JettyAllocator* allocatorHandle = nullptr;
    auto ret = GetJettyAllocator(feId, allocatorHandle);
    CHK_PRT_RET(
        ret != HcclResult::HCCL_SUCCESS,
        HCCL_WARNING(
            "[CcuJettyCtxMgrV1][%s] failed, failed to get jetty allocator handle, "
            "devLogicId[%d], dieId[%u], feId[%u].",
            __func__, devLogicId, dieId, feId),
        HcclResult::HCCL_E_INTERNAL);

    vector<ResInfo> jettyResInfos; // jettys分配必须要求连续，故返回一个元素
    ret = allocatorHandle->idAllocator->Alloc(jettyNum, true, jettyResInfos, "ResType::Jetty");
    CHK_PRT_RET(
        ret != HcclResult::HCCL_SUCCESS,
        HCCL_WARNING(
            "[CcuJettyCtxMgrV1][%s] failed, allocator failed to allocate, "
            "devLogicId[%d], dieId[%u], feId[%u].",
            __func__, devLogicId, dieId, feId),
        ret);

    const auto& strategy = allocatorHandle->strategy;
    const uint32_t jettyResStartId = jettyResInfos[0].startId; // 资源分配器中的索引号
    const uint32_t jettyCtxStartId = strategy.startLocalJettyCtxId + jettyResStartId;
    const uint32_t taJettyStartId = strategy.startTaJettyId + jettyResStartId;
    CHK_PRT_RET(
        jettyCtxStartId > jettySpecNum - jettyNum,
        HCCL_WARNING(
            "[CcuJettyCtxMgrV1][%s] jetty resource is not enough, allocated "
            "jettyCtxId[%u] should be less than %u, devLogicId[%d], dieId[%u], feId[%u].",
            __func__, jettyCtxStartId, jettySpecNum, devLogicId, dieId, feId),
        HcclResult::HCCL_E_UNAVAIL);

    constexpr CcuJettyType type_ = CcuJettyType::CCUM_CACHED_JETTY;
    jettyInfos.resize(jettyNum);
    ret = TryAllocWqeBBResource(sqSize, jettyCtxStartId, taJettyStartId, type_, jettyInfos);
    if (ret != HcclResult::HCCL_SUCCESS) {
        HCCL_WARNING(
            "[CcuJettyCtxMgrV1][%s] failed, try to release temp resource, "
            "devLogicId[%d], dieId[%u], feId[%u].",
            __func__, devLogicId, dieId, feId);
        CHK_RET(ReleaseWqeBBResource(jettyInfos)); // jettyInfos已分配的wqeBB资源信息
        jettyInfos.clear();                        // 清理wqebb资源信息
        CHK_RET(allocatorHandle->idAllocator->Release(jettyResStartId, jettyNum));
    }
    return ret;
}

HcclResult CcuJettyCtxMgrV1::GetJettyAllocator(uint32_t feId, JettyAllocator*& allocatorHandle)
{
    if (allocator_ == nullptr) {
        HCCL_INFO("[CcuJettyCtxMgrV1][%s] allocator is null, create an allocator", __func__);
        PfeJettyStrategy strategy = {};
        CHK_RET(pfeMgr.GetPfeStrategy(feId, strategy)); // 如果strategy为0，后续按资源不足处理
        TRY_CATCH_RETURN(allocator_ = std::make_unique<JettyAllocator>(strategy));
    }

    allocatorHandle = allocator_.get();
    CHK_PTR_NULL(allocatorHandle);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuJettyCtxMgrV1::Config(
    const uint32_t feId, const std::vector<JettyInfo>& jettyInfos, const std::vector<JettyCfg>& jettyCfgs)
{
    CHK_RET(CheckIfJettyCfgsValid(jettyInfos, jettyCfgs));

    std::vector<LocalJettyCtxData> jettyCtxData;
    const uint32_t jettyNum = jettyInfos.size();
    for (size_t i = 0; i < jettyNum; i++) {
        jettyCtxData.emplace_back(BuildJettyCtxData(dieId, feId, jettyInfos[i], jettyCfgs[i]));
    }

    const uint32_t startJettyCtxId = jettyInfos[0].jettyCtxId;
    TRY_CATCH_RETURN(ConfigJettyCtxData(devLogicId, dieId, startJettyCtxId, jettyCtxData));
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuJettyCtxMgrV1::Release(const uint32_t feId, const std::vector<JettyInfo>& jettyInfos)
{
    if (jettyInfos.empty()) {
        HCCL_INFO(
            "[CcuJettyCtxMgrV1][%s] passed, jettyInfos is empty, no need to release, ",
            "devLogicId[%d], dieId[%u], feId[%u].", __func__, devLogicId, dieId, feId);
        return HcclResult::HCCL_SUCCESS;
    }
    JettyAllocator* allocatorHandle = nullptr;
    CHK_RET(GetJettyAllocator(feId, allocatorHandle));
    CHK_RET(ReleaseWqeBBResource(jettyInfos));

    const uint32_t jettyStartCtxId = static_cast<uint32_t>(jettyInfos[0].jettyCtxId);
    const uint32_t jettyResStartId = jettyStartCtxId - allocatorHandle->strategy.startLocalJettyCtxId;
    CHK_RET(allocatorHandle->idAllocator->Release(jettyResStartId, jettyInfos.size()));
    return HcclResult::HCCL_SUCCESS;
}

}; // namespace Hccl
