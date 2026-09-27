/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_jetty_ctx_mgr_v2.h"

#include "ccu_pfe_cfg_generator.h"

#include "ccu_wqebb_mgr_v2.h"

namespace hcomm {

CcuJettyCtxMgrV2::CcuJettyCtxMgrV2(const int32_t userDevId, const uint8_t dieId, const uint32_t devPhyId)
    : CcuJettyCtxMgr(userDevId, dieId, devPhyId)
{
    (void)CcuResSpecifications::GetInstance(userDevId).GetChannelJettyMap(dieId, channelJettyMap_);
}

HcclResult CcuJettyCtxMgrV2::Init()
{
    // 获取失败或为0场景，分配将按资源不足操作
    (void)CcuResSpecifications::GetInstance(userDevId_).GetJettyNum(dieId_, jettySpecNum_);
    (void)CcuResSpecifications::GetInstance(userDevId_).GetResourceAddr(dieId_, ccuResBaseVa_);

    wqeBBMgr_.reset(new (std::nothrow) CcuWqeBBMgrV2(userDevId_, dieId_));
    CHK_PTR_NULL(wqeBBMgr_);
    CHK_RET(wqeBBMgr_->Init());
    CHK_RET(pfeMgr_.Init());
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuJettyCtxMgrV2::CheckCtxGroupsByFeId(const uint32_t feId)
{
    // 当前a5与a6待机，所有pfe配置均使用所有jetty，故基于fe生成一份资源即可
    if (ctxGroups_.size() != 0) {
        HCCL_INFO("[CcuJettyCtxMgrV2][%s]ctxGroups_.size[%zu]", __func__, ctxGroups_.size());
        return HcclResult::HCCL_SUCCESS;
    }
    PfeJettyStrategy strategy = {};
    CHK_RET(pfeMgr_.GetPfeStrategy(feId, strategy));

    const uint32_t jettyNum = static_cast<uint32_t>(strategy.size);
    const uint32_t startTaJettyId = strategy.startTaJettyId;
    const uint32_t startJettyCtxId = strategy.startLocalJettyCtxId;
    const uint32_t jettyGroupSize = channelJettyMap_.jettyNum;
    CHK_PRT_RET(
        (jettySpecNum_ < jettyNum || startJettyCtxId > jettySpecNum_ - jettyNum),
        HCCL_ERROR(
            "[CcuJettyCtxMgrV2][%s] failed, allocated jettyCtxId[%u] is invalid, "
            "jettyCtxId should in [0, %u).",
            __func__, startJettyCtxId, jettySpecNum_ - jettyNum),
        HcclResult::HCCL_E_INTERNAL);

    CHK_PRT_RET(
        jettyNum < jettyGroupSize || jettyGroupSize == 0, // fe策略不够分1个jetty组，认为资源不足
        HCCL_WARNING(
            "[CcuJettyCtxMgrV2][%s] failed, jettyNum[%u] of feId[%u] is too small to "
            "allocate a jetty group, groupSize[%u], userDevId[%d], dieId[%u].",
            __func__, jettyNum, feId, jettyGroupSize, userDevId_, dieId_),
        HcclResult::HCCL_E_UNAVAIL);

    // ccu v2 按组的粒度分配，保证jettyNum为不大于分配规格的整除最大数
    const uint32_t groupNum = jettyNum / jettyGroupSize; // 未对齐是仍向下取整
    ctxGroups_.resize(groupNum);
    for (uint32_t i = 0; i < groupNum; i++) {
        ctxGroups_[i].startJettyCtxId = startJettyCtxId + i * jettyGroupSize;
        ctxGroups_[i].startTaJettyId = startTaJettyId + i * jettyGroupSize;
    }
    return HcclResult::HCCL_SUCCESS;
}

static HcclResult FindFreeCtxGroup(const std::vector<JettyCtxGroup>& ctxGroups, uint32_t& freeGroupId)
{
    CHK_PRT_RET(
        ctxGroups.empty(), HCCL_ERROR("[CcuJettyCtxMgrV2][%s] failed, ctxGroups is empty.", __func__),
        HcclResult::HCCL_E_PARA);
    // 任意jettyCtx组均可，选择首个可用的
    const uint32_t groupSize = ctxGroups.size();
    for (uint32_t i = 0; i < groupSize; i++) {
        if (ctxGroups[i].useCnt == 0) {
            freeGroupId = i;
            return HcclResult::HCCL_SUCCESS;
        }
    }

    HCCL_WARNING("[CcuJettyCtxMgrV2][%s] failed, no free ctxGroup left.", __func__);
    return HcclResult::HCCL_E_UNAVAIL;
}

HcclResult CcuJettyCtxMgrV2::Alloc(
    const uint32_t feId, const uint32_t jettyNum, const uint32_t sqSize, std::vector<JettyInfo>& jettyInfos)
{
    // 检查feId对应jettyCtx信息是否初始化
    auto ret = CheckCtxGroupsByFeId(feId);
    CHK_PRT_RET(
        ret != HcclResult::HCCL_SUCCESS,
        HCCL_WARNING(
            "[CcuJettyCtxMgrV2][%s] failed to find jetty contexts by feId[%u], "
            "userDevId[%d], dieId[%u].",
            __func__, feId, userDevId_, dieId_),
        ret);

    uint32_t freeGroupId = 0;
    ret = FindFreeCtxGroup(ctxGroups_, freeGroupId);
    CHK_PRT_RET(
        ret != HcclResult::HCCL_SUCCESS,
        HCCL_WARNING(
            "[CcuJettyCtxMgrV2][%s] failed to find free jetty contexts of feId[%u], "
            "userDevId[%d], dieId[%u].",
            __func__, feId, userDevId_, dieId_),
        ret);
    HCCL_INFO("[CcuJettyCtxMgrV2][%s] freeGroupId[%u].", __func__, freeGroupId);
    const uint32_t jettyCtxStartId = ctxGroups_[freeGroupId].startJettyCtxId;
    const uint32_t taJettyStartId = ctxGroups_[freeGroupId].startTaJettyId;
    constexpr CcuJettyType jettyType_ = CcuJettyType::TA_CACHED_JETTY;
    jettyInfos.resize(jettyNum);

    uint32_t allocSqSize = sqSize;
    if (allocSqSize != CCU_V2_FIXED_SQ_SIZE) {
        HCCL_RUN_WARNING(
            "[CcuJettyCtxMgr][%s] failed, sqSize is not equal to 32, "
            "sqSize is [%u]",
            __func__, allocSqSize);
        allocSqSize = CCU_V2_FIXED_SQ_SIZE;
    }

    ret = TryAllocWqeBBResource(allocSqSize, jettyCtxStartId, taJettyStartId, jettyType_, jettyInfos);
    if (ret != HCCL_SUCCESS) {
        HCCL_RUN_WARNING(
            "[CcuJettyCtxMgrV2][%s] failed to alloc wqebb resource to jetty contexts "
            "of feId[%u], request sq size[%u], userDevId[%d], dieId[%u].",
            __func__, feId, allocSqSize, userDevId_, dieId_);
        CHK_RET(ReleaseWqeBBResource(jettyInfos));
        return ret;
    }

    const uint32_t channelGroupSize = channelJettyMap_.channelNum;
    ctxGroups_[freeGroupId].useCnt = channelGroupSize; // 一次性分配整组channel
    ctxGroups_[freeGroupId].configured = false;
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuJettyCtxMgrV2::Config(
    const uint32_t feId, const std::vector<JettyInfo>& jettyInfos, const std::vector<JettyCfg>& jettyCfgs)
{
    CHK_RET(CheckIfJettyCfgsValid(jettyInfos, jettyCfgs));
    CHK_RET(CheckCtxGroupsByFeId(feId));
    // jettyInfo由channelMgr提供，属于内部数据，分配与配置的校验保证是增序，不会越界
    const uint32_t startJettyCtxId = jettyInfos[0].jettyCtxId;
    // 创建ctxGroup校验已保证ctxGroupMap[feId]非空
    const uint32_t feStartjettyCtxId = ctxGroups_[0].startJettyCtxId;
    const uint32_t groupId = (startJettyCtxId - feStartjettyCtxId) / channelJettyMap_.jettyNum;
    auto& ctxGroup = ctxGroups_[groupId];
    HCCL_INFO(
        "[CcuJettyCtxMgrV2][%s]jetty contexts[start id[%u], num[%u], feStartjettyCtxId[%u], groupId[%u]] "
        "of feId[%u] have not been allocated yet, userDevId[%d], dieId[%u].",
        __func__, startJettyCtxId, channelJettyMap_.jettyNum, feStartjettyCtxId, groupId, feId, userDevId_, dieId_);
    CHK_PRT_RET(
        ctxGroup.useCnt == 0,
        HCCL_ERROR(
            "[CcuJettyCtxMgrV2][%s] failed, jetty contexts[start id[%u], num[%u]] "
            "of feId[%u] have not been allocated yet, userDevId[%d], dieId[%u].",
            __func__, startJettyCtxId, channelJettyMap_.jettyNum, feId, userDevId_, dieId_),
        HcclResult::HCCL_E_PARA);

    if (ctxGroup.configured) {
        HCCL_INFO(
            "[CcuJettyCtxMgrV2][%s] passed, jetty contexts[start id[%u], num[%u]] "
            "of feId[%u] have been configured, userDevId[%d], dieId[%u].",
            __func__, startJettyCtxId, channelJettyMap_.jettyNum, feId, userDevId_, dieId_);
        return HcclResult::HCCL_SUCCESS;
    }

    std::vector<LocalJettyCtxData> jettyCtxData;
    uint32_t jettyNum = jettyInfos.size();
    for (size_t i = 0; i < jettyNum; i++) {
        jettyCtxData.emplace_back(BuildJettyCtxData(dieId_, feId, jettyInfos[i], jettyCfgs[i]));
    }

    CHK_RET(ConfigJettyCtxData(userDevId_, dieId_, devPhyId_, startJettyCtxId, jettyCtxData));
    ctxGroup.configured = true;
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuJettyCtxMgrV2::Release(const uint32_t feId, const std::vector<JettyInfo>& jettyInfos)
{
    if (jettyInfos.empty()) {
        HCCL_INFO(
            "[CcuJettyCtxMgrV2][%s] passed, jettyInfos is empty, no need to release, ",
            "userDevId[%d], dieId[%u], feId[%u].", __func__, userDevId_, dieId_, feId);
        return HcclResult::HCCL_SUCCESS;
    }
    CHK_RET(CheckCtxGroupsByFeId(feId));
    // jettyInfo由channelMgr提供，属于内部数据，分配与配置的校验保证是增序，不会越界
    const uint32_t startJettyCtxId = jettyInfos[0].jettyCtxId;
    // 创建ctxGroup校验已保证ctxGroupMap[feId]非空
    const uint32_t feStartjettyCtxId = ctxGroups_[0].startJettyCtxId;
    const uint32_t groupId = (startJettyCtxId - feStartjettyCtxId) / channelJettyMap_.jettyNum;
    auto& ctxGroup = ctxGroups_[groupId];
    CHK_PRT_RET(
        ctxGroup.useCnt == 0,
        HCCL_ERROR(
            "[CcuJettyCtxMgrV2][%s] failed, jetty contexts[start id[%u], num[%u]] "
            "of feId[%u] have not been allocated yet, userDevId[%d], dieId[%u].",
            __func__, startJettyCtxId, channelJettyMap_.jettyNum, feId, userDevId_, dieId_),
        HcclResult::HCCL_E_PARA);

    if (ctxGroup.useCnt > 1) {
        ctxGroup.useCnt -= 1;
        HCCL_INFO(
            "[CcuJettyCtxMgr][%s] passed, jetty contexts[start id[%u], num[%u]] is "
            "still in use, left use count is %u, userDevId[%d], dieId[%u].",
            __func__, startJettyCtxId, channelJettyMap_.jettyNum, ctxGroup.useCnt, userDevId_, dieId_);
        return HcclResult::HCCL_SUCCESS;
    }
    CHK_RET(ReleaseWqeBBResource(jettyInfos));

    ctxGroup.configured = false;
    ctxGroup.useCnt = 0;
    return HcclResult::HCCL_SUCCESS;
}

}; // namespace hcomm
