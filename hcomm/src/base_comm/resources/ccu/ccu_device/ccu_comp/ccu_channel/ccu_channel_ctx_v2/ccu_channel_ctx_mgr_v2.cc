/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_channel_ctx_mgr_v2.h"

#include "orion_adapter_hccp.h"

#include "ccu_res_specs.h"

#include "env_config.h"
#include "string_util.h"

namespace hcomm {

CcuChannelCtxMgrV2::CcuChannelCtxMgrV2(const int32_t userDevId, const uint8_t dieId, const uint32_t devPhyId)
    : CcuChannelCtxMgr(userDevId, dieId, devPhyId),
      jettyCtxMgr_(userDevId, dieId, devPhyId)
{
    (void)CcuResSpecifications::GetInstance(userDevId).GetChannelJettyMap(dieId, channelJettyMap_);
}

HcclResult CcuChannelCtxMgrV2::Init()
{
    uint32_t strategy = 0; // 获取失败或为0场景，分配将按资源不足操作
    (void)CcuResSpecifications::GetInstance(userDevId_).GetChannelNum(dieId_, strategy);
    channelResInfos_.resize(strategy);
    CHK_RET(jettyCtxMgr_.Init());
    return HcclResult::HCCL_SUCCESS;
}

static uint32_t CheckAndAdjustJettyNum(const ChannelPara& channelPara, const CcuChannelJettyMap& channelJettyMap)
{
    uint32_t jettyNum = channelPara.jettyNum;
    const uint32_t jettyGroupSize = channelJettyMap.jettyNum;
    if (jettyNum != jettyGroupSize) {
        HCCL_INFO(
            "[CcuChannelCtxMgrV2][%s] jetty num[%u] reset to channelJettyMap."
            "jettyNum[%u], feId[%u].",
            __func__, jettyNum, jettyGroupSize, channelPara.feId);
        jettyNum = jettyGroupSize;
    }
    return jettyNum;
}

static HcclResult
GetStartChannelId(const uint32_t jettyCtxStartId, const CcuChannelJettyMap& channelJettyMap, uint32_t& channelId)
{
    // channelJettyMap来自静态定义，认为其不会为0
    const uint32_t channelGroupSize = channelJettyMap.channelNum;
    const uint32_t jettyGroupSize = channelJettyMap.jettyNum;
    const uint32_t jettyGroupId = jettyCtxStartId / jettyGroupSize;
    if (UINT32_MAX / channelGroupSize < jettyGroupId) {
        HCCL_ERROR(
            "[CcuChannelCtxMgrV2][%s] failed, channelId result overflow "
            "UINT32_MAX, jettyStartId[%u].",
            __func__, jettyCtxStartId);
        return HcclResult::HCCL_E_INTERNAL;
    }

    channelId = channelGroupSize * jettyGroupId;
    return HcclResult::HCCL_SUCCESS;
}

static HcclResult CheckChannelRangeAllocatable(
    const uint32_t startChannelId, const uint32_t channelNum, std::vector<ChannelResInfo>& channelResInfos)
{
    const uint32_t endChannelId = startChannelId + channelNum;
    CHK_PRT_RET(
        channelResInfos.size() < endChannelId || startChannelId >= endChannelId,
        HCCL_ERROR(
            "[CcuChannelCtxMgrV2][%s] failed, channel id range[%u, %u) is not expected, "
            "should be less than channelResInfos size[%zu].",
            __func__, startChannelId, endChannelId, channelResInfos.size()),
        HcclResult::HCCL_E_INTERNAL);

    for (uint32_t i = startChannelId; i < endChannelId; i++) {
        if (channelResInfos[i].allocated) {
            HCCL_WARNING(
                "[CcuChannelCtxMgrV2][%s] failed, channel id[%u] is already allocated, "
                "channel group range[%u, %u).",
                __func__, i, startChannelId, endChannelId);
            return HcclResult::HCCL_E_UNAVAIL;
        }
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuChannelCtxMgrV2::Alloc(const ChannelPara& channelPara, std::vector<ChannelInfo>& channelInfos)
{
    const uint32_t feId = channelPara.feId;
    uint32_t jettyNum = CheckAndAdjustJettyNum(channelPara, channelJettyMap_);

    std::lock_guard<std::mutex> lock(innerMutex_);

    std::vector<JettyInfo> jettyInfos;
    // sqsize 每个jetty预留32分配
    auto ret = jettyCtxMgr_.Alloc(feId, jettyNum, channelPara.sqSize, jettyInfos);
    CHK_PRT_RET(
        ret != HcclResult::HCCL_SUCCESS,
        HCCL_WARNING(
            "[CcuChannelCtxMgrV2][%s] failed to allocate jetty contexts of feId[%u], "
            "userDevId[%d], dieId[%u].",
            __func__, feId, userDevId_, dieId_),
        ret);

    const uint32_t channelGroupSize = channelJettyMap_.channelNum;
    // 分配成功保证数量不为0
    const uint32_t jettyCtxStartId = static_cast<uint32_t>(jettyInfos[0].jettyCtxId);
    uint32_t startChannelId = 0;
    CHK_RET(GetStartChannelId(jettyCtxStartId, channelJettyMap_, startChannelId));
    ret = CheckChannelRangeAllocatable(startChannelId, channelGroupSize, channelResInfos_);
    if (ret != HcclResult::HCCL_SUCCESS) {
        HCCL_WARNING(
            "[CcuChannelCtxMgrV2][%s] failed to find free channels, "
            "jettyCtxStartId[%u], jettyNum[%u], need to release temp jetty contexts.",
            __func__, jettyCtxStartId, jettyNum);

        for (uint32_t i = 0; i < channelGroupSize; i++) { // 存在借用计数故需多次释放
            CHK_RET(jettyCtxMgr_.Release(feId, jettyInfos));
        }
        return ret;
    }

    AllocateChannelResources(channelPara, jettyInfos, startChannelId, channelInfos);
    return HcclResult::HCCL_SUCCESS;
}

void CcuChannelCtxMgrV2::AllocateChannelResources(
    const ChannelPara& channelPara, const std::vector<CcuJettyInfo>& jettyInfos, uint32_t startChannelId,
    std::vector<ChannelInfo>& channelInfos)
{
    // ccu v2按配比关系以组的粒度分配channel，同channel组复用jettyCtx
    // 调用者不处理channel组的概念，认为各channel独立
    channelInfos.clear();
    const uint32_t feId = channelPara.feId;
    const uint32_t channelGroupSize = channelJettyMap_.channelNum;
    for (uint32_t i = 0; i < channelGroupSize; i++) {
        uint32_t channelId = i + startChannelId;
        ChannelInfo channelInfo{};
        channelInfo.channelId = channelId;
        channelInfo.dieId = dieId_;
        channelInfo.jettyInfos = jettyInfos; // 拷贝相同的jetty信息

        auto& channelResInfo = channelResInfos_[channelId];
        channelResInfo.feId = feId;
        channelResInfo.channelInfo = channelInfo;
        channelResInfo.allocated = true;
        channelInfos.emplace_back(std::move(channelInfo));
    }

    HCCL_INFO(
        "[CcuChannelCtxMgrV2][%s] allocated channels[%u, %u) successfully, "
        "userDevId[%d], ioDie[%u], channelNum[%u].",
        __func__, startChannelId, startChannelId + channelGroupSize, userDevId_, dieId_, channelGroupSize);
    // 只打印首channel，避免刷屏，分配成功保证数量不为0
    HCCL_INFO("[CcuChannelCtxMgrV2][%s] the start channel: ", __func__);
    DumpChannelResInfo(feId, channelInfos[0]);
}

static ChannelDataV2 BuildChannelDataV2(const ChannelCfg& cfg, const uint8_t dieId)
{
    ChannelDataV2 data{};
    (void)memcpy_s(&data.eidRaw[0], URMA_EID_LEN, &cfg.remoteEid, URMA_EID_LEN);

    data.vtpLow = cfg.tpn & MASK_VTP_LOW;
    data.vtpHigh = ((cfg.tpn & MASK_VTP) >> Hccl::SHIFT_16BITS) & MASK_VTP_HIGH;
    data.ioDieId = static_cast<uint16_t>(dieId);

    return data;
}

static void DumpChannelDataV2(struct ChannelDataV2& tmp)
{
    if (IsEidEmpty(tmp.eidRaw)) {
        return;
    }

    std::string dstEidInfo = "eidRaw: ";
    for (uint32_t i = 0; i < URMA_EID_LEN - 1; i++) {
        dstEidInfo += Hccl::StringFormat("0x%02x, ", tmp.eidRaw[i]);
    }
    dstEidInfo += Hccl::StringFormat("0x%02x", tmp.eidRaw[URMA_EID_LEN - 1]);
    HCCL_INFO("[ChannelDataV2][%s] dstEidInfo is %s ", __func__, dstEidInfo.c_str());
    HCCL_INFO("vtpLow: 0x%04x, vtpHigh: 0x%04x, ioDieId: 0x%04x.", tmp.vtpLow, tmp.vtpHigh, tmp.ioDieId);
}

static HcclResult ConfigChannelCtxDataV2(
    int32_t userDevId, const uint32_t devPhyId, const uint8_t dieId, const uint32_t channelId,
    ChannelDataV2& channelData)
{
    CustomChannelInfoIn inBuff{};
    CustomChannelInfoOut outBuff{};

    constexpr uint32_t dataArraySize = 1; // 每次配置1个Channel
    inBuff.op = CcuOpcodeType::CCU_U_OP_SET_CHANNEL;
    inBuff.data.dataInfo.udieIdx = dieId;
    inBuff.data.dataInfo.dataArraySize = dataArraySize;
    inBuff.data.dataInfo.dataLen = sizeof(struct ChannelDataV2) * dataArraySize;
    inBuff.offsetStartIdx = channelId;

    HCCL_INFO(
        "[CcuChannelCtxMgrV2][%s] config data to ccu driver, devPhyId[%u], "
        "ioDie[%u], idx[%u], size[%u].",
        __func__, devPhyId, dieId, channelId, sizeof(struct ChannelDataV2));
    DumpChannelDataV2(channelData);

    (void)memcpy_s(
        inBuff.data.dataInfo.dataArray, sizeof(struct ChannelDataV2), &channelData, sizeof(struct ChannelDataV2));

    auto ret = HccpRaTlvCcuCustomChannel(userDevId, static_cast<void*>(&inBuff), static_cast<void*>(&outBuff));
    if (ret != 0) {
        HCCL_ERROR(
            "[CcuChannelCtxMgrV2][%s] failed to call ccu driver, "
            "userDevId[%d] devPhyId[%u] dieId[%d] op[%s] ret[%d].",
            __func__, userDevId, devPhyId, dieId, "SET_CHANNEL", ret);
        return ret;
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuChannelCtxMgrV2::Config(const ChannelCfg& channelCfg)
{
    std::lock_guard<std::mutex> lock(innerMutex_);
    uint32_t channelId = channelCfg.channelId;
    if (!CheckIfChannelAllocated(channelId)) {
        return HcclResult::HCCL_E_PARA; // 日志已在判断处处理
    }

    const auto& channelResInfo = channelResInfos_[channelId];
    const uint32_t feId = channelResInfo.feId;
    const std::vector<JettyInfo>& jettyInfos = channelResInfo.channelInfo.jettyInfos;
    auto ret = jettyCtxMgr_.Config(feId, jettyInfos, channelCfg.jettyCfgs);
    CHK_PRT_RET(
        ret != HcclResult::HCCL_SUCCESS,
        HCCL_ERROR(
            "[CcuChannelCtxMgrV2][%s] failed to config jetty contexts of channelId[%u], "
            "feId[%u], userDevId[%d], dieId[%u].",
            __func__, channelId, feId, userDevId_, static_cast<uint32_t>(dieId_)),
        ret);

    ChannelDataV2 data = BuildChannelDataV2(channelCfg, dieId_);
    CHK_RET(ConfigChannelCtxDataV2(userDevId_, devPhyId_, dieId_, channelId, data));
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuChannelCtxMgrV2::Release(const uint32_t channelId)
{
    std::lock_guard<std::mutex> lock(innerMutex_);
    if (!CheckIfChannelAllocated(channelId)) {
        return HcclResult::HCCL_E_PARA; // 日志已在判断处处理
    };

    const auto& channelResInfo = channelResInfos_[channelId];
    auto ret = jettyCtxMgr_.Release(channelResInfo.feId, channelResInfo.channelInfo.jettyInfos);
    CHK_PRT_RET(
        ret != HcclResult::HCCL_SUCCESS,
        HCCL_WARNING(
            "[CcuChannelCtxMgrV2][%s] failed to release jetty contexts "
            "of channelId[%u], feId[%u], userDevId[%d], dieId[%u].",
            __func__, channelId, channelResInfos_[channelId].feId, userDevId_, dieId_),
        ret);

    channelResInfos_[channelId] = ChannelResInfo{};
    // V2 验证阶段未启用动态channel，保持重置Channel配置表，避免错误复用
    ChannelDataV2 data = {};
    CHK_RET(ConfigChannelCtxDataV2(userDevId_, devPhyId_, dieId_, channelId, data));
    return HcclResult::HCCL_SUCCESS;
}

}; // namespace hcomm
