/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_channel_ctx_mgr_v1.h"

#include <vector>
#include <string>

#include "ccu_res_specs.h"
#include "hcomm_adapter_hccp.h"

namespace hcomm {

constexpr uint32_t CCU_V1_CHANNEL_DEFAULT_JETTY_NUM = 1;

HcclResult CcuChannelCtxMgrV1::Init()
{
    uint32_t strategy = 0; // 获取失败或为0场景，分配将按资源不足操作
    (void)CcuResSpecifications::GetInstance(userDevId_).GetChannelNum(dieId_, strategy);
    channelResInfos_.resize(strategy);
    CHK_RET(jettyCtxMgr_.Init());
    return HcclResult::HCCL_SUCCESS;
}

static HcclResult FindFreeChannelId(std::vector<ChannelResInfo>& channelResInfos, uint32_t& channelId)
{
    // ccu v1每次都分配新的channel与jettyCtx
    // 故直接选择首个可用channel即可
    const uint32_t channelNum = channelResInfos.size();
    for (uint32_t i = 0; i < channelNum; i++) {
        if (!channelResInfos[i].allocated) {
            channelId = i;
            return HcclResult::HCCL_SUCCESS;
        }
    }
    return HcclResult::HCCL_E_UNAVAIL;
}

HcclResult CcuChannelCtxMgrV1::Alloc(const ChannelPara& channelPara, std::vector<ChannelInfo>& channelInfos)
{
    const uint32_t feId = channelPara.feId;
    uint32_t jettyNum = channelPara.jettyNum;
    if (jettyNum == 0) {
        jettyNum = CCU_V1_CHANNEL_DEFAULT_JETTY_NUM;
        HCCL_INFO(
            "[CcuJettyCtxMgrV1][%s] jettyNum is 0, reset to default[%u], "
            "feId[%u], userDevId[%d], dieId[%u].",
            __func__, jettyNum, feId, userDevId_, dieId_);
    }

    std::lock_guard<std::mutex> lock(innerMutex_);
    uint32_t channelId = 0;
    auto ret = FindFreeChannelId(channelResInfos_, channelId);
    CHK_PRT_RET(
        ret == HcclResult::HCCL_E_UNAVAIL,
        HCCL_WARNING(
            "[CcuChannelCtxMgrV1][%s] failed to find free channel, channel strategy[%zu], "
            "left resources are not enough, feId[%u], userDevId[%d], dieId[%u].",
            __func__, channelResInfos_.size(), feId, userDevId_, dieId_),
        ret);
    CHK_RET(ret);

    ChannelInfo channelInfo = {};
    ret = jettyCtxMgr_.Alloc(feId, jettyNum, channelPara.sqSize, channelInfo.jettyInfos);
    CHK_PRT_RET(
        ret == HcclResult::HCCL_E_UNAVAIL,
        HCCL_WARNING(
            "[CcuChannelCtxMgrV1][%s] failed to allocate jetty contexts to channelId[%u], "
            "left resources are not enough, feId[%u], userDevId[%d], dieId[%u].",
            __func__, channelId, feId, userDevId_, dieId_),
        ret);
    CHK_RET(ret);

    channelInfo.channelId = channelId;
    channelInfo.dieId = dieId_;
    channelResInfos_[channelId].feId = feId;
    channelResInfos_[channelId].channelInfo = channelInfo;
    channelResInfos_[channelId].allocated = true;
    DumpChannelResInfo(feId, channelInfo);

    channelInfos.clear(); // ccu v1每次仅分配1个channel，不同channel不复用jettyCtx
    channelInfos.emplace_back(std::move(channelInfo));
    return ret;
}

static ChannelCtxDataV1
BuildChannelCtxDataV1(const ChannelCfg& cfg, const uint32_t feId, const uint8_t dieId, const uint16_t startTaJettyId)
{
    ChannelCtxDataV1 data = {};
    (void)memcpy_s(&data.eidRaw[0], URMA_EID_LEN, &cfg.remoteEid, URMA_EID_LEN);

    data.vtpLow = cfg.tpn & MASK_VTP_LOW;
    data.vtpHigh = ((cfg.tpn & MASK_VTP) >> Hccl::SHIFT_16BITS) & MASK_VTP_HIGH;

    data.srcPfeId = static_cast<uint16_t>(feId);

    data.startJettyIdLow = startTaJettyId & MASK_START_JETTY_ID_LOW;
    data.startJettyIdHigh = (startTaJettyId >> Hccl::SHIFT_4BITS) & MASK_START_JETTY_ID_HIGH;

    // 写入硬件减 1，cfgs的数量一定小于jetty规格数，不会超过uint8_t范围
    uint8_t jettyNum = static_cast<uint8_t>(cfg.jettyCfgs.size()) - 1;
    data.jettyNumLow = jettyNum & MASK_JETTY_NUM_LOW;
    data.jettyNumHigh = (jettyNum >> Hccl::SHIFT_4BITS) & MASK_JETTY_NUM_HIGH;

    data.ioDieId = static_cast<uint16_t>(dieId);

    data.dstTokenIdLow = cfg.memTokenId & MASK_TOKEN_ID_LOW;
    data.dstTokenIdHigh = (cfg.memTokenId >> Hccl::SHIFT_12BITS) & MASK_TOKEN_ID_HIGH;

    data.dstTokenValueLow = cfg.memTokenValue & MASK_TOKEN_VALUE_LOW;
    data.dstTokenValueMiddle = (cfg.memTokenValue >> Hccl::SHIFT_8BITS) & MASK_TOKEN_VALUE_MID;
    data.dstTokenValueHigh = (cfg.memTokenValue >> Hccl::SHIFT_24BITS) & MASK_TOKEN_VALUE_HIGH;

    uint64_t dstVa = (cfg.remoteCcuVa >> REMOTE_CCU_VA_RIGHT_SHIFT_NUM);
    data.dstVaLow = dstVa & MASK_VA_LOW;
    data.dstVaMiddle = (dstVa >> Hccl::SHIFT_8BITS) & MASK_VA_MID;
    data.dstVaHigh = (dstVa >> Hccl::SHIFT_24BITS) & MASK_VA_HIGH;
    data.dstVaHigher = (dstVa >> Hccl::SHIFT_40BITS) & MASK_VA_HIGHER;
    data.dstTokenValueValid = TOKEN_VALUE_VALID;
    return data;
}

static void DumpChannelCtxDataV1(const struct ChannelCtxDataV1& data)
{
    if (IsEidEmpty(data.eidRaw)) {
        return;
    }
    std::string dstEidInfo = "eidRaw: ";
    for (uint32_t i = 0; i < URMA_EID_LEN - 1; i++) {
        dstEidInfo += Hccl::StringFormat("0x%02x, ", data.eidRaw[i]);
    }
    dstEidInfo += Hccl::StringFormat("0x%02x", data.eidRaw[URMA_EID_LEN - 1]);
    HCCL_INFO("%s.", dstEidInfo.c_str());

    HCCL_INFO(
        "vtpLow: 0x%04x, vtpHigh: 0x%04x, srcPfeId: 0x%04x, "
        "startJettyIdLow: 0x%04x, startJettyIdHigh: 0x%04x, "
        "JettyNumLow: 0x%04x, JettyNumHigh: 0x%04x, ioDieId: 0x%04x, ",
        data.vtpLow, data.vtpHigh, data.srcPfeId, data.startJettyIdLow, data.startJettyIdHigh, data.jettyNumLow,
        data.jettyNumHigh, data.ioDieId);

    HCCL_INFO(
        "dstVaLow: 0x%04x, dstVaMiddle: 0x%04x, "
        "dstVaHigh: 0x%04x, dstVaHigher: 0x%04x, dstTokenValueValid: 0x%04x",
        data.dstVaLow, data.dstVaMiddle, data.dstVaHigh, data.dstVaHigher, data.dstTokenValueValid);
}

static HcclResult ConfigChannelCtxDataV1(
    const int32_t userDevId, const uint32_t devPhyId, const uint8_t dieId, const uint32_t channelId,
    const ChannelCtxDataV1& channelCtxData)
{
    CustomChannelInfoIn inBuff{};
    CustomChannelInfoOut outBuff{};

    constexpr uint32_t dataArraySize = 1; // 每次配置1个Channel
    inBuff.op = CcuOpcodeType::CCU_U_OP_SET_CHANNEL;
    inBuff.data.dataInfo.udieIdx = dieId;
    inBuff.data.dataInfo.dataArraySize = dataArraySize;
    inBuff.data.dataInfo.dataLen = sizeof(struct ChannelCtxDataV1) * dataArraySize;
    inBuff.offsetStartIdx = channelId;

    HCCL_INFO(
        "[CcuChannelCtxMgrV1][%s] set data to ccu driver, userDevId[%d] devPhyId[%u], "
        "ioDie[%u], idx[%u], size[%u].",
        __func__, userDevId, devPhyId, dieId, channelId, sizeof(struct ChannelCtxDataV1));
    DumpChannelCtxDataV1(channelCtxData);

    (void)memcpy_s(
        inBuff.data.dataInfo.dataArray, sizeof(struct ChannelCtxDataV1), &channelCtxData,
        sizeof(struct ChannelCtxDataV1));

    auto ret = HccpRaTlvCcuCustomChannel(userDevId, static_cast<void*>(&inBuff), static_cast<void*>(&outBuff));
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[CcuChannelCtxMgrV1][%s] failed to call ccu driver, "
            "userDevId[%d] devPhyId[%u] dieId[%d] op[%s] ret[%d].",
            __func__, userDevId, devPhyId, dieId, "SET_CHANNEL", ret);
        return ret;
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuChannelCtxMgrV1::Config(const ChannelCfg& channelCfg)
{
    std::lock_guard<std::mutex> lock(innerMutex_);
    const uint32_t channelId = channelCfg.channelId;
    if (!CheckIfChannelAllocated(channelId)) {
        return HcclResult::HCCL_E_PARA; // 日志已在判断处处理
    };

    const auto& channelResInfo = channelResInfos_[channelId];
    const uint32_t feId = channelResInfo.feId;
    const std::vector<JettyInfo>& jettyInfos = channelResInfo.channelInfo.jettyInfos;
    auto ret = jettyCtxMgr_.Config(feId, jettyInfos, channelCfg.jettyCfgs);
    if (ret != HcclResult::HCCL_SUCCESS) {
        HCCL_ERROR(
            "[CcuChannelCtxMgrV1][%s] failed to config jetty contexts of channelId[%u], "
            "feId[%u], userDevId[%d], dieId[%u].",
            __func__, channelId, feId, userDevId_, dieId_);
        return ret;
    }
    // 因jettyCtx连续，从起始jettyCtx配置
    const uint16_t startTaJettyId = jettyInfos[0].taJettyId;
    const ChannelCtxDataV1& data = BuildChannelCtxDataV1(channelCfg, feId, dieId_, startTaJettyId);
    CHK_RET(ConfigChannelCtxDataV1(userDevId_, devPhyId_, dieId_, channelId, data));
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuChannelCtxMgrV1::Release(const uint32_t channelId)
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
            "[CcuChannelCtxMgrV1][%s] failed to release jetty contexts "
            "of channelId[%u], feId[%u], userDevId[%d], dieId[%u].",
            __func__, channelId, channelResInfos_[channelId].feId, userDevId_, dieId_),
        ret);
    // 重置并配置Channel表，避免错误复用
    channelResInfos_[channelId] = ChannelResInfo{};
    ChannelCtxDataV1 data = {};
    CHK_RET(ConfigChannelCtxDataV1(userDevId_, devPhyId_, dieId_, channelId, data));
    return HcclResult::HCCL_SUCCESS;
}

}; // namespace hcomm
