/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_pfe_mgr_legacy.h"

#include "ccu_res_specs_legacy.h"
#include "orion_adapter_hccp.h"
#include "ccu_pfe_cfg_generator.h"
#include "hccp_tlv_hdc_manager.h"

namespace Hccl {

inline PfeCtx BuildPfeCtx(const PfeJettyCtxCfg& cfg)
{
    struct PfeCtx ctx = {};
    ctx.startJettyId = cfg.startTaJettyId;
    ctx.jettyNum = cfg.size - 1; // PFE给CCU分配的Jetty个数，配置硬件时需减1
    ctx.startLocalJettyCtxId = cfg.startJettyCtxId;

    HCCL_INFO(
        "CcuPfeManager[%s]: feId[%u], startJettyId[%u], jettyNum(-1)[%u], "
        "startLocalJettyCtxId[%u], pfe ctx size[%u]",
        __func__, cfg.feId, ctx.startJettyId, ctx.jettyNum, ctx.startLocalJettyCtxId, sizeof(PfeCtx));
    return ctx;
}

inline PfeJettyStrategy BuildStrategy(const PfeJettyCtxCfg& cfg)
{
    struct PfeJettyStrategy pfeJettyStrategy = {};
    pfeJettyStrategy.feId = cfg.feId;
    pfeJettyStrategy.pfeId = cfg.feId;
    pfeJettyStrategy.size = cfg.size;
    pfeJettyStrategy.startTaJettyId = cfg.startTaJettyId;
    pfeJettyStrategy.startLocalJettyCtxId = cfg.startJettyCtxId;
    return pfeJettyStrategy;
}

static void ConfigPfeTable(
    const int32_t devLogicId, const uint32_t devPhyId, const uint8_t dieId, const uint32_t feId,
    const uint32_t pfeReservedNum, const PfeCtx& pfeCtx)
{
    auto tlvHandle = HccpTlvHdcManager::GetInstance().GetTlvHandle(devLogicId);
    CHECK_NULLPTR(
        tlvHandle, StringFormat("[CcuPfeMgr][%s] tlvHandle is nullptr, devLogicId[%d]", __func__, devLogicId));

    struct CustomChannelInfoIn inBuff;
    struct CustomChannelInfoOut outBuff;
    (void)memset_s(inBuff.data.raw, sizeof(inBuff.data.raw), 0, sizeof(inBuff.data.raw));

    if (UNLIKELY(feId > UINT32_MAX - static_cast<uint32_t>(dieId) * pfeReservedNum)) {
        THROW<InvalidParamsException>(
            "[CcuPfeMgr][%s] failed, feId[%u] is greater than expected, "
            "pfeReservedNum[%u], will exceed the range of uint32_t, devPhyId[%u], "
            "dieId[%u].",
            __func__, feId, pfeReservedNum, devPhyId, dieId);
    }

    const uint32_t pfeTableOffset = static_cast<uint32_t>(dieId) * pfeReservedNum + feId;

    inBuff.op = CcuOpcodeType::CCU_U_OP_SET_PFE;
    inBuff.data.dataInfo.udieIdx = static_cast<uint32_t>(dieId);
    inBuff.data.dataInfo.dataArraySize = 1;
    inBuff.data.dataInfo.dataLen = sizeof(struct PfeCtx); // 单个8B
    // die1 使用后半部分pfe表项，故根据pfe预留数量偏移
    inBuff.offsetStartIdx = pfeTableOffset;

    (void)memcpy_s(inBuff.data.dataInfo.dataArray, inBuff.data.dataInfo.dataLen, &pfeCtx, inBuff.data.dataInfo.dataLen);
    HrtRaTlvRequestForCustomChannel(
        tlvHandle, MSG_TYPE_CCU_DISPATCH_CMD, static_cast<void*>(&inBuff), static_cast<void*>(&outBuff));
}

CcuPfeMgr::CcuPfeMgr(const int32_t devLogicId, const uint8_t dieId, const uint32_t devPhyId)
    : devLogicId_(devLogicId),
      dieId_(dieId),
      devPhyId_(devPhyId)
{
    std::vector<PfeJettyCtxCfg> cfgs = CcuPfeCfgGenerator::GetInstance(devLogicId).GetPfeJettyCtxCfg(dieId);
    if (UNLIKELY(cfgs.empty())) { // 此处不中断流程，后续jettyCtx分配时会因无pfe配置报错停止
        HCCL_WARNING(
            "[CcuJettyCtxMgr] config pfe table passed, pfe cfgs size is 0, "
            "devLogicId[%d], dieId[%u].",
            devLogicId, dieId);
        return;
    }

    uint32_t pfeReservedNum = 0;
    (void)CcuResSpecifications::GetInstance(devLogicId).GetPfeReservedNum(dieId, pfeReservedNum);
    if (UNLIKELY(pfeReservedNum == 0)) {
        HCCL_WARNING(
            "[CcuPfeMgr] config pfe table passed, pfe reserved num is 0, "
            "devLogicId[%d], dieId[%u].",
            devLogicId, dieId);
        return;
    }

    for (const auto& cfg : cfgs) {
        // 分配策略保证cfg中feId均为ccu可用设备
        const uint32_t feId = cfg.feId;
        const auto& iter = pfeMap.find(feId);
        if (iter != pfeMap.end()) {
            continue; // 跳过已配置pfe
        }
        pfeMap[feId] = BuildStrategy(cfg);

        const auto& pfeCtx = BuildPfeCtx(cfg);
        ConfigPfeTable(devLogicId, devPhyId, dieId, feId, pfeReservedNum, pfeCtx);
        HCCL_INFO("[CcuPfeMgr] config pfe table end, devLogicId[%d] dieId[%u] feId[%u]", devLogicId, dieId, feId);
    }
}

HcclResult CcuPfeMgr::GetPfeStrategy(uint32_t feId, PfeJettyStrategy& pfeJettyStrategy) const
{
    const auto& iter = pfeMap.find(feId);
    if (iter == pfeMap.end()) {
        HCCL_ERROR(
            "[CcuPfeMgr][%s] failed, feId[%u] is not found, "
            "devLogicId[%d], dieId[%u].",
            __func__, feId, devLogicId_, dieId_);
        return HCCL_E_NOT_FOUND;
    }

    pfeJettyStrategy = iter->second;
    HCCL_INFO(
        "[CcuPfeMgr][%s] find pfe strategy: dieId[%u] feId[%u] pfeId[%u] size[%u] "
        "startTaJettyId[%u] startLocalJettyCtxId[%u]",
        __func__, dieId_, pfeJettyStrategy.feId, pfeJettyStrategy.pfeId, pfeJettyStrategy.size,
        pfeJettyStrategy.startTaJettyId, pfeJettyStrategy.startLocalJettyCtxId);
    return HCCL_SUCCESS;
}

}; // namespace Hccl
