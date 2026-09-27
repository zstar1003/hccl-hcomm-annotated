/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_jetty_ctx_mgr_legacy.h"

#include "ccu_res_specs_legacy.h"
#include "orion_adapter_hccp.h"
#include "resources_not_exist_exception.h"
#include "hccp_tlv_hdc_manager.h"

namespace Hccl {

// 对一个数求以2为底的对数，num已保证不为0
inline uint16_t Log2OfPowerOfTwo(uint32_t num)
{
    uint16_t log2 = 0;
    while (num > 1) {
        num >>= 1;
        log2++;
    }
    return log2;
}

union DoorbellAddr {
    uint64_t dbAddr;
    uint16_t dbAddr16[4];
};

LocalJettyCtxData
BuildJettyCtxData(const uint8_t dieId, const uint32_t pfeId, const JettyInfo& jettyInfo, const JettyCfg& jettyCfg)
{
    LocalJettyCtxData data{};

    DoorbellAddr dbAddr;
    dbAddr.dbAddr = jettyCfg.dbVa;
    data.doorbellAddr[0] = dbAddr.dbAddr16[0];
    data.doorbellAddr[1] = dbAddr.dbAddr16[1];
    data.doorbellAddr[2] = dbAddr.dbAddr16[2]; // 2: doorbell 地址访问
    data.doorbellAddr[3] = dbAddr.dbAddr16[3]; // 3: doorbell 地址访问

    data.pfeIdx = static_cast<uint8_t>(pfeId);
    data.ioDieId = dieId;

    data.doorbellAddrType = DB_ADDR_TYPE;
    data.tokenValueIsValid = TOKEN_VALUE_IS_VALIDE;

    data.tokenIdLow = jettyCfg.dbTokenId & MASK_TK_ID_LOW;
    data.tokenIdHigh = (jettyCfg.dbTokenId >> SHIFT_8BITS) & MASK_TK_ID_HIGH; // tokenId右移8位

    data.tokenValueLow = jettyCfg.dbTokenValue & MASK_TK_VALUE_LOW;
    data.tokenValueMiddle = (jettyCfg.dbTokenValue >> SHIFT_4BITS) & MASK_TK_VALUE_MID; // tokenValue右移4位
    data.tokenValueHigh = (jettyCfg.dbTokenValue >> SHIFT_20BITS) & MASK_TK_VALUE_HIGH; // tokenValue右移20位

    const uint16_t wqeBBShift = Log2OfPowerOfTwo(jettyInfo.sqDepth * CCU_WQE_NUM_PER_SQE);
    data.sqeBasicBlockLeftShifts = wqeBBShift;

    const uint16_t wqeBBIdx = jettyInfo.wqeBBStartId;
    data.startWqeBasicBlockIdxLow = wqeBBIdx & MASK_WQEBB_IDX_LOW;
    data.startWqeBasicBlockIdxHigh = (wqeBBIdx >> SHIFT_4BITS) & MASK_WQEBB_IDX_HIGH; // 右移4位

    data.pi = CCU_HARDWARE_DEFAULT_VALUE;
    data.ci = CCU_HARDWARE_DEFAULT_VALUE;
    data.maxCi = CCU_HARDWARE_DEFAULT_VALUE;
    data.oooCqeCnt = CCU_HARDWARE_DEFAULT_VALUE;
    data.doorbellSendState = CCU_HARDWARE_DEFAULT_VALUE;

    return data;
}

void DumpJettyCtxData(const LocalJettyCtxData& tmp)
{
    HCCL_RUN_INFO(
        "doorbellAddr: [3]0x%04x, [2]0x%04x, [1]0x%04x, [0]0x%04x",
        tmp.doorbellAddr[3], // 3: doorbell 地址访问
        tmp.doorbellAddr[2], // 2: doorbell 地址访问
        tmp.doorbellAddr[1], tmp.doorbellAddr[0]);

    // 安全问题：禁止打印token相关信息
    HCCL_RUN_INFO(
        "pfeIdx: 0x%04x, ioDieId: 0x%04x, doorbellAddrType: 0x%04x, "
        "tokenValueIsValid: 0x%04x",
        tmp.pfeIdx, tmp.ioDieId, tmp.doorbellAddrType, tmp.tokenValueIsValid);

    HCCL_RUN_INFO(
        "sqeBasicBlockLeftShifts: 0x%04x, pi: 0x%04x, ci: 0x%04x, "
        "maxCi: 0x%04x, oooCqeCnt: 0x%04x, startWqeBasicBlockIdxLow: 0x%04x, "
        "startWqeBasicBlockIdxHigh: 0x%04x, doorbellSendState: 0x%04x",
        tmp.sqeBasicBlockLeftShifts, tmp.pi, tmp.ci, tmp.maxCi, tmp.oooCqeCnt, tmp.startWqeBasicBlockIdxLow,
        tmp.startWqeBasicBlockIdxHigh, tmp.doorbellSendState);
}

void ConfigJettyCtxData(
    const int32_t devLogicId, const uint8_t dieId, const uint16_t startJettyCtxId,
    std::vector<LocalJettyCtxData>& jettyCtxData)
{
    const uint32_t jettyNum = jettyCtxData.size(); // 分配与配置前校验已保证不为0
    auto tlvHandle = HccpTlvHdcManager::GetInstance().GetTlvHandle(devLogicId);
    CHECK_NULLPTR(
        tlvHandle, StringFormat("[CcuJettyCtxMgr][%s] tlvHandle is nullptr, devLogicId[%d]", __func__, devLogicId));

    struct CustomChannelInfoIn inBuff;
    struct CustomChannelInfoOut outBuff;

    inBuff.op = CcuOpcodeType::CCU_U_OP_SET_JETTY_CTX;
    (void)memset_s(inBuff.data.raw, sizeof(inBuff.data.raw), 0, sizeof(inBuff.data.raw));

    inBuff.data.dataInfo.udieIdx = dieId;
    inBuff.data.dataInfo.dataArraySize = jettyNum;

    // 设置数据长度，目前设备管理最多使用5个JettyCtx，需要长度上限为 32 * 5 = 160B
    inBuff.data.dataInfo.dataLen = sizeof(struct LocalJettyCtxData) * inBuff.data.dataInfo.dataArraySize;
    inBuff.offsetStartIdx = startJettyCtxId; // 设置起始Jetty上下文ID，注意应从0开始，非TaJettyId

    HCCL_INFO(
        "[CcuJettyCtxMgr][%s] iodie[%u], startJettyCtxId[%u], jettyCtxData.size[%u]", __func__, dieId, startJettyCtxId,
        jettyNum);

    for (size_t i = 0; i < jettyNum; i++) {
        DumpJettyCtxData(jettyCtxData[i]);

        (void)memcpy_s(
            &inBuff.data.dataInfo.dataArray[i], sizeof(struct LocalJettyCtxData), &jettyCtxData[i],
            sizeof(struct LocalJettyCtxData));
    }

    HrtRaTlvRequestForCustomChannel(
        tlvHandle, MSG_TYPE_CCU_DISPATCH_CMD, static_cast<void*>(&inBuff), static_cast<void*>(&outBuff));
}

CcuJettyCtxMgr::CcuJettyCtxMgr(const int32_t devLogicId, const uint8_t dieId, const uint32_t devPhyId)
    : devLogicId(devLogicId),
      dieId(dieId),
      devPhyId(devPhyId),
      wqeBBMgr(devLogicId, dieId)
{
    // 获取失败或为0场景，分配将按资源不足操作
    (void)CcuResSpecifications::GetInstance(devLogicId).GetJettyNum(dieId, jettySpecNum);
    (void)CcuResSpecifications::GetInstance(devLogicId).GetResourceAddr(dieId, ccuResBaseVa);
    if (UNLIKELY(ccuResBaseVa == 0)) { // 直接抛异常终止，避免访问非法地址
        THROW<ResourcesNotExistException>(
            "[CcuJettyCtxMgr] init failed, ccu resource base addr is 0, "
            "devLogicId[%d] dieId[%u].",
            devLogicId, dieId);
    }
    pfeMgr = CcuPfeMgr(devLogicId, dieId, devPhyId);
}

static HcclResult GetSqeBuffVa(const uint64_t ccuResBaseVa, const uint32_t jettyCtxId, uint64_t& sqeBuffVa)
{
    sqeBuffVa = 0;
    if (UINT32_MAX / CCU_SQ_BUFFER_SIZE < jettyCtxId) {
        HCCL_ERROR(
            "[CcuJettyCtxMgr][%s] jetty context id[%u] is greater "
            "than expected, CCU_SQ_BUFFER_SIZE[%u], their product will exceed the "
            "range of uint32_t.",
            __func__, jettyCtxId, CCU_SQ_BUFFER_SIZE);
        return HcclResult::HCCL_E_INTERNAL;
    }
    const uint64_t jettyCtxOffset = static_cast<uint64_t>(jettyCtxId) * CCU_SQ_BUFFER_SIZE;

    if (UINT64_MAX - CCU_V1_WQE_BASIC_BLOCK_OFFSET - jettyCtxOffset < ccuResBaseVa) {
        HCCL_ERROR(
            "[CcuJettyCtxMgr][%s] ccu resource space base va[%llu] "
            "is greater than expected, jettyCtxId[%u], the sqe buff va exceed the "
            "range of uint64_t.",
            __func__, ccuResBaseVa, jettyCtxId);
        return HcclResult::HCCL_E_INTERNAL;
    }

    // 内部分配保证jettyCtxId 小于 jettyCtx规格数量，地址不应越界
    sqeBuffVa = ccuResBaseVa + CCU_V1_WQE_BASIC_BLOCK_OFFSET + jettyCtxOffset;
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuJettyCtxMgr::TryAllocWqeBBResource(
    const uint32_t sqSize, const uint32_t jettyCtxStartId, const uint32_t taJettyStartId, const CcuJettyType jettyType,
    std::vector<JettyInfo>& jettyInfos)
{
    const uint32_t jettyNum = jettyInfos.size();
    if (jettyNum == 0) {
        HCCL_WARNING(
            "[CcuJettyCtxMgr][%s] failed, jettyInfos size is 0, "
            "devLogicId[%d], dieId[%u].",
            __func__, devLogicId, dieId);
        return HcclResult::HCCL_E_PARA;
    }

    for (uint32_t i = 0; i < jettyNum; i++) {
        ResInfo wqeBBInfo(0, 0);
        HcclResult ret = wqeBBMgr.Alloc(sqSize, wqeBBInfo);
        if (ret != HcclResult::HCCL_SUCCESS) {
            HCCL_WARNING(
                "[CcuJettyCtxMgr][%s] failed to alloc wqe basic block resource, "
                "devLogicId[%d], dieId[%u].",
                __func__, devLogicId, dieId);
            return ret;
        }

        auto& jettyInfo = jettyInfos[i];
        jettyInfo.jettyType = jettyType;
        jettyInfo.jettyCtxId = static_cast<uint16_t>(jettyCtxStartId + i);
        jettyInfo.taJettyId = static_cast<uint16_t>(taJettyStartId + i);

        const uint32_t wqeBBReqNum = wqeBBInfo.num;
        jettyInfo.sqDepth = wqeBBReqNum / CCU_WQE_NUM_PER_SQE;
        jettyInfo.wqeBBStartId = wqeBBInfo.startId;
        if (jettyType == CcuJettyType::CCUM_CACHED_JETTY) {
            jettyInfo.sqBufSize = wqeBBReqNum * CCU_ONE_WQE_SIZE;
            CHK_RET(GetSqeBuffVa(
                ccuResBaseVa, static_cast<uint32_t>(jettyInfo.jettyCtxId),
                jettyInfo.sqBufVa)); // 检查溢出，分配成功的wqeBB资源已经记录
        }
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuJettyCtxMgr::ReleaseWqeBBResource(const std::vector<JettyInfo>& jettyInfos)
{
    for (const auto& jettyInfo : jettyInfos) {
        if (jettyInfo.sqDepth == 0) {
            continue; // 该jetty未分配完成，跳过wqeBB资源释放
        }

        uint32_t wqeBBIdx = static_cast<uint32_t>(jettyInfo.wqeBBStartId);
        // jettyInfo 为内部数据，分配保证不会溢出
        uint32_t wqeBBNum = jettyInfo.sqDepth * CCU_WQE_NUM_PER_SQE;
        const auto resInfo = ResInfo(wqeBBIdx, wqeBBNum);
        CHK_RET(wqeBBMgr.Release(resInfo));
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuJettyCtxMgr::CheckIfJettyCfgsValid(
    const std::vector<JettyInfo>& jettyInfos, const std::vector<JettyCfg>& jettyCfgs) const
{
    const uint32_t jettyNum = jettyInfos.size();
    const uint32_t jettyCfgNum = jettyCfgs.size();
    CHK_PRT_RET(
        jettyCfgNum != jettyNum,
        HCCL_ERROR(
            "[CcuJettyCtxMgr][%s] failed, jettyCfgs size[%u] is not expected, "
            "which should be equal to jettyInfo size[%u], devLogicId[%d], dieId[%u].",
            __func__, jettyCfgNum, jettyNum, devLogicId, dieId),
        HcclResult::HCCL_E_PARA);

    for (uint32_t i = 0; i < jettyNum; i++) {
        if (jettyInfos[i].jettyCtxId != jettyCfgs[i].jettyCtxId) {
            HCCL_ERROR(
                "[CcuJettyCtxMgr][%s] failed, jettyCtxId of jettyInfo[%u] and "
                "jettyCfg[%u] are not same, devLogicId[%d], dieId[%u].",
                __func__, jettyInfos[i].jettyCtxId, jettyCfgs[i].jettyCtxId, devLogicId, dieId);
            return HcclResult::HCCL_E_PARA;
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

}; // namespace Hccl
