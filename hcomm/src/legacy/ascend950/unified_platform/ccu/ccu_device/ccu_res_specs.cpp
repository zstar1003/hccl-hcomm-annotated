/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_res_specs_legacy.h"

#include "hccl_common_v2.h"
#include "orion_adapter_rts.h"
#include "orion_adapter_hccp.h"
#include "ccu_device_manager.h"
#include "hccp_tlv_hdc_manager.h"

namespace Hccl {

CcuResSpecifications& CcuResSpecifications::GetInstance(const int32_t deviceLogicId)
{
    static CcuResSpecifications ccuResSpecifications[MAX_MODULE_DEVICE_NUM + 1];
    if (deviceLogicId < 0 || static_cast<uint32_t>(deviceLogicId) > MAX_MODULE_DEVICE_NUM) {
        THROW<InvalidParamsException>(StringFormat(
            "[CcuResSpecifications][GetInstance] Failed to get instance. "
            "devLogicId should be less than %u.",
            MAX_MODULE_DEVICE_NUM));
    }

    ccuResSpecifications[deviceLogicId].devLogicId = deviceLogicId;

    return ccuResSpecifications[deviceLogicId];
}

void CcuResSpecifications::Init()
{
    if (ifInit) {
        return;
    }

    if (Init_() != HcclResult::HCCL_SUCCESS) {
        devPhyId = MAX_MODULE_DEVICE_NUM;
        ccuVersion = CcuVersion::CCU_INVALID;
        for (uint32_t i = 0; i < MAX_CCU_IODIE_NUM; i++) {
            dieEnableFlags[i] = false;
            resSpecs[i] = CcuResSpecInfo{};
        }
    }

    ifInit = true;
}

void CcuResSpecifications::Deinit()
{
    for (uint32_t i = 0; i < MAX_CCU_IODIE_NUM; i++) {
        dieEnableFlags[i] = false;
        resSpecs[i] = CcuResSpecInfo{};
    }

    ifInit = false;
}

static CcuVersion CheckCcuVersion()
{
    return CcuVersion::CCU_V1; // CCU驱动未更新前临时使用
}

static bool CheckDieEnable(const int32_t devLogicId, const uint8_t dieId)
{
    auto tlvHandle = HccpTlvHdcManager::GetInstance().GetTlvHandle(devLogicId);
    CHECK_NULLPTR(tlvHandle, StringFormat("[%s] tlvHandle is nullptr, devLogicId[%d]", __func__, devLogicId));
    struct CustomChannelInfoIn inBuff;
    struct CustomChannelInfoOut outBuff;
    inBuff.op = CcuOpcodeType::CCU_U_OP_GET_DIE_WORKING;
    inBuff.offsetStartIdx = 0;
    inBuff.data.dataInfo.udieIdx = dieId;

    HrtRaTlvRequestForCustomChannel(
        tlvHandle, MSG_TYPE_CCU_DISPATCH_CMD, static_cast<void*>(&inBuff), static_cast<void*>(&outBuff));

    const uint32_t enableFlag = outBuff.data.dataInfo.dataArray[0].dieinfo.enableFlag;
    return enableFlag == CCU_ENABLE_FLAG;
}

static CcuBaseInfoData ParseOutBuffToBaseInfoData(const CustomChannelInfoOut& outBuff)
{
    CcuBaseInfoData baseInfoData{};
    baseInfoData.resourceAddr = outBuff.data.dataInfo.dataArray[0].baseinfo.resourceAddr;
    baseInfoData.missionKey = outBuff.data.dataInfo.dataArray[0].baseinfo.missionKey;
    baseInfoData.msId = outBuff.data.dataInfo.dataArray[0].baseinfo.msId;
    baseInfoData.caps.cap0 = outBuff.data.dataInfo.dataArray[0].baseinfo.caps.cap0;
    baseInfoData.caps.cap1 = outBuff.data.dataInfo.dataArray[0].baseinfo.caps.cap1;
    baseInfoData.caps.cap2 = outBuff.data.dataInfo.dataArray[0].baseinfo.caps.cap2;
    baseInfoData.caps.cap3 = outBuff.data.dataInfo.dataArray[0].baseinfo.caps.cap3;
    baseInfoData.caps.cap4 = outBuff.data.dataInfo.dataArray[0].baseinfo.caps.cap4;
    return baseInfoData;
}

static CcuResSpecInfo ParseOutBuffToResSpecInfo(const CcuVersion ccuVersion, const CustomChannelInfoOut& outBuff)
{
    if (ccuVersion != CcuVersion::CCU_V1) {
        HCCL_WARNING(
            "[CcuResSpecifications][%s] failed to parse out buff, ccu driver "
            "version[%s] is not expected.",
            __func__, ccuVersion.Describe().c_str());
        return {};
    }

    const auto& baseInfoData = ParseOutBuffToBaseInfoData(outBuff);

    CcuResSpecInfo ccuResSpecInfo{};
    ccuResSpecInfo.msId = baseInfoData.msId;
    ccuResSpecInfo.resourceAddr = baseInfoData.resourceAddr;
    ccuResSpecInfo.missionKey = baseInfoData.missionKey;

    ccuResSpecInfo.instructionNum = (baseInfoData.caps.cap0 & 0x0000FFFF) + 1;
    ccuResSpecInfo.xnNum = ((baseInfoData.caps.cap1 >> MOVE_16_BITS) & 0x0000FFFF) + 1;
    ccuResSpecInfo.msNum = ((baseInfoData.caps.cap2 >> MOVE_16_BITS) & 0x0000FFFF) + 1;
    ccuResSpecInfo.ckeNum = (baseInfoData.caps.cap2 & 0x0000FFFF) + 1;
    ccuResSpecInfo.jettyNum = ((baseInfoData.caps.cap3 >> MOVE_16_BITS) & 0x0000FFFF) + 1;
    ccuResSpecInfo.channelNum = (baseInfoData.caps.cap3 & 0x0000FFFF) + 1;
    ccuResSpecInfo.pfeNum = (baseInfoData.caps.cap4 & 0x000000FF) + 1;

    ccuResSpecInfo.missionNum = ((baseInfoData.caps.cap0 >> MOVE_16_BITS) & 0x000000FF) + 1;
    ccuResSpecInfo.loopEngineNum = ((baseInfoData.caps.cap0 >> MOVE_24_BITS) & 0x000000FF) + 1;
    ccuResSpecInfo.gsaNum = (baseInfoData.caps.cap1 & 0x0000FFFF) + 1;
    return ccuResSpecInfo;
}

static CcuResSpecInfo CheckResSpecifications(const int32_t devLogicId, const uint8_t dieId, const CcuVersion ccuVersion)
{
    auto tlvHandle = HccpTlvHdcManager::GetInstance().GetTlvHandle(devLogicId);
    CHECK_NULLPTR(tlvHandle, StringFormat("[%s] tlvHandle is nullptr, devLogicId[%d]", __func__, devLogicId));
    struct CustomChannelInfoIn inBuff;
    struct CustomChannelInfoOut outBuff;
    inBuff.op = CcuOpcodeType::CCU_U_OP_GET_BASIC_INFO;
    inBuff.offsetStartIdx = 0;
    inBuff.data.dataInfo.udieIdx = dieId;

    HrtRaTlvRequestForCustomChannel(
        tlvHandle, MSG_TYPE_CCU_DISPATCH_CMD, static_cast<void*>(&inBuff), static_cast<void*>(&outBuff));
    return ParseOutBuffToResSpecInfo(ccuVersion, outBuff);
}

HcclResult CcuResSpecifications::Init_()
{
    TRY_CATCH_RETURN(devPhyId = HrtGetDevicePhyIdByUserDevId(devLogicId); ccuVersion = CheckCcuVersion();
                     auto tlvHandle = HccpTlvHdcManager::GetInstance().GetTlvHandle(devLogicId);
                     auto memTypeBitmap = GetCombinedMemTypeBitmap(); auto count = GetMemTypeVector().size();
                     for (uint8_t dieId = 0; dieId < MAX_CCU_IODIE_NUM; dieId++) {
                         dieEnableFlags[dieId] = CheckDieEnable(devLogicId, dieId);
                         if (!dieEnableFlags[dieId]) {
                             resSpecs[dieId] = CcuResSpecInfo{};
                             continue;
                         }
                         resSpecs[dieId] = CheckResSpecifications(devLogicId, dieId, ccuVersion);
                         HrtGetCcuMemInfo(tlvHandle, dieId, memTypeBitmap, resSpecs[dieId].memInfoList.data(), count);
                     } HcclMainboardId hcclMainboardId;
                     CHK_RET(HrtGetMainboardId(devLogicId, hcclMainboardId));
                     isAX
                     = (hcclMainboardId == HcclMainboardId::MAINBOARD_A_X_SERVER
                        || hcclMainboardId == HcclMainboardId::MAINBOARD_PCIE_STD);
                     HCCL_INFO(
                         "[CcuResSpecifications]HrtGetMainboardId devLogicId[%d] hcclMainboardId[%s] isAX[%d].",
                         devLogicId, hcclMainboardId.Describe().c_str(), static_cast<int>(isAX)););

    return HcclResult::HCCL_SUCCESS;
}

CcuVersion CcuResSpecifications::GetCcuVersion() const { return ccuVersion; }

bool CcuResSpecifications::GetAXFlag() const { return isAX; }

HcclResult CcuResSpecifications::GetDieEnableFlag(const uint8_t dieId, bool& dieEnableFlag) const
{
    // 只校验dieId合法性，不校验die是否使能
    CHK_RET(CheckDieValid(__func__, devLogicId, dieId, {true, true}));
    dieEnableFlag = dieEnableFlags[dieId];
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuResSpecifications::GetCcuMemInfoList(const uint8_t dieId, struct CcuMemInfo* memInfoList, uint32_t& count)
{
    CHK_RET(CheckDieValid(__func__, devLogicId, dieId, dieEnableFlags));
    count = static_cast<uint32_t>(GetMemTypeVector().size());
    // 使用 std::copy 将 std::array 的内容拷贝到 C 风格指针数组
    std::copy_n(resSpecs[dieId].memInfoList.begin(), count, memInfoList);

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuResSpecifications::GetResourceAddr(const uint8_t dieId, uint64_t& resourceAddr) const
{
    CHK_RET(CheckDieValid(__func__, devLogicId, dieId, dieEnableFlags));
    resourceAddr = resSpecs[dieId].resourceAddr;
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuResSpecifications::GetXnBaseAddr(const uint8_t dieId, uint64_t& xnBaseAddr) const
{
    CHK_RET(CheckDieValid(__func__, devLogicId, dieId, dieEnableFlags));

    const uint64_t ccuResAddr = resSpecs[dieId].resourceAddr;
    if (ccuResAddr == 0) {
        HCCL_WARNING(
            "[CcuResSpecifications][%s] failed, CCU resource base address is 0, "
            "devLogicId[%d] dieId[%u].",
            __func__, devLogicId, dieId);
        return HcclResult::HCCL_E_INTERNAL;
    }

    // xn位于ins与gsa之后，xn偏移 = CCUM偏移 + 指令空间大小 + GSA大小，常量计算不会溢出
    constexpr uint64_t instrRevserveSize = CCU_RESOURCE_INS_RESERVE_SIZE;
    constexpr uint64_t gsaReserveSize = CCU_V1_RESOURCE_GSA_RESERVE_SIZE;
    constexpr uint64_t ccum_offset = CCU_V1_CCUM_OFFSET;
    constexpr uint32_t ccuXnOffset = ccum_offset + instrRevserveSize + gsaReserveSize;
    if (ccuResAddr > UINT64_MAX - ccuXnOffset) {
        HCCL_ERROR(
            "[CcuResSpecifications][%s] failed, CCU resource base address[%llu] is "
            "greater than expected, ccu xn offset[%llu], their sum will exceed the range "
            "of uint64_t.",
            __func__, ccuResAddr, ccuXnOffset);
    }

    xnBaseAddr = ccuResAddr + ccuXnOffset;
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuResSpecifications::GetMsId(const uint8_t dieId, uint32_t& msId) const
{
    CHK_RET(CheckDieValid(__func__, devLogicId, dieId, dieEnableFlags));
    msId = resSpecs[dieId].msId;
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuResSpecifications::GetMissionKey(const uint8_t dieId, uint32_t& missionKey) const
{
    CHK_RET(CheckDieValid(__func__, devLogicId, dieId, dieEnableFlags));
    missionKey = resSpecs[dieId].missionKey;
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuResSpecifications::GetInstructionNum(const uint8_t dieId, uint32_t& instrNum) const
{
    CHK_RET(CheckDieValid(__func__, devLogicId, dieId, dieEnableFlags));
    instrNum = resSpecs[dieId].instructionNum;
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuResSpecifications::GetMissionNum(const uint8_t dieId, uint32_t& missionNum) const
{
    CHK_RET(CheckDieValid(__func__, devLogicId, dieId, dieEnableFlags));
    missionNum = resSpecs[dieId].missionNum;
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuResSpecifications::GetLoopEngineNum(const uint8_t dieId, uint32_t& loopNum) const
{
    CHK_RET(CheckDieValid(__func__, devLogicId, dieId, dieEnableFlags));
    loopNum = resSpecs[dieId].loopEngineNum;
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuResSpecifications::GetGsaNum(const uint8_t dieId, uint32_t& gsaNum) const
{
    CHK_RET(CheckDieValid(__func__, devLogicId, dieId, dieEnableFlags));
    gsaNum = resSpecs[dieId].gsaNum;
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuResSpecifications::GetXnNum(const uint8_t dieId, uint32_t& xnNum) const
{
    CHK_RET(CheckDieValid(__func__, devLogicId, dieId, dieEnableFlags));
    xnNum = resSpecs[dieId].xnNum;
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuResSpecifications::GetCkeNum(const uint8_t dieId, uint32_t& ckeNum) const
{
    CHK_RET(CheckDieValid(__func__, devLogicId, dieId, dieEnableFlags));
    ckeNum = resSpecs[dieId].ckeNum;
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuResSpecifications::GetMsNum(const uint8_t dieId, uint32_t& msNum) const
{
    CHK_RET(CheckDieValid(__func__, devLogicId, dieId, dieEnableFlags));
    msNum = resSpecs[dieId].msNum;
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuResSpecifications::GetChannelNum(const uint8_t dieId, uint32_t& channelNum) const
{
    CHK_RET(CheckDieValid(__func__, devLogicId, dieId, dieEnableFlags));
    channelNum = resSpecs[dieId].channelNum;
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuResSpecifications::GetJettyNum(const uint8_t dieId, uint32_t& jettyNum) const
{
    CHK_RET(CheckDieValid(__func__, devLogicId, dieId, dieEnableFlags));
    jettyNum = resSpecs[dieId].jettyNum;
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuResSpecifications::GetPfeReservedNum(const uint8_t dieId, uint32_t& pfeNum) const
{
    CHK_RET(CheckDieValid(__func__, devLogicId, dieId, dieEnableFlags));
    pfeNum = CCU_V1_PER_DIE_PFE_RESERVED_NUM;
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuResSpecifications::GetPfeNum(const uint8_t dieId, uint32_t& pfeNum) const
{
    CHK_RET(CheckDieValid(__func__, devLogicId, dieId, dieEnableFlags));
    pfeNum = resSpecs[dieId].pfeNum;
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuResSpecifications::GetWqeBBNum(const uint8_t dieId, uint32_t& wqeBBNum) const
{
    CHK_RET(CheckDieValid(__func__, devLogicId, dieId, dieEnableFlags));
    wqeBBNum = resSpecs[dieId].wqeBBNum;
    return HcclResult::HCCL_SUCCESS;
}

}; // namespace Hccl
