/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_res_specs.h"

#include "hccp_ctx.h"
#include "hccl_common.h"
#include "hcomm_adapter_hccp.h"

namespace hcomm {

CcuResSpecifications& CcuResSpecifications::GetInstance(const int32_t userDevId)
{
    static CcuResSpecifications ccuResSpecifications[MAX_MODULE_DEVICE_NUM + 1];
    int32_t validUserDevId = userDevId;
    if (validUserDevId < 0 || static_cast<uint32_t>(validUserDevId) >= MAX_MODULE_DEVICE_NUM) {
        HCCL_WARNING(
            "[CcuResSpecifications][%s] use the backup device, userDevId[%d] "
            "should be less than %u.",
            __func__, validUserDevId, MAX_MODULE_DEVICE_NUM);
        validUserDevId = MAX_MODULE_DEVICE_NUM; // 使用备份设备
    }
    ccuResSpecifications[validUserDevId].userDevId_ = validUserDevId;
    return ccuResSpecifications[validUserDevId];
}

static CcuVersion CheckCcuVersion(int32_t userDevId)
{
    CustomChannelInfoIn inBuff{};
    CustomChannelInfoOut outBuff{};
    inBuff.op = CcuOpcodeType::CCU_U_OP_GET_VERSION;
    inBuff.offsetStartIdx = 0;
    inBuff.data.dataInfo.udieIdx = 0; // 查询版本访问任意die均可

    auto ret = HccpRaTlvCcuCustomChannel(userDevId, static_cast<void*>(&inBuff), static_cast<void*>(&outBuff));
    if (ret != 0) {
        HCCL_ERROR(
            "[%s] failed to call ccu driver, "
            "userDevId[%d], op[%s].",
            __func__, userDevId, "GET_CCU_VERSION");
        return CcuVersion::CCU_INVALID;
    }
    const CcuVersionEnum ccuVersionEnum = outBuff.data.dataInfo.dataArray[0].ccuVersion;

    static const std::unordered_map<int, CcuVersion> ccuVersionMap
        = {{static_cast<int>(CcuVersionEnum::CCU_V1), CcuVersion::CCU_V1},
           {static_cast<int>(CcuVersionEnum::CCU_V2), CcuVersion::CCU_V2}};

    const auto& iter = ccuVersionMap.find(static_cast<int>(ccuVersionEnum));
    if (iter == ccuVersionMap.end()) {
        HCCL_ERROR(
            "[CcuResSpecifications][%s] failed, ccu driver version[%d] "
            "is invalid, userDevId[%d].",
            __func__, static_cast<int>(ccuVersionEnum), userDevId);
        return CcuVersion::CCU_INVALID;
    }

    return iter->second;
}

static bool CheckDieEnable(const uint32_t userDevId, const uint8_t dieId)
{
    CustomChannelInfoIn inBuff{};
    CustomChannelInfoOut outBuff{};
    inBuff.op = CcuOpcodeType::CCU_U_OP_GET_DIE_WORKING;
    inBuff.offsetStartIdx = 0;
    inBuff.data.dataInfo.udieIdx = dieId;

    auto ret = HccpRaTlvCcuCustomChannel(userDevId, static_cast<void*>(&inBuff), static_cast<void*>(&outBuff));
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[CcuResSpecifications][%s] failed to call ccu driver, "
            "userDevId[%d] dieId[%d] op[%s] ret[%d].",
            __func__, userDevId, dieId, "GET_DIE_WORKING", ret);
        return false;
    }

    const uint32_t enableFlag = outBuff.data.dataInfo.dataArray[0].dieinfo.enableFlag;
    return enableFlag == Hccl::CCU_ENABLE_FLAG;
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
    if (ccuVersion == CcuVersion::CCU_INVALID || ccuVersion == CcuVersion::INVALID) {
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

    if (ccuVersion == CcuVersion::CCU_V1) {
        ccuResSpecInfo.missionNum = ((baseInfoData.caps.cap0 >> MOVE_16_BITS) & 0x000000FF) + 1;
        ccuResSpecInfo.loopEngineNum = ((baseInfoData.caps.cap0 >> MOVE_24_BITS) & 0x000000FF) + 1;
        ccuResSpecInfo.gsaNum = (baseInfoData.caps.cap1 & 0x0000FFFF) + 1;
        return ccuResSpecInfo;
    }

    ccuResSpecInfo.loopEngineNum = ((baseInfoData.caps.cap0 >> MOVE_20_BITS) & 0x00000FFF) + 1;
    ccuResSpecInfo.missionNum = ((baseInfoData.caps.cap0 >> MOVE_16_BITS) & 0x0000000F) + 1;
    ccuResSpecInfo.gsaNum = 0; // v2废弃字段，固定为0;

    // loop cke 数量与loop一致，与cke数量寄存器隔离
    // loop cke 从全局 cke 划分后半部分，资源不足时cke为0
    HCCL_INFO(
        "ParseOutBuffToResSpecInfo, loopEngineNum[%u], ckeNum[%u]", ccuResSpecInfo.loopEngineNum,
        ccuResSpecInfo.ckeNum);
    if (ccuResSpecInfo.loopEngineNum < ccuResSpecInfo.ckeNum) {
        HCCL_INFO("ccuResSpecInfo.loopEngineNum < ccuResSpecInfo.ckeNum");
        ccuResSpecInfo.loopCkeNum = ccuResSpecInfo.loopEngineNum; // ccu v2 loop cke要求与loop静态对应
    } else {
        HCCL_INFO("ccuResSpecInfo.loopEngineNum >= ccuResSpecInfo.ckeNum");
        ccuResSpecInfo.ckeNum = 0; // cke资源信息错误
    }

    // count xn 从全局xn划分后半部分，资源不足时count xn为0
    if (ccuResSpecInfo.xnNum > CCU_V2_COUNT_XN_NUM) {
        ccuResSpecInfo.xnNum -= CCU_V2_COUNT_XN_NUM;
        ccuResSpecInfo.countXnNum = CCU_V2_COUNT_XN_NUM;
    }

    HCCL_INFO(
        "[CcuResSpecifications][%s] msId[%u] resourceAddr[%llx] channelJettyMap[%u, %u].", __func__,
        ccuResSpecInfo.msId, ccuResSpecInfo.resourceAddr, ccuResSpecInfo.channelJettyMap.channelNum,
        ccuResSpecInfo.channelJettyMap.jettyNum);
    return ccuResSpecInfo;
}

static HcclResult CheckResSpecifications(
    const int32_t userDevId, const uint8_t dieId, const CcuVersion ccuVersion, CcuResSpecInfo& resSpecs)
{
    CustomChannelInfoIn inBuff{};
    CustomChannelInfoOut outBuff{};
    inBuff.op = CcuOpcodeType::CCU_U_OP_GET_BASIC_INFO;
    inBuff.offsetStartIdx = 0;
    inBuff.data.dataInfo.udieIdx = dieId;

    auto ret = HccpRaTlvCcuCustomChannel(userDevId, static_cast<void*>(&inBuff), static_cast<void*>(&outBuff));
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[CcuResSpecifications][%s] failed to call ccu driver, "
            "userDevId[%d] dieId[%d] op[%s] ret[%d].",
            __func__, userDevId, dieId, "GET_BASIC_INFO", ret);
        return ret;
    }
    resSpecs = ParseOutBuffToResSpecInfo(ccuVersion, outBuff);
    return HcclResult::HCCL_SUCCESS;
}

/*
 * 获取Mainboard ID 5-7位，输出整机形态枚举值
 * Mainboard ID描述说明
 * Mainboard ID采用了16bit，区分形态，主从，以及端口配置
 * bit[7:5] 区分整机形态(当前POD和EVB没有区分A+X或A+K)
 * {
 *  000: 天成 POD
 *  001: A+K Server
 *  010: A+X Server
 *  011: PCIE标卡
 *  100-101: RSV
 *  110: 装备
 *  111: EVB
 * }
 * bit[4:1] 整机形态细分
 * {
 *  0000-1111
 * }
 * bit[0] 主从或池化
 * {
 *  0: 主从（NPU作为某个Host的从设备，Host主控）
 *  1: 池化（NPU作为资源池，其它Host对等访问）
 * }
 */
constexpr uint64_t POD_MAINBOARD = 0x0;
constexpr uint64_t A_K_SERVER_MAINBOARD = 0x1;
constexpr uint64_t A_X_SERVER_MAINBOARD = 0x2;
constexpr uint64_t PCIE_STD_MAINBOARD = 0x3;
constexpr uint64_t RSV1_MAINBOARD = 0x4;
constexpr uint64_t RSV2_MAINBOARD = 0x5;
constexpr uint64_t EQUIP_MAINBOARD = 0x6;
constexpr uint64_t EVB_MAINBOARD = 0x7;
const std::unordered_map<uint64_t, Hccl::HcclMainboardId> rtMainboardIdToHcclMainboardId
    = {{POD_MAINBOARD, Hccl::HcclMainboardId::MAINBOARD_POD},
       {A_K_SERVER_MAINBOARD, Hccl::HcclMainboardId::MAINBOARD_A_K_SERVER},
       {A_X_SERVER_MAINBOARD, Hccl::HcclMainboardId::MAINBOARD_A_X_SERVER},
       {PCIE_STD_MAINBOARD, Hccl::HcclMainboardId::MAINBOARD_PCIE_STD},
       {RSV1_MAINBOARD, Hccl::HcclMainboardId::MAINBOARD_RSV},
       {RSV2_MAINBOARD, Hccl::HcclMainboardId::MAINBOARD_RSV},
       {EQUIP_MAINBOARD, Hccl::HcclMainboardId::MAINBOARD_EQUIPMENT},
       {EVB_MAINBOARD, Hccl::HcclMainboardId::MAINBOARD_EVB}};

HcclResult CcuGetMainboardId(uint32_t userDevId, Hccl::HcclMainboardId& hcclMainboardId)
{
    constexpr aclrtDevAttr devAttr = aclrtDevAttr::ACL_DEV_ATTR_MAINBOARD_ID;
    constexpr uint64_t BITS_5 = 5;
    constexpr uint64_t MASK_7 = 0x7;
    int64_t val = 0;
    auto ret = aclrtGetDeviceInfo(userDevId, devAttr, &val);
    if (ret != RT_ERROR_NONE) {
        HCCL_ERROR(
            "[%s]errNo[0x%016llx] rt get device info failed, "
            "userDevId=%u, devAttr=%d",
            __func__, HCCL_ERROR_CODE(HcclResult::HCCL_E_RUNTIME), userDevId, devAttr);
        return HcclResult::HCCL_E_RUNTIME;
    }

    HCCL_INFO("[%s] userDevId[%u] val[%lld].", __func__, userDevId, static_cast<long long>(val));
    uint64_t mainboardId = (static_cast<uint64_t>(val) >> BITS_5) & MASK_7; // 提取val的5-7位，判断整机形态
    hcclMainboardId = Hccl::HcclMainboardId::MAINBOARD_OTHERS;
    auto it = rtMainboardIdToHcclMainboardId.find(mainboardId);
    if (it != rtMainboardIdToHcclMainboardId.end()) {
        hcclMainboardId = it->second;
    }
    HCCL_INFO(
        "[%s] userDevId[%u] mainboardId[%llu] hcclMainboardId[%s].", __func__, userDevId,
        static_cast<unsigned long long>(mainboardId), hcclMainboardId.Describe().c_str());
    return HcclResult::HCCL_SUCCESS;
}

static HcclResult CheckServeMode(int32_t userDevId, ServeMode& serveMode)
{
    Hccl::HcclMainboardId hcclMainboardId{Hccl::HcclMainboardId::MAINBOARD_RSV};
    CHK_RET(CcuGetMainboardId(userDevId, hcclMainboardId));
    if (hcclMainboardId == Hccl::HcclMainboardId::MAINBOARD_A_X_SERVER
        || hcclMainboardId == Hccl::HcclMainboardId::MAINBOARD_PCIE_STD) {
        serveMode = ServeMode::ARMX86;
    } else {
        serveMode = ServeMode::NORMAL;
    }
    HCCL_INFO(
        "[CcuResSpecifications][%s] userDevId[%d] "
        "hcclMainboardId[%s] serveMode_[%d].",
        __func__, userDevId, hcclMainboardId.Describe().c_str(), static_cast<int>(serveMode));
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuResSpecifications::Init()
{
    if (initFlag_) {
        return HcclResult::HCCL_SUCCESS;
    }

    CHK_RET(hrtGetDevicePhyIdByIndex(static_cast<uint32_t>(userDevId_), devPhyId_));
    CHK_RET(CheckServeMode(userDevId_, serveMode_));
    ccuVersion_ = CheckCcuVersion(userDevId_);
    if (ccuVersion_ == CcuVersion::CCU_INVALID) {
        HCCL_WARNING("[CcuResSpecifications][%s] check ccu version failed.", __func__);
        return HcclResult::HCCL_E_UNAVAIL;
    }
    for (uint8_t dieId = 0; dieId < CCU_MAX_IODIE_NUM; dieId++) {
        dieEnableFlags_[dieId] = CheckDieEnable(userDevId_, dieId);
        if (!dieEnableFlags_[dieId]) {
            resSpecs_[dieId] = CcuResSpecInfo{};
            continue;
        }

        CHK_RET(CheckResSpecifications(userDevId_, dieId, ccuVersion_, resSpecs_[dieId]));
    }

    initFlag_ = true;
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuResSpecifications::Deinit()
{
    for (uint32_t i = 0; i < CCU_MAX_IODIE_NUM; i++) {
        dieEnableFlags_[i] = false;
        resSpecs_[i] = CcuResSpecInfo{};
    }

    serveMode_ = ServeMode::ARMX86;
    initFlag_ = false;
    return HcclResult::HCCL_SUCCESS;
}

CcuVersion CcuResSpecifications::GetCcuVersion() const { return ccuVersion_; }

HcclResult CcuResSpecifications::CheckDieValid(
    const std::string& funcName, const int32_t userDevId, const uint8_t dieId,
    const std::array<bool, CCU_MAX_IODIE_NUM>& dieEnableFlags) const
{
    CHK_PRT_RET(
        dieId >= CCU_MAX_IODIE_NUM,
        HCCL_ERROR(
            "[%s] failed, dieId[%u] is invalid, should be in [0-%u), userDevId[%d].", funcName.c_str(), dieId,
            CCU_MAX_IODIE_NUM, userDevId),
        HcclResult::HCCL_E_PARA);

    CHK_PRT_RET(
        !dieEnableFlags[dieId],
        HCCL_WARNING("[%s] failed, dieId[%u] is disable, userDevId[%d].", funcName.c_str(), dieId, userDevId),
        HcclResult::HCCL_E_PARA);

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuResSpecifications::GetDieEnableFlag(const uint8_t dieId, bool& dieEnableFlag) const
{
    // 只校验dieId合法性，不校验die是否使能
    CHK_RET(CheckDieValid(__func__, userDevId_, dieId, {true, true}));
    dieEnableFlag = dieEnableFlags_[dieId];
    return HcclResult::HCCL_SUCCESS;
}

ServeMode CcuResSpecifications::GetServeMode() const { return serveMode_; }

HcclResult CcuResSpecifications::GetResourceAddr(const uint8_t dieId, uint64_t& resourceAddr) const
{
    CHK_RET(CheckDieValid(__func__, userDevId_, dieId, dieEnableFlags_));
    resourceAddr = resSpecs_[dieId].resourceAddr;
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuResSpecifications::GetXnBaseAddr(const uint8_t dieId, uint64_t& xnBaseAddr) const
{
    // xn位于ins与gsa之后，xn偏移 = CCUM偏移 + 指令空间大小 + GSA大小，常量计算不会溢出
    constexpr uint64_t instrRevserveSize = CCU_RESOURCE_INS_RESERVE_SIZE;
    constexpr uint64_t gsaReserveSize = CCU_V1_RESOURCE_GSA_RESERVE_SIZE;

    CHK_RET(CheckDieValid(__func__, userDevId_, dieId, dieEnableFlags_));
    const uint64_t ccuResAddr = resSpecs_[dieId].resourceAddr;
    if (ccuResAddr == 0) {
        HCCL_ERROR(
            "[CcuResSpecifications][%s] failed, CCU resource base address is 0, "
            "userDevId[%d] dieId[%u].",
            __func__, userDevId_, dieId);
        return HcclResult::HCCL_E_INTERNAL;
    }

    uint64_t ccum_offset = 0;
    uint32_t ccuXnOffset = 0;
    if (ccuVersion_ == CcuVersion::CCU_V1) {
        ccum_offset = CCU_V1_CCUM_OFFSET;
        ccuXnOffset = ccum_offset + instrRevserveSize + gsaReserveSize;
    } else if (ccuVersion_ == CcuVersion::CCU_V2) {
        ccum_offset = CCU_V2_CCUM_OFFSET;
        ccuXnOffset = ccum_offset + instrRevserveSize;
    } else {
        HCCL_ERROR(
            "[CcuResSpecifications][%s]current version[%s] not support", __func__, ccuVersion_.Describe().c_str());
        return HcclResult::HCCL_E_INTERNAL;
    }

    if (ccuResAddr > UINT64_MAX - ccuXnOffset) {
        HCCL_ERROR(
            "[CcuResSpecifications][%s] failed, CCU resource base address[0x%llx] is "
            "greater than expected, ccu xn offset[%u], their sum will exceed the range "
            "of uint64_t.",
            __func__, static_cast<unsigned long long>(ccuResAddr), ccuXnOffset);
        return HcclResult::HCCL_E_INTERNAL;
    }

    xnBaseAddr = ccuResAddr + ccuXnOffset;
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuResSpecifications::GetCkeBaseAddr(const uint8_t dieId, uint64_t& ckeBaseAddr) const
{
    constexpr uint64_t instrRevserveSize = CCU_RESOURCE_INS_RESERVE_SIZE;
    constexpr uint64_t gsaReserveSize = CCU_V1_RESOURCE_GSA_RESERVE_SIZE;
    constexpr uint64_t xnReserveSizeV1 = CCU_RESOURCE_XN_V1_RESERVE_SIZE;
    constexpr uint64_t xnReserveSizeV2 = CCU_RESOURCE_XN_V2_RESERVE_SIZE;

    CHK_RET(CheckDieValid(__func__, userDevId_, dieId, dieEnableFlags_));
    const uint64_t ccuResAddr = resSpecs_[dieId].resourceAddr;
    if (ccuResAddr == 0) {
        HCCL_ERROR(
            "[CcuResSpecifications][%s] failed, CCU resource base address is 0, "
            "userDevId[%d] dieId[%u].",
            __func__, userDevId_, dieId);
        return HcclResult::HCCL_E_INTERNAL;
    }

    uint64_t ccum_offset = CCU_V1_CCUM_OFFSET;
    uint32_t ccuCkeOffset = 0;
    // ccuCkeOffset 为常量计算，不涉及溢出
    if (ccuVersion_ == CcuVersion::CCU_V1) {
        ccuCkeOffset = ccum_offset + instrRevserveSize + gsaReserveSize + xnReserveSizeV1;
    } else if (ccuVersion_ == CcuVersion::CCU_V2) {
        ccum_offset = CCU_V2_CCUM_OFFSET;
        ccuCkeOffset = ccum_offset + instrRevserveSize + xnReserveSizeV2;
    } else {
        HCCL_ERROR(
            "[CcuResSpecifications][%s]current version[%s] not support", __func__, ccuVersion_.Describe().c_str());
        return HcclResult::HCCL_E_INTERNAL;
    }

    if (ccuResAddr > UINT64_MAX - ccuCkeOffset) {
        HCCL_ERROR(
            "[CcuResSpecifications][%s] failed, CCU resource base address[0x%llx] is "
            "greater than expected, ccu cke offset[%u], their sum will exceed the range "
            "of uint64_t.",
            __func__, static_cast<unsigned long long>(ccuResAddr), ccuCkeOffset);
        return HcclResult::HCCL_E_INTERNAL;
    }
    ckeBaseAddr = ccuResAddr + ccuCkeOffset;
    HCCL_DEBUG(
        "[CcuResSpecifications][%s] userDevId[%d], dieId[%u], ckeBaseAddr[0x%llx]", __func__, userDevId_, dieId,
        ckeBaseAddr);
    return HcclResult::HCCL_SUCCESS;
}

uint64_t CcuResSpecifications::GetXnOffsetCcumBaseAddr(const uint8_t dieId) const
{
    CHK_PRT_RET(
        dieId >= CCU_MAX_IODIE_NUM || !dieEnableFlags_[dieId],
        HCCL_ERROR("[CcuResSpecifications][GetCcuMBaseAddr] userDevId[%d], dieId[%u] is invalid.", userDevId_, dieId),
        INVALID_ADDR);
    uint64_t xnOffsetCcumBaseAddr{0};
    if (ccuVersion_ == CcuVersion::CCU_V1) {
        xnOffsetCcumBaseAddr
            = CCU_RESOURCE_INS_RESERVE_SIZE + CCU_V1_RESOURCE_GSA_RESERVE_SIZE; // CCUM + 1M Instruction + 32K GSA
    } else if (ccuVersion_ == CcuVersion::CCU_V2) {
        xnOffsetCcumBaseAddr = CCU_RESOURCE_INS_RESERVE_SIZE; // CCUM + 1M Instruction
    } else {
        xnOffsetCcumBaseAddr = 0;
    }
    HCCL_DEBUG(
        "[CcuResSpecifications][%s] userDevId[%d], dieId[%u], xnOffsetCcumBaseAddr[0x%llx]", __func__, userDevId_,
        dieId, xnOffsetCcumBaseAddr);
    return xnOffsetCcumBaseAddr;
}

static HcclResult CheckResOffsetAddrIsValid(uint32_t id, uint64_t baseAddr, uint32_t resoucePerSize)
{
    if (baseAddr == INVALID_ADDR) {
        HCCL_ERROR("[CcuResSpecifications][%s] failed, base addr is invalid.", __func__);
        return HcclResult::HCCL_E_INTERNAL;
    }

    if (id > UINT32_MAX / resoucePerSize) {
        HCCL_ERROR("[CcuResSpecifications][%s] failed, id[%u] is invalid.", __func__, id);
        return HcclResult::HCCL_E_PARA;
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuResSpecifications::GetXnOffsetCcumAddrById(const uint8_t dieId, uint32_t id, uint64_t& xnAddr) const
{
    uint64_t xnBaseAddr = GetXnOffsetCcumBaseAddr(dieId);
    CHK_RET(CheckResOffsetAddrIsValid(id, xnBaseAddr, CCU_RESOURCE_XN_PER_SIZE));

    xnAddr = xnBaseAddr + id * CCU_RESOURCE_XN_PER_SIZE;
    HCCL_DEBUG(
        "[CcuResSpecifications][%s] userDevId[%d], dieId[%u], xnBaseAddr[0x%llx], xn[%d][0x%llx]", __func__, userDevId_,
        dieId, xnBaseAddr, id, xnAddr);
    return HcclResult::HCCL_SUCCESS;
}

uint64_t CcuResSpecifications::GetCkeOffsetCcumBaseAddr(const uint8_t dieId) const
{
    CHK_PRT_RET(
        dieId >= CCU_MAX_IODIE_NUM || !dieEnableFlags_[dieId],
        HCCL_ERROR("[CcuResSpecifications][GetCcuMBaseAddr] userDevId[%d], dieId[%u] is invalid.", userDevId_, dieId),
        INVALID_ADDR);

    uint64_t ckeBaseAddr{0};
    if (ccuVersion_ == CcuVersion::CCU_V1) {
        ckeBaseAddr = CCU_RESOURCE_INS_RESERVE_SIZE + CCU_V1_RESOURCE_GSA_RESERVE_SIZE
                      + CCU_RESOURCE_XN_V1_RESERVE_SIZE; // CCUM + 1M Instruction + 32K GSA + 32K Xn
    } else if (ccuVersion_ == CcuVersion::CCU_V2) {
        ckeBaseAddr
            = CCU_RESOURCE_INS_RESERVE_SIZE + CCU_RESOURCE_XN_V2_RESERVE_SIZE; // CCUM + 1M Instruction + 256K Xn
    } else {
        ckeBaseAddr = INVALID_ADDR;
    }
    HCCL_DEBUG(
        "[CcuResSpecifications][%s] userDevId[%d], dieId[%u], ckeBaseAddr[0x%llx]", __func__, userDevId_, dieId,
        ckeBaseAddr);
    return ckeBaseAddr;
}

HcclResult CcuResSpecifications::GetCkeOffsetCcumAddrById(const uint8_t dieId, uint32_t id, uint64_t& ckeAddr) const
{
    uint64_t ckeBaseAddr = GetCkeOffsetCcumBaseAddr(dieId);
    CHK_RET(CheckResOffsetAddrIsValid(id, ckeBaseAddr, CCU_RESOURCE_CKE_PER_SIZE));

    ckeAddr = ckeBaseAddr + id * CCU_RESOURCE_CKE_PER_SIZE;
    HCCL_DEBUG(
        "[CcuResSpecifications][%s] userDevId[%d], dieId[%u], ckeBaseAddr[0x%llx], ckeAddr[%d][0x%llx]", __func__,
        userDevId_, dieId, ckeBaseAddr, id, ckeAddr);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuResSpecifications::GetMsId(const uint8_t dieId, uint32_t& msId) const
{
    CHK_RET(CheckDieValid(__func__, userDevId_, dieId, dieEnableFlags_));
    msId = resSpecs_[dieId].msId;
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuResSpecifications::GetMissionKey(const uint8_t dieId, uint32_t& missionKey) const
{
    CHK_RET(CheckDieValid(__func__, userDevId_, dieId, dieEnableFlags_));
    missionKey = resSpecs_[dieId].missionKey;
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuResSpecifications::GetInstructionNum(const uint8_t dieId, uint32_t& instrNum) const
{
    CHK_RET(CheckDieValid(__func__, userDevId_, dieId, dieEnableFlags_));
    instrNum = resSpecs_[dieId].instructionNum;
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuResSpecifications::GetMissionNum(const uint8_t dieId, uint32_t& missionNum) const
{
    CHK_RET(CheckDieValid(__func__, userDevId_, dieId, dieEnableFlags_));
    missionNum = resSpecs_[dieId].missionNum;
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuResSpecifications::GetLoopEngineNum(const uint8_t dieId, uint32_t& loopNum) const
{
    CHK_RET(CheckDieValid(__func__, userDevId_, dieId, dieEnableFlags_));
    loopNum = resSpecs_[dieId].loopEngineNum;
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuResSpecifications::GetGsaNum(const uint8_t dieId, uint32_t& gsaNum) const
{
    CHK_RET(CheckDieValid(__func__, userDevId_, dieId, dieEnableFlags_));
    gsaNum = resSpecs_[dieId].gsaNum;
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuResSpecifications::GetXnNum(const uint8_t dieId, uint32_t& xnNum) const
{
    CHK_RET(CheckDieValid(__func__, userDevId_, dieId, dieEnableFlags_));
    xnNum = resSpecs_[dieId].xnNum;
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuResSpecifications::GetCountXnNum(const uint8_t dieId, uint32_t& countXnNum) const
{
    CHK_RET(CheckDieValid(__func__, userDevId_, dieId, dieEnableFlags_));
    countXnNum = resSpecs_[dieId].countXnNum;
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuResSpecifications::GetCascCntNum(const uint8_t dieId, uint32_t& cascCntNum) const
{
    CHK_RET(CheckDieValid(__func__, userDevId_, dieId, dieEnableFlags_));
    if (ccuVersion_ != CcuVersion::CCU_V2) {
        cascCntNum = 0;
        return HcclResult::HCCL_SUCCESS;
    }
    cascCntNum = CCU_V2_RESOURCE_TOTAL_CNT_XNS_NUM;
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuResSpecifications::GetCkeNum(const uint8_t dieId, uint32_t& ckeNum) const
{
    CHK_RET(CheckDieValid(__func__, userDevId_, dieId, dieEnableFlags_));
    ckeNum = resSpecs_[dieId].ckeNum;
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuResSpecifications::GetLoopCkeNum(const uint8_t dieId, uint32_t& loopCkeNum) const
{
    CHK_RET(CheckDieValid(__func__, userDevId_, dieId, dieEnableFlags_));
    loopCkeNum = resSpecs_[dieId].loopCkeNum;
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuResSpecifications::GetMsNum(const uint8_t dieId, uint32_t& msNum) const
{
    CHK_RET(CheckDieValid(__func__, userDevId_, dieId, dieEnableFlags_));
    msNum = resSpecs_[dieId].msNum;
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuResSpecifications::GetChannelNum(const uint8_t dieId, uint32_t& channelNum) const
{
    CHK_RET(CheckDieValid(__func__, userDevId_, dieId, dieEnableFlags_));
    channelNum = resSpecs_[dieId].channelNum;
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuResSpecifications::GetJettyNum(const uint8_t dieId, uint32_t& jettyNum) const
{
    CHK_RET(CheckDieValid(__func__, userDevId_, dieId, dieEnableFlags_));
    jettyNum = resSpecs_[dieId].jettyNum;
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuResSpecifications::GetPfeReservedNum(const uint8_t dieId, uint32_t& pfeNum) const
{
    CHK_RET(CheckDieValid(__func__, userDevId_, dieId, dieEnableFlags_));
    if (ccuVersion_ == CcuVersion::CCU_V1) {
        pfeNum = CCU_V1_PER_DIE_PFE_RESERVED_NUM;
        return HcclResult::HCCL_SUCCESS;
    }
    pfeNum = CCU_V2_PER_DIE_PFE_RESERVED_NUM;
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuResSpecifications::GetPfeNum(const uint8_t dieId, uint32_t& pfeNum) const
{
    CHK_RET(CheckDieValid(__func__, userDevId_, dieId, dieEnableFlags_));
    pfeNum = resSpecs_[dieId].pfeNum;
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuResSpecifications::GetWqeBBNum(const uint8_t dieId, uint32_t& wqeBBNum) const
{
    CHK_RET(CheckDieValid(__func__, userDevId_, dieId, dieEnableFlags_));
    wqeBBNum = resSpecs_[dieId].wqeBBNum;
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuResSpecifications::GetChannelJettyMap(const uint8_t dieId, CcuChannelJettyMap& channelJettyMap) const
{
    CHK_RET(CheckDieValid(__func__, userDevId_, dieId, dieEnableFlags_));
    channelJettyMap = resSpecs_[dieId].channelJettyMap;
    return HcclResult::HCCL_SUCCESS;
}
} // namespace hcomm
