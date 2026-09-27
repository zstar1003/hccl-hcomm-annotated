/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "network/hccp_ping.h"
#include "ping_mesh.h"
#include "hccl_ip_address.h"
#include "dlra_function.h"
#include "adapter_rts_common.h"
#include "hccn_rping.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

#define HCCN_RPING_MIN_TIMEOUT 1
#define HCCN_RPING_MAX_TIMEOUT 3600000
#define HCCN_RPING_DEFAULT_TIMEOUT 120000

using namespace hccl;
constexpr u32 BUFFER_SIZE_UNIT = 4096;
constexpr u32 NPU_NUM_MAX = 32768;
constexpr u32 NPU_NUM_MIN = 128;
constexpr u32 NPU_NUM_MAX_BITWIDTH = 128;
constexpr u32 LINK_TYPE_MODE_ROCE = 3;
constexpr u32 LINK_TYPE_MODE_UB = 7;
constexpr u32 RPING_RESULT_STATE_VALID = 2;
constexpr u32 TARGET_NUM_MAX = 16;

inline HccnResult HccnRpingInitInter(
    uint32_t& devLogicIdInter, HccnRpingInitAttr* initAttrInter, PingMesh* rpingInter, u32& npuNumInter,
    std::string& ipAddrDesInter)
{
    u32 bufferSizeInter = 0;
    // npuNum必须为2的整次幂
    npuNumInter = (initAttrInter->npuNum > NPU_NUM_MIN) ? initAttrInter->npuNum : NPU_NUM_MIN;
    npuNumInter = (npuNumInter > NPU_NUM_MAX) ? (NPU_NUM_MAX - 1) : (npuNumInter - 1);
    for (u32 i = 0; i < NPU_NUM_MAX_BITWIDTH; i++) {
        npuNumInter |= (npuNumInter >> 1);
    }
    npuNumInter++;
    npuNumInter = npuNumInter < NPU_NUM_MIN ? NPU_NUM_MIN : npuNumInter;
    // bufferSize必须为4k的倍数
    if (initAttrInter->bufferSize > 0) {
        bufferSizeInter = ((initAttrInter->bufferSize - 1) / BUFFER_SIZE_UNIT + 1) * BUFFER_SIZE_UNIT;
    }
    // 初始化pingmesh实例,利用用户输入的ip地址构造HcclIpAddress类
    HcclIpAddress ipAddr;
    HcclResult ret = HCCL_SUCCESS;
    if (initAttrInter->mode == HCCN_RPING_MODE_ROCE) {
        if (!HcclIpAddress::IsIPv4(std::string(initAttrInter->ipAddr))
            && !HcclIpAddress::IsIPv6(std::string(initAttrInter->ipAddr))) {
            HCCL_ERROR("[HccnRpingInitInter] invalid ip: %s, bufferSize:%u", initAttrInter->ipAddr, bufferSizeInter);
            return HCCN_E_PARA;
        }
        ipAddr = HcclIpAddress(std::string(initAttrInter->ipAddr));
        ret = rpingInter->HccnRpingInit(
            devLogicIdInter, LINK_TYPE_MODE_ROCE, ipAddr, initAttrInter->port, npuNumInter, bufferSizeInter,
            initAttrInter->sl, initAttrInter->tc);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("[HccnRpingInitInter] init fail, bufferSize:%u", bufferSizeInter);
            return HCCN_E_FAIL;
        }
        ipAddrDesInter = ipAddr.GetReadableIP();
    }
    const char* socNamePtr = aclrtGetSocName();
    CHK_PRT_RET(socNamePtr == nullptr, HCCL_ERROR("[HccnRpingInitInter]socNamePtr is null."), HCCN_E_PARA);
    ;
    if (initAttrInter->mode == HCCN_RPING_MODE_UB && IsSupportHCCLV2(socNamePtr)) {
        if (!HcclIpAddress::IsEID(std::string(initAttrInter->eid))) {
            HCCL_ERROR("[HccnRpingInitInter] invalid eid: %s, bufferSize:%u", initAttrInter->eid, bufferSizeInter);
            return HCCN_E_PARA;
        }
        Eid eid = HcclIpAddress::StrToEID(std::string(initAttrInter->eid));
        ipAddr = HcclIpAddress(eid);
        ret = rpingInter->HccnRpingInit(
            devLogicIdInter, LINK_TYPE_MODE_UB, ipAddr, initAttrInter->port, npuNumInter, bufferSizeInter,
            initAttrInter->sl, initAttrInter->tc);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("[HccnRpingInitInter] init fail, bufferSize:%u, ret:%d", bufferSizeInter, ret);
            return HCCN_E_FAIL;
        }
        ipAddrDesInter = ipAddr.Describe();
    }
    HCCL_RUN_INFO("[HccnRpingInitInter]bufferSize:%u, ret:%d", bufferSizeInter, ret);
    return HCCN_SUCCESS;
}

HccnResult HccnRpingInit(uint32_t devLogicId, HccnRpingInitAttr* initAttr, HccnRpingCtx* rpingCtx)
{
    // 校验入参
    CHK_PRT_RET(initAttr == nullptr, HCCL_ERROR("[HccnRpingInit]initAttr is null."), HCCN_E_PARA);
    CHK_PRT_RET(
        initAttr->mode >= HCCN_RPING_MODE_RESERVED,
        HCCL_ERROR("[HccnRpingInit]LinkMode[%d] not support.", initAttr->mode), HCCN_E_PARA);
    CHK_PRT_RET(rpingCtx == nullptr, HCCL_ERROR("[HccnRpingInit]rpingCtx is null."), HCCN_E_PARA);
    // 初始化ra接口
    CHK_PRT_RET(
        DlRaFunction::GetInstance().DlRaFunctionInit() != HCCL_SUCCESS,
        HCCL_ERROR("[HccnRpingInit]dlrafunction failed."), HCCN_E_FAIL);
    std::string initAttrDes;
    if (initAttr->mode == HCCN_RPING_MODE_ROCE) {
        initAttrDes = std::string(initAttr->ipAddr).c_str();
    }
    const char* socNamePtr = aclrtGetSocName();
    CHK_PRT_RET(socNamePtr == nullptr, HCCL_ERROR("[HccnRpingInit]socNamePtr is null."), HCCN_E_PARA);
    if (initAttr->mode == HCCN_RPING_MODE_UB && IsSupportHCCLV2(socNamePtr)) {
        initAttrDes = std::string(initAttr->eid).c_str();
    }
    HCCL_RUN_INFO(
        "[HccnRpingInit]devLogicId:%u, mode:%d port:%u npuNum:%u bufferSize:%u sl:%u tc:%u ip:%s", devLogicId,
        initAttr->mode, initAttr->port, initAttr->npuNum, initAttr->bufferSize, initAttr->sl, initAttr->tc,
        initAttrDes);
    // 获取device id
    s32 currDevLogicId = 0;
    HcclResult ret = hrtGetDeviceRefresh(&currDevLogicId);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS, HCCL_ERROR("[HccnRpingInit]cannot get device logic id, ret[%d].", ret), HCCN_E_FAIL);

    // 判断devLogicId与pingmesh所记录的是否一致
    if (devLogicId != static_cast<u32>(currDevLogicId)) {
        HCCL_ERROR(
            "[HccnRpingInit]Input device logicId[%u] don't match current logicId[%d].", devLogicId, currDevLogicId);
        return HCCN_E_FAIL;
    }
    // 构造一个pingmesh实例, 用于存放初始化的pingmesh信息
    PingMesh* rping = new (std::nothrow) PingMesh();
    CHK_PRT_RET(rping == nullptr, HCCL_ERROR("[HccnRpingInit]rping alloc failed."), HCCN_E_MEM);
    rping->init(initAttr); // 定义现在pingmesh的mode

    u32 npuNum = 0;
    std::string ipAddrDes;

    if (HccnRpingInitInter(devLogicId, initAttr, rping, npuNum, ipAddrDes) != 0) {
        HCCL_ERROR(
            "[HccnRpingInit]Pingmesh init failed, devLogicId:%u, port:%u, npuNum:%u, sl:%u"
            " tc:%u ip:%s",
            devLogicId, initAttr->port, npuNum, initAttr->sl, initAttr->tc, ipAddrDes);
        delete rping; // 初始化失败释放内存
        return HCCN_E_FAIL;
    }

    HCCL_RUN_INFO(
        "[HccnRpingInit]Pingmesh init success, devLogicId:%u, mode:%d port:%u npuNum:%u sl:%u"
        " tc:%u ip:%s",
        devLogicId, initAttr->mode, initAttr->port, npuNum, initAttr->sl, initAttr->tc, ipAddrDes);
    // 记录pingmesh指针
    *rpingCtx = rping;
    return HCCN_SUCCESS;
}

HccnResult HccnRpingDeinit(HccnRpingCtx rpingCtx)
{
    // 初始化ra接口
    CHK_PRT_RET(
        DlRaFunction::GetInstance().DlRaFunctionInit() != HCCL_SUCCESS,
        HCCL_ERROR("[HccnRpingDeinit]dlrafunction failed."), HCCN_E_FAIL);
    // 校验指针，并将其转换为pingmesh指针
    CHK_PRT_RET(rpingCtx == nullptr, HCCL_ERROR("[HccnRpingDeinit]rpingCtx is null."), HCCN_E_PARA);
    PingMesh* rping = static_cast<PingMesh*>(rpingCtx);
    // 获取device id
    s32 devLogicId = 0;
    HcclResult ret = hrtGetDeviceRefresh(&devLogicId);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS, HCCL_ERROR("[HccnRpingDeinit]cannot get device logic id, ret[%d].", ret), HCCN_E_FAIL);

    // 判断devLogicId与pingmesh所记录的是否一致
    if (devLogicId != rping->GetDeviceLogicId()) {
        HCCL_ERROR(
            "[HccnRpingDeinit]curr devId[%d] don't match record logicId[%d].", devLogicId, rping->GetDeviceLogicId());
        return HCCN_E_PARA;
    }

    ret = rping->HccnRpingDeinit(static_cast<u32>(devLogicId));
    delete rping; // 先释放资源
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[HccnRpingDeinit]Device[%d] deinit fail, ret[%d]", devLogicId, ret);
        return HCCN_E_FAIL;
    }

    HCCL_RUN_INFO("[HccnRpingDeinit]Device[%d] deinit success", devLogicId);
    return HCCN_SUCCESS;
}

HccnResult HccnRpingAddTarget(HccnRpingCtx rpingCtx, uint32_t targetNum, HccnRpingTargetInfo* target)
{
    HccnRpingAddTargetConfig config;
    config.connectTimeout = HCCN_RPING_DEFAULT_TIMEOUT;
    return HccnRpingAddTargetWithCfg(rpingCtx, targetNum, target, &config);
}

inline HccnResult HccnRpingInitInputTargetAttr(HccnRpingTargetInfo* targetInter, RpingInput* inputInter, uint32_t& n)
{
    u32 addressType = 0;
    HcclResult addrTypeRet = GetAddrType(&addressType);
    if (addrTypeRet != HCCL_SUCCESS) {
        HCCL_ERROR("[HccnRpingInitInputTargetAttr]GetAddrType Fail ret %d", addrTypeRet);
        return HCCN_E_PARA;
    }
    if (addressType == HCCN_RPING_ADDR_TYPE_IP) {
        if (!HcclIpAddress::IsIPv4(std::string(targetInter[n].srcIp))
            && !HcclIpAddress::IsIPv6(std::string(targetInter[n].srcIp))) {
            HCCL_ERROR("[HccnRpingInitInputTargetAttr] invalid source ip.");
            return HCCN_E_PARA;
        }
        inputInter[n].sip = HcclIpAddress(std::string(targetInter[n].srcIp));
        if (!HcclIpAddress::IsIPv4(std::string(targetInter[n].dstIp))
            && !HcclIpAddress::IsIPv6(std::string(targetInter[n].dstIp))) {
            HCCL_ERROR("[HccnRpingInitInputTargetAttr] invalid destination ip.");
            return HCCN_E_PARA;
        }
        inputInter[n].dip = HcclIpAddress(std::string(targetInter[n].dstIp));
    }
    const char* socNamePtr = aclrtGetSocName();
    CHK_PRT_RET(socNamePtr == nullptr, HCCL_ERROR("[HccnRpingInitInputTargetAttr]socNamePtr is null."), HCCN_E_PARA);
    if (addressType == HCCN_RPING_ADDR_TYPE_EID && IsSupportHCCLV2(socNamePtr)) {
        if (!HcclIpAddress::IsEID(std::string(targetInter[n].srcEid))
            && !HcclIpAddress::IsEID(std::string(targetInter[n].dstEid))) {
            HCCL_ERROR("[HccnRpingInitInputTargetAttr] invalid eid");
            return HCCN_E_PARA;
        }
        inputInter[n].sip = HcclIpAddress(HcclIpAddress::StrToEID(std::string(targetInter[n].srcEid)));
        inputInter[n].dip = HcclIpAddress(HcclIpAddress::StrToEID(std::string(targetInter[n].dstEid)));
    }
    return HCCN_SUCCESS;
}

inline HccnResult HccnRpingInitTargetAttr(HccnRpingTargetInfo* targetInter, RpingInput* inputInter, uint32_t& n)
{
    HccnResult ret = HccnRpingInitInputTargetAttr(targetInter, inputInter, n);
    if (ret != HCCN_SUCCESS) {
        HCCL_ERROR("[HccnRpingInitTargetAttr] invalid eid");
        return HCCN_E_PARA;
    }
    inputInter[n].srcPort = targetInter[n].srcPort;
    inputInter[n].sl = targetInter[n].sl;
    inputInter[n].tc = targetInter[n].tc;
    inputInter[n].port = targetInter[n].port;
    inputInter[n].len = targetInter[n].payloadLen;
    inputInter[n].addrType = targetInter[n].addrType;

    return HCCN_SUCCESS;
}

HccnResult HccnRpingAddTargetV2(
    HccnRpingCtx rpingCtx, uint32_t targetNum, HccnRpingTargetInfo* target, HccnRpingAddTargetConfig* config)
{
    return HccnRpingAddTargetWithCfg(rpingCtx, targetNum, target, config);
}

HccnResult HccnRpingAddTargetWithCfg(
    HccnRpingCtx rpingCtx, uint32_t targetNum, HccnRpingTargetInfo* target, HccnRpingAddTargetConfig* config)
{
    // 校验入参
    CHK_PRT_RET(config == nullptr, HCCL_ERROR("[HccnRpingAddTargetWithCfg]config is null."), HCCN_E_PARA);
    // 超时设置判断 小于1ms,大于1h
    if (config->connectTimeout < HCCN_RPING_MIN_TIMEOUT || config->connectTimeout > HCCN_RPING_MAX_TIMEOUT) {
        HCCL_ERROR(
            "[HCCN][HccnRpingAddTargetWithCfg] timeout [%u ms], need to be in the range 1 to 3600000",
            config->connectTimeout);
        return HCCN_E_PARA;
    }
    // 校验入参
    CHK_PRT_RET(target == nullptr, HCCL_ERROR("[HccnRpingAddTargetWithCfg]target is null."), HCCN_E_PARA);
    CHK_PRT_RET(
        targetNum > TARGET_NUM_MAX,
        HCCL_ERROR("[HccnRpingAddTargetWithCfg]targetNum[%u] is more than %u!", targetNum, TARGET_NUM_MAX),
        HCCN_E_PARA);
    // 校验指针，并将其转换为pingmesh指针
    CHK_PRT_RET(rpingCtx == nullptr, HCCL_ERROR("[HccnRpingAddTargetWithCfg]rpingCtx is null."), HCCN_E_PARA);
    HCCL_DEBUG("[HccnRpingAddTargetWithCfg]targetNum:[%u] connectTimeout:[%u ms]", targetNum, config->connectTimeout);
    PingMesh* rping = static_cast<PingMesh*>(rpingCtx);
    // 获取device id
    s32 devLogicId = 0;
    HcclResult ret = hrtGetDeviceRefresh(&devLogicId);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS, HCCL_ERROR("[HccnRpingAddTargetWithCfg]cannot get device logic id, ret[%d].", ret),
        HCCN_E_FAIL);

    // 判断devLogicId与pingmesh所记录的是否一致
    if (devLogicId != rping->GetDeviceLogicId()) {
        HCCL_ERROR(
            "[HccnRpingAddTargetWithCfg]curr devId[%d] don't match record logicId[%d].", devLogicId,
            rping->GetDeviceLogicId());
        return HCCN_E_PARA;
    }
    HCCL_DEBUG("[HccnRpingAddTargetWithCfg] device id is %d.", devLogicId);

    // 将入参转换为内部接口可以使用的类型
    RpingInput* input = new (std::nothrow) RpingInput[targetNum];
    CHK_PRT_RET(input == nullptr, HCCL_ERROR("[HccnRpingAddTargetWithCfg]memory alloc failed."), HCCN_E_MEM);
    for (uint32_t m = 0; m < targetNum; m++) {
        HccnResult res = HccnRpingInitTargetAttr(target, input, m);
        if (res != HCCN_SUCCESS) {
            delete[] input;
            HCCL_ERROR("[HccnRpingAddTargetWithCfg]init target attr fail, ret[%d].", res);
            return HCCN_E_PARA;
        }
        s32 sRet = memcpy_s(input[m].payload, input[m].len, target[m].payload, target[m].payloadLen);
        if (sRet != EOK) {
            HCCL_ERROR(
                "[HccnRpingAddTargetWithCfg]memcpy payload fail. errorno[%d] params:dstMaxSize[%u] srclen[%d]", sRet,
                input[m].len, target[m].payloadLen);
            delete[] input;
            return HCCN_E_MEM;
        }
    }
    ret = rping->HccnRpingAddTarget(static_cast<u32>(devLogicId), targetNum, input, config);
    if (ret != 0) {
        HCCL_ERROR("[HccnRpingAddTarget]Device[%d] add targetNum %u fail, ret[%d]", devLogicId, targetNum, ret);
        delete[] input;
        return HCCN_E_FAIL;
    }

    delete[] input;
    HCCL_RUN_INFO("[HccnRpingAddTargetWithCfg]Device[%d] add targetNum %u success.", devLogicId, targetNum);
    return HCCN_SUCCESS;
}

HccnResult HccnRpingRemoveTarget(HccnRpingCtx rpingCtx, uint32_t targetNum, HccnRpingTargetInfo* target)
{
    // 校验入参
    CHK_PRT_RET(target == nullptr, HCCL_ERROR("[HccnRpingRemoveTarget]target is null."), HCCN_E_PARA);
    CHK_PRT_RET(
        targetNum > TARGET_NUM_MAX,
        HCCL_ERROR("[HccnRpingRemoveTarget]targetNum[%u] is more than %u!", targetNum, TARGET_NUM_MAX), HCCN_E_PARA);
    // 校验指针，并将其转换为pingmesh指针
    CHK_PRT_RET(rpingCtx == nullptr, HCCL_ERROR("[HccnRpingRemoveTarget]rpingCtx is null."), HCCN_E_PARA);
    HCCL_DEBUG("[HccnRpingRemoveTarget]targetNum:%u", targetNum);
    PingMesh* rping = static_cast<PingMesh*>(rpingCtx);
    // 获取device id
    s32 devLogicId = 0;
    HcclResult ret = hrtGetDeviceRefresh(&devLogicId);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS, HCCL_ERROR("[HccnRpingRemoveTarget]cannot get device logic id, ret[%d].", ret),
        HCCN_E_FAIL);

    // 判断devLogicId与pingmesh所记录的是否一致
    if (devLogicId != rping->GetDeviceLogicId()) {
        HCCL_ERROR(
            "[HccnRpingRemoveTarget]curr devId[%d] don't match record logicId[%d].", devLogicId,
            rping->GetDeviceLogicId());
        return HCCN_E_PARA;
    }
    HCCL_DEBUG("[HccnRpingRemoveTarget] device id is %d", devLogicId);

    // 将入参转换为内部接口可以使用的类型
    RpingInput* input = new (std::nothrow) RpingInput[targetNum];
    CHK_PRT_RET(input == nullptr, HCCL_ERROR("[HccnRpingRemoveTarget]memory alloc failed."), HCCN_E_FAIL);
    for (uint32_t in = 0; in < targetNum; in++) {
        HccnResult ret = HccnRpingInitInputTargetAttr(target, input, in);
        if (ret != HCCN_SUCCESS) {
            HCCL_ERROR("[HccnRpingRemoveTarget] invalid eid, ret[%d]", ret);
            delete[] input;
            return HCCN_E_PARA;
        }
        input[in].sl = target[in].sl;
        input[in].tc = target[in].tc;
        input[in].addrType = target[in].addrType;
    }

    ret = rping->HccnRpingRemoveTarget(static_cast<u32>(devLogicId), targetNum, input);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[HccnRpingRemoveTarget]Device[%d] remove targetNum %u fail, ret[%d]", devLogicId, targetNum, ret);
        delete[] input;
        return HCCN_E_FAIL;
    }

    delete[] input;
    HCCL_RUN_INFO("[HccnRpingRemoveTarget]Device[%d] remove targetNum %u success.", devLogicId, targetNum);
    return HCCN_SUCCESS;
}

inline void ConvertTargetState(uint32_t targetNum, int* state, HccnRpingAddTargetState* targetState)
{
    for (uint32_t i = 0; i < targetNum; i++) {
        switch (static_cast<RpingLinkState>(state[i])) {
            case RpingLinkState::CONNECTED:
                targetState[i] = HCCN_RPING_ADDTARGET_STATE_DONE;
                break;
            case RpingLinkState::CONNECTING:
                targetState[i] = HCCN_RPING_ADDTARGET_STATE_DOING;
                break;
            case RpingLinkState::DISCONNECTED:
                targetState[i] = HCCN_RPING_ADDTARGET_STATE_FAIL;
                break;
            case RpingLinkState::TIMEOUT:
                targetState[i] = HCCN_RPING_ADDTARGET_STATE_TIMEOUT;
                break;
            default:
                targetState[i] = HCCN_RPING_ADDTARGET_STATE_RESERVED;
                break;
        }
    }
}

HccnResult HccnRpingGetTarget(
    HccnRpingCtx rpingCtx, uint32_t targetNum, HccnRpingTargetInfo* target, HccnRpingAddTargetState* targetState)
{
    // 校验入参
    CHK_PRT_RET(target == nullptr, HCCL_ERROR("[HccnRpingGetTarget]target is null."), HCCN_E_PARA);
    CHK_PRT_RET(targetState == nullptr, HCCL_ERROR("[HccnRpingGetTarget]targetState is null."), HCCN_E_PARA);
    CHK_PRT_RET(
        targetNum > TARGET_NUM_MAX,
        HCCL_ERROR("[HccnRpingGetTarget]targetNum[%u] is more than %u!", targetNum, TARGET_NUM_MAX), HCCN_E_PARA);
    // 校验指针，并将其转换为pingmesh指针
    CHK_PRT_RET(rpingCtx == nullptr, HCCL_ERROR("[HccnRpingGetTarget]rpingCtx is null."), HCCN_E_PARA);
    HCCL_DEBUG("[HccnRpingGetTarget]targetNum:%u", targetNum);
    PingMesh* rping = static_cast<PingMesh*>(rpingCtx);
    // 获取device id
    s32 devLogicId = 0;
    HcclResult ret = hrtGetDeviceRefresh(&devLogicId);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS, HCCL_ERROR("[HccnRpingGetTarget]cannot get device logic id, ret[%d].", ret), HCCN_E_FAIL);

    // 判断devLogicId与pingmesh所记录的是否一致
    if (devLogicId != rping->GetDeviceLogicId()) {
        HCCL_ERROR(
            "[HccnRpingGetTarget]curr devId[%d] don't match record logicId[%d].", devLogicId,
            rping->GetDeviceLogicId());
        return HCCN_E_PARA;
    }
    HCCL_DEBUG("[HccnRpingGetTarget] device id is %d", devLogicId);

    // 将入参转换为内部接口可以使用的类型
    RpingInput* input = new (std::nothrow) RpingInput[targetNum];
    CHK_PRT_RET(input == nullptr, HCCL_ERROR("[HccnRpingGetTarget]memory alloc failed."), HCCN_E_MEM);
    for (uint32_t h = 0; h < targetNum; h++) {
        HccnResult ret = HccnRpingInitInputTargetAttr(target, input, h);
        if (ret != HCCN_SUCCESS) {
            HCCL_ERROR("[HccnRpingGetTarget] invalid eid");
            delete[] input;
            return HCCN_E_FAIL;
        }
        input[h].addrType = target[h].addrType;
    }
    int* state = new (std::nothrow) int[targetNum];
    if (state == nullptr) {
        HCCL_ERROR("[HccnRpingGetTarget]memory alloc failed.");
        delete[] input;
        return HCCN_E_MEM;
    }
    ret = rping->HccnRpingGetTarget(static_cast<u32>(devLogicId), targetNum, input, state);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[HccnRpingGetTarget]Device[%d] get targetNum %u fail, ret[%d]", devLogicId, targetNum, ret);
        delete[] input;
        delete[] state;
        return HCCN_E_FAIL;
    }

    ConvertTargetState(targetNum, state, targetState);
    delete[] input;
    delete[] state;
    HCCL_RUN_INFO("[HccnRpingGetTarget]Device[%d] get targetNum %u success.", devLogicId, targetNum);
    return HCCN_SUCCESS;
}

HccnResult HccnRpingBatchPingStart(HccnRpingCtx rpingCtx, uint32_t pktNum, uint32_t interval, uint32_t timeout)
{
    // 校验指针，并将其转换为pingmesh指针
    CHK_PRT_RET(rpingCtx == nullptr, HCCL_ERROR("[HccnRpingBatchPingStart]rpingCtx is null.", rpingCtx), HCCN_E_PARA);
    HCCL_DEBUG("[HccnRpingBatchPingStart]pktNum:[%u], interval:[%u s], timeout:[%u s]", pktNum, interval, timeout);
    PingMesh* rping = static_cast<PingMesh*>(rpingCtx);
    // 获取device id
    s32 devLogicId = 0;
    HcclResult ret = hrtGetDeviceRefresh(&devLogicId);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS, HCCL_ERROR("[HccnRpingBatchPingStart]cannot get device logic id, ret[%d].", ret),
        HCCN_E_PARA);

    // 判断devLogicId与pingmesh所记录的是否一致
    if (devLogicId != rping->GetDeviceLogicId()) {
        HCCL_ERROR(
            "[HccnRpingBatchPingStart]curr devId[%d] don't match record logicId[%d].", devLogicId,
            rping->GetDeviceLogicId());
        return HCCN_E_PARA;
    }

    ret = rping->HccnRpingBatchPingStart(static_cast<u32>(devLogicId), pktNum, interval, timeout);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[HccnRpingBatchPingStart]task start failed, devLogicId[%d], ret[%d], pktNum:[%u], interval:[%u s], "
            "timeout:[%u s].",
            devLogicId, ret, pktNum, interval, timeout),
        HCCN_E_FAIL);

    HCCL_RUN_INFO(
        "[HccnRpingBatchPingStart]task start success, devLogicId:[%d], pktNum:[%u], interval:[%u s], timeout:[%u s].",
        devLogicId, pktNum, interval, timeout);
    return HCCN_SUCCESS;
}

HccnResult HccnRpingBatchPingStop(HccnRpingCtx rpingCtx)
{
    // 校验指针，并将其转换为pingmesh指针
    CHK_PRT_RET(rpingCtx == nullptr, HCCL_ERROR("[HccnRpingBatchPingStop]rpingCtx is null.", rpingCtx), HCCN_E_PARA);
    PingMesh* rping = static_cast<PingMesh*>(rpingCtx);
    // 获取device id
    s32 devLogicId = 0;
    HcclResult ret = hrtGetDeviceRefresh(&devLogicId);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS, HCCL_ERROR("[HccnRpingBatchPingStop]cannot get device logic id, ret[%d].", ret),
        HCCN_E_PARA);

    // 判断devLogicId与pingmesh所记录的是否一致
    if (devLogicId != rping->GetDeviceLogicId()) {
        HCCL_ERROR(
            "[HccnRpingBatchPingStop]curr devId[%d] don't match record logicId[%d].", devLogicId,
            rping->GetDeviceLogicId());
        return HCCN_E_PARA;
    }

    ret = rping->HccnRpingBatchPingStop(static_cast<u32>(devLogicId));
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[HccnRpingBatchPingStop]task stop failed, devLogicId[%d] ret[%d].", devLogicId, ret), HCCN_E_FAIL);

    HCCL_RUN_INFO("[HccnRpingBatchPingStop]task stop success, devLogicId[%d].", devLogicId);
    return HCCN_SUCCESS;
}

inline void ConvertResultState(uint32_t state, HccnRpingResultState& resultState)
{
    if (state == RPING_RESULT_STATE_VALID) {
        resultState = HCCN_RPING_RESULT_STATE_VALID;
    } else {
        resultState = HCCN_RPING_RESULT_STATE_INVALID;
    }
}

inline HccnResult ResultGetTarget(uint32_t& targetNumInter, HccnRpingTargetInfo* targetInter, RpingInput* inputInter)
{
    for (uint32_t k = 0; k < targetNumInter; k++) {
        HccnResult ret = HccnRpingInitInputTargetAttr(targetInter, inputInter, k);
        if (ret != HCCN_SUCCESS) {
            HCCL_ERROR("[ResultGetTarget] invalid eid, ret[%d].", ret);
            return HCCN_E_PARA;
        }
        inputInter[k].addrType = targetInter[k].addrType;
    }
    return HCCN_SUCCESS;
}

inline void PutResult(uint32_t& targetNumInter, HccnRpingResultInfo* resultInter, RpingOutput* outputInter)
{
    for (uint32_t i = 0; i < targetNumInter; i++) {
        ConvertResultState(outputInter[i].state, resultInter[i].state);
        if (outputInter[i].state != PingResultState::PING_RESULT_STATE_VALID) {
            HCCL_INFO("[HccnRpingGetResult]Target[%u]'s state is not valid, state[%d].", i, outputInter[i].state);
            continue;
        }
        resultInter[i].txPkt = outputInter[i].txPkt;
        resultInter[i].rxPkt = outputInter[i].rxPkt;
        resultInter[i].minRTT = outputInter[i].minRTT;
        resultInter[i].maxRTT = outputInter[i].maxRTT;
        resultInter[i].avgRTT = outputInter[i].avgRTT;
    }
}

HccnResult
HccnRpingGetResult(HccnRpingCtx rpingCtx, uint32_t targetNum, HccnRpingTargetInfo* target, HccnRpingResultInfo* result)
{
    // 校验入参
    CHK_PRT_RET(target == nullptr, HCCL_ERROR("[HccnRpingGetResult]target is null."), HCCN_E_PARA);
    CHK_PRT_RET(result == nullptr, HCCL_ERROR("[HccnRpingGetResult]result is null."), HCCN_E_PARA);
    CHK_PRT_RET(
        targetNum > TARGET_NUM_MAX,
        HCCL_ERROR("[HccnRpingGetResult]targetNum[%u] is more than %u!", targetNum, TARGET_NUM_MAX), HCCN_E_PARA);
    // 校验指针，并将其转换为pingmesh指针
    CHK_PRT_RET(rpingCtx == nullptr, HCCL_ERROR("[HccnRpingGetResult]rpingCtx is null."), HCCN_E_PARA);
    HCCL_DEBUG("[HccnRpingGetResult]targetNum:%u", targetNum);
    PingMesh* rping = static_cast<PingMesh*>(rpingCtx);
    // 获取device id
    s32 devLogicId = 0;
    HcclResult ret = hrtGetDeviceRefresh(&devLogicId);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS, HCCL_ERROR("[HccnRpingGetResult]cannot get device logic id, ret[%d].", ret), HCCN_E_FAIL);

    // 判断devLogicId与pingmesh所记录的是否一致
    if (devLogicId != rping->GetDeviceLogicId()) {
        HCCL_ERROR("[HccnRpingGetResult]curr devId[%d] not match record[%d].", devLogicId, rping->GetDeviceLogicId());
        return HCCN_E_PARA;
    }

    // 将入参转换为内部接口可以使用的类型
    RpingInput* input = new (std::nothrow) RpingInput[targetNum];
    CHK_PRT_RET(input == nullptr, HCCL_ERROR("[HccnRpingGetResult]memory alloc failed."), HCCN_E_MEM);
    RpingOutput* output = new (std::nothrow) RpingOutput[targetNum];
    if (output == nullptr) {
        delete[] input;
        HCCL_ERROR("[HccnRpingGetResult]memory alloc failed.");
        return HCCN_E_MEM;
    }
    HccnResult res = ResultGetTarget(targetNum, target, input);
    if (res != HCCN_SUCCESS) {
        delete[] input;
        delete[] output;
        return HCCN_E_FAIL;
    }
    ret = rping->HccnRpingGetResult(static_cast<u32>(devLogicId), targetNum, input, output);
    if (ret == HCCL_E_AGAIN) {
        delete[] input;
        delete[] output;
        return HCCN_E_AGAIN;
    }
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[HccnRpingGetResult]Device[%d] get result fail, targetNum[%u] ret[%d]", devLogicId, targetNum, ret);
        delete[] input;
        delete[] output;
        return HCCN_E_FAIL;
    }
    PutResult(targetNum, result, output);
    delete[] input;
    delete[] output;
    HCCL_RUN_INFO("[HccnRpingGetResult]Device[%d] get result success, targetNum[%u].", devLogicId, targetNum);
    return HCCN_SUCCESS;
}

HccnResult HccnRpingGetPayload(HccnRpingCtx rpingCtx, void** payload, uint32_t* payloadLen)
{
    // 校验入参
    CHK_PRT_RET(payload == nullptr, HCCL_ERROR("[HccnRpingGetPayload]payload is null."), HCCN_E_PARA);
    CHK_PRT_RET(payloadLen == nullptr, HCCL_ERROR("[HccnRpingGetPayload]payloadLen is null."), HCCN_E_PARA);
    // 校验指针，并将其转换为pingmesh指针
    CHK_PRT_RET(rpingCtx == nullptr, HCCL_ERROR("[HccnRpingGetPayload]rpingCtx is null."), HCCN_E_PARA);
    PingMesh* rping = static_cast<PingMesh*>(rpingCtx);
    // 获取device id
    s32 devLogicId = 0;
    HcclResult ret = hrtGetDeviceRefresh(&devLogicId);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS, HCCL_ERROR("[HccnRpingGetPayload]cannot get device logic id, ret[%d].", ret), HCCN_E_FAIL);

    // 判断devLogicId与pingmesh所记录的是否一致
    if (devLogicId != rping->GetDeviceLogicId()) {
        HCCL_ERROR("[HccnRpingGetPayload]curr devId[%d] not match record[%d].", devLogicId, rping->GetDeviceLogicId());
        return HCCN_E_PARA;
    }

    ret = rping->HccnRpingGetPayload(static_cast<u32>(devLogicId), payload, payloadLen, rping->GetMode());
    CHK_PRT_RET(
        ret != HCCL_SUCCESS, HCCL_ERROR("[HccnRpingGetPayload]Device[%d] get payload fail, ret[%d]", devLogicId, ret),
        HCCN_E_FAIL);

    HCCL_RUN_INFO("[HccnRpingGetPayload]Device[%d] get payload success, payloadLen[%u].", devLogicId, *payloadLen);
    return HCCN_SUCCESS;
}

#ifdef __cplusplus
}
#endif // __cplusplus
