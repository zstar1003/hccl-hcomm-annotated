/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccuTaskException.h"
#include "ccu_dfx_schema.h"
#include <algorithm>
#include <memory>
#include "log.h"
#include "comm_addr_logger.h"
#include "coll_comm.h"
#include <adapter_error_manager_pub.h>
#include "hccl_log_keywords.h"
#include "task_param.h"
#include "ccu_kernel_mgr.h"
#include "hcomm_c_adpt.h"
#include "../../endpoint_pairs/channels/ccu/ccu_urma_channel.h"
#include "orion_adpt_utils.h"
#include "orion_adapter_hccp.h"
#include "aubdfx_api.h"
#include "hcomm_adapter_hccp.h"
#include "hccp_tlv_hdc_manager.h"
#include "adapter_rts_common.h"
#include "ccu_rep_loopgroup_bundle_v1.h"
#include "ccu_rep_type_v1.h"
#include "ccu_rep_loc_record_event.h"
#include "ccu_rep_v1.h"
#include "ccu_comp.h"
#include "string_util.h"

#include "net_instance.h"
#include "hccl_communicator.h"

#include "sal.h"

namespace hcomm {

using namespace std;
constexpr int BYTE = 8;
constexpr size_t CCU_CTX_RAW_CAPACITY = 64;
constexpr uint64_t CCU_MSG_256MB_LEN = 256ULL * 1024 * 1024; // CCU消息长度不能大于256MB
constexpr uint16_t INVALID_U16 = 65535;
constexpr uint8_t CCUM_EXECUTE_ERROR = 0X09;
constexpr uint8_t CCU_MISSION_TASK_KILLED = 0X02;

const map<uint8_t, string> MISSION_STATUS_MAP{
    {0x01, "Unsupported Opcode(0x01)"},      {0x02, "Local Operation Error(0x02)"},
    {0x03, "Remote Operation Error(0x03)"},  {0x04, "Transaction Retry Counter Exceeded(0x04)"},
    {0x05, "Transaction ACK Timeout(0x05)"}, {0x06, "Jetty Work Request Flushed(0x06)"},
    {0x07, "CCUA Alg Task Error(0x07)"},     {0x08, "Memory ECC Error(0x08)"},
    {0x09, "CCUM Execute Error(0x09)"},      {0x0A, "CCUA Execute Error(0x0A)"},
};

const map<uint8_t, map<uint8_t, string>> MISSION_SUB_STATUS_MAP{
    {0x02,
     {{0x01, "Local Length Error(0x01)"},
      {0x02, "Local Access Error(0x02)"},
      {0x03, "Remote Response Length Error(0x03)"},
      {0x04, "Local Data Poison(0x04)"}}},
    {0x03,
     {{0x01, "Remote Unsupported Request(0x01)"},
      {0x02, "Remote Access Abort(0x02)"},
      {0x04, "Remote Data Poison(0x04)"}}},
    {0x07,
     {{0x01, "Overflow(0x01)"},
      {0x02, "Underflow(0x02)"},
      {0x04, "NaN(0x04)"},
      {0x08, "Inf(0x08)"},
      {0x09, "Inf Overflow(0x09)"}}},
    {0x09, {{0x01, "SQE instr and key not match(0x01)"}, {0x02, "CCU Mission Task Killed(0x02)"}}},
    {0x0A,
     {{0x01, "EXOKAY(0x01)"},
      {0x11, "EXOKAY(0x11)"},
      {0x02, "SLVERR(0x02)"},
      {0x12, "SLVERR(0x12)"},
      {0x03, "DECERR(0x03)"},
      {0x13, "DECERR(0x13)"},
      {0x04, "Abort(0x04)"},
      {0x14, "Abort(0x14)"},
      {0x05, "Write Permission Err(0x05)"},
      {0x15, "Write Permission Err(0x15)"},
      {0x06, "Read Permission Err(0x06)"},
      {0x16, "Read Permission Err(0x16)"},
      {0x07, "Atomic Permission Err(0x07)"},
      {0x17, "Atomic Permission Err(0x17)"},
      {0x08, "Tokenval Err(0x08)"},
      {0x18, "Tokenval Err(0x18)"},
      {0x09, "Page Fault(0x09)"},
      {0x0a, "Page Fault(0x0A)"},
      {0x0b, "Page Fault(0x0B)"},
      {0x19, "Page Fault(0x19)"},
      {0x1a, "Page Fault(0x1A)"},
      {0x1b, "Page Fault(0x1B)"},
      {0x0c, "Read Local Mem Poison(0x0C)"}}},
};

const unordered_set<CcuRep::CcuRepType> REP_WITH_CHANNEL = {
    {CcuRep::CcuRepType::REM_POST_SEM}, {CcuRep::CcuRepType::REM_WAIT_SEM}, {CcuRep::CcuRepType::REM_POST_VAR},
    {CcuRep::CcuRepType::READ},         {CcuRep::CcuRepType::WRITE},        {CcuRep::CcuRepType::BUF_READ},
    {CcuRep::CcuRepType::BUF_WRITE},
};

MAKE_ENUM(AuxInfoInType, AUX_INFO_IN_TYPE_CQE, AUX_INFO_IN_TYPE_AE, AUX_INFO_IN_TYPE_MAX);
struct AuxInfoIn {
    AuxInfoInType auxInfoInType;
    union {
        struct {
            uint32_t status;
            uint8_t sR;
        } cqe;
        struct {
            uint32_t eventType;
        } ae;
    };
    u8 resv[7];
};

constexpr u32 MAX_AUX_INFO_NUM = 256;
struct AuxInfoOut {
    uint32_t auxInfoTypes[MAX_AUX_INFO_NUM];
    uint32_t auxInfoValues[MAX_AUX_INFO_NUM];
    uint32_t auxInfoNum{0};
};

std::mutex g_channelMapMutex;
std::unordered_map<uint16_t, uint64_t> g_channelIdToHandle;
std::atomic<bool> isGetCqeErrInfo(
    false); // 是否已获取过CQE错误信息，获取过后再次获取的概率较小，且获取过程可能较慢，因此设置标志位避免重复获取
CcuGetErrStatusVecCallBack g_CcuGetErrStatusVecCallBack = nullptr;
GetCcuCqeErrInfoCallBackHcomm g_getCqeErrInfoCallBack = nullptr;

void RegisterCcuGetErrStatusVecCallBack(CcuGetErrStatusVecCallBack callback)
{
    g_CcuGetErrStatusVecCallBack = callback;
    return;
}

std::vector<std::string> CcuGetErrStatusVec(s32 deviceLogicID)
{
    if (g_CcuGetErrStatusVecCallBack != nullptr) {
        return g_CcuGetErrStatusVecCallBack(deviceLogicID);
    } else {
        HCCL_RUN_WARNING("[GetErrStatusVec]g_CcuGetErrStatusVecCallBack is nullptr.");
    }
    return std::vector<std::string>();
}

std::string CcuGetAndPrintClusterMonitorErr(const u32 deviceId)
{
    auto errStatusVec = CcuGetErrStatusVec(deviceId);
    std::string errMsg = "";
    int errSize = errStatusVec.size();
    if (errSize > 0) {
        int maxListSize = 3; // 放入errMsg中的异常事件最多只有3个
        if (errSize <= maxListSize) {
            errMsg = "\nthere are(is) " + std::to_string(errSize) + " abnormal device(s):\n";
        } else {
            errMsg = "\nthere are " + std::to_string(errSize) + " abnormal device(s), "
                     + "only the first 3 devices are listed:\n";
        }

        for (int i = 0; i < errSize; i++) {
            HCCL_ERROR(
                "[%s][%s][%s] %s", LOG_KEYWORDS_TASK_EXEC.c_str(), LOG_KEYWORDS_RUN_FAILED.c_str(),
                LOG_KEYWORDS_CCU.c_str(), errStatusVec[i].c_str());
            if (i < maxListSize) {
                errMsg += ("\t" + errStatusVec[i] + "\n");
            }
        }
    }
    return errMsg;
}

void RegisterGetCcuCqeErrInfoCallBackHcomm(GetCcuCqeErrInfoCallBackHcomm p1)
{
    g_getCqeErrInfoCallBack = p1;
    return;
}
void CcuTaskException::ClusterMoniterGetCcuCqeErrInfo(
    u32 RemoteDeviceId, u32 locDeviceId, uint16_t status, std::string LocalEid, std::string RemoteEid,
    std::string RemoteInsId)
{
    if (g_getCqeErrInfoCallBack != nullptr) {
        g_getCqeErrInfoCallBack(RemoteDeviceId, locDeviceId, status, LocalEid, RemoteEid, RemoteInsId);
    }
    return;
}

static void PrintPanicLogWithOps(const uint8_t* panicLog, const CcuVersionOps* ops)
{
    if (panicLog == nullptr) {
        HCCL_ERROR("[CcuTaskException][PrintPanicLogWithOps] panicLog is nullptr.");
        return;
    }
    if (ops == nullptr || ops->printCcumDfxInfo == nullptr) {
        HCCL_ERROR("[CcuTaskException][PrintPanicLogWithOps] ops or printCcumDfxInfo is nullptr.");
        return;
    }
    std::ostringstream oss;
    oss << "[CCU DFX][ops=" << ops->name << "] CCU DFX INFO:";
    ops->printCcumDfxInfo(panicLog, oss);
    std::string logStr = oss.str();
    HCCL_ERROR("%s", logStr.c_str());
}

void CcuTaskException::ProcessCcuException(const rtExceptionInfo_t* exceptionInfo, const Hccl::TaskInfo& taskInfo)
{
    auto deviceId = exceptionInfo->deviceid;
    HCCL_ERROR("[CcuTaskException][%s]Task from HCCL run failed.", __func__);
    HCCL_ERROR(
        "[CcuTaskException]Task run failed, base information is deviceID:[%u], %s.", deviceId,
        taskInfo.GetIndopBaseInfo().c_str());
    HCCL_ERROR("[CcuTaskException]Task run failed, groupRank information is %s.", GetGroupRankInfo(taskInfo).c_str());
    HCCL_ERROR("[CcuTaskException]Task run failed, opData information is %s.", taskInfo.GetIndopDataInfo().c_str());
    CHK_PRT(InitChannelMap(deviceId, taskInfo.taskParam_.taskPara.Ccu.ccuKernelHandle));
    auto& ccuExDetailInfo = exceptionInfo->expandInfo.u.ccuInfo;
    isGetCqeErrInfo = true;

    const CcuVersionOps* ops = nullptr;
    if (GetCcuOps(ops) != HCCL_SUCCESS) {
        ops = nullptr;
    }

    for (uint32_t i = 0; i < ccuExDetailInfo.ccuMissionNum; ++i) { // ccuExDetailInfo.ccuMissionNum为1
        const auto& missionInfo = ccuExDetailInfo.missionInfo[i];  // 异常mission
        uint16_t status = static_cast<uint16_t>(missionInfo.status) << BYTE | missionInfo.subStatus;
        PrintCcuErrorInfo(deviceId, status, taskInfo);
        // 打印寄存器信息
        PrintPanicLogWithOps(missionInfo.panicLog, ops);
    }

    const int32_t devLogicId = static_cast<int32_t>(deviceId);
    if (CcuComponent::GetInstance(devLogicId).CleanTaskKillState() != HcclResult::HCCL_SUCCESS) {
        HCCL_ERROR("[CcuTaskException][%s] failed to clean ccu task kill state, devLogicId[%d].", __func__, devLogicId);
    }

    const uint8_t dieId = taskInfo.taskParam_.taskPara.Ccu.dieId;
    if (CcuComponent::GetInstance(devLogicId).CleanDieCkes(dieId) != HcclResult::HCCL_SUCCESS) {
        HCCL_ERROR(
            "[CcuTaskException][%s] failed to clean ccu die ckes, dieId[%u], devLogicId[%d].", __func__, dieId,
            devLogicId);
    }
}

HcclResult CcuTaskException::InitChannelMap(s32 deviceId, u64 ccuKernelHandle)
{
    std::lock_guard<std::mutex> lock(g_channelMapMutex);

    auto& kernelMgr = hcomm::CcuKernelMgr::GetInstance(deviceId);
    auto* kernel = kernelMgr.GetKernel(ccuKernelHandle);
    CHK_PRT_RET(
        kernel == nullptr,
        HCCL_ERROR("[%s]GetKernel nullptr, deviceId[%d], ccuKernelHandle[0x%llx]", __func__, deviceId, ccuKernelHandle),
        HCCL_E_PARA);

    const auto& ccuChannels = kernel->GetChannels();
    for (const auto& it : ccuChannels) {
        void* channelPtr = nullptr;
        CHK_RET(static_cast<HcclResult>(HcommChannelGet(it, &channelPtr)));
        auto* channelImpl = dynamic_cast<CcuUrmaChannel*>(static_cast<Channel*>(channelPtr));
        CHK_PTR_NULL(channelImpl);
        g_channelIdToHandle[channelImpl->GetChannelId()] = it;
        HCCL_RUN_INFO(
            "[%s]deviceId[%d], ccuKernelHandle[0x%llx], channelId[%u], channelHandle[0x%llx]", __func__, deviceId,
            ccuKernelHandle, channelImpl->GetChannelId(), it);
    }
    return HCCL_SUCCESS;
}

std::string CcuTaskException::GetGroupRankInfo(const Hccl::TaskInfo& taskInfo)
{
    if (taskInfo.dfxOpInfo_ == nullptr || taskInfo.dfxOpInfo_->comm_ == nullptr) {
        HCCL_ERROR("[TaskInfo][%s]TaskInfo communicator is nullptr.", __func__);
        return "";
    }

    hccl::CollComm* communicator = static_cast<hccl::CollComm*>(taskInfo.dfxOpInfo_->comm_);
    return Hccl::StringFormat(
        "group:[%s], rankSize[%u], rankId[%u]", communicator->GetCommId().c_str(), communicator->GetRankSize(),
        communicator->GetMyRankId());
}

void CcuTaskException::PrintPanicLogInfo(const uint8_t* panicLog)
{
    if (panicLog == nullptr) {
        HCCL_ERROR("[CcuTaskException][%s] panicLog is nullptr.", __func__);
        return;
    }

    const CcuVersionOps* ops = nullptr;
    if (GetCcuOps(ops) != HCCL_SUCCESS) {
        HCCL_ERROR("[CcuTaskException][%s] Failed to get printer ops for ccum_dfxInfo", __func__);
        return;
    }
    PrintPanicLogWithOps(panicLog, ops);
}

namespace {
    HcclResult QueryCcuCtxRaw(
        int32_t deviceId, uint32_t dieId, uint32_t ctxId, CcuOpcodeType op, const char* tag, uint8_t* buf,
        size_t bufLen, size_t& copiedLen)
    {
        copiedLen = 0;
        if (buf == nullptr || bufLen < CCU_CTX_RAW_CAPACITY) {
            HCCL_ERROR("[%s] invalid buffer: buf=%p, bufLen=%zu, required=%zu", tag, buf, bufLen, CCU_CTX_RAW_CAPACITY);
            return HCCL_E_PARA;
        }
        const auto memsetRet = memset_s(buf, bufLen, 0, bufLen);
        if (memsetRet != EOK) {
            HCCL_ERROR("[%s] memset_s failed, ret[%d]", tag, memsetRet);
            return HCCL_E_INTERNAL;
        }

        auto tlvHandle = Hccl::HccpTlvHdcManager::GetInstance().GetTlvHandle(deviceId);
        CHK_PRT_RET(
            tlvHandle == nullptr, HCCL_ERROR("[%s]tlvHandle is null, deviceId[%d]", __func__, deviceId), HCCL_E_PTR);

        CustomChannelInfoIn inBuff{};
        CustomChannelInfoOut outBuff{};
        inBuff.op = op;
        inBuff.data.dataInfo.udieIdx = dieId;
        inBuff.offsetStartIdx = ctxId;
        inBuff.data.dataInfo.dataArraySize = 1; // 读1个MissionContext
        inBuff.data.dataInfo.dataLen = CCU_CTX_RAW_CAPACITY;

        HcclResult ret = HccpRaTlvRequestForCustomChannel(
            tlvHandle, MSG_TYPE_CCU_DISPATCH_CMD, static_cast<void*>(&inBuff), static_cast<void*>(&outBuff));

        CHK_PRT_RET(
            ret != HCCL_SUCCESS, HCCL_ERROR("[%s]HccpRaTlvRequestForCustomChannel fail, ret[%u]", tag, ret), ret);
        const auto sret = memcpy_s(buf, bufLen, outBuff.data.dataInfo.dataArray, inBuff.data.dataInfo.dataLen);
        CHK_PRT_RET(sret != EOK, HCCL_ERROR("[%s]memcpy_s failed, ret[%d]", tag, sret), HCCL_E_MEMORY);
        copiedLen = inBuff.data.dataInfo.dataLen;
        return HCCL_SUCCESS;
    }
} // namespace

using GetMissionRawFunc = HcclResult (*)(
    int32_t deviceId, uint32_t dieId, uint32_t missionId, uint8_t* buf, size_t bufLen, size_t& copiedLen);
using GenErrorInfoLoopGroupFunc = HcclResult (*)(
    const ErrorInfoBase& baseInfo, std::shared_ptr<CcuRep::CcuRepBase> repBase, CcuRep::CcuRepContext& ctx,
    std::vector<CcuErrorInfo>& errorInfo);
using GenErrorInfoByRepTypeFunc = void (*)(
    const ErrorInfoBase& baseInfo, std::shared_ptr<CcuRep::CcuRepBase> repBase, std::vector<CcuErrorInfo>& errorInfo);

static HcclResult DecodeRawMissionInfo(const uint8_t* raw, CcuMissionInfo& out)
{
    const CcuVersionOps* ops = nullptr;
    if (GetCcuOps(ops) != HCCL_SUCCESS) {
        HCCL_ERROR("[DecodeRawMissionInfo] Failed to get ops for mission context");
        return HCCL_E_INTERNAL;
    }
    if (ops->getMissionInfo == nullptr) {
        HCCL_ERROR("[DecodeRawMissionInfo] getMissionInfo is nullptr, ops[%s]", ops->name);
        return HCCL_E_INTERNAL;
    }
    return ops->getMissionInfo(raw, &out);
}

static HcclResult FetchAndDecodeLoopInfo(int32_t deviceId, uint32_t dieId, uint32_t loopCtxId, CcuLoopInfo& out)
{
    uint8_t loopRaw[CCU_CTX_RAW_CAPACITY] = {0};
    size_t loopRawLen = 0;
    if (QueryCcuCtxRaw(
            deviceId, dieId, loopCtxId, CcuOpcodeType::CCU_U_OP_GET_LOOP_CTX, "FetchAndDecodeLoopInfo", loopRaw,
            sizeof(loopRaw), loopRawLen)
        != HCCL_SUCCESS) {
        HCCL_ERROR("[FetchAndDecodeLoopInfo] Failed to fetch loop raw context, deviceId[%d]", deviceId);
        return HCCL_E_INTERNAL;
    }
    const CcuVersionOps* ops = nullptr;
    if (GetCcuOps(ops) != HCCL_SUCCESS) {
        HCCL_ERROR("[FetchAndDecodeLoopInfo] Failed to get ops for loop context");
        return HCCL_E_INTERNAL;
    }
    if (ops->getLoopInfo == nullptr) {
        HCCL_ERROR("[FetchAndDecodeLoopInfo] getLoopInfo is nullptr, ops[%s]", ops->name);
        return HCCL_E_INTERNAL;
    }
    return ops->getLoopInfo(loopRaw, &out);
}

// 计算报错指令附近可用 Rep 的起始指令，提取自 GetCcuErrorMsg 以降低其圈复杂度
static uint16_t FindSurroundingBeginInstr(CcuRep::CcuRepContext& ctx, uint16_t startIns, uint16_t currIns)
{
    uint16_t loopUpInstrNum = 10; // 出错指令前 10 条
    uint16_t beginIns = (currIns < loopUpInstrNum) ?
                            startIns :
                            ((currIns - loopUpInstrNum) > startIns ? (currIns - loopUpInstrNum) : startIns);
    // 从第一个非空 rep 开始，使用有符号变量避免 uint16_t 下溢无限循环
    for (int32_t instrId = static_cast<int32_t>(currIns); instrId >= static_cast<int32_t>(beginIns); instrId--) {
        if (ctx.GetRepByInstrId(static_cast<uint16_t>(instrId)) == nullptr) {
            beginIns = static_cast<uint16_t>(instrId) + 1U;
            break;
        }
    }
    return beginIns;
}

static HcclResult ValidateAndDecodeMissionContext(
    int32_t deviceId, uint16_t missionStatus, const Hccl::ParaCcu& ccuTaskParam, GetMissionRawFunc getMissionRaw,
    CcuMissionInfo& missionInfo)
{
    CHK_PRT_RET(
        (deviceId < 0 || static_cast<u32>(deviceId) >= MAX_MODULE_DEVICE_NUM),
        HCCL_ERROR("[CcuTaskException][GetCcuErrorMsg]deviceId[%d] error.", deviceId), HcclResult::HCCL_E_PARA);

    if (missionStatus == 0) {
        HCCL_ERROR(
            "[CcuErrorHandler][%s] no err found, mission status is 0, deviceId[%d], dieId[%u], execMissionId[%u]",
            __func__, deviceId, static_cast<u32>(ccuTaskParam.dieId), static_cast<u32>(ccuTaskParam.execMissionId));
        return HCCL_E_PARA;
    }

    uint8_t missionRaw[CCU_CTX_RAW_CAPACITY] = {0};
    size_t missionRawLen = 0;
    if (getMissionRaw(
            deviceId, ccuTaskParam.dieId, ccuTaskParam.execMissionId, missionRaw, sizeof(missionRaw), missionRawLen)
        != HCCL_SUCCESS) {
        HCCL_ERROR("[CcuErrorHandler][%s] Failed to fetch mission raw context, deviceId[%d]", __func__, deviceId);
        return HCCL_E_INTERNAL;
    }
    if (DecodeRawMissionInfo(missionRaw, missionInfo) != HCCL_SUCCESS) {
        HCCL_ERROR("[CcuErrorHandler][%s] Failed to decode mission info, deviceId[%d]", __func__, deviceId);
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

static HcclResult ResolveRepContextAndCurrentRep(
    int32_t deviceId, const Hccl::ParaCcu& ccuTaskParam, uint16_t currIns, CcuRep::CcuRepContext*& ctx,
    std::shared_ptr<CcuRep::CcuRepBase>& rep, std::shared_ptr<CcuRep::CcuRepBase>& prevRep)
{
    auto& kernelMgr = hcomm::CcuKernelMgr::GetInstance(deviceId);
    auto* kernel = kernelMgr.GetKernel(ccuTaskParam.ccuKernelHandle);
    CHK_PRT_RET(
        kernel == nullptr,
        HCCL_ERROR(
            "[%s]GetKernel nullptr, deviceId[%d], ccuKernelHandle[0x%llx]", __func__, deviceId,
            ccuTaskParam.ccuKernelHandle),
        HCCL_E_PARA);

    ctx = reinterpret_cast<CcuRep::CcuRepContext*>(kernel);
    CHK_PRT_RET(
        ctx == nullptr,
        HCCL_ERROR(
            "CcuContext not found, deviceId[%d], dieId[%u], missionId[%u], executeId[%llu]", deviceId,
            static_cast<u32>(ccuTaskParam.dieId), static_cast<u32>(ccuTaskParam.missionId), ccuTaskParam.executeId),
        HCCL_E_PARA);

    rep = ctx->GetRepByInstrId(currIns);
    CHK_PRT_RET(
        rep == nullptr,
        HCCL_ERROR("[CcuErrorHandler][%s] cannot find REP from current CcuContext, instrId[%u]", __func__, currIns),
        HCCL_E_PARA);

    prevRep = nullptr;
    if (currIns > 0) {
        prevRep = ctx->GetRepByInstrId(currIns - 1);
    }

    while (rep->Type() == CcuRep::CcuRepType::FUNC_BLOCK) {
        auto blockRep = static_pointer_cast<CcuRepBlock>(rep);
        rep = blockRep->GetRepByInstrId(currIns);
        CHK_PRT_RET(
            rep == nullptr,
            HCCL_ERROR(
                "Failed to find REP from FuncBlock, instrId[%u], FuncBlock[%s]", currIns, blockRep->GetLabel().c_str()),
            HcclResult::HCCL_E_PARA);
    }
    return HCCL_SUCCESS;
}

static void HandleSurroundingRepErrorInfo(
    int32_t deviceId, uint32_t execMissionId, const ErrorInfoBase& baseInfo, CcuRep::CcuRepContext& ctx,
    uint16_t startIns, uint16_t endIns, uint16_t currIns, GenErrorInfoByRepTypeFunc genByRepType,
    std::vector<CcuErrorInfo>& errorInfo)
{
    HCCL_ERROR(
        "[CcuErrorHandler]device %d, execMissionId[%u], startIns[%u], endIns[%u], currIns[%u]", deviceId, execMissionId,
        startIns, endIns, currIns);
    if (endIns == currIns) {
        HCCL_ERROR("[CcuErrorHandler]device %d SQE != CQE, endIns[%u], currIns[%u]", deviceId, endIns, currIns);
    }

    const uint16_t beginIns = FindSurroundingBeginInstr(ctx, startIns, currIns);
    for (uint16_t instrId = beginIns; instrId <= currIns; instrId++) {
        auto surroundingRep = ctx.GetRepByInstrId(instrId);
        if (surroundingRep == nullptr) {
            HCCL_WARNING(
                "[CcuErrorHandler][%s] cannot find REP from current CcuContext, instrId[%u]", __func__, instrId);
            continue;
        }
        genByRepType(baseInfo, surroundingRep, errorInfo);
    }
}

HcclResult CcuTaskException::GetCcuMissionContextRaw(
    int32_t deviceId, uint32_t dieId, uint32_t missionId, uint8_t* buf, size_t bufLen, size_t& copiedLen)
{
    return QueryCcuCtxRaw(
        deviceId, dieId, missionId, CcuOpcodeType::CCU_U_OP_GET_MISSION_CTX, "GetCcuMissionContextRaw", buf, bufLen,
        copiedLen);
}

static string StatusCode2Str(uint8_t highPart, uint8_t lowPart)
{
    HCCL_INFO("Mission Status Code: highPart[0x%02x], lowPart[0x%02x]", highPart, lowPart);
    const auto status = MISSION_STATUS_MAP.find(highPart);
    if (status == MISSION_STATUS_MAP.end()) {
        return "Unknown Status";
    }
    stringstream result;
    result << status->second;

    const auto lowMap = MISSION_SUB_STATUS_MAP.find(highPart);
    if (lowMap == MISSION_SUB_STATUS_MAP.end()) {
        return result.str();
    }

    const auto subStatus = lowMap->second.find(lowPart);
    const string subStatusMsg = subStatus == lowMap->second.end() ? "Unknown Status" : subStatus->second;
    result << ", " << subStatusMsg;
    return result.str();
}

void CcuTaskException::GenStatusInfo(const ErrorInfoBase& baseInfo, vector<CcuErrorInfo>& errorInfo)
{
    CcuErrorInfo errorMsg{};
    errorMsg.type = CcuErrorType::MISSION;
    errorMsg.SetBaseInfo(CcuRep::CcuRepType::BASE, baseInfo.dieId, baseInfo.missionId, baseInfo.currentInsId);
    const auto baseInformation = Hccl::StringFormat("dieId[%u], missionId[%u]", baseInfo.dieId, baseInfo.missionId);
    const auto taskInformation
        = Hccl::StringFormat("currentInsId[%u], status[%u]", baseInfo.currentInsId, baseInfo.status);
    const uint8_t highPart = (baseInfo.status >> 8) & 0xFF; // 高8位
    const uint8_t lowPart = baseInfo.status & 0xFF;         // 低8位
    string clusterMonitorErrMsg = CcuGetAndPrintClusterMonitorErr(baseInfo.deviceId);

    if (highPart == CCUM_EXECUTE_ERROR && lowPart == CCU_MISSION_TASK_KILLED) {
        RPT_INPUT_ERR(
            true, "EI0002",
            std::vector<std::string>({"remote_rankid", "base_information", "task_information", "group_rank_content"}),
            std::vector<std::string>(
                {std::to_string(baseInfo.deviceId), baseInformation.c_str(),
                 (taskInformation + clusterMonitorErrMsg).c_str(), "none"}));
        HCCL_ERROR(
            "[%s][%s][%s] Task run failed, EI0002, deviceID:[%u], %s, %s.", LOG_KEYWORDS_TASK_EXEC.c_str(),
            LOG_KEYWORDS_TIMEOUT.c_str(), LOG_KEYWORDS_CCU.c_str(), baseInfo.deviceId, baseInformation.c_str(),
            taskInformation.c_str());
    }
    const string statusMsg = StatusCode2Str(highPart, lowPart);
    const auto sRet
        = strncpy_s(errorMsg.msg.mission.missionError, MISSION_STATUS_MSG_LEN, statusMsg.c_str(), statusMsg.length());
    if (sRet != EOK) {
        HCCL_ERROR("[CcuErrorHandler][%s] strcpy failed, statusMsg: [%s].sRet:[%d]", __func__, statusMsg.c_str(), sRet);
    }

    errorInfo.push_back(errorMsg);
}

uint16_t CcuTaskException::GetCcuCKEValue(int32_t deviceId, uint32_t dieId, uint32_t ckeId)
{
    CustomChannelInfoIn inBuff{};
    CustomChannelInfoOut outBuff{};
    auto tlvHandle = Hccl::HccpTlvHdcManager::GetInstance().GetTlvHandle(deviceId);
    CHK_PRT_RET(
        tlvHandle == nullptr, HCCL_ERROR("[%s]tlvHandle is null, deviceId[%d]", __func__, deviceId), INVALID_U16);

    inBuff.op = CcuOpcodeType::CCU_U_OP_GET_CKE;
    inBuff.data.dataInfo.udieIdx = dieId;
    inBuff.offsetStartIdx = ckeId;
    inBuff.data.dataInfo.dataArraySize = 1; // 读1个CKE
    inBuff.data.dataInfo.dataLen = sizeof(uint64_t) * inBuff.data.dataInfo.dataArraySize;

    HcclResult ret = HccpRaTlvRequestForCustomChannel(
        tlvHandle, MSG_TYPE_CCU_DISPATCH_CMD, static_cast<void*>(&inBuff), static_cast<void*>(&outBuff));
    CHK_PRT_RET(
        ret != HCCL_SUCCESS, HCCL_ERROR("[%s]HccpRaTlvRequestForCustomChannel fail, ret[%d]", __func__, ret),
        INVALID_U16);

    uint64_t ckeVal{0};
    s32 sret = memcpy_s(&ckeVal, sizeof(ckeVal), outBuff.data.dataInfo.dataArray, inBuff.data.dataInfo.dataLen);
    CHK_PRT_RET(sret != EOK, HCCL_ERROR("[%s]memcpy failed. errno[%d]", __func__, sret), INVALID_U16);
    return static_cast<uint16_t>(ckeVal);
}

HcclResult CcuTaskException::GetCcuUrmaChannel(const ChannelHandle& channel, CcuUrmaChannel*& channelImpl)
{
    void* channelPtr = nullptr;
    CHK_RET(static_cast<HcclResult>(HcommChannelGet(channel, &channelPtr)));
    CHK_PTR_NULL(channelPtr);
    channelImpl = dynamic_cast<CcuUrmaChannel*>(static_cast<Channel*>(channelPtr));
    CHK_PTR_NULL(channelImpl);
    return HCCL_SUCCESS;
}

HcclResult CcuTaskException::GetChannelIdByHandle(const ChannelHandle& channel, uint32_t& channelId)
{
    CcuUrmaChannel* channelImpl = nullptr;
    CHK_RET(GetCcuUrmaChannel(channel, channelImpl));
    channelId = channelImpl->GetChannelId();
    return HCCL_SUCCESS;
}

HcclResult
CcuTaskException::GetSignalIdByHandle(const ChannelHandle& channel, uint16_t semIdx, bool isRmtSig, uint32_t& signalId)
{
    CcuUrmaChannel* channelImpl = nullptr;
    CHK_RET(GetCcuUrmaChannel(channel, channelImpl));
    if (isRmtSig) {
        CHK_PRT_RET(
            channelImpl->GetRmtCkeByIndex(semIdx, signalId) != HcclResult::HCCL_SUCCESS,
            HCCL_ERROR("[%s] failed to get remote cke id, channelHandle[0x%llx].", __func__, channel), HCCL_E_UNAVAIL);
    } else {
        CHK_PRT_RET(
            channelImpl->GetLocCkeByIndex(semIdx, signalId) != HcclResult::HCCL_SUCCESS,
            HCCL_ERROR("[%s] failed to get local cke id, channelHandle[0x%llx]--sem[%u].", __func__, channel, semIdx),
            HCCL_E_UNAVAIL);
    }
    return HCCL_SUCCESS;
}

HcclResult CcuTaskException::GetVariableIdByHandle(const ChannelHandle& channel, uint16_t varIdx, uint32_t& varId)
{
    CcuUrmaChannel* channelImpl = nullptr;
    CHK_RET(GetCcuUrmaChannel(channel, channelImpl));
    CHK_PRT_RET(
        channelImpl->GetRmtXnByIndex(varIdx, varId) != HcclResult::HCCL_SUCCESS,
        HCCL_ERROR("[%s] failed to get remote xn id, channelHandle[0x%llx]--var[%u].", __func__, channel, varIdx),
        HCCL_E_UNAVAIL);
    return HCCL_SUCCESS;
}

void CcuTaskException::GenErrorInfoLocRecordEvent(
    const ErrorInfoBase& baseInfo, shared_ptr<CcuRep::CcuRepBase> repBase, vector<CcuErrorInfo>& errorInfo)
{
    CcuErrorInfo errorMsg{};
    errorMsg.type = CcuErrorType::WAIT_SIGNAL;
    errorMsg.SetBaseInfo(repBase->Type(), baseInfo.dieId, baseInfo.missionId, repBase->StartInstrId());

    const auto rep = static_pointer_cast<CcuRep::CcuRepLocRecordEvent>(repBase);
    errorMsg.msg.waitSignal.signalId = rep->GetEventId();
    errorMsg.msg.waitSignal.signalValue = GetCcuCKEValue(baseInfo.deviceId, baseInfo.dieId, rep->GetEventId());
    errorMsg.msg.waitSignal.signalMask = rep->GetMask();

    errorInfo.push_back(errorMsg);
}

void CcuTaskException::GenErrorInfoLocWaitEvent(
    const ErrorInfoBase& baseInfo, shared_ptr<CcuRep::CcuRepBase> repBase, vector<CcuErrorInfo>& errorInfo)
{
    CcuErrorInfo errorMsg{};
    errorMsg.type = CcuErrorType::WAIT_SIGNAL;
    errorMsg.SetBaseInfo(repBase->Type(), baseInfo.dieId, baseInfo.missionId, repBase->StartInstrId());

    const auto rep = static_pointer_cast<CcuRep::CcuRepLocWaitEvent>(repBase);
    errorMsg.msg.waitSignal.signalId = rep->GetEventId();
    errorMsg.msg.waitSignal.signalValue = GetCcuCKEValue(baseInfo.deviceId, baseInfo.dieId, rep->GetEventId());
    errorMsg.msg.waitSignal.signalMask = rep->GetMask();

    errorInfo.push_back(errorMsg);
}

void CcuTaskException::GenErrorInfoLocWaitNotify(
    const ErrorInfoBase& baseInfo, shared_ptr<CcuRep::CcuRepBase> repBase, vector<CcuErrorInfo>& errorInfo)
{
    CcuErrorInfo errorMsg{};
    errorMsg.type = CcuErrorType::WAIT_SIGNAL;
    errorMsg.SetBaseInfo(repBase->Type(), baseInfo.dieId, baseInfo.missionId, repBase->StartInstrId());

    const auto rep = static_pointer_cast<CcuRep::CcuRepLocWaitNotify>(repBase);
    errorMsg.msg.waitSignal.signalId = rep->GetNotifyId();
    errorMsg.msg.waitSignal.signalValue = GetCcuCKEValue(baseInfo.deviceId, baseInfo.dieId, rep->GetNotifyId());
    errorMsg.msg.waitSignal.signalMask = rep->GetMask();

    errorInfo.push_back(errorMsg);
}

uint64_t CcuTaskException::GetCcuGSAValue(int32_t deviceId, uint32_t dieId, uint32_t gsaId)
{
    uint64_t gsaVal{0};

    auto tlvHandle = Hccl::HccpTlvHdcManager::GetInstance().GetTlvHandle(deviceId);
    CHK_PRT_RET(
        tlvHandle == nullptr, HCCL_ERROR("[%s]tlvHandle is null, deviceId[%d]", __func__, deviceId), DFX_INVALID_U64);

    CustomChannelInfoIn inBuff{};
    CustomChannelInfoOut outBuff{};

    inBuff.op = CcuOpcodeType::CCU_U_OP_GET_GSA;
    inBuff.data.dataInfo.udieIdx = dieId;
    inBuff.offsetStartIdx = gsaId;
    inBuff.data.dataInfo.dataArraySize = 1; // 读1个GSA
    inBuff.data.dataInfo.dataLen = sizeof(uint64_t) * inBuff.data.dataInfo.dataArraySize;

    HcclResult ret = HccpRaTlvRequestForCustomChannel(
        tlvHandle, MSG_TYPE_CCU_DISPATCH_CMD, static_cast<void*>(&inBuff), static_cast<void*>(&outBuff));
    CHK_PRT_RET(
        ret != HCCL_SUCCESS, HCCL_ERROR("[%s]HccpRaTlvRequestForCustomChannel fail, ret[%d]", __func__, ret),
        DFX_INVALID_U64);

    auto sret = memcpy_s(&gsaVal, sizeof(gsaVal), outBuff.data.dataInfo.dataArray, inBuff.data.dataInfo.dataLen);
    CHK_PRT_RET(sret != EOK, HCCL_ERROR("[%s]memcpy failed. errno[%d]", __func__, sret), DFX_INVALID_U64);
    return gsaVal;
}

uint64_t CcuTaskException::GetCcuXnValue(int32_t deviceId, uint32_t dieId, uint32_t xnId)
{
    auto tlvHandle = Hccl::HccpTlvHdcManager::GetInstance().GetTlvHandle(deviceId);
    CHK_PRT_RET(
        tlvHandle == nullptr, HCCL_ERROR("[%s]tlvHandle is null, deviceId[%d]", __func__, deviceId), DFX_INVALID_U64);
    uint64_t xnVal{0};

    CustomChannelInfoIn inBuff{};
    CustomChannelInfoOut outBuff{};

    inBuff.op = CcuOpcodeType::CCU_U_OP_GET_XN;
    inBuff.data.dataInfo.udieIdx = dieId;
    inBuff.offsetStartIdx = xnId;
    inBuff.data.dataInfo.dataArraySize = 1; // 读1个Xn
    inBuff.data.dataInfo.dataLen = sizeof(uint64_t) * inBuff.data.dataInfo.dataArraySize;

    HcclResult ret = HccpRaTlvRequestForCustomChannel(
        tlvHandle, MSG_TYPE_CCU_DISPATCH_CMD, static_cast<void*>(&inBuff), static_cast<void*>(&outBuff));

    CHK_PRT_RET(
        ret != HCCL_SUCCESS, HCCL_ERROR("[%s]HccpRaTlvRequestForCustomChannel fail, ret[%d]", __func__, ret),
        DFX_INVALID_U64);

    auto sret = memcpy_s(&xnVal, sizeof(xnVal), outBuff.data.dataInfo.dataArray, inBuff.data.dataInfo.dataLen);
    CHK_PRT_RET(sret != EOK, HCCL_ERROR("[%s]memcpy failed. errno[%d]", __func__, sret), DFX_INVALID_U64);
    return xnVal;
}

void CcuTaskException::GenErrorInfoRemPostSem(
    const ErrorInfoBase& baseInfo, shared_ptr<CcuRep::CcuRepBase> repBase, vector<CcuErrorInfo>& errorInfo)
{
    CcuErrorInfo errorMsg{};
    errorMsg.type = CcuErrorType::WAIT_SIGNAL;
    errorMsg.SetBaseInfo(repBase->Type(), baseInfo.dieId, baseInfo.missionId, repBase->StartInstrId());
    const auto rep = static_pointer_cast<CcuRep::CcuRepRemPostSem>(repBase);

    uint32_t channelId = 0;
    CHK_PRT_RET(
        GetChannelIdByHandle(rep->GetChannel(), channelId) != HcclResult::HCCL_SUCCESS,
        HCCL_ERROR("GetChannelIdByHandle[0x%llx] failed", rep->GetChannel()), );
    uint32_t signalId = 0;
    CHK_PRT_RET(
        GetSignalIdByHandle(rep->GetChannel(), rep->GetSemIndex(), 1, signalId) != HcclResult::HCCL_SUCCESS,
        HCCL_ERROR("GetSignalIdByHandle[0x%llx]--SemIdx[%d] failed", rep->GetChannel(), rep->GetSemIndex()), );

    errorMsg.msg.waitSignal.signalId = signalId;
    errorMsg.msg.waitSignal.signalMask = rep->GetMask();
    auto sret = memset_s(
        errorMsg.msg.waitSignal.channelId, sizeof(errorMsg.msg.waitSignal.channelId), 0xFF,
        sizeof(errorMsg.msg.waitSignal.channelId));
    CHK_PRT_RET(sret != EOK, HCCL_ERROR("[%s]memset_s failed. errno[%d]", __func__, sret), );
    errorMsg.msg.waitSignal.channelId[0] = channelId;

    errorInfo.push_back(errorMsg);
}

void CcuTaskException::GenErrorInfoRemWaitSem(
    const ErrorInfoBase& baseInfo, shared_ptr<CcuRep::CcuRepBase> repBase, vector<CcuErrorInfo>& errorInfo)
{
    CcuErrorInfo errorMsg{};
    errorMsg.type = CcuErrorType::WAIT_SIGNAL;
    errorMsg.SetBaseInfo(repBase->Type(), baseInfo.dieId, baseInfo.missionId, repBase->StartInstrId());
    const auto rep = static_pointer_cast<CcuRep::CcuRepRemWaitSem>(repBase);

    uint32_t channelId = 0;
    CHK_PRT_RET(
        GetChannelIdByHandle(rep->GetChannel(), channelId) != HcclResult::HCCL_SUCCESS,
        HCCL_ERROR("GetChannelIdByHandle[0x%llx] failed", rep->GetChannel()), );
    uint32_t signalId = 0;
    CHK_PRT_RET(
        GetSignalIdByHandle(rep->GetChannel(), rep->GetSemIndex(), 0, signalId) != HcclResult::HCCL_SUCCESS,
        HCCL_ERROR("GetSignalIdByHandle[0x%llx]--SemIdx[%d] failed", rep->GetChannel(), rep->GetSemIndex()), );

    errorMsg.msg.waitSignal.signalId = signalId;
    errorMsg.msg.waitSignal.signalValue
        = GetCcuCKEValue(baseInfo.deviceId, baseInfo.dieId, errorMsg.msg.waitSignal.signalId);
    errorMsg.msg.waitSignal.signalMask = rep->GetMask();
    auto sret = memset_s(
        errorMsg.msg.waitSignal.channelId, sizeof(errorMsg.msg.waitSignal.channelId), 0xFF,
        sizeof(errorMsg.msg.waitSignal.channelId));
    CHK_PRT_RET(sret != EOK, HCCL_ERROR("[%s]memset_s failed. errno[%d]", __func__, sret), );

    errorMsg.msg.waitSignal.channelId[0] = channelId;

    errorInfo.push_back(errorMsg);
}

void CcuTaskException::GenErrorInfoRemPostVar(
    const ErrorInfoBase& baseInfo, shared_ptr<CcuRep::CcuRepBase> repBase, vector<CcuErrorInfo>& errorInfo)
{
    CcuErrorInfo errorMsg{};
    errorMsg.type = CcuErrorType::WAIT_SIGNAL;
    errorMsg.SetBaseInfo(repBase->Type(), baseInfo.dieId, baseInfo.missionId, repBase->StartInstrId());
    const auto rep = static_pointer_cast<CcuRep::CcuRepRemPostVar>(repBase);

    // 通过channelHandle从rep中获取channelId，rmtCkeId，rmtXnId
    uint32_t channelId = 0;
    CHK_PRT_RET(
        GetChannelIdByHandle(rep->GetChannel(), channelId) != HcclResult::HCCL_SUCCESS,
        HCCL_ERROR("GetChannelIdByHandle[0x%llx] failed", rep->GetChannel()), );
    uint32_t rmtCkeId = 0;
    CHK_PRT_RET(
        GetSignalIdByHandle(rep->GetChannel(), rep->GetSemIndex(), 1, rmtCkeId) != HcclResult::HCCL_SUCCESS,
        HCCL_ERROR("GetSignalIdByHandle[0x%llx]--SemIdx[%d] failed", rep->GetChannel(), rep->GetSemIndex()), );
    uint32_t rmtXnId = 0;
    CHK_PRT_RET(
        GetVariableIdByHandle(rep->GetChannel(), rep->GetParamIndex(), rmtXnId) != HcclResult::HCCL_SUCCESS,
        HCCL_ERROR("GetVariableIdByHandle[0x%llx]--VarIdx[%d] failed", rep->GetChannel(), rep->GetParamIndex()), );

    errorMsg.msg.waitSignal.signalId = rmtCkeId;
    errorMsg.msg.waitSignal.signalMask = rep->GetMask();
    auto sret = memset_s(
        errorMsg.msg.waitSignal.channelId, sizeof(errorMsg.msg.waitSignal.channelId), 0xFF,
        sizeof(errorMsg.msg.waitSignal.channelId));
    CHK_PRT_RET(sret != EOK, HCCL_ERROR("[%s]memset_s failed. errno[%d]", __func__, sret), );

    errorMsg.msg.waitSignal.channelId[0] = channelId;
    errorMsg.msg.waitSignal.paramId = rmtXnId;
    errorMsg.msg.waitSignal.paramValue = GetCcuXnValue(baseInfo.deviceId, baseInfo.dieId, rep->GetParam().Id());

    errorInfo.push_back(errorMsg);
}

void CcuTaskException::GenErrorInfoPostSharedSem(
    const ErrorInfoBase& baseInfo, shared_ptr<CcuRep::CcuRepBase> repBase, vector<CcuErrorInfo>& errorInfo)
{
    CcuErrorInfo errorMsg{};
    errorMsg.type = CcuErrorType::WAIT_SIGNAL;
    errorMsg.SetBaseInfo(repBase->Type(), baseInfo.dieId, baseInfo.missionId, repBase->StartInstrId());

    const auto rep = static_pointer_cast<CcuRep::CcuRepRecordSharedNotify>(repBase);
    errorMsg.msg.waitSignal.signalId = rep->GetNotifyId();
    errorMsg.msg.waitSignal.signalMask = rep->GetMask();

    errorInfo.push_back(errorMsg);
}

void CcuTaskException::GenErrorInfoRead(
    const ErrorInfoBase& baseInfo, shared_ptr<CcuRep::CcuRepBase> repBase, vector<CcuErrorInfo>& errorInfo)
{
    CcuErrorInfo errorMsg{};
    errorMsg.type = CcuErrorType::TRANS_MEM;
    errorMsg.SetBaseInfo(repBase->Type(), baseInfo.dieId, baseInfo.missionId, repBase->StartInstrId());
    const auto rep = static_pointer_cast<CcuRep::CcuRepRead>(repBase);

    uint32_t channelId = 0;
    CHK_PRT_RET(
        GetChannelIdByHandle(rep->GetChannel(), channelId) != HcclResult::HCCL_SUCCESS,
        HCCL_ERROR("GetChannelIdByHandle[0x%llx] failed", rep->GetChannel()), );

    errorMsg.msg.transMem.locAddr = GetCcuGSAValue(baseInfo.deviceId, baseInfo.dieId, rep->GetLocAddrId());
    errorMsg.msg.transMem.locToken = GetCcuXnValue(baseInfo.deviceId, baseInfo.dieId, rep->GetLocTokenId());
    errorMsg.msg.transMem.rmtAddr = GetCcuGSAValue(baseInfo.deviceId, baseInfo.dieId, rep->GetRemAddrId());
    errorMsg.msg.transMem.rmtToken = GetCcuXnValue(baseInfo.deviceId, baseInfo.dieId, rep->GetRemTokenId());
    errorMsg.msg.transMem.len = GetCcuXnValue(baseInfo.deviceId, baseInfo.dieId, rep->GetLenId());
    errorMsg.msg.transMem.signalMask = rep->GetMask();
    errorMsg.msg.transMem.signalId = rep->GetSemId();
    errorMsg.msg.transMem.channelId = channelId;
    errorInfo.push_back(errorMsg);
}

void CcuTaskException::GenErrorInfoWrite(
    const ErrorInfoBase& baseInfo, shared_ptr<CcuRep::CcuRepBase> repBase, vector<CcuErrorInfo>& errorInfo)
{
    CcuErrorInfo errorMsg{};
    errorMsg.type = CcuErrorType::TRANS_MEM;
    errorMsg.SetBaseInfo(repBase->Type(), baseInfo.dieId, baseInfo.missionId, repBase->StartInstrId());
    const auto rep = static_pointer_cast<CcuRep::CcuRepWrite>(repBase);

    uint32_t channelId = 0;
    CHK_PRT_RET(
        GetChannelIdByHandle(rep->GetChannel(), channelId) != HcclResult::HCCL_SUCCESS,
        HCCL_ERROR("GetChannelIdByHandle[0x%llx] failed", rep->GetChannel()), );

    errorMsg.msg.transMem.locAddr = GetCcuGSAValue(baseInfo.deviceId, baseInfo.dieId, rep->GetLocAddrId());
    errorMsg.msg.transMem.locToken = GetCcuXnValue(baseInfo.deviceId, baseInfo.dieId, rep->GetLocTokenId());
    errorMsg.msg.transMem.rmtAddr = GetCcuGSAValue(baseInfo.deviceId, baseInfo.dieId, rep->GetRemAddrId());
    errorMsg.msg.transMem.rmtToken = GetCcuXnValue(baseInfo.deviceId, baseInfo.dieId, rep->GetRemTokenId());
    errorMsg.msg.transMem.len = GetCcuXnValue(baseInfo.deviceId, baseInfo.dieId, rep->GetLenId());
    errorMsg.msg.transMem.signalId = rep->GetSemId();
    errorMsg.msg.transMem.signalMask = rep->GetMask();
    errorMsg.msg.transMem.channelId = channelId;

    errorInfo.push_back(errorMsg);
}

void CcuTaskException::GenErrorInfoLocalCpy(
    const ErrorInfoBase& baseInfo, shared_ptr<CcuRep::CcuRepBase> repBase, vector<CcuErrorInfo>& errorInfo)
{
    CcuErrorInfo errorMsg{};
    errorMsg.type = CcuErrorType::TRANS_MEM;
    errorMsg.SetBaseInfo(repBase->Type(), baseInfo.dieId, baseInfo.missionId, repBase->StartInstrId());

    const auto rep = static_pointer_cast<CcuRep::CcuRepLocCpy>(repBase);
    errorMsg.msg.transMem.locAddr = GetCcuGSAValue(baseInfo.deviceId, baseInfo.dieId, rep->GetSrcAddrId());
    errorMsg.msg.transMem.rmtAddr = GetCcuGSAValue(baseInfo.deviceId, baseInfo.dieId, rep->GetDstAddrId());
    errorMsg.msg.transMem.locToken = GetCcuXnValue(baseInfo.deviceId, baseInfo.dieId, rep->GetSrcTokenId());
    errorMsg.msg.transMem.rmtToken = GetCcuXnValue(baseInfo.deviceId, baseInfo.dieId, rep->GetDstTokenId());
    errorMsg.msg.transMem.len = GetCcuXnValue(baseInfo.deviceId, baseInfo.dieId, rep->GetLenId());
    errorMsg.msg.transMem.signalId = rep->GetSemId();
    errorMsg.msg.transMem.signalMask = rep->GetMask();

    errorInfo.push_back(errorMsg);
}

void CcuTaskException::GenErrorInfoLocalReduce(
    const ErrorInfoBase& baseInfo, shared_ptr<CcuRep::CcuRepBase> repBase, vector<CcuErrorInfo>& errorInfo)
{
    CcuErrorInfo errorMsg{};
    errorMsg.type = CcuErrorType::TRANS_MEM;
    errorMsg.SetBaseInfo(repBase->Type(), baseInfo.dieId, baseInfo.missionId, repBase->StartInstrId());

    const auto rep = static_pointer_cast<CcuRep::CcuRepLocCpy>(repBase);
    errorMsg.msg.transMem.locAddr = GetCcuGSAValue(baseInfo.deviceId, baseInfo.dieId, rep->GetSrcAddrId());
    errorMsg.msg.transMem.locToken = GetCcuXnValue(baseInfo.deviceId, baseInfo.dieId, rep->GetSrcTokenId());
    errorMsg.msg.transMem.rmtAddr = GetCcuGSAValue(baseInfo.deviceId, baseInfo.dieId, rep->GetDstAddrId());
    errorMsg.msg.transMem.rmtToken = GetCcuXnValue(baseInfo.deviceId, baseInfo.dieId, rep->GetDstTokenId());
    errorMsg.msg.transMem.len = GetCcuXnValue(baseInfo.deviceId, baseInfo.dieId, rep->GetLenId());
    errorMsg.msg.transMem.signalId = rep->GetSemId();
    errorMsg.msg.transMem.signalMask = rep->GetMask();
    errorMsg.msg.transMem.opType = rep->GetOpType();
    errorMsg.msg.transMem.dataType = rep->GetDataType();

    errorInfo.push_back(errorMsg);
}

void CcuTaskException::GenErrorInfoBufRead(
    const ErrorInfoBase& baseInfo, shared_ptr<CcuRep::CcuRepBase> repBase, vector<CcuErrorInfo>& errorInfo)
{
    CcuErrorInfo errorMsg{};
    errorMsg.type = CcuErrorType::BUF_TRANS_MEM;
    errorMsg.SetBaseInfo(repBase->Type(), baseInfo.dieId, baseInfo.missionId, repBase->StartInstrId());
    const auto rep = static_pointer_cast<CcuRep::CcuRepBufRead>(repBase);

    uint32_t channelId = 0;
    CHK_PRT_RET(
        GetChannelIdByHandle(rep->GetChannel(), channelId) != HcclResult::HCCL_SUCCESS,
        HCCL_ERROR("GetChannelIdByHandle[0x%llx] failed", rep->GetChannel()), );

    errorMsg.msg.bufTransMem.bufId = GetMSIdPerDie(rep->GetDstAddrId());
    errorMsg.msg.bufTransMem.addr = GetCcuGSAValue(baseInfo.deviceId, baseInfo.dieId, rep->GetSrcAddrId());
    errorMsg.msg.bufTransMem.token = GetCcuXnValue(baseInfo.deviceId, baseInfo.dieId, rep->GetSrcTokenId());
    errorMsg.msg.bufTransMem.len = GetCcuXnValue(baseInfo.deviceId, baseInfo.dieId, rep->GetLenId());
    errorMsg.msg.bufTransMem.signalId = rep->GetSemId();
    errorMsg.msg.bufTransMem.signalMask = rep->GetMask();
    errorMsg.msg.bufTransMem.channelId = channelId;
    errorInfo.push_back(errorMsg);
}

void CcuTaskException::GenErrorInfoBufWrite(
    const ErrorInfoBase& baseInfo, shared_ptr<CcuRep::CcuRepBase> repBase, vector<CcuErrorInfo>& errorInfo)
{
    CcuErrorInfo errorMsg{};
    errorMsg.type = CcuErrorType::BUF_TRANS_MEM;
    errorMsg.SetBaseInfo(repBase->Type(), baseInfo.dieId, baseInfo.missionId, repBase->StartInstrId());
    const auto rep = static_pointer_cast<CcuRep::CcuRepBufWrite>(repBase);

    uint32_t channelId = 0;
    CHK_PRT_RET(
        GetChannelIdByHandle(rep->GetChannel(), channelId) != HcclResult::HCCL_SUCCESS,
        HCCL_ERROR("GetChannelIdByHandle[0x%llx] failed", rep->GetChannel()), );

    errorMsg.msg.bufTransMem.bufId = GetMSIdPerDie(rep->GetSrcId());
    errorMsg.msg.bufTransMem.addr = GetCcuGSAValue(baseInfo.deviceId, baseInfo.dieId, rep->GetDstAddrId());
    errorMsg.msg.bufTransMem.token = GetCcuXnValue(baseInfo.deviceId, baseInfo.dieId, rep->GetDstTokenId());
    errorMsg.msg.bufTransMem.len = GetCcuXnValue(baseInfo.deviceId, baseInfo.dieId, rep->GetLenId());
    errorMsg.msg.bufTransMem.signalId = rep->GetSemId();
    errorMsg.msg.bufTransMem.signalMask = rep->GetMask();
    errorMsg.msg.bufTransMem.channelId = channelId;

    errorInfo.push_back(errorMsg);
}

void CcuTaskException::GenErrorInfoBufLocRead(
    const ErrorInfoBase& baseInfo, shared_ptr<CcuRep::CcuRepBase> repBase, vector<CcuErrorInfo>& errorInfo)
{
    CcuErrorInfo errorMsg{};
    errorMsg.type = CcuErrorType::BUF_TRANS_MEM;
    errorMsg.SetBaseInfo(repBase->Type(), baseInfo.dieId, baseInfo.missionId, repBase->StartInstrId());

    const auto rep = static_pointer_cast<CcuRep::CcuRepBufLocRead>(repBase);
    errorMsg.msg.bufTransMem.bufId = GetMSIdPerDie(rep->GetDstId());
    errorMsg.msg.bufTransMem.addr = GetCcuGSAValue(baseInfo.deviceId, baseInfo.dieId, rep->GetSrcAddrId());
    errorMsg.msg.bufTransMem.token = GetCcuXnValue(baseInfo.deviceId, baseInfo.dieId, rep->GetSrcTokenId());
    errorMsg.msg.bufTransMem.len = GetCcuXnValue(baseInfo.deviceId, baseInfo.dieId, rep->GetLenId());
    errorMsg.msg.bufTransMem.signalId = rep->GetSemId();
    errorMsg.msg.bufTransMem.signalMask = rep->GetMask();

    errorInfo.push_back(errorMsg);
}

void CcuTaskException::GenErrorInfoBufLocWrite(
    const ErrorInfoBase& baseInfo, shared_ptr<CcuRep::CcuRepBase> repBase, vector<CcuErrorInfo>& errorInfo)
{
    CcuErrorInfo errorMsg{};
    errorMsg.type = CcuErrorType::BUF_TRANS_MEM;
    errorMsg.SetBaseInfo(repBase->Type(), baseInfo.dieId, baseInfo.missionId, repBase->StartInstrId());

    const auto rep = static_pointer_cast<CcuRep::CcuRepBufLocWrite>(repBase);
    errorMsg.msg.bufTransMem.bufId = GetMSIdPerDie(rep->GetSrcAddrId());
    errorMsg.msg.bufTransMem.addr = GetCcuGSAValue(baseInfo.deviceId, baseInfo.dieId, rep->GetDstAddrId());
    errorMsg.msg.bufTransMem.token = GetCcuXnValue(baseInfo.deviceId, baseInfo.dieId, rep->GetDstTokenId());
    errorMsg.msg.bufTransMem.len = GetCcuXnValue(baseInfo.deviceId, baseInfo.dieId, rep->GetLenId());
    errorMsg.msg.bufTransMem.signalId = rep->GetSemId();
    errorMsg.msg.bufTransMem.signalMask = rep->GetMask();

    errorInfo.push_back(errorMsg);
}

void CcuTaskException::GenErrorInfoBufReduce(
    const ErrorInfoBase& baseInfo, shared_ptr<CcuRep::CcuRepBase> repBase, vector<CcuErrorInfo>& errorInfo)
{
    CcuErrorInfo errorMsg{};
    errorMsg.type = CcuErrorType::BUF_REDUCE;
    errorMsg.SetBaseInfo(repBase->Type(), baseInfo.dieId, baseInfo.missionId, repBase->StartInstrId());

    const auto rep = static_pointer_cast<CcuRep::CcuRepBufReduce>(repBase);
    errorMsg.msg.bufReduce.count = rep->GetCount();
    errorMsg.msg.bufReduce.dataType = rep->GetDataType();
    errorMsg.msg.bufReduce.outputDataType = rep->GetOutputDataType();
    errorMsg.msg.bufReduce.opType = rep->GetOpType();
    errorMsg.msg.bufReduce.signalId = rep->GetSemId();
    errorMsg.msg.bufReduce.signalMask = rep->GetMask();
    errorMsg.msg.bufReduce.xnIdLength = rep->GetXnLengthId();
    const auto& buffs = rep->GetMem();
    auto sret = memset_s(
        errorMsg.msg.bufReduce.bufIds, sizeof(errorMsg.msg.bufReduce.bufIds), 0xFF,
        sizeof(errorMsg.msg.bufReduce.bufIds));
    CHK_PRT_RET(sret != EOK, HCCL_ERROR("[%s]memset failed. errno[%d]", __func__, sret), );
    for (uint32_t i = 0; i < buffs.size() && i < BUF_REDUCE_ID_SIZE; ++i) {
        errorMsg.msg.bufReduce.bufIds[i] = GetMSIdPerDie(buffs[i].Id());
    }

    errorInfo.push_back(errorMsg);
}
void CcuTaskException::GenErrorInfoDefault(
    const ErrorInfoBase& baseInfo, shared_ptr<CcuRep::CcuRepBase> repBase, vector<CcuErrorInfo>& errorInfo)
{
    CcuErrorInfo errorMsg{};
    errorMsg.type = CcuErrorType::DEFAULT;
    errorMsg.SetBaseInfo(repBase->Type(), baseInfo.dieId, baseInfo.missionId, repBase->StartInstrId());
    errorInfo.push_back(errorMsg);
}

void CcuTaskException::GenErrorInfoWriteVarAtomic(
    const ErrorInfoBase& baseInfo, shared_ptr<CcuRep::CcuRepBase> repBase, vector<CcuErrorInfo>& errorInfo)
{
    CcuErrorInfo errorMsg{};
    errorMsg.type = CcuErrorType::WRITE_VAR_ATOMIC;
    errorMsg.SetBaseInfo(repBase->Type(), baseInfo.dieId, baseInfo.missionId, repBase->StartInstrId());
    const auto rep = static_pointer_cast<CcuRep::CcuRepWriteVarAtomic>(repBase);
    errorMsg.msg.writeVarAtomic.dstId = rep->GetTargetId();
    errorMsg.msg.writeVarAtomic.dstValue = GetCcuXnValue(baseInfo.deviceId, baseInfo.dieId, rep->GetTargetId());
    errorMsg.msg.writeVarAtomic.signalId = rep->GetSemId();
    errorMsg.msg.writeVarAtomic.signalMask = rep->GetMask();
    errorInfo.push_back(errorMsg);
}

void CcuTaskException::GenErrorInfoWriteWithCntInc(
    const ErrorInfoBase& baseInfo, shared_ptr<CcuRep::CcuRepBase> repBase, vector<CcuErrorInfo>& errorInfo)
{
    CcuErrorInfo errorMsg{};
    errorMsg.type = CcuErrorType::WRITE_WITH_CNT_INC;
    errorMsg.SetBaseInfo(repBase->Type(), baseInfo.dieId, baseInfo.missionId, repBase->StartInstrId());
    const auto rep = static_pointer_cast<CcuRep::CcuRepWriteWithCntInc>(repBase);
    errorMsg.msg.writeWithCntInc.locAddr = GetCcuXnValue(baseInfo.deviceId, baseInfo.dieId, rep->GetLocAddrId());
    errorMsg.msg.writeWithCntInc.rmtAddr = GetCcuXnValue(baseInfo.deviceId, baseInfo.dieId, rep->GetRemAddrId());
    errorMsg.msg.writeWithCntInc.len = GetCcuXnValue(baseInfo.deviceId, baseInfo.dieId, rep->GetLenId());
    errorMsg.msg.writeWithCntInc.incCntAddr = GetCcuXnValue(baseInfo.deviceId, baseInfo.dieId, rep->GetIncCntAddrId());
    errorInfo.push_back(errorMsg);
}

void CcuTaskException::GenErrorInfoCascCntWait(
    const ErrorInfoBase& baseInfo, shared_ptr<CcuRep::CcuRepBase> repBase, vector<CcuErrorInfo>& errorInfo)
{
    CcuErrorInfo errorMsg{};
    errorMsg.type = CcuErrorType::CASC_CNT_WAIT;
    errorMsg.SetBaseInfo(repBase->Type(), baseInfo.dieId, baseInfo.missionId, repBase->StartInstrId());
    const auto rep = static_pointer_cast<CcuRep::CcuRepCascCntWait>(repBase);
    errorMsg.msg.cascCntWait.targetValue = rep->GetOutCntTarget();
    auto cascCntBlock = rep->GetCascCntBlock();
    errorMsg.msg.cascCntWait.currentCntValue
        = GetCcuXnValue(baseInfo.deviceId, baseInfo.dieId, cascCntBlock.totalCntXn);
    errorInfo.push_back(errorMsg);
}

void CcuTaskException::GenErrorInfoCascCntClear(
    const ErrorInfoBase& baseInfo, shared_ptr<CcuRep::CcuRepBase> repBase, vector<CcuErrorInfo>& errorInfo)
{
    CcuErrorInfo errorMsg{};
    errorMsg.type = CcuErrorType::CASC_CNT_CLEAR;
    errorMsg.SetBaseInfo(repBase->Type(), baseInfo.dieId, baseInfo.missionId, repBase->StartInstrId());
    const auto rep = static_pointer_cast<CcuRep::CcuRepCascCntClear>(repBase);
    auto cascCntBlock = rep->GetCascCntBlock();
    errorMsg.msg.cascCntClear.wishCntXnIdFirst = static_cast<uint16_t>(cascCntBlock.wishCntXns.first);
    errorMsg.msg.cascCntClear.expectedCntXn = static_cast<uint16_t>(cascCntBlock.expectedCntXn);
    errorInfo.push_back(errorMsg);
}

void CcuTaskException::GenErrorInfoLoadAddImm(
    const ErrorInfoBase& baseInfo, shared_ptr<CcuRep::CcuRepBase> repBase, vector<CcuErrorInfo>& errorInfo)
{
    CcuErrorInfo errorMsg{};
    errorMsg.type = CcuErrorType::LOAD_ADD_IMM;
    errorMsg.SetBaseInfo(repBase->Type(), baseInfo.dieId, baseInfo.missionId, repBase->StartInstrId());
    const auto rep = static_pointer_cast<CcuRep::CcuRepLoadAddImm>(repBase);
    errorMsg.msg.loadAddImm.srcId = rep->GetSrcId();
    errorMsg.msg.loadAddImm.srcOffsetId = rep->GetSrcOffsetId();
    errorMsg.msg.loadAddImm.dstId = rep->GetDstId();
    errorMsg.msg.loadAddImm.immAddValue = rep->GetImmAddValue();
    errorMsg.msg.loadAddImm.srcValue = GetCcuXnValue(baseInfo.deviceId, baseInfo.dieId, rep->GetSrcId());
    errorMsg.msg.loadAddImm.srcOffsetValue = GetCcuXnValue(baseInfo.deviceId, baseInfo.dieId, rep->GetSrcOffsetId());
    errorInfo.push_back(errorMsg);
}

void CcuTaskException::GenErrorInfoStoreAddImm(
    const ErrorInfoBase& baseInfo, shared_ptr<CcuRep::CcuRepBase> repBase, vector<CcuErrorInfo>& errorInfo)
{
    CcuErrorInfo errorMsg{};
    errorMsg.type = CcuErrorType::STORE_ADD_IMM;
    errorMsg.SetBaseInfo(repBase->Type(), baseInfo.dieId, baseInfo.missionId, repBase->StartInstrId());
    const auto rep = static_pointer_cast<CcuRep::CcuRepStoreAddImm>(repBase);
    errorMsg.msg.storeAddImm.dstId = rep->GetDstId();
    errorMsg.msg.storeAddImm.dstOffsetId = rep->GetDstOffsetId();
    errorMsg.msg.storeAddImm.srcId = rep->GetSrcId();
    errorMsg.msg.storeAddImm.immAddValue = rep->GetImmAddValue();
    errorMsg.msg.storeAddImm.dstValue = GetCcuXnValue(baseInfo.deviceId, baseInfo.dieId, rep->GetDstId());
    errorMsg.msg.storeAddImm.dstOffsetValue = GetCcuXnValue(baseInfo.deviceId, baseInfo.dieId, rep->GetDstOffsetId());
    errorInfo.push_back(errorMsg);
}

void CcuTaskException::GenErrorInfoByRepType(
    const ErrorInfoBase& baseInfo, shared_ptr<CcuRep::CcuRepBase> repBase, vector<CcuErrorInfo>& errorInfo)
{
    using GenErrorInfoFunc = void (*)(
        const ErrorInfoBase& baseInfo, shared_ptr<CcuRep::CcuRepBase> repBase, vector<CcuErrorInfo>& errorInfo);
    static const map<CcuRep::CcuRepType, GenErrorInfoFunc> HANDLER_MAP{
        // WAIT_SIGNAL
        {CcuRep::CcuRepType::LOC_RECORD_EVENT, &CcuTaskException::GenErrorInfoLocRecordEvent},
        {CcuRep::CcuRepType::LOC_WAIT_EVENT, &CcuTaskException::GenErrorInfoLocWaitEvent},
        {CcuRep::CcuRepType::LOC_WAIT_NOTIFY, &CcuTaskException::GenErrorInfoLocWaitNotify},
        {CcuRep::CcuRepType::REM_POST_SEM, &CcuTaskException::GenErrorInfoRemPostSem},
        {CcuRep::CcuRepType::REM_WAIT_SEM, &CcuTaskException::GenErrorInfoRemWaitSem},
        {CcuRep::CcuRepType::REM_POST_VAR, &CcuTaskException::GenErrorInfoRemPostVar},
        {CcuRep::CcuRepType::RECORD_SHARED_NOTIFY, &CcuTaskException::GenErrorInfoPostSharedSem},
        // TRANS_MEM
        {CcuRep::CcuRepType::READ, &CcuTaskException::GenErrorInfoRead},
        {CcuRep::CcuRepType::WRITE, &CcuTaskException::GenErrorInfoWrite},
        {CcuRep::CcuRepType::LOCAL_CPY, &CcuTaskException::GenErrorInfoLocalCpy},
        {CcuRep::CcuRepType::LOCAL_REDUCE, &CcuTaskException::GenErrorInfoLocalReduce},
        // BUF_TRANS_MEM
        {CcuRep::CcuRepType::BUF_READ, &CcuTaskException::GenErrorInfoBufRead},
        {CcuRep::CcuRepType::BUF_WRITE, &CcuTaskException::GenErrorInfoBufWrite},
        {CcuRep::CcuRepType::BUF_LOC_READ, &CcuTaskException::GenErrorInfoBufLocRead},
        {CcuRep::CcuRepType::BUF_LOC_WRITE, &CcuTaskException::GenErrorInfoBufLocWrite},
        // BUF_REDUCE
        {CcuRep::CcuRepType::BUF_REDUCE, &CcuTaskException::GenErrorInfoBufReduce},
        {CcuRep::CcuRepType::WRITE_VAR_ATOMIC, &CcuTaskException::GenErrorInfoWriteVarAtomic},
        {CcuRep::CcuRepType::WRITE_WITH_CNT_INC, &CcuTaskException::GenErrorInfoWriteWithCntInc},
        {CcuRep::CcuRepType::CASC_CNT_WAIT, &CcuTaskException::GenErrorInfoCascCntWait},
        {CcuRep::CcuRepType::CASC_CNT_CLEAR, &CcuTaskException::GenErrorInfoCascCntClear},
        {CcuRep::CcuRepType::LOAD_ADD_IMM, &CcuTaskException::GenErrorInfoLoadAddImm},
        {CcuRep::CcuRepType::STORE_ADD_IMM, &CcuTaskException::GenErrorInfoStoreAddImm}};
    const auto funcIt = HANDLER_MAP.find(repBase->Type());
    HCCL_INFO("[%s]type[%d]", __func__, repBase->Type());
    if (funcIt == HANDLER_MAP.end()) {
        // DEFAULT, chip error
        GenErrorInfoDefault(baseInfo, repBase, errorInfo);
    } else {
        (funcIt->second)(baseInfo, repBase, errorInfo);
    }
}

HcclResult CcuTaskException::GetCcuLoopContextRaw(
    int32_t deviceId, uint32_t dieId, uint32_t loopCtxId, uint8_t* buf, size_t bufLen, size_t& copiedLen)
{
    return QueryCcuCtxRaw(
        deviceId, dieId, loopCtxId, CcuOpcodeType::CCU_U_OP_GET_LOOP_CTX, "GetCcuLoopContextRaw", buf, bufLen,
        copiedLen);
}

HcclResult CcuTaskException::GenErrorInfoLoop(
    const ErrorInfoBase& baseInfo, CcuRep::CcuRepContext& ctx, vector<CcuErrorInfo>& errorInfo)
{
    // 找LoopRep
    auto repBase = ctx.GetRepByInstrId(baseInfo.currentInsId);
    if (repBase == nullptr || repBase->Type() != CcuRep::CcuRepType::LOOP) {
        HCCL_ERROR("Failed to find Loop REP from CcuContext, instrId[%u]", baseInfo.currentInsId);
        return HCCL_E_PARA;
    }
    const auto rep = static_pointer_cast<CcuRep::CcuRepLoop>(repBase);

    CcuErrorInfo errorMsg{};
    errorMsg.type = CcuErrorType::LOOP;
    errorMsg.SetBaseInfo(repBase->Type(), baseInfo.dieId, baseInfo.missionId, baseInfo.currentInsId);

    LoopXm loopXm{};
    loopXm.value = GetCcuXnValue(baseInfo.deviceId, baseInfo.dieId, rep->GetLoopParam()->Id());
    CcuLoopInfo loopInfo{};
    if (FetchAndDecodeLoopInfo(baseInfo.deviceId, baseInfo.dieId, loopXm.loopCtxId, loopInfo) != HCCL_SUCCESS) {
        HCCL_ERROR("[GenErrorInfoLoop] Failed to fetch/decode loop info, deviceId[%d]", baseInfo.deviceId);
        return HCCL_E_INTERNAL;
    }
    errorMsg.msg.loop.startInstrId = rep->GetLoopBlock()->StartInstrId();
    errorMsg.msg.loop.endInstrId = rep->GetLoopBlock()->StartInstrId() + rep->GetLoopBlock()->InstrCount() - 1;
    errorMsg.msg.loop.loopEngineId = loopXm.loopCtxId;
    errorMsg.msg.loop.loopCnt = static_cast<uint16_t>(loopXm.loopCnt);
    errorMsg.msg.loop.loopCurrentCnt = loopInfo.currentCnt;
    errorMsg.msg.loop.addrStride = loopInfo.addrStride;

    errorInfo.push_back(errorMsg);

    // 解析Loop内的异常Rep
    for (uint16_t loopCurrentIns = errorMsg.msg.loop.startInstrId; loopCurrentIns <= errorMsg.msg.loop.endInstrId;
         loopCurrentIns++) {
        auto inLoopExRep = rep->GetLoopBlock()->GetRepByInstrId(loopCurrentIns);
        if (inLoopExRep == nullptr) {
            HCCL_ERROR("Failed to find REP from Loop, instrId[%u], Loop[%s]", loopCurrentIns, rep->GetLabel().c_str());
            return HCCL_E_PARA;
        }
        ErrorInfoBase loopErrBase{
            baseInfo.deviceId, baseInfo.dieId, baseInfo.missionId, loopCurrentIns, baseInfo.status};
        GenErrorInfoByRepType(loopErrBase, inLoopExRep, errorInfo);
    }
    return HCCL_SUCCESS;
}

HcclResult CcuTaskException::GenErrorInfoLoopGroup(
    const ErrorInfoBase& baseInfo, shared_ptr<CcuRep::CcuRepBase> repBase, CcuRep::CcuRepContext& ctx,
    vector<CcuErrorInfo>& errorInfo)
{
    CcuErrorInfo errorMsg{};
    errorMsg.type = CcuErrorType::LOOP_GROUP;
    errorMsg.SetBaseInfo(repBase->Type(), baseInfo.dieId, baseInfo.missionId, repBase->StartInstrId());

    const auto rep = static_pointer_cast<CcuRep::CcuRepLoopGroupBundle>(repBase);
    const auto startLoopInstrId = rep->GetStartLoopInstrId();
    LoopGroupXn loopGroupXn{};
    loopGroupXn.value = GetCcuXnValue(baseInfo.deviceId, baseInfo.dieId, rep->GetOffsetParam().Id());
    errorMsg.msg.loopGroup.startLoopInsId = startLoopInstrId;
    errorMsg.msg.loopGroup.loopInsCnt = static_cast<uint16_t>(loopGroupXn.loopInsCnt);
    errorMsg.msg.loopGroup.expandOffset = static_cast<uint16_t>(loopGroupXn.expandOffset);
    errorMsg.msg.loopGroup.expandCnt = static_cast<uint16_t>(loopGroupXn.expandCnt);

    errorInfo.push_back(errorMsg);

    // 处理loop
    for (uint16_t i = 0; i < loopGroupXn.loopInsCnt; ++i) {
        uint16_t loopInsId = startLoopInstrId + i;
        ErrorInfoBase loopErrInfoBase{
            baseInfo.deviceId, baseInfo.dieId, baseInfo.missionId, loopInsId, baseInfo.status};
        CHK_RET(GenErrorInfoLoop(loopErrInfoBase, ctx, errorInfo));
    }
    return HCCL_SUCCESS;
}

static HcclResult HandleCurrentRepErrorInfo(
    const ErrorInfoBase& baseInfo, CcuRep::CcuRepContext& ctx, const std::shared_ptr<CcuRep::CcuRepBase>& rep,
    const std::shared_ptr<CcuRep::CcuRepBase>& prevRep, GenErrorInfoLoopGroupFunc genLoopGroup,
    GenErrorInfoByRepTypeFunc genByRepType, std::vector<CcuErrorInfo>& errorInfo)
{
    if ((prevRep != nullptr && prevRep->Type() == CcuRep::CcuRepType::LOOPGROUP)
        || (rep->Type() == CcuRep::CcuRepType::LOOPGROUP)) {
        CHK_RET(genLoopGroup(baseInfo, prevRep, ctx, errorInfo));
        return HCCL_SUCCESS;
    }

    if (rep->Type() == CcuRep::CcuRepType::LOC_WAIT_EVENT || rep->Type() == CcuRep::CcuRepType::LOC_WAIT_NOTIFY) {
        genByRepType(baseInfo, rep, errorInfo);
        uint16_t actValue = errorInfo.back().msg.waitSignal.signalValue;
        uint16_t expValue = errorInfo.back().msg.waitSignal.signalMask;
        for (uint16_t i = 0; i < 16; ++i) { // CKE的bit数最多为16
            uint16_t mask = 1 << i;         // 创建一个用于检查第 i 位的掩码
            if ((expValue & mask) != 0 && (actValue & mask) == 0) {
                auto depRepVec = static_pointer_cast<CcuRep::CcuRepLocWaitEvent>(rep)->GetDependencyInfo(mask);
                for (const auto& depRep : depRepVec) {
                    genByRepType(baseInfo, depRep, errorInfo);
                }
            }
        }
        return HCCL_SUCCESS;
    }

    genByRepType(baseInfo, rep, errorInfo);
    return HCCL_SUCCESS;
}

HcclResult CcuTaskException::GetCcuErrorMsg(
    int32_t deviceId, uint16_t missionStatus, const Hccl::ParaCcu& ccuTaskParam, std::vector<CcuErrorInfo>& errorInfo)
{
    HCCL_INFO(
        "[CcuTaskException]%s: deviceId[%d], dieId[%u], missionId[%u], execMissionId[%u], executeId[%llu].", __func__,
        deviceId, static_cast<u32>(ccuTaskParam.dieId), static_cast<u32>(ccuTaskParam.missionId),
        static_cast<u32>(ccuTaskParam.execMissionId), ccuTaskParam.executeId);

    CcuMissionInfo missionInfo{};
    CHK_RET(
        ValidateAndDecodeMissionContext(deviceId, missionStatus, ccuTaskParam, GetCcuMissionContextRaw, missionInfo));

    const uint16_t currIns = missionInfo.currentIns;
    CcuRep::CcuRepContext* ctx = nullptr;
    std::shared_ptr<CcuRep::CcuRepBase> rep = nullptr;
    std::shared_ptr<CcuRep::CcuRepBase> prevRep = nullptr;
    CHK_RET(ResolveRepContextAndCurrentRep(deviceId, ccuTaskParam, currIns, ctx, rep, prevRep));

    // 分类处理Rep, 返回异常信息
    ErrorInfoBase baseInfo{deviceId, ccuTaskParam.dieId, ccuTaskParam.missionId, currIns, missionStatus};
    GenStatusInfo(baseInfo, errorInfo);
    CHK_RET(HandleCurrentRepErrorInfo(
        baseInfo, *ctx, rep, prevRep, GenErrorInfoLoopGroup, GenErrorInfoByRepType, errorInfo));

    const uint16_t endIns = missionInfo.endIns;
    const uint16_t startIns = missionInfo.startIns;
    HandleSurroundingRepErrorInfo(
        deviceId, ccuTaskParam.execMissionId, baseInfo, *ctx, startIns, endIns, currIns, GenErrorInfoByRepType,
        errorInfo);
    return HCCL_SUCCESS;
}

void CcuTaskException::GetCcuCqeErrRemoteLocalIdByRankId(hccl::CollComm* collComm, uint32_t rankid, u32& remoteLocalId)
{
    if (collComm == nullptr) {
        HCCL_ERROR("[GetCcuCqeErrRemoteLocalIdByRankId] collComm is nullptr");
        return;
    }

    if (rankid == Hccl::DFX_INVALID_RANKID) {
        HCCL_ERROR("[GetCcuCqeErrRemoteLocalIdByRankId] RemoteLocalId is already set, rankId[%u]", rankid);
        return;
    }

    Hccl::HcclCommunicator* commV2 = static_cast<Hccl::HcclCommunicator*>(collComm->GetCommunicatorV2());
    if (commV2 == nullptr) {
        HCCL_ERROR("[GetCcuCqeErrRemoteLocalIdByRankId] commV2 is nullptr");
        return;
    }
    void* rankGraph = nullptr;
    HcclResult ret = commV2->GetRankGraphV2(rankGraph);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[GetCcuCqeErrRemoteLocalIdByRankId]GetRankGraphV2 failed, rankId[%u], ret[%d]", rankid, ret);
        return;
    }
    if (rankGraph == nullptr) {
        HCCL_ERROR("[GetCcuCqeErrRemoteLocalIdByRankId] rankGraph is nullptr for rankid[%u]", rankid);
        return;
    }

    Hccl::RankGraph* rankGraphv2 = static_cast<Hccl::RankGraph*>(rankGraph);
    if (rankGraphv2 == nullptr) {
        HCCL_ERROR("[GetCcuCqeErrRemoteLocalIdByRankId] rankGraphv2 is nullptr for rankid[%u]", rankid);
        return;
    }
    u32 LocalId = rankGraphv2->GetLocalId(rankid);
    remoteLocalId = LocalId;
    return;
}

void CcuTaskException::GetCcuCqeErrNetInstanceByRankId(
    hccl::CollComm* collComm, uint32_t rankid, std::string& netInstanceId)
{
    if (collComm == nullptr) {
        HCCL_ERROR("[GetCcuCqeErrNetInstanceByRankId] collComm is nullptr");
        return;
    }
    if (rankid == Hccl::DFX_INVALID_RANKID) {
        HCCL_ERROR("[GetCcuCqeErrNetInstanceByRankId] RemoteLocalId is already set, rankId[%u]", rankid);
        return;
    }
    Hccl::HcclCommunicator* commV2 = static_cast<Hccl::HcclCommunicator*>(collComm->GetCommunicatorV2());
    if (commV2 == nullptr) {
        HCCL_ERROR("[GetCcuCqeErrNetInstanceByRankId] commV2 is nullptr");
        return;
    }
    void* rankGraph = nullptr;
    HcclResult ret = commV2->GetRankGraphV2(rankGraph);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[GetCcuCqeErrNetInstanceByRankId]GetRankGraphV2 failed, rankId[%u], ret[%d]", rankid, ret);
        return;
    }
    Hccl::RankGraph* rankGraphv2 = static_cast<Hccl::RankGraph*>(rankGraph);
    const Hccl::NetInstance* netInstance = rankGraphv2->GetNetInstanceByRankId(0, rankid);
    if (netInstance == nullptr) {
        HCCL_ERROR("[GetCcuCqeErrNetInstanceByRankId] netInstance is nullptr for rankid[%u]", rankid);
        return;
    }
    std::string netInsId = netInstance->GetNetInstId();
    netInstanceId = netInsId;
    return;
}

void CcuTaskException::GetCcuCqeErrorInfo(
    const CcuErrorInfo& ccuErrorInfo, const Hccl::TaskInfo& taskInfo, u32 locDeviceId, uint8_t missionStatus)
{
    uint16_t channelId = GetChannleIdByCcuErrorInfo(ccuErrorInfo);
    if (channelId == INVALID_U16) {
        HCCL_ERROR("[%s]cannot get channelId from repType[%d]", __func__, ccuErrorInfo.repType);
        return;
    }
    auto pair = GetAddrPairByChannelId(channelId, taskInfo, locDeviceId);
    RankId remoteRankId = GetRankIdByChannelId(channelId, taskInfo, locDeviceId);
    hccl::CollComm* collComm = static_cast<hccl::CollComm*>(taskInfo.dfxOpInfo_->comm_);
    u32 remoteLocalId = Hccl::DFX_INVALID_RANKID;
    GetCcuCqeErrRemoteLocalIdByRankId(collComm, remoteRankId, remoteLocalId);
    std::string netInstanceId = "";
    GetCcuCqeErrNetInstanceByRankId(collComm, remoteRankId, netInstanceId);
    std::string srcEid = pair.first.Describe();
    std::string dstEid = pair.second.Describe();
    ClusterMoniterGetCcuCqeErrInfo(remoteLocalId, locDeviceId, missionStatus, srcEid, dstEid, netInstanceId);
    return;
}

void CcuTaskException::PrintCcuErrorInfo(uint32_t deviceId, uint16_t status, const Hccl::TaskInfo& taskInfo)
{
    const Hccl::ParaCcu& ccuTaskParam = taskInfo.taskParam_.taskPara.Ccu;
    vector<CcuErrorInfo> errorInfos{};
    HcclResult ret = GetCcuErrorMsg(deviceId, status, ccuTaskParam, errorInfos);
    if (ret != HcclResult::HCCL_SUCCESS || errorInfos.empty()) {
        HCCL_ERROR(
            "Get CCU error info failed. deviceId[%u], dieId[%u], missionId[%u], executeId[%llu].", deviceId,
            ccuTaskParam.dieId, ccuTaskParam.missionId, ccuTaskParam.executeId);
        return;
    }
    PrintCcuErrorLog(errorInfos, taskInfo, deviceId);
    const uint8_t missionStatus = (status >> 8) & 0xFF;
    if (missionStatus >= 0x01
        && missionStatus <= 0x05) { // 如果是UB错误(missionStatus为[0x01, 0x05])，打印Ub Dfx寄存器信息
        PrintCcuUbRegisters(errorInfos, static_cast<s32>(deviceId), taskInfo);
        if (isGetCqeErrInfo) {
            isGetCqeErrInfo = false; // 只获取一次CQE错误信息，避免重复获取
            for (const auto& errorInfo : errorInfos) {
                if (errorInfo.repType == CcuRep::CcuRepType::READ || errorInfo.repType == CcuRep::CcuRepType::WRITE
                    || errorInfo.repType == CcuRep::CcuRepType::BUF_READ
                    || errorInfo.repType == CcuRep::CcuRepType::BUF_WRITE) {
                    GetCcuCqeErrorInfo(errorInfo, taskInfo, deviceId, missionStatus);
                }
            }
        }
        NotifyControlPlaneOnUbError(errorInfos, taskInfo, deviceId, missionStatus);
    }
}

void CcuTaskException::PrintCcuErrorLog(
    const std::vector<CcuErrorInfo>& errorInfos, const Hccl::TaskInfo& taskInfo, u32 deviceId)
{
    if (errorInfos.empty()) {
        return;
    }
    HCCL_ERROR("[CcuTaskException]Task run failed, ccu runtime information is: %s", __func__);
    for (const auto& errorInfo : errorInfos) {
        HCCL_ERROR("[CcuTaskException][%s]", GetCcuErrorMsgByType(errorInfo, taskInfo, deviceId).c_str());
    }
}

HcclResult CcuTaskException::PrintCcuUbRegisters(
    const std::vector<CcuErrorInfo>& errorInfos, s32 devLogicId, [[maybe_unused]] const Hccl::TaskInfo& taskInfo)
{
    std::vector<CcuJetty*> ccuJettys;
    for (const CcuErrorInfo& errorInfo : errorInfos) {
        std::pair<CcuChannelInfo, std::vector<CcuJetty*>> ctx;
        (void)GetCcuJettys(errorInfo, ctx);
        ccuJettys.insert(ccuJettys.end(), ctx.second.begin(), ctx.second.end());
    }
    std::sort(ccuJettys.begin(), ccuJettys.end());
    ccuJettys.erase(std::unique(ccuJettys.begin(), ccuJettys.end()), ccuJettys.end());

    u32 jettyNum = static_cast<u32>(ccuJettys.size());
    CHK_PRT_RET(jettyNum == 0, HCCL_RUN_INFO("[%s]jettyNum[%u], skip", __func__, jettyNum), HCCL_SUCCESS);

    std::vector<JettyStatus> jettyStatusVec;
    std::unordered_map<CtxHandle, std::vector<std::pair<JettyHandle, u32>>> ctxGroups;
    for (u32 i = 0; i < jettyNum; ++i) {
        ctxGroups[ccuJettys[i]->GetCtxHandle()].emplace_back(ccuJettys[i]->GetJettyHandle(), i);
    }
    jettyStatusVec.resize(jettyNum);
    for (auto& [ctxHandle, group] : ctxGroups) {
        std::vector<JettyHandle> handles;
        handles.reserve(group.size());
        for (auto& item : group) {
            handles.push_back(item.first);
        }
        u32 num = static_cast<u32>(handles.size());
        std::vector<JettyStatus> statusVec;
        CHK_RET(HccpBatchQueryJettyStatus(ctxHandle, handles, statusVec, num));
        for (u32 j = 0; j < num; ++j) {
            jettyStatusVec[group[j].second] = statusVec[j];
        }
    }

    for (u32 i = 0; i < jettyNum; ++i) {
        if (jettyStatusVec[i] == JettyStatus::ERROR || jettyStatusVec[i] == JettyStatus::SUSPENDED) {
            HrtRaDumpJettyContext(ccuJettys[i]->GetJettyHandle(), ccuJettys[i]->GetJettyId());
        }
    }

    for (u32 i = 0; i < jettyNum; ++i) {
        if (jettyStatusVec[i] == JettyStatus::ERROR || jettyStatusVec[i] == JettyStatus::SUSPENDED) {
            auto rdmaHandle = ccuJettys[i]->GetRdmaHandle();
            HCCL_ERROR("[%s]jettyId[%u]", __func__, ccuJettys[i]->GetJettyId());
            PrintUbRegisters(devLogicId, rdmaHandle);
            break;
        }
    }
    return HCCL_SUCCESS;
}

uint16_t CcuTaskException::GetChannleIdByCcuErrorInfo(const CcuErrorInfo& errorInfo)
{
    uint16_t channelId = INVALID_U16;
    switch (errorInfo.repType) {
        case CcuRep::CcuRepType::REM_POST_SEM:
        case CcuRep::CcuRepType::REM_WAIT_SEM:
        case CcuRep::CcuRepType::REM_POST_VAR:
            channelId = errorInfo.msg.waitSignal.channelId[0];
            break;
        case CcuRep::CcuRepType::READ:
        case CcuRep::CcuRepType::WRITE:
            channelId = errorInfo.msg.transMem.channelId;
            break;
        case CcuRep::CcuRepType::BUF_READ:
        case CcuRep::CcuRepType::BUF_WRITE:
            channelId = errorInfo.msg.bufTransMem.channelId;
            break;
        default:
            HCCL_RUN_INFO("[%s]repType[%d] does not have jetty", __func__, errorInfo.repType);
            break;
    }
    HCCL_INFO("[%s]repType[%d], channelId[%u]", __func__, errorInfo.repType, channelId);
    return channelId;
}

HcclResult
CcuTaskException::GetCcuJettys(const CcuErrorInfo& errorInfo, std::pair<CcuChannelInfo, std::vector<CcuJetty*>>& ctx)
{
    uint16_t channelId = GetChannleIdByCcuErrorInfo(errorInfo);
    CHK_PRT_RET(channelId == INVALID_U16, HCCL_RUN_INFO("[%s]channelId is invalid, skip", __func__), HCCL_SUCCESS);

    // channelId -> channelHandle
    u64 channelHandle = DFX_INVALID_U64;
    CHK_RET(GetCcuChannelHandleById(channelId, channelHandle));

    // channelHandle -> CcuUrmaChannel
    void* channelPtr{nullptr};
    CHK_RET(static_cast<HcclResult>(HcommChannelGet(channelHandle, &channelPtr)));
    CHK_PTR_NULL(channelPtr);
    auto* channelImpl = dynamic_cast<CcuUrmaChannel*>(static_cast<Channel*>(channelPtr));

    // CcuUrmaChannel -> UrmaEndpoint
    EndpointHandle locEndPointHandle = channelImpl->GetlocEndPointHandle();
    void* endpoint{nullptr};
    CHK_RET(static_cast<HcclResult>(HcommEndpointGet(locEndPointHandle, &endpoint)));
    CHK_PTR_NULL(endpoint);
    UrmaEndpoint* ccuEndpoint = dynamic_cast<UrmaEndpoint*>(static_cast<Endpoint*>(endpoint));

    // 打印localEid和remoteEid
    CHK_PTR_NULL(ccuEndpoint);
    HCCL_ERROR(
        "[GetCcuJettys]localEid[%s], remoteEid[%s]",
        hcomm::logger::CommAddrLogger::ToString(ccuEndpoint->GetEndpointDesc().commAddr).c_str(),
        hcomm::logger::CommAddrLogger::ToString(channelImpl->GetChannelDesc().remoteEndpoint.commAddr).c_str());

    // UrmaEndpoint -> CcuChannelCtxPool
    CcuChannelCtxPool* ccuChannelCtxPool = ccuEndpoint->GetCcuChannelCtxPool();
    CHK_PTR_NULL(ccuChannelCtxPool);

    // CcuChannelCtxPool -> CcuJetty
    auto channelIdKey = std::make_pair(errorInfo.dieId, channelId);
    CHK_RET(ccuChannelCtxPool->GetCcuChannelCtxById(channelIdKey, ctx));
    return HCCL_SUCCESS;
}

HcclResult RaGetAuxInfo(const RdmaHandle rdmaHandle, AuxInfoIn auxInfoIn, AuxInfoOut& auxInfoOut)
{
    HccpAuxInfoIn in;
    in.type = static_cast<HccpAuxInfoInType>(static_cast<int>(auxInfoIn.auxInfoInType));
    if (auxInfoIn.auxInfoInType == AuxInfoInType::AUX_INFO_IN_TYPE_CQE) {
        in.cqe.status = auxInfoIn.cqe.status;
        in.cqe.sR = auxInfoIn.cqe.sR;
    } else if (auxInfoIn.auxInfoInType == AuxInfoInType::AUX_INFO_IN_TYPE_AE) {
        in.ae.eventType = auxInfoIn.ae.eventType;
    }

    HccpAuxInfoOut out;
    auto ret = RaCtxGetAuxInfo(rdmaHandle, &in, &out);
    if (ret != 0) {
        HCCL_ERROR("RaGetAuxInfo failed, ret=[%d]", ret);
        return HCCL_E_NETWORK;
    }

    auxInfoOut.auxInfoNum = out.auxInfoNum;
    for (uint32_t i = 0; i < out.auxInfoNum; i++) {
        auxInfoOut.auxInfoTypes[i] = out.auxInfoType[i];
        auxInfoOut.auxInfoValues[i] = out.auxInfoValue[i];
    }
    return HCCL_SUCCESS;
}
HcclResult CcuTaskException::PrintUbRegisters(s32 devLogicId, RdmaHandle rdmaHandle)
{
    HCCL_INFO("[PrintUbRegister] start, devLogicId[%d], rdmaHandle[%p]", devLogicId, rdmaHandle);
    AuxInfoIn in;
    in.cqe.status = 0xffffffff; // 0xffffffff代表查询所有寄存器
    in.auxInfoInType = AuxInfoInType::AUX_INFO_IN_TYPE_CQE;
    in.cqe.sR = 0;
    AuxInfoOut auxInfo;
    auto ret = RaGetAuxInfo(rdmaHandle, in, auxInfo);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[PrintUbRegister] RaGetAuxInfo failed, devLogicId[%d], rdmaHandle[%p], ret[%d]", devLogicId, rdmaHandle,
            ret);
        return ret;
    }

    uint16_t isAuxInfoExisted{false};
    for (u32 i = 0; i < auxInfo.auxInfoNum; i++) {
        if (auxInfo.auxInfoValues[i]) { // 非零进行打印
            isAuxInfoExisted = true;
            HCCL_ERROR(
                "devLogicId[%d], cqe_aux_info_type[%u], cqe_aux_info_value[0x%x]", devLogicId, auxInfo.auxInfoTypes[i],
                auxInfo.auxInfoValues[i]);
        } else {
            HCCL_INFO(
                "devLogicId[%d], cqe_aux_info_type[%u], cqe_aux_info_value[0x%x]", devLogicId, auxInfo.auxInfoTypes[i],
                auxInfo.auxInfoValues[i]);
        }
    }
    if (!isAuxInfoExisted) {
        HCCL_ERROR("devLogicId[%d], all aux_info values are zero.", devLogicId);
    }
    return HCCL_SUCCESS;
}

string CcuTaskException::GetCcuLenErrorMsg(const uint64_t len)
{
    if ((0 < len) && (len <= CCU_MSG_256MB_LEN)) {
        return "";
    }
    return Hccl::StringFormat("ccu transMem Len[%llu]B > 256MB or is zero, not support!", len);
}

string
CcuTaskException::GetCcuErrorMsgLoop(const CcuErrorInfo& ccuErrorInfo, const Hccl::TaskInfo& taskInfo, u32 deviceId)
{
    (void)taskInfo;
    (void)deviceId;
    return Hccl::StringFormat(
        "InstrId[%u]: Loop startInstrId[%u], endInstrId[%u], executorId[%u], "
        "totalIter[%u], curIter[%u], addressStride[0x%x]",
        ccuErrorInfo.instrId, ccuErrorInfo.msg.loop.startInstrId, ccuErrorInfo.msg.loop.endInstrId,
        ccuErrorInfo.msg.loop.loopEngineId, ccuErrorInfo.msg.loop.loopCnt, ccuErrorInfo.msg.loop.loopCurrentCnt,
        ccuErrorInfo.msg.loop.addrStride);
}

string CcuTaskException::GetCcuErrorMsgLoopGroup(
    const CcuErrorInfo& ccuErrorInfo, const Hccl::TaskInfo& taskInfo, u32 deviceId)
{
    (void)taskInfo;
    (void)deviceId;
    return Hccl::StringFormat(
        "InstrId[%u]: LoopGroup startLoopInsId[%u], loopInsCnt[%u], "
        "expandOffset[%u], expandCnt[%u]",
        ccuErrorInfo.instrId, ccuErrorInfo.msg.loopGroup.startLoopInsId, ccuErrorInfo.msg.loopGroup.loopInsCnt,
        ccuErrorInfo.msg.loopGroup.expandOffset, ccuErrorInfo.msg.loopGroup.expandCnt);
}

string CcuTaskException::GetCcuErrorMsgLocPostSem(
    const CcuErrorInfo& ccuErrorInfo, const Hccl::TaskInfo& taskInfo, u32 deviceId)
{
    (void)taskInfo;
    (void)deviceId;
    return Hccl::StringFormat(
        "InstrId[%u]: Set sem[%u], semValue[0x%04x], mask[0x%04x]", ccuErrorInfo.instrId,
        ccuErrorInfo.msg.waitSignal.signalId, ccuErrorInfo.msg.waitSignal.signalValue,
        ccuErrorInfo.msg.waitSignal.signalMask);
}

string CcuTaskException::GetCcuErrorMsgLocWaitEvent(
    const CcuErrorInfo& ccuErrorInfo, const Hccl::TaskInfo& taskInfo, u32 deviceId)
{
    (void)taskInfo;
    (void)deviceId;
    return Hccl::StringFormat(
        "InstrId[%u]: LocWaitEvent[%u], semValue[0x%04x], mask[0x%04x]", ccuErrorInfo.instrId,
        ccuErrorInfo.msg.waitSignal.signalId, ccuErrorInfo.msg.waitSignal.signalValue,
        ccuErrorInfo.msg.waitSignal.signalMask);
}

string CcuTaskException::GetCcuErrorMsgLocWaitNotify(
    const CcuErrorInfo& ccuErrorInfo, const Hccl::TaskInfo& taskInfo, u32 deviceId)
{
    (void)taskInfo;
    (void)deviceId;
    return Hccl::StringFormat(
        "InstrId[%u]: LocWaitNotify[%u], semValue[0x%04x], mask[0x%04x]", ccuErrorInfo.instrId,
        ccuErrorInfo.msg.waitSignal.signalId, ccuErrorInfo.msg.waitSignal.signalValue,
        ccuErrorInfo.msg.waitSignal.signalMask);
}

string CcuTaskException::GetCcuErrorMsgRemPostSem(
    const CcuErrorInfo& ccuErrorInfo, const Hccl::TaskInfo& taskInfo, u32 deviceId)
{
    return Hccl::StringFormat(
        "InstrId[%u]: Post, Use sem[%u], mask[0x%04x], rankId[%d]", ccuErrorInfo.instrId,
        ccuErrorInfo.msg.waitSignal.signalId, ccuErrorInfo.msg.waitSignal.signalMask,
        GetRankIdByChannelId(ccuErrorInfo.msg.waitSignal.channelId[0], taskInfo, deviceId));
}

string CcuTaskException::GetCcuErrorMsgRemWaitSem(
    const CcuErrorInfo& ccuErrorInfo, const Hccl::TaskInfo& taskInfo, u32 deviceId)
{
    return Hccl::StringFormat(
        "InstrId[%u]: Wait, Use sem[%u], semValue[0x%04x], mask[0x%04x], rankId[%d]", ccuErrorInfo.instrId,
        ccuErrorInfo.msg.waitSignal.signalId, ccuErrorInfo.msg.waitSignal.signalValue,
        ccuErrorInfo.msg.waitSignal.signalMask,
        GetRankIdByChannelId(ccuErrorInfo.msg.waitSignal.channelId[0], taskInfo, deviceId));
}

string CcuTaskException::GetCcuErrorMsgRemPostVar(
    const CcuErrorInfo& ccuErrorInfo, const Hccl::TaskInfo& taskInfo, u32 deviceId)
{
    return Hccl::StringFormat(
        "InstrId[%u]: Post Variable[0x%016llx] To Param[%u], Use sem[%u], mask[0x%04x], rankId[%d]",
        ccuErrorInfo.instrId, ccuErrorInfo.msg.waitSignal.paramValue, ccuErrorInfo.msg.waitSignal.paramId,
        ccuErrorInfo.msg.waitSignal.signalId, ccuErrorInfo.msg.waitSignal.signalMask,
        GetRankIdByChannelId(ccuErrorInfo.msg.waitSignal.channelId[0], taskInfo, deviceId));
}

string CcuTaskException::GetCcuErrorMsgPostSharedSem(
    const CcuErrorInfo& ccuErrorInfo, const Hccl::TaskInfo& taskInfo, u32 deviceId)
{
    (void)taskInfo;
    (void)deviceId;
    return Hccl::StringFormat(
        "InstrId[%u]: Post, Use sem[%u], mask[0x%04x]", ccuErrorInfo.instrId, ccuErrorInfo.msg.waitSignal.signalId,
        ccuErrorInfo.msg.waitSignal.signalMask);
}

string
CcuTaskException::GetCcuErrorMsgRead(const CcuErrorInfo& ccuErrorInfo, const Hccl::TaskInfo& taskInfo, u32 deviceId)
{
    auto pair = GetAddrPairByChannelId(ccuErrorInfo.msg.transMem.channelId, taskInfo, deviceId);
    string printMsg = GetCcuLenErrorMsg(ccuErrorInfo.msg.transMem.len);
    return Hccl::StringFormat(
        "InstrId[%u]: Read Memory[0x%016llx] To Memory[0x%016llx], Len[%llu], "
        "Set sem[%u] with mask[0x%04x], remoteRankId[%d], srcEID[%s], dstEID[%s] %s",
        ccuErrorInfo.instrId, ccuErrorInfo.msg.transMem.rmtAddr, ccuErrorInfo.msg.transMem.locAddr,
        ccuErrorInfo.msg.transMem.len, ccuErrorInfo.msg.transMem.signalId, ccuErrorInfo.msg.transMem.signalMask,
        GetRankIdByChannelId(ccuErrorInfo.msg.transMem.channelId, taskInfo, deviceId), pair.first.Describe().c_str(),
        pair.second.Describe().c_str(), printMsg.c_str());
}

string
CcuTaskException::GetCcuErrorMsgWrite(const CcuErrorInfo& ccuErrorInfo, const Hccl::TaskInfo& taskInfo, u32 deviceId)
{
    auto pair = GetAddrPairByChannelId(ccuErrorInfo.msg.transMem.channelId, taskInfo, deviceId);
    string printMsg = GetCcuLenErrorMsg(ccuErrorInfo.msg.transMem.len);
    return Hccl::StringFormat(
        "InstrId[%u]: Write Memory[0x%016llx] to Memory[0x%016llx], Len[%llu], "
        "Set sem[%u] with mask[0x%04x], remoteRankId[%d], srcEID[%s], dstEID[%s] %s",
        ccuErrorInfo.instrId, ccuErrorInfo.msg.transMem.locAddr, ccuErrorInfo.msg.transMem.rmtAddr,
        ccuErrorInfo.msg.transMem.len, ccuErrorInfo.msg.transMem.signalId, ccuErrorInfo.msg.transMem.signalMask,
        GetRankIdByChannelId(ccuErrorInfo.msg.transMem.channelId, taskInfo, deviceId), pair.first.Describe().c_str(),
        pair.second.Describe().c_str(), printMsg.c_str());
}

string
CcuTaskException::GetCcuErrorMsgLocalCpy(const CcuErrorInfo& ccuErrorInfo, const Hccl::TaskInfo& taskInfo, u32 deviceId)
{
    (void)taskInfo;
    (void)deviceId;
    string printMsg = GetCcuLenErrorMsg(ccuErrorInfo.msg.transMem.len);
    return Hccl::StringFormat(
        "InstrId[%u]: Read Memory[0x%016llx] to Memory[0x%016llx], Len[%llu], "
        "Set sem[%u] with mask[0x%04x] %s",
        ccuErrorInfo.instrId, ccuErrorInfo.msg.transMem.locAddr, ccuErrorInfo.msg.transMem.rmtAddr,
        ccuErrorInfo.msg.transMem.len, ccuErrorInfo.msg.transMem.signalId, ccuErrorInfo.msg.transMem.signalMask,
        printMsg.c_str());
}

string CcuTaskException::GetCcuErrorMsgLocalReduce(
    const CcuErrorInfo& ccuErrorInfo, const Hccl::TaskInfo& taskInfo, u32 deviceId)
{
    (void)taskInfo;
    (void)deviceId;
    string printMsg = GetCcuLenErrorMsg(ccuErrorInfo.msg.transMem.len);
    return Hccl::StringFormat(
        "InstrId[%u]: Read Memory[0x%016llx] to Memory[0x%016llx], Len[%llu], "
        "Set sem[%u] with mask[0x%04x], dataType[%u], opType[%u] %s",
        ccuErrorInfo.instrId, ccuErrorInfo.msg.transMem.locAddr, ccuErrorInfo.msg.transMem.rmtAddr,
        ccuErrorInfo.msg.transMem.len, ccuErrorInfo.msg.transMem.signalId, ccuErrorInfo.msg.transMem.signalMask,
        ccuErrorInfo.msg.transMem.dataType, ccuErrorInfo.msg.transMem.opType, printMsg.c_str());
}

string
CcuTaskException::GetCcuErrorMsgBufRead(const CcuErrorInfo& ccuErrorInfo, const Hccl::TaskInfo& taskInfo, u32 deviceId)
{
    auto pair = GetAddrPairByChannelId(ccuErrorInfo.msg.bufTransMem.channelId, taskInfo, deviceId);
    string printMsg = GetCcuLenErrorMsg(ccuErrorInfo.msg.bufTransMem.len);
    return Hccl::StringFormat(
        "InstrId[%u]: Read Rmt Mem[0x%016llx] To CcuBuffer[%u], Len[%llu], "
        "sem[%u], mask[0x%04x], remoteRankId[%d], srcEID[%s], dstEID[%s] %s",
        ccuErrorInfo.instrId, ccuErrorInfo.msg.bufTransMem.addr, ccuErrorInfo.msg.bufTransMem.bufId,
        ccuErrorInfo.msg.bufTransMem.len, ccuErrorInfo.msg.bufTransMem.signalId,
        ccuErrorInfo.msg.bufTransMem.signalMask,
        GetRankIdByChannelId(ccuErrorInfo.msg.bufTransMem.channelId, taskInfo, deviceId), pair.first.Describe().c_str(),
        pair.second.Describe().c_str(), printMsg.c_str());
}

string
CcuTaskException::GetCcuErrorMsgBufWrite(const CcuErrorInfo& ccuErrorInfo, const Hccl::TaskInfo& taskInfo, u32 deviceId)
{
    auto pair = GetAddrPairByChannelId(ccuErrorInfo.msg.bufTransMem.channelId, taskInfo, deviceId);
    string printMsg = GetCcuLenErrorMsg(ccuErrorInfo.msg.bufTransMem.len);
    return Hccl::StringFormat(
        "InstrId[%u]: Write CcuBuffer[%u] To Rmt Mem[0x%016llx], Len[%llu], "
        "sem[%u], mask[0x%04x], remoteRankId[%d], srcEID[%s], dstEID[%s] %s",
        ccuErrorInfo.instrId, ccuErrorInfo.msg.bufTransMem.bufId, ccuErrorInfo.msg.bufTransMem.addr,
        ccuErrorInfo.msg.bufTransMem.len, ccuErrorInfo.msg.bufTransMem.signalId,
        ccuErrorInfo.msg.bufTransMem.signalMask,
        GetRankIdByChannelId(ccuErrorInfo.msg.bufTransMem.channelId, taskInfo, deviceId), pair.first.Describe().c_str(),
        pair.second.Describe().c_str(), printMsg.c_str());
}

string CcuTaskException::GetCcuErrorMsgBufLocRead(
    const CcuErrorInfo& ccuErrorInfo, const Hccl::TaskInfo& taskInfo, u32 deviceId)
{
    (void)taskInfo;
    (void)deviceId;
    string printMsg = GetCcuLenErrorMsg(ccuErrorInfo.msg.bufTransMem.len);
    return Hccl::StringFormat(
        "InstrId[%u]: Read Loc Mem[0x%016llx] To CcuBuffer[%u], Len[%llu], sem[%u], mask[0x%04x] %s",
        ccuErrorInfo.instrId, ccuErrorInfo.msg.bufTransMem.addr, ccuErrorInfo.msg.bufTransMem.bufId,
        ccuErrorInfo.msg.bufTransMem.len, ccuErrorInfo.msg.bufTransMem.signalId,
        ccuErrorInfo.msg.bufTransMem.signalMask, printMsg.c_str());
}

string CcuTaskException::GetCcuErrorMsgBufLocWrite(
    const CcuErrorInfo& ccuErrorInfo, const Hccl::TaskInfo& taskInfo, u32 deviceId)
{
    (void)taskInfo;
    (void)deviceId;
    string printMsg = GetCcuLenErrorMsg(ccuErrorInfo.msg.bufTransMem.len);
    return Hccl::StringFormat(
        "InstrId[%u]: Write CcuBuffer[%u] To Loc Mem[0x%016llx], Len[%llu], sem[%u], mask[0x%04x] %s",
        ccuErrorInfo.instrId, ccuErrorInfo.msg.bufTransMem.bufId, ccuErrorInfo.msg.bufTransMem.addr,
        ccuErrorInfo.msg.bufTransMem.len, ccuErrorInfo.msg.bufTransMem.signalId,
        ccuErrorInfo.msg.bufTransMem.signalMask, printMsg.c_str());
}

string CcuTaskException::GetCcuErrorMsgBufReduce(
    const CcuErrorInfo& ccuErrorInfo, const Hccl::TaskInfo& taskInfo, u32 deviceId)
{
    (void)taskInfo;
    (void)deviceId;
    stringstream buffIds;
    for (uint32_t i = 0; i < BUF_REDUCE_ID_SIZE; ++i) {
        const auto buffId = ccuErrorInfo.msg.bufReduce.bufIds[i];
        if (buffId == UINT16_MAX) {
            break;
        }
        if (i != 0) {
            buffIds << ", ";
        }
        buffIds << to_string(buffId);
    }

    return Hccl::StringFormat(
        "InstrId[%u]: Buffer Reduce count[%u], dataType[%u], outputDataType[%u], opType[%u], "
        "sem[%u], mask[0x%04x], CcuBuffers[%s]",
        ccuErrorInfo.instrId, ccuErrorInfo.msg.bufReduce.count, ccuErrorInfo.msg.bufReduce.dataType,
        ccuErrorInfo.msg.bufReduce.outputDataType, ccuErrorInfo.msg.bufReduce.opType,
        ccuErrorInfo.msg.bufReduce.signalId, ccuErrorInfo.msg.bufReduce.signalMask, buffIds.str().c_str());
}

string CcuTaskException::GetCcuErrorMsgWriteVarAtomic(
    const CcuErrorInfo& ccuErrorInfo, const Hccl::TaskInfo& taskInfo, u32 deviceId)
{
    (void)taskInfo;
    (void)deviceId;
    return Hccl::StringFormat(
        "InstrId[%u]: WriteVarAtomic, dstId[%u], dstValue[0x%016llx], Set sem[%u], mask[0x%04x]", ccuErrorInfo.instrId,
        ccuErrorInfo.msg.writeVarAtomic.dstId, ccuErrorInfo.msg.writeVarAtomic.dstValue,
        ccuErrorInfo.msg.writeVarAtomic.signalId, ccuErrorInfo.msg.writeVarAtomic.signalMask);
}

string CcuTaskException::GetCcuErrorMsgWriteWithCntInc(
    const CcuErrorInfo& ccuErrorInfo, const Hccl::TaskInfo& taskInfo, u32 deviceId)
{
    (void)taskInfo;
    (void)deviceId;
    string printMsg = GetCcuLenErrorMsg(ccuErrorInfo.msg.writeWithCntInc.len);
    return Hccl::StringFormat(
        "InstrId[%u]: WriteWithCntInc, Memory[0x%016llx] to Memory[0x%016llx], Len[%llu], incCntAddr[0x%016llx] %s",
        ccuErrorInfo.instrId, ccuErrorInfo.msg.writeWithCntInc.locAddr, ccuErrorInfo.msg.writeWithCntInc.rmtAddr,
        ccuErrorInfo.msg.writeWithCntInc.len, ccuErrorInfo.msg.writeWithCntInc.incCntAddr, printMsg.c_str());
}

string CcuTaskException::GetCcuErrorMsgCascCntWait(
    const CcuErrorInfo& ccuErrorInfo, const Hccl::TaskInfo& taskInfo, u32 deviceId)
{
    (void)taskInfo;
    (void)deviceId;
    return Hccl::StringFormat(
        "InstrId[%u]: CascCntWait, targetValue[0x%016llx], currentCntValue[0x%016llx]", ccuErrorInfo.instrId,
        ccuErrorInfo.msg.cascCntWait.targetValue, ccuErrorInfo.msg.cascCntWait.currentCntValue);
}

string CcuTaskException::GetCcuErrorMsgCascCntClear(
    const CcuErrorInfo& ccuErrorInfo, const Hccl::TaskInfo& taskInfo, u32 deviceId)
{
    (void)taskInfo;
    (void)deviceId;
    return Hccl::StringFormat(
        "InstrId[%u]: CascCntClear, Xns[%u~%u]", ccuErrorInfo.instrId, ccuErrorInfo.msg.cascCntClear.wishCntXnIdFirst,
        ccuErrorInfo.msg.cascCntClear.expectedCntXn);
}

string CcuTaskException::GetCcuErrorMsgLoadAddImm(
    const CcuErrorInfo& ccuErrorInfo, const Hccl::TaskInfo& taskInfo, u32 deviceId)
{
    (void)taskInfo;
    (void)deviceId;
    return Hccl::StringFormat(
        "InstrId[%u]: LoadAddImm, srcId[%u], srcOffsetId[%u], dstId[%u], immAddValue[%u], "
        "srcValue[0x%016llx], srcOffsetValue[0x%016llx]",
        ccuErrorInfo.instrId, ccuErrorInfo.msg.loadAddImm.srcId, ccuErrorInfo.msg.loadAddImm.srcOffsetId,
        ccuErrorInfo.msg.loadAddImm.dstId, ccuErrorInfo.msg.loadAddImm.immAddValue,
        ccuErrorInfo.msg.loadAddImm.srcValue, ccuErrorInfo.msg.loadAddImm.srcOffsetValue);
}

string CcuTaskException::GetCcuErrorMsgStoreAddImm(
    const CcuErrorInfo& ccuErrorInfo, const Hccl::TaskInfo& taskInfo, u32 deviceId)
{
    (void)taskInfo;
    (void)deviceId;
    return Hccl::StringFormat(
        "InstrId[%u]: StoreAddImm, dstId[%u], dstOffsetId[%u], srcId[%u], immAddValue[%u], "
        "dstValue[0x%016llx], dstOffsetValue[0x%016llx]",
        ccuErrorInfo.instrId, ccuErrorInfo.msg.storeAddImm.dstId, ccuErrorInfo.msg.storeAddImm.dstOffsetId,
        ccuErrorInfo.msg.storeAddImm.srcId, ccuErrorInfo.msg.storeAddImm.immAddValue,
        ccuErrorInfo.msg.storeAddImm.dstValue, ccuErrorInfo.msg.storeAddImm.dstOffsetValue);
}

string CcuTaskException::GetCcuErrorMsgDefault(const CcuErrorInfo& ccuErrorInfo)
{
    return Hccl::StringFormat(
        "InstrId[%u]: CcuErrorType[%s]", ccuErrorInfo.instrId, ccuErrorInfo.type.Describe().c_str());
}

string CcuTaskException::GetCcuErrorMsgMission(const CcuErrorInfo& ccuErrorInfo)
{
    return Hccl::StringFormat(
        "InstrId[%u]: dieId[%u], missionId[%u], missionError[%s]", ccuErrorInfo.instrId, ccuErrorInfo.dieId,
        ccuErrorInfo.missionId, ccuErrorInfo.msg.mission.missionError);
}

HcclResult CcuTaskException::GetCcuChannelHandleById(u16 channelId, u64& channelHandle)
{
    std::lock_guard<std::mutex> lock(g_channelMapMutex);
    auto it = g_channelIdToHandle.find(channelId);
    if (it == g_channelIdToHandle.end()) {
        HCCL_ERROR("[%s]channelId[%u] not found", __func__, channelId);
        for (auto info : g_channelIdToHandle) {
            HCCL_ERROR("[%s]g_channelIdToHandle[%u]=[0x%llx]", __func__, info.first, info.second);
        }
        return HCCL_E_NOT_FOUND;
    }
    channelHandle = it->second;
    return HCCL_SUCCESS;
}

RankId CcuTaskException::GetRankIdByChannelId(uint16_t channelId, const Hccl::TaskInfo& taskInfo, u32 deviceId)
{
    if (taskInfo.taskParam_.taskType != Hccl::TaskParamType::TASK_CCU) {
        HCCL_ERROR("[%s]taskType[%s] is not CCU.", __func__, taskInfo.taskParam_.taskType.Describe().c_str());
        return INVALID_UINT;
    }
    if (taskInfo.dfxOpInfo_ == nullptr || taskInfo.dfxOpInfo_->comm_ == nullptr) {
        HCCL_ERROR("[%s]dfxOpInfo[%p] or comm is nullptr.", __func__, taskInfo.dfxOpInfo_);
        return INVALID_UINT;
    }

    u64 channelHandle = DFX_INVALID_U64;
    if (GetCcuChannelHandleById(channelId, channelHandle) != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[%s]GetCcuChannelHandleById fail, deviceId[%u], channelId[%u], channelHandle[0x%llx]", __func__, deviceId,
            channelId, channelHandle);
        return INVALID_UINT;
    }

    hccl::CollComm* collComm = static_cast<hccl::CollComm*>(taskInfo.dfxOpInfo_->comm_);
    u32 remoteRank = INVALID_UINT;
    if (hccl::HcclCommDfx::GetChannelRemoteRankId(collComm->GetCommId(), channelHandle, remoteRank) != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[%s]GetChannelRemoteRankId fail, channelHandle[0x%llx], commId[%s]", __func__, channelHandle,
            collComm->GetCommId().c_str());
        return INVALID_UINT;
    }

    HCCL_INFO(
        "[%s]channelId[%u], deviceId[%u], channelHandle[0x%llx], commId[%s], remoteRank[%u]", __func__, channelId,
        deviceId, channelHandle, collComm->GetCommId().c_str(), remoteRank);
    return remoteRank;
}

std::pair<Hccl::IpAddress, Hccl::IpAddress>
CcuTaskException::GetAddrPairByChannelId(uint16_t channelId, const Hccl::TaskInfo& taskInfo, u32 deviceId)
{
    std::pair<Hccl::IpAddress, Hccl::IpAddress> dummy = {Hccl::IpAddress(), Hccl::IpAddress()};
    if (taskInfo.taskParam_.taskType != Hccl::TaskParamType::TASK_CCU) {
        HCCL_ERROR(
            "[TaskException][%s]Get AddrPair failed, task type error[%s]", __func__,
            taskInfo.taskParam_.Describe().c_str());
        return dummy;
    }
    if (taskInfo.dfxOpInfo_ == nullptr || taskInfo.dfxOpInfo_->comm_ == nullptr) {
        HCCL_ERROR("[TaskException][%s]Get AddrPair failed, communicator is nullptr.", __func__);
        return dummy;
    }

    u64 channelHandle = DFX_INVALID_U64;
    if (GetCcuChannelHandleById(channelId, channelHandle) != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[%s]GetCcuChannelHandleById fail, deviceId[%u], channelId[%u], channelHandle[0x%llx]", __func__, deviceId,
            channelId, channelHandle);
        return dummy;
    }

    void* channelPtr{nullptr};
    HcclResult ret = static_cast<HcclResult>(HcommChannelGet(channelHandle, &channelPtr));
    if (ret != HCCL_SUCCESS || channelPtr == nullptr) {
        HCCL_ERROR(
            "[%s]HcommChannelGet failed, ret[%d], channelHandle[0x%llx], channelPtr[%p]", __func__, ret, channelHandle,
            channelPtr);
        return dummy;
    }
    auto* channelImpl = dynamic_cast<CcuUrmaChannel*>(static_cast<Channel*>(channelPtr));

    // 获取locAddr
    EndpointHandle locEndPointHandle = channelImpl->GetlocEndPointHandle();
    void* endpoint{nullptr};
    ret = static_cast<HcclResult>(HcommEndpointGet(locEndPointHandle, &endpoint));
    if (ret != HCCL_SUCCESS || endpoint == nullptr) {
        HCCL_ERROR(
            "[%s]HcommEndpointGet failed, ret[%d], locEndPointHandle[%p], endpoint[%p], channelId[%u]", __func__, ret,
            locEndPointHandle, endpoint, channelId);
        return dummy;
    }
    UrmaEndpoint* ccuEndpoint = dynamic_cast<UrmaEndpoint*>(static_cast<Endpoint*>(endpoint));
    const EndpointDesc& locEndpointDesc = ccuEndpoint->GetEndpointDesc();
    HcclResult locRet = CommAddrToIpAddress(locEndpointDesc.commAddr, dummy.first);

    // 获取remoteAddr
    HcommChannelDesc remoteChannelDesc = channelImpl->GetChannelDesc();
    HcclResult remRet = CommAddrToIpAddress(remoteChannelDesc.remoteEndpoint.commAddr, dummy.second);
    HCCL_INFO(
        "[%s]channelId[%u], channelHandle[0x%llx], locRet[%d], locIpAddr[%s], remRet[%d], remIpAddr[%s]", __func__,
        channelId, channelHandle, locRet, dummy.first.Describe().c_str(), remRet, dummy.second.Describe().c_str());
    return dummy;
}

string
CcuTaskException::GetCcuErrorMsgByType(const CcuErrorInfo& ccuErrorInfo, const Hccl::TaskInfo& taskInfo, u32 deviceId)
{
    if (ccuErrorInfo.type == CcuErrorType::MISSION) {
        return GetCcuErrorMsgMission(ccuErrorInfo);
    }

    using GetCcuErrorMsgFunc
        = string (*)(const CcuErrorInfo& ccuErrorInfo, const Hccl::TaskInfo& taskInfo, u32 deviceId);
    static const map<CcuRep::CcuRepType, GetCcuErrorMsgFunc> HANDLER_MAP{
        {CcuRep::CcuRepType::LOOP, &CcuTaskException::GetCcuErrorMsgLoop},
        {CcuRep::CcuRepType::LOOPGROUP, &CcuTaskException::GetCcuErrorMsgLoopGroup},
        {CcuRep::CcuRepType::LOC_RECORD_EVENT, &CcuTaskException::GetCcuErrorMsgLocPostSem},
        {CcuRep::CcuRepType::LOC_WAIT_EVENT, &CcuTaskException::GetCcuErrorMsgLocWaitEvent},
        {CcuRep::CcuRepType::LOC_WAIT_NOTIFY, &CcuTaskException::GetCcuErrorMsgLocWaitNotify},
        {CcuRep::CcuRepType::REM_POST_SEM, &CcuTaskException::GetCcuErrorMsgRemPostSem},
        {CcuRep::CcuRepType::REM_WAIT_SEM, &CcuTaskException::GetCcuErrorMsgRemWaitSem},
        {CcuRep::CcuRepType::REM_POST_VAR, &CcuTaskException::GetCcuErrorMsgRemPostVar},
        {CcuRep::CcuRepType::RECORD_SHARED_NOTIFY, &CcuTaskException::GetCcuErrorMsgPostSharedSem},
        {CcuRep::CcuRepType::READ, &CcuTaskException::GetCcuErrorMsgRead},
        {CcuRep::CcuRepType::WRITE, &CcuTaskException::GetCcuErrorMsgWrite},
        {CcuRep::CcuRepType::LOCAL_CPY, &CcuTaskException::GetCcuErrorMsgLocalCpy},
        {CcuRep::CcuRepType::LOCAL_REDUCE, &CcuTaskException::GetCcuErrorMsgLocalReduce},
        {CcuRep::CcuRepType::BUF_READ, &CcuTaskException::GetCcuErrorMsgBufRead},
        {CcuRep::CcuRepType::BUF_WRITE, &CcuTaskException::GetCcuErrorMsgBufWrite},
        {CcuRep::CcuRepType::BUF_LOC_READ, &CcuTaskException::GetCcuErrorMsgBufLocRead},
        {CcuRep::CcuRepType::BUF_LOC_WRITE, &CcuTaskException::GetCcuErrorMsgBufLocWrite},
        {CcuRep::CcuRepType::BUF_REDUCE, &CcuTaskException::GetCcuErrorMsgBufReduce},
        {CcuRep::CcuRepType::WRITE_VAR_ATOMIC, &CcuTaskException::GetCcuErrorMsgWriteVarAtomic},
        {CcuRep::CcuRepType::WRITE_WITH_CNT_INC, &CcuTaskException::GetCcuErrorMsgWriteWithCntInc},
        {CcuRep::CcuRepType::CASC_CNT_WAIT, &CcuTaskException::GetCcuErrorMsgCascCntWait},
        {CcuRep::CcuRepType::CASC_CNT_CLEAR, &CcuTaskException::GetCcuErrorMsgCascCntClear},
        {CcuRep::CcuRepType::LOAD_ADD_IMM, &CcuTaskException::GetCcuErrorMsgLoadAddImm},
        {CcuRep::CcuRepType::STORE_ADD_IMM, &CcuTaskException::GetCcuErrorMsgStoreAddImm}};

    const auto funcIt = HANDLER_MAP.find(ccuErrorInfo.repType);
    if (funcIt == HANDLER_MAP.end()) {
        return GetCcuErrorMsgDefault(ccuErrorInfo);
    } else {
        return funcIt->second(ccuErrorInfo, taskInfo, deviceId);
    }
}

void CcuTaskException::NotifyControlPlaneOnUbError(
    const std::vector<CcuErrorInfo>& errorInfos, const Hccl::TaskInfo& taskInfo, u32 deviceId, uint8_t missionStatus)
{
    uint32_t devPhyId = 0;
    EXCEPTION_CATCH(devPhyId = Hccl::HrtGetDevicePhyIdByUserDevId(static_cast<s32>(deviceId)), return);
    struct RaInfo raInfo = {};
    raInfo.mode = NETWORK_OFFLINE;
    raInfo.phyId = devPhyId;

    const bool supported = RaHasCapability(&raInfo, RA_CAP_UDMA_NOTIFY_EVENT);
    if (!supported) {
        HCCL_WARNING("[%s]RaHasCapability returned false, skip notify control plane, devPhyId[%u]", __func__, devPhyId);
        return;
    }

    std::vector<CcuJetty*> ccuJettys;
    std::vector<uint16_t> channelIds;
    std::vector<JettyHandle> jettyHandles;
    for (const CcuErrorInfo& errorInfo : errorInfos) {
        uint16_t channelId = GetChannleIdByCcuErrorInfo(errorInfo);
        if (channelId == INVALID_U16) {
            continue;
        }

        std::pair<CcuChannelInfo, std::vector<CcuJetty*>> ctx;
        if (GetCcuJettys(errorInfo, ctx) != HCCL_SUCCESS || ctx.second.empty()) {
            continue;
        }

        for (auto* jetty : ctx.second) {
            if (std::find(ccuJettys.begin(), ccuJettys.end(), jetty) == ccuJettys.end()) {
                ccuJettys.push_back(jetty);
                jettyHandles.push_back(jetty->GetJettyHandle());
                // 取遍历到的第一个channelId，通知jetty(tpn)出错了，只带这个channelId的eid信息
                channelIds.push_back(channelId);
            }
        }
    }

    u32 jettyNum = static_cast<u32>(ccuJettys.size());
    if (jettyNum == 0) {
        return;
    }

    std::vector<JettyStatus> jettyStatusVec;
    std::unordered_map<CtxHandle, std::vector<std::pair<JettyHandle, u32>>> ctxGroups;
    for (u32 i = 0; i < jettyNum; ++i) {
        ctxGroups[ccuJettys[i]->GetCtxHandle()].emplace_back(jettyHandles[i], i);
    }
    jettyStatusVec.resize(jettyNum);
    for (auto& [ctxHandle, group] : ctxGroups) {
        std::vector<JettyHandle> handles;
        handles.reserve(group.size());
        for (auto& item : group) {
            handles.push_back(item.first);
        }
        u32 num = static_cast<u32>(handles.size());
        std::vector<JettyStatus> statusVec;

        if (HccpBatchQueryJettyStatus(ctxHandle, handles, statusVec, num) != HCCL_SUCCESS) {
            HCCL_ERROR(
                "[%s]HccpBatchQueryJettyStatus failed, skip this ctx group, ctxHandle[%p]", __func__,
                static_cast<const void*>(ctxHandle));
            continue; // 跳过失败分组，其余分组继续；该组 jettyStatusVec 保持默认值，后续状态过滤自然跳过
        }

        for (u32 j = 0; j < num; ++j) {
            jettyStatusVec[group[j].second] = statusVec[j];
        }
    }

    for (u32 i = 0; i < jettyNum; ++i) {
        if (jettyStatusVec[i] != JettyStatus::ERROR && jettyStatusVec[i] != JettyStatus::SUSPENDED) {
            continue;
        }

        RdmaHandle rdmaHandle = static_cast<RdmaHandle>(ccuJettys[i]->GetCtxHandle());
        if (rdmaHandle == nullptr) {
            HCCL_ERROR("[%s]rdmaHandle is nullptr, skip", __func__);
            continue;
        }

        auto addrPair = GetAddrPairByChannelId(channelIds[i], taskInfo, deviceId);

        struct CtxNotifyEvent event = {};
        event.serviceType = URMA_TYPE;
        event.errorType = missionStatus;
        s32 sRet = memcpy_s(
            event.srcEid.raw, sizeof(event.srcEid.raw), addrPair.first.GetEid().raw,
            sizeof(addrPair.first.GetEid().raw));
        if (sRet != EOK) {
            HCCL_ERROR("[%s]memcpy_s srcEid failed, ret[%d]", __func__, sRet);
            continue;
        }
        sRet = memcpy_s(
            event.dstEid.raw, sizeof(event.dstEid.raw), addrPair.second.GetEid().raw,
            sizeof(addrPair.second.GetEid().raw));
        if (sRet != EOK) {
            HCCL_ERROR("[%s]memcpy_s dstEid failed, ret[%d]", __func__, sRet);
            continue;
        }
        event.errorInfo.tpn = ccuJettys[i]->GetTpn();

        int32_t retCode = RaCtxNotifyEvent(rdmaHandle, &event);
        std::string eventInfo = Hccl::StringFormat(
            "devPhyId[%u], rdmaHandle[%p], jettyHandle[%p], serviceType[%u], errorType[%u], tpn[%u], "
            "srcEid[%s], dstEid[%s]",
            devPhyId, static_cast<const void*>(rdmaHandle), static_cast<const void*>(jettyHandles[i]),
            event.serviceType, event.errorType, event.errorInfo.tpn, addrPair.first.Describe().c_str(),
            addrPair.second.Describe().c_str());
        if (retCode != 0) {
            HCCL_ERROR("[%s]RaCtxNotifyEvent failed, ret[%d], %s", __func__, retCode, eventInfo.c_str());
        } else {
            HCCL_ERROR("[%s]notify control plane finish, %s", __func__, eventInfo.c_str());
        }
    }
}

} // namespace hcomm
