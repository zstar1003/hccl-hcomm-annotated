/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aicpu_kfc_process.h"

#include <numeric>
#include "log_control.h"
#include "common/aicpu_hccl_common.h"
#include "aicpu_kfc_batchwrite_process.h"
#include "aicpu_kfc_retry_process.h"
#include "framework/aicpu_communicator.h"
#include "algorithm/task_orchestrator.h"
#include "common/aicpu_sqe_context.h"
#include "dfx/mc2_trace_utils.h"
#include "utils/hccl_aicpu_utils.h"
#include "common/aicpu_kfc_utils.h"
#include "utils/aicpu_hdc_utils.h"
#include "framework/aicpu_hccl_process.h"
#include "framework/aicpu_kfc_rpc_serverv2.h"
#include "framework/aicpu_kfc_prof.h"
#include "common/aicpu_kfc_tiling_utils.h"
#include "coll_batch_write_executor.h"
#include "dfx/aicpu_profiling_manager.h"
#include <shared_mutex>

using namespace hccl;
using namespace HcclApi;

ANONYMOUS_NAMESPACE_BEGIN
static constexpr uint64_t KERNEL_TIMEOUT = 16 * 60;
static constexpr uint64_t LOGCOUNT_PRINT_TIMEOUT = 10000;
struct TimeOutCheckInfo {
    u64 kernelStartTime;
    std::unordered_map<u32, bool> msgFlag;
    std::unordered_map<u32, u64> msgStartTime;
    std::unordered_map<u32, u32> invalidMsgCount;
};
thread_local TimeOutCheckInfo g_timeOutInfoInst{};
void SetMsgEnableFlag(u32 groupIdx, bool flag) { g_timeOutInfoInst.msgFlag[groupIdx] = flag; }

bool CheckMsgEnableFlag(u32 groupIdx)
{
    if (g_timeOutInfoInst.msgFlag.find(groupIdx) == g_timeOutInfoInst.msgFlag.end()) {
        return false;
    }
    return g_timeOutInfoInst.msgFlag[groupIdx];
}

void SetMsgStartTime(u32 groupIdx) { g_timeOutInfoInst.msgStartTime[groupIdx] = GetCurCpuTimestamp(); }

u64 GetMsgStartTime(u32 groupIdx)
{
    if (g_timeOutInfoInst.msgStartTime.find(groupIdx) == g_timeOutInfoInst.msgStartTime.end()) {
        return 0UL;
    }
    return g_timeOutInfoInst.msgStartTime[groupIdx];
}

void SetKernelStartTime(void) { g_timeOutInfoInst.kernelStartTime = GetCurCpuTimestamp(); }

void AddMsgInValidCount(u32 groupIdx) { g_timeOutInfoInst.invalidMsgCount[groupIdx]++; }

void ClearMsgInValidCount(u32 groupIdx) { g_timeOutInfoInst.invalidMsgCount[groupIdx] = 0; }

uint32_t GetMsgInValidCount(u32 groupIdx)
{
    if (g_timeOutInfoInst.invalidMsgCount.find(groupIdx) == g_timeOutInfoInst.invalidMsgCount.end()) {
        return 0U;
    }
    return g_timeOutInfoInst.invalidMsgCount[groupIdx];
}

struct CommInstMgr {
    HcclOpResParam* resParam;
    hccl::HcclCommAicpu* hcclCommAicpu;
    AicpuKfcRpcServerV2 rpcServer;
};

struct KfcGroupIndexInfo {
    std::shared_mutex mutex;
    u32 nextId{0U};
    std::unordered_map<std::string, int32_t> groupNameToId{};
    std::unordered_map<int32_t, CommInstMgr> instMap{};
} g_commIdMap;

int32_t InsertComIdMap(const std::string& group)
{
    std::unique_lock<std::shared_mutex> rwlock(g_commIdMap.mutex);
    if (g_commIdMap.groupNameToId.find(group) == g_commIdMap.groupNameToId.end()) {
        HCCL_INFO("Insert group %s at index %u.", group.c_str(), g_commIdMap.nextId);
        g_commIdMap.groupNameToId[group] = g_commIdMap.nextId++;
    } else {
        HCCL_INFO("Group %s is already at index %u.", group.c_str(), g_commIdMap.groupNameToId[group]);
    }
    return g_commIdMap.groupNameToId[group];
}

int32_t GetComGroupIdx(const std::string& group)
{
    std::shared_lock<std::shared_mutex> rwlock(g_commIdMap.mutex);
    int32_t idx;
    if (g_commIdMap.groupNameToId.find(group) == g_commIdMap.groupNameToId.end()) {
        HCCL_ERROR("Failed to find group %s in index map.", group.c_str());
        idx = -1;
    } else {
        idx = g_commIdMap.groupNameToId[group];
    }
    return idx;
}

HcclResult InsertCommInst(uint32_t idx, hccl::HcclCommAicpu* comm, HcclOpResParam* resParam)
{
    g_commIdMap.instMap[idx].resParam = resParam;
    g_commIdMap.instMap[idx].hcclCommAicpu = comm;
    return HCCL_SUCCESS;
}

hccl::HcclCommAicpu* GetCommAicpuCommInst(uint32_t idx)
{
    if (g_commIdMap.instMap.find(idx) == g_commIdMap.instMap.end()) {
        return nullptr;
    }
    return g_commIdMap.instMap[idx].hcclCommAicpu;
}

HcclOpResParam* GetCommAicpuResInst(uint32_t idx)
{
    if (g_commIdMap.instMap.find(idx) == g_commIdMap.instMap.end()) {
        return nullptr;
    }
    return g_commIdMap.instMap[idx].resParam;
}

AicpuKfcRpcServerV2* GetCommRpcServer(uint32_t idx)
{
    if (g_commIdMap.instMap.find(idx) == g_commIdMap.instMap.end()) {
        return nullptr;
    }
    return &(g_commIdMap.instMap[idx].rpcServer);
}

static thread_local uint8_t g_expectPrepareId[MAX_QUE_NUM];
void SetExpectPrepareId(uint8_t queueId, uint8_t msgId) { g_expectPrepareId[queueId] = msgId; }

uint8_t GetExpectPrepareId(uint8_t queueId) { return g_expectPrepareId[queueId]; }

struct CommInfoCtx {
    AlgType algType;
    std::string algName;
    std::string tag;
};
static std::unordered_map<std::string, std::unordered_map<u8, CommInfoCtx>> g_commTypeInfoMap;
static std::shared_mutex g_mutexForTypeInfoMap;
void SetCommInfoCtx(const std::string& groupName, u8 commType, const CommInfoCtx& ctx)
{
    std::unique_lock<std::shared_mutex> rwlock(g_mutexForTypeInfoMap);
    g_commTypeInfoMap[groupName][commType] = ctx;
}

HcclResult GetCommInfoCtx(const std::string& commName, u8 commType, CommInfoCtx& ctx)
{
    std::shared_lock<std::shared_mutex> rwlock(g_mutexForTypeInfoMap);
    const auto groupIter = g_commTypeInfoMap.find(commName);
    if (groupIter == g_commTypeInfoMap.end()) {
        HCCL_ERROR("Failed to find group %s in type info map.", commName.c_str());
        return HCCL_E_INTERNAL;
    }

    const auto commIter = groupIter->second.find(commType);
    if (commIter == groupIter->second.end()) {
        HCCL_ERROR("Failed to find type %u in map for group %s.", static_cast<u32>(commType), commName.c_str());
        return HCCL_E_INTERNAL;
    }

    ctx = commIter->second;
    return HCCL_SUCCESS;
}

const std::unordered_map<std::string, std::string> g_algName
    = {{"AllGather=level0:ring", "AllGatherRingFor91093Executor"},
       {"AllGather=level0:fullmesh", "AllGatherMeshOpbaseExecutor"},
       {"AllGather=level0:doublering", "AlignedAllGatherDoubleRingFor91093Executor"},
       {"ReduceScatter=level0:ring", "ReduceScatterRingFor91093Executor"},
       {"ReduceScatter=level0:fullmesh", "ReduceScatterMeshDmaEliminationExecutor"},
       {"ReduceScatter=level0:doublering", "AlignedReduceScatterDoubleRingFor91093Executor"},
       {"AllReduce=level0:ring", "AllReduceRingFor91093Executor"},
       {"AllReduce=level0:fullmesh", "AllReduceMeshOpbaseLoopExecutor"},
       {"AllReduce=level0:doublering", "AlignedAllReduceDoubleRingFor91093Executor"},
       {"AlltoAll=level0:pairwise", "RunAlltoAllVStaged"},
       {"AlltoAll=level0:fullmesh", "RunAlltoAllDirectFullmesh"},
       {"BatchWrite=level0:fullmesh", BATCH_WRITE_ALG_NAME}};
ANONYMOUS_NAMESPACE_END

AicpuAddOneNotifyWaitSqe g_addOneNotifyWaitSqe = nullptr;
AicpuAddOneRecordSqe g_addOneRecordSqe = nullptr;
AicpuAddOneWriteValueRecordSqe g_addOneWriteValueRecordSqe = nullptr;
AicpuAddOneMemcpySqe g_addOneMemcpySqe = nullptr;
AicpuAddOneEventResetSqe g_addOneEventResetSqe = nullptr;
AicpuAddOneEventRecordSqe g_addOneEventRecordSqe = nullptr;
AicpuAddOneEventWaitSqe g_addOneEventWaitSqe = nullptr;
AicpuAddOneRdmaDbSendSqe g_addOneRdmaDbSendSqe = nullptr;
AicpuAddOneFlipPlaceHolderSqe g_addOneFlipPlaceHolderSqe = nullptr;
AicpuAddOneNotifyWaitSqe AicpuGetAddOneNotifyWaitSqe() { return g_addOneNotifyWaitSqe; }
AicpuAddOneRecordSqe AicpuGetAddOneRecordSqe() { return g_addOneRecordSqe; }
AicpuAddOneWriteValueRecordSqe AicpuGetAddOneWriteValueRecordSqe() { return g_addOneWriteValueRecordSqe; }
AicpuAddOneMemcpySqe AicpuGetAddOneMemcpySqe() { return g_addOneMemcpySqe; }
AicpuAddOneEventResetSqe AicpuGetAddOneEventResetSqe() { return g_addOneEventResetSqe; }
AicpuAddOneEventRecordSqe AicpuGetAddOneEventRecordSqe() { return g_addOneEventRecordSqe; }
AicpuAddOneEventWaitSqe AicpuGetAddOneEventWaitSqe() { return g_addOneEventWaitSqe; }
AicpuAddOneRdmaDbSendSqe AicpuGetAddOneRdmaDbSendSqe() { return g_addOneRdmaDbSendSqe; }
AicpuAddOneFlipPlaceHolderSqe AicpuGetAddOneFlipPlaceHolderSqe() { return g_addOneFlipPlaceHolderSqe; }

ANONYMOUS_NAMESPACE_BEGIN
void InitSqCqFun(AicpuComContext* ctx)
{
    if (ctx->devType == DevType::DEV_TYPE_310P1 || ctx->devType == DevType::DEV_TYPE_310P3) {
        g_addOneNotifyWaitSqe = AddOneNotifyWaitSqeV2;
        g_addOneRecordSqe = AddOneRecordSqeV2;
        g_addOneWriteValueRecordSqe = AddOneWriteValueRecordSqeV2;
        g_addOneMemcpySqe = AddOneMemcpySqeV2;
        g_addOneEventResetSqe = AddOneEventResetSqeV2;
        g_addOneEventRecordSqe = AddOneEventRecordSqeV2;
        g_addOneEventWaitSqe = AddOneEventWaitSqeV2;
    } else {
        g_addOneNotifyWaitSqe = AddOneNotifyWaitSqeV1;
        g_addOneRecordSqe = AddOneRecordSqeV1;
        g_addOneWriteValueRecordSqe = AddOneWriteValueRecordSqeV1;
        g_addOneMemcpySqe = AddOneMemcpySqeV1;
        g_addOneEventResetSqe = AddOneEventResetSqeV1;
        g_addOneEventRecordSqe = AddOneEventRecordSqeV1;
        g_addOneEventWaitSqe = AddOneEventWaitSqeV1;
        g_addOneFlipPlaceHolderSqe = AddOneFlipPlaceHolderSqeV1;
        g_addOneRdmaDbSendSqe = AddOneRdmaDbSendSqeV1;
    }
}

HcclResult InitIbversData(HccCommResParamTask* commParam, AicpuComContext* ctx)
{
    HCCL_INFO("commParam->ibverbsData:%llu", commParam->ibverbsData);
    if (commParam->ibverbsDataSize != static_cast<u64>(ctx->rankNum) * sizeof(TransportDeviceNormalData)) {
        HCCL_ERROR(
            "ibverbsData size[%llu] is not valid, expect size[%llu]", commParam->ibverbsDataSize,
            static_cast<u64>(ctx->rankNum) * sizeof(TransportDeviceNormalData));
        return HCCL_E_PARA;
    }
    ctx->ibversData.resize(ctx->rankNum);
    for (u32 i = 0; i < ctx->rankNum; i++) {
        void* memPtr = reinterpret_cast<void*>(commParam->ibverbsData + sizeof(TransportDeviceNormalData) * i);
        ctx->ibversData[i] = *(static_cast<TransportDeviceNormalData*>(memPtr));
        ctx->ibversData[i].Print();
    }
    return HCCL_SUCCESS;
}

void InitRankInfo(HccCommResParamTask* commParam, AicpuComContext* ctx)
{
    for (u32 i = 0; i < ctx->rankNum; i++) {
        ctx->rankInfo[i].rankId = i;
        ctx->rankInfo[i].window = commParam->windowsIn[i];
        ctx->rankInfo[i].windowOut = commParam->windowsOut[i];
    }
}

template <typename T>
HcclResult InitAndVerifySignal(const HcclSignalInfo& signalInfo, std::shared_ptr<T>& notify, u64& addr)
{
    if (signalInfo.resId == INVALID_U64) {
        HCCL_INFO("[HcclCommAicpu][%s] resId is invalid, need not check", __func__);
        return HCCL_SUCCESS;
    }

    EXCEPTION_CATCH((notify = std::make_shared<T>()), return HCCL_E_PTR);
    CHK_SMART_PTR_NULL(notify);
    CHK_RET(notify->Init(signalInfo, NotifyLoadType::DEVICE_NOTIFY));
    HcclSignalInfo notifyInfo;
    CHK_RET(notify->GetNotifyData(notifyInfo));
    addr = notifyInfo.addr;
    HCCL_INFO(
        "[HcclCommAicpu][%s] success, resId[%u], tsId:%d, devId[%u]", __func__, signalInfo.resId, signalInfo.tsId,
        signalInfo.devId);
    return HCCL_SUCCESS;
}

HcclResult InitSignalInfo(HccCommResParamTask* commParam, AicpuComContext* ctx)
{
    for (u32 i = 0; i < ctx->rankNum; i++) {
        // 跨片notify只用在其它rank上，本片位置未填写有效值
        if (ctx->rankId == i) {
            continue;
        }

        // no ipc pre sync
        u64 address = 0;
        std::shared_ptr<LocalNotify> localNotify;
        HcclSignalInfo* sigInfo = &commParam->signalInfo.noIpcNotifys[i];
        CHK_RET(InitAndVerifySignal(*sigInfo, localNotify, address));
        ctx->noIpcPreNotify[i].actualNotifyId = static_cast<s32>(sigInfo->resId);

        if (sigInfo->rankId != ctx->rankInfo[i].rankId) {
            HCCL_DEBUG(
                "rankId mismatch. current process rank:%d, sigInfo rank:%d", ctx->rankInfo[i].rankId, sigInfo->rankId);
            return HCCL_E_INTERNAL;
        }

        // no ipc post sync
        sigInfo = &commParam->signalInfo.noIpcNotifys[ctx->rankNum + i];
        CHK_RET(InitAndVerifySignal(*sigInfo, localNotify, address));
        ctx->noIpcPostNotify[i].actualNotifyId = static_cast<s32>(sigInfo->resId);

        // ipc pre record
        sigInfo = &commParam->signalInfo.ipcNotifys[i];
        std::shared_ptr<RemoteNotify> remoteNotify;
        CHK_RET(InitAndVerifySignal(*sigInfo, remoteNotify, ctx->ipcPreRecordNotify[i].address));
        ctx->ipcPreRecordNotify[i].actualNotifyId = static_cast<s32>(sigInfo->resId);

        // ipc pre wait
        sigInfo = &commParam->signalInfo.ipcNotifys[ctx->rankNum + i];
        CHK_RET(InitAndVerifySignal(*sigInfo, localNotify, ctx->ipcPreWaitNotify[i].address));
        ctx->ipcPreWaitNotify[i].actualNotifyId = static_cast<s32>(sigInfo->resId);

        // ipc post record
        sigInfo = &commParam->signalInfo.ipcNotifys[2 * ctx->rankNum + i]; // 2 is ipc post record(8-15)
        CHK_RET(InitAndVerifySignal(*sigInfo, remoteNotify, ctx->ipcPostRecordNotify[i].address));
        ctx->ipcPostRecordNotify[i].actualNotifyId = static_cast<s32>(sigInfo->resId);

        // ipc post wait
        sigInfo = &commParam->signalInfo.ipcNotifys[3 * ctx->rankNum + i]; // 3 is ipc post wait(16-23)
        CHK_RET(InitAndVerifySignal(*sigInfo, localNotify, ctx->ipcPostWaitNotify[i].address));
        ctx->ipcPostWaitNotify[i].actualNotifyId = static_cast<s32>(sigInfo->resId);
    }
    return HCCL_SUCCESS;
}

HcclResult InitEventId(HccCommResParamTask* commParam, AicpuComContext* ctx)
{
    for (u32 i = 0; i < ctx->rankNum; i++) {
        // eventid只用在片内，放全局
        HcclSignalInfo* sigInfo = &commParam->signalInfo.noIpcEvents[i];
        if (sigInfo->rankId == ctx->rankId) {
            // 盘古230B入图场景连续跑第二次会出现eventId校验失败，当前不使用event，删除KfcResIsInvalid校验
            ctx->eventIds[i] = sigInfo->resId;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult InitAicpuOpNotify(HccCommResParamTask* commParam, AicpuComContext* ctx)
{
    for (u32 i = 0; i < sizeof(ctx->aicpuOpNotify) / sizeof(ctx->aicpuOpNotify[0]); i++) {
        HcclSignalInfo* sigInfo = &commParam->signalInfo.aicpuOpNotify[i];
        std::shared_ptr<LocalNotify> localNitfy;
        EXCEPTION_CATCH((localNitfy = std::make_shared<LocalNotify>()), return HCCL_E_PTR);
        CHK_RET(localNitfy->Init(*sigInfo, NotifyLoadType::DEVICE_NOTIFY));
        HcclSignalInfo signalInfo;
        CHK_RET(localNitfy->GetNotifyData(signalInfo));
        ctx->aicpuOpNotify[i].actualNotifyId = static_cast<s32>(sigInfo->resId);
        ctx->aicpuOpNotify[i].address = signalInfo.addr;
    }
    return HCCL_SUCCESS;
}

HcclResult InitTimeOutConfig(HccCommResParamTask* commParam, AicpuComContext* ctx)
{
    ctx->dfxExtendInfo.dfxTimeOutConfig.sqeTimeOutTimeOut = commParam->config.notifyWaitTime;
    ctx->dfxExtendInfo.dfxTimeOutConfig.sqeCreditTimeOut = RT_STARS_NEVER_TIMEOUT_KERNEL_CREDIT;
    ctx->dfxExtendInfo.dfxTimeOutConfig.sqeWaitTimeOut = dfx::kKfcTimeOut;
    ctx->dfxExtendInfo.dfxTimeOutConfig.sqFullWaitTimeOut = dfx::kSqFullWaitTimeOut;
    HCCL_INFO(
        "DFX timeout config init successfully with details: [%s]",
        ctx->dfxExtendInfo.dfxTimeOutConfig.ToString().c_str());
    return HCCL_SUCCESS;
}

HcclResult InitChipType(AicpuComContext* ctx)
{
    CHK_RET(hrtHalGetDeviceType(ctx->devId, ctx->devType));
    CHK_RET(hrtHalGetDeviceInfo(ctx->devId, MODULE_TYPE_SYSTEM, INFO_TYPE_PHY_CHIP_ID, &ctx->chipId));
    if (ctx->devType == DevType::DEV_TYPE_910 || ctx->devType == DevType::DEV_TYPE_NOSOC
        || ctx->devType == DevType::DEV_TYPE_COUNT) {
        HCCL_ERROR("Get devtype [%d] is invalid", ctx->devType);
        return HCCL_E_DRV;
    }
    if (ctx->devType == DevType::DEV_TYPE_310P3 || ctx->devType == DevType::DEV_TYPE_310P1) {
        uint32_t ssid;
        const HcclResult ret = hrtDrvMemSmmuQuery(ctx->devId, &ssid);
        HCCL_DEBUG("ssid %u", ssid);
        ctx->ssid = ssid;
        ctx->determinism = false;
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("hrtDrvMemSmmuQuery error"), HCCL_E_DRV);
    }
    InitSqCqFun(ctx);
    return HCCL_SUCCESS;
}

void GetNextMsgFromMsg(AivAicpuOpParam* msg, AivAicpuOpParam* nextMsg, u64 dataLen, u32 rankNum)
{
    *(nextMsg) = *(msg);
    // nextMsg的偏移同UpdateMsg
    if (nextMsg->commType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER) {
        nextMsg->sendBuffer = nextMsg->sendBuffer + dataLen / rankNum;
        nextMsg->recvBuffer = nextMsg->recvBuffer + dataLen / rankNum;
    } else {
        nextMsg->sendBuffer = nextMsg->sendBuffer + dataLen;
        nextMsg->recvBuffer = nextMsg->recvBuffer + dataLen;
    }
    nextMsg->PrintMsg("nextMsg");
}

void GetCommonHcclMsg(HcclMsg* hcclMsg, CommonHcclMsg* commonHcclMsg, u64 tilingBase)
{
    const HcclTilingVersion ver = hcclMsg->addMsg.v0Msg.version;
    if (ver != HcclTilingVersion::DEPRECATED_TILING_VERSION) {
        const size_t copyOffset = offsetof(HcclMsg, addMsg);
        (void)memcpy_s(commonHcclMsg, copyOffset, hcclMsg, copyOffset);
        if (ver == HcclTilingVersion::ONLINE_COMPILATION_TILING_VERSION) {
            commonHcclMsg->ccOpTilingData = hcclMsg->addMsg.v1Msg.ccOpTilingData + tilingBase;
        } else {
            commonHcclMsg->ccOpTilingData = hcclMsg->addMsg.v1Msg.ccOpTilingData;
        }
        commonHcclMsg->valid = hcclMsg->addMsg.v1Msg.valid;
        commonHcclMsg->hcclDataType = static_cast<HcclDataType>(hcclMsg->addMsg.v1Msg.hcclDataType);
        commonHcclMsg->repeatCnt = hcclMsg->addMsg.v1Msg.repeatCnt;
        commonHcclMsg->selfHandleID = hcclMsg->addMsg.v1Msg.selfHandleID;
        commonHcclMsg->seqNum = hcclMsg->addMsg.v1Msg.seqNum;
        commonHcclMsg->version = hcclMsg->addMsg.v1Msg.version;
        commonHcclMsg->xorCheck = hcclMsg->addMsg.v1Msg.xorCheck;
    } else {
        (void)memcpy_s(commonHcclMsg, sizeof(HcclMsg), hcclMsg, sizeof(HcclMsg));
        commonHcclMsg->ccOpTilingData = 0UL;
    }
}

AicpuCCExecOp GetCcOpType(u64 comDataLen, u64 rankNum)
{
    AicpuCCExecOp ccType;
    AicpuComContext* ctx = AicpuGetComContext();
    if (ctx->devType == DevType::DEV_TYPE_310P1 || ctx->devType == DevType::DEV_TYPE_310P3) {
        if (ctx->onlyRead > 0) {
            HCCL_DEBUG("Only read mode enabled");
            ccType = CC_EXE_ONE_SHOT_SINGLE_RING;
        } else if (rankNum == 2) { // 2 卡
            if (comDataLen < HCCL_SMALL_COUNT_1_M) {
                ccType = CC_EXE_ONE_SHOT_1_STREAM;
            } else {
                ccType = CC_EXE_TWO_SHOT_1_STREAM;
            }
        } else { // 2 卡以上
            if (comDataLen < HCCL_SMALL_COUNT_256K && (rankNum & (rankNum - 1)) == 0) {
                ccType = CC_EXE_ONE_SHOT_HD;
            } else {
                ccType = CC_EXE_ONE_SHOT_SINGLE_RING;
            }
        }
    } else {
        if ((comDataLen < AC_DEFAULT_ONE_SHOT_SIZE) && ((rankNum % AC_DEFAULT_RANK_GROUP) == 0)) {
            ccType = CC_EXE_ONE_SHOT_8_STREAM;
        } else {
            ccType = CC_EXE_TWO_SHOT_8_STREAM;
        }
    }
    return ccType;
}

void UpdateMsg(AivAicpuOpParam* msg, u64 dataLen, u32 rankNum)
{
    // 如果是reduceScatter算法，sendBuffer和recvBuffer的偏移为recvCnt，即sendCnt/rankNum
    // allgather和allreduce算法，sendBuffer和recvBuffer的偏移为recvCnt=sendCnt
    // all2all算法，sendBuffer和recvBuffer的偏移为 sendCnt / rankNum
    // 如果recvBuffer是非连续存储的，则recvBuffer的偏移将变更为 sendCnt
    if (msg->commType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER) {
        msg->sendBuffer = msg->sendBuffer + dataLen / rankNum;
        msg->recvBuffer = msg->recvBuffer + dataLen / rankNum;
    } else {
        msg->sendBuffer = msg->sendBuffer + dataLen;
        msg->recvBuffer = msg->recvBuffer + dataLen;
    }
    if (msg->commType == HcclCMDType::HCCL_CMD_ALLREDUCE || msg->commType == HcclCMDType::HCCL_CMD_ALLTOALL) {
        msg->winOffset = msg->winOffset + dataLen;
    }
    msg->PrintMsg("update msg");
}

HcclResult SetMsgWinOffset(AicpuComContext* ctx, AivAicpuOpParam* msg)
{
    if (msg->useBufferType == MC2_BUFFER_TYPE_WINDOW_IN
        && ((msg->commType == HcclCMDType::HCCL_CMD_ALLREDUCE && !ctx->determinism)
            || msg->commType == HcclCMDType::HCCL_CMD_ALLTOALL)) {
        // sendBuffer 减去本卡的winIn
        AicpuComRankInfo* selfRankInfo = &ctx->rankInfo[ctx->rankId];
        if (msg->sendBuffer < selfRankInfo->window) {
            HCCL_ERROR("sendBuffer addr[%p] must bigger than window addr[%p].", msg->sendBuffer, selfRankInfo->window);
            return HCCL_E_PARA;
        }
        msg->winOffset = msg->sendBuffer - selfRankInfo->window;
    }
    HCCL_INFO("Offsetting winOffset %lu", msg->winOffset);
    return HCCL_SUCCESS;
}

bool CheckNsCommand(hccl::HcclCommAicpu* comm)
{
    KfcCommand cmd;
    if (comm->BackGroundGetCmd(cmd) != HCCL_SUCCESS || cmd != KfcCommand::NsStopLaunch) {
        return false;
    }
    comm->SetNsStopLaunchStatus(true);
    HCCL_WARNING("N second stop launch for receiving stop launch cmd.");
    return true;
}

HcclResult CheckNsStopLaunchStatus(const std::vector<u32>& groupIds)
{
    for (const auto i : groupIds) {
        hccl::HcclCommAicpu* comm = GetCommAicpuCommInst(i);
        if (comm != nullptr && comm->GetNsStopLaunchStatus()) {
            return HCCL_E_SUSPENDING;
        }
    }
    return HCCL_SUCCESS;
}

bool GetOpRetryEnable(const std::vector<u32>& groupIds)
{
    for (const auto i : groupIds) {
        hccl::HcclCommAicpu* comm = GetCommAicpuCommInst(i);
        if (comm == nullptr || !comm->GetOpRetryEnable()) {
            return false;
        }
    }
    return true;
}

HcclResult CheckRestartError(hccl::HcclCommAicpu* comm)
{
    // 支持重执行时，检测是否有可重执行的sdma异常, 或者kStopLaunch命令
    if (comm->GetOpRetryEnable()) {
        if (comm->IsTaskExceptionForHccs()) {
            HCCL_WARNING("MC2 restart Sdma error happened.");
            return HCCL_E_SUSPENDING;
        }

        KfcCommand cmd = KfcCommand::kNone;
        CHK_RET(comm->BackGroundGetCmd(cmd));
        if (cmd == KfcCommand::kStopLaunch) {
            HCCL_WARNING("MC2 restart receive kfc command stop launch.");
            return HCCL_E_SUSPENDING;
        }
    }
    return HCCL_SUCCESS;
}

static constexpr u32 LOG_INTERVAL = 10000U;
HcclResult CheckFinishByStream(HcclCommAicpu& comm, size_t streamIdx, bool tailQueryFlag = true)
{
    uint32_t sqHead, sqTail;
    Stream& stream = (streamIdx == SIZE_MAX ? comm.GetMainStream() : comm.GetSlaveStream()[streamIdx]);
    const uint32_t sqId = stream.sqId();
    if (tailQueryFlag) {
        CHK_RET(QuerySqStatusByType(comm.GetDevId(), sqId, DRV_SQCQ_PROP_SQ_TAIL, sqTail));
    } else {
        sqTail = stream.GetSqeContextPtr()->buffer.sqTail;
    }
    CHK_RET(QuerySqStatusByType(comm.GetDevId(), sqId, DRV_SQCQ_PROP_SQ_HEAD, sqHead));
    if (sqTail == sqHead) {
        HCCL_DEBUG("Stream %u finished, sq id %u, head&tail %u.", stream.id(), stream.sqId(), sqHead);
        return HCCL_SUCCESS;
    }

    static uint32_t logHead = UINT32_MAX;
    static uint32_t logTail = UINT32_MAX;
    static uint32_t loopCnt;
    if (++loopCnt % LOG_INTERVAL == 0U) {
        if (logHead != sqHead || logTail != sqTail) {
            logHead = sqHead;
            logTail = sqTail;
            HCCL_RUN_INFO(
                "Current state. devId:%u sqid:%d, head:%u, tail:%u, group[%s]", comm.GetDevId(), sqId, sqHead, sqTail,
                comm.GetGroupName().c_str());
        }
    }
    return HCCL_E_UNAVAIL;
}

HcclResult RpcServerPreCheck(AicpuKfcRpcServerV2* rpc, hccl::HcclCommAicpu* comm, bool& finalizeFlag)
{
    if (CheckNsCommand(comm)) {
        return HCCL_E_SUSPENDING;
    }
    if (CheckRestartError(comm) == HCCL_E_SUSPENDING) {
        return HCCL_E_SUSPENDING;
    }
    if (comm->GetDfxExtendInfo()->pollStatus == PollStatus::kStopAsException) {
        if (comm->GetOpRetryEnable() && comm->IsTaskExceptionForHccs()) {
            HCCL_WARNING("MC2 restart Sdma error happened.");
            return HCCL_E_SUSPENDING;
        }
        HCCL_ERROR("MC2 hccl aicpu exec failed, for task exception.");
        return HCCL_E_INTERNAL;
    }
    if (rpc->GetIsFinalize()) {
        if (CheckFinishByStream(*comm, SIZE_MAX) == HCCL_SUCCESS) {
            finalizeFlag = true;
            rpc->WriteFinishWhenAllFinalize();
        }
        return HCCL_E_AGAIN;
    }
    return HCCL_SUCCESS;
}

static constexpr u64 BARRIER_TIMEOUT = static_cast<u64>(NSEC_PER_SEC) * 60UL;
HcclResult BarrierProcess(u32 groupIdx, u32 localGroupIdx, u32 queueId, BarrierStatus& status)
{
    AicpuKfcRpcServerV2* rpc = GetCommRpcServer(groupIdx);
    BarrierInfo* barrierInfos = rpc->GetBarrierInfoByGroupIdx(localGroupIdx);
    BarrierStatus& selfFlag = barrierInfos[queueId].status;
    if (selfFlag == BarrierStatus::NO_BARRIER) {
        barrierInfos[queueId].lastTimeStamp = GetCurCpuTimestamp();
        status = BarrierStatus::NO_BARRIER;
        return HCCL_SUCCESS;
    }

    u32& barrierFinishCnt = rpc->GetBarrierFinishCnts()[HcclAicpuUtils::GetBlockIdx()];
    if (selfFlag == BarrierStatus::SELF_BARRIER) {
        if (CheckFinishByStream(*GetCommAicpuCommInst(groupIdx), queueId, false) == HCCL_SUCCESS) {
            barrierInfos[queueId].lastTimeStamp = GetCurCpuTimestamp();
            selfFlag = BarrierStatus::INTER_BARRIER;
            ++barrierFinishCnt;
            HCCL_INFO(
                "[%s][Queue %u]All tasks in queue are finished in block %u, finish count %u.", __func__, queueId,
                HcclAicpuUtils::GetBlockIdx(), barrierFinishCnt);
        }
    }

    if (selfFlag == BarrierStatus::INTER_BARRIER) {
        u32 start = 0U;
        u32 end = 0U;
        rpc->GetLocalQueueRange(start, end);
        if (barrierFinishCnt == end + 1U - start) {
            CHK_PRT_RET(
                AicpuKfcUtils::ThreadBarrier(BARRIER_TIMEOUT) != HCCL_SUCCESS,
                HCCL_ERROR(
                    "[%s]Failed to wait in block %u, finish count %u.", __func__, HcclAicpuUtils::GetBlockIdx(),
                    barrierFinishCnt),
                HCCL_E_AGAIN);
            rpc->ClearBarrierStatus(localGroupIdx, start, barrierFinishCnt);
            barrierFinishCnt = 0U;
            return HCCL_SUCCESS;
        }
    }

    status = selfFlag;
    const u64 ts = GetCurCpuTimestamp();
    CHK_PRT_RET(
        ts - barrierInfos[queueId].lastTimeStamp > BARRIER_TIMEOUT,
        HCCL_ERROR("[%s]Timeout when checking queue %u, finish count %u.", __func__, queueId, barrierFinishCnt),
        HCCL_E_AGAIN);

    return HCCL_SUCCESS;
}

void FinalizeProcess(u32 queueIdx, hccl::HcclCommAicpu& commAicpu, AicpuKfcRpcServerV2& rpcServer)
{
    if (AicpuKfcProf::IsDebugModeEquals(MC2_DEBUG_PRINT_BUFF)) {
        rpcServer.PrintAllHcclMsgAreaData();
    }
    rpcServer.SetIsFinalize(queueIdx, true);
    if (rpcServer.GetTotalQueueNum() == 0U) {
        rpcServer.ResetCommitTaskAdd(commAicpu.GetDispatcher(), &(commAicpu.GetMainStream()));
        LaunchTask(commAicpu.GetDispatcher(), commAicpu.GetMainStream());
    }
    SetExpectPrepareId(queueIdx, 0U);
}

HcclResult AddTaskForGroupSyncMsg(const std::vector<u32>& groupIds, u32 localGroupIdx, CommonHcclMsg* hcclMsg)
{
    if (static_cast<uint32_t>(hcclMsg->commDepGroupID) == localGroupIdx) {
        HCCL_ERROR(
            "InterHcclGroupSync must be used for cross-domain synchronization, group id %d", hcclMsg->commDepGroupID);
        return HCCL_E_INTERNAL;
    }

    CHK_PRT_RET(
        static_cast<size_t>(hcclMsg->commDepGroupID) >= groupIds.size(),
        HCCL_ERROR("Invalid group id %d.", hcclMsg->commDepGroupID), HCCL_E_INTERNAL);

    AicpuKfcRpcServerV2* rpcServerDep = GetCommRpcServer(groupIds[hcclMsg->commDepGroupID]);
    if (rpcServerDep == nullptr) {
        HCCL_ERROR("get rpc server failed, group id %d", hcclMsg->commDepGroupID);
        return HCCL_E_INTERNAL;
    }
    uint64_t waitAddr = rpcServerDep->GetFinishAddrByHandleId(hcclMsg->commDepHandleID);
    if (waitAddr == 0) {
        HCCL_INFO("%s waitAddr is not ready, group id %d", __func__, hcclMsg->commDepGroupID);
        return HCCL_E_UNAVAIL;
    }
    int32_t turnNum = rpcServerDep->GetMsgRepeatCnt(hcclMsg->commDepHandleID);
    if (turnNum < 0) {
        HCCL_INFO("%s comm group %d idx %d is not ready", __func__, hcclMsg->commDepGroupID, hcclMsg->commDepHandleID);
        return HCCL_E_UNAVAIL;
    }

    const u32 groupIdx = groupIds[localGroupIdx];
    hccl::HcclCommAicpu* commAicpu = GetCommAicpuCommInst(groupIdx);
    AicpuKfcRpcServerV2* rpcServer = GetCommRpcServer(groupIdx);
    CHK_PRT_RET(
        commAicpu == nullptr || rpcServer == nullptr, HCCL_ERROR("Invalid group index %u.", groupIdx), HCCL_E_INTERNAL);
    rpcServer->SetNeedRetryFlag(false);
    CHK_RET(rpcServer->AddCcoreWait(
        commAicpu->GetDispatcher(), waitAddr, static_cast<uint32_t>(turnNum), &(commAicpu->GetMainStream()), false));
    return HCCL_SUCCESS;
}

void PrepareOpParam(
    hccl::OpParam* opParam, CommonHcclMsg* hcclMsg, AicpuKfcRpcServerV2& rpc, hccl::HcclCommAicpu* commAicpu)
{
    if (AicpuKfcProf::IsDebugModeEquals(MC2_DEBUG_SDMA_ERROR)) {
        opParam->inputPtr = reinterpret_cast<void*>(0xdeadbeef);
        opParam->outputPtr = reinterpret_cast<void*>(0xdeadbeef);
    } else {
        opParam->inputPtr = reinterpret_cast<void*>(hcclMsg->sendBuffer);
        opParam->outputPtr = reinterpret_cast<void*>(hcclMsg->recvBuffer);
    }
    opParam->reduceType = hcclMsg->opType;
    opParam->stream = commAicpu->GetMainStream();
    opParam->syncMode = SyncMode::DEFAULT_TIMEWAITSYNCMODE;
    opParam->opBaseAtraceInfo = nullptr;
    opParam->opType = static_cast<HcclCMDType>(hcclMsg->commType);
    if (hcclMsg->commType == HcclCMDType::HCCL_CMD_ALLTOALLV || hcclMsg->commType == HcclCMDType::HCCL_CMD_ALLTOALL) {
        HcclMsgExt* hcclMsgExt = rpc.GetHcclMsgExtPtr();
        opParam->All2AllDataDes.sendType = opParam->All2AllDataDes.recvType = hcclMsg->hcclDataType;
        opParam->All2AllDataDes.sendCount = hcclMsg->dataCnt;
        if (hcclMsg->commType == HcclCMDType::HCCL_CMD_ALLTOALL && hcclMsg->strideCount > 0UL) {
            for (uint32_t i = 0U; i < commAicpu->GetRankSize(); ++i) {
                hcclMsgExt->sendCounts[i] = hcclMsgExt->recvCounts[i] = hcclMsg->dataCnt;
                hcclMsgExt->sendOffset[i] = hcclMsgExt->recvOffset[i] = hcclMsg->strideCount * i;
            }
            opParam->opType = static_cast<HcclCMDType>(HcclCMDType::HCCL_CMD_ALLTOALLV);
        }
        if (opParam->opType == static_cast<HcclCMDType>(HcclCMDType::HCCL_CMD_ALLTOALLV)) {
            opParam->All2AllDataDes.sendCounts = static_cast<void*>(hcclMsgExt->sendCounts);
            opParam->All2AllDataDes.recvCounts = static_cast<void*>(hcclMsgExt->recvCounts);
            opParam->All2AllDataDes.sdispls = static_cast<void*>(hcclMsgExt->sendOffset);
            opParam->All2AllDataDes.rdispls = static_cast<void*>(hcclMsgExt->recvOffset);
        }
    } else if (hcclMsg->commType == HcclCMDType::HCCL_CMD_BATCH_WRITE) {
        opParam->BatchWriteDataDes.itemNum = hcclMsg->dataCnt;
        opParam->BatchWriteDataDes.queueNum = rpc.GetTotalQueueNum();
        opParam->BatchWriteDataDes.queueIdx = static_cast<u32>(hcclMsg->opType);
        HCCL_DEBUG(
            "[Sdma-BatchWrite]Queue size %u, global queue id %u, item number %u.", opParam->BatchWriteDataDes.queueNum,
            opParam->BatchWriteDataDes.queueIdx, opParam->BatchWriteDataDes.itemNum);
    } else {
        const u64 totalSize = hcclMsg->dataCnt * DataUnitSize(hcclMsg->hcclDataType);
        opParam->DataDes.count = hcclMsg->dataCnt;
        opParam->DataDes.dataType = hcclMsg->hcclDataType;
        opParam->DataDes.strideCount = hcclMsg->strideCount;
        opParam->inputSize = totalSize;
        opParam->outputSize = totalSize;
    }
}

bool SelectAlgName(const std::string& algConfig, u32 topoType, std::string& algName)
{
    std::string curConfig;
    std::size_t found = algConfig.find(";");
    if (found == 0) {
        return false;
    } else if (found == std::string::npos) {
        curConfig = algConfig;
    } else {
        curConfig = algConfig.substr(0, found);
    }
    if (static_cast<TopoType>(topoType) == TopoType::TOPO_TYPE_NP_SINGLE_RING) {
        if (curConfig == "AllGather=level0:doublering" || curConfig == "ReduceScatter=level0:doublering"
            || curConfig == "AllReduce=level0:doublering") {
            std::size_t pos = curConfig.find(":");
            std::string algConfigTmp = curConfig.substr(0, pos + 1) + "ring";
            algName = g_algName.at(algConfigTmp);
            return true;
        }
    }
    auto res = g_algName.find(curConfig);
    if (res != g_algName.end()) {
        algName = res->second;
        return true;
    }
    HCCL_ERROR(
        "[AicpuHcclProcess][%s] algo_name does not exist, algConfig %s is invalid.", __func__, algConfig.c_str());
    return false;
}

bool SplitHcclAlgoGetLevel1Res(std::string& algoConfig, std::string& algos)
{
    std::string remainAlgoConfig;
    std::size_t found = algoConfig.find(";");
    if ((found == 0) || (found == (algoConfig.length() - 1)) || (found == std::string::npos)) {
        HCCL_INFO("algoConfig %s: there is no level1 algo config", algoConfig.c_str());
        return true;
    }
    remainAlgoConfig = algoConfig.substr(found + 1);
    found = remainAlgoConfig.find(";");
    std::size_t msgPos = 0;
    if (found != std::string::npos) {
        msgPos = found;
        HCCL_WARNING("[AicpuHcclProcess] algo level is more than 1, not supported !");
    } else {
        msgPos = remainAlgoConfig.size();
    }
    algos = (remainAlgoConfig.substr(0, msgPos));
    return false;
}

HcclResult ParserHcclAlgoLevel1(std::string& algoLevel, uint32_t& level, HcclAlgoType& algoType)
{
    std::size_t found = algoLevel.find(":");
    if ((found == 0) || (found == (algoLevel.length() - 1))) {
        HCCL_ERROR("[Parser][HcclAlgoLevel] algo config is invalid.");
        return HCCL_E_PARA;
    }

    std::string orginalLevel = algoLevel.substr(0, found);
    std::string orginalAlgo = algoLevel.substr(found + 1);

    const std::map<std::string, HcclAlgoType> hcclAlgoTypeMap = {
        {"null", HcclAlgoType::HCCL_ALGO_TYPE_NULL},
        {"ring", HcclAlgoType::HCCL_ALGO_TYPE_RING},
        {"pipeline", HcclAlgoType::HCCL_ALGO_TYPE_PIPELINE},
        {"fullmesh", HcclAlgoType::HCCL_ALGO_TYPE_FULLMESH},
        {"H-D_R", HcclAlgoType::HCCL_ALGO_TYPE_HDR},
        {"pairwise", HcclAlgoType::HCCL_ALGO_TYPE_PAIRWISE},
        {"NHR", HcclAlgoType::HCCL_ALGO_TYPE_NHR},
        {"NHR_V1", HcclAlgoType::HCCL_ALGO_TYPE_NHR_V1},
        {"NB", HcclAlgoType::HCCL_ALGO_TYPE_NB},
        {"NA", HcclAlgoType::HCCL_ALGO_TYPE_NA},
    };

    auto iterAlgoType = hcclAlgoTypeMap.find(orginalAlgo);
    if (iterAlgoType == hcclAlgoTypeMap.end()) {
        HCCL_ERROR("[Parser][HcclAlgoLevel] algo config is invalid, algo %s is not supported.", orginalAlgo.c_str());
        return HCCL_E_PARA;
    }
    level = HCCL_ALGO_LEVEL_1;
    algoType = iterAlgoType->second;
    return HCCL_SUCCESS;
}

bool SetAlgTypeLevel1(HcclAlgoType algoConfig, AlgTypeLevel1& algType, uint32_t moduleNum)
{
    switch (algoConfig) {
        case HcclAlgoType::HCCL_ALGO_TYPE_HDR:
            algType = AlgTypeLevel1::ALG_LEVEL1_HD;
            break;
        case HcclAlgoType::HCCL_ALGO_TYPE_RING:
            algType = AlgTypeLevel1::ALG_LEVEL1_RING;
            HCCL_INFO("server num[%u]: level1:ring algo is set.", moduleNum);
            break;
        case HcclAlgoType::HCCL_ALGO_TYPE_NHR:
            algType = AlgTypeLevel1::ALG_LEVEL1_NHR;
            HCCL_INFO("server num[%u]: level1:nhr algo is set.", moduleNum);
            break;
        case HcclAlgoType::HCCL_ALGO_TYPE_NHR_V1:
            algType = AlgTypeLevel1::ALG_LEVEL1_NHR_V1;
            HCCL_INFO("server num[%u]: level1:nhr_v1 algo is set.", moduleNum);
            break;
        case HcclAlgoType::HCCL_ALGO_TYPE_NB:
            algType = AlgTypeLevel1::ALG_LEVEL1_NB;
            HCCL_INFO("server num[%u]: level1:nb algo is set.", moduleNum);
            break;
        case HcclAlgoType::HCCL_ALGO_TYPE_PIPELINE:
            algType = AlgTypeLevel1::ALG_LEVEL1_PIPELINE;
            HCCL_INFO("server num[%u]: level1:pipeline algo is set.", moduleNum);
            break;
        case HcclAlgoType::HCCL_ALGO_TYPE_FULLMESH:
        case HcclAlgoType::HCCL_ALGO_TYPE_PAIRWISE:
            HCCL_WARNING("level1:fullmesh algo is not supported. the config is ignored.");
            [[fallthrough]];
        default:
            HCCL_WARNING("algo is not supported. the config is ignored.");
            return false;
    }
    return true;
}

void SetAlgoLevel1(
    hccl::HcclCommAicpu* commAicpu, HcclAlgoType algoConfig, uint32_t moduleNum, AlgTypeLevel1& algType, bool isDefault)
{
    if ((isDefault == false) && (SetAlgTypeLevel1(algoConfig, algType, moduleNum))) {
        // 不使用default配置
        HCCL_INFO("[AicpuHcclProcess][%s] algType[%u], moduleNum[%u]", __func__, algType, moduleNum);
        return;
    }
    if (moduleNum >= HCCL_INTER_SERVER_RING_ALGO_MAX_SUPPORT_SERVER_NUM) {
        // server 数为 8 以上：使用 HD 算法
        algType = AlgTypeLevel1::ALG_LEVEL1_HD;
    } else {
        // server 数为 2 的非整数次幂：使用 RING 算法
        // server 数为 2 的整数次幂：使用 HD 算法
        algType = (((moduleNum & (moduleNum - 1)) != 0) || (moduleNum == 1)) ? AlgTypeLevel1::ALG_LEVEL1_RING :
                                                                               AlgTypeLevel1::ALG_LEVEL1_HD;
    }
    DevType devType = commAicpu->GetDevType();
    if (algType == AlgTypeLevel1::ALG_LEVEL1_HD && devType == DevType::DEV_TYPE_910_93) {
        algType = AlgTypeLevel1::ALG_LEVEL1_NHR;
    }
    HCCL_INFO("[AicpuHcclProcess][%s] algType[%u], moduleNum[%u]", __func__, algType, moduleNum);
}

void SelectAlgType(hccl::HcclCommAicpu* commAicpu, const std::string& algConfig, uint32_t moduleNum, AlgType& algType)
{
    // 当前默认只会穿入0 1两层算法配置，多余层数穿入不做解析.
    // 0层算法 当前先写死
    // 1层算法 按默认值取
    AlgTypeLevel0 algType0 = AlgTypeLevel0::ALG_LEVEL0_NP_DOUBLE_RING;
    // 构造 1层 algoType, 未填写则取默认值
    HcclAlgoType level1AlgoConfig = HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT;
    std::string algos;
    uint32_t level = 0;
    AlgTypeLevel1 algType1 = AlgTypeLevel1::ALG_LEVEL1_RESERVED;

    std::size_t found = algConfig.find("=");
    std::string curAlgConfig = algConfig.substr(found + 1);
    bool useDefault = SplitHcclAlgoGetLevel1Res(curAlgConfig, algos);
    if (useDefault == false) {
        ParserHcclAlgoLevel1(algos, level, level1AlgoConfig);
    }
    SetAlgoLevel1(commAicpu, level1AlgoConfig, moduleNum, algType1, useDefault);
    algType.algoLevel0 = algType0;
    algType.algoLevel1 = algType1;
}

static const std::unordered_set<std::string> STEP_SIZE_SUPPORT_LIST = {"AlltoAll=level0:fullmesh;level1:pairwise"};
HcclResult ParseCcOpTilingData(CommonHcclMsg* commonHcclMsg, int32_t groupIdx)
{
    const HcclTilingVersion version = commonHcclMsg->version;
    HCCL_INFO("Hccl client message version %u", static_cast<u32>(version));
    AicpuKfcRpcServerV2* rpc = GetCommRpcServer(groupIdx);
    rpc->SetStepSize(0U);
    rpc->SetTotalStep((0U));
    if (version == HcclTilingVersion::DEPRECATED_TILING_VERSION) {
        return HCCL_SUCCESS;
    }

    Mc2CcTilingInner* mc2CcTiling = reinterpret_cast<Mc2CcTilingInner*>(commonHcclMsg->ccOpTilingData);
    if (mc2CcTiling == nullptr) {
        HCCL_ERROR("Tiling is nullptr.");
        return HCCL_E_PARA;
    }

    // 校验tiling的groupName与当前接收数据的group 的index是否一致
    int32_t tilingGroupIdx = GetComGroupIdx(std::string(mc2CcTiling->groupName));
    if (tilingGroupIdx != groupIdx) {
        HCCL_ERROR(
            "Failed to check groupName %s, groupIdx %d, tiling GroupIdx %d", mc2CcTiling->groupName, groupIdx,
            tilingGroupIdx);
        return HCCL_E_PARA;
    }

    HcclOpResParam* commParam = GetCommAicpuResInst(groupIdx);
    std::string curAlgName;
    CHK_PRT_RET(
        !SelectAlgName(mc2CcTiling->algConfig, commParam->topoInfo.topoType, curAlgName),
        HCCL_ERROR("Failed to select algname."), HCCL_E_PARA);
    AlgType algType;
    HcclCommAicpu* commAicpu = GetCommAicpuCommInst(groupIdx);
    SelectAlgType(commAicpu, mc2CcTiling->algConfig, commParam->topoInfo.moduleNum, algType);
    std::string curTag = std::string(mc2CcTiling->groupName) + std::to_string(mc2CcTiling->opType);
    SetCommInfoCtx(
        std::string(mc2CcTiling->groupName), static_cast<u8>(mc2CcTiling->opType),
        CommInfoCtx{algType, curAlgName, curTag});

    if (mc2CcTiling->stepSize > 0U) {
        CHK_PRT_RET(
            STEP_SIZE_SUPPORT_LIST.find(mc2CcTiling->algConfig) == STEP_SIZE_SUPPORT_LIST.end(),
            HCCL_ERROR("Alg %s is not supported when step size is %u.", mc2CcTiling->algConfig, mc2CcTiling->stepSize),
            HCCL_E_PARA);
        rpc->SetStepSize(mc2CcTiling->stepSize);
        rpc->SetTotalStep(commParam->rankSize);
    }
    return HCCL_SUCCESS;
}

void RepeatUpdateOpParam(
    hccl::OpParam& opParam, CommonHcclMsg* hcclMsg, HcclMsgExt* hcclMsgExt, hccl::HcclCommAicpu* commAicpu)
{
    uint64_t dataLen = hcclMsg->dataCnt * DataUnitSize(hcclMsg->hcclDataType);
    if (hcclMsg->commType == HcclCMDType::HCCL_CMD_ALLTOALLV
        || (hcclMsg->commType == HcclCMDType::HCCL_CMD_ALLTOALL && hcclMsg->strideCount > 0)) {
        for (uint32_t i = 0; i < commAicpu->GetRankSize(); i++) {
            hcclMsgExt->sendOffset[i] += hcclMsgExt->sendCounts[i];
            hcclMsgExt->recvOffset[i] += hcclMsgExt->recvCounts[i];
        }
    } else {
        opParam.outputPtr = reinterpret_cast<void*>(reinterpret_cast<int8_t*>(opParam.outputPtr) + dataLen);
        opParam.inputPtr = reinterpret_cast<void*>(reinterpret_cast<int8_t*>(opParam.inputPtr) + dataLen);
    }
}

HcclResult AddTaskForHcclMsgV2(
    hccl::HcclCommAicpu* comm, AicpuKfcRpcServerV2* rpc, CommonHcclMsg* hcclMsg, const HcclOpResParam* commParam)
{
    uint32_t curTurnCntForKernel = 0;
    rpc->SetMsgPosForKernel(0);
    CommInfoCtx curCtx;
    HcclResult ret = GetCommInfoCtx(comm->GetGroupName(), static_cast<uint8_t>(hcclMsg->commType), curCtx);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("Failed to get comm info from aicpu instance.");
        return HCCL_E_INTERNAL;
    }
    hccl::OpParam opParam;
    std::string algName = curCtx.algName;
    opParam.tag = curCtx.tag;
    std::string newTag = opParam.tag + "_mc2" + algName + "_device";

    u32 aicpuAlgType = (static_cast<u32>(curCtx.algType.algoLevel2) << (HCCL_LEVEL_ALGO_WIDTH + HCCL_LEVEL_ALGO_WIDTH))
                       + (static_cast<u32>(curCtx.algType.algoLevel1) << HCCL_LEVEL_ALGO_WIDTH)
                       + static_cast<u32>(curCtx.algType.algoLevel0);
    comm->SetAlgType(static_cast<u64>(aicpuAlgType));
    PrepareOpParam(&opParam, hcclMsg, *rpc, comm);
    hccl::AlgResourceResponse* algResResponse;
    std::unique_ptr<hccl::CollExecutorBase> executor;
    while (curTurnCntForKernel < hcclMsg->repeatCnt) {
        HCCL_INFO("Orchestrate curTurnCntForKernel %u, hcclMsg->repeatCnt %u", curTurnCntForKernel, hcclMsg->repeatCnt);
        curTurnCntForKernel++;
        rpc->SetMsgPosForKernel(curTurnCntForKernel);
        CHK_RET(comm->GetAlgResponseRes(newTag, algName, opParam, commParam, executor, algResResponse));
        HcclResult hcclRet = comm->Orchestrate(newTag, algName, opParam, executor, *algResResponse, commParam);
        AicpuKfcProf::GetCurrentAicpuProf()->workCnt++;
        CHK_PRT_RET(
            hcclRet != HCCL_SUCCESS,
            HCCL_ERROR("Executor op fail, opParam.tag[%s], algName[%s]", newTag.c_str(), algName.c_str()), hcclRet);
        RepeatUpdateOpParam(opParam, hcclMsg, rpc->GetHcclMsgExtPtr(), comm);
    }
    return HCCL_SUCCESS;
}

HcclResult RunRpcServerLoopProcess(const std::vector<u32>& groupIds, u32 localGroupIdx, bool& finalizeFlag)
{
    HcclMsg hcclMsg;
    CommonHcclMsg commonHcclMsg;
    const u32 groupIdx = groupIds[localGroupIdx];
    AicpuKfcRpcServerV2* rpc = GetCommRpcServer(groupIdx);
    HcclCommAicpu* comm = GetCommAicpuCommInst(groupIdx);
    HcclOpResParam* commParam = GetCommAicpuResInst(groupIdx);
    u32 start = 0U;
    u32 end = 0U;
    rpc->GetLocalQueueRange(start, end);
    const u64 tilingBase = rpc->GetTilingBaseAddr();
    HcclResult ret;
    do {
        ret = RpcServerPreCheck(rpc, comm, finalizeFlag);
        if (ret == HCCL_E_AGAIN) {
            return HCCL_SUCCESS;
        } else if (ret != HCCL_SUCCESS) {
            return ret;
        }

        HcclMsg(*msgLists)[HCCL_MSG_CNT] = rpc->GetMsgWorkSpace();
        SetMsgEnableFlag(groupIdx, false);
        for (u32 i = start; i <= end; ++i) {
            if (rpc->GetIsFinalize(i)) {
                continue;
            }

            BarrierStatus status = BarrierStatus::NO_BARRIER;
            if (BarrierProcess(groupIdx, localGroupIdx, i, status) != HCCL_SUCCESS) {
                rpc->DumpBarrierInfo(localGroupIdx, comm->GetSlaveStream()[i].sqId(), comm->GetDevId());
                rpc->PrintAllHcclMsgArea(commParam->rankSize);
                return HCCL_E_INTERNAL;
            }

            if (status != BarrierStatus::NO_BARRIER) {
                SetMsgEnableFlag(groupIdx, true);
                continue;
            }

            uint32_t currMsgPos = rpc->GetMsgPos(i);
            if (!rpc->ReadAddrMsg(&hcclMsg, msgLists[i], i, currMsgPos, commParam->rankSize)) {
                if (rpc->IsExceedLimit(static_cast<HcclCMDType>(hcclMsg.commType.prepareType), commParam->rankSize)) {
                    return HCCL_E_INTERNAL;
                }
                AddMsgInValidCount(groupIdx);
                if (GetMsgInValidCount(groupIdx) == LOGCOUNT_PRINT_TIMEOUT) {
                    HCCL_WARNING(
                        "Fail to get msg, addr is %p, queue %u, msgPos %u, group %s", msgLists[i], i, currMsgPos,
                        comm->GetGroupName().c_str());
                }
                if (rpc->IsPrintLog()) {
                    LogControl logControl(false, true);
                    comm->PrintTaskExceptionAllComm();
                }
                continue;
            }

            if (GetMsgInValidCount(groupIdx) > LOGCOUNT_PRINT_TIMEOUT) {
                HCCL_WARNING(
                    "Msg channel restores, addr is %p, queue %u, msgPos %u, group %s", msgLists[i], i, currMsgPos,
                    comm->GetGroupName().c_str());
            }
            SetMsgStartTime(groupIdx);
            ClearMsgInValidCount(groupIdx);
            SetMsgEnableFlag(groupIdx, true);

            GetCommonHcclMsg(&hcclMsg, &commonHcclMsg, tilingBase);
            HCCL_INFO(
                "Process message queue %u pos %u seq num %u type %u group %s.", i, currMsgPos, commonHcclMsg.seqNum,
                commonHcclMsg.commType, comm->GetGroupName().c_str());
            if (commonHcclMsg.commType == HcclCMDType::HCCL_CMD_FINALIZE) {
                FinalizeProcess(i, *comm, *rpc);
                continue;
            } else if (commonHcclMsg.commType == HcclCMDType::HCCL_CMD_INTER_GROUP_SYNC) {
                ret = AddTaskForGroupSyncMsg(groupIds, localGroupIdx, &commonHcclMsg);
                if (ret == HCCL_E_UNAVAIL) {
                    SetMsgEnableFlag(groupIdx, false);
                    rpc->SetNeedRetryFlag(true);
                    continue;
                } else if (ret != HCCL_SUCCESS) {
                    return ret;
                }
            } else if (commonHcclMsg.commType == HcclCMDType::HCCL_CMD_BARRIER) {
                rpc->GetBarrierInfoByGroupIdx(localGroupIdx)[i].status = BarrierStatus::SELF_BARRIER;
            } else {
                ret = rpc->ProcessExpectPrepareMsg(commonHcclMsg.seqNum, GetExpectPrepareId(i));
                if (ret == HCCL_E_UNAVAIL) {
                    SetMsgEnableFlag(groupIdx, false);
                    rpc->SetNeedRetryFlag(true);
                    continue;
                } else if (ret != HCCL_SUCCESS) {
                    return ret;
                }
                rpc->SetNeedRetryFlag(false);
                rpc->SetMsgRepeatCnt(commonHcclMsg.repeatCnt);
                rpc->SetMsgHandlePos(currMsgPos, commonHcclMsg.selfHandleID);
                if (commonHcclMsg.commType == HcclCMDType::HCCL_CMD_BATCH_WRITE) {
                    hccl::OpParam opParam;
                    PrepareOpParam(&opParam, &commonHcclMsg, *rpc, comm);
                    CHK_RET(AicpuKfcBatchwriteProcess::BatchWriteProcess(opParam, *comm, *commParam));
                } else {
                    CHK_RET(ParseCcOpTilingData(&commonHcclMsg, groupIdx));
                    CHK_RET(TaskOrchestrator::IsSupportRDMAReduce(
                        commonHcclMsg.commType, commonHcclMsg.hcclDataType, commonHcclMsg.opType));
                    CHK_RET(AddTaskForHcclMsgV2(comm, rpc, &commonHcclMsg, commParam));
                }
                SetExpectPrepareId(i, commonHcclMsg.seqNum + 1U);
            }
            rpc->SetMsgPos(i, (currMsgPos + 1) % HCCL_MSG_CNT);
        }
    } while (CheckMsgEnableFlag(groupIdx));
    return HCCL_SUCCESS;
}

std::string GetNewTag(uint32_t groupIdx)
{
    hccl::HcclCommAicpu* comm = GetCommAicpuCommInst(groupIdx);
    AicpuKfcRpcServerV2* rpc = GetCommRpcServer(groupIdx);
    uint32_t currMsgPos = rpc->GetMsgPos();
    currMsgPos = currMsgPos > 0 ? currMsgPos - 1 : currMsgPos;
    HcclMsg(*msgLists)[HCCL_MSG_CNT] = rpc->GetMsgWorkSpace();
    CommInfoCtx curCtx;
    GetCommInfoCtx(
        comm->GetGroupName(), static_cast<HcclCMDType>(msgLists[0U][currMsgPos].commType.prepareType), curCtx);
    return curCtx.tag + "_mc2" + curCtx.algName + "_device";
    ;
}

void ResetRestartParam(RestartParam& restartParam)
{
    restartParam.restartCnt++;
    restartParam.restartFlag = false;
    restartParam.consultationAllEnd = 0;
    for (uint32_t i = 0; i < MAX_COMM_CTX_NUM; i++) {
        restartParam.consultationResult[i] = false;
        restartParam.linkChanged[i] = false;
        restartParam.fsmState[i] = HcclOpExecFSM::HCCL_OP_EXEC_FSM_WAIT_END;
        restartParam.errorCode[i] = KfcError::kNone;
    }
}

HcclResult RestartProcessConsulation(
    RestartParam& restartParam, bool& finalizeAllEnd, bool* finalizeMask, std::vector<u32> groupIds)
{
    for (size_t i = 0U; i < groupIds.size(); ++i) {
        if (restartParam.consultationResult[i]) {
            continue;
        }
        hccl::HcclCommAicpu* comm = GetCommAicpuCommInst(groupIds[i]);
        if (comm == nullptr) {
            HCCL_ERROR("Failed to obtain the AICPU communication domain pointer."
                       "Check whether the parameters are correct.");
            return HCCL_E_PARA;
        }
        std::string newTag = GetNewTag(groupIds[i]);
        HcclResult ret = AicpuKfcRetryProcess::RetryProcess(*comm, restartParam, i);
        if (ret == HCCL_SUCCESS) {
            if (restartParam.consultationResult[i]) {
                HCCL_RUN_INFO("[MC2][AICPU]MC2 restart process success, groupIdx %u , tag %s", i, newTag.c_str());
                restartParam.consultationAllEnd++;
            }
        } else {
            // 重执行协商流程失败，直接返回错误
            HCCL_ERROR(
                "[MC2][AICPU]MC2 restart process groupIdx %u failed at state %u ret is %u tag is %s", i,
                restartParam.fsmState[i], ret, newTag.c_str());
            return ret;
        }
    }

    // 全部协商重执行完成
    if (restartParam.consultationAllEnd >= groupIds.size()) {
        HCCL_RUN_INFO("[MC2][AICPU]MC2 restart process all group success, reset param and write restart");
        SetExpectPrepareId(0U, 0U);
        ResetRestartParam(restartParam);
        finalizeAllEnd = false;
        for (size_t i = 0U; i < groupIds.size(); ++i) {
            // 重置结束标志
            finalizeMask[i] = false;
            // 重置rpc
            AicpuKfcRpcServerV2* rpc = GetCommRpcServer(groupIds[i]);
            rpc->Reset();
            rpc->WriteRestartFlag();
            SetMsgStartTime(groupIds[i]);
            HCCL_INFO("MC2 restart process reset rpc param end. groupIndex = %u", i);
        }
        SetKernelStartTime();
    }
    return HCCL_SUCCESS;
}

void RecordReportStatus(const std::vector<u32>& groupIds, dfx::ReportStatus status)
{
    for (const auto i : groupIds) {
        hccl::HcclCommAicpu* comm = GetCommAicpuCommInst(i);
        if (comm != nullptr) {
            comm->RecordReportStatus(status);
        }
    }
}

bool CheckMsgTimeOut(const std::vector<u32>& groupIds)
{
    if ((GetCurCpuTimestamp() - g_timeOutInfoInst.kernelStartTime)
        > static_cast<unsigned long long>(NSEC_PER_SEC * KERNEL_TIMEOUT)) {
        HCCL_ERROR("Kernel Execute TimeOut %lus...", KERNEL_TIMEOUT);
        return true;
    }
    int timeoutFlag = 0;
    for (u32 idx : groupIds) {
        if (CheckMsgEnableFlag(idx)
            && (GetCurCpuTimestamp() - GetMsgStartTime(idx))
                   > static_cast<unsigned long long>(NSEC_PER_SEC * KERNEL_TIMEOUT)) {
            HCCL_ERROR("comm group idx %d ReadValidMsg timeout %lus... ", idx, KERNEL_TIMEOUT);
            timeoutFlag++;
        }
    }
    if (timeoutFlag) {
        return true;
    }
    return false;
}

HcclResult SetNsOpStatus(const std::vector<u32>& groupIds, bool state)
{
    for (const auto i : groupIds) {
        hccl::HcclCommAicpu* comm = GetCommAicpuCommInst(i);
        if (comm != nullptr) {
            comm->SetNsOpStatus(state);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult RunRpcServerInnerProcessV2(const std::vector<u32>& groupIds)
{
    const bool retryEnable = GetOpRetryEnable(groupIds);
    RestartParam restartParam;
    auto opStartTime = std::chrono::steady_clock::now();
    bool finalizeMask[MAX_COMM_CTX_NUM] = {false, false, false};
    SetKernelStartTime();
    AicpuKfcProf::GetCurrentAicpuProf()->commInitEndTime = GetCurCpuTimestamp(true);
    if (CheckNsStopLaunchStatus(groupIds) != HCCL_SUCCESS) {
        HCCL_WARNING("the op should not be launched in the suspending status");
        return HCCL_E_SUSPENDING;
    }
    CHK_RET(SetNsOpStatus(groupIds, true));
    while (true) {
        bool finishFlag = true;
        for (uint32_t i = 0; i < groupIds.size(); i++) {
            if (finalizeMask[i]) {
                continue;
            }
            finishFlag = false;
            if (restartParam.restartFlag) {
                continue;
            }
            HcclResult res = RunRpcServerLoopProcess(groupIds, i, finalizeMask[i]);
            if (res == HCCL_E_SUSPENDING) {
                if (retryEnable) {
                    restartParam.restartFlag = true;
                    break;
                }
                HcclCommAicpu* comm = GetCommAicpuCommInst(groupIds[i]);
                if (comm != nullptr && comm->GetNsStopLaunchStatus()) {
                    finalizeMask[i] = true;
                    AicpuKfcRpcServerV2* rpc = GetCommRpcServer(groupIds[i]);
                    rpc->SetNeedRetryFlag(false);
                    comm->SetCommRecoveryFlag(true);
                    (void)comm->BackGroundSetStatus(KfcStatus::kStoplaunch);
                } else {
                    HCCL_ERROR("[MC2][Restart]Mc2 can not retry, not all comm retryEnable are true");
                    return res;
                }
            } else if (res != HCCL_SUCCESS) {
                HCCL_ERROR("RPC server failed to run.");
                return res;
            }
        }

        if (restartParam.restartFlag && HcclAicpuUtils::GetBlockIdx() == 0U) {
            HcclResult res = RestartProcessConsulation(restartParam, finishFlag, finalizeMask, groupIds);
            if (res != HCCL_SUCCESS) {
                HCCL_ERROR(
                    "[MC2][AICPU]MC2 restart process failed, restartCnt = %u, res = %u", restartParam.restartCnt, res);
                RecordReportStatus(groupIds, dfx::ReportStatus::kRetryFail);
                return res;
            }
        }
        // 全部结束
        if (finishFlag) {
            HCCL_INFO("RPC server process ends.");
            AicpuKfcProf::GetCurrentAicpuProf()->receiveFinalizeTime = GetCurCpuTimestamp(true);
            CHK_RET(SetNsOpStatus(groupIds, false));
            if (restartParam.restartCnt > 0) {
                auto opEndTime = std::chrono::steady_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::seconds>(opEndTime - opStartTime).count();
                HCCL_RUN_INFO(
                    "[MC2][AICPU]MC2 restart exec success, restartCnt = %u, take time = %ld s", restartParam.restartCnt,
                    duration);
                RecordReportStatus(groupIds, dfx::ReportStatus::kRetrySuccess);
            }
            return HCCL_SUCCESS;
        }
        // 消息超时或总执行时间超时
        if (CheckMsgTimeOut(groupIds)) {
            HCCL_ERROR("RPC server process Timeout.");
            for (uint32_t i : groupIds) {
                AicpuKfcRpcServerV2* rpc = GetCommRpcServer(i);
                HcclOpResParam* commParam = GetCommAicpuResInst(i);
                if (rpc != nullptr && commParam != nullptr) {
                    rpc->PrintAllHcclMsgArea(commParam->rankSize);
                }
            }
            return HCCL_E_TIMEOUT;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult RunRpcServerApiV2(void* tilingData, const std::vector<u32>& groupIds)
{
    // 待适配 startthread DFX
    uint32_t commNum = MC2TilingGetHcommCnt(tilingData);
    for (uint32_t i = 0; i < commNum; i++) {
        Mc2HcommCfg* cfg = MC2TilingGetHcommCfg(tilingData, i);
        int32_t groupIdx = GetComGroupIdx(std::string(cfg->groupName));
        if (groupIdx < 0) {
            HCCL_ERROR("%s idx %d cannot get group by hcomId %s", __func__, i, cfg->groupName);
            return HCCL_E_INTERNAL;
        }
        hccl::HcclCommAicpu* comm = GetCommAicpuCommInst(groupIdx);
        if (comm == nullptr) {
            HCCL_ERROR("%s cannot get CommAicpu by groupIdx %d", __func__, groupIdx);
            return HCCL_E_INTERNAL;
        }
        HcclOpResParam* commParam = GetCommAicpuResInst(groupIdx);
        std::string curAlgName;
        if (!SelectAlgName(cfg->algConfig, commParam->topoInfo.topoType, curAlgName)) {
            return HCCL_E_INTERNAL;
        }
        std::string curTag = std::string(cfg->groupName) + std::to_string(cfg->opType);
        uint32_t moduleNum = commParam->topoInfo.moduleNum;
        AlgType algType;
        SelectAlgType(comm, cfg->algConfig, moduleNum, algType);
        SetCommInfoCtx(
            std::string(cfg->groupName), static_cast<u8>(cfg->opType), CommInfoCtx{algType, curAlgName, curTag});
    }
    CHK_RET(RunRpcServerInnerProcessV2(groupIds));
    return HCCL_SUCCESS;
}

HcclResult KfcStepSizeHandler(const std::vector<u64>& args)
{
    CHK_PRT_RET(args.size() != 3U, HCCL_ERROR("Invalid args size %zu.", args.size()), HCCL_E_INTERNAL);
    const AicpuKfcRpcServerV2* rpc = reinterpret_cast<const AicpuKfcRpcServerV2*>(args[0]);
    u8 stepSize = rpc->GetStepSize();
    if (stepSize == 0U) {
        HCCL_INFO("The orchestrating OP is not a fine-grained one.");
        return HCCL_SUCCESS;
    }

    Mc2Handler* handler = reinterpret_cast<Mc2Handler*>(args[1]);
    handler->version = 0U;
    handler->commitAddr = rpc->GetCommitareaAddr(rpc->GetMsgPos());
    handler->finishAddr = rpc->GetFinishAddr(rpc->GetMsgPos());
    handler->valueAddr = rpc->GetTurnNumAddr();
    handler->rankSize = args[2];
    handler->repeatCnt = rpc->GetMsgPosForKernel();
    handler->stepSize = stepSize;
    handler->skipLocalRankCopy = 0U;
    handler->skipBufferWindowCopy = 0U;
    HCCL_INFO(
        "Prepare MC2 handler: commitAddr %p, finishAddr %p, valueAddr %p, rankSize %u, repeat %u, stepSize %u.",
        handler->commitAddr, handler->finishAddr, handler->valueAddr, handler->rankSize, handler->repeatCnt,
        handler->stepSize);
    return HCCL_SUCCESS;
}

HcclResult KfcNotifyPost(const std::vector<u64>& args)
{
    CHK_PRT_RET(args.size() != 3U, HCCL_ERROR("Invalid args size %zu.", args.size()), HCCL_E_INTERNAL);
    AicpuKfcRpcServerV2* rpc = reinterpret_cast<AicpuKfcRpcServerV2*>(args[0]);
    CHK_PRT_RET(rpc == nullptr, HCCL_ERROR("Failed to get rpc pointer."), HCCL_E_INTERNAL);
    if (rpc->GetStepSize() != 0 || rpc->GetTotalQueueNum() > 0U) {
        HCCL_DEBUG("No need to add notify for MC2.");
        return HCCL_SUCCESS;
    }
    return rpc->AddCcoreNotify(
        reinterpret_cast<HcclDispatcher>(args[1]), rpc->GetFinishAddr(rpc->GetMsgPos()), rpc->GetMsgPosForKernel(),
        reinterpret_cast<Stream*>(args[2]));
}

HcclResult KfcNotifyWait(const std::vector<u64>& args)
{
    CHK_PRT_RET(args.size() != 3U, HCCL_ERROR("Invalid args size %zu.", args.size()), HCCL_E_INTERNAL);
    AicpuKfcRpcServerV2* rpc = reinterpret_cast<AicpuKfcRpcServerV2*>(args[0]);
    CHK_PRT_RET(rpc == nullptr, HCCL_ERROR("Failed to get rpc pointer."), HCCL_E_INTERNAL);
    if (rpc->GetStepSize() != 0 || rpc->GetTotalQueueNum() > 0U) {
        HCCL_DEBUG("No need to add wait for MC2.");
        return HCCL_SUCCESS;
    }
    return rpc->AddCcoreWait(
        reinterpret_cast<HcclDispatcher>(args[1]), rpc->GetCommitareaAddr(rpc->GetMsgPos()), rpc->GetMsgPosForKernel(),
        reinterpret_cast<Stream*>(args[2]), false);
}

HcclResult KfcClearMsgArea(const std::vector<u64>& args)
{
    CHK_PRT_RET(args.size() != 1U, HCCL_ERROR("Invalid args size %zu.", args.size()), HCCL_E_INTERNAL);
    AicpuKfcRpcServerV2* rpc = reinterpret_cast<AicpuKfcRpcServerV2*>(args[0]);
    HcclMsgArea* hcclMsgArea = rpc->GetHcclMsgArea();
    if (hcclMsgArea != nullptr) {
        (void)memset_s(hcclMsgArea, sizeof(HcclMsgArea), 0, sizeof(HcclMsgArea));
    }
    hcclMsgArea->controlMsg.resetSeq = 1;
    return HCCL_SUCCESS;
}

HcclResult KfcClearCommitTurn(const std::vector<u64>& args)
{
    CHK_PRT_RET(args.size() != 1U, HCCL_ERROR("Invalid args size %zu.", args.size()), HCCL_E_INTERNAL);
    AicpuKfcRpcServerV2* rpc = reinterpret_cast<AicpuKfcRpcServerV2*>(args[0]);
    HcclMsgArea* hcclMsgArea = rpc->GetHcclMsgArea();
    if (hcclMsgArea != nullptr) {
        for (uint32_t i = 0; i < HCCL_MSG_CNT; i++) {
            hcclMsgArea->commMsg.singleMsg.commitTurnCnt[i].cnt = 0xFF;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult PrepareHcommInstance(HcclOpResParam* commParam, const Mc2InitTilingInner* tiling = nullptr)
{
    const std::string& group = commParam->hcomId;
    hccl::HcclCommAicpu* hcclCommAicpu = AicpuHcclProcess::AicpuGetCommbyGroup(group);
    CHK_PRT_RET(
        hcclCommAicpu == nullptr, HCCL_ERROR("RunAicpuRpcSrvLaunchV2 get Hcclcomm error group [%s]", group.c_str()),
        HCCL_E_INTERNAL);

    DevType devType = hcclCommAicpu->GetDevType();
    CHK_PRT_RET(
        devType != DevType::DEV_TYPE_910_93,
        HCCL_ERROR("Platform %u not support, please use 910_93 platform.", static_cast<u32>(devType)), HCCL_E_INTERNAL);

    const DfxExtendInfo* dfxInfo = hcclCommAicpu->GetDfxExtendInfo();
    CHK_PRT_RET(
        dfxInfo->cqeStatus != dfx::CqeStatus::kDefault || dfxInfo->pollStatus == PollStatus::kStopAsException,
        HCCL_ERROR(
            "Exist errors before, cqeStatus:%d, pollStatus:%d, group[%s]", dfxInfo->cqeStatus, dfxInfo->pollStatus,
            group.c_str()),
        HCCL_E_INTERNAL);

    const u32 groupIdx = InsertComIdMap(group);
    HcclResult ret = InsertCommInst(groupIdx, hcclCommAicpu, commParam);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("Failed to insert comm inst."), HCCL_E_INTERNAL);

    AicpuKfcRpcServerV2* rpcServer = GetCommRpcServer(groupIdx);
    CHK_PRT_RET(
        rpcServer == nullptr,
        HCCL_ERROR("RunAicpuRpcSrvLaunchV2 get rpc inst error idx %d group [%s]", groupIdx, group.c_str()),
        HCCL_E_INTERNAL);

    ret = rpcServer->Init(commParam->mc2WorkSpace, tiling);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("Failed to init for group [%s]", group.c_str()), HCCL_E_INTERNAL);

    hcclCommAicpu->SetIsDeviceMode(true);
    hcclCommAicpu->SetAicpuRpcServer(reinterpret_cast<u64>(rpcServer));
    hcclCommAicpu->RegisterKfcHandler(AicpuKfcHandlerType::kSetStepSize, KfcStepSizeHandler);
    hcclCommAicpu->RegisterKfcHandler(AicpuKfcHandlerType::kNotifyRecord, KfcNotifyPost);
    hcclCommAicpu->RegisterKfcHandler(AicpuKfcHandlerType::kNotifyWait, KfcNotifyWait);
    hcclCommAicpu->RegisterKfcHandler(AicpuKfcHandlerType::kClearMsgArea, KfcClearMsgArea);
    hcclCommAicpu->RegisterKfcHandler(AicpuKfcHandlerType::kClearCommitTurn, KfcClearCommitTurn);
    hcclCommAicpu->RegisterKfcHandler(
        AicpuKfcHandlerType::kSetProfTimeStart, []([[maybe_unused]] const std::vector<u64>& args) -> HcclResult {
            AicpuKfcProf::SetKfcTimeLine(KfcTimeLine::HCC_EXEC_START_TIME);
            return HCCL_SUCCESS;
        });
    hcclCommAicpu->RegisterKfcHandler(
        AicpuKfcHandlerType::kSetProfTimeOrch, []([[maybe_unused]] const std::vector<u64>& args) -> HcclResult {
            AicpuKfcProf::SetKfcTimeLine(KfcTimeLine::SEND_TASK_START_TIME);
            return HCCL_SUCCESS;
        });
    hcclCommAicpu->RegisterKfcHandler(
        AicpuKfcHandlerType::kSetProfTimeEnd, []([[maybe_unused]] const std::vector<u64>& args) -> HcclResult {
            AicpuKfcProf::SetKfcTimeLine(KfcTimeLine::SEND_SQE_FINISH_TIME);
            return HCCL_SUCCESS;
        });
    return HCCL_SUCCESS;
}
ANONYMOUS_NAMESPACE_END

u32 AicpuKfcProcess::AicpuRpcResInit(HccCommResParamTask* commParam)
{
    HcclAicpuUtils::PrintHcclCombinOpParam(*commParam);

    AicpuComContext* ctx = AicpuGetComContext();
    if (ctx->alreadyInit) {
        if (strncmp(ctx->hcomId, commParam->hcomId, HCCL_COMM_DOMAIN_KEY_MAX_LEN)) {
            HCCL_ERROR("the comm domain is not valid old [%s] != new[%s].", ctx->hcomId, commParam->hcomId);
            return AC_ERROR_INVALID_PARAM;
        }
        HCCL_INFO("The ctx was already inited");
        return 0;
    }
    AicpuSqeContext::InitSqeContext();
    (void)memset_s(ctx, sizeof(AicpuComContext), 0, sizeof(AicpuComContext));
    s32 enableEvent = 0;
    ctx->logLevel = dlog_getlevel(HCCL, &enableEvent);
    ctx->rankId = commParam->rankId;
    ctx->rankNum = commParam->rankNum;
    ctx->windowSize = commParam->winSize;
    ctx->workSpaceAddr = commParam->mc2WorkSpace.workSpace;
    ctx->curTurnCnt = 0;
    ctx->commAlg = 0;
    ctx->multiServerFlag = commParam->multiServerFlag;
    std::iota(ctx->turnValue, ctx->turnValue + TILING_TURN_MAX * AC_MAX_RANK_NUM, 0);
    HcclSignalInfo* sigInfo = &commParam->signalInfo.aicpuNotify;
    std::shared_ptr<LocalNotify> localNitfy;
    EXCEPTION_CATCH((localNitfy = std::make_shared<LocalNotify>()), return HCCL_E_PTR);
    CHK_RET(localNitfy->Init(*sigInfo, NotifyLoadType::DEVICE_NOTIFY));
    ctx->kfcNotifyId = sigInfo->resId;

    CHK_RET(hrtDrvGetLocalDevIDByHostDevID(sigInfo->devId, &(ctx->devId)));

    if (ctx->multiServerFlag) {
        CHK_RET(InitIbversData(commParam, ctx));
    } else {
        InitRankInfo(commParam, ctx);
        CHK_RET(InitSignalInfo(commParam, ctx));
        CHK_RET(InitEventId(commParam, ctx));
    }

    CHK_RET(AicpuKfcProcess::InitStreamInfo(commParam, ctx));
    CHK_RET(InitAicpuOpNotify(commParam, ctx));
    CHK_RET(InitTimeOutConfig(commParam, ctx));
    HCCL_INFO("remote_udevid: %u, local_devid: %u, ssid: %u", sigInfo->devId, ctx->devId, ctx->ssid);
    ctx->directlySendMainSteramSqe = false;
    ctx->clusterId = HcclAicpuUtils::GetCurClusterId();
    auto ret = strcpy_s(ctx->hcomId, sizeof(ctx->hcomId), commParam->hcomId);
    HCCL_DEBUG("Init hcom group [%s] strcpy ret %d", ctx->hcomId, static_cast<int>(ret));
    ctx->determinism = (commParam->config.deterministic != 0);
    ctx->retryEnable = (commParam->config.retryEnable == 1);
    ctx->retryHoldTime = commParam->config.retryHoldTime;
    ctx->retryIntervalTime = commParam->config.retryIntervalTime;
    HCCL_DEBUG(
        "[%s] ctx->retryEnable [%d], ctx->retryHoldTime [%u ms], ctx->retryIntervalTime [%u ms]", __func__,
        ctx->retryEnable, ctx->retryHoldTime, ctx->retryIntervalTime);
    CHK_RET(InitChipType(ctx));
    ctx->overflowAddr = commParam->overFlowAddr;
    ctx->onlyRead = commParam->onlyRead;
    ctx->dfxExtendInfo.dfxTimeOutConfig.useCredit = true;
    dfx::AicpuProfilingManager::Init(ctx);
    ctx->alreadyInit = true;
    ctx->commOpenStatus = true;
    ctx->opIndex = 0;
    if (commParam->kfcControlTransferH2DParams.buffLen != 0) {
        EXCEPTION_CATCH((ctx->kfcControlTransferH2D = std::make_shared<hccl::HDCommunicate>()), return HCCL_E_PTR);
        CHK_SMART_PTR_NULL(ctx->kfcControlTransferH2D);
        CHK_RET(ctx->kfcControlTransferH2D->InitDevice(commParam->kfcControlTransferH2DParams));
    }
    if (commParam->kfcStatusTransferD2HParams.buffLen != 0) {
        EXCEPTION_CATCH((ctx->kfcStatusTransferD2H = std::make_shared<hccl::HDCommunicate>()), return HCCL_E_PTR);
        CHK_SMART_PTR_NULL(ctx->kfcStatusTransferD2H);
        CHK_RET(ctx->kfcStatusTransferD2H->InitDevice(commParam->kfcStatusTransferD2HParams));
    }
    AicpuHcclProcess::CopyCtxInfo(ctx);
    AicpuHcclProcess::CallMC2MaintenanceThread(ctx);
    if (MC2TraceUtils::Init() != HCCL_SUCCESS) {
        HCCL_ERROR("Init trace failed.");
        return static_cast<u32>(HCCL_E_INTERNAL);
    }
    HCCL_RUN_INFO("End %s", __func__);
    return 0;
}

std::unordered_map<int32_t, uint32_t> g_streamIdMap;
u32 AicpuKfcProcess::GetStreamRankIdx(s32 actualStreamId)
{
    auto it = g_streamIdMap.find(actualStreamId);
    return it == g_streamIdMap.cend() ? UINT32_MAX : it->second;
}

HcclResult AicpuKfcProcess::DealReturnValue(const AicpuComContext* ctx, const HcclResult ret)
{
    if (ctx->isStopLaunch) {
        AicpuHcclProcess::CopyCtxForBackGroundDfx(ctx);
        CHK_RET(AicpuHdcUtils::SetOpExecStatus(ctx->kfcStatusTransferD2H, KfcStatus::kStoplaunch, KfcError::kNone, 0));
        return HCCL_E_SUSPENDING;
    } else if (ctx->endStopLaunch) {
        return HCCL_E_SUSPENDING;
    } else {
        CHK_RET(AicpuHdcUtils::SetOpExecStatus(ctx->kfcStatusTransferD2H, KfcStatus::kError, KfcError::kInner, 0));
        return ret;
    }
}

HcclResult AicpuKfcProcess::AddTaskForHcclMsg(
    AicpuComContext* ctx, AicpuKfcRpcServer& rpc, CommonHcclMsg* hcclMsg, AivAicpuOpParam* msg, u64 tilingBase)
{
    // reduce scatter:在strideLen使能的情况下，如果recvCount * repeat > strideLen 则偏移越界，报错
    if (hcclMsg->commType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER && hcclMsg->strideCount != 0
        && hcclMsg->dataCnt * hcclMsg->repeatCnt > hcclMsg->strideCount) {
        HCCL_ERROR("In ReduceScatter algorithm, when stride Count is not zero, repeatCnt * dataCnt"
                   " should not be greater than strideCount.");
        hcclMsg->PrintMsg("");
        return HCCL_E_PARA;
    }

    AivAicpuOpParam* tmpptr = nullptr;
    AivAicpuOpParam nextMsg;
    u64 dataLen = DataUnitSize(msg->hcclDataType) * msg->count;
    ctx->curTurnCntForKernel = 0;
    ctx->totalTurnCntForKernel = hcclMsg->repeatCnt;
    while (ctx->curTurnCntForKernel < hcclMsg->repeatCnt) {
        HCCL_INFO("ctx->curTurnCntForKernel %u, hcclMsg->repeatCnt %u", ctx->curTurnCntForKernel, hcclMsg->repeatCnt);
        // 当前msg预取仅支持当前及下一条msg都为allgather
        if (hcclMsg->commType == HcclCMDType::HCCL_CMD_ALLGATHER
            && (hcclMsg->hcclDataType == HCCL_DATA_TYPE_FP16 || hcclMsg->hcclDataType == HCCL_DATA_TYPE_BFP16)) {
            HCCL_INFO("Try get AllGather next msg");
            HcclMsg tmpMsg;
            if (ctx->curTurnCntForKernel < (hcclMsg->repeatCnt - 1)) {
                GetNextMsgFromMsg(msg, &nextMsg, dataLen, ctx->rankNum);
                tmpptr = &nextMsg;
            } else if (rpc.CheckRcvAddrMsg(&tmpMsg, ctx->msgPosForKernel + 1)) {
                CommonHcclMsg commonHcclMsg;
                GetCommonHcclMsg(&tmpMsg, &commonHcclMsg, tilingBase);
                rpc.HcclMsg2AicAicpuOpParam(&commonHcclMsg, &nextMsg);
                tmpptr = &nextMsg;
            } else {
                HCCL_INFO("nextMsg is not ready. msgPos %u", ctx->msgPosForKernel + 1);
                tmpptr = nullptr;
            }
            // 如果nextMsg和hcclMsg不同commtype或datatype，nextMsg要置为nullptr
            if (tmpptr != nullptr
                && (tmpptr->commType != hcclMsg->commType || tmpptr->hcclDataType != hcclMsg->hcclDataType)) {
                HCCL_INFO("Set nextMsg nullptr");
                tmpptr = nullptr;
            }
        }
        ctx->curTurnCntForKernel++;
        CHK_RET(AicpuKfcProcess::AicpuCcOpExe(msg, tmpptr, ctx));
        TaskOrchestrator::ActiveRecordMain(AicpuKfcProcess::GetActiveSqId(ctx));
        // 更新msg
        UpdateMsg(msg, dataLen, ctx->rankNum);
    }
    return HCCL_SUCCESS;
}

HcclResult AicpuKfcProcess::RunRpcServerApi(AicpuComContext* ctx, AicpuKfcRpcServer& rpc, u64 tilingBase)
{
    if (ctx->devType != DevType::DEV_TYPE_910B) {
        HCCL_ERROR("Platform not support, please use 910B platform.");
        return HCCL_E_PARA;
    }
    HcclMsg hcclMsg;
    CommonHcclMsg commonHcclMsg;
    AivAicpuOpParam msg;
    AicpuUpdatComContextMumber(offsetof(AicpuComContext, dfxExtendInfo.kfcStatus), DfxKfcStatus::kOneStart);
    AicpuHcclProcess::CallMC2MaintenanceThread(ctx);
    ctx->directlySendMainSteramSqe = true;
    ctx->msgPosForKernel = 0;

    msg.opId.index = ctx->opIndex + 1;
    AicpuUpdatComContextMumber(offsetof(AicpuComContext, opIndex), msg.opId.index);
    if (ctx->endStopLaunch) {
        HCCL_WARNING("the op should not be launched in suspending status");
        return HCCL_E_SUSPENDING;
    }
    CHK_RET(AicpuHdcUtils::InitOpExecStatus(ctx->kfcStatusTransferD2H, msg.opId));
    AicpuUpdatComContextMumber(offsetof(AicpuComContext, isOpLaunch), true);
    while (true) {
        HCCL_INFO("start to read the [%u] msg", ctx->msgPosForKernel);
        if (!rpc.ReadAddrMsg(&hcclMsg, ctx->msgPosForKernel)) {
            HCCL_ERROR("fail to get addr msg, msgPos %u", ctx->msgPosForKernel);
            rpc.PrintAllHcclMsgArea();
            TaskOrchestrator::PrintTimeOutSqInfo(ctx, ctx->dfxExtendInfo.dfxTimeOutConfig.sqeWaitTimeOut);
            return HCCL_E_TIMEOUT;
        }
        GetCommonHcclMsg(&hcclMsg, &commonHcclMsg, tilingBase);
        // 处理finalize消息
        if (commonHcclMsg.commType == HcclCMDType::HCCL_CMD_FINALIZE) {
            AicpuKfcProf::GetProInst(*ctx).receiveFinalizeTime = GetCurCpuTimestamp(true);
            if (ctx->debugMode == MC2_DEBUG_PRINT_BUFF) {
                rpc.PrintAllHcclMsgAreaData();
            }
            break;
        } else if (commonHcclMsg.commType == HcclCMDType::HCCL_CMD_INIT) {
            continue;
        } else if (
            commonHcclMsg.commType == HcclCMDType::HCCL_CMD_INTER_GROUP_SYNC
            || commonHcclMsg.commType == HcclCMDType::HCCL_CMD_BARRIER) {
            HCCL_ERROR("Msg %u is not supported.", static_cast<uint32_t>(commonHcclMsg.commType));
            return HCCL_E_PARA;
        } else if (commonHcclMsg.commType == HcclCMDType::HCCL_CMD_BATCH_WRITE) {
            // 校验多机场景，multiServerFlag必须为true
            if (!ctx->multiServerFlag) {
                HCCL_ERROR("Batch write is only support in multi server.");
                return HCCL_E_PARA;
            }
            // 处理BatchWrite的操作从直接发送->队列发送。
            CHK_RET(AicpuKfcBatchwriteProcess::HandleBatchWriteOperation(commonHcclMsg, ctx));
            // 刷一下标记内存 commitTUrnCnt=0, finsihTurnCnt++
            rpc.WriteTurnCnt(ctx->msgPosForKernel);
        } else {
            rpc.HcclMsg2AicAicpuOpParam(&commonHcclMsg, &msg);
            if (msg.sendBuffer == 0UL || msg.recvBuffer == 0UL) {
                HCCL_ERROR("Get msg buffer is nullptr.");
                msg.PrintMsg("Invalid msg buffer");
                rpc.PrintAllHcclMsgArea();
                return HCCL_E_PARA;
            }
            CHK_RET(SetMsgWinOffset(ctx, &msg));
            CHK_RET(AicpuKfcProcess::AddTaskForHcclMsg(ctx, rpc, &commonHcclMsg, &msg, tilingBase));
        }
        // 切换到下一个msg
        ctx->msgPosForKernel = (ctx->msgPosForKernel + 1) % HCCL_MSG_CNT;
    }
    // 添加结束任务
    if (!ctx->multiServerFlag) {
        CHK_RET(AicpuDispatcher::AddAllEndTaskOnMainStream(AicpuKfcProcess::GetActiveSqId(ctx)));
        TaskOrchestrator::ActiveRecordMain(AicpuKfcProcess::GetActiveSqId(ctx));
        ctx->directlySendMainSteramSqe = false;
        CHK_RET(AicpuKfcProcess::WaitTaskFinish(ctx));
    } else {
        AicpuKfcBatchwriteProcess::FinishProcess();
    }
    rpc.WriteFinishWhenAllFinalize(ctx->msgPosForKernel);
    AicpuUpdatComContextMumber(offsetof(AicpuComContext, dfxExtendInfo.kfcStatus), DfxKfcStatus::kOneFinished);
    return HCCL_SUCCESS;
}

HcclResult AicpuKfcProcess::AicpuRunRpcServerForApi(AicpuComContext* ctx, u64 tilingBase)
{
    static AicpuKfcRpcServer rpc;
    rpc.Init(ctx->workSpaceAddr);
    AicpuKfcProf::GetProInst(*ctx).commInitEndTime = GetCurCpuTimestamp(true);
    const HcclResult ret = RunRpcServerApi(ctx, rpc, tilingBase);
    AicpuUpdatComContextMumber(offsetof(AicpuComContext, isOpLaunch), false);
    if (ret != HCCL_SUCCESS) {
        return DealReturnValue(ctx, ret);
    } else {
        CHK_RET(AicpuHdcUtils::SetOpExecStatus(ctx->kfcStatusTransferD2H, KfcStatus::kEnd, KfcError::kNone, 0));
        return ret;
    }
}

u32 AicpuKfcProcess::AicpuRunRpcServerForMC2V2(KFCTaskV2* task, const Mc2InitTilingInner* tilingData)
{
    static std::atomic<bool> initFlag(false);
    if (HcclAicpuUtils::GetBlockNum() <= 1U || !initFlag.exchange(true)) {
        for (u64 i = 0UL; i < task->ctxNum; i++) {
            HcclOpResParam* ctx = reinterpret_cast<HcclOpResParam*>(task->context[i]);
            HcclAicpuUtils::PrintHcclOpResParam(ctx);
            CHK_PRT_RET(
                PrepareHcommInstance(ctx, tilingData) != HCCL_SUCCESS,
                AicpuHcclProcess::AicpuReleaseCommbyGroup(ctx->hcomId), HCCL_E_INTERNAL);
        }
    }
    CHK_PRT_RET(
        AicpuKfcUtils::ThreadBarrier(BARRIER_TIMEOUT) != HCCL_SUCCESS,
        HCCL_ERROR("[%s]Timeout during instance preparation.", __func__), HCCL_E_INTERNAL);

    std::vector<u32> groupIds{};
    for (u64 i = 0UL; i < task->ctxNum; i++) {
        HcclOpResParam* ctx = reinterpret_cast<HcclOpResParam*>(task->context[i]);
        groupIds.emplace_back(GetComGroupIdx(ctx->hcomId));
    }
    HcclResult ret = RunRpcServerInnerProcessV2(groupIds);
    CHK_PRT_RET(
        AicpuKfcUtils::ThreadBarrier(BARRIER_TIMEOUT) != HCCL_SUCCESS,
        HCCL_ERROR("[%s]Timeout during instance finalize.", __func__), HCCL_E_INTERNAL);

    if (HcclAicpuUtils::GetBlockIdx() == 0U) {
        for (u64 i = 0UL; i < task->ctxNum; i++) {
            HcclOpResParam* ctx = reinterpret_cast<HcclOpResParam*>(task->context[i]);
            AicpuHcclProcess::AicpuReleaseCommbyGroup(ctx->hcomId);
        }
        initFlag = false;
        if (CheckNsStopLaunchStatus(groupIds) == HCCL_E_SUSPENDING) {
            SetExpectPrepareId(0U, 0U);
            HCCL_INFO("mc2 opp is suspended");
            return AICPUSUSPENDING_ERROR;
        }
    }
    return ret;
}

u32 AicpuKfcProcess::AicpuRunRpcServerForMC2(KFCTaskV2* task)
{
    HcclOpResParam* commParam[MAX_COMM_CTX_NUM]{};
    std::vector<u32> groupIds{};
    for (int i = 0; i < static_cast<int>(task->ctxNum); i++) {
        commParam[i] = reinterpret_cast<HcclOpResParam*>(task->context[i]);
        CHK_RET(static_cast<HcclResult>(PrepareHcommInstance(commParam[i])));
        groupIds.emplace_back(GetComGroupIdx(commParam[i]->hcomId));
    }
    HcclResult ret = RunRpcServerApiV2(reinterpret_cast<void*>(task->tilingData), groupIds);
    for (int i = 0; i < static_cast<int>(task->ctxNum); i++) {
        std::string group = commParam[i]->hcomId;
        AicpuHcclProcess::AicpuReleaseCommbyGroup(group);
    }
    if (CheckNsStopLaunchStatus(groupIds) == HCCL_E_SUSPENDING) {
        SetExpectPrepareId(0U, 0U);
        HCCL_INFO("mc2 opp is suspended");
        return AICPUSUSPENDING_ERROR;
    }
    return ret;
}

HcclResult
AicpuKfcProcess::AicpuCcOpExe(AivAicpuOpParam* commParam, AivAicpuOpParam* commParamNext, AicpuComContext* ctx)
{
    HCCL_DEBUG("----------start %s -------", __func__);
    if (commParam == nullptr || ctx == nullptr) {
        HCCL_ERROR("%s commParam or ctx is null.", __func__);
        return HCCL_E_PARA;
    }

    // 1. process global resource, update context.
    ctx->unitSize = DataUnitSize(commParam->hcclDataType);
    CHK_PRT_RET(ctx->unitSize == 0, HCCL_ERROR("[%s]ctx->unitSize is zero.", __func__), HCCL_E_PARA);
    ctx->commLen = ctx->unitSize * commParam->count;
    ctx->commType = commParam->commType;
    ctx->reducekind = commParam->opType;
    ctx->commOpType = GetCcOpType(ctx->commLen, ctx->rankNum); // twoshot.oneshot...
    ctx->totalTurnCnt = commParam->totalTurnCnt;
    ctx->useBufferType = commParam->useBufferType;
    ctx->winOffset = commParam->winOffset;

    auto profInst = AicpuKfcProf::GetProInst(*ctx);
    if (AicpuKfcUtils::NeedRecordTimeTaken(*ctx)) {
        u32 index = profInst.workCnt;
        index = (index >= AC_MAX_PROF_COMM_CNT) ? (AC_MAX_PROF_COMM_CNT - 1) : index;
        profInst.commLoop[index].dataLen = ctx->commLen;
    }

    HcclResult result = TaskOrchestrator::RunConcreteAlgorithm(commParam, commParamNext, ctx);
    if (result != HCCL_SUCCESS) {
        HCCL_ERROR("Run comm alg failed, rankId:%d, result:%u.", ctx->rankId, result);
        return result;
    }
    profInst.workCnt = ctx->curTurnCnt;
    // 所有轮次执行完毕后通知aclnn
    if (ctx->curTurnCnt == ctx->totalTurnCnt
        && (ctx->devType != DevType::DEV_TYPE_310P1 && ctx->devType != DevType::DEV_TYPE_310P3)
        && ctx->preparePosition != TASK_PREPARE_KERNEL) {
        CHK_RET(AicpuDispatcher::AddAllEndTaskOnMainStream(AicpuKfcProcess::GetActiveSqId(ctx)));
    }

    return HCCL_SUCCESS;
}

HcclResult AicpuKfcProcess::WaitTaskFinish(AicpuComContext* ctx, bool isWaitTask)
{
    HcclResult ret = HCCL_SUCCESS;
    CHK_RET(AicpuKfcUtils::TraceProfSubmit());
    if (isWaitTask || ctx->retryEnable) {
        ret = TaskOrchestrator::WaitMainStreamFinish(ctx);
        CHK_PRT_RET(
            (ret != HCCL_SUCCESS && ret != HCCL_E_SUSPENDING), HCCL_ERROR("wait main stream finish failed"), ret);
    }
    return ret;
}

HcclResult AicpuKfcProcess::ResetSqBuff(AicpuComContext* ctx)
{
    CHK_RET(AicpuSqeContext::ClearLocalBuff());
    SqeContext* sqeContext = GetSqeContext();
    u32 streamNum = (ctx->multiServerFlag) ? 1 : ctx->rankNum;
    for (u32 i = 0; i < streamNum; i++) {
        auto& buff = sqeContext->buffPtr[i];
        CHK_RET(QuerySqStatusByType(ctx->devId, ctx->streamInfo[i].sqId, DRV_SQCQ_PROP_SQ_TAIL, buff.sqTail));
        CHK_RET(QuerySqStatusByType(ctx->devId, ctx->streamInfo[i].sqId, DRV_SQCQ_PROP_SQ_HEAD, buff.sqHead));
        HCCL_INFO(
            "hccl aicpu reset stream buffer, sqid:%d head:%u tail:%u.", ctx->streamInfo[i].sqId, buff.sqHead,
            buff.sqTail);
    }
    HCCL_INFO("reset stream sq buffer success.");
    return HCCL_SUCCESS;
}

u32 AicpuKfcProcess::GetActiveSqId(AicpuComContext* ctx) { return ctx->rankId; }

HcclResult AicpuKfcProcess::InitStreamInfo(HccCommResParamTask* commParam, AicpuComContext* ctx)
{
    g_streamIdMap.clear();
    u32 streamNum = (ctx->multiServerFlag) ? 1U : ctx->rankNum;
    for (u32 i = 0; i < streamNum; i++) {
        auto& streamInfo = ctx->streamInfo[i];
        streamInfo.sqId = commParam->streamInfo[i].sqIds;
        streamInfo.logicCqId = commParam->streamInfo[i].logicCqids;
        streamInfo.actualStreamId = commParam->streamInfo[i].streamIds;
        HCCL_INFO("streamInfo.sqId :%d, streamId:%d", streamInfo.sqId, streamInfo.actualStreamId);
        u64 sq_addr = 0;
        CHK_RET(QuerySqBaseAddr(ctx->devId, streamInfo.sqId, sq_addr));
        streamInfo.sqBaseAddr = reinterpret_cast<void*>(sq_addr);
        CHK_RET(QuerySqStatusByType(ctx->devId, streamInfo.sqId, DRV_SQCQ_PROP_SQ_DEPTH, streamInfo.sqDepth));
        g_streamIdMap[streamInfo.actualStreamId] = i;
    }
    CHK_RET(AicpuKfcProcess::ResetSqBuff(ctx));
    return HCCL_SUCCESS;
}
