/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <adapter_error_manager_pub.h>
#include "ccu_error_handler.h"
#include "ccu_context_mgr_imp.h"
#include "orion_adapter_hccp.h"
#include "orion_adapter_rts.h"
#include "hccp_tlv_hdc_manager.h"
#include "hccl_log_keywords.h"

namespace Hccl {
using namespace std;
using namespace CcuRep;
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

void CcuErrorHandler::GetCcuErrorMsg(
    int32_t deviceId, uint16_t missionStatus, const ParaCcu& ccuTaskParam, const std::string& groupRankContent,
    std::vector<CcuErrorInfo>& errorInfo)
{
    const auto missionContext = GetCcuMissionContext(deviceId, ccuTaskParam.dieId, ccuTaskParam.execMissionId);
    if (missionStatus == 0) {
        HCCL_INFO(
            "[CcuErrorHandler][%s] no err found, mission status is 0, deviceId[%d], dieId[%u], execMissionId[%u]",
            __func__, deviceId, static_cast<u32>(ccuTaskParam.dieId), static_cast<u32>(ccuTaskParam.execMissionId));
        return;
    }

    CcuRepContext* ctx
        = CtxMgrImp::GetInstance(deviceId).GetCtx(ccuTaskParam.executeId, ccuTaskParam.dieId, ccuTaskParam.missionId);
    if (ctx == nullptr) {
        THROW<CcuApiException>(
            "CcuContext not found, deviceId[%d], dieId[%u], missionId[%u], executeId[%llu]", deviceId,
            static_cast<u32>(ccuTaskParam.dieId), static_cast<u32>(ccuTaskParam.missionId), ccuTaskParam.executeId);
    }

    const uint16_t currIns = missionContext.GetCurrentIns();

    auto rep = ctx->GetRepByInstrId(currIns);
    auto prevRep = ctx->GetRepByInstrId(currIns - 1);
    if (rep == nullptr) {
        HCCL_WARNING("[CcuErrorHandler][%s] cannot find REP from current CcuContext, instrId[%u]", __func__, currIns);
        return;
    }

    // 分类处理Rep, 返回异常信息
    ErrorInfoBase baseInfo{deviceId, ccuTaskParam.dieId, ccuTaskParam.missionId, currIns, missionStatus};
    GenStatusInfo(baseInfo, groupRankContent, errorInfo);

    // 处理Rep为FUNC_BLOCK的场景
    while (rep->Type() == CcuRepType::FUNC_BLOCK) {
        auto blockRep = static_pointer_cast<CcuRepBlock>(rep);
        rep = blockRep->GetRepByInstrId(currIns);
        if (rep == nullptr) {
            THROW<CcuApiException>(
                "Failed to find REP from FuncBlock, instrId[%u], FuncBlock[%s]", currIns, blockRep->GetLabel().c_str());
        }
    }

    if ((prevRep != nullptr && prevRep->Type() == CcuRepType::LOOPGROUP) || (rep->Type() == CcuRepType::LOOPGROUP)) {
        // 处理LoopGroup
        GenErrorInfoLoopGroup(baseInfo, prevRep, *ctx, errorInfo);
    } else if (rep->Type() == CcuRepType::LOC_WAIT_SEM) {
        GenErrorInfoByRepType(baseInfo, rep, errorInfo);
        uint16_t actValue = errorInfo.back().msg.waitSignal.signalValue;
        uint16_t expValue = errorInfo.back().msg.waitSignal.signalMask;
        for (uint16_t i = 0; i < 16; ++i) { // CKE的bit数最多为16
            uint16_t mask = 1 << i;         // 创建一个用于检查第 i 位的掩码
            if ((expValue & mask) != 0 && (actValue & mask) == 0) {
                auto depRepVec = std::static_pointer_cast<CcuRepLocWaitSem>(rep)->GetDependencyInfo(mask);
                for (const auto& depRep : depRepVec) {
                    GenErrorInfoByRepType(baseInfo, depRep, errorInfo);
                }
            }
        }
    } else {
        // 处理可直接解析的Rep
        GenErrorInfoByRepType(baseInfo, rep, errorInfo);
    }

    const uint16_t endIns = missionContext.GetEndIns();
    const uint16_t startIns = missionContext.GetStartIns();
    // 获取异常指令对应的Rep
    HCCL_ERROR(
        "[CcuErrorHandler]device %d, execMissionId[%u], startIns[%u], endIns[%u], currIns[%u]", deviceId,
        ccuTaskParam.execMissionId, startIns, endIns, currIns);
    if (endIns == currIns) {
        HCCL_ERROR("[CcuErrorHandler]device %d SQE != CQE, endIns[%u], currIns[%u]", deviceId, endIns, currIns);
    }

    // 安全地获取currIns - 10的值
    uint16_t loopUpInstrNum = 10; // 获取出错指令前10条指令
    uint16_t beginIns = (currIns < loopUpInstrNum) ?
                            startIns :
                            ((currIns - loopUpInstrNum) > startIns ? (currIns - loopUpInstrNum) : startIns);
    // 打印报错的前10条指令，并且从第一个非空rep开始
    for (uint16_t instrId = currIns - 1; instrId >= beginIns; instrId--) {
        auto rep = ctx->GetRepByInstrId(instrId);
        if (rep == nullptr) {
            beginIns = instrId + 1;
            break;
        }
    }
    for (uint16_t instrId = beginIns; instrId <= currIns; instrId++) {
        auto rep = ctx->GetRepByInstrId(instrId);
        if (rep == nullptr) {
            HCCL_WARNING(
                "[CcuErrorHandler][%s] cannot find REP from current CcuContext, instrId[%u]", __func__, instrId);
            continue;
        }

        GenErrorInfoByRepType(baseInfo, rep, errorInfo);
    }
}

void CcuErrorHandler::GetCcuJettys(int32_t deviceId, const ParaCcu& ccuTaskParam, std::vector<CcuJetty*> ccuJettys)
{
    // 获取异常指令对应的Rep
    CcuContext* ctx
        = CtxMgrImp::GetInstance(deviceId).GetCtx(ccuTaskParam.executeId, ccuTaskParam.dieId, ccuTaskParam.missionId);
    if (ctx == nullptr) {
        THROW<CcuApiException>(
            "CcuContext not found, deviceId[%d], dieId[%u], missionId[%u], executeId[%llu]", deviceId,
            ccuTaskParam.dieId, ccuTaskParam.missionId, ccuTaskParam.executeId);
    }

    std::vector<CcuTransport*> ccuTransports = ctx->GetCcuTransports();
    for (auto ccuTransport : ccuTransports) {
        CcuConnection* ccuConn = ccuTransport->GetCcuConnection();
        std::vector<CcuJetty*> ccuJettysOneConn = ccuConn->GetCcuJettys();
        ccuJettys.insert(ccuJettys.end(), ccuJettysOneConn.begin(), ccuJettysOneConn.end());
    }
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

void CcuErrorHandler::GenStatusInfo(
    const ErrorInfoBase& baseInfo, const std::string& groupRankContent, vector<CcuErrorInfo>& errorInfo)
{
    CcuErrorInfo errorMsg{};
    errorMsg.type = CcuErrorType::MISSION;
    errorMsg.SetBaseInfo(CcuRepType::BASE, baseInfo.dieId, baseInfo.missionId, baseInfo.currentInsId);
    const auto baseInformation = Hccl::StringFormat("dieId[%u], missionId[%u]", baseInfo.dieId, baseInfo.missionId);
    const auto taskInformation
        = Hccl::StringFormat("currentInsId[%u], status[%u]", baseInfo.currentInsId, baseInfo.status);
    const uint8_t highPart = (baseInfo.status >> 8) & 0xFF; // 高8位
    const uint8_t lowPart = baseInfo.status & 0xFF;         // 低8位
    if (highPart == CCUM_EXECUTE_ERROR && lowPart == CCU_MISSION_TASK_KILLED) {
        RPT_INPUT_ERR(
            true, "EI0002",
            std::vector<std::string>({"remote_rankid", "base_information", "task_information", "group_rank_content"}),
            std::vector<std::string>(
                {std::to_string(baseInfo.deviceId), baseInformation.c_str(), taskInformation.c_str(),
                 groupRankContent.c_str()}));
        HCCL_ERROR(
            "[%s][%s][%s] Task run failed, EI0002, deviceID:[%u], %s, %s.", LOG_KEYWORDS_TASK_EXEC.c_str(),
            LOG_KEYWORDS_TIMEOUT.c_str(), LOG_KEYWORDS_CCU.c_str(), baseInfo.deviceId, baseInformation.c_str(),
            taskInformation.c_str());
    }
    const string statusMsg = StatusCode2Str(highPart, lowPart);
    const auto sRet
        = strncpy_s(errorMsg.msg.mission.missionError, MISSION_STATUS_MSG_LEN, statusMsg.c_str(), statusMsg.length());
    if (sRet != EOK) {
        HCCL_ERROR("[CcuErrorHandler][%s] strcpy failed, statusMsg: %s.", __func__, statusMsg.c_str());
    }

    errorInfo.push_back(errorMsg);
}

void CcuErrorHandler::GenErrorInfoLoopGroup(
    const ErrorInfoBase& baseInfo, shared_ptr<CcuRepBase> repBase, CcuRepContext& ctx, vector<CcuErrorInfo>& errorInfo)
{
    CcuErrorInfo errorMsg{};
    errorMsg.type = CcuErrorType::LOOP_GROUP;
    errorMsg.SetBaseInfo(repBase->Type(), baseInfo.dieId, baseInfo.missionId, repBase->StartInstrId());

    const auto rep = static_pointer_cast<CcuRepLoopGroup>(repBase);
    const auto startLoopInstrId = rep->GetStartLoopInstrId();
    LoopGroupXn loopGroupXn{};
    loopGroupXn.value = GetCcuXnValue(baseInfo.deviceId, baseInfo.dieId, rep->parallelParam.Id());
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
        GenErrorInfoLoop(loopErrInfoBase, ctx, errorInfo);
    }
}

void CcuErrorHandler::GenErrorInfoLoop(
    const ErrorInfoBase& baseInfo, CcuRepContext& ctx, vector<CcuErrorInfo>& errorInfo)
{
    // 找LoopRep
    auto repBase = ctx.GetRepByInstrId(baseInfo.currentInsId);
    if (repBase == nullptr || repBase->Type() != CcuRepType::LOOP) {
        THROW<CcuApiException>("Failed to find Loop REP from CcuContext, instrId[%u]", baseInfo.currentInsId);
    }
    const auto rep = static_pointer_cast<CcuRepLoop>(repBase);

    CcuErrorInfo errorMsg{};
    errorMsg.type = CcuErrorType::LOOP;
    errorMsg.SetBaseInfo(repBase->Type(), baseInfo.dieId, baseInfo.missionId, baseInfo.currentInsId);

    LoopXm loopXm{};
    loopXm.value = GetCcuXnValue(baseInfo.deviceId, baseInfo.dieId, rep->loopParam.Id());
    const auto ccuLoopContext = GetCcuLoopContext(baseInfo.deviceId, baseInfo.dieId, loopXm.loopCtxId);
    errorMsg.msg.loop.startInstrId = rep->loopBlock->StartInstrId();
    errorMsg.msg.loop.endInstrId = rep->loopBlock->StartInstrId() + rep->loopBlock->InstrCount() - 1;
    errorMsg.msg.loop.loopEngineId = loopXm.loopCtxId;
    errorMsg.msg.loop.loopCnt = static_cast<uint16_t>(loopXm.loopCnt);
    errorMsg.msg.loop.loopCurrentCnt = ccuLoopContext.GetCurrentCnt();
    errorMsg.msg.loop.addrStride = ccuLoopContext.GetAddrStride();

    errorInfo.push_back(errorMsg);

    // 解析Loop内的异常Rep
    for (uint16_t loopCurrentIns = errorMsg.msg.loop.startInstrId; loopCurrentIns <= errorMsg.msg.loop.endInstrId;
         loopCurrentIns++) {
        auto inLoopExRep = rep->loopBlock->GetRepByInstrId(loopCurrentIns);
        if (inLoopExRep == nullptr) {
            THROW<CcuApiException>(
                "Failed to find REP from Loop, instrId[%u], Loop[%s]", loopCurrentIns, rep->GetLabel().c_str());
        }
        ErrorInfoBase loopErrBase{
            baseInfo.deviceId, baseInfo.dieId, baseInfo.missionId, loopCurrentIns, baseInfo.status};
        GenErrorInfoByRepType(loopErrBase, inLoopExRep, errorInfo);
    }
}

void CcuErrorHandler::GenErrorInfoByRepType(
    const ErrorInfoBase& baseInfo, shared_ptr<CcuRepBase> repBase, vector<CcuErrorInfo>& errorInfo)
{
    using GenErrorInfoFunc
        = void (*)(const ErrorInfoBase& baseInfo, shared_ptr<CcuRepBase> repBase, vector<CcuErrorInfo>& errorInfo);
    static const map<CcuRepType, GenErrorInfoFunc> handlerMap{
        // WAIT_SIGNAL
        {CcuRepType::LOC_POST_SEM, &CcuErrorHandler::GenErrorInfoLocPostSem},
        {CcuRepType::LOC_WAIT_SEM, &CcuErrorHandler::GenErrorInfoLocWaitSem},
        {CcuRepType::REM_POST_SEM, &CcuErrorHandler::GenErrorInfoRemPostSem},
        {CcuRepType::REM_WAIT_SEM, &CcuErrorHandler::GenErrorInfoRemWaitSem},
        {CcuRepType::REM_POST_VAR, &CcuErrorHandler::GenErrorInfoRemPostVar},
        {CcuRepType::REM_WAIT_GROUP, &CcuErrorHandler::GenErrorInfoRemWaitGroup},
        {CcuRepType::POST_SHARED_VAR, &CcuErrorHandler::GenErrorInfoPostSharedVar},
        {CcuRepType::POST_SHARED_SEM, &CcuErrorHandler::GenErrorInfoPostSharedSem},
        // TRANS_MEM
        {CcuRepType::READ, &CcuErrorHandler::GenErrorInfoRead},
        {CcuRepType::WRITE, &CcuErrorHandler::GenErrorInfoWrite},
        {CcuRepType::LOCAL_CPY, &CcuErrorHandler::GenErrorInfoLocalCpy},
        {CcuRepType::LOCAL_REDUCE, &CcuErrorHandler::GenErrorInfoLocalReduce},
        // BUF_TRANS_MEM
        {CcuRepType::BUF_READ, &CcuErrorHandler::GenErrorInfoBufRead},
        {CcuRepType::BUF_WRITE, &CcuErrorHandler::GenErrorInfoBufWrite},
        {CcuRepType::BUF_LOC_READ, &CcuErrorHandler::GenErrorInfoBufLocRead},
        {CcuRepType::BUF_LOC_WRITE, &CcuErrorHandler::GenErrorInfoBufLocWrite},
        // BUF_REDUCE
        {CcuRepType::BUF_REDUCE, &CcuErrorHandler::GenErrorInfoBufReduce}};
    const auto funcIt = handlerMap.find(repBase->Type());
    if (funcIt == handlerMap.end()) {
        // DEFAULT, chip error
        GenErrorInfoDefault(baseInfo, repBase, errorInfo);
    } else {
        (funcIt->second)(baseInfo, repBase, errorInfo);
    }
}

void CcuErrorHandler::GenErrorInfoDefault(
    const ErrorInfoBase& baseInfo, shared_ptr<CcuRepBase> repBase, vector<CcuErrorInfo>& errorInfo)
{
    CcuErrorInfo errorMsg{};
    errorMsg.type = CcuErrorType::DEFAULT;
    errorMsg.SetBaseInfo(repBase->Type(), baseInfo.dieId, baseInfo.missionId, repBase->StartInstrId());
    errorInfo.push_back(errorMsg);
}

void CcuErrorHandler::GenErrorInfoLocPostSem(
    const ErrorInfoBase& baseInfo, shared_ptr<CcuRepBase> repBase, vector<CcuErrorInfo>& errorInfo)
{
    CcuErrorInfo errorMsg{};
    errorMsg.type = CcuErrorType::WAIT_SIGNAL;
    errorMsg.SetBaseInfo(repBase->Type(), baseInfo.dieId, baseInfo.missionId, repBase->StartInstrId());

    const auto rep = static_pointer_cast<CcuRepLocPostSem>(repBase);
    errorMsg.msg.waitSignal.signalId = rep->sem.Id();
    errorMsg.msg.waitSignal.signalValue = GetCcuCKEValue(baseInfo.deviceId, baseInfo.dieId, rep->sem.Id());
    errorMsg.msg.waitSignal.signalMask = rep->mask;

    errorInfo.push_back(errorMsg);
}

void CcuErrorHandler::GenErrorInfoLocWaitSem(
    const ErrorInfoBase& baseInfo, shared_ptr<CcuRepBase> repBase, vector<CcuErrorInfo>& errorInfo)
{
    CcuErrorInfo errorMsg{};
    errorMsg.type = CcuErrorType::WAIT_SIGNAL;
    errorMsg.SetBaseInfo(repBase->Type(), baseInfo.dieId, baseInfo.missionId, repBase->StartInstrId());

    const auto rep = static_pointer_cast<CcuRepLocWaitSem>(repBase);
    errorMsg.msg.waitSignal.signalId = rep->sem.Id();
    errorMsg.msg.waitSignal.signalValue = GetCcuCKEValue(baseInfo.deviceId, baseInfo.dieId, rep->sem.Id());
    errorMsg.msg.waitSignal.signalMask = rep->mask;

    errorInfo.push_back(errorMsg);
}

void CcuErrorHandler::GenErrorInfoRemPostSem(
    const ErrorInfoBase& baseInfo, shared_ptr<CcuRepBase> repBase, vector<CcuErrorInfo>& errorInfo)
{
    CcuErrorInfo errorMsg{};
    errorMsg.type = CcuErrorType::WAIT_SIGNAL;
    errorMsg.SetBaseInfo(repBase->Type(), baseInfo.dieId, baseInfo.missionId, repBase->StartInstrId());

    const auto rep = static_pointer_cast<CcuRepRemPostSem>(repBase);
    errorMsg.msg.waitSignal.signalId = rep->transport.GetRmtCntCkeByIndex(rep->semIndex);
    errorMsg.msg.waitSignal.signalMask = rep->mask;
    (void)memset_s(
        errorMsg.msg.waitSignal.channelId, sizeof(errorMsg.msg.waitSignal.channelId), 0xFF,
        sizeof(errorMsg.msg.waitSignal.channelId));
    errorMsg.msg.waitSignal.channelId[0] = rep->transport.GetChannelId();

    errorInfo.push_back(errorMsg);
}

void CcuErrorHandler::GenErrorInfoRemWaitSem(
    const ErrorInfoBase& baseInfo, shared_ptr<CcuRepBase> repBase, vector<CcuErrorInfo>& errorInfo)
{
    CcuErrorInfo errorMsg{};
    errorMsg.type = CcuErrorType::WAIT_SIGNAL;
    errorMsg.SetBaseInfo(repBase->Type(), baseInfo.dieId, baseInfo.missionId, repBase->StartInstrId());

    const auto rep = static_pointer_cast<CcuRepRemWaitSem>(repBase);
    errorMsg.msg.waitSignal.signalId = rep->transport.GetLocCntCkeByIndex(rep->semIndex);
    errorMsg.msg.waitSignal.signalValue
        = GetCcuCKEValue(baseInfo.deviceId, baseInfo.dieId, errorMsg.msg.waitSignal.signalId);
    errorMsg.msg.waitSignal.signalMask = rep->mask;
    (void)memset_s(
        errorMsg.msg.waitSignal.channelId, sizeof(errorMsg.msg.waitSignal.channelId), 0xFF,
        sizeof(errorMsg.msg.waitSignal.channelId));
    errorMsg.msg.waitSignal.channelId[0] = rep->transport.GetChannelId();

    errorInfo.push_back(errorMsg);
}

void CcuErrorHandler::GenErrorInfoRemPostVar(
    const ErrorInfoBase& baseInfo, shared_ptr<CcuRepBase> repBase, vector<CcuErrorInfo>& errorInfo)
{
    CcuErrorInfo errorMsg{};
    errorMsg.type = CcuErrorType::WAIT_SIGNAL;
    errorMsg.SetBaseInfo(repBase->Type(), baseInfo.dieId, baseInfo.missionId, repBase->StartInstrId());

    const auto rep = static_pointer_cast<CcuRepRemPostVar>(repBase);
    errorMsg.msg.waitSignal.signalId = rep->transport.GetRmtCntCkeByIndex(rep->semIndex);
    errorMsg.msg.waitSignal.signalMask = rep->mask;
    (void)memset_s(
        errorMsg.msg.waitSignal.channelId, sizeof(errorMsg.msg.waitSignal.channelId), 0xFF,
        sizeof(errorMsg.msg.waitSignal.channelId));
    errorMsg.msg.waitSignal.channelId[0] = rep->transport.GetChannelId();
    errorMsg.msg.waitSignal.paramId = rep->transport.GetRmtXnByIndex(rep->paramIndex);
    errorMsg.msg.waitSignal.paramValue = GetCcuXnValue(baseInfo.deviceId, baseInfo.dieId, rep->param.Id());

    errorInfo.push_back(errorMsg);
}

void CcuErrorHandler::GenErrorInfoRemWaitGroup(
    const ErrorInfoBase& baseInfo, shared_ptr<CcuRepBase> repBase, vector<CcuErrorInfo>& errorInfo)
{
    CcuErrorInfo errorMsg{};
    errorMsg.type = CcuErrorType::WAIT_SIGNAL;
    errorMsg.SetBaseInfo(repBase->Type(), baseInfo.dieId, baseInfo.missionId, repBase->StartInstrId());

    const auto rep = static_pointer_cast<CcuRepWaitGroup>(repBase);
    u32 cntCkeId = 0;
    HcclResult ret = rep->transportGroup.GetCntCkeId(rep->semIndex, cntCkeId);
    if (ret != HcclResult::HCCL_SUCCESS) {
        string msg = StringFormat(
            "[GenErrorInfoRemWaitGroup]rt get CntCkeId failed. "
            "semIndex[%u], cntCkeId[%u] return[%d].",
            rep->semIndex, cntCkeId, ret);
        MACRO_THROW(CcuApiException, msg);
    }
    errorMsg.msg.waitSignal.signalId = cntCkeId;
    errorMsg.msg.waitSignal.signalValue
        = GetCcuCKEValue(baseInfo.deviceId, baseInfo.dieId, errorMsg.msg.waitSignal.signalId);
    errorMsg.msg.waitSignal.signalMask = rep->mask;
    const vector<CcuTransport*>& transports = rep->transportGroup.GetTransports();
    (void)memset_s(
        errorMsg.msg.waitSignal.channelId, sizeof(errorMsg.msg.waitSignal.channelId), 0xFF,
        sizeof(errorMsg.msg.waitSignal.channelId));
    for (uint32_t i = 0; i < transports.size() && i < WAIT_SIGNAL_CHANNEL_SIZE; ++i) {
        errorMsg.msg.waitSignal.channelId[i] = transports[i]->GetChannelId();
    }

    errorInfo.push_back(errorMsg);
}

void CcuErrorHandler::GenErrorInfoPostSharedVar(
    const ErrorInfoBase& baseInfo, shared_ptr<CcuRepBase> repBase, vector<CcuErrorInfo>& errorInfo)
{
    CcuErrorInfo errorMsg{};
    errorMsg.type = CcuErrorType::WAIT_SIGNAL;
    errorMsg.SetBaseInfo(repBase->Type(), baseInfo.dieId, baseInfo.missionId, repBase->StartInstrId());

    const auto rep = static_pointer_cast<CcuRepPostSharedVar>(repBase);
    errorMsg.msg.waitSignal.signalId = rep->sem.Id();
    errorMsg.msg.waitSignal.signalMask = rep->mask;
    errorMsg.msg.waitSignal.paramId = rep->dstVar.Id();
    errorMsg.msg.waitSignal.paramValue = GetCcuXnValue(baseInfo.deviceId, baseInfo.dieId, rep->srcVar.Id());

    errorInfo.push_back(errorMsg);
}

void CcuErrorHandler::GenErrorInfoPostSharedSem(
    const ErrorInfoBase& baseInfo, shared_ptr<CcuRepBase> repBase, vector<CcuErrorInfo>& errorInfo)
{
    CcuErrorInfo errorMsg{};
    errorMsg.type = CcuErrorType::WAIT_SIGNAL;
    errorMsg.SetBaseInfo(repBase->Type(), baseInfo.dieId, baseInfo.missionId, repBase->StartInstrId());

    const auto rep = static_pointer_cast<CcuRepPostSharedSem>(repBase);
    errorMsg.msg.waitSignal.signalId = rep->sem.Id();
    errorMsg.msg.waitSignal.signalMask = rep->mask;

    errorInfo.push_back(errorMsg);
}

void CcuErrorHandler::GenErrorInfoRead(
    const ErrorInfoBase& baseInfo, shared_ptr<CcuRepBase> repBase, vector<CcuErrorInfo>& errorInfo)
{
    CcuErrorInfo errorMsg{};
    errorMsg.type = CcuErrorType::TRANS_MEM;
    errorMsg.SetBaseInfo(repBase->Type(), baseInfo.dieId, baseInfo.missionId, repBase->StartInstrId());

    const auto rep = static_pointer_cast<CcuRepRead>(repBase);
    errorMsg.msg.transMem.locAddr = GetCcuGSAValue(baseInfo.deviceId, baseInfo.dieId, rep->loc.addr.Id());
    errorMsg.msg.transMem.locToken = GetCcuXnValue(baseInfo.deviceId, baseInfo.dieId, rep->loc.token.Id());
    errorMsg.msg.transMem.rmtAddr = GetCcuGSAValue(baseInfo.deviceId, baseInfo.dieId, rep->rem.addr.Id());
    errorMsg.msg.transMem.rmtToken = GetCcuXnValue(baseInfo.deviceId, baseInfo.dieId, rep->rem.token.Id());
    errorMsg.msg.transMem.len = GetCcuXnValue(baseInfo.deviceId, baseInfo.dieId, rep->len.Id());
    errorMsg.msg.transMem.signalId = rep->sem.Id();
    errorMsg.msg.transMem.signalMask = rep->mask;
    errorMsg.msg.transMem.channelId = rep->transport.GetChannelId();

    errorInfo.push_back(errorMsg);
}

void CcuErrorHandler::GenErrorInfoWrite(
    const ErrorInfoBase& baseInfo, shared_ptr<CcuRepBase> repBase, vector<CcuErrorInfo>& errorInfo)
{
    CcuErrorInfo errorMsg{};
    errorMsg.type = CcuErrorType::TRANS_MEM;
    errorMsg.SetBaseInfo(repBase->Type(), baseInfo.dieId, baseInfo.missionId, repBase->StartInstrId());

    const auto rep = static_pointer_cast<CcuRepWrite>(repBase);
    errorMsg.msg.transMem.locAddr = GetCcuGSAValue(baseInfo.deviceId, baseInfo.dieId, rep->loc.addr.Id());
    errorMsg.msg.transMem.locToken = GetCcuXnValue(baseInfo.deviceId, baseInfo.dieId, rep->loc.token.Id());
    errorMsg.msg.transMem.rmtAddr = GetCcuGSAValue(baseInfo.deviceId, baseInfo.dieId, rep->rem.addr.Id());
    errorMsg.msg.transMem.rmtToken = GetCcuXnValue(baseInfo.deviceId, baseInfo.dieId, rep->rem.token.Id());
    errorMsg.msg.transMem.len = GetCcuXnValue(baseInfo.deviceId, baseInfo.dieId, rep->len.Id());
    errorMsg.msg.transMem.signalId = rep->sem.Id();
    errorMsg.msg.transMem.signalMask = rep->mask;
    errorMsg.msg.transMem.channelId = rep->transport.GetChannelId();

    errorInfo.push_back(errorMsg);
}

void CcuErrorHandler::GenErrorInfoLocalCpy(
    const ErrorInfoBase& baseInfo, shared_ptr<CcuRepBase> repBase, vector<CcuErrorInfo>& errorInfo)
{
    CcuErrorInfo errorMsg{};
    errorMsg.type = CcuErrorType::TRANS_MEM;
    errorMsg.SetBaseInfo(repBase->Type(), baseInfo.dieId, baseInfo.missionId, repBase->StartInstrId());

    const auto rep = static_pointer_cast<CcuRepLocCpy>(repBase);
    errorMsg.msg.transMem.locAddr = GetCcuGSAValue(baseInfo.deviceId, baseInfo.dieId, rep->src.addr.Id());
    errorMsg.msg.transMem.locToken = GetCcuXnValue(baseInfo.deviceId, baseInfo.dieId, rep->src.token.Id());
    errorMsg.msg.transMem.rmtAddr = GetCcuGSAValue(baseInfo.deviceId, baseInfo.dieId, rep->dst.addr.Id());
    errorMsg.msg.transMem.rmtToken = GetCcuXnValue(baseInfo.deviceId, baseInfo.dieId, rep->dst.token.Id());
    errorMsg.msg.transMem.len = GetCcuXnValue(baseInfo.deviceId, baseInfo.dieId, rep->len.Id());
    errorMsg.msg.transMem.signalId = rep->sem.Id();
    errorMsg.msg.transMem.signalMask = rep->mask;

    errorInfo.push_back(errorMsg);
}

void CcuErrorHandler::GenErrorInfoLocalReduce(
    const ErrorInfoBase& baseInfo, shared_ptr<CcuRepBase> repBase, vector<CcuErrorInfo>& errorInfo)
{
    CcuErrorInfo errorMsg{};
    errorMsg.type = CcuErrorType::TRANS_MEM;
    errorMsg.SetBaseInfo(repBase->Type(), baseInfo.dieId, baseInfo.missionId, repBase->StartInstrId());

    const auto rep = static_pointer_cast<CcuRepLocCpy>(repBase);
    errorMsg.msg.transMem.locAddr = GetCcuGSAValue(baseInfo.deviceId, baseInfo.dieId, rep->src.addr.Id());
    errorMsg.msg.transMem.locToken = GetCcuXnValue(baseInfo.deviceId, baseInfo.dieId, rep->src.token.Id());
    errorMsg.msg.transMem.rmtAddr = GetCcuGSAValue(baseInfo.deviceId, baseInfo.dieId, rep->dst.addr.Id());
    errorMsg.msg.transMem.rmtToken = GetCcuXnValue(baseInfo.deviceId, baseInfo.dieId, rep->dst.token.Id());
    errorMsg.msg.transMem.len = GetCcuXnValue(baseInfo.deviceId, baseInfo.dieId, rep->len.Id());
    errorMsg.msg.transMem.signalId = rep->sem.Id();
    errorMsg.msg.transMem.signalMask = rep->mask;
    errorMsg.msg.transMem.opType = rep->opType;
    errorMsg.msg.transMem.dataType = rep->dataType;

    errorInfo.push_back(errorMsg);
}

/**
 * @brief Convert a unified (per-device) MSId into a per-die MSId.
 *
 * In the instruction format, MSId is encoded in a unified address space for
 * two dies. The most significant bit (bit 15) is used to encode the DieId,
 * and the remaining 15 bits represent the per-die MS/buffer identifier.
 *
 * When generating error information or human-readable output, the DieId
 * component must be removed so that the MSId is reported relative to the
 * current die only.
 */
inline uint16_t GetMSIdPerDie(uint16_t msId)
{
    // Mask off the DieId bit (bit 15, value 0x8000), keeping only the lower
    // 15 bits that encode the per-die MS/buffer identifier.
    return msId & 0x7fff;
}

void CcuErrorHandler::GenErrorInfoBufRead(
    const ErrorInfoBase& baseInfo, shared_ptr<CcuRepBase> repBase, vector<CcuErrorInfo>& errorInfo)
{
    CcuErrorInfo errorMsg{};
    errorMsg.type = CcuErrorType::BUF_TRANS_MEM;
    errorMsg.SetBaseInfo(repBase->Type(), baseInfo.dieId, baseInfo.missionId, repBase->StartInstrId());

    const auto rep = static_pointer_cast<CcuRepBufRead>(repBase);
    errorMsg.msg.bufTransMem.bufId = GetMSIdPerDie(rep->dst.Id());
    errorMsg.msg.bufTransMem.addr = GetCcuGSAValue(baseInfo.deviceId, baseInfo.dieId, rep->src.addr.Id());
    errorMsg.msg.bufTransMem.token = GetCcuXnValue(baseInfo.deviceId, baseInfo.dieId, rep->src.token.Id());
    errorMsg.msg.bufTransMem.len = GetCcuXnValue(baseInfo.deviceId, baseInfo.dieId, rep->len.Id());
    errorMsg.msg.bufTransMem.signalId = rep->sem.Id();
    errorMsg.msg.bufTransMem.signalMask = rep->mask;
    errorMsg.msg.bufTransMem.channelId = rep->transport.GetChannelId();

    errorInfo.push_back(errorMsg);
}

void CcuErrorHandler::GenErrorInfoBufWrite(
    const ErrorInfoBase& baseInfo, shared_ptr<CcuRepBase> repBase, vector<CcuErrorInfo>& errorInfo)
{
    CcuErrorInfo errorMsg{};
    errorMsg.type = CcuErrorType::BUF_TRANS_MEM;
    errorMsg.SetBaseInfo(repBase->Type(), baseInfo.dieId, baseInfo.missionId, repBase->StartInstrId());

    const auto rep = static_pointer_cast<CcuRepBufWrite>(repBase);
    errorMsg.msg.bufTransMem.bufId = GetMSIdPerDie(rep->src.Id());
    errorMsg.msg.bufTransMem.addr = GetCcuGSAValue(baseInfo.deviceId, baseInfo.dieId, rep->dst.addr.Id());
    errorMsg.msg.bufTransMem.token = GetCcuXnValue(baseInfo.deviceId, baseInfo.dieId, rep->dst.token.Id());
    errorMsg.msg.bufTransMem.len = GetCcuXnValue(baseInfo.deviceId, baseInfo.dieId, rep->len.Id());
    errorMsg.msg.bufTransMem.signalId = rep->sem.Id();
    errorMsg.msg.bufTransMem.signalMask = rep->mask;
    errorMsg.msg.bufTransMem.channelId = rep->transport.GetChannelId();

    errorInfo.push_back(errorMsg);
}

void CcuErrorHandler::GenErrorInfoBufLocRead(
    const ErrorInfoBase& baseInfo, shared_ptr<CcuRepBase> repBase, vector<CcuErrorInfo>& errorInfo)
{
    CcuErrorInfo errorMsg{};
    errorMsg.type = CcuErrorType::BUF_TRANS_MEM;
    errorMsg.SetBaseInfo(repBase->Type(), baseInfo.dieId, baseInfo.missionId, repBase->StartInstrId());

    const auto rep = static_pointer_cast<CcuRepBufLocRead>(repBase);
    errorMsg.msg.bufTransMem.bufId = GetMSIdPerDie(rep->dst.Id());
    errorMsg.msg.bufTransMem.addr = GetCcuGSAValue(baseInfo.deviceId, baseInfo.dieId, rep->src.addr.Id());
    errorMsg.msg.bufTransMem.token = GetCcuXnValue(baseInfo.deviceId, baseInfo.dieId, rep->src.token.Id());
    errorMsg.msg.bufTransMem.len = GetCcuXnValue(baseInfo.deviceId, baseInfo.dieId, rep->len.Id());
    errorMsg.msg.bufTransMem.signalId = rep->sem.Id();
    errorMsg.msg.bufTransMem.signalMask = rep->mask;

    errorInfo.push_back(errorMsg);
}

void CcuErrorHandler::GenErrorInfoBufLocWrite(
    const ErrorInfoBase& baseInfo, shared_ptr<CcuRepBase> repBase, vector<CcuErrorInfo>& errorInfo)
{
    CcuErrorInfo errorMsg{};
    errorMsg.type = CcuErrorType::BUF_TRANS_MEM;
    errorMsg.SetBaseInfo(repBase->Type(), baseInfo.dieId, baseInfo.missionId, repBase->StartInstrId());

    const auto rep = static_pointer_cast<CcuRepBufLocWrite>(repBase);
    errorMsg.msg.bufTransMem.bufId = GetMSIdPerDie(rep->src.Id());
    errorMsg.msg.bufTransMem.addr = GetCcuGSAValue(baseInfo.deviceId, baseInfo.dieId, rep->dst.addr.Id());
    errorMsg.msg.bufTransMem.token = GetCcuXnValue(baseInfo.deviceId, baseInfo.dieId, rep->dst.token.Id());
    errorMsg.msg.bufTransMem.len = GetCcuXnValue(baseInfo.deviceId, baseInfo.dieId, rep->len.Id());
    errorMsg.msg.bufTransMem.signalId = rep->sem.Id();
    errorMsg.msg.bufTransMem.signalMask = rep->mask;

    errorInfo.push_back(errorMsg);
}

void CcuErrorHandler::GenErrorInfoBufReduce(
    const ErrorInfoBase& baseInfo, shared_ptr<CcuRepBase> repBase, vector<CcuErrorInfo>& errorInfo)
{
    CcuErrorInfo errorMsg{};
    errorMsg.type = CcuErrorType::BUF_REDUCE;
    errorMsg.SetBaseInfo(repBase->Type(), baseInfo.dieId, baseInfo.missionId, repBase->StartInstrId());

    const auto rep = static_pointer_cast<CcuRepBufReduce>(repBase);
    errorMsg.msg.bufReduce.count = rep->count;
    errorMsg.msg.bufReduce.dataType = rep->dataType;
    errorMsg.msg.bufReduce.outputDataType = rep->outputDataType;
    errorMsg.msg.bufReduce.opType = rep->opType;
    errorMsg.msg.bufReduce.signalId = rep->sem.Id();
    errorMsg.msg.bufReduce.signalMask = rep->mask;
    errorMsg.msg.bufReduce.xnIdLength = rep->xnIdLength_.Id();
    const auto& buffs = rep->mem;
    (void)memset_s(
        errorMsg.msg.bufReduce.bufIds, sizeof(errorMsg.msg.bufReduce.bufIds), 0xFF,
        sizeof(errorMsg.msg.bufReduce.bufIds));
    for (uint32_t i = 0; i < buffs.size() && i < BUF_REDUCE_ID_SIZE; ++i) {
        errorMsg.msg.bufReduce.bufIds[i] = GetMSIdPerDie(buffs[i].Id());
    }

    errorInfo.push_back(errorMsg);
}

CcuMissionContext CcuErrorHandler::GetCcuMissionContext(int32_t deviceId, uint32_t dieId, uint32_t missionId)
{
    auto tlvHandle = HccpTlvHdcManager::GetInstance().GetTlvHandle(deviceId);
    CHECK_NULLPTR(
        tlvHandle, StringFormat("[CcuErrorHandler][%s] tlvHandle is nullptr, deviceId[%d]", __func__, deviceId));

    struct CustomChannelInfoIn inBuff;
    struct CustomChannelInfoOut outBuff;

    inBuff.op = CcuOpcodeType::CCU_U_OP_GET_MISSION_CTX;
    inBuff.data.dataInfo.udieIdx = dieId;
    inBuff.offsetStartIdx = missionId;
    inBuff.data.dataInfo.dataArraySize = 1; // 读1个MissionContext
    inBuff.data.dataInfo.dataLen = sizeof(CcuMissionContext) * inBuff.data.dataInfo.dataArraySize;

    HrtRaTlvRequestForCustomChannel(tlvHandle, MSG_TYPE_CCU_DISPATCH_CMD, &inBuff, &outBuff);

    CcuMissionContext missionCtx{};
    (void)memcpy_s(&missionCtx, sizeof(missionCtx), outBuff.data.dataInfo.dataArray, inBuff.data.dataInfo.dataLen);
    return missionCtx;
}

CcuLoopContext CcuErrorHandler::GetCcuLoopContext(int32_t deviceId, uint32_t dieId, uint32_t loopCtxId)
{
    auto tlvHandle = HccpTlvHdcManager::GetInstance().GetTlvHandle(deviceId);
    CHECK_NULLPTR(
        tlvHandle, StringFormat("[CcuErrorHandler][%s] tlvHandle is nullptr, deviceId[%d]", __func__, deviceId));

    struct CustomChannelInfoIn inBuff;
    struct CustomChannelInfoOut outBuff;

    inBuff.op = CcuOpcodeType::CCU_U_OP_GET_LOOP_CTX;
    inBuff.data.dataInfo.udieIdx = dieId;
    inBuff.offsetStartIdx = loopCtxId;
    inBuff.data.dataInfo.dataArraySize = 1; // 读1个LoopContext
    inBuff.data.dataInfo.dataLen = sizeof(CcuLoopContext) * inBuff.data.dataInfo.dataArraySize;

    HrtRaTlvRequestForCustomChannel(tlvHandle, MSG_TYPE_CCU_DISPATCH_CMD, &inBuff, &outBuff);

    CcuLoopContext loopCtx{};
    (void)memcpy_s(&loopCtx, sizeof(loopCtx), outBuff.data.dataInfo.dataArray, inBuff.data.dataInfo.dataLen);
    return loopCtx;
}

uint64_t CcuErrorHandler::GetCcuXnValue(int32_t deviceId, uint32_t dieId, uint32_t xnId)
{
    auto tlvHandle = HccpTlvHdcManager::GetInstance().GetTlvHandle(deviceId);
    CHECK_NULLPTR(
        tlvHandle, StringFormat("[CcuErrorHandler][%s] tlvHandle is nullptr, deviceId[%d]", __func__, deviceId));

    struct CustomChannelInfoIn inBuff;
    struct CustomChannelInfoOut outBuff;

    inBuff.op = CcuOpcodeType::CCU_U_OP_GET_XN;
    inBuff.data.dataInfo.udieIdx = dieId;
    inBuff.offsetStartIdx = xnId;
    inBuff.data.dataInfo.dataArraySize = 1; // 读1个Xn
    inBuff.data.dataInfo.dataLen = sizeof(uint64_t) * inBuff.data.dataInfo.dataArraySize;

    HrtRaTlvRequestForCustomChannel(tlvHandle, MSG_TYPE_CCU_DISPATCH_CMD, &inBuff, &outBuff);

    uint64_t xnVal{0};
    (void)memcpy_s(&xnVal, sizeof(xnVal), outBuff.data.dataInfo.dataArray, inBuff.data.dataInfo.dataLen);
    return xnVal;
}

uint64_t CcuErrorHandler::GetCcuGSAValue(int32_t deviceId, uint32_t dieId, uint32_t gsaId)
{
    auto tlvHandle = HccpTlvHdcManager::GetInstance().GetTlvHandle(deviceId);
    CHECK_NULLPTR(
        tlvHandle, StringFormat("[CcuErrorHandler][%s] tlvHandle is nullptr, deviceId[%d]", __func__, deviceId));

    struct CustomChannelInfoIn inBuff;
    struct CustomChannelInfoOut outBuff;

    inBuff.op = CcuOpcodeType::CCU_U_OP_GET_GSA;
    inBuff.data.dataInfo.udieIdx = dieId;
    inBuff.offsetStartIdx = gsaId;
    inBuff.data.dataInfo.dataArraySize = 1; // 读1个GSA
    inBuff.data.dataInfo.dataLen = sizeof(uint64_t) * inBuff.data.dataInfo.dataArraySize;

    HrtRaTlvRequestForCustomChannel(tlvHandle, MSG_TYPE_CCU_DISPATCH_CMD, &inBuff, &outBuff);

    uint64_t gsaVal{0};
    (void)memcpy_s(&gsaVal, sizeof(gsaVal), outBuff.data.dataInfo.dataArray, inBuff.data.dataInfo.dataLen);
    return gsaVal;
}

uint16_t CcuErrorHandler::GetCcuCKEValue(int32_t deviceId, uint32_t dieId, uint32_t ckeId)
{
    auto tlvHandle = HccpTlvHdcManager::GetInstance().GetTlvHandle(deviceId);
    CHECK_NULLPTR(
        tlvHandle, StringFormat("[CcuErrorHandler][%s] tlvHandle is nullptr, deviceId[%d]", __func__, deviceId));

    struct CustomChannelInfoIn inBuff;
    struct CustomChannelInfoOut outBuff;

    inBuff.op = CcuOpcodeType::CCU_U_OP_GET_CKE;
    inBuff.data.dataInfo.udieIdx = dieId;
    inBuff.offsetStartIdx = ckeId;
    inBuff.data.dataInfo.dataArraySize = 1; // 读1个CKE
    inBuff.data.dataInfo.dataLen = sizeof(uint64_t) * inBuff.data.dataInfo.dataArraySize;

    HrtRaTlvRequestForCustomChannel(tlvHandle, MSG_TYPE_CCU_DISPATCH_CMD, &inBuff, &outBuff);

    uint64_t ckeVal{0};
    (void)memcpy_s(&ckeVal, sizeof(ckeVal), outBuff.data.dataInfo.dataArray, inBuff.data.dataInfo.dataLen);
    return static_cast<uint16_t>(ckeVal);
}

} // namespace Hccl
