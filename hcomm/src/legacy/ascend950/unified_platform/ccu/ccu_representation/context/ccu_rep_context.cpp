/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_rep_context.h"

#include "exception_util.h"
#include "ccu_api_exception.h"
#include "ccu_rep_assign.h"
#include "const_val.h"
namespace Hccl {
namespace CcuRep {

    CcuRepContext::CcuRepContext()
    {
        mainBlock = std::make_shared<CcuRep::CcuRepBlock>();
        activeBlock = mainBlock;
    }

    CcuRepContext::~CcuRepContext() {}

    std::shared_ptr<CcuRep::CcuRepBlock> CcuRepContext::CurrentBlock()
    {
        if (activeBlock == nullptr) {
            THROW<CcuApiException>("Invalid ActiveBlock");
        }
        return activeBlock;
    }

    void CcuRepContext::SetCurrentBlock(std::shared_ptr<CcuRep::CcuRepBlock> repBlock) { activeBlock = repBlock; }

    void CcuRepContext::CollectProfilingReps(std::shared_ptr<CcuRep::CcuRepBase> rep)
    {
        if (rep->Type() == CcuRepType::ASSIGN) {
            auto assignRep = dynamic_cast<CcuRepAssign*>(rep.get());
            if (assignRep->subType == AssignSubType::VAR_TO_VAR) {
                lgProfilingInfo.assignProfilingReps.push_back(rep);
            }
        } else if (
            CurrentBlock()->Type() != CcuRep::CcuRepType::LOOP_BLOCK
            && (rep->Type() == CcuRepType::LOC_WAIT_SEM || rep->Type() == CcuRepType::REM_WAIT_SEM
                || rep->Type() == CcuRepType::REM_WAIT_GROUP)) {
            waitCkeProfilingReps.push_back(rep);
        } else if (rep->Type() == CcuRepType::LOOPGROUP) {
            allLgProfilingReps.push_back(rep);
        }
    }

    void CcuRepContext::Append(std::shared_ptr<CcuRep::CcuRepBase> rep)
    {
        CollectProfilingReps(rep);
        CurrentBlock()->Append(rep);
    }

    const std::vector<std::shared_ptr<CcuRep::CcuRepBase>>& CcuRepContext::GetRepSequence()
    {
        return mainBlock->GetReps();
    }

    std::shared_ptr<CcuRep::CcuRepBase> CcuRepContext::GetRepByInstrId(uint16_t instrId)
    {
        for (const auto& rep : GetRepSequence()) {
            const uint16_t instrCount = rep->InstrCount();
            if (instrCount == 0) {
                continue;
            }
            const uint16_t startId = rep->StartInstrId();
            const uint16_t endId = startId + instrCount - 1;
            if (instrId >= startId && instrId <= endId) {
                return rep;
            }
        }
        return nullptr;
    }

    void CcuRepContext::DumpReprestation()
    {
        HCCL_INFO("Rep Count: %lu", GetRepSequence().size());
        for (uint32_t index = 0; index < GetRepSequence().size(); index++) {
            HCCL_INFO("index[%u]: %s", index, GetRepSequence()[index]->Describe().c_str());
        }
    }

    void CcuRepContext::SetDieId(uint32_t dieId) { this->dieId = dieId; }

    uint32_t CcuRepContext::GetDieId() const { return dieId; }

    void CcuRepContext::SetMissionId(uint32_t missionId)
    {
        if (this->missionId == INVALID_U32) {
            this->missionId = missionId;
        }
    }

    uint32_t CcuRepContext::GetMissionId() const { return missionId; }

    void CcuRepContext::SetMissionKey(uint32_t missionKey) { this->missionKey = missionKey; }

    uint32_t CcuRepContext::GetMissionKey() const { return missionKey; }

    std::vector<CcuProfilingInfo>& CcuRepContext::GetProfilingInfo() { return profilingInfo; }

    const std::vector<std::shared_ptr<CcuRepBase>>& CcuRepContext::GetWaiteCkeProfilingReps() const
    {
        return waitCkeProfilingReps;
    }

    LoopGroupProfilingInfo& CcuRepContext::GetLGProfilingInfo() { return lgProfilingInfo; }

    void CcuRepContext::AddSqeProfiling(const CcuCtxArg& arg)
    {
        profilingInfo.clear();
        // 生成SQE粒度profiling信息
        ccuProfilingInfoCache.type = CcuProfilinType::CCU_TASK_PROFILING;
        ccuProfilingInfoCache.name = arg.GetCtxSignature().Describe();
        ccuProfilingInfoCache.dieId = GetDieId();

        profilingInfo.push_back(ccuProfilingInfoCache);
    }

    void CcuRepContext::AddProfiling(const std::string& name, uint32_t mask)
    {
        ccuProfilingInfoCache.type = CcuProfilinType::CCU_WAITCKE_PROFILING;
        ccuProfilingInfoCache.name = name;
        ccuProfilingInfoCache.ckeId = INVALID_CKE_ID;
        ccuProfilingInfoCache.mask = mask;
        (void)memset_s(
            ccuProfilingInfoCache.channelId, sizeof(ccuProfilingInfoCache.channelId), INVALID_VALUE_CHANNELID,
            sizeof(ccuProfilingInfoCache.channelId));

        profilingInfo.push_back(ccuProfilingInfoCache);
    }

    void CcuRepContext::AddProfiling(
        const CcuTransport& transport, const std::string& name, uint32_t signalIndex, uint32_t mask)
    {
        ccuProfilingInfoCache.type = CcuProfilinType::CCU_WAITCKE_PROFILING;
        ccuProfilingInfoCache.name = name;
        ccuProfilingInfoCache.ckeId = transport.GetLocCntCkeByIndex(signalIndex);
        ccuProfilingInfoCache.mask = mask;
        (void)memset_s(
            ccuProfilingInfoCache.channelId, sizeof(ccuProfilingInfoCache.channelId), INVALID_VALUE_CHANNELID,
            sizeof(ccuProfilingInfoCache.channelId));
        ccuProfilingInfoCache.channelId[0] = transport.GetChannelId();

        profilingInfo.push_back(ccuProfilingInfoCache);
    }

    void CcuRepContext::AddProfiling(
        const CcuTransportGroup& transportGroup, const std::string& name, uint32_t signalIndex, uint32_t mask)
    {
        ccuProfilingInfoCache.type = CcuProfilinType::CCU_WAITCKE_PROFILING;
        ccuProfilingInfoCache.name = name;
        u32 cntCkeId = 0;
        HcclResult ret = transportGroup.GetCntCkeId(signalIndex, cntCkeId);
        if (ret != HcclResult::HCCL_SUCCESS) {
            string msg = StringFormat(
                "[AddProfiling]rt get cntCkeId failed. "
                "signalIndex[%u], cntCkeId[%u], return[%d].",
                signalIndex, cntCkeId, ret);
            MACRO_THROW(CcuApiException, msg);
        }
        ccuProfilingInfoCache.ckeId = cntCkeId;
        ccuProfilingInfoCache.mask = mask;

        (void)memset_s(
            ccuProfilingInfoCache.channelId, sizeof(ccuProfilingInfoCache.channelId), INVALID_VALUE_CHANNELID,
            sizeof(ccuProfilingInfoCache.channelId));
        auto& transports = transportGroup.GetTransports();
        for (u32 i = 0; i < transports.size(); i++) {
            ccuProfilingInfoCache.channelId[i] = transports[i]->GetChannelId();
        }

        profilingInfo.push_back(ccuProfilingInfoCache);
    }

    void CcuRepContext::AddProfiling(const std::vector<CcuTransport*>& transports)
    {
        ccuProfilingInfoCache.type = CcuProfilinType::CCU_LOOPGROUP_PROFILING;
        ccuProfilingInfoCache.name = "GroupBroadcast";
        ccuProfilingInfoCache.reduceOpType = 0xFF;   // 0xFF 无效值
        ccuProfilingInfoCache.inputDataType = 0xFF;  // 0xFF 无效值
        ccuProfilingInfoCache.outputDataType = 0xFF; // 0xFF 无效值
        ccuProfilingInfoCache.missionId = GetMissionId();

        (void)memset_s(
            ccuProfilingInfoCache.channelId, sizeof(ccuProfilingInfoCache.channelId), INVALID_VALUE_CHANNELID,
            sizeof(ccuProfilingInfoCache.channelId));
        for (u32 i = 0; i < transports.size(); i++) {
            ccuProfilingInfoCache.channelId[i] = transports[i]->GetChannelId();
        }

        lgProfilingInfo.ccuProfilingInfos.push_back(ccuProfilingInfoCache);
        lgProfilingInfo.lgProfilingReps.push_back(allLgProfilingReps.back());
    }

    void CcuRepContext::AddProfiling(
        const std::vector<CcuTransport*>& transports, DataType dataType, DataType outputDataType, ReduceOp opType)
    {
        ccuProfilingInfoCache.type = CcuProfilinType::CCU_LOOPGROUP_PROFILING;
        ccuProfilingInfoCache.name = "GroupReduce";
        ccuProfilingInfoCache.reduceOpType = opType;
        ccuProfilingInfoCache.inputDataType = dataType;
        ccuProfilingInfoCache.outputDataType = outputDataType;
        ccuProfilingInfoCache.missionId = GetMissionId();

        (void)memset_s(
            ccuProfilingInfoCache.channelId, sizeof(ccuProfilingInfoCache.channelId), INVALID_VALUE_CHANNELID,
            sizeof(ccuProfilingInfoCache.channelId));
        for (u32 i = 0; i < transports.size(); i++) {
            ccuProfilingInfoCache.channelId[i] = transports[i]->GetChannelId();
        }

        lgProfilingInfo.ccuProfilingInfos.push_back(ccuProfilingInfoCache);
        lgProfilingInfo.lgProfilingReps.push_back(allLgProfilingReps.back());
    }

    void CcuRepContext::SetDependencyInfo(uint32_t id, uint32_t mask, std::shared_ptr<CcuRepBase> rep)
    {
        if (mask == 0 || (mask & (mask - 1)) != 0) {
            THROW<CcuApiException>("Invalid Mask[%u]", mask);
        }
        // 查找 id 是否已存在于外层 map 中
        auto idIt = depInfo.find(id);
        if (idIt == depInfo.end()) {
            // 如果不存在，插入一个新的内层 unordered_map
            idIt = depInfo.emplace(id, std::unordered_map<uint32_t, std::vector<std::shared_ptr<CcuRepBase>>>()).first;
        }

        // 现在查找 mask 是否存在于内层 map 中
        auto maskIt = idIt->second.find(mask);
        if (maskIt == idIt->second.end()) {
            // 如果不存在，插入一个新的 vector
            maskIt = idIt->second.emplace(mask, std::vector<std::shared_ptr<CcuRepBase>>()).first;
        }

        // 将 rep 添加到 vector 中
        maskIt->second.push_back(rep);
    }

    std::unordered_map<uint32_t, std::vector<std::shared_ptr<CcuRepBase>>> CcuRepContext::GetDependencyInfo(uint32_t id)
    {
        // 查找给定 id 是否存在于 depInfo 中
        auto it = depInfo.find(id);
        // 如果找到 id，返回与之关联的内层 unordered_map
        if (it != depInfo.end()) {
            return it->second;
        }
        // 如果未找到 id，返回一个空的 unordered_map
        return std::unordered_map<uint32_t, std::vector<std::shared_ptr<CcuRepBase>>>();
    }

    void CcuRepContext::ClearDependencyInfo() { depInfo.clear(); }

}; // namespace CcuRep
}; // namespace Hccl
