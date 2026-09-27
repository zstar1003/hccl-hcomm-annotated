/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_rep_context_v1.h"

#include "exception_util.h"
#include "ccu_api_exception.h"
#include "ccu_rep_assign_v1.h"
#include "const_val.h"

#include "hcomm_c_adpt.h"                                          // 需优化
#include "../../../endpoint_pairs/channels/ccu/ccu_urma_channel.h" // 需优化
#include "ccu_dev_mgr_imp.h"

namespace hcomm {
namespace CcuRep {

    CcuRepContext::CcuRepContext()
    {
        mainBlock = std::make_shared<CcuRep::CcuRepBlock>(insGenerator);
        activeBlock = mainBlock;
    }

    CcuRepContext::~CcuRepContext() {}

    std::shared_ptr<CcuRep::CcuRepBlock> CcuRepContext::CurrentBlock()
    {
        if (activeBlock == nullptr) {
            Hccl::THROW<Hccl::CcuApiException>("Invalid ActiveBlock");
        }
        return activeBlock;
    }

    void CcuRepContext::SetCurrentBlock(std::shared_ptr<CcuRep::CcuRepBlock> repBlock) { activeBlock = repBlock; }

    /**
     * @details:保存rep信息，后续有arg后配合补全profiling信息
     */
    void CcuRepContext::CollectProfilingReps(std::shared_ptr<CcuRep::CcuRepBase> rep)
    {
        if (rep->Type() == CcuRepType::ASSIGN) {
            auto assignRep = dynamic_cast<CcuRepAssign*>(rep.get());
            if (assignRep->GetSubType() == AssignSubType::VAR_TO_VAR) {
                lgProfilingInfo.assignProfilingReps.push_back(rep);
            }
        } else if (
            CurrentBlock()->Type() != CcuRep::CcuRepType::LOOP_BLOCK
            && (rep->Type() == CcuRepType::LOC_WAIT_EVENT || rep->Type() == CcuRepType::REM_WAIT_SEM
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

    const std::vector<std::shared_ptr<CcuRep::CcuRepBase>>& CcuRepContext::GetRepSequence() const
    {
        return mainBlock->GetReps();
    }

    std::shared_ptr<CcuRep::CcuRepBase> CcuRepContext::GetRepByInstrId(uint16_t instrId)
    {
        for (const auto& rep : GetRepSequence()) {
            CHK_PRT_RET(rep == nullptr, HCCL_ERROR("[%s]fail, rep is nullptr", __func__), nullptr);
            const uint16_t repInstrCount = rep->InstrCount();
            if (repInstrCount == 0) {
                continue;
            }
            const uint16_t startId = rep->StartInstrId();
            const uint16_t endId = startId + repInstrCount - 1;
            HCCL_INFO("[%s]startId[%u], endId[%u], instrId[%u]", __func__, startId, endId, instrId);
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

    void CcuRepContext::SetDieId(uint32_t dieId)
    {
        HCCL_INFO("set dieId[%u]", dieId);
        this->dieId = dieId;
    }

    uint32_t CcuRepContext::GetDieId() const { return dieId; }

    void CcuRepContext::SetMissionId(uint32_t missionId)
    {
        if (this->missionId == Hccl::INVALID_U32) {
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

    void CcuRepContext::AddSqeProfiling(const std::string& kernelName)
    {
        constexpr uint32_t defaultDieId = 0; // 首次填写profiling时dieId未确定
        // 生成SQE粒度profiling信息
        ccuProfilingInfoCache.type = static_cast<uint8_t>(CcuProfilinType::CCU_TASK_PROFILING);
        ccuProfilingInfoCache.name = kernelName.c_str();
        ccuProfilingInfoCache.dieId = defaultDieId;
        HCCL_DEBUG(
            "[%s]type[%d], name[%s], defaultDieId[%u]", __func__, ccuProfilingInfoCache.type,
            ccuProfilingInfoCache.name.c_str(), ccuProfilingInfoCache.dieId);
        profilingInfo.push_back(ccuProfilingInfoCache);
    }

    int32_t CcuRepContext::AddProfiling(const std::string& name, uint32_t mask)
    {
        ccuProfilingInfoCache.type = static_cast<uint8_t>(CcuProfilinType::CCU_WAITCKE_PROFILING);
        ccuProfilingInfoCache.name = name;
        ccuProfilingInfoCache.ckeId = INVALID_CKE_ID;
        ccuProfilingInfoCache.mask = mask;
        CHK_SAFETY_FUNC_RET(memset_s(
            ccuProfilingInfoCache.channelId, sizeof(ccuProfilingInfoCache.channelId), INVALID_VALUE_CHANNELID,
            sizeof(ccuProfilingInfoCache.channelId)));

        HCCL_INFO("[%s]name[%s], mask[%u], type[%d]", __func__, name.c_str(), mask, ccuProfilingInfoCache.type);
        profilingInfo.push_back(ccuProfilingInfoCache);
        return HCCL_SUCCESS;
    }

    int32_t CcuRepContext::AddProfiling(
        const ChannelHandle channel, const std::string& name, uint32_t signalIndex, uint32_t mask)
    {
        void* channelPtr{nullptr};
        CHK_RET(static_cast<HcclResult>(HcommChannelGet(channel, &channelPtr)));
        auto* channelImpl = dynamic_cast<CcuUrmaChannel*>(static_cast<Channel*>(channelPtr));
        CHK_PTR_NULL(channelImpl);

        ccuProfilingInfoCache.type = static_cast<uint8_t>(CcuProfilinType::CCU_WAITCKE_PROFILING);
        ccuProfilingInfoCache.name = name;
        CHK_RET(channelImpl->GetLocCkeByIndex(signalIndex, ccuProfilingInfoCache.ckeId));
        ccuProfilingInfoCache.mask = mask;
        CHK_SAFETY_FUNC_RET(memset_s(
            ccuProfilingInfoCache.channelId, sizeof(ccuProfilingInfoCache.channelId), INVALID_VALUE_CHANNELID,
            sizeof(ccuProfilingInfoCache.channelId)));
        ccuProfilingInfoCache.channelId[0] = channelImpl->GetChannelId();
        ccuProfilingInfoCache.channelHandle[0] = channel;

        HCCL_INFO(
            "[%s]channelHandle[0x%llx], name[%s], signalIndex[%u], mask[%u], type[%d], ckeId[%u], channelId[%u]",
            __func__, channel, name.c_str(), signalIndex, mask, ccuProfilingInfoCache.type, ccuProfilingInfoCache.ckeId,
            ccuProfilingInfoCache.channelId[0]);
        profilingInfo.push_back(ccuProfilingInfoCache);
        return HCCL_SUCCESS;
    }

    int32_t CcuRepContext::AddProfiling(const ChannelHandle* channels, uint32_t channelNum)
    {
        CHK_PTR_NULL(channels);
        ccuProfilingInfoCache.type = static_cast<uint8_t>(CcuProfilinType::CCU_LOOPGROUP_PROFILING);
        ccuProfilingInfoCache.name = "GroupBroadcast";
        ccuProfilingInfoCache.reduceOpType = 0xFF;   // 0xFF 无效值
        ccuProfilingInfoCache.inputDataType = 0xFF;  // 0xFF 无效值
        ccuProfilingInfoCache.outputDataType = 0xFF; // 0xFF 无效值
        ccuProfilingInfoCache.missionId = GetMissionId();

        CHK_SAFETY_FUNC_RET(memset_s(
            ccuProfilingInfoCache.channelId, sizeof(ccuProfilingInfoCache.channelId), INVALID_VALUE_CHANNELID,
            sizeof(ccuProfilingInfoCache.channelId)));
        for (u32 i = 0; i < channelNum; i++) {
            void* channelPtr{nullptr};
            CHK_RET(static_cast<HcclResult>(HcommChannelGet(channels[i], &channelPtr)));
            auto* channelImpl = dynamic_cast<CcuUrmaChannel*>(static_cast<Channel*>(channelPtr));
            CHK_PTR_NULL(channelImpl);
            ccuProfilingInfoCache.channelId[i] = channelImpl->GetChannelId();
            ccuProfilingInfoCache.channelHandle[i] = channels[i];
            HCCL_INFO(
                "[%s]type[%d], name[%s], missionId[%u], channelHandle[0x%llx], channelId[%u]", __func__,
                ccuProfilingInfoCache.type, ccuProfilingInfoCache.name.c_str(), ccuProfilingInfoCache.missionId,
                ccuProfilingInfoCache.channelHandle[i], ccuProfilingInfoCache.channelId[i]);
        }

        lgProfilingInfo.ccuProfilingInfos.push_back(ccuProfilingInfoCache);
        if (!allLgProfilingReps.empty()) {
            lgProfilingInfo.lgProfilingReps.push_back(allLgProfilingReps.back());
        }
        return HCCL_SUCCESS;
    }

    int32_t CcuRepContext::AddProfiling(
        const ChannelHandle* channels, uint32_t channelNum, HcommDataType hcommDataType,
        HcommDataType hcommOutputDataType, HcommReduceOp hcommOpType)
    {
        HcclDataType dataType = static_cast<HcclDataType>(hcommDataType);
        HcclDataType outputDataType = static_cast<HcclDataType>(hcommOutputDataType);
        HcclReduceOp opType = static_cast<HcclReduceOp>(hcommOpType);

        CHK_PTR_NULL(channels);
        ccuProfilingInfoCache.type = static_cast<uint8_t>(CcuProfilinType::CCU_LOOPGROUP_PROFILING);
        ccuProfilingInfoCache.name = "GroupReduce";
        ccuProfilingInfoCache.reduceOpType = opType;
        ccuProfilingInfoCache.inputDataType = dataType;
        ccuProfilingInfoCache.outputDataType = outputDataType;
        ccuProfilingInfoCache.missionId = GetMissionId();

        CHK_SAFETY_FUNC_RET(memset_s(
            ccuProfilingInfoCache.channelId, sizeof(ccuProfilingInfoCache.channelId), INVALID_VALUE_CHANNELID,
            sizeof(ccuProfilingInfoCache.channelId)));
        for (u32 i = 0; i < channelNum; ++i) {
            void* channelPtr{nullptr};
            CHK_RET(static_cast<HcclResult>(HcommChannelGet(channels[i], &channelPtr)));
            auto* channelImpl = dynamic_cast<CcuUrmaChannel*>(static_cast<Channel*>(channelPtr));
            CHK_PTR_NULL(channelImpl);
            ccuProfilingInfoCache.channelId[i] = channelImpl->GetChannelId();
            ccuProfilingInfoCache.channelHandle[i] = channels[i];
            HCCL_INFO(
                "[%s]type[%d], name[%s], opType[%d], dataType[%d], outputDataType[%d], missionId[%u], "
                "channelHandle[0x%llx], channelId[%u]",
                __func__, ccuProfilingInfoCache.type, ccuProfilingInfoCache.name.c_str(), opType, dataType,
                outputDataType, ccuProfilingInfoCache.missionId, ccuProfilingInfoCache.channelHandle[i],
                ccuProfilingInfoCache.channelId[i]);
        }

        lgProfilingInfo.ccuProfilingInfos.push_back(ccuProfilingInfoCache);
        lgProfilingInfo.lgProfilingReps.push_back(allLgProfilingReps.back());
        return HCCL_SUCCESS;
    }

    void CcuRepContext::SetDependencyInfo(uint32_t id, uint32_t mask, const std::shared_ptr<CcuRepBase>& rep)
    {
        // 按 mask 各置位 bit 分别登记：异常侧按 1<<i 单 bit 查询，多 bit mask 需拆解到每个单 bit key
        constexpr uint32_t CCU_CKE_BIT_NUM = 16; // CKE 的 bit 数最多为 16
        auto& inner = depInfo[id];
        for (uint32_t i = 0; i < CCU_CKE_BIT_NUM; i++) {
            uint32_t bit = 1u << i;
            if ((mask & bit) != 0u) {
                inner[bit].push_back(rep);
            }
        }
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

    void CcuRepContext::EraseDependencyInfo(uint32_t id) { depInfo.erase(id); }

    void CcuRepContext::ClearDependencyInfo() { depInfo.clear(); }

}; // namespace CcuRep
}; // namespace hcomm
