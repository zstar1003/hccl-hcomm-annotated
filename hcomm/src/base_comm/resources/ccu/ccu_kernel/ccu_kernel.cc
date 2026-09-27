/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel.h"

#include <algorithm>
#include <map>

#include "ccu_rep_v1.h"
#include "ccu_kernel_resource.h"
#include "ccu_microcode_v1.h"

#include "ccu_types.h"
#include "exception_util.h"
#include "ccu_api_exception.h"
#include "ccu_dev_mgr_imp.h"
#include "env_config.h"
#include "ccu_rep_type_v1.h"

#include "hcomm_c_adpt.h"

#include "ccu_rep_context_v1.h"
#include "ccu_rep_funccall_v1.h"
#include "../../endpoint_pairs/channels/ccu/ccu_urma_channel.h"
#include "../ccu_instance/ccu_var_event_res_mgr.h"

#include "ccu_log.h"

#include "hcom_common.h"

// todo: 引入头文件需要检查
#include "ccu_assist_v1.h"
#include "hccl_comm_pub.h"
#include "hccl_types.h"
#include "task_param.h"

#include "ccu_ins_generator_base.h"
#include "ccu_ins_generator_v1.h"
#include "config_plf_log_v2.h"

namespace hcomm {

using Hccl::PLF_DATA_OP;
using Hccl::PLF_RES;

constexpr uint32_t TOKEN_VALUE_INDEX = 2;
constexpr uint16_t INVALID_U16 = 65535;
constexpr uint32_t MAX_LOOP_ENGINE_POOL_SIZE_V1 = 128;
constexpr uint32_t MAX_LOOP_ENGINE_POOL_SIZE_V2 = 512;
constexpr uint32_t MIN_PINNED_REG_GROUP_NUM = 2; // 连续变量数 >= 2 时才需添加 pinned 寄存器组

using CcuRep::CcuInsGeneratorBase;
using CcuRep::CcuInsGeneratorV1;

template <typename T>
T CcuKernel::CreateResAssist(std::array<std::vector<T>, CCU_MAX_IODIE_NUM>& resRecord)
{
    // kernel确认die之前默认为0，需要刷新资源
    // 确认die之后按实际使用die分配资源
    const uint32_t dieId = GetDieId();
    resRecord[dieId].emplace_back(this);
    auto& item = resRecord[dieId].back();
    item.Reset(resRecord[dieId].size(), dieId);
    return item;
}

template <typename T>
std::vector<T>
CcuKernel::CreateBlockResAssist(const uint32_t count, std::array<std::vector<T>, CCU_MAX_IODIE_NUM>& resRecord)
{
    constexpr uint16_t CCU_BLOCK_RES_ID_BASE = 0x1000; // block 批量分配资源 id 基址，与单资源 id 区间隔离便于 DFX 定位
    std::vector<T> block;
    block.reserve(count);
    const uint32_t dieId = GetDieId();
    for (size_t i = 0; i < count; i++) {
        block.emplace_back(this);
        block.back().Reset(static_cast<uint16_t>(CCU_BLOCK_RES_ID_BASE + resRecord[dieId].size() + i), dieId);
    }
    resRecord[dieId].insert(resRecord[dieId].end(), block.begin(), block.end());
    return block;
}

CcuKernel::~CcuKernel() {}

static HcclResult GetDieIdByChannel(const ChannelHandle channel, uint32_t& dieId)
{
    void* channelPtr{nullptr};
    CHK_RET(static_cast<HcclResult>(HcommChannelGet(channel, &channelPtr)));
    auto* channelImpl = dynamic_cast<CcuUrmaChannel*>(static_cast<Channel*>(channelPtr));
    if (channelImpl == nullptr) {
        HCCL_ERROR("[%s] failed to cast channel[0x%llx] to CcuUrmaChannel", __func__, channel);
        return HcclResult::HCCL_E_PTR;
    }
    dieId = channelImpl->GetDieId();
    HCCL_INFO("[%s], channelHandle[0x%llx], dieId[%u]", __func__, channel, dieId);
    return HcclResult::HCCL_SUCCESS;
}

CcuResult GetChannelIdByHandle(ChannelHandle channel, uint32_t& channelId)
{
    void* channelPtr{nullptr};
    // HcommResult 与 HcclResult 数值对齐（见 hcomm_result_defs.h 与 hcomm_res_defs.h 注释），
    // 经 static_cast<HcclResult> 再由 HCCL_TO_CCU_RET 转为 CcuResult
    HcommResult hret = HcommChannelGet(channel, &channelPtr);
    if (static_cast<HcclResult>(hret) != HCCL_SUCCESS) {
        HCCL_ERROR("[%s] HcommChannelGet failed, ret[%d], channel[0x%llx].", __func__, hret, channel);
        return HCCL_TO_CCU_RET(static_cast<HcclResult>(hret));
    }
    auto* channelImpl = dynamic_cast<CcuUrmaChannel*>(static_cast<Channel*>(channelPtr));
    if (channelImpl == nullptr) {
        HCCL_ERROR("[%s] failed to cast channel[0x%llx] to CcuUrmaChannel", __func__, channel);
        return CcuResult::CCU_E_PTR;
    }
    channelId = channelImpl->GetChannelId();
    if (channelId == UINT32_MAX) {
        HCCL_ERROR("[%s] channel[0x%llx] channelId is invalid", __func__, channel);
        return CcuResult::CCU_E_UNAVAIL;
    }
    HCCL_INFO("[%s], channelHandle[0x%llx], channelId[%u]", __func__, channel, channelId);
    return CcuResult::CCU_SUCCESS;
}

static HcclResult GetDieIdByChannels(const std::unordered_set<ChannelHandle>& channels, uint32_t& dieId)
{
    if (channels.empty()) {
        int32_t userDevId = HcclGetThreadDeviceId();
        for (uint32_t die = 0; die < CCU_MAX_IODIE_NUM; die++) {
            bool enableFlag = false;
            CHK_RET(static_cast<HcclResult>(CcuGetDieEnableInfo(userDevId, die, enableFlag)));
            if (enableFlag) {
                dieId = die;
                return HcclResult::HCCL_SUCCESS;
            }
        }

        HCCL_ERROR("[CcuKernel][%s] failed, all dies are disable, userDevId[%d].", __func__, userDevId);
        return HcclResult::HCCL_E_INTERNAL;
    }

    uint32_t firstDieId = 0;
    CHK_RET(GetDieIdByChannel(*channels.begin(), firstDieId));
    for (const auto channel : channels) {
        uint32_t nextDieId = 0;
        CHK_RET(GetDieIdByChannel(channel, nextDieId));
        if (firstDieId != nextDieId) {
            HCCL_ERROR("[%s] failed, the dies of channels are not same.", __func__);
            return HcclResult::HCCL_E_PARA;
        }
    }

    dieId = firstDieId;
    return HcclResult::HCCL_SUCCESS;
}

static HcclResult CheckChannelsDie(const std::unordered_set<ChannelHandle>& channels, const uint32_t targetDieId)
{
    for (const auto channel : channels) {
        uint32_t channelDieId = 0;
        CHK_RET(GetDieIdByChannel(channel, channelDieId));
        if (channelDieId != targetDieId) {
            HCCL_ERROR(
                "[%s] failed, channel[0x%llx] dieId[%u] differs from target dieId[%u].", __func__, channel,
                channelDieId, targetDieId);
            return HcclResult::HCCL_E_PARA;
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

static void MoveResourcesToDie(CcuRepResource& res, uint32_t targetDieId)
{
    if (targetDieId == 0)
        return; // 初始资源位于die0，不用设置

    auto moveAndSet = [&targetDieId](auto& arr) {
        arr[targetDieId] = std::move(arr[0]);
        for (auto& item : arr[targetDieId])
            item.SetDieId(targetDieId);
    };

    moveAndSet(res.ccubufs);
    moveAndSet(res.blockCcubufs);
    moveAndSet(res.executor);
    moveAndSet(res.blockExecutor);
    moveAndSet(res.completedEvent);
    moveAndSet(res.blockCompletedEvent);
    moveAndSet(res.address);
    moveAndSet(res.blockAddress);
    moveAndSet(res.continuousVariable);
    moveAndSet(res.variable);
    moveAndSet(res.localNotify);
}

HcclResult CcuKernel::SetupProfilingInfo(const char* kernelFuncName)
{
    if (kernelFuncName == nullptr || strlen(kernelFuncName) == 0) {
        name_ = std::string("CCU_KERNEL"); // 默认名称
        AddSqeProfiling(name_);
        return HcclResult::HCCL_SUCCESS;
    }

    constexpr size_t MAX_KERNEL_FUNC_NAME_LEN = 128;
    const auto nameLen = strlen(kernelFuncName);
    if (nameLen > MAX_KERNEL_FUNC_NAME_LEN) {
        name_ = std::string(kernelFuncName, MAX_KERNEL_FUNC_NAME_LEN);
        HCCL_WARNING("[CcuKernel][%s] kernelFuncName is too long, reset to %s.", __func__, name_.c_str());
    }

    // 生成SQE粒度profiling信息，此时未选择die，默认die 0
    AddSqeProfiling(name_);
    return HcclResult::HCCL_SUCCESS;
}

static HcclResult
UpdateProfilingInfo(std::vector<CcuProfilingInfo>& profilingInfos, uint32_t dieId, const std::string& kernelName)
{
    if (dieId == 0) {
        // 与默认dieId相同，不需要修改
        return HcclResult::HCCL_SUCCESS;
    }

    // 正常情况仅首个info包含die信息，仅应为CCU_TASK_PROFILING类型
    if (UNLIKELY(profilingInfos.empty())) {
        // profiling不属于主流程，不打断算子业务
        HCCL_INFO("[%s] passed, profiling infos are empty, ccu kernel func[%s].", __func__, kernelName.c_str());
        return HcclResult::HCCL_SUCCESS;
    }

    // 根据选择的die跟新profiling信息
    for (auto& info : profilingInfos) {
        info.dieId = dieId;
    }

    HCCL_INFO("[%s] reset profiling info dieId to [%u], ccu kernel func[%s].", __func__, dieId, kernelName.c_str());
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuKernel::ApplyDieFromChannels()
{
    uint32_t dieId{0};
    CHK_RET(GetDieIdByChannels(channels_, dieId));
    CHK_PRT_RET(
        dieId >= CCU_MAX_IODIE_NUM,
        HCCL_ERROR("[CcuKernel][%s] failed, dieId[%u] should be less than [%u].", __func__, dieId, CCU_MAX_IODIE_NUM),
        HcclResult::HCCL_E_PARA);
    SetDieId(dieId);
    MoveResourcesToDie(res_, dieId);
    (void)UpdateProfilingInfo(profilingInfo, dieId, name_);

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuKernel::ValidateAndApplyDie(uint32_t targetDieId)
{
    CHK_PRT_RET(
        targetDieId >= CCU_MAX_IODIE_NUM,
        HCCL_ERROR(
            "[CcuKernel][%s] failed, dieId[%u] should be less than [%u].", __func__, targetDieId, CCU_MAX_IODIE_NUM),
        HcclResult::HCCL_E_PARA);

    const int32_t userDevId = HcclGetThreadDeviceId();
    bool enableFlag = false;
    CHK_RET(static_cast<HcclResult>(CcuGetDieEnableInfo(userDevId, static_cast<uint8_t>(targetDieId), enableFlag)));
    CHK_PRT_RET(
        !enableFlag,
        HCCL_ERROR(
            "[CcuKernel][%s] failed, target dieId[%u] is disabled, userDevId[%d].", __func__, targetDieId, userDevId),
        HcclResult::HCCL_E_PARA);
    CHK_RET(CheckChannelsDie(channels_, targetDieId));

    SetDieId(targetDieId);
    MoveResourcesToDie(res_, targetDieId);
    (void)UpdateProfilingInfo(profilingInfo, targetDieId, name_);

    return HcclResult::HCCL_SUCCESS;
}

void CcuKernel::SetInsGenerater(CcuInsGeneratorBase* insGeneratorBase) { insGenerator = insGeneratorBase; }

CcuResult CcuKernel::ValidateTaskArgs(const uint64_t* taskArgs, uint32_t argsNum) const
{
    if (loadArgUsedSet_.size() != argsNum) {
        HCCL_ERROR(
            "[CcuKernel][%s] failed, args number does not match the Load instruction, "
            "argsNum = %u, loaded = %zu",
            __func__, argsNum, loadArgUsedSet_.size());
        return CcuResult::CCU_E_INTERNAL;
    }
    for (uint32_t i = 0; i < argsNum; ++i) {
        if (loadArgUsedSet_.count(i) == 0) {
            HCCL_ERROR("[CcuKernel][%s] failed, argId %u not loaded (argsNum=%u)", __func__, i, argsNum);
            return CcuResult::CCU_E_INTERNAL;
        }
    }
    if (argsNum != 0) {
        CCU_CHK_PTR_NULL(taskArgs);
    }
    if (instrInfo_.missionInstrCount == 0 || instrInfo_.instrVec.empty()) {
        HCCL_ERROR(
            "[CcuKernel][%s] failed, mission instructions are empty, "
            "the kernel is not been translated yet.",
            __func__);
        return CcuResult::CCU_E_INTERNAL;
    }
    return CcuResult::CCU_SUCCESS;
}

void CcuKernel::FillTaskParam(
    CcuTaskParam& param, uint32_t index, uint32_t seqNum, const uint64_t* taskArgs, uint32_t argsNum) const
{
    param.dieId = GetDieId();
    param.missionId = GetMissionId();
    param.instStartId = instrInfo_.missionStartInstrId + index * CCU_SQE_ARGS_LEN;
    param.key = GetMissionKey();
    param.argSize = CCU_SQE_ARGS_LEN;

    const uint32_t preMissionInsCnt = index * CCU_SQE_ARGS_LEN;
    const bool isLast = (index == seqNum - 1);
    param.instCnt = isLast ? (instrInfo_.missionInstrCount - preMissionInsCnt) : CCU_SQE_ARGS_LEN;

    if (argsNum > preMissionInsCnt) {
        const uint32_t argsToCopy = isLast ? std::min(argsNum - preMissionInsCnt, CCU_SQE_ARGS_LEN) : CCU_SQE_ARGS_LEN;
        std::copy(taskArgs + preMissionInsCnt, taskArgs + preMissionInsCnt + argsToCopy, std::begin(param.args));
    }

    HCCL_INFO(
        "[GeneTaskParam]task Param, dieId[%u] missionId[%u] instStartId[%u] instCnt[%u], argSize[%u]", param.dieId,
        param.missionId, param.instStartId, param.instCnt, param.argSize);
}

CcuResult CcuKernel::GeneTaskParams(const uint64_t* taskArgs, uint32_t argsNum, std::vector<CcuTaskParam>& taskParams)
{
    CCU_CHK_RET(ValidateTaskArgs(taskArgs, argsNum));

    // 如果args数量超过sqe arg的最大数量，则返回多个TaskParam，前面几个只从sqe中加载args;
    // args数量大于等于0、小于等于最大值时，返回1个TaskParam
    const uint32_t seqNum
        = (argsNum / CCU_SQE_ARGS_LEN) + ((argsNum % CCU_SQE_ARGS_LEN) == 0 ? 0 : 1) + (argsNum == 0 ? 1 : 0);

    const uint32_t preMissionSqeInsCnt = (seqNum - 1) * CCU_SQE_ARGS_LEN;
    if (instrInfo_.missionInstrCount < preMissionSqeInsCnt) {
        HCCL_ERROR(
            "[CcuKernel][%s] failed, missionInstrCount[%u] should be greater "
            "than preMissionSqeInsCnt[%u].",
            __func__, instrInfo_.missionInstrCount, preMissionSqeInsCnt);
        return CcuResult::CCU_E_INTERNAL;
    }

    taskParams.resize(seqNum);
    for (uint32_t index = 0; index < seqNum; index++) {
        FillTaskParam(taskParams[index], index, seqNum, taskArgs, argsNum);
    }

    return CcuResult::CCU_SUCCESS;
}

HcclResult CcuKernel::CreateVariable(const ChannelHandle channel, uint32_t varIndex, CcuRep::Variable* var)
{
    channels_.insert(channel);

    void* channelPtr{nullptr};
    CHK_RET(static_cast<HcclResult>(HcommChannelGet(channel, &channelPtr)));
    auto* channelImpl = dynamic_cast<CcuUrmaChannel*>(static_cast<Channel*>(channelPtr));
    if (channelImpl == nullptr) {
        HCCL_ERROR("[%s] failed to cast channel[0x%llx] to CcuUrmaChannel", __func__, channel);
        return HcclResult::HCCL_E_PTR;
    }
    uint32_t locXnId{0};
    CHK_RET(channelImpl->GetLocXnByIndex(varIndex, locXnId));
    var->Reset(locXnId, channelImpl->GetDieId());
    return HcclResult::HCCL_SUCCESS;
}

void CcuKernel::AddPinnedRegGroup(const CcuRep::Variable& baseVar, uint16_t count)
{
    if (count == 0) {
        return;
    }
    const uint16_t baseId = baseVar.Id();
    for (const auto& g : pinnedRegGroups_) {
        if (g.baseVar.Id() == baseId && g.count == count) {
            return;
        }
    }
    pinnedRegGroups_.push_back({baseVar, count});
}

CcuRepResource& CcuKernel::GetResource() { return res_; }

CcuResReq CcuKernel::GetResourceRequest()
{
    CcuResReq req;
    uint32_t dieId = GetDieId();
    req.msReq[dieId] = res_.ccubufs[dieId].size();
    req.blockMsReq[dieId] = res_.blockCcubufs[dieId].size();
    req.ckeReq[dieId] = res_.completedEvent[dieId].size();
    req.blockCkeReq[dieId] = res_.blockCompletedEvent[dieId].size() + res_.localNotify[dieId].size();
    req.loopEngineReq[dieId] = res_.executor[dieId].size();
    req.blockLoopEngineReq[dieId] = res_.blockExecutor[dieId].size();
    req.gsaReq[dieId] = res_.address[dieId].size();
    req.blockGsaReq[dieId] = res_.blockAddress[dieId].size();
    req.xnReq[dieId] = res_.variable[dieId].size();
    req.blockXnReq[dieId] = res_.continuousVariable[dieId].size();

    req.missionReq.reqType = MissionReqType::FUSION_MULTIPLE_DIE;
    req.missionReq.req[dieId] = 1;

    auto info = Hccl::StringFormat(
        "resource request: dieId[%u], ms[%u], blockMs[%u], cke[%u], blockCke[%u], "
        "loopEngine[%u], blockLoopEngine[%u], gsa[%u], blockGsa[%u], xn[%u], blockXn[%u], "
        "missionId[%u]",
        dieId, req.msReq[dieId], req.blockMsReq[dieId], req.ckeReq[dieId], req.blockCkeReq[dieId],
        req.loopEngineReq[dieId], req.blockLoopEngineReq[dieId], req.gsaReq[dieId], req.blockGsaReq[dieId],
        req.xnReq[dieId], req.blockXnReq[dieId], req.missionReq.req[dieId]);

    PLF_CONFIG_INFO(PLF_RES, "%s", info.c_str());

    return req;
}

template <typename HandleType, typename ResourceType>
static CcuResult GetResourceByHandle(
    std::unordered_map<HandleType, ResourceType>& resourceMap, HandleType handle, ResourceType** resource,
    const char* resourceType)
{
    auto iter = resourceMap.find(handle);
    if (iter == resourceMap.end()) {
        HCCL_ERROR("[%s] failed to find %s by handle: 0x%llx", __func__, resourceType, handle);
        return CcuResult::CCU_E_NOT_FOUND;
    }

    // ccu资源本身可能重载=，对象赋值会被转换成指令，导致流程失败
    *resource = &(iter->second);
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuKernel::GetVariableByHandle(CcuVariableHandle varHandle, CcuRep::Variable** variable)
{
    return GetResourceByHandle(ccuVarMap_, varHandle, variable, "variable");
}
// Alloc 相关接口
CcuResult CcuKernel::VariableAlloc(CcuVariableHandle* varHandle)
{
    PLF_CONFIG_INFO(PLF_DATA_OP, "[VariableAlloc]");
    const auto& var = CreateBlockResAssist(1, res_.continuousVariable);
    CcuVariableHandle handle = ccuVarMap_.size();
    ccuVarMap_.emplace(handle, var[0]);

    *varHandle = handle;
    return CcuResult::CCU_SUCCESS;
}
CcuResult CcuKernel::AddressAlloc(CcuAddressHandle* addrHandle)
{
    PLF_CONFIG_INFO(PLF_DATA_OP, "[AddressAlloc]");
    const auto addr = CreateAddress();
    CcuAddressHandle handle = ccuAddrMap_.size();
    ccuAddrMap_.emplace(handle, addr);
    *addrHandle = handle;
    return CcuResult::CCU_SUCCESS;
}
CcuResult CcuKernel::EventAlloc(CcuEventHandle* eventHandle)
{
    PLF_CONFIG_INFO(PLF_DATA_OP, "[EventAlloc]");
    const auto& event = CreateResAssist(res_.blockCompletedEvent);
    CcuEventHandle handle = ccuEventMap_.size();
    ccuEventMap_.emplace(handle, event);
    *eventHandle = handle;
    return CcuResult::CCU_SUCCESS;
}
CcuResult CcuKernel::BufferAlloc(CcuBufferHandle* bufHandle)
{
    PLF_CONFIG_INFO(PLF_DATA_OP, "[BufferAlloc]");
    const auto& buffer = CreateBlockResAssist(1, res_.blockCcubufs);
    CcuBufferHandle handle = ccuBufferMap_.size();
    ccuBufferMap_.emplace(handle, buffer[0]);
    *bufHandle = handle;
    return CcuResult::CCU_SUCCESS;
}
CcuResult CcuKernel::LocalAddrAlloc(
    CcuLocalAddrHandle* localAddrHandle, CcuAddressHandle* addrHandle, CcuVariableHandle* tokenHandle)
{
    PLF_CONFIG_INFO(PLF_DATA_OP, "[LocalAddrAlloc]");
    auto localAddr = CreateLocalAddr();

    CcuAddressHandle aHandle = ccuAddrMap_.size();
    ccuAddrMap_.emplace(aHandle, localAddr.addr);

    CcuVariableHandle tHandle = ccuVarMap_.size();
    ccuVarMap_.emplace(tHandle, localAddr.token);

    CcuLocalAddrHandle laHandle = ccuLocalAddrMap_.size();
    ccuLocalAddrMap_.emplace(laHandle, localAddr);

    *localAddrHandle = laHandle;
    *addrHandle = aHandle;
    *tokenHandle = tHandle;
    return CcuResult::CCU_SUCCESS;
}
CcuResult CcuKernel::RemoteAddrAlloc(
    CcuRemoteAddrHandle* remoteAddrHandle, CcuAddressHandle* addrHandle, CcuVariableHandle* tokenHandle)
{
    PLF_CONFIG_INFO(PLF_DATA_OP, "[RemoteAddrAlloc]");
    auto remoteAddr = CreateRemoteAddr();

    CcuAddressHandle aHandle = ccuAddrMap_.size();
    ccuAddrMap_.emplace(aHandle, remoteAddr.addr);

    CcuVariableHandle tHandle = ccuVarMap_.size();
    ccuVarMap_.emplace(tHandle, remoteAddr.token);

    CcuRemoteAddrHandle raHandle = ccuRemoteAddrMap_.size();
    ccuRemoteAddrMap_.emplace(raHandle, remoteAddr);

    *remoteAddrHandle = raHandle;
    *addrHandle = aHandle;
    *tokenHandle = tHandle;
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuKernel::BlockVariableAlloc(CcuVariableHandle* varHandles, uint32_t count)
{
    PLF_CONFIG_INFO(PLF_DATA_OP, "[BlockVariableAlloc] count=%u", count);
    const auto& var = CreateBlockResAssist(count, res_.continuousVariable);
    for (uint32_t i = 0; i < count; i++) {
        CcuVariableHandle handle = ccuVarMap_.size();
        ccuVarMap_.emplace(handle, var[i]);
        varHandles[i] = handle;
    }
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuKernel::BlockEventAlloc(CcuEventHandle* eventHandles, uint32_t count)
{
    PLF_CONFIG_INFO(PLF_DATA_OP, "[BlockEventAlloc] count=%u", count);
    const auto& event = CreateBlockResAssist(count, res_.blockCompletedEvent);
    for (uint32_t i = 0; i < count; i++) {
        CcuEventHandle handle = ccuEventMap_.size();
        ccuEventMap_.emplace(handle, event[i]);
        eventHandles[i] = handle;
    }
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuKernel::BlockBufferAlloc(CcuBufferHandle* bufHandles, uint32_t count)
{
    PLF_CONFIG_INFO(PLF_DATA_OP, "[BlockBufferAlloc] count=%u", count);
    const auto& buffer = CreateBlockResAssist(count, res_.blockCcubufs);
    for (uint32_t i = 0; i < count; i++) {
        CcuBufferHandle handle = ccuBufferMap_.size();
        ccuBufferMap_.emplace(handle, buffer[i]);
        bufHandles[i] = handle;
    }
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuKernel::VariableCreateByChannel(ChannelHandle channel, uint32_t varIndex, CcuVariableHandle* varHandle)
{
    PLF_CONFIG_INFO(PLF_DATA_OP, "[VariableCreateByChannel] channel=%llu, varIndex=%u", channel, varIndex);
    channels_.insert(channel);
    CcuRep::Variable var(this);
    CCU_CHK_RET(CreateVariable(channel, varIndex, &var));
    CcuVariableHandle handle = ccuVarMap_.size();
    ccuVarMap_.emplace(handle, var);
    *varHandle = handle;
    declaredLocXns_.insert(var.Id());
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuKernel::VariableCreateByAcquire(CcuVariableHandle acqHandle, uint32_t index, CcuVariableHandle* varHandle)
{
    uint8_t dieId = 0;
    uint32_t xnId = 0;
    CCU_CHK_RET(CcuVarEventResMgr::GetInstance(HcclGetThreadDeviceId()).GetVariableXnId(acqHandle, index, dieId, xnId));

    CcuRep::Variable var(this);
    var.Reset(static_cast<uint16_t>(xnId), static_cast<uint16_t>(dieId));
    CcuVariableHandle handle = ccuVarMap_.size();
    ccuVarMap_.emplace(handle, var);
    *varHandle = handle;
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuKernel::EventCreateByAcquire(CcuEventHandle acqHandle, uint32_t index, CcuEventHandle* eventHandle)
{
    uint8_t dieId = 0;
    uint32_t ckeId = 0;
    CCU_CHK_RET(CcuVarEventResMgr::GetInstance(HcclGetThreadDeviceId()).GetEventCkeId(acqHandle, index, dieId, ckeId));

    CcuRep::CompletedEvent event(this);
    event.Reset(static_cast<uint16_t>(ckeId), static_cast<uint16_t>(dieId));
    CcuEventHandle handle = ccuEventMap_.size();
    ccuEventMap_.emplace(handle, event);
    *eventHandle = handle;
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuKernel::VariableAssignImm(CcuVariableHandle varHandle, uint64_t immediate)
{
    PLF_CONFIG_INFO(PLF_DATA_OP, "[VariableAssignImm] varHandle=%llu, immediate=%llu", varHandle, immediate);
    CcuRep::Variable* variable{nullptr};
    CCU_CHK_RET(GetVariableByHandle(varHandle, &variable));
    // 通过符号重载实现，内部记录rep；异常由入口 HcommCcuKernelRegister 的
    // CCU_EXCEPTION_HANDLE_BEGIN/END 统一接住，无需在此局部 try/catch。
    (*variable) = immediate;
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuKernel::VariableAssignVar(CcuVariableHandle varHandle, CcuVariableHandle varA)
{
    PLF_CONFIG_INFO(PLF_DATA_OP, "[VariableAssignVar] varHandle=%llu, varA=%llu", varHandle, varA);
    CcuRep::Variable* variable{nullptr};
    CCU_CHK_RET(GetVariableByHandle(varHandle, &variable));
    CcuRep::Variable* variableA{nullptr};
    CCU_CHK_RET(GetVariableByHandle(varA, &variableA));
    // 通过符号重载实现，内部记录rep；异常由入口统一 catch。
    (*variable) = (*variableA);
    return CcuResult::CCU_SUCCESS;
}

CcuResult
CcuKernel::VariableAddVarToVar(CcuVariableHandle varHandle, CcuVariableHandle varAHandle, CcuVariableHandle varBHandle)
{
    PLF_CONFIG_INFO(
        PLF_DATA_OP, "[VariableAddVarToVar] varHandle=%llu, varAHandle=%llu, varBHandle=%llu", varHandle, varAHandle,
        varBHandle);
    CcuRep::Variable *resVar{nullptr}, *leftVar{nullptr}, *rightVar{nullptr};
    CCU_CHK_RET(GetVariableByHandle(varHandle, &resVar));
    CCU_CHK_RET(GetVariableByHandle(varAHandle, &leftVar));
    CCU_CHK_RET(GetVariableByHandle(varBHandle, &rightVar));

    // 通过符号重载实现，内部记录rep；异常由入口统一 catch。
    *resVar = *leftVar + *rightVar;
    return CcuResult::CCU_SUCCESS;
}

CcuResult
CcuKernel::VariableSubVarToVar(CcuVariableHandle varHandle, CcuVariableHandle varAHandle, CcuVariableHandle varBHandle)
{
    CcuRep::Variable *resVar{nullptr}, *leftVar{nullptr}, *rightVar{nullptr};
    CCU_CHK_RET(GetVariableByHandle(varHandle, &resVar));
    CCU_CHK_RET(GetVariableByHandle(varAHandle, &leftVar));
    CCU_CHK_RET(GetVariableByHandle(varBHandle, &rightVar));

    *resVar = *leftVar - *rightVar;
    return CcuResult::CCU_SUCCESS;
}

CcuResult
CcuKernel::VariableMulVarToVar(CcuVariableHandle varHandle, CcuVariableHandle varAHandle, CcuVariableHandle varBHandle)
{
    CcuRep::Variable *resVar{nullptr}, *leftVar{nullptr}, *rightVar{nullptr};
    CCU_CHK_RET(GetVariableByHandle(varHandle, &resVar));
    CCU_CHK_RET(GetVariableByHandle(varAHandle, &leftVar));
    CCU_CHK_RET(GetVariableByHandle(varBHandle, &rightVar));

    *resVar = *leftVar * *rightVar;
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuKernel::VariableAddImmToVar(CcuVariableHandle varHandle, CcuVariableHandle varAHandle, uint16_t immediate)
{
    CcuRep::Variable *resVar{nullptr}, *leftVar{nullptr};
    CCU_CHK_RET(GetVariableByHandle(varHandle, &resVar));
    CCU_CHK_RET(GetVariableByHandle(varAHandle, &leftVar));

    *resVar = *leftVar + immediate;
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuKernel::VariableSubImmToVar(CcuVariableHandle varHandle, CcuVariableHandle varAHandle, uint16_t immediate)
{
    CcuRep::Variable *resVar{nullptr}, *leftVar{nullptr};
    CCU_CHK_RET(GetVariableByHandle(varHandle, &resVar));
    CCU_CHK_RET(GetVariableByHandle(varAHandle, &leftVar));

    *resVar = *leftVar - immediate;
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuKernel::VariableMulImmToVar(CcuVariableHandle varHandle, CcuVariableHandle varAHandle, uint16_t immediate)
{
    CcuRep::Variable *resVar{nullptr}, *leftVar{nullptr};
    CCU_CHK_RET(GetVariableByHandle(varHandle, &resVar));
    CCU_CHK_RET(GetVariableByHandle(varAHandle, &leftVar));

    *resVar = *leftVar * immediate;
    return CcuResult::CCU_SUCCESS;
}

CcuResult
CcuKernel::VariableAndVarToVar(CcuVariableHandle varHandle, CcuVariableHandle varAHandle, CcuVariableHandle varBHandle)
{
    CcuRep::Variable *resVar{nullptr}, *leftVar{nullptr}, *rightVar{nullptr};
    CCU_CHK_RET(GetVariableByHandle(varHandle, &resVar));
    CCU_CHK_RET(GetVariableByHandle(varAHandle, &leftVar));
    CCU_CHK_RET(GetVariableByHandle(varBHandle, &rightVar));

    *resVar = *leftVar & *rightVar;
    return CcuResult::CCU_SUCCESS;
}

CcuResult
CcuKernel::VariableOrVarToVar(CcuVariableHandle varHandle, CcuVariableHandle varAHandle, CcuVariableHandle varBHandle)
{
    CcuRep::Variable *resVar{nullptr}, *leftVar{nullptr}, *rightVar{nullptr};
    CCU_CHK_RET(GetVariableByHandle(varHandle, &resVar));
    CCU_CHK_RET(GetVariableByHandle(varAHandle, &leftVar));
    CCU_CHK_RET(GetVariableByHandle(varBHandle, &rightVar));

    *resVar = *leftVar | *rightVar;
    return CcuResult::CCU_SUCCESS;
}

CcuResult
CcuKernel::VariableXorVarToVar(CcuVariableHandle varHandle, CcuVariableHandle varAHandle, CcuVariableHandle varBHandle)
{
    CcuRep::Variable *resVar{nullptr}, *leftVar{nullptr}, *rightVar{nullptr};
    CCU_CHK_RET(GetVariableByHandle(varHandle, &resVar));
    CCU_CHK_RET(GetVariableByHandle(varAHandle, &leftVar));
    CCU_CHK_RET(GetVariableByHandle(varBHandle, &rightVar));

    *resVar = *leftVar ^ *rightVar;
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuKernel::VariableNotVar(CcuVariableHandle varHandle, CcuVariableHandle varAHandle)
{
    CcuRep::Variable *resVar{nullptr}, *leftVar{nullptr};
    CCU_CHK_RET(GetVariableByHandle(varHandle, &resVar));
    CCU_CHK_RET(GetVariableByHandle(varAHandle, &leftVar));

    *resVar = ~(*leftVar);
    return CcuResult::CCU_SUCCESS;
}

CcuResult
CcuKernel::VariableShlVarToVar(CcuVariableHandle varHandle, CcuVariableHandle varAHandle, CcuVariableHandle varBHandle)
{
    CcuRep::Variable *resVar{nullptr}, *leftVar{nullptr}, *rightVar{nullptr};
    CCU_CHK_RET(GetVariableByHandle(varHandle, &resVar));
    CCU_CHK_RET(GetVariableByHandle(varAHandle, &leftVar));
    CCU_CHK_RET(GetVariableByHandle(varBHandle, &rightVar));

    *resVar = *leftVar << *rightVar;
    return CcuResult::CCU_SUCCESS;
}

CcuResult
CcuKernel::VariableShrVarToVar(CcuVariableHandle varHandle, CcuVariableHandle varAHandle, CcuVariableHandle varBHandle)
{
    CcuRep::Variable *resVar{nullptr}, *leftVar{nullptr}, *rightVar{nullptr};
    CCU_CHK_RET(GetVariableByHandle(varHandle, &resVar));
    CCU_CHK_RET(GetVariableByHandle(varAHandle, &leftVar));
    CCU_CHK_RET(GetVariableByHandle(varBHandle, &rightVar));

    *resVar = *leftVar >> *rightVar;
    return CcuResult::CCU_SUCCESS;
}

/*========== Event信号同步类 相关接口 ==========*/
CcuResult CcuKernel::EventRecord(CcuEventHandle eventHandle, uint32_t mask)
{
    PLF_CONFIG_INFO(PLF_DATA_OP, "[EventRecord] eventHandle=%llu, mask=%u", eventHandle, mask);
    CcuRep::CompletedEvent* event{nullptr};
    CCU_CHK_RET(GetEventByHandle(eventHandle, &event));
    // 复用已有的 RecordEvent 实现（内部 Append CcuRepLocRecordEvent）
    CCU_CHK_RET(RecordEvent(*event, mask));
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuKernel::EventWait(CcuEventHandle eventHandle, uint32_t mask)
{
    PLF_CONFIG_INFO(PLF_DATA_OP, "[EventWait] eventHandle=%llu, mask=%u", eventHandle, mask);
    CcuRep::CompletedEvent* event{nullptr};
    CCU_CHK_RET(GetEventByHandle(eventHandle, &event));
    // 复用已有的 WaitEvent 实现（内部 Append CcuRepLocWaitEvent）
    CCU_CHK_RET(WaitEvent(*event, mask));
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuKernel::LocalNotifyRecord(const char* notifyTag, const uint32_t mask)
{
    PLF_CONFIG_INFO(PLF_DATA_OP, "[LocalNotifyRecord] tag=%s, mask=%u", (notifyTag ? notifyTag : "null"), mask);
    if (notifyTag == nullptr) {
        HCCL_ERROR("[CcuKernel][%s] notifyTag is nullptr, please check.", __func__);
        return CcuResult::CCU_E_PTR;
    }
    if (CurrentBlock()->Type() == CcuRep::CcuRepType::LOOP_BLOCK) {
        HCCL_ERROR("[CcuKernel][%s] is not supported in loop block, please check.", __func__);
        return LatchBodyError(CcuResult::CCU_E_NOT_SUPPORT);
    }

    const std::string tagKey(notifyTag);

    auto& sharedNotifies = importedRes_.sharedNotifies;
    if (sharedNotifies.find(tagKey) == sharedNotifies.end()) {
        CcuRep::LocalNotify localNotify;
        sharedNotifies.insert({tagKey, localNotify});
    }

    Append(std::make_shared<CcuRep::CcuRepRecordSharedNotify>(insGenerator, sharedNotifies.at(tagKey), mask));

    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuKernel::LocalNotifyWait(const char* notifyTag, const uint32_t mask)
{
    PLF_CONFIG_INFO(PLF_DATA_OP, "[LocalNotifyWait] tag=%s, mask=%u", (notifyTag ? notifyTag : "null"), mask);
    if (notifyTag == nullptr) {
        HCCL_ERROR("[CcuKernel][%s] notifyTag is nullptr, please check.", __func__);
        return CcuResult::CCU_E_PTR;
    }

    const std::string tagKey(notifyTag);

    auto& sharedNotifies = exportedRes_.sharedNotifies;
    if (sharedNotifies.find(tagKey) == sharedNotifies.end()) {
        CcuRep::LocalNotify notify = CreateLocalNotify();
        exportedRes_.sharedNotifies.insert({tagKey, notify});
    }

    bool isProfiling = CurrentBlock()->Type() != CcuRep::CcuRepType::LOOP_BLOCK;
    Append(std::make_shared<CcuRep::CcuRepLocWaitNotify>(
        insGenerator, exportedRes_.sharedNotifies.at(tagKey), mask, isProfiling));
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuKernel::NotifyRecord(const ChannelHandle channel, uint32_t remoteNotifyIdx, uint32_t mask)
{
    PLF_CONFIG_INFO(
        PLF_DATA_OP, "[NotifyRecord] channel=%llu, remoteNotifyIdx=%u, mask=%u", channel, remoteNotifyIdx, mask);
    if (CurrentBlock()->Type() == CcuRep::CcuRepType::LOOP_BLOCK) {
        HCCL_ERROR("[%s] NotifyRecord is not allowed inside a ccu::Loop body", __func__);
        return LatchBodyError(CcuResult::CCU_E_NOT_SUPPORT);
    }
    channels_.insert(channel);
    Append(std::make_shared<CcuRep::CcuRepRemPostSem>(insGenerator, channel, remoteNotifyIdx, mask));
    return CCU_SUCCESS;
}

CcuResult CcuKernel::NotifyWait(const ChannelHandle channel, uint32_t localNotifyIdx, uint32_t mask)
{
    PLF_CONFIG_INFO(
        PLF_DATA_OP, "[NotifyWait] channel=%llu, localNotifyIdx=%u, mask=%u", channel, localNotifyIdx, mask);
    channels_.insert(channel);
    bool isProfiling = CurrentBlock()->Type() != CcuRep::CcuRepType::LOOP_BLOCK;
    if (isProfiling) {
        CCU_CHK_RET(static_cast<HcclResult>(AddProfiling(channel, "NotifyWait", localNotifyIdx, mask)));
    }
    Append(std::make_shared<CcuRep::CcuRepRemWaitSem>(insGenerator, channel, localNotifyIdx, mask, isProfiling));
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuKernel::CascCntWait(HcommCcuCascCntHandle cntHandle, uint64_t tgtValue)
{
    PLF_CONFIG_INFO(PLF_DATA_OP, "[CascCntWait] cntHandle=%llu, tgtValue=%llu", cntHandle, tgtValue);
    auto rep = std::make_shared<CcuRep::CcuRepCascCntWait>(insGenerator, cntHandle, tgtValue);
    Append(rep);
    return CcuResult::CCU_SUCCESS;
}
CcuResult CcuKernel::CascCntClear(HcommCcuCascCntHandle cntHandle, CcuEventHandle eventHandle, uint32_t mask)
{
    PLF_CONFIG_INFO(
        PLF_DATA_OP, "[CascCntClear] cntHandle=%llu, eventHandle=%llu, mask=%u", cntHandle, eventHandle, mask);
    CcuRep::CompletedEvent* event{nullptr};
    CCU_CHK_RET(GetEventByHandle(eventHandle, &event));
    auto rep = std::make_shared<CcuRep::CcuRepCascCntClear>(insGenerator, cntHandle, *event, mask);
    Append(rep);
    SetDependencyInfo(event->Id(), mask, rep);
    return CcuResult::CCU_SUCCESS;
}
CcuResult CcuKernel::WriteVariableWithNotify(
    const ChannelHandle channel, CcuVariableHandle varHandle, uint32_t remoteVarIdx, uint32_t remoteNotifyIdx,
    uint32_t mask)
{
    PLF_CONFIG_INFO(
        PLF_DATA_OP,
        "[WriteVariableWithNotify] channel=%llu, varHandle=%llu, remoteVarIdx=%u,"
        " remoteNotifyIdx=%u, mask=%u",
        channel, varHandle, remoteVarIdx, remoteNotifyIdx, mask);
    if (CurrentBlock()->Type() == CcuRep::CcuRepType::LOOP_BLOCK) {
        HCCL_ERROR("[%s] WriteVariableWithNotify is not allowed inside a ccu::Loop body", __func__);
        return LatchBodyError(CcuResult::CCU_E_NOT_SUPPORT);
    }
    channels_.insert(channel);
    CcuRep::Variable* var{nullptr};
    CCU_CHK_RET(GetVariableByHandle(varHandle, &var));
    Append(
        std::make_shared<CcuRep::CcuRepRemPostVar>(insGenerator, *var, channel, remoteVarIdx, remoteNotifyIdx, mask));
    return CcuResult::CCU_SUCCESS;
}

// 加载类 相关接口
CcuResult CcuKernel::LoadArg(CcuVariableHandle varHandle, uint32_t argId)
{
    PLF_CONFIG_INFO(PLF_DATA_OP, "[LoadArg] varHandle=%llu, argId=%u", varHandle, argId);
    loadArgUsedSet_.insert(argId);
    CcuRep::Variable* var{nullptr};
    CCU_CHK_RET(GetVariableByHandle(varHandle, &var));
    auto loadArgRep = std::make_shared<CcuRep::CcuRepLoadArg>(
        insGenerator, *var, argId % CCU_SQE_ARGS_LEN, static_cast<uint16_t>(argId));
    Append(loadArgRep);
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuKernel::GetCcuKernelInfo(CcuKernelInfo& info) const
{
    uint32_t maxTaskArgsNum = 0;
    // 如果没有LoadArg指令，则说明不需要传入参数，直接返回0
    if (loadArgUsedSet_.empty()) {
        info.maxTaskArgsNum = 0;
        return CcuResult::CCU_SUCCESS;
    }
    // 遍历loadArgUsedSet_，获取最大的argId，加1得到最大任务参数数量（argId从0连续）
    for (const auto& argId : loadArgUsedSet_) {
        maxTaskArgsNum = std::max(maxTaskArgsNum, argId);
    }
    maxTaskArgsNum += 1;
    info.maxTaskArgsNum = maxTaskArgsNum;
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuKernel::CheckContinuousVariables(
    CcuVariableHandle varHandle, uint32_t num, const CcuRep::Variable& baseVar, const char* tag)
{
    if (num <= 1) {
        return CcuResult::CCU_SUCCESS;
    }
    for (uint32_t i = 1; i < num; i++) {
        CcuRep::Variable* nextVar{nullptr};
        CCU_CHK_RET(GetVariableByHandle(varHandle + i, &nextVar));
        if (nextVar->Id() != baseVar.Id() + i) {
            HCCL_ERROR(
                "[CcuKernel][%s] variables not continuous at index %u, "
                "expected Id %u but got %u",
                tag, i, baseVar.Id() + i, nextVar->Id());
            return HCCL_TO_CCU_RET(HCCL_E_PARA);
        }
    }
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuKernel::LoadVar(uint64_t addr, CcuVariableHandle varHandle, uint32_t num)
{
    PLF_CONFIG_INFO(PLF_DATA_OP, "[LoadVar] addr=0x%llx, varHandle=%llu, num=%u", addr, varHandle, num);
    CcuRep::Variable* var{nullptr};
    CCU_CHK_RET(GetVariableByHandle(varHandle, &var));
    CCU_CHK_RET(CheckContinuousVariables(varHandle, num, *var, "LoadVariable"));
    Append(std::make_shared<CcuRep::CcuRepLoad>(insGenerator, addr, *var, num));
    // 数据面 num>=2 走 Array<Variable>: microcode 会用到 [baseVReg, baseVReg+num)
    // 这段连续 Xn 且组内相对偏移固定, 交给 microcode 后端优化 Pass 2 作为 pinned 组.
    if (num >= 2) {
        AddPinnedRegGroup(*var, static_cast<uint16_t>(num));
    }
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuKernel::CcuLoadVarFromVarAddr(CcuVariableHandle addrHandle, CcuVariableHandle varHandle, uint32_t num)
{
    PLF_CONFIG_INFO(
        PLF_DATA_OP,
        "[CcuLoadVarFromVarAddr] addrHandle=%llu, varHandle=%llu,"
        " num=%u",
        addrHandle, varHandle, num);
    CcuRep::Variable* addrVar{nullptr};
    CCU_CHK_RET(GetVariableByHandle(addrHandle, &addrVar));
    CcuRep::Variable* var{nullptr};
    CCU_CHK_RET(GetVariableByHandle(varHandle, &var));
    CCU_CHK_RET(CheckContinuousVariables(varHandle, num, *var, "LoadVar dst"));
    Append(std::make_shared<CcuRep::CcuRepLoadVar>(insGenerator, *addrVar, *var, num));
    if (num >= MIN_PINNED_REG_GROUP_NUM) {
        AddPinnedRegGroup(*var, static_cast<uint16_t>(num));
    }
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuKernel::LoadAddImm(
    CcuVariableHandle varHandle, uint16_t srcNum, CcuVariableHandle offsetHandle, uint16_t immAddValue,
    CcuVariableHandle dstHandle)
{
    PLF_CONFIG_INFO(
        PLF_DATA_OP, "[LoadAddImm] varHandle=%llu, offsetHandle=%llu, immAddValue=%u, dstHandle=%llu", varHandle,
        offsetHandle, immAddValue, dstHandle);
    CcuRep::Variable* baseVar(nullptr);
    CCU_CHK_RET(GetVariableByHandle(varHandle, &baseVar));
    CcuRep::Variable* offsetVar(nullptr);
    CCU_CHK_RET(GetVariableByHandle(offsetHandle, &offsetVar));
    CcuRep::Variable* dstVar(nullptr);
    CCU_CHK_RET(GetVariableByHandle(dstHandle, &dstVar));
    auto rep
        = std::make_shared<CcuRep::CcuRepLoadAddImm>(insGenerator, *baseVar, srcNum, *offsetVar, immAddValue, *dstVar);
    Append(rep);
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuKernel::AddImmStore(
    CcuVariableHandle varHandle, uint16_t dstNum, CcuVariableHandle offsetHandle, uint16_t immAddValue,
    CcuVariableHandle srcHandle)
{
    PLF_CONFIG_INFO(
        PLF_DATA_OP, "[AddImmStore] varHandle=%llu, offsetHandle=%llu, immAddValue=%u, srcHandle=%llu", varHandle,
        offsetHandle, immAddValue, srcHandle);
    CcuRep::Variable* baseVar(nullptr);
    CCU_CHK_RET(GetVariableByHandle(varHandle, &baseVar));
    CcuRep::Variable* offsetVar(nullptr);
    CCU_CHK_RET(GetVariableByHandle(offsetHandle, &offsetVar));
    CcuRep::Variable* srcVar(nullptr);
    CCU_CHK_RET(GetVariableByHandle(srcHandle, &srcVar));
    auto rep
        = std::make_shared<CcuRep::CcuRepStoreAddImm>(insGenerator, *baseVar, dstNum, *offsetVar, immAddValue, *srcVar);
    Append(rep);
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuKernel::StoreVar(uint64_t addr, CcuVariableHandle varHandle, uint32_t num)
{
    PLF_CONFIG_INFO(PLF_DATA_OP, "[StoreVar] addr=0x%llx, varHandle=%llu, num=%u", addr, varHandle, num);
    CcuRep::Variable* var{nullptr};
    CCU_CHK_RET(GetVariableByHandle(varHandle, &var));
    CCU_CHK_RET(CheckContinuousVariables(varHandle, num, *var, "StoreVariable"));
    Append(std::make_shared<CcuRep::CcuRepStore>(insGenerator, *var, addr, num));
    if (num >= MIN_PINNED_REG_GROUP_NUM) {
        AddPinnedRegGroup(*var, static_cast<uint16_t>(num));
    }
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuKernel::CcuStoreVarToVarAddr(CcuVariableHandle addrHandle, CcuVariableHandle varHandle, uint32_t num)
{
    PLF_CONFIG_INFO(
        PLF_DATA_OP,
        "[CcuStoreVarToVarAddr] addrHandle=%llu, varHandle=%llu,"
        " num=%u",
        addrHandle, varHandle, num);
    CcuRep::Variable* addrVar{nullptr};
    CCU_CHK_RET(GetVariableByHandle(addrHandle, &addrVar));
    CcuRep::Variable* var{nullptr};
    CCU_CHK_RET(GetVariableByHandle(varHandle, &var));
    CCU_CHK_RET(CheckContinuousVariables(varHandle, num, *var, "StoreVar src"));
    Append(std::make_shared<CcuRep::CcuRepStoreVar>(insGenerator, *var, *addrVar, num));
    if (num >= MIN_PINNED_REG_GROUP_NUM) {
        AddPinnedRegGroup(*var, static_cast<uint16_t>(num));
    }
    return CcuResult::CCU_SUCCESS;
}

// 本地数据拷贝 相关实现
CcuResult CcuKernel::LocalCopyMemToBuffer(
    CcuBufferHandle dstHandle, CcuLocalAddrHandle srcHandle, CcuVariableHandle lenHandle, CcuEventHandle eventHandle,
    uint32_t mask)
{
    PLF_CONFIG_INFO(
        PLF_DATA_OP,
        "[LocalCopyMemToBuffer] dstHandle=%llu, srcHandle=%llu, lenHandle=%llu,"
        " eventHandle=%llu, mask=%u",
        dstHandle, srcHandle, lenHandle, eventHandle, mask);
    CcuRep::CcuBuf* dst{nullptr};
    CCU_CHK_RET(GetBufferByHandle(dstHandle, &dst));
    CcuRep::LocalAddr* src{nullptr};
    CCU_CHK_RET(GetLocalAddrByHandle(srcHandle, &src));
    CcuRep::Variable* len{nullptr};
    CCU_CHK_RET(GetVariableByHandle(lenHandle, &len));
    CcuRep::CompletedEvent* event{nullptr};
    CCU_CHK_RET(GetEventByHandle(eventHandle, &event));
    auto ret = LocalCopyNb(*dst, *src, *len, *event, mask); // 复用 protected
    return HCCL_TO_CCU_RET(ret);
}

CcuResult CcuKernel::LocalCopyBufferToMem(
    CcuLocalAddrHandle dstHandle, CcuBufferHandle srcHandle, CcuVariableHandle lenHandle, CcuEventHandle eventHandle,
    uint32_t mask)
{
    PLF_CONFIG_INFO(
        PLF_DATA_OP,
        "[LocalCopyBufferToMem] dstHandle=%llu, srcHandle=%llu, lenHandle=%llu,"
        " eventHandle=%llu, mask=%u",
        dstHandle, srcHandle, lenHandle, eventHandle, mask);
    CcuRep::LocalAddr* dst{nullptr};
    CCU_CHK_RET(GetLocalAddrByHandle(dstHandle, &dst));
    CcuRep::CcuBuf* src{nullptr};
    CCU_CHK_RET(GetBufferByHandle(srcHandle, &src));
    CcuRep::Variable* len{nullptr};
    CCU_CHK_RET(GetVariableByHandle(lenHandle, &len));
    CcuRep::CompletedEvent* event{nullptr};
    CCU_CHK_RET(GetEventByHandle(eventHandle, &event));
    auto ret = LocalCopyNb(*dst, *src, *len, *event, mask);
    return HCCL_TO_CCU_RET(ret);
}

CcuResult CcuKernel::LocalCopyMemToMem(
    CcuLocalAddrHandle dstHandle, CcuLocalAddrHandle srcHandle, CcuVariableHandle lenHandle, CcuEventHandle eventHandle,
    uint32_t mask)
{
    PLF_CONFIG_INFO(
        PLF_DATA_OP,
        "[LocalCopyMemToMem] dstHandle=%llu, srcHandle=%llu, lenHandle=%llu,"
        " eventHandle=%llu, mask=%u",
        dstHandle, srcHandle, lenHandle, eventHandle, mask);
    CcuRep::LocalAddr* dst{nullptr};
    CCU_CHK_RET(GetLocalAddrByHandle(dstHandle, &dst));
    CcuRep::LocalAddr* src{nullptr};
    CCU_CHK_RET(GetLocalAddrByHandle(srcHandle, &src));
    CcuRep::Variable* len{nullptr};
    CCU_CHK_RET(GetVariableByHandle(lenHandle, &len));
    CcuRep::CompletedEvent* event{nullptr};
    CCU_CHK_RET(GetEventByHandle(eventHandle, &event));
    auto ret = LocalCopyNb(*dst, *src, *len, *event, mask);
    return HCCL_TO_CCU_RET(ret);
}

// 本地reduce 相关实现
CcuResult CcuKernel::LocalMemReduce(
    CcuLocalAddrHandle dstHandle, CcuLocalAddrHandle srcHandle, CcuVariableHandle lenHandle, HcclDataType dataType,
    HcclReduceOp opType, CcuEventHandle eventHandle, uint32_t mask)
{
    PLF_CONFIG_INFO(
        PLF_DATA_OP,
        "[LocalMemReduce] dstHandle=%llu, srcHandle=%llu, lenHandle=%llu, dataType=%d,"
        " op=%d, eventHandle=%llu, mask=%u",
        dstHandle, srcHandle, lenHandle, dataType, opType, eventHandle, mask);
    CcuRep::LocalAddr* dst{nullptr};
    CCU_CHK_RET(GetLocalAddrByHandle(dstHandle, &dst));
    CcuRep::LocalAddr* src{nullptr};
    CCU_CHK_RET(GetLocalAddrByHandle(srcHandle, &src));
    CcuRep::Variable* len{nullptr};
    CCU_CHK_RET(GetVariableByHandle(lenHandle, &len));
    CcuRep::CompletedEvent* event{nullptr};
    CCU_CHK_RET(GetEventByHandle(eventHandle, &event));
    auto ret = LocalReduceNb(*dst, *src, *len, dataType, opType, *event, mask);
    return HCCL_TO_CCU_RET(ret);
}

CcuResult CcuKernel::LocalBufferReduce(
    CcuBufferHandle* bufHandles, uint32_t count, HcclDataType dataType, HcclDataType outputDataType,
    HcclReduceOp opType, CcuVariableHandle lenHandle, CcuEventHandle eventHandle, uint32_t mask)
{
    PLF_CONFIG_INFO(
        PLF_DATA_OP,
        "[LocalBufferReduce] count=%u, dataType=%d, outDataType=%d,"
        " op=%d, lenHandle=%llu, eventHandle=%llu, mask=%u",
        count, dataType, outputDataType, opType, lenHandle, eventHandle, mask);
    CcuRep::Variable* len{nullptr};
    CCU_CHK_RET(GetVariableByHandle(lenHandle, &len));
    CcuRep::CompletedEvent* event{nullptr};
    CCU_CHK_RET(GetEventByHandle(eventHandle, &event));
    std::vector<CcuRep::CcuBuf> bufs(count);
    for (uint32_t i = 0; i < count; i++) {
        CcuRep::CcuBuf* buf{nullptr};
        CCU_CHK_RET(GetBufferByHandle(bufHandles[i], &buf));
        bufs[i] = *buf;
    }
    auto ret = LocalReduceNb(bufs.data(), count, dataType, outputDataType, opType, *len, *event, mask);
    return HCCL_TO_CCU_RET(ret);
}

/*========== 远端数据传输操作 ==========*/

CcuResult CcuKernel::ResolveBufRemoteLenEvent(
    CcuBufferHandle bufHandle, CcuRemoteAddrHandle remoteHandle, CcuVariableHandle lenHandle,
    CcuEventHandle eventHandle, CcuRep::CcuBuf** buf, CcuRep::RemoteAddr** remote, CcuRep::Variable** len,
    CcuRep::CompletedEvent** event)
{
    CCU_CHK_RET(GetBufferByHandle(bufHandle, buf));
    CCU_CHK_RET(GetRemoteAddrByHandle(remoteHandle, remote));
    CCU_CHK_RET(GetVariableByHandle(lenHandle, len));
    CCU_CHK_RET(GetEventByHandle(eventHandle, event));
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuKernel::ResolveLocalRemoteLenEvent(
    CcuLocalAddrHandle localHandle, CcuRemoteAddrHandle remoteHandle, CcuVariableHandle lenHandle,
    CcuEventHandle eventHandle, CcuRep::LocalAddr** local, CcuRep::RemoteAddr** remote, CcuRep::Variable** len,
    CcuRep::CompletedEvent** event)
{
    CCU_CHK_RET(GetLocalAddrByHandle(localHandle, local));
    CCU_CHK_RET(GetRemoteAddrByHandle(remoteHandle, remote));
    CCU_CHK_RET(GetVariableByHandle(lenHandle, len));
    CCU_CHK_RET(GetEventByHandle(eventHandle, event));
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuKernel::ResolveRemoteLocalLenEvent(
    CcuRemoteAddrHandle remoteHandle, CcuLocalAddrHandle localHandle, CcuVariableHandle lenHandle,
    CcuEventHandle eventHandle, CcuRep::RemoteAddr** remote, CcuRep::LocalAddr** local, CcuRep::Variable** len,
    CcuRep::CompletedEvent** event)
{
    CCU_CHK_RET(GetRemoteAddrByHandle(remoteHandle, remote));
    CCU_CHK_RET(GetLocalAddrByHandle(localHandle, local));
    CCU_CHK_RET(GetVariableByHandle(lenHandle, len));
    CCU_CHK_RET(GetEventByHandle(eventHandle, event));
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuKernel::ReadMemToMem(
    ChannelHandle channel, CcuLocalAddrHandle localHandle, CcuRemoteAddrHandle remoteHandle,
    CcuVariableHandle lenHandle, CcuEventHandle eventHandle, uint32_t mask)
{
    PLF_CONFIG_INFO(
        PLF_DATA_OP,
        "[ReadMemToMem] channel=%llu, localHandle=%llu, remoteHandle=%llu,"
        " lenHandle=%llu, eventHandle=%llu, mask=%u",
        channel, localHandle, remoteHandle, lenHandle, eventHandle, mask);
    channels_.insert(channel);
    CcuRep::LocalAddr* local{nullptr};
    CcuRep::RemoteAddr* remote{nullptr};
    CcuRep::Variable* len{nullptr};
    CcuRep::CompletedEvent* event{nullptr};
    CCU_CHK_RET(
        ResolveLocalRemoteLenEvent(localHandle, remoteHandle, lenHandle, eventHandle, &local, &remote, &len, &event));
    auto ret = ReadNb(channel, *local, *remote, *len, *event, mask);
    return HCCL_TO_CCU_RET(ret);
}

CcuResult CcuKernel::ReadMemToBuffer(
    ChannelHandle channel, CcuBufferHandle localHandle, CcuRemoteAddrHandle remoteHandle, CcuVariableHandle lenHandle,
    CcuEventHandle eventHandle, uint32_t mask)
{
    PLF_CONFIG_INFO(
        PLF_DATA_OP,
        "[ReadMemToBuffer] channel=%llu, bufHandle=%llu, remoteHandle=%llu,"
        " lenHandle=%llu, eventHandle=%llu, mask=%u",
        channel, localHandle, remoteHandle, lenHandle, eventHandle, mask);
    channels_.insert(channel);
    CcuRep::CcuBuf* local{nullptr};
    CcuRep::RemoteAddr* remote{nullptr};
    CcuRep::Variable* len{nullptr};
    CcuRep::CompletedEvent* event{nullptr};
    CCU_CHK_RET(
        ResolveBufRemoteLenEvent(localHandle, remoteHandle, lenHandle, eventHandle, &local, &remote, &len, &event));
    auto ret = ReadNb(channel, *local, *remote, *len, *event, mask);
    return HCCL_TO_CCU_RET(ret);
}

CcuResult CcuKernel::ReadMemToMemReduce(
    ChannelHandle channel, CcuLocalAddrHandle localHandle, CcuRemoteAddrHandle remoteHandle,
    CcuVariableHandle lenHandle, HcclDataType dataType, HcclReduceOp opType, CcuEventHandle eventHandle, uint32_t mask)
{
    PLF_CONFIG_INFO(
        PLF_DATA_OP,
        "[ReadMemToMemReduce] channel=%llu, lenHandle=%llu, dataType=%d, op=%d,"
        " eventHandle=%llu, mask=%u",
        channel, lenHandle, dataType, opType, eventHandle, mask);
    channels_.insert(channel);
    CcuRep::LocalAddr* local{nullptr};
    CcuRep::RemoteAddr* remote{nullptr};
    CcuRep::Variable* len{nullptr};
    CcuRep::CompletedEvent* event{nullptr};
    CCU_CHK_RET(
        ResolveLocalRemoteLenEvent(localHandle, remoteHandle, lenHandle, eventHandle, &local, &remote, &len, &event));
    auto ret = ReadReduceNb(channel, *local, *remote, *len, dataType, opType, *event, mask);
    return HCCL_TO_CCU_RET(ret);
}

CcuResult CcuKernel::WriteMemToMem(
    ChannelHandle channel, CcuRemoteAddrHandle remoteHandle, CcuLocalAddrHandle localHandle,
    CcuVariableHandle lenHandle, CcuEventHandle eventHandle, uint32_t mask)
{
    PLF_CONFIG_INFO(
        PLF_DATA_OP,
        "[WriteMemToMem] channel=%llu, remoteHandle=%llu, localHandle=%llu,"
        " lenHandle=%llu, eventHandle=%llu, mask=%u",
        channel, remoteHandle, localHandle, lenHandle, eventHandle, mask);
    channels_.insert(channel);
    CcuRep::RemoteAddr* remote{nullptr};
    CcuRep::LocalAddr* local{nullptr};
    CcuRep::Variable* len{nullptr};
    CcuRep::CompletedEvent* event{nullptr};
    CCU_CHK_RET(
        ResolveRemoteLocalLenEvent(remoteHandle, localHandle, lenHandle, eventHandle, &remote, &local, &len, &event));
    auto ret = WriteNb(channel, *remote, *local, *len, *event, mask);
    return HCCL_TO_CCU_RET(ret);
}

CcuResult CcuKernel::WriteBufferToMem(
    ChannelHandle channel, CcuRemoteAddrHandle remoteHandle, CcuBufferHandle localHandle, CcuVariableHandle lenHandle,
    CcuEventHandle eventHandle, uint32_t mask)
{
    PLF_CONFIG_INFO(
        PLF_DATA_OP,
        "[WriteBufferToMem] channel=%llu, remoteHandle=%llu, bufHandle=%llu,"
        " lenHandle=%llu, eventHandle=%llu, mask=%u",
        channel, remoteHandle, localHandle, lenHandle, eventHandle, mask);
    channels_.insert(channel);
    CcuRep::CcuBuf* local{nullptr};
    CcuRep::RemoteAddr* remote{nullptr};
    CcuRep::Variable* len{nullptr};
    CcuRep::CompletedEvent* event{nullptr};
    CCU_CHK_RET(
        ResolveBufRemoteLenEvent(localHandle, remoteHandle, lenHandle, eventHandle, &local, &remote, &len, &event));
    auto ret = WriteNb(channel, *remote, *local, *len, *event, mask);
    return HCCL_TO_CCU_RET(ret);
}

CcuResult CcuKernel::WriteMemToMemReduce(
    ChannelHandle channel, CcuRemoteAddrHandle remoteHandle, CcuLocalAddrHandle localHandle,
    CcuVariableHandle lenHandle, HcclDataType dataType, HcclReduceOp opType, CcuEventHandle eventHandle, uint32_t mask)
{
    PLF_CONFIG_INFO(
        PLF_DATA_OP,
        "[WriteMemToMemReduce] channel=%llu, lenHandle=%llu, dataType=%d, op=%d,"
        " eventHandle=%llu, mask=%u",
        channel, lenHandle, dataType, opType, eventHandle, mask);
    channels_.insert(channel);
    CcuRep::RemoteAddr* remote{nullptr};
    CcuRep::LocalAddr* local{nullptr};
    CcuRep::Variable* len{nullptr};
    CcuRep::CompletedEvent* event{nullptr};
    CCU_CHK_RET(
        ResolveRemoteLocalLenEvent(remoteHandle, localHandle, lenHandle, eventHandle, &remote, &local, &len, &event));
    auto ret = WriteReduceNb(channel, *remote, *local, *len, dataType, opType, *event, mask);
    return HCCL_TO_CCU_RET(ret);
}

CcuResult CcuKernel::WriteVarAtomicAdd(
    CcuVariableHandle channelIdHandle, CcuRemoteAddrHandle varAddrHandle, CcuVariableHandle addValueHandle,
    CcuEventHandle eventHandle, uint32_t mask)
{
    PLF_CONFIG_INFO(
        PLF_DATA_OP,
        "[WriteVarAtomicAdd] channelIdHandle=%llu, varAddrHandle=%llu, addValueHandle=%llu, eventHandle=%llu,"
        " mask=%u",
        channelIdHandle, varAddrHandle, addValueHandle, eventHandle, mask);
    CcuRep::Variable* channelId{nullptr};
    CCU_CHK_RET(GetVariableByHandle(channelIdHandle, &channelId));
    CcuRep::RemoteAddr* varAddr{nullptr};
    CCU_CHK_RET(GetRemoteAddrByHandle(varAddrHandle, &varAddr));
    CcuRep::Variable* addValue{nullptr};
    CCU_CHK_RET(GetVariableByHandle(addValueHandle, &addValue));
    CcuRep::CompletedEvent* event{nullptr};
    CCU_CHK_RET(GetEventByHandle(eventHandle, &event));
    auto rep
        = std::make_shared<CcuRep::CcuRepWriteVarAtomic>(insGenerator, *channelId, *varAddr, *addValue, *event, mask);
    Append(rep);
    SetDependencyInfo(event->Id(), mask, rep);
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuKernel::WriteWithCascCntInc(
    CcuVariableHandle channelIdHandle, CcuRemoteAddrHandle remoteHandle, CcuLocalAddrHandle localHandle,
    CcuVariableHandle lenHandle, CcuRemoteAddrHandle inCntAddrHandle)
{
    PLF_CONFIG_INFO(
        PLF_DATA_OP,
        "[WriteWithCascCntInc] channelIdHandle=%llu, remoteHandle=%llu, localHandle=%llu, lenHandle=%llu,"
        " inCntAddrHandle=%llu",
        channelIdHandle, remoteHandle, localHandle, lenHandle, inCntAddrHandle);
    CcuRep::Variable* channelId{nullptr};
    CCU_CHK_RET(GetVariableByHandle(channelIdHandle, &channelId));
    CcuRep::RemoteAddr* remoteAddr{nullptr};
    CCU_CHK_RET(GetRemoteAddrByHandle(remoteHandle, &remoteAddr));
    CcuRep::LocalAddr* local{nullptr};
    CCU_CHK_RET(GetLocalAddrByHandle(localHandle, &local));
    CcuRep::Variable* len{nullptr};
    CCU_CHK_RET(GetVariableByHandle(lenHandle, &len));
    CcuRep::RemoteAddr* inCntAddr{nullptr};
    CCU_CHK_RET(GetRemoteAddrByHandle(inCntAddrHandle, &inCntAddr));
    auto rep = std::make_shared<CcuRep::CcuRepWriteWithCntInc>(
        insGenerator, *channelId, *remoteAddr, *local, *len, *inCntAddr);
    Append(rep);
    return CcuResult::CCU_SUCCESS;
}

void CcuKernel::FlushClosablePendingIfs()
{
    if (isFlushing_) {
        return;
    }
    isFlushing_ = true;
    while (IfLabelStackTopIsClosable()) {
        const char* lbl = IfLabelStackPop();
        if (lbl != nullptr) {
            IfEnd(lbl);
        }
    }
    isFlushing_ = false;
}

void CcuKernel::Append(std::shared_ptr<CcuRep::CcuRepBase> rep)
{
    FlushClosablePendingIfs();
    CcuRep::CcuRepContext::Append(rep);
}

namespace {
    std::shared_ptr<CcuRep::CcuRepJumpBase> MakeInvertedCondJumpImm(
        CcuInsGeneratorBase* insGenerator, const std::string& destLabelStr, const CcuRep::Variable& targetVar,
        const CcuRep::Variable& expectVar, const CcuRep::Variable& variable, uint64_t immediate,
        CcuConditionType condType, const char* funcName)
    {
        switch (condType) {
            case CCU_CONDITION_EQ:
                return std::make_shared<CcuRep::CcuRepJumpNE>(
                    insGenerator, destLabelStr, targetVar, expectVar, variable, immediate);
            case CCU_CONDITION_NE:
                return std::make_shared<CcuRep::CcuRepJumpEQ>(
                    insGenerator, destLabelStr, targetVar, expectVar, variable, immediate);
            case CCU_CONDITION_LT:
                return std::make_shared<CcuRep::CcuRepJumpGE>(
                    insGenerator, destLabelStr, targetVar, expectVar, variable, immediate);
            case CCU_CONDITION_LE:
                return std::make_shared<CcuRep::CcuRepJumpGT>(
                    insGenerator, destLabelStr, targetVar, expectVar, variable, immediate);
            case CCU_CONDITION_GT:
                return std::make_shared<CcuRep::CcuRepJumpLE>(
                    insGenerator, destLabelStr, targetVar, expectVar, variable, immediate);
            case CCU_CONDITION_GE:
                return std::make_shared<CcuRep::CcuRepJumpLT>(
                    insGenerator, destLabelStr, targetVar, expectVar, variable, immediate);
            default:
                HCCL_ERROR("[%s] unsupported condition type: %d", funcName, condType);
                return nullptr;
        }
    }

    // 双变量版本：当 (lhsVar OP rhsVar) 为假时跳转到 destLabelStr。
    std::shared_ptr<CcuRep::CcuRepJumpBase> MakeInvertedCondJumpVar(
        CcuInsGeneratorBase* insGenerator, const std::string& destLabelStr, const CcuRep::Variable& targetVar,
        const CcuRep::Variable& lhsVar, const CcuRep::Variable& rhsVar, CcuConditionType condType, const char* funcName)
    {
        switch (condType) {
            case CCU_CONDITION_EQ:
                return std::make_shared<CcuRep::CcuRepJumpNE>(insGenerator, destLabelStr, targetVar, lhsVar, rhsVar);
            case CCU_CONDITION_NE:
                return std::make_shared<CcuRep::CcuRepJumpEQ>(insGenerator, destLabelStr, targetVar, lhsVar, rhsVar);
            case CCU_CONDITION_LT:
                return std::make_shared<CcuRep::CcuRepJumpGE>(insGenerator, destLabelStr, targetVar, lhsVar, rhsVar);
            case CCU_CONDITION_LE:
                return std::make_shared<CcuRep::CcuRepJumpGT>(insGenerator, destLabelStr, targetVar, lhsVar, rhsVar);
            case CCU_CONDITION_GT:
                return std::make_shared<CcuRep::CcuRepJumpLE>(insGenerator, destLabelStr, targetVar, lhsVar, rhsVar);
            case CCU_CONDITION_GE:
                return std::make_shared<CcuRep::CcuRepJumpLT>(insGenerator, destLabelStr, targetVar, lhsVar, rhsVar);
            default:
                HCCL_ERROR("[%s] unsupported condition type: %d", funcName, condType);
                return nullptr;
        }
    }
} // namespace

CcuResult
CcuKernel::IfBegin(CcuVariableHandle varHandle, uint64_t immediate, CcuConditionType condType, const char* label)
{
    PLF_CONFIG_INFO(
        PLF_DATA_OP, "[IfBegin] varHandle=%llu, immediate=%llu, condType=%d, label=%s", varHandle, immediate, condType,
        (label ? label : "null"));
    if (CurrentBlock()->Type() == CcuRep::CcuRepType::LOOP_BLOCK) {
        HCCL_ERROR(
            "[%s] CCU_IF is not allowed inside a ccu::Loop body (label='%s')", __func__,
            label != nullptr ? label : "(null)");
        return LatchBodyError(CcuResult::CCU_E_INTERNAL);
    }
    CcuRep::Variable* variable{nullptr};
    CCU_CHK_RET(GetVariableByHandle(varHandle, &variable));

    FlushClosablePendingIfs();

    std::string labelStr(label);
    if (pendingIfCtx_.find(labelStr) != pendingIfCtx_.end()) {
        HCCL_ERROR("[%s] label '%s' already has a pending IfBegin without IfEnd", __func__, label);
        return CcuResult::CCU_E_PARA;
    }

    std::string elseLabelStr = labelStr + "_else";
    std::string endLabelStr = labelStr + "_end";
    auto elseLabel = std::make_shared<CcuRep::CcuRepJumpLabel>(insGenerator, elseLabelStr);
    auto endLabel = std::make_shared<CcuRep::CcuRepJumpLabel>(insGenerator, endLabelStr);
    auto targetVar = CreateJumpTargetVar();
    auto expectVar = CreateExpectVar();

    // 反转条件："if <cond>, 执行块" 等价于 "!<cond> 时跳过块"。
    auto jump = MakeInvertedCondJumpImm(
        insGenerator, elseLabelStr, targetVar, expectVar, *variable, immediate, condType, __func__);
    if (jump == nullptr) {
        return CcuResult::CCU_E_PARA;
    }
    jump->Reference(elseLabel);
    Append(jump);

    PendingIfContext ctx;
    ctx.elseLabel = elseLabel;
    ctx.endLabel = endLabel;
    ctx.hasElse = false;
    pendingIfCtx_.emplace(labelStr, std::move(ctx));

    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuKernel::IfBeginVar(
    CcuVariableHandle lhsHandle, CcuVariableHandle rhsHandle, CcuConditionType condType, const char* label)
{
    if (CurrentBlock()->Type() == CcuRep::CcuRepType::LOOP_BLOCK) {
        HCCL_ERROR(
            "[%s] CCU_IF is not allowed inside a ccu::Loop body (label='%s')", __func__,
            label != nullptr ? label : "(null)");
        return LatchBodyError(CcuResult::CCU_E_INTERNAL);
    }
    CcuRep::Variable* lhsVar{nullptr};
    CcuRep::Variable* rhsVar{nullptr};
    CCU_CHK_RET(GetVariableByHandle(lhsHandle, &lhsVar));
    CCU_CHK_RET(GetVariableByHandle(rhsHandle, &rhsVar));

    FlushClosablePendingIfs();

    std::string labelStr(label);
    if (pendingIfCtx_.find(labelStr) != pendingIfCtx_.end()) {
        HCCL_ERROR("[%s] label '%s' already has a pending IfBegin without IfEnd", __func__, label);
        return CcuResult::CCU_E_PARA;
    }

    std::string endLabelStr = labelStr + "_end";
    std::string elseLabelStr = labelStr + "_else";
    auto endLabel = std::make_shared<CcuRep::CcuRepJumpLabel>(insGenerator, endLabelStr);
    auto elseLabel = std::make_shared<CcuRep::CcuRepJumpLabel>(insGenerator, elseLabelStr);
    auto targetVar = CreateJumpTargetVar();

    auto jump = MakeInvertedCondJumpVar(insGenerator, elseLabelStr, targetVar, *lhsVar, *rhsVar, condType, __func__);
    if (jump == nullptr) {
        return CcuResult::CCU_E_PARA;
    }
    jump->Reference(elseLabel);
    Append(jump);

    PendingIfContext ctx;
    ctx.hasElse = false;
    ctx.endLabel = endLabel;
    ctx.elseLabel = elseLabel;
    pendingIfCtx_.emplace(labelStr, std::move(ctx));

    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuKernel::IfElse(const char* label)
{
    PLF_CONFIG_INFO(PLF_DATA_OP, "[IfElse] label=%s", (label ? label : "null"));
    if (CurrentBlock()->Type() == CcuRep::CcuRepType::LOOP_BLOCK) {
        HCCL_ERROR(
            "[%s] CCU_ELSE is not allowed inside a ccu::Loop body (label='%s')", __func__,
            label != nullptr ? label : "(null)");
        return LatchBodyError(CcuResult::CCU_E_INTERNAL);
    }

    std::string labelStr(label);
    auto iter = pendingIfCtx_.find(labelStr);
    if (iter == pendingIfCtx_.end()) {
        HCCL_ERROR("[%s] no matching IfBegin for label '%s'", __func__, label);
        return CcuResult::CCU_E_NOT_FOUND;
    }

    if (iter->second.hasElse) {
        HCCL_ERROR("[%s] label '%s' already has an IfElse", __func__, label);
        return CcuResult::CCU_E_PARA;
    }

    // At end of then-block: unconditional jump past else-block to endLabel
    std::string endLabelStr = labelStr + "_end";
    auto skipElseVar = CreateJumpTargetVar();
    auto skipElseJump = std::make_shared<CcuRep::CcuRepJump>(insGenerator, endLabelStr, skipElseVar);
    skipElseJump->Reference(iter->second.endLabel);
    Append(skipElseJump);

    // Place the else label (entry point of else-block)
    Append(iter->second.elseLabel);

    iter->second.hasElse = true;

    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuKernel::IfEnd(const char* label)
{
    PLF_CONFIG_INFO(PLF_DATA_OP, "[IfEnd] label=%s", (label ? label : "null"));
    std::string labelStr(label);
    auto iter = pendingIfCtx_.find(labelStr);
    if (iter == pendingIfCtx_.end()) {
        HCCL_ERROR("[%s] no matching IfBegin for label '%s'", __func__, label);
        return CcuResult::CCU_E_NOT_FOUND;
    }

    if (iter->second.hasElse) {
        // Had else-block: place endLabel after else-block
        Append(iter->second.endLabel);
    } else {
        // No else-block: place elseLabel as the skip target
        Append(iter->second.elseLabel);
    }

    pendingIfCtx_.erase(iter);

    return CcuResult::CCU_SUCCESS;
}

CcuResult
CcuKernel::WhileBegin(CcuVariableHandle varHandle, uint64_t immediate, CcuConditionType condType, const char* label)
{
    PLF_CONFIG_INFO(
        PLF_DATA_OP, "[WhileBegin] varHandle=%llu, immediate=%llu, condType=%d, label=%s", varHandle, immediate,
        condType, (label ? label : "null"));
    if (CurrentBlock()->Type() == CcuRep::CcuRepType::LOOP_BLOCK) {
        HCCL_ERROR(
            "[%s] CCU_WHILE is not allowed inside a ccu::Loop body (label='%s')", __func__,
            label != nullptr ? label : "(null)");
        return LatchBodyError(CcuResult::CCU_E_INTERNAL);
    }

    CcuRep::Variable* variable{nullptr};
    CCU_CHK_RET(GetVariableByHandle(varHandle, &variable));

    std::string labelStr(label);
    if (pendingWhileCtx_.find(labelStr) != pendingWhileCtx_.end()) {
        HCCL_ERROR("[%s] label '%s' already has a pending WhileBegin without WhileEnd", __func__, label);
        return CcuResult::CCU_E_PARA;
    }

    std::string beginLabelStr = labelStr + "_begin";
    std::string endLabelStr = labelStr + "_end";
    auto beginLabel = std::make_shared<CcuRep::CcuRepJumpLabel>(insGenerator, beginLabelStr);
    auto endLabel = std::make_shared<CcuRep::CcuRepJumpLabel>(insGenerator, endLabelStr);

    Append(beginLabel);

    auto targetVar = CcuRep::CreateVariable(this);
    auto expectVar = CreateExpectVar();
    auto jump = MakeInvertedCondJumpImm(
        insGenerator, endLabelStr, targetVar, expectVar, *variable, immediate, condType, __func__);
    if (jump == nullptr) {
        return CcuResult::CCU_E_PARA;
    }
    jump->Reference(endLabel);
    Append(jump);

    PendingWhileContext ctx;
    ctx.beginLabel = beginLabel;
    ctx.endLabel = endLabel;
    ctx.varHandle = varHandle;
    ctx.immediate = immediate;
    ctx.condType = condType;
    pendingWhileCtx_.emplace(labelStr, std::move(ctx));

    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuKernel::WhileBeginVar(
    CcuVariableHandle lhsHandle, CcuVariableHandle rhsHandle, CcuConditionType condType, const char* label)
{
    if (CurrentBlock()->Type() == CcuRep::CcuRepType::LOOP_BLOCK) {
        HCCL_ERROR(
            "[%s] CCU_WHILE is not allowed inside a ccu::Loop body (label='%s')", __func__,
            label != nullptr ? label : "(null)");
        return LatchBodyError(CcuResult::CCU_E_INTERNAL);
    }

    CcuRep::Variable* lhsVar{nullptr};
    CcuRep::Variable* rhsVar{nullptr};
    CCU_CHK_RET(GetVariableByHandle(lhsHandle, &lhsVar));
    CCU_CHK_RET(GetVariableByHandle(rhsHandle, &rhsVar));

    std::string labelStr(label);
    if (pendingWhileCtx_.find(labelStr) != pendingWhileCtx_.end()) {
        HCCL_ERROR("[%s] label '%s' already has a pending WhileBegin without WhileEnd", __func__, label);
        return CcuResult::CCU_E_PARA;
    }

    std::string endLabelStr = labelStr + "_end";
    std::string beginLabelStr = labelStr + "_begin";
    auto endLabel = std::make_shared<CcuRep::CcuRepJumpLabel>(insGenerator, endLabelStr);
    auto beginLabel = std::make_shared<CcuRep::CcuRepJumpLabel>(insGenerator, beginLabelStr);

    Append(beginLabel);

    auto targetVar = CcuRep::CreateVariable(this);
    auto jump = MakeInvertedCondJumpVar(insGenerator, endLabelStr, targetVar, *lhsVar, *rhsVar, condType, __func__);
    if (jump == nullptr) {
        return CcuResult::CCU_E_PARA;
    }
    jump->Reference(endLabel);
    Append(jump);

    PendingWhileContext ctx;
    ctx.beginLabel = beginLabel;
    ctx.endLabel = endLabel;
    ctx.varHandle = lhsHandle;
    ctx.immediate = 0;
    ctx.condType = condType;
    pendingWhileCtx_.emplace(labelStr, std::move(ctx));

    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuKernel::WhileEnd(const char* label)
{
    PLF_CONFIG_INFO(PLF_DATA_OP, "[WhileEnd] label=%s", (label ? label : "null"));
    std::string labelStr(label);
    auto iter = pendingWhileCtx_.find(labelStr);
    if (iter == pendingWhileCtx_.end()) {
        HCCL_ERROR("[%s] no matching WhileBegin for label '%s'", __func__, label);
        return CcuResult::CCU_E_NOT_FOUND;
    }

    std::string beginLabelStr = labelStr + "_begin";
    auto loopBackVar = CcuRep::CreateVariable(this);
    auto loopBackJump = std::make_shared<CcuRep::CcuRepJump>(insGenerator, beginLabelStr, loopBackVar);
    loopBackJump->Reference(iter->second.beginLabel);
    Append(loopBackJump);

    Append(iter->second.endLabel);

    pendingWhileCtx_.erase(iter);

    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuKernel::DoWhileBegin(const char* label)
{
    PLF_CONFIG_INFO(PLF_DATA_OP, "[DoWhileBegin] label=%s", (label ? label : "null"));
    if (CurrentBlock()->Type() == CcuRep::CcuRepType::LOOP_BLOCK) {
        HCCL_ERROR(
            "[%s] CCU_DO is not allowed inside a ccu::Loop body (label='%s')", __func__,
            label != nullptr ? label : "(null)");
        return LatchBodyError(CcuResult::CCU_E_INTERNAL);
    }

    std::string labelStr(label);
    if (pendingDoWhileCtx_.find(labelStr) != pendingDoWhileCtx_.end()) {
        HCCL_ERROR("[%s] label '%s' already has a pending DoWhileBegin without DoWhileEnd", __func__, label);
        return CcuResult::CCU_E_PARA;
    }

    std::string beginLabelStr = labelStr + "_begin";
    auto beginLabel = std::make_shared<CcuRep::CcuRepJumpLabel>(insGenerator, beginLabelStr);
    Append(beginLabel);

    PendingDoWhileContext ctx;
    ctx.beginLabel = beginLabel;
    pendingDoWhileCtx_.emplace(labelStr, std::move(ctx));

    return CcuResult::CCU_SUCCESS;
}

CcuResult
CcuKernel::DoWhileEnd(CcuVariableHandle varHandle, uint64_t immediate, CcuConditionType condType, const char* label)
{
    PLF_CONFIG_INFO(
        PLF_DATA_OP, "[DoWhileEnd] varHandle=%llu, immediate=%llu, condType=%d, label=%s", varHandle, immediate,
        condType, (label ? label : "null"));
    CcuRep::Variable* variable{nullptr};
    CCU_CHK_RET(GetVariableByHandle(varHandle, &variable));

    std::string labelStr(label);
    auto iter = pendingDoWhileCtx_.find(labelStr);
    if (iter == pendingDoWhileCtx_.end()) {
        HCCL_ERROR("[%s] no matching DoWhileBegin for label '%s'", __func__, label);
        return CcuResult::CCU_E_NOT_FOUND;
    }

    std::string beginLabelStr = labelStr + "_begin";
    auto targetVar = CcuRep::CreateVariable(this);
    auto expectVar = CreateExpectVar();
    std::shared_ptr<CcuRep::CcuRepJumpBase> jump{nullptr};

    // "condition true => continue looping" means jump back to begin when condition holds
    if (condType == CCU_CONDITION_EQ) {
        jump = std::make_shared<CcuRep::CcuRepJumpEQ>(
            insGenerator, beginLabelStr, targetVar, expectVar, *variable, immediate);
    } else if (condType == CCU_CONDITION_NE) {
        jump = std::make_shared<CcuRep::CcuRepJumpNE>(
            insGenerator, beginLabelStr, targetVar, expectVar, *variable, immediate);
    } else if (condType == CCU_CONDITION_LT) {
        jump = std::make_shared<CcuRep::CcuRepJumpLT>(
            insGenerator, beginLabelStr, targetVar, expectVar, *variable, immediate);
    } else if (condType == CCU_CONDITION_LE) {
        jump = std::make_shared<CcuRep::CcuRepJumpLE>(
            insGenerator, beginLabelStr, targetVar, expectVar, *variable, immediate);
    } else if (condType == CCU_CONDITION_GT) {
        jump = std::make_shared<CcuRep::CcuRepJumpGT>(
            insGenerator, beginLabelStr, targetVar, expectVar, *variable, immediate);
    } else if (condType == CCU_CONDITION_GE) {
        jump = std::make_shared<CcuRep::CcuRepJumpGE>(
            insGenerator, beginLabelStr, targetVar, expectVar, *variable, immediate);
    } else {
        HCCL_ERROR("[%s] unsupported condition type: %d", __func__, condType);
        return CcuResult::CCU_E_PARA;
    }

    jump->Reference(iter->second.beginLabel);
    Append(jump);

    pendingDoWhileCtx_.erase(iter);

    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuKernel::DoWhileEndVar(
    CcuVariableHandle lhsHandle, CcuVariableHandle rhsHandle, CcuConditionType condType, const char* label)
{
    CcuRep::Variable* lhsVar{nullptr};
    CcuRep::Variable* rhsVar{nullptr};
    CCU_CHK_RET(GetVariableByHandle(lhsHandle, &lhsVar));
    CCU_CHK_RET(GetVariableByHandle(rhsHandle, &rhsVar));

    std::string labelStr(label);
    auto iter = pendingDoWhileCtx_.find(labelStr);
    if (iter == pendingDoWhileCtx_.end()) {
        HCCL_ERROR("[%s] no matching DoWhileBegin for label '%s'", __func__, label);
        return CcuResult::CCU_E_NOT_FOUND;
    }

    std::string beginLabelStr = labelStr + "_begin";
    auto targetVar = CcuRep::CreateVariable(this);
    std::shared_ptr<CcuRep::CcuRepJumpBase> jump{nullptr};

    if (condType == CCU_CONDITION_EQ) {
        jump = std::make_shared<CcuRep::CcuRepJumpEQ>(insGenerator, beginLabelStr, targetVar, *lhsVar, *rhsVar);
    } else if (condType == CCU_CONDITION_NE) {
        jump = std::make_shared<CcuRep::CcuRepJumpNE>(insGenerator, beginLabelStr, targetVar, *lhsVar, *rhsVar);
    } else if (condType == CCU_CONDITION_LT) {
        jump = std::make_shared<CcuRep::CcuRepJumpLT>(insGenerator, beginLabelStr, targetVar, *lhsVar, *rhsVar);
    } else if (condType == CCU_CONDITION_LE) {
        jump = std::make_shared<CcuRep::CcuRepJumpLE>(insGenerator, beginLabelStr, targetVar, *lhsVar, *rhsVar);
    } else if (condType == CCU_CONDITION_GT) {
        jump = std::make_shared<CcuRep::CcuRepJumpGT>(insGenerator, beginLabelStr, targetVar, *lhsVar, *rhsVar);
    } else if (condType == CCU_CONDITION_GE) {
        jump = std::make_shared<CcuRep::CcuRepJumpGE>(insGenerator, beginLabelStr, targetVar, *lhsVar, *rhsVar);
    } else {
        HCCL_ERROR("[%s] unsupported condition type: %d", __func__, condType);
        return CcuResult::CCU_E_PARA;
    }

    jump->Reference(iter->second.beginLabel);
    Append(jump);

    pendingDoWhileCtx_.erase(iter);

    return CcuResult::CCU_SUCCESS;
}

// 控制流标签栈实体
void CcuKernel::IfLabelStackPush(const char* label) { iflabelStack_.push_back({label, false}); }
void CcuKernel::IfLabelStackMarkBodyDone()
{
    if (iflabelStack_.empty()) {
        HCCL_ERROR("[CcuKernel::IfLabelStack][MarkBodyDone] stack is empty");
        return;
    }
    iflabelStack_.back().bodyDone = true;
}
const char* CcuKernel::IfLabelStackPopForElse()
{
    if (iflabelStack_.empty()) {
        HCCL_ERROR("[CcuKernel::IfLabelStack][PopForElse] orphan CCU_ELSE: "
                   "no matching CCU_IF on the stack");
        return nullptr;
    }
    if (!iflabelStack_.back().bodyDone) {
        HCCL_ERROR(
            "[CcuKernel::IfLabelStack][PopForElse] CCU_ELSE called while "
            "top if-body is still InBody (label='%s')",
            iflabelStack_.back().label != nullptr ? iflabelStack_.back().label : "(null)");
        return nullptr;
    }
    const char* label = iflabelStack_.back().label;
    iflabelStack_.pop_back();
    return label;
}
bool CcuKernel::IfLabelStackTopIsClosable() { return !iflabelStack_.empty() && iflabelStack_.back().bodyDone; }

const char* CcuKernel::IfLabelStackPop()
{
    if (iflabelStack_.empty()) {
        return nullptr;
    }
    const char* label = iflabelStack_.back().label;
    iflabelStack_.pop_back();
    return label;
}

void CcuKernel::DoWhileLabelStackPush(const char* label)
{
    DoWhileLabelEntry entry;
    entry.label = label;
    entry.snapshotBlock = CurrentBlock();
    entry.snapshotRepCount = (entry.snapshotBlock != nullptr) ? entry.snapshotBlock->GetReps().size() : 0;
    doWhileLabelStack_.push_back(std::move(entry));
}

const char* CcuKernel::DoWhileLabelStackPopForWhile()
{
    if (CurrentBlock()->Type() == CcuRep::CcuRepType::LOOP_BLOCK) {
        return nullptr;
    }
    if (doWhileLabelStack_.empty()) {
        return nullptr;
    }
    DoWhileLabelEntry entry = doWhileLabelStack_.back();
    doWhileLabelStack_.pop_back();

    auto currentBlock = CurrentBlock();
    size_t currentRepCount = (currentBlock != nullptr) ? currentBlock->GetReps().size() : 0;
    if (currentBlock != entry.snapshotBlock || currentRepCount != entry.snapshotRepCount) {
        HCCL_ERROR(
            "[CcuKernel::DoWhileLabelStackPopForWhile] dangling CCU calls between CCU_DO end "
            "and CCU_WHILE (label='%s', snapRep=%zu, curRep=%zu, blockChanged=%d); they must "
            "be syntactically adjacent, otherwise the code in between is pulled into the body.",
            entry.label != nullptr ? entry.label : "(null)", entry.snapshotRepCount, currentRepCount,
            currentBlock != entry.snapshotBlock ? 1 : 0);
        return nullptr;
    }
    return entry.label;
}

CcuResult CcuKernel::GetAddressByHandle(CcuAddressHandle addrHandle, CcuRep::Address** address)
{
    return GetResourceByHandle(ccuAddrMap_, addrHandle, address, "address");
}

// addr = 立即数 → CcuRepAssign(Address, uint64_t)
CcuResult CcuKernel::AddressAssignImm(CcuAddressHandle addrHandle, uint64_t immediate)
{
    PLF_CONFIG_INFO(PLF_DATA_OP, "[AddressAssignImm] addrHandle=%llu, immediate=%llu", addrHandle, immediate);
    CcuRep::Address* address{nullptr};
    CCU_CHK_RET(GetAddressByHandle(addrHandle, &address));
    (*address) = immediate;
    return CcuResult::CCU_SUCCESS;
}

// addr = variable → CcuRepAssign(Address, Variable)
CcuResult CcuKernel::AddressAssignVar(CcuAddressHandle addrHandle, CcuVariableHandle varHandle)
{
    PLF_CONFIG_INFO(PLF_DATA_OP, "[AddressAssignVar] addrHandle=%llu, varHandle=%llu", addrHandle, varHandle);
    CcuRep::Address* address{nullptr};
    CCU_CHK_RET(GetAddressByHandle(addrHandle, &address));

    CcuRep::Variable* variable{nullptr};
    CCU_CHK_RET(GetVariableByHandle(varHandle, &variable));

    (*address) = (*variable);
    return CcuResult::CCU_SUCCESS;
}
// addr = addr → CcuRepAssign(Address, Address)
CcuResult CcuKernel::AddressAssignAddr(CcuAddressHandle dstAddrHandle, CcuAddressHandle srcAddrHandle)
{
    PLF_CONFIG_INFO(
        PLF_DATA_OP, "[AddressAssignAddr] dstAddrHandle=%llu, srcAddrHandle=%llu", dstAddrHandle, srcAddrHandle);
    CcuRep::Address* dstAddress{nullptr};
    CCU_CHK_RET(GetAddressByHandle(dstAddrHandle, &dstAddress));

    CcuRep::Address* srcAddress{nullptr};
    CCU_CHK_RET(GetAddressByHandle(srcAddrHandle, &srcAddress));

    (*dstAddress) = (*srcAddress);
    return CcuResult::CCU_SUCCESS;
}

// resAddr = lhsAddr + rhsVar → CcuRepAdd(Address, Address, Variable)
CcuResult CcuKernel::AddressAddVarToAddr(
    CcuAddressHandle resAddrHandle, CcuAddressHandle lhsAddrHandle, CcuVariableHandle rhsVarHandle)
{
    PLF_CONFIG_INFO(
        PLF_DATA_OP, "[AddressAddVarToAddr] resAddrHandle=%llu, lhsAddrHandle=%llu, rhsVarHandle=%llu", resAddrHandle,
        lhsAddrHandle, rhsVarHandle);
    CcuRep::Address *resAddr{nullptr}, *lhsAddr{nullptr};
    CCU_CHK_RET(GetAddressByHandle(resAddrHandle, &resAddr));
    CCU_CHK_RET(GetAddressByHandle(lhsAddrHandle, &lhsAddr));

    CcuRep::Variable* rhsVar{nullptr};
    CCU_CHK_RET(GetVariableByHandle(rhsVarHandle, &rhsVar));

    *resAddr = *lhsAddr + *rhsVar;
    return CcuResult::CCU_SUCCESS;
}

// resAddr = addrA + addrB → CcuRepAdd(Address, Address, Address)
CcuResult CcuKernel::AddressAddAddrToAddr(
    CcuAddressHandle resAddrHandle, CcuAddressHandle addrAHandle, CcuAddressHandle addrBHandle)
{
    PLF_CONFIG_INFO(
        PLF_DATA_OP, "[AddressAddAddrToAddr] resAddrHandle=%llu, addrAHandle=%llu, addrBHandle=%llu", resAddrHandle,
        addrAHandle, addrBHandle);
    CcuRep::Address *resAddr{nullptr}, *addrA{nullptr}, *addrB{nullptr};
    CCU_CHK_RET(GetAddressByHandle(resAddrHandle, &resAddr));
    CCU_CHK_RET(GetAddressByHandle(addrAHandle, &addrA));
    CCU_CHK_RET(GetAddressByHandle(addrBHandle, &addrB));

    *resAddr = *addrA + *addrB;
    return CcuResult::CCU_SUCCESS;
}

// addr += variable → CcuRepAdd(Address, Variable) 就地加
CcuResult CcuKernel::AddressAddAssignVar(CcuAddressHandle addrHandle, CcuVariableHandle varHandle)
{
    PLF_CONFIG_INFO(PLF_DATA_OP, "[AddressAddAssignVar] addrHandle=%llu, varHandle=%llu", addrHandle, varHandle);
    CcuRep::Address* address{nullptr};
    CCU_CHK_RET(GetAddressByHandle(addrHandle, &address));

    CcuRep::Variable* variable{nullptr};
    CCU_CHK_RET(GetVariableByHandle(varHandle, &variable));

    (*address) += (*variable);
    return CcuResult::CCU_SUCCESS;
}

// addr += addr → 等价于 addr = addr + otherAddr
CcuResult CcuKernel::AddressAddAssignAddr(CcuAddressHandle addrHandle, CcuAddressHandle otherHandle)
{
    PLF_CONFIG_INFO(PLF_DATA_OP, "[AddressAddAssignAddr] addrHandle=%llu, otherHandle=%llu", addrHandle, otherHandle);
    CcuRep::Address* address{nullptr};
    CCU_CHK_RET(GetAddressByHandle(addrHandle, &address));

    CcuRep::Address* other{nullptr};
    CCU_CHK_RET(GetAddressByHandle(otherHandle, &other));

    (*address) = (*address) + (*other);
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuKernel::AddressAddImmToAddr(CcuAddressHandle resAddrHandle, CcuAddressHandle addrAHandle, uint16_t imm)
{
    CcuRep::Address *resAddr{nullptr}, *addrA{nullptr};
    CCU_CHK_RET(GetAddressByHandle(resAddrHandle, &resAddr));
    CCU_CHK_RET(GetAddressByHandle(addrAHandle, &addrA));

    *resAddr = *addrA + imm;
    return CcuResult::CCU_SUCCESS;
}

void CcuKernel::Load(const CcuRep::Variable& var)
{
    auto loadArgRep = std::make_shared<CcuRep::CcuRepLoadArg>(
        insGenerator, var, loadArgIndex_ % CCU_SQE_ARGS_LEN, static_cast<uint16_t>(loadArgIndex_));
    GetLGProfilingInfo().loadRep2ArgIdxMap[loadArgRep] = loadArgIndex_;
    Append(loadArgRep);
    loadArgIndex_++;
}

void CcuKernel::StoreVariable(const CcuRep::Variable& var, uint64_t addr)
{
    Append(std::make_shared<CcuRep::CcuRepStore>(insGenerator, var, addr));
}

void CcuKernel::LoadVariable(const CcuRep::Variable& src, const CcuRep::Variable& var)
{
    Append(std::make_shared<CcuRep::CcuRepLoadVar>(insGenerator, src, var));
}

HcclResult CcuKernel::RecordEvent(CcuRep::CompletedEvent event, uint32_t mask)
{
    if (CurrentBlock()->Type() == CcuRep::CcuRepType::LOOP_BLOCK) {
        HCCL_ERROR("[CcuKernel][%s] is not supported in loop block, please check.", __func__);
        LatchBodyError(HCCL_TO_CCU_RET(HcclResult::HCCL_E_NOT_SUPPORT));
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }

    auto rep = std::make_shared<CcuRep::CcuRepLocRecordEvent>(insGenerator, event, mask);
    Append(rep);
    SetDependencyInfo(event.Id(), mask, rep);
    return HCCL_SUCCESS;
}

HcclResult CcuKernel::WaitEvent(CcuRep::CompletedEvent event, uint32_t mask)
{
    bool isProfiling = CurrentBlock()->Type() != CcuRep::CcuRepType::LOOP_BLOCK;
    auto rep = std::make_shared<CcuRep::CcuRepLocWaitEvent>(insGenerator, event, mask, isProfiling);
    if (isProfiling) {
        CHK_RET(static_cast<HcclResult>(AddProfiling("WaitEvent", rep->GetMask())));
    }
    rep->SetDependencyInfo(GetDependencyInfo(event.Id()));
    EraseDependencyInfo(event.Id());
    Append(rep);
    return HCCL_SUCCESS;
}

CcuResult CcuKernel::GetEventByHandle(CcuEventHandle eventHandle, CcuRep::CompletedEvent** event)
{
    return GetResourceByHandle(ccuEventMap_, eventHandle, event, "completedEvent");
}

/*
LocalAddr / RemoteAddr 相关接口
*/
CcuResult CcuKernel::GetLocalAddrByHandle(CcuLocalAddrHandle handle, CcuRep::LocalAddr** localAddr)
{
    return GetResourceByHandle(ccuLocalAddrMap_, handle, localAddr, "localAddr");
}

CcuResult CcuKernel::GetRemoteAddrByHandle(CcuRemoteAddrHandle handle, CcuRep::RemoteAddr** remoteAddr)
{
    return GetResourceByHandle(ccuRemoteAddrMap_, handle, remoteAddr, "remoteAddr");
}

/*Read新接口*/
HcclResult CcuKernel::ReadNb(
    const ChannelHandle channel, const CcuRep::CcuBuf& loc, const CcuRep::RemoteAddr& rem, const CcuRep::Variable& len,
    CcuRep::CompletedEvent event, uint32_t mask)
{
    channels_.insert(channel);
    auto rep = std::make_shared<CcuRep::CcuRepBufRead>(insGenerator, channel, rem, loc, len, event, mask);
    Append(rep);
    SetDependencyInfo(event.Id(), mask, rep);
    return HCCL_SUCCESS;
}

/*Write新接口*/
HcclResult CcuKernel::WriteNb(
    const ChannelHandle channel, const CcuRep::RemoteAddr& rem, const CcuRep::CcuBuf& loc, const CcuRep::Variable& len,
    CcuRep::CompletedEvent event, uint32_t mask)
{
    channels_.insert(channel);
    auto rep = std::make_shared<CcuRep::CcuRepBufWrite>(insGenerator, channel, loc, rem, len, event, mask);
    Append(rep);
    SetDependencyInfo(event.Id(), mask, rep);
    return HCCL_SUCCESS;
}

static bool isLowPrecisionIn(Hccl::DataType dataType)
{
    return dataType == Hccl::DataType::INT8 || dataType == Hccl::DataType::HIF8 || dataType == Hccl::DataType::FP8E4M3
           || dataType == Hccl::DataType::FP8E5M2;
}

static bool isLowPrecisionOut(Hccl::DataType dataType)
{
    return dataType == Hccl::DataType::FP16 || dataType == Hccl::DataType::BFP16 || dataType == Hccl::DataType::FP32;
}

static Hccl::DataType HcommDataTypeToHcclDataType(const HcclDataType dataType)
{
    static const std::map<HcclDataType, Hccl::DataType> hcclDataTypeMap = {
        {HCCL_DATA_TYPE_INT8, Hccl::DataType::INT8},       {HCCL_DATA_TYPE_INT16, Hccl::DataType::INT16},
        {HCCL_DATA_TYPE_INT32, Hccl::DataType::INT32},     {HCCL_DATA_TYPE_FP16, Hccl::DataType::FP16},
        {HCCL_DATA_TYPE_FP32, Hccl::DataType::FP32},       {HCCL_DATA_TYPE_INT64, Hccl::DataType::INT64},
        {HCCL_DATA_TYPE_UINT64, Hccl::DataType::UINT64},   {HCCL_DATA_TYPE_UINT8, Hccl::DataType::UINT8},
        {HCCL_DATA_TYPE_UINT16, Hccl::DataType::UINT16},   {HCCL_DATA_TYPE_UINT32, Hccl::DataType::UINT32},
        {HCCL_DATA_TYPE_FP64, Hccl::DataType::FP64},       {HCCL_DATA_TYPE_BFP16, Hccl::DataType::BFP16},
        {HCCL_DATA_TYPE_INT128, Hccl::DataType::INT128},
#if !defined(OPEN_BUILD_PROJECT) || defined(ORION_MODE)
        {HCCL_DATA_TYPE_HIF8, Hccl::DataType::HIF8},       {HCCL_DATA_TYPE_FP8E4M3, Hccl::DataType::FP8E4M3},
        {HCCL_DATA_TYPE_FP8E5M2, Hccl::DataType::FP8E5M2}, {HCCL_DATA_TYPE_FP8E8M0, Hccl::DataType::FP8E8M0},
#endif
    };

    auto it = hcclDataTypeMap.find(dataType);
    if (it == hcclDataTypeMap.end()) {
        return Hccl::DataType::INVALID;
    }

    return it->second;
}

static Hccl::ReduceOp HcommReduceOpToHcclReduceOp(const HcclReduceOp reduceOp)
{
    static const std::map<HcclReduceOp, Hccl::ReduceOp> hcclReduceOpMap = {
        {HCCL_REDUCE_SUM, Hccl::ReduceOp::SUM},
        {HCCL_REDUCE_PROD, Hccl::ReduceOp::PROD},
        {HCCL_REDUCE_MAX, Hccl::ReduceOp::MAX},
        {HCCL_REDUCE_MIN, Hccl::ReduceOp::MIN},
    };

    auto it = hcclReduceOpMap.find(reduceOp);
    if (it == hcclReduceOpMap.end()) {
        return Hccl::ReduceOp::INVALID;
    }

    return it->second;
}

HcclResult CcuKernel::LocalReduceNb(
    const CcuRep::CcuBuf* bufs, uint32_t count, HcclDataType dataType, HcclDataType outputDataType, HcclReduceOp opType,
    const CcuRep::Variable& len, CcuRep::CompletedEvent event, uint32_t mask)
{
    auto opType_ = HcommReduceOpToHcclReduceOp(opType);
    auto dataType_ = HcommDataTypeToHcclDataType(dataType);
    auto outputDataType_ = HcommDataTypeToHcclDataType(outputDataType);

    if ((opType_ == Hccl::ReduceOp::SUM && isLowPrecisionIn(dataType_) && !isLowPrecisionOut(outputDataType_))
        || (opType_ == Hccl::ReduceOp::SUM && !isLowPrecisionIn(dataType_) && dataType_ != outputDataType_)
        || (opType_ != Hccl::ReduceOp::SUM && dataType_ != outputDataType_)) {
        return HCCL_E_NOT_SUPPORT;
    }

    std::vector<CcuRep::CcuBuf> ccuBufs(count);
    for (uint32_t i = 0; i < count; i++) {
        ccuBufs[i] = bufs[i];
    }

    auto rep = std::make_shared<CcuRep::CcuRepBufReduce>(
        insGenerator, ccuBufs, count, CcuRep::GetCcuDataType(dataType_, opType_),
        CcuRep::GetCcuDataType(outputDataType_, opType_), CcuRep::GetCcuReduceType(opType_), event, len, mask);
    Append(rep);
    SetDependencyInfo(event.Id(), mask, rep);
    return HCCL_SUCCESS;
}

/*Read新接口*/
HcclResult CcuKernel::ReadNb(
    const ChannelHandle channel, const CcuRep::LocalAddr& loc, const CcuRep::RemoteAddr& rem,
    const CcuRep::Variable& len, CcuRep::CompletedEvent event, uint32_t mask)
{
    channels_.insert(channel);
    auto rep = std::make_shared<CcuRep::CcuRepRead>(insGenerator, channel, loc, rem, len, event, mask);
    Append(rep);
    SetDependencyInfo(event.Id(), mask, rep);
    return HCCL_SUCCESS;
}

/*ReadReduce新接口*/
HcclResult CcuKernel::ReadReduceNb(
    const ChannelHandle channel, const CcuRep::LocalAddr& loc, const CcuRep::RemoteAddr& rem,
    const CcuRep::Variable& len, HcclDataType dataType, HcclReduceOp opType, CcuRep::CompletedEvent event,
    uint32_t mask)
{
    channels_.insert(channel);
    auto opType_ = HcommReduceOpToHcclReduceOp(opType);
    auto dataType_ = HcommDataTypeToHcclDataType(dataType);

    auto rep = std::make_shared<CcuRep::CcuRepRead>(
        insGenerator, channel, loc, rem, len, CcuRep::GetUBDataType(dataType_), CcuRep::GetUBReduceType(opType_), event,
        mask);
    Append(rep);
    SetDependencyInfo(event.Id(), mask, rep);
    return HCCL_SUCCESS;
}

/*Write新接口*/
HcclResult CcuKernel::WriteNb(
    const ChannelHandle channel, const CcuRep::RemoteAddr& rem, const CcuRep::LocalAddr& loc,
    const CcuRep::Variable& len, CcuRep::CompletedEvent event, uint32_t mask)
{
    channels_.insert(channel);
    auto rep = std::make_shared<CcuRep::CcuRepWrite>(insGenerator, channel, rem, loc, len, event, mask);
    Append(rep);
    SetDependencyInfo(event.Id(), mask, rep);
    return HCCL_SUCCESS;
}

HcclResult CcuKernel::WriteReduceNb(
    const ChannelHandle channel, const CcuRep::RemoteAddr& rem, const CcuRep::LocalAddr& loc,
    const CcuRep::Variable& len, HcclDataType dataType, HcclReduceOp opType, CcuRep::CompletedEvent event,
    uint32_t mask)
{
    channels_.insert(channel);
    auto opType_ = HcommReduceOpToHcclReduceOp(opType);
    auto dataType_ = HcommDataTypeToHcclDataType(dataType);

    auto rep = std::make_shared<CcuRep::CcuRepWrite>(
        insGenerator, channel, rem, loc, len, CcuRep::GetUBDataType(dataType_), CcuRep::GetUBReduceType(opType_), event,
        mask);
    Append(rep);
    SetDependencyInfo(event.Id(), mask, rep);
    return HCCL_SUCCESS;
}

CcuResult CcuKernel::GetBufferByHandle(CcuBufferHandle bufferHandle, CcuRep::CcuBuf** buffer)
{
    return GetResourceByHandle(ccuBufferMap_, bufferHandle, buffer, "buffer");
}

HcclResult CcuKernel::LocalCopyNb(
    const CcuRep::LocalAddr& dst, const CcuRep::LocalAddr& src, const CcuRep::Variable& len,
    CcuRep::CompletedEvent event, uint32_t mask)
{
    auto rep = std::make_shared<CcuRep::CcuRepLocCpy>(insGenerator, dst, src, len, event, mask);
    Append(rep);
    SetDependencyInfo(event.Id(), mask, rep);
    return HCCL_SUCCESS;
}

HcclResult CcuKernel::LocalCopyNb(
    const CcuRep::CcuBuf& dst, const CcuRep::LocalAddr& src, const CcuRep::Variable& len, CcuRep::CompletedEvent event,
    uint32_t mask)
{
    auto rep = std::make_shared<CcuRep::CcuRepBufLocRead>(insGenerator, src, dst, len, event, mask);
    Append(rep);
    SetDependencyInfo(event.Id(), mask, rep);
    return HCCL_SUCCESS;
}

HcclResult CcuKernel::LocalCopyNb(
    const CcuRep::LocalAddr& dst, const CcuRep::CcuBuf& src, const CcuRep::Variable& len, CcuRep::CompletedEvent event,
    uint32_t mask)
{
    auto rep = std::make_shared<CcuRep::CcuRepBufLocWrite>(insGenerator, src, dst, len, event, mask);
    Append(rep);
    SetDependencyInfo(event.Id(), mask, rep);
    return HCCL_SUCCESS;
}

HcclResult CcuKernel::LocalReduceNb(
    const CcuRep::LocalAddr& dst, const CcuRep::LocalAddr& src, const CcuRep::Variable& len, HcclDataType dataType,
    HcclReduceOp opType, CcuRep::CompletedEvent event, uint32_t mask)
{
    auto opType_ = HcommReduceOpToHcclReduceOp(opType);
    auto dataType_ = HcommDataTypeToHcclDataType(dataType);

    auto rep = std::make_shared<CcuRep::CcuRepLocCpy>(
        insGenerator, dst, src, len, CcuRep::GetUBDataType(dataType_), CcuRep::GetUBReduceType(opType_), event, mask);
    Append(rep);
    SetDependencyInfo(event.Id(), mask, rep);
    return HCCL_SUCCESS;
}

CcuRep::FuncCall CcuKernel::Func(const std::string& label) { return CcuRep::FuncCall(this, label); }

CcuRep::FuncCall CcuKernel::Func(const CcuRep::Variable& funcAddr) { return CcuRep::FuncCall(this, funcAddr); }

CcuRep::LoopCall CcuKernel::Loop(const std::string& label) { return CcuRep::LoopCall(this, label); }

CcuResult CcuKernel::LoopCreate(CcuLoop* loop)
{
    PLF_CONFIG_INFO(PLF_DATA_OP, "[LoopCreate]");
    if (loop == nullptr) {
        HCCL_ERROR("[CcuKernel::LoopCreate] null pointer");
        return CcuResult::CCU_E_PTR;
    }
    if (CurrentBlock()->Type() == CcuRep::CcuRepType::LOOP_BLOCK) {
        HCCL_ERROR("[CcuKernel::LoopCreate] cannot create loop inside a loop body");
        return LatchBodyError(CcuResult::CCU_E_INTERNAL);
    }
    if (inFuncBody_) {
        HCCL_ERROR("[CcuKernel::LoopCreate] cannot create loop inside a func body");
        return LatchBodyError(CcuResult::CCU_E_INTERNAL);
    }

    CcuLoop handle = ++loopHandleCounter_;
    std::string label = "loop_" + std::to_string(handle);

    LoopDescriptor desc;
    desc.label = label;
    desc.repLoopBlock = std::make_shared<CcuRep::CcuRepLoopBlock>(insGenerator, label);

    loopMap_[handle] = std::move(desc);
    *loop = handle;
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuKernel::LatchBodyError(CcuResult err)
{
    if ((CurrentBlock()->Type() == CcuRep::CcuRepType::LOOP_BLOCK || inFuncBody_)
        && bodyError_ == CcuResult::CCU_SUCCESS) {
        bodyError_ = err;
    }
    return err;
}

CcuResult CcuKernel::LoopBodyEnter(CcuLoop loop)
{
    PLF_CONFIG_INFO(PLF_DATA_OP, "[LoopBodyEnter] loop=%llu", loop);
    auto it = loopMap_.find(loop);
    if (it == loopMap_.end()) {
        HCCL_ERROR("[CcuKernel::LoopBodyEnter] invalid loop handle %lu", loop);
        return CcuResult::CCU_E_PARA;
    }
    auto& desc = it->second;
    if (desc.bodyDefined) {
        HCCL_ERROR("[CcuKernel::LoopBodyEnter] loop %lu body already defined", loop);
        return CcuResult::CCU_E_INTERNAL;
    }

    Append(desc.repLoopBlock);
    desc.prevActiveBlock = CurrentBlock();
    SetCurrentBlock(desc.repLoopBlock);
    ++loopBodyDepth_;
    bodyError_ = CcuResult::CCU_SUCCESS;

    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuKernel::LoopBodyExit(CcuLoop loop)
{
    PLF_CONFIG_INFO(PLF_DATA_OP, "[LoopBodyExit] loop=%llu", loop);
    auto it = loopMap_.find(loop);
    if (it == loopMap_.end()) {
        HCCL_ERROR("[CcuKernel::LoopBodyExit] invalid loop handle %lu", loop);
        return CcuResult::CCU_E_PARA;
    }
    auto& desc = it->second;

    SetCurrentBlock(desc.prevActiveBlock);
    desc.bodyDefined = true;
    --loopBodyDepth_;

    if (bodyError_ != CcuResult::CCU_SUCCESS) {
        const CcuResult err = bodyError_;
        HCCL_ERROR("[CcuKernel::LoopBodyExit] illegal operation inside loop body, err=%d", err);
        bodyError_ = CcuResult::CCU_SUCCESS;
        return err;
    }

    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuKernel::FuncBlockLookup(const void* funcPtr, uint64_t* outHandle)
{
    PLF_CONFIG_INFO(PLF_DATA_OP, "[FuncBlockLookup] funcPtr=%p", funcPtr);
    if (funcPtr == nullptr || outHandle == nullptr) {
        HCCL_ERROR("[CcuKernel::FuncBlockLookup] null pointer");
        return CcuResult::CCU_E_PTR;
    }
    if (CurrentBlock()->Type() == CcuRep::CcuRepType::LOOP_BLOCK || inFuncBody_) {
        HCCL_ERROR("[CcuKernel::FuncBlockLookup] ccu::CallFunc only allowed at top level");
        return LatchBodyError(CcuResult::CCU_E_INTERNAL);
    }

    auto it = funcInstanceMap_.find(funcPtr);
    *outHandle = (it == funcInstanceMap_.end()) ? 0 : it->second;
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuKernel::FuncBlockBegin(const void* funcPtr, uint64_t* outHandle)
{
    PLF_CONFIG_INFO(PLF_DATA_OP, "[FuncBlockBegin] funcPtr=%p", funcPtr);
    if (funcPtr == nullptr || outHandle == nullptr) {
        HCCL_ERROR("[CcuKernel::FuncBlockBegin] null pointer");
        return CcuResult::CCU_E_PTR;
    }
    if (CurrentBlock()->Type() == CcuRep::CcuRepType::LOOP_BLOCK || inFuncBody_) {
        HCCL_ERROR("[CcuKernel::FuncBlockBegin] ccu::CallFunc only allowed at top level");
        return LatchBodyError(CcuResult::CCU_E_INTERNAL);
    }

    auto exist = funcInstanceMap_.find(funcPtr);
    if (exist != funcInstanceMap_.end()) {
        *outHandle = exist->second;
        return CcuResult::CCU_SUCCESS;
    }

    const uint64_t handle = ++funcHandleCounter_;
    std::string label = "func_" + std::to_string(handle);

    FuncDescriptor desc;
    desc.funcPtr = funcPtr;
    desc.label = label;
    desc.repFuncBlock = std::make_shared<CcuRep::CcuRepFuncBlock>(insGenerator, label);
    desc.prevActiveBlock = CurrentBlock();

    funcMap_[handle] = desc;
    SetCurrentBlock(desc.repFuncBlock);
    inFuncBody_ = true;
    bodyError_ = CcuResult::CCU_SUCCESS;
    *outHandle = handle;
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuKernel::FuncBlockEnd(uint64_t handle)
{
    PLF_CONFIG_INFO(PLF_DATA_OP, "[FuncBlockEnd] handle=%llu", handle);
    auto it = funcMap_.find(handle);
    if (it == funcMap_.end()) {
        HCCL_ERROR("[CcuKernel::FuncBlockEnd] invalid func handle %lu", handle);
        return CcuResult::CCU_E_PARA;
    }
    auto& desc = it->second;

    SetCurrentBlock(desc.prevActiveBlock);
    inFuncBody_ = false;

    if (bodyError_ != CcuResult::CCU_SUCCESS) {
        const CcuResult err = bodyError_;
        HCCL_ERROR("[CcuKernel::FuncBlockEnd] illegal operation inside func body, err=%d", err);
        bodyError_ = CcuResult::CCU_SUCCESS;
        funcMap_.erase(it);
        return err;
    }

    Append(desc.repFuncBlock);
    desc.bodyDefined = true;
    funcInstanceMap_[desc.funcPtr] = handle;
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuKernel::FuncDefineInArg(uint64_t handle, CcuVariableHandle formal)
{
    PLF_CONFIG_INFO(PLF_DATA_OP, "[FuncDefineInArg] handle=%llu, formal=%llu", handle, formal);
    auto it = funcMap_.find(handle);
    if (it == funcMap_.end()) {
        HCCL_ERROR("[CcuKernel::FuncDefineInArg] invalid func handle %lu", handle);
        return CcuResult::CCU_E_PARA;
    }

    CcuRep::Variable* formalVar = nullptr;
    CCU_CHK_RET(GetVariableByHandle(formal, &formalVar));
    it->second.repFuncBlock->DefineInArg(*formalVar);
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuKernel::FuncCall(uint64_t handle, const CcuVariableHandle* inArgs, uint32_t numIn)
{
    PLF_CONFIG_INFO(PLF_DATA_OP, "[FuncCall] handle=%llu, numIn=%u", handle, numIn);
    if (CurrentBlock()->Type() == CcuRep::CcuRepType::LOOP_BLOCK || inFuncBody_) {
        HCCL_ERROR("[CcuKernel::FuncCall] ccu::CallFunc only allowed at top level");
        return LatchBodyError(CcuResult::CCU_E_INTERNAL);
    }
    if (numIn > 0 && inArgs == nullptr) {
        HCCL_ERROR("[CcuKernel::FuncCall] null input args");
        return CcuResult::CCU_E_PTR;
    }

    auto it = funcMap_.find(handle);
    if (it == funcMap_.end() || !it->second.bodyDefined) {
        HCCL_ERROR("[CcuKernel::FuncCall] invalid func handle %lu", handle);
        return CcuResult::CCU_E_PARA;
    }

    auto repFuncCall = std::make_shared<CcuRep::CcuRepFuncCall>(insGenerator, it->second.label);
    for (uint32_t i = 0; i < numIn; i++) {
        CcuRep::Variable* actual = nullptr;
        CCU_CHK_RET(GetVariableByHandle(inArgs[i], &actual));
        repFuncCall->SetInArg(*actual);
    }
    Append(repFuncCall);
    return CcuResult::CCU_SUCCESS;
}

// 按 maxLoopNum 把 res_.blockExecutor[0] 扩容到至少 maxLoopNum 个 LoopEngine。
// 与 CreateBlockResAssist 对齐：所有 LoopEngine 资源先落在 die0 池，待实际 die 确定后
// 再由 MoveResourcesToDie 迁移到目标 die。
// 不同 LoopGroup 通过 local loopIdx 复用同一池低位 executorId，所以这里只
// "补足"而不是"累加"。
CcuResult CcuKernel::EnsureLoopEnginePool(uint32_t maxLoopNum)
{
    if (maxLoopNum == 0) {
        HCCL_ERROR("[CcuKernel::EnsureLoopEnginePool] maxLoopNum must be > 0");
        return CcuResult::CCU_E_PARA;
    }
    const uint32_t maxPoolSize
        = (ccuVersion_ == CcuVersion::CCU_V2) ? MAX_LOOP_ENGINE_POOL_SIZE_V2 : MAX_LOOP_ENGINE_POOL_SIZE_V1;
    if (maxLoopNum > maxPoolSize) {
        HCCL_ERROR(
            "[CcuKernel::EnsureLoopEnginePool] maxLoopNum(%u) exceeds max supported %u", maxLoopNum, maxPoolSize);
        return CcuResult::CCU_E_PARA;
    }
    constexpr uint32_t poolDieId = 0;
    auto& loopEnginePool = res_.blockExecutor[poolDieId];
    if (maxLoopNum <= loopEnginePool.size()) {
        return CcuResult::CCU_SUCCESS;
    }
    const uint32_t deficit = maxLoopNum - static_cast<uint32_t>(loopEnginePool.size());
    std::vector<CcuRep::Executor> tmp(deficit, CcuRep::Executor(this));
    (void)CreateBlockExecutor(deficit, tmp.data());
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuKernel::LoopGroupCreate(CcuLoopGroup* group, uint32_t maxLoopNum, const CcuLoopGroupCfg* cfg)
{
    PLF_CONFIG_INFO(PLF_DATA_OP, "[LoopGroupCreate] maxLoopNum=%u", maxLoopNum);
    if (group == nullptr || cfg == nullptr) {
        HCCL_ERROR("[CcuKernel::LoopGroupCreate] null pointer");
        return CcuResult::CCU_E_PTR;
    }
    if (CurrentBlock()->Type() == CcuRep::CcuRepType::LOOP_BLOCK) {
        HCCL_ERROR("[CcuKernel::LoopGroupCreate] cannot create loop group inside a loop body");
        return LatchBodyError(CcuResult::CCU_E_INTERNAL);
    }
    if (inFuncBody_) {
        HCCL_ERROR("[CcuKernel::LoopGroupCreate] cannot create loop group inside a func body");
        return LatchBodyError(CcuResult::CCU_E_INTERNAL);
    }

    // 按需扩 LoopEngine 池；池足够则复用低位 executorId，跨组共享。
    CCU_CHK_RET(EnsureLoopEnginePool(maxLoopNum));

    CcuLoopGroup handle = ++loopGroupHandleCounter_;

    LoopGroupDescriptor desc;
    desc.config = *cfg;
    desc.parallelVar = CreateVariable();
    desc.offsetVar = CreateVariable();
    desc.isVarBased = false;

    auto bundle = std::make_shared<CcuRep::CcuRepLoopGroupBundle>(insGenerator, *cfg, desc.parallelVar, desc.offsetVar);
    if (ccuVersion_ == CcuVersion::CCU_V2) {
        bundle->SetXnOffsetVar(CreateVariable());
    }
    desc.bundleRep = bundle;
    Append(bundle);

    loopGroupMap_[handle] = std::move(desc);
    *group = handle;
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuKernel::LoopGroupCreateFromVar(
    CcuLoopGroup* group, uint32_t maxLoopNum, CcuVariableHandle parallelVarHandle, CcuVariableHandle offsetVarHandle)
{
    PLF_CONFIG_INFO(
        PLF_DATA_OP, "[LoopGroupCreateFromVar] maxLoopNum=%u, parallelVarHandle=%llu, offsetVarHandle=%llu", maxLoopNum,
        parallelVarHandle, offsetVarHandle);
    if (group == nullptr) {
        HCCL_ERROR("[CcuKernel::LoopGroupCreateFromVar] null pointer for group");
        return CcuResult::CCU_E_PTR;
    }
    if (CurrentBlock()->Type() == CcuRep::CcuRepType::LOOP_BLOCK) {
        HCCL_ERROR("[CcuKernel::LoopGroupCreateFromVar] cannot create loop group inside a loop body");
        return LatchBodyError(CcuResult::CCU_E_INTERNAL);
    }
    if (inFuncBody_) {
        HCCL_ERROR("[CcuKernel::LoopGroupCreateFromVar] cannot create loop group inside a func body");
        return LatchBodyError(CcuResult::CCU_E_INTERNAL);
    }

    CCU_CHK_RET(EnsureLoopEnginePool(maxLoopNum));

    CcuRep::Variable* parallelVarPtr = nullptr;
    CcuRep::Variable* offsetVarPtr = nullptr;
    CCU_CHK_RET(GetVariableByHandle(parallelVarHandle, &parallelVarPtr));
    CCU_CHK_RET(GetVariableByHandle(offsetVarHandle, &offsetVarPtr));

    CcuLoopGroup handle = ++loopGroupHandleCounter_;

    LoopGroupDescriptor desc;
    desc.parallelVar = CcuRep::Variable(*parallelVarPtr);
    desc.offsetVar = CcuRep::Variable(*offsetVarPtr);
    desc.isVarBased = true;

    auto bundle = std::make_shared<CcuRep::CcuRepLoopGroupBundle>(insGenerator, desc.parallelVar, desc.offsetVar);
    if (ccuVersion_ == CcuVersion::CCU_V2) {
        bundle->SetCompatRemapVars(CreateVariable(), CreateVariable());
    }
    desc.bundleRep = bundle;
    Append(bundle);

    loopGroupMap_[handle] = std::move(desc);
    *group = handle;
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuKernel::LoopGroupCreateFromVarV2(
    CcuLoopGroup* group, uint32_t maxLoopNum, CcuVariableHandle parallelVarV2Handle,
    CcuVariableHandle offsetVarV2Handle, CcuVariableHandle varOffsetVarHandle)
{
    if (ccuVersion_ != CcuVersion::CCU_V2) {
        HCCL_ERROR("[CcuKernel::LoopGroupCreateFromVarV2] only supported on V2");
        return CcuResult::CCU_E_NOT_SUPPORT;
    }
    if (group == nullptr) {
        HCCL_ERROR("[CcuKernel::LoopGroupCreateFromVarV2] null pointer for group");
        return CcuResult::CCU_E_PTR;
    }
    if (CurrentBlock()->Type() == CcuRep::CcuRepType::LOOP_BLOCK) {
        HCCL_ERROR("[CcuKernel::LoopGroupCreateFromVarV2] cannot create loop group inside a loop body");
        return LatchBodyError(CcuResult::CCU_E_INTERNAL);
    }
    if (inFuncBody_) {
        HCCL_ERROR("[CcuKernel::LoopGroupCreateFromVarV2] cannot create loop group inside a func body");
        return LatchBodyError(CcuResult::CCU_E_INTERNAL);
    }

    CCU_CHK_RET(EnsureLoopEnginePool(maxLoopNum));

    CcuRep::Variable* parallelVarPtr = nullptr;
    CcuRep::Variable* offsetVarPtr = nullptr;
    CcuRep::Variable* xnOffsetVarPtr = nullptr;
    CCU_CHK_RET(GetVariableByHandle(parallelVarV2Handle, &parallelVarPtr));
    CCU_CHK_RET(GetVariableByHandle(offsetVarV2Handle, &offsetVarPtr));
    CCU_CHK_RET(GetVariableByHandle(varOffsetVarHandle, &xnOffsetVarPtr));

    CcuLoopGroup handle = ++loopGroupHandleCounter_;

    LoopGroupDescriptor desc;
    desc.parallelVar = CcuRep::Variable(*parallelVarPtr);
    desc.offsetVar = CcuRep::Variable(*offsetVarPtr);
    desc.xnOffsetVar = CcuRep::Variable(*xnOffsetVarPtr);
    desc.isVarBased = true;
    desc.isVersionV2 = true;

    auto bundle = std::make_shared<CcuRep::CcuRepLoopGroupBundle>(insGenerator, desc.parallelVar, desc.offsetVar);
    bundle->SetLayout(CcuRep::CcuRepLoopGroupBundle::Layout::VersionV2);
    bundle->SetXnOffsetVar(desc.xnOffsetVar);
    desc.bundleRep = bundle;
    Append(bundle);

    loopGroupMap_[handle] = std::move(desc);
    *group = handle;
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuKernel::LookupLoopGroupAndLoop(
    CcuLoopGroup group, CcuLoop loop, const char* fnName, const char* createFnName, LoopGroupDescriptor*& grpDesc,
    LoopDescriptor*& loopDesc, uint32_t& loopIdx)
{
    auto grpIt = loopGroupMap_.find(group);
    if (grpIt == loopGroupMap_.end()) {
        HCCL_ERROR("[CcuKernel::%s] invalid group handle %lu", fnName, group);
        return CcuResult::CCU_E_PARA;
    }
    grpDesc = &grpIt->second;

    auto loopIt = loopMap_.find(loop);
    if (loopIt == loopMap_.end()) {
        HCCL_ERROR("[CcuKernel::%s] invalid loop handle %lu", fnName, loop);
        return CcuResult::CCU_E_PARA;
    }
    loopDesc = &loopIt->second;

    if (!loopDesc->bodyDefined) {
        HCCL_ERROR("[CcuKernel::%s] loop %lu body not defined", fnName, loop);
        return CcuResult::CCU_E_INTERNAL; // CCU_E_LOOP_BODY_UNDEFINED
    }

    auto& loopEnginePool = res_.blockExecutor[0];
    loopIdx = grpDesc->loopCount;
    if (loopIdx >= loopEnginePool.size()) {
        HCCL_ERROR(
            "[CcuKernel::%s] loopEngine pool exhausted (pool size %zu, loopIdx %u). "
            "Pass a larger maxLoopNum to %s so the pool can be extended at create time.",
            fnName, loopEnginePool.size(), loopIdx, createFnName);
        return CcuResult::CCU_E_PARA;
    }
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuKernel::LoopGroupAddLoop(CcuLoopGroup group, CcuLoop loop, const CcuLoopCfg* cfg)
{
    PLF_CONFIG_INFO(PLF_DATA_OP, "[LoopGroupAddLoop] group=%llu, loop=%llu", group, loop);
    if (cfg == nullptr) {
        HCCL_ERROR("[CcuKernel::LoopGroupAddLoop] null pointer for cfg");
        return CcuResult::CCU_E_PTR;
    }
    LoopGroupDescriptor* grpDesc = nullptr;
    LoopDescriptor* loopDesc = nullptr;
    uint32_t loopIdx = 0;
    CCU_CHK_RET(
        LookupLoopGroupAndLoop(group, loop, "LoopGroupAddLoop", "CcuLoopGroupCreate", grpDesc, loopDesc, loopIdx));
    auto& loopEnginePool = res_.blockExecutor[0];

    grpDesc->loopCount++;
    grpDesc->totalLoopNum = grpDesc->loopCount;

    CcuRep::CcuRepLoopGroupBundle::LoopEntry entry;
    entry.config = *cfg;
    entry.executor = loopEnginePool[loopIdx];
    entry.repLoopBlock = loopDesc->repLoopBlock;
    entry.loopParamVar = CreateVariable();
    entry.layout = CcuRep::CcuRepLoopGroupBundle::Layout::Config;
    if (ccuVersion_ == CcuVersion::CCU_V2) {
        entry.iterNumVar = CreateVariable();
        entry.addrOffsetVar = CreateVariable();
        entry.ctxIdVar = CreateVariable();
    }

    auto bundle = std::static_pointer_cast<CcuRep::CcuRepLoopGroupBundle>(grpDesc->bundleRep);
    bundle->AddLoop(entry);

    if (!grpDesc->isVarBased) {
        bundle->SetRepeatLoopIdx(grpDesc->config.cloneLoopOffset);
        bundle->SetTotalLoopNum(grpDesc->totalLoopNum);
    }

    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuKernel::LoopGroupAddLoopFromVar(CcuLoopGroup group, CcuLoop loop, CcuVariableHandle loopParamVarHandle)
{
    PLF_CONFIG_INFO(
        PLF_DATA_OP, "[LoopGroupAddLoopFromVar] group=%llu, loop=%llu, loopParamVarHandle=%llu", group, loop,
        loopParamVarHandle);
    LoopGroupDescriptor* grpDesc = nullptr;
    LoopDescriptor* loopDesc = nullptr;
    uint32_t loopIdx = 0;
    CCU_CHK_RET(LookupLoopGroupAndLoop(
        group, loop, "LoopGroupAddLoopFromVar", "CcuLoopGroupCreateFromVar", grpDesc, loopDesc, loopIdx));
    auto& loopEnginePool = res_.blockExecutor[0];

    CcuRep::Variable* loopParamVarPtr = nullptr;
    CCU_CHK_RET(GetVariableByHandle(loopParamVarHandle, &loopParamVarPtr));

    CcuRep::CcuRepLoopGroupBundle::LoopEntry entry;
    entry.executor = loopEnginePool[loopIdx];
    entry.repLoopBlock = loopDesc->repLoopBlock;
    entry.loopParamVar = CcuRep::Variable(*loopParamVarPtr);
    entry.layout = CcuRep::CcuRepLoopGroupBundle::Layout::PackedVar;
    if (ccuVersion_ == CcuVersion::CCU_V2) {
        entry.iterNumVar = CreateVariable();
        entry.addrOffsetVar = CreateVariable();
        entry.ctxIdVar = CreateVariable();
    }

    auto bundle = std::static_pointer_cast<CcuRep::CcuRepLoopGroupBundle>(grpDesc->bundleRep);
    bundle->AddLoop(entry);

    // 计数放在入 bundle 之后统一更新：totalLoopNum 仅供 config 组同步编码使用，与 AddLoop 无先后依赖。
    grpDesc->loopCount++;
    grpDesc->totalLoopNum = grpDesc->loopCount;
    if (!grpDesc->isVarBased) {
        bundle->SetRepeatLoopIdx(grpDesc->config.cloneLoopOffset);
        bundle->SetTotalLoopNum(grpDesc->totalLoopNum);
    }

    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuKernel::LoopGroupAddLoopFromVarV2(
    CcuLoopGroup group, CcuLoop loop, CcuVariableHandle iterNumVarHandle, CcuVariableHandle addrOffsetVarHandle,
    CcuVariableHandle ctxIdVarHandle)
{
    if (ccuVersion_ != CcuVersion::CCU_V2) {
        HCCL_ERROR("[CcuKernel::LoopGroupAddLoopFromVarV2] only supported on V2");
        return CcuResult::CCU_E_NOT_SUPPORT;
    }

    LoopGroupDescriptor* grpDesc = nullptr;
    LoopDescriptor* loopDesc = nullptr;
    uint32_t loopIdx = 0;
    CCU_CHK_RET(LookupLoopGroupAndLoop(
        group, loop, "LoopGroupAddLoopFromVarV2", "CcuLoopGroupCreateFromVarV2", grpDesc, loopDesc, loopIdx));
    auto& loopEnginePool = res_.blockExecutor[0];

    CcuRep::Variable* iterNumVarPtr = nullptr;
    CcuRep::Variable* addrOffsetVarPtr = nullptr;
    CcuRep::Variable* ctxIdVarPtr = nullptr;
    CCU_CHK_RET(GetVariableByHandle(iterNumVarHandle, &iterNumVarPtr));
    CCU_CHK_RET(GetVariableByHandle(addrOffsetVarHandle, &addrOffsetVarPtr));
    CCU_CHK_RET(GetVariableByHandle(ctxIdVarHandle, &ctxIdVarPtr));

    grpDesc->loopCount++;
    grpDesc->totalLoopNum = grpDesc->loopCount;

    VersionV2LoopRecord record;
    record.iterNumVar = CcuRep::Variable(*iterNumVarPtr);
    record.addrOffsetVar = CcuRep::Variable(*addrOffsetVarPtr);
    record.ctxIdVar = CcuRep::Variable(*ctxIdVarPtr);
    grpDesc->versionV2Loops.push_back(record);

    CcuRep::CcuRepLoopGroupBundle::LoopEntry entry;
    entry.executor = loopEnginePool[loopIdx];
    entry.repLoopBlock = loopDesc->repLoopBlock;
    entry.iterNumVar = CcuRep::Variable(*iterNumVarPtr);
    entry.addrOffsetVar = CcuRep::Variable(*addrOffsetVarPtr);
    entry.ctxIdVar = CcuRep::Variable(*ctxIdVarPtr);
    entry.layout = CcuRep::CcuRepLoopGroupBundle::Layout::VersionV2;

    auto bundle = std::static_pointer_cast<CcuRep::CcuRepLoopGroupBundle>(grpDesc->bundleRep);
    bundle->AddLoop(entry);

    if (!grpDesc->isVarBased) {
        bundle->SetRepeatLoopIdx(grpDesc->config.cloneLoopOffset);
        bundle->SetTotalLoopNum(grpDesc->totalLoopNum);
    }

    return CcuResult::CCU_SUCCESS;
}

void CcuKernel::SetInstrId(uint32_t instrId) { instrInfo_.startInstrId = instrId; }

uint32_t CcuKernel::GetInstrId() const { return instrInfo_.startInstrId; }

uint32_t CcuKernel::GetInstrCount()
{
    uint32_t instrCount = 0;
    for (const auto& rep : GetRepSequence()) {
        instrCount += rep->InstrCount();
    }
    instrInfo_.instrCount = instrCount;
    HCCL_INFO("Kernel inst %u", instrCount);
    return instrCount;
}

namespace {
    // 识别"会翻译出 waitCKEId!=0 && clearType=1 的 set/clearCKE (即后端优化 ExtractOperandsV2 眼里
    // 的 CKE 读者+写者)"的 rep. 后端优化 (CkeOnly) 会为每个此类 CKE 读者最多补 (CCU_CKE_RAW_LATENCY-1)
    // 条 NOP, 故每个此类 rep 都要按 CCU_CKE_RAW_LATENCY 预留指令空间, 保证优化后指令数不越界.
    // 下列每类 rep 恰好翻译出 1 条此形态指令 (RECORD_SHARED_NOTIFY 仅跨 die 分支出, 本 die 分支
    // waitCKEId=0 不是读者; 因预留为静态统计无法预知 die, 保守全计, 顶多多留 L 条, 无害), 故计 1 次即可:
    //   - 三种 wait 类 rep: waitCKEId=事件/通知/信号 CKE, profiling->setcke, 非 profiling->clearcke;
    //   - LOAD / LOAD_VAR / STORE / STORE_VAR: 搬运后一条 SetCKE(waitCKEId=commSignal) 等搬运完成;
    //   - RECORD_SHARED_NOTIFY (跨 die): store 对端 cke 后一条 SetCKE(waitCKEId=commSignal) 等 store 完成.
    // commSignal 在 kernel/die 级共享同一 CKE, 多个搬运类 rep 的 SetCKE 会构成写后读链, 触发补 NOP,
    // 故必须与三种 wait 类 rep 同等预留. 详见 ccu_ins_generator_v2.cc 各 Translate 与 extract_operands.cc.
    bool IsCkeWaitRep(const std::shared_ptr<CcuRep::CcuRepBase>& rep)
    {
        if (rep == nullptr) {
            return false;
        }
        switch (rep->Type()) {
            case CcuRep::CcuRepType::LOC_WAIT_EVENT:
            case CcuRep::CcuRepType::LOC_WAIT_NOTIFY:
            case CcuRep::CcuRepType::REM_WAIT_SEM:
            case CcuRep::CcuRepType::LOAD:
            case CcuRep::CcuRepType::LOAD_VAR:
            case CcuRep::CcuRepType::STORE:
            case CcuRep::CcuRepType::STORE_VAR:
            case CcuRep::CcuRepType::RECORD_SHARED_NOTIFY:
                return true;
            default:
                return false;
        }
    }

    // 与 CcuKernelMgr::PrepareConstValueResources 一致地下钻 block 子 rep, 统计其中会翻译出
    // waitCKEId!=0 && clearType=1 的 set/clearCKE (CKE 读者+写者) 的 rep 个数 (判定见 IsCkeWaitRep).
    // 只下钻一层是充分的: 表示层禁止 block 嵌套 —— CcuKernel::LoopCreate / FuncBlockBegin /
    // FuncBlockLookup 在 CurrentBlock() 为 LOOP_BLOCK 或 inFuncBody_ 时直接报错 (见 ccu_kernel.cc),
    // 故 LOOP_BLOCK / FUNC_BLOCK 内不可能再出现 block 类型子 rep, 无需递归下钻.
    // 若后续放开 block 嵌套, 此处 (与 PrepareConstValueResources) 需同步改为递归统计, 否则会漏计.
    uint32_t CountCkeWaitRepInBlock(const std::shared_ptr<CcuRep::CcuRepBase>& rep)
    {
        const auto repType = rep->Type();
        if (repType != CcuRep::CcuRepType::BLOCK && repType != CcuRep::CcuRepType::FUNC_BLOCK
            && repType != CcuRep::CcuRepType::LOOP_BLOCK) {
            return 0;
        }
        auto* blockPtr = static_cast<CcuRep::CcuRepBlock*>(rep.get());
        if (blockPtr == nullptr) {
            return 0;
        }
        uint32_t count = 0;
        for (const auto& child : blockPtr->GetReps()) {
            if (IsCkeWaitRep(child)) {
                count++;
            }
        }
        return count;
    }

    // 识别"会翻译出 LoadX/StoreX/ClearX (half-rtt 特殊指令) 的 rep". 后端优化 (CkeOnly) 对该指令双向补
    // NOP: 作为写者其后读者、作为读者其前写者各最多补 (CCU_XN_RAW_LATENCY - 1) 条, 故每个此类 rep 都要按
    // 2 * CCU_XN_RAW_LATENCY 预留指令空间, 保证优化后指令数不越界. 各 rep 恰出 1 条对应微指令 (见
    // ccu_ins_generator_v2.cc):
    //   LOAD_ADD_IMM   -> 1 条 LoadX;
    //   STORE_ADD_IMM  -> 1 条 StoreX;
    //   CASC_CNT_CLEAR -> 1 条 ClearX.
    // CASC_CNT_CLEAR 的 ClearX 同片自依赖 (两条相同区间 ClearX) 也会触发后端补 NOP
    // (见 instruction_scheduler.cc EarliestXnIssueCycle 的 ClearX 区间 key 探测与 UT
    // SchedulerCkeOnly_ClearXSameRangeInsertsXnNops), 故必须与 LoadX/StoreX 同等预留, 否则预留与
    // 实际插入漂移会导致越界.
    bool IsLdStXRep(const std::shared_ptr<CcuRep::CcuRepBase>& rep)
    {
        if (rep == nullptr) {
            return false;
        }
        switch (rep->Type()) {
            case CcuRep::CcuRepType::LOAD_ADD_IMM:
            case CcuRep::CcuRepType::STORE_ADD_IMM:
            case CcuRep::CcuRepType::CASC_CNT_CLEAR:
                return true;
            default:
                return false;
        }
    }

    // 与 CountCkeWaitRepInBlock 一致地下钻一层 block 子 rep, 统计其中会翻译出 LoadX/StoreX/ClearX 的 rep 个数.
    uint32_t CountLdStXRepInBlock(const std::shared_ptr<CcuRep::CcuRepBase>& rep)
    {
        const auto repType = rep->Type();
        if (repType != CcuRep::CcuRepType::BLOCK && repType != CcuRep::CcuRepType::FUNC_BLOCK
            && repType != CcuRep::CcuRepType::LOOP_BLOCK) {
            return 0;
        }
        auto* blockPtr = static_cast<CcuRep::CcuRepBlock*>(rep.get());
        if (blockPtr == nullptr) {
            return 0;
        }
        uint32_t count = 0;
        for (const auto& child : blockPtr->GetReps()) {
            if (IsLdStXRep(child)) {
                count++;
            }
        }
        return count;
    }
} // namespace

// 统计当前 kernel 中"需要按 cke 写后读 latency 补 NOP"的 rep 个数 (含 block 子 rep).
// 凡会翻译出 waitCKEId!=0 && clearType=1 的 set/clearCKE (后端优化眼里的 CKE 读者) 的 rep 均计入
// (判定集合见 IsCkeWaitRep: 三种 wait 类 + LOAD/LOAD_VAR/STORE/STORE_VAR/RECORD_SHARED_NOTIFY),
// 每个此类 rep 恰出 1 条此形态指令, 后端优化最多为该读者补 (CCU_CKE_RAW_LATENCY - 1) 条 NOP,
// 上层据此为每个此类 rep 预留同等指令空间, 保证优化后指令数不越界.
uint32_t CcuKernel::GetRepNeedToAddLatency() const
{
    // cke 写后读补 NOP 与指令空间预留只属于 A6(CCU_V2) 后端优化; A5(CCU_V1) 不跑后端优化,
    // 不做任何预留, 直接返回 0, 避免影响 A5 的申请/释放口径.
    if (ccuVersion_ != CcuVersion::CCU_V2) {
        return 0;
    }
    uint32_t count = 0;
    for (const auto& rep : GetRepSequence()) {
        if (rep == nullptr) {
            continue;
        }
        if (IsCkeWaitRep(rep)) {
            count++;
        }
        count += CountCkeWaitRepInBlock(rep);
    }
    HCCL_INFO("[CcuKernel] cke wait rep count %u (reserve %u instrs)", count, count * CcuRep::CCU_CKE_RAW_LATENCY);
    return count;
}

// 统计当前 kernel 中"会翻译出 LoadX/StoreX/ClearX (half-rtt 特殊指令) 的 rep 个数" (含 block 子 rep),
// 每个此类 rep 恰出 1 条对应微指令. 后端优化对该指令存在两个方向的写后读补 NOP:
//   (1) 该 lsx 写 xn/array -> 后续任意指令读 (作为写者, 其后读者最多补 CCU_XN_RAW_LATENCY - 1 条);
//   (2) 任意前序指令写 xn/array -> 本 lsx/clearx 读 (硬件对 lsx/clearx 读操作数 interlock 失效,
//       作为读者, 其前最多补 CCU_XN_RAW_LATENCY - 1 条).
// 单个 rep 前后两侧最坏合计补 2*(CCU_XN_RAW_LATENCY - 1) 条, 故上层据此为每个此类 rep 预留
// 2 * CCU_XN_RAW_LATENCY 条指令空间 (预留倍数见 ccu_kernel_mgr.cc ComputeKernelInstrRegionSize).
// 本函数只返回 rep 个数, latency 倍数由预留计算处统一乘. 判定集合见 IsLdStXRep: LOAD_ADD_IMM /
// STORE_ADD_IMM / CASC_CNT_CLEAR.
uint32_t CcuKernel::GetLsxRepReserveCount() const
{
    // XN 写后读补 NOP 与指令空间预留只属于 A6(CCU_V2) 后端优化; A5(CCU_V1) 不跑后端优化,
    // 不做任何预留, 直接返回 0, 避免影响 A5 的申请/释放口径.
    if (ccuVersion_ != CcuVersion::CCU_V2) {
        return 0;
    }
    uint32_t count = 0;
    for (const auto& rep : const_cast<CcuKernel*>(this)->GetRepSequence()) {
        if (rep == nullptr) {
            continue;
        }
        if (IsLdStXRep(rep)) {
            count++;
        }
        count += CountLdStXRepInBlock(rep);
    }
    HCCL_INFO("[CcuKernel] lsx rep count %u (reserve %u instrs)", count, count * 2u * CcuRep::CCU_XN_RAW_LATENCY);
    return count;
}

void CcuKernel::SetCcuInstrInfo(const CcuRep::CcuInstrInfo& instrInfo) { this->instrInfo_ = instrInfo; }

CcuRep::Variable CcuKernel::CreateVariable()
{
    const auto& var = CreateBlockResAssist(1, res_.continuousVariable);
    return var[0];
}

CcuRep::Variable CcuKernel::CreateExpectVar()
{
    // v2(A6) 的 jump 只支持 var-var 比较，需要真实 XN 承载立即数；
    // v1(A5) 的 jump 直接支持 var-imm，expectVar 不参与翻译，
    // 用不进 res_ 账本的壳 Variable 即可，避免 v1 资源虚高。
    if (ccuVersion_ == CcuVersion::CCU_V2) {
        return CreateVariable();
    }
    return CcuRep::Variable(this);
}

CcuRep::Variable CcuKernel::CreateJumpTargetVar()
{
    if (ccuVersion_ == CcuVersion::CCU_V1) {
        // A5(CCU_V1): 没有指令重排，可复用同一个 Xn 存放跳转目标地址。targetVar 仅用于 LoadImdToXn + 紧接 Jump。
        if (!sharedJumpTargetVar_) {
            sharedJumpTargetVar_ = std::make_unique<CcuRep::Variable>(CreateVariable());
        }
        return *sharedJumpTargetVar_;
    }
    return CcuRep::CreateVariable(this);
}

CcuRep::Address CcuKernel::CreateAddress()
{
    if (ccuVersion_ == CcuVersion::CCU_V2) {
        // A6创建Address时，需要添加到Variable的列表中，但是仍以Address返回
        return CcuRep::Address(CreateResAssist(res_.continuousVariable));
    }
    return CreateResAssist(res_.blockAddress);
}

CcuRep::LocalNotify CcuKernel::CreateLocalNotify() { return CreateResAssist(res_.localNotify); }

CcuRep::CompletedEvent CcuKernel::CreateCompletedEvent() { return CreateResAssist(res_.blockCompletedEvent); }

CcuRep::CcuBuf CcuKernel::CreateCcuBuf()
{
    const auto& buffer = CreateBlockResAssist(1, res_.blockCcubufs);
    return buffer[0];
}

CcuRep::Executor CcuKernel::CreateExecutor() { return CreateResAssist(res_.blockExecutor); }

CcuRep::LocalAddr CcuKernel::CreateLocalAddr() { return CcuRep::LocalAddr(CreateAddress(), CreateVariable()); }

CcuRep::RemoteAddr CcuKernel::CreateRemoteAddr() { return CcuRep::RemoteAddr(CreateAddress(), CreateVariable()); }

CcuRep::RemoteAddr CcuKernel::GetRemoteAddr(const ChannelHandle channel, uint32_t index)
{
    (void)index;
    channels_.insert(channel);
    auto mem = CcuRep::RemoteAddr(CreateAddress(), CreateVariable());
    Append(std::make_shared<CcuRep::CcuRepRemMem>(insGenerator, channel, mem));
    return mem;
}

CcuRep::LocalAddr CcuKernel::CreateLocalAddr(const CcuRep::Variable& token)
{
    return CcuRep::LocalAddr(CreateAddress(), token);
}

HcclResult CcuKernel::CreateBlockCcuBuf(const uint32_t count, CcuRep::CcuBuf* ccuBufs)
{
    CHK_PTR_NULL(ccuBufs);
    auto resources = CreateBlockResAssist(count, res_.blockCcubufs);

    for (uint32_t i = 0; i < count; i++) {
        ccuBufs[i] = resources[i]; // 拷贝虚拟资源，通过shared_ptr链接到物理资源
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuKernel::CreateBlockExecutor(const uint32_t count, CcuRep::Executor* ccuExes)
{
    CHK_PTR_NULL(ccuExes);
    auto resources = CreateBlockResAssist(count, res_.blockExecutor);

    for (uint32_t i = 0; i < count; i++) {
        ccuExes[i] = resources[i]; // 拷贝虚拟资源，通过shared_ptr链接到物理资源
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuKernel::CreateBlockCompletedEvent(const uint32_t count, CcuRep::CompletedEvent* ccuEvents)
{
    CHK_PTR_NULL(ccuEvents);
    auto resources = CreateBlockResAssist(count, res_.blockCompletedEvent);

    for (uint32_t i = 0; i < count; i++) {
        ccuEvents[i] = resources[i]; // 拷贝虚拟资源，通过shared_ptr链接到物理资源
    }

    return HcclResult::HCCL_SUCCESS;
}

void CcuKernel::SetResRepository(const CcuResRepository& resRepo) { resRepo_ = resRepo; }

void CcuKernel::SetCascCntBlock(const std::unordered_map<HcommCcuCascCntHandle, CntXnBlock>& cascCntBlocks)
{
    std::lock_guard<std::mutex> lock(cascCntBlockMutex_);
    cascCntBlocks_ = cascCntBlocks;
}

HcclResult CcuKernel::GetCascCntBlock(HcommCcuCascCntHandle cntHandle, CntXnBlock& cascCntBlock)
{
    std::lock_guard<std::mutex> lock(cascCntBlockMutex_);
    auto cascCntBlockIter = cascCntBlocks_.find(cntHandle);
    if (cascCntBlockIter == cascCntBlocks_.end()) {
        HCCL_ERROR("Invalid cascCntHandle(%llu)", static_cast<unsigned long long>(cntHandle));
        return HCCL_E_NOT_FOUND;
    }
    cascCntBlock = cascCntBlockIter->second;
    return HcclResult::HCCL_SUCCESS;
}

CcuResRepository& CcuKernel::GetResRepository() { return resRepo_; }

CcuSharedResource& CcuKernel::GetExportedRes() { return exportedRes_; }

CcuSharedResource& CcuKernel::GetImportedRes() { return importedRes_; }

static HcclResult GetArgIndex(
    const std::unordered_map<uint16_t, uint16_t>& varId2VarIdMap,
    const std::unordered_map<uint16_t, uint32_t>& varId2ArgIndexMap, const uint64_t* taskArgs, uint32_t argSize,
    uint16_t varId, uint64_t& argIndex)
{
    HCCL_INFO("[GetArgIndex] Enter varId(%u)", varId);
    auto item = varId2ArgIndexMap.find(varId);
    if (item == varId2ArgIndexMap.end()) {
        uint16_t oriVarId = varId;
        auto iter = varId2VarIdMap.find(varId);
        while (iter != varId2VarIdMap.end()) { // 循环查找中间assign Rep，找到起始varId
            oriVarId = iter->second;
            iter = varId2VarIdMap.find(oriVarId);
        }
        if (oriVarId != varId) { // 起始varId预期通过LoadArg赋值
            item = varId2ArgIndexMap.find(oriVarId);
            if (item == varId2ArgIndexMap.end()) {
                HCCL_ERROR("[%s]fail, Invalid goSize variable id(%u), oriVarId = %u", __func__, varId, oriVarId);
                return HCCL_E_PARA;
            }
        } else {
            HCCL_ERROR("[%s]fail, Invalid goSize variable id(%u)", __func__, varId);
            return HCCL_E_PARA;
        }
    }
    HCCL_INFO("[GetArgIndex] find end");
    if (item->second >= argSize) {
        HCCL_ERROR("Invalid goSize variable index(%u).", item->second);
        return HCCL_E_PARA;
    }
    HCCL_INFO(
        "GetArgIndex success: varId(%u) varId2VarIdMapSize(%u) varId2ArgIndexMapSize(%u) taskArgsSize(%u)", varId,
        varId2VarIdMap.size(), varId2ArgIndexMap.size(), argSize);
    argIndex = taskArgs[item->second];
    return HCCL_SUCCESS;
}

void DumpCcuProfilingInfo(const std::vector<CcuProfilingInfo>& ccuProfilingInfo)
{
    auto dumpLinkInfo = [](const CcuProfilingInfo& info) -> void {
        for (int i = 0; i < CCU_MAX_CHANNEL_NUM; i++) {
            if (info.channelId[i] == INVALID_VALUE_CHANNELID) {
                continue;
            }
            HCCL_INFO("channelId(%u), remoteRankId(%u).", info.channelId[i], info.remoteRankId[i]);
        }
    };

    for (const auto& profInfo : ccuProfilingInfo) {
        if (profInfo.type == static_cast<uint8_t>(CcuProfilinType::CCU_TASK_PROFILING)) {
            HCCL_INFO(
                "Dump CCU Profiling Info:SQE Profiling Info: ctxSignature(%s), "
                "dieId(%d), missionId(%d), instrId(%d).",
                profInfo.name.c_str(), static_cast<int>(profInfo.dieId), static_cast<int>(profInfo.missionId),
                static_cast<int>(profInfo.instrId));
        } else if (profInfo.type == static_cast<uint8_t>(CcuProfilinType::CCU_WAITCKE_PROFILING)) {
            HCCL_INFO(
                "Microcode WaitCKE Profiling Info: name(%s), "
                "dieId(%d), missionId(%d), instrId(%d), ckeId(%u), mask(%u).",
                profInfo.name.c_str(), static_cast<int>(profInfo.dieId), static_cast<int>(profInfo.missionId),
                static_cast<int>(profInfo.instrId), profInfo.ckeId, profInfo.mask);
            dumpLinkInfo(profInfo);
        } else if (profInfo.type == static_cast<uint8_t>(CcuProfilinType::CCU_LOOPGROUP_PROFILING)) {
            HCCL_INFO(
                "Microcode LoopGroup Profiling Info: name(%s), "
                "dieId(%d), missionId(%d), instrId(%d), reduceOpType(%d), inputDataType(%d), "
                "outputDataType(%d), dataSize(%llu).",
                profInfo.name.c_str(), static_cast<int>(profInfo.dieId), static_cast<int>(profInfo.missionId),
                static_cast<int>(profInfo.instrId), static_cast<int>(profInfo.reduceOpType),
                static_cast<int>(profInfo.inputDataType), static_cast<int>(profInfo.outputDataType), profInfo.dataSize);
            dumpLinkInfo(profInfo);
        }
    }
}

constexpr uint64_t SetBits(uint16_t end) { return ((uint64_t(1) << (end + 1)) - uint64_t(1)); }

static uint16_t ParseRepeatNumFromParallelParam(uint64_t parallelParam)
{
    constexpr uint16_t repeatBitNum = 7;       // 7： repeat num 占 7 bits
    constexpr uint16_t repeatNumShiftBit = 55; // 55： repeat num占[61:55]位置
    return (parallelParam >> repeatNumShiftBit) & SetBits(repeatBitNum);
}

HcclResult CcuKernel::CollectSqeAndWaitCkeProfilingInfo()
{
    auto& ccuProfilingCache = GetProfilingInfo();
    uint32_t count{0};
    HCCL_INFO("[GetCcuProfilingInfo] Process sqe&waitcke profiling info start.");
    for (auto& profInfo : ccuProfilingCache) {
        profInfo.missionId = GetMissionId();
        if (profInfo.type == static_cast<uint8_t>(hcomm::CcuProfilinType::CCU_TASK_PROFILING)) {
            profInfo.instrId = GetInstrId();
            allCcuProfilingInfos_.push_back(profInfo);
            continue;
        }
        if (count >= GetWaiteCkeProfilingReps().size()) {
            HCCL_ERROR(
                "count[%u] out of range[0, %u], cache size(%u).", count, GetWaiteCkeProfilingReps().size(),
                ccuProfilingCache.size());
            return HCCL_E_INTERNAL;
        }
        auto waitCkeRep = GetWaiteCkeProfilingReps()[count];
        profInfo.instrId = waitCkeRep->StartInstrId();
        if (profInfo.ckeId == INVALID_CKE_ID) { // localWait Rep
            if (waitCkeRep.get() == nullptr) {
                HCCL_ERROR("[GetCcuProfilingInfo] localWaitRep is nullptr.");
                return HCCL_E_PTR;
            }
            profInfo.ckeId = waitCkeRep->GetId();
            HCCL_INFO("[CcuKernel][GetCcuProfilingInfo] waitcke[%u]", profInfo.ckeId);
        }
        allCcuProfilingInfos_.push_back(profInfo);
        count++;
    }
    return HCCL_SUCCESS;
}

HcclResult CcuKernel::BuildLoopGroupVarIdMaps(
    std::unordered_map<uint16_t, uint32_t>& varId2ArgIndexMap, std::unordered_map<uint16_t, uint16_t>& varId2VarIdMap)
{
    auto& lgProfInfo = GetLGProfilingInfo();
    HCCL_INFO("[GetCcuProfilingInfo] create varId2ArgIndexMap start. size=%lu", lgProfInfo.loadRep2ArgIdxMap.size());
    for (auto& iter : lgProfInfo.loadRep2ArgIdxMap) {
        if (iter.first.get() == nullptr) {
            HCCL_ERROR("[GetCcuProfilingInfo] loadRep is nullptr.");
            return HCCL_E_PTR;
        }
        auto loadRep = dynamic_cast<CcuRep::CcuRepLoadArg*>(iter.first.get());
        varId2ArgIndexMap[loadRep->GetVarId()] = iter.second;
    }

    HCCL_INFO("[GetCcuProfilingInfo] create varId2VarIdMap start. size=%lu", lgProfInfo.assignProfilingReps.size());
    for (auto& iter : lgProfInfo.assignProfilingReps) {
        if (iter.get() == nullptr) {
            HCCL_ERROR("[GetCcuProfilingInfo] assignRep is nullptr.");
            return HCCL_E_PTR;
        }
        auto assignRep = dynamic_cast<CcuRep::CcuRepAssign*>(iter.get());
        varId2VarIdMap[assignRep->GetVarB().Id()] = assignRep->GetVarA().Id();
    }
    return HCCL_SUCCESS;
}

HcclResult CcuKernel::CollectLoopGroupProfilingInfo(
    const uint64_t* taskArgs, uint32_t argSize, const std::unordered_map<uint16_t, uint32_t>& varId2ArgIndexMap,
    const std::unordered_map<uint16_t, uint16_t>& varId2VarIdMap)
{
    auto& lgProfInfo = GetLGProfilingInfo();
    HCCL_INFO(
        "[GetCcuProfilingInfo] process loop group profiling start: "
        "lgsize(%lu), goSize(%lu)",
        lgProfInfo.lgProfilingReps.size(), groupOpSizeInfo_.size());
    // lgProfilingReps、groupOpSizeInfo_、ccuProfilingInfos 设计上同步 push；
    // 但 CcuRepContext::AddProfiling 可能只 push ccuProfilingInfos 而不 push lgProfilingReps，
    // 取三者最小值作为循环安全上界，防止任一容器较短时下标越界。
    size_t repSize = lgProfInfo.lgProfilingReps.size();
    size_t goSizeNum = groupOpSizeInfo_.size();
    size_t profSize = lgProfInfo.ccuProfilingInfos.size();
    size_t safeSize = (repSize < goSizeNum) ? repSize : goSizeNum;
    safeSize = (profSize < safeSize) ? profSize : safeSize;
    if (safeSize != repSize) {
        HCCL_WARNING(
            "[CollectLoopGroupProfilingInfo] size mismatch: lgReps[%zu], goSize[%zu], ccuProf[%zu],"
            " use safeSize[%zu].",
            repSize, goSizeNum, profSize, safeSize);
    }
    for (uint32_t i = 0; i < safeSize; i += 2) { // 2: 一个goSize对应一个CcuProfilingInfo，对应1个loopGroup Rep
        CHK_RET(CollectSingleLoopGroupProfiling(i, repSize, taskArgs, argSize, varId2ArgIndexMap, varId2VarIdMap));
    }
    return HCCL_SUCCESS;
}

HcclResult CcuKernel::CollectSingleLoopGroupProfiling(
    uint32_t i, size_t repSize, const uint64_t* taskArgs, uint32_t argSize,
    const std::unordered_map<uint16_t, uint32_t>& varId2ArgIndexMap,
    const std::unordered_map<uint16_t, uint16_t>& varId2VarIdMap)
{
    if (argSize == 0 || varId2ArgIndexMap.empty()) {
        return HCCL_SUCCESS;
    }
    auto& lgProfInfo = GetLGProfilingInfo();
    uint64_t loopParam{0};
    CHK_RET(
        GetArgIndex(varId2VarIdMap, varId2ArgIndexMap, taskArgs, argSize, groupOpSizeInfo_[i].loopParamId, loopParam));
    uint64_t parallelParam{0};
    CHK_RET(GetArgIndex(
        varId2VarIdMap, varId2ArgIndexMap, taskArgs, argSize, groupOpSizeInfo_[i].parallelParamId, parallelParam));
    HCCL_INFO(
        "Collect loopgroup profiling info: repSize[%u], index[%u],"
        "loopParam[%llu], parallelParam[%llu].",
        lgProfInfo.lgProfilingReps.size(), i, loopParam, parallelParam);

    if (loopParam != 0) {
        lgProfInfo.ccuProfilingInfos[i].dataSize = loopParam * moConfig_.loopCount * moConfig_.memSlice;
        lgProfInfo.ccuProfilingInfos[i].instrId
            = dynamic_cast<CcuRep::CcuRepLoopGroupBundle*>(lgProfInfo.lgProfilingReps[i].get())->StartInstrId();
        allCcuProfilingInfos_.push_back(lgProfInfo.ccuProfilingInfos[i]);
    }

    if (parallelParam != 0) {
        HCCL_INFO("[GetCcuProfilingInfo] collect lg, residual start i=%lu", i);
        uint64_t residual{0};
        CHK_RET(GetArgIndex(
            varId2VarIdMap, varId2ArgIndexMap, taskArgs, argSize, groupOpSizeInfo_[i].residualId, residual));
        uint64_t repeatNum = ParseRepeatNumFromParallelParam(parallelParam);
        lgProfInfo.ccuProfilingInfos[i].dataSize = repeatNum * moConfig_.memSlice + residual;
        // rep[i] 对应 loopParam 分支，rep[i+1] 对应 parallelParam 分支；尾项无配对 rep 时跳过避免越界
        if (i + 1 >= repSize) {
            HCCL_WARNING(
                "[CollectLoopGroupProfilingInfo] parallelParam != 0 but no paired rep,"
                " index(%u), repSize(%zu), skip parallelParam profiling.",
                i, repSize);
            return HCCL_SUCCESS;
        }
        lgProfInfo.ccuProfilingInfos[i].instrId
            = dynamic_cast<CcuRep::CcuRepLoopGroupBundle*>(lgProfInfo.lgProfilingReps[i + 1].get())->StartInstrId();
        allCcuProfilingInfos_.push_back(lgProfInfo.ccuProfilingInfos[i]);
    }
    return HCCL_SUCCESS;
}

/*
 * variable/maskSignal等资源变量Id，一定要在获取ccu profiling时才获取；
 * 原因：在创建context Rep时，其资源Id属于虚拟资源；翻译时，才会绑定固定的物理资源。
 */
HcclResult CcuKernel::GetCcuProfilingInfo(
    const uint64_t* taskArgs, uint32_t argSize, std::vector<CcuProfilingInfo>& allCcuProfilingInfo)
{
    HCCL_INFO("[GetCcuProfilingInfo] Enter.");
    allCcuProfilingInfos_.clear();

    CHK_RET(CollectSqeAndWaitCkeProfilingInfo());

    std::unordered_map<uint16_t, uint32_t> varId2ArgIndexMap;
    std::unordered_map<uint16_t, uint16_t> varId2VarIdMap;
    CHK_RET(BuildLoopGroupVarIdMaps(varId2ArgIndexMap, varId2VarIdMap));

    CHK_RET(CollectLoopGroupProfilingInfo(taskArgs, argSize, varId2ArgIndexMap, varId2VarIdMap));

    DumpCcuProfilingInfo(allCcuProfilingInfos_);
    allCcuProfilingInfo = allCcuProfilingInfos_;
    return HCCL_SUCCESS;
}

HcclResult CcuKernel::AddProfilingInfo(
    const ChannelHandle* channels, uint32_t channelNum, HcclDataType dataType, HcclDataType outputDataType,
    HcclReduceOp opType, const std::string& opName)
{
    CHK_PTR_NULL(channels);
    ccuProfilingInfoCache.type = static_cast<uint8_t>(CcuProfilinType::CCU_LOOPGROUP_PROFILING);
    ccuProfilingInfoCache.name = opName;
    ccuProfilingInfoCache.reduceOpType = opType;
    ccuProfilingInfoCache.inputDataType = dataType;
    ccuProfilingInfoCache.outputDataType = outputDataType;
    ccuProfilingInfoCache.missionId = GetMissionId();

    CHK_SAFETY_FUNC_RET(memset_s(
        ccuProfilingInfoCache.channelId, sizeof(ccuProfilingInfoCache.channelId), INVALID_VALUE_CHANNELID,
        sizeof(ccuProfilingInfoCache.channelId)));
    for (uint32_t i = 0; i < channelNum; i++) {
        void* channelPtr{nullptr};
        CHK_RET(static_cast<HcclResult>(HcommChannelGet(channels[i], &channelPtr)));
        auto* channelImpl = dynamic_cast<CcuUrmaChannel*>(static_cast<Channel*>(channelPtr));
        CHK_PTR_NULL(channelImpl);
        ccuProfilingInfoCache.channelId[i] = channelImpl->GetChannelId();
        ccuProfilingInfoCache.channelHandle[i] = channels[i];
        HCCL_INFO(
            "[%s]type[%d], name[%s], opType[%d], dataType[%d], outputDataType[%d], missionId[%u], "
            "channelHandle[0x%llx], channelId[%u]",
            __func__, ccuProfilingInfoCache.type, ccuProfilingInfoCache.name.c_str(), opType, dataType, outputDataType,
            ccuProfilingInfoCache.missionId, ccuProfilingInfoCache.channelHandle[i],
            ccuProfilingInfoCache.channelId[i]);
    }
    lgProfilingInfo.ccuProfilingInfos.push_back(ccuProfilingInfoCache);
    lgProfilingInfo.lgProfilingReps.push_back(allLgProfilingReps.back());
    return HCCL_SUCCESS;
}

HcclResult CcuKernel::AddCcuProfiling(
    GroupInfo groupInfo, const std::vector<ChannelHandle> channelHandle, HcclDataType dataType,
    HcclDataType outputDataType, HcclReduceOp opType, const std::string& opName)
{
    CHK_RET(AddCcuProfiling(channelHandle.data(), channelHandle.size(), dataType, outputDataType, opType, opName));
    groupOpSizeInfo_.push_back(groupInfo);
    return HCCL_SUCCESS;
}

HcclResult CcuKernel::AddCcuProfiling(
    const ChannelHandle* channels, uint32_t channelNum, HcclDataType dataType, HcclDataType outputDataType,
    HcclReduceOp opType, const std::string& opName)
{
    CHK_PTR_NULL(channels);
    CHK_RET(AddProfilingInfo(channels, channelNum, dataType, outputDataType, opType, opName));
    return HCCL_SUCCESS;
}

HcclResult CcuKernel::Add2ConstValue2VarMap(std::vector<uint64_t>& values)
{
    // 记录当前context所需的常量，仅A6场景适用
    for (uint64_t value : values) {
        if (constValue2VarMap.find(value) == constValue2VarMap.end()) {
            constValue2VarMap[value] = CreateVariable();
        }
    }
    return HCCL_SUCCESS;
}

}; // namespace hcomm
