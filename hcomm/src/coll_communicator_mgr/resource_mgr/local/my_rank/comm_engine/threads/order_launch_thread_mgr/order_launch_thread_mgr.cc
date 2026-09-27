/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "order_launch_thread_mgr.h"
#include "acl/acl_rt.h"
#include "orion_adapter_rts.h"
#include "coll_comm.h"
#include "hcclCommOp.h"
#include "sal_pub.h"
#include "dfx_dlprof_function.h"
#include "hcomm_c_adpt.h"

namespace hccl {

constexpr u32 ORDER_THREAD_NOTIFY_NUM = 1;

static u32 GetAicpuBlockNum(s32 deviceLogicId)
{
    int64_t coreNum = 0;
    aclError ret = aclrtGetDeviceInfo(static_cast<uint32_t>(deviceLogicId), ACL_DEV_ATTR_AICPU_CORE_NUM, &coreNum);
    HCCL_INFO(
        "[GetAicpuBlockNum] deviceLogicId[%d], aclrtGetDeviceInfo ret[%d], coreNum[%lld]", deviceLogicId, ret, coreNum);
    if (ret != ACL_SUCCESS || coreNum <= 0) {
        HCCL_WARNING(
            "[GetAicpuBlockNum] aclrtGetDeviceInfo failed, ret[%d], coreNum[%lld], use default 1", ret, coreNum);
        return 1U;
    }
    HCCL_INFO("[GetAicpuBlockNum] success, aicpuCoreNum[%u]", static_cast<u32>(coreNum));
    return static_cast<u32>(coreNum);
}

/* ============================ OrderLaunchContextRes ============================ */

void OrderLaunchContextRes::DestroyResources()
{
    if (opbaseThread != 0) {
        HcommResult ret = HcommThreadFree(&opbaseThread, 1);
        if (ret != HCCL_SUCCESS) {
            HCCL_WARNING(
                "[OrderLaunchContextRes] HcommThreadFree opbaseThread[0x%llx] failed, ret[%d]", opbaseThread, ret);
        }
        opbaseThread = 0;
    }
    if (aclgraphThread != 0) {
        HcommResult ret = HcommThreadFree(&aclgraphThread, 1);
        if (ret != HCCL_SUCCESS) {
            HCCL_WARNING(
                "[OrderLaunchContextRes] HcommThreadFree aclgraphThread[0x%llx] failed, ret[%d]", aclgraphThread, ret);
        }
        aclgraphThread = 0;
    }
    resValid = false;
    HCCL_INFO("[OrderLaunchContextRes] resources destroyed, context[0x%llx]", context);
}

/* ============================ OrderLaunchThreadMgr ============================ */

OrderLaunchThreadMgr::OrderLaunchThreadMgr() {}

OrderLaunchThreadMgr::~OrderLaunchThreadMgr()
{
    std::unique_lock<std::mutex> lock(mutex_);
    Destroy();
}

void OrderLaunchThreadMgr::Destroy()
{
    for (auto& entry : contextResMap_) {
        entry.second.DestroyResources();
    }
    contextResMap_.clear();
    contextGroupsMap_.clear();
    groupCtxMap_.clear();

    for (auto& entry : hcomAttachedThreadMap_) {
        if (entry.second != 0) {
            HcommResult ret = HcommThreadFreeWithStream(&entry.second, 1);
            if (ret != HCCL_SUCCESS) {
                HCCL_WARNING(
                    "[OrderLaunchThreadMgr] HcommThreadFreeWithStream hcomAttachedThread[0x%llx] failed, ret[%d]",
                    entry.second, ret);
            }
            entry.second = 0;
        }
    }
    hcomAttachedThreadMap_.clear();
    groupGraphMap_.clear();
}

HcclResult OrderLaunchThreadMgr::RegisterOrderLaunch(const std::string& group)
{
    std::unique_lock<std::mutex> lock(mutex_);
    if (groupCtxMap_.find(group) != groupCtxMap_.end()) {
        HCCL_WARNING("%s skip, group[%s] has already been registered", __func__, group.c_str());
        return HCCL_SUCCESS;
    }
    groupCtxMap_.insert({group, UINT64_MAX});
    HCCL_INFO("%s success, group[%s]", __func__, group.c_str());
    return HCCL_SUCCESS;
}

HcclResult OrderLaunchThreadMgr::UnRegisterOrderLaunch(const std::string& group)
{
    std::unique_lock<std::mutex> lock(mutex_);
    auto it = groupCtxMap_.find(group);
    if (it == groupCtxMap_.end()) {
        HCCL_WARNING("%s skip, group[%s] has not been registered", __func__, group.c_str());
        return HCCL_SUCCESS;
    }

    u64 context = it->second;
    HCCL_INFO("[OrderLaunchThreadMgr][%s] group[%s] context[0x%llx]", __func__, group.c_str(), context);

    auto ctxIt = contextGroupsMap_.find(context);
    if (ctxIt != contextGroupsMap_.end()) {
        ctxIt->second.erase(group);
        if (ctxIt->second.empty()) {
            contextGroupsMap_.erase(ctxIt);
            auto resIt = contextResMap_.find(context);
            if (resIt != contextResMap_.end()) {
                resIt->second.DestroyResources();
                contextResMap_.erase(resIt);
            }
            HCCL_INFO("%s contextGroupsMap_ erase context[0x%llx]", __func__, context);
        }
    }

    groupCtxMap_.erase(it);

    HCCL_INFO("%s success, group[%s]", __func__, group.c_str());
    return HCCL_SUCCESS;
}

HcclResult OrderLaunchThreadMgr::GetCurrentContext(u64& currentContext) const
{
    aclrtContext rtCtx = nullptr;
    aclError ret = aclrtGetCurrentContext(&rtCtx);
    CHK_PRT_RET(
        ret != ACL_SUCCESS, HCCL_ERROR("[%s]aclrtGetCurrentContext failed, ret[%d]", __func__, ret), HCCL_E_RUNTIME);
    currentContext = reinterpret_cast<u64>(rtCtx);
    CHK_PRT_RET(
        currentContext == UINT64_MAX, HCCL_ERROR("[%s]GetCurrentContext failed, context is INVALID_U64", __func__),
        HCCL_E_RUNTIME);
    return HCCL_SUCCESS;
}

HcclResult OrderLaunchThreadMgr::EnsureContextRes(u64 context)
{
    if (contextResMap_.find(context) == contextResMap_.end()) {
        contextResMap_.emplace(context, OrderLaunchContextRes());
        HCCL_INFO(
            "[OrderLaunchThreadMgr][%s] created new OrderLaunchContextRes for context[0x%llx]", __func__, context);
    }
    return HCCL_SUCCESS;
}

HcclResult OrderLaunchThreadMgr::SetAttachedStream(const std::string& group, u32 graphId, void* stream)
{
    CHK_PRT_RET(stream == nullptr, HCCL_ERROR("[%s] stream is nullptr, graphId[%u]", __func__, graphId), HCCL_E_PTR);

    std::unique_lock<std::mutex> lock(mutex_);

    ThreadHandle thread = 0;
    HcommResult ret = HcommThreadAllocWithStream(COMM_ENGINE_CPU_TS, stream, ORDER_THREAD_NOTIFY_NUM, &thread);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[%s] HcommThreadAllocWithStream failed, ret[%d], graphId[%u]", __func__, ret, graphId),
        static_cast<HcclResult>(ret));

    hcomAttachedThreadMap_[graphId] = thread;
    groupGraphMap_[group] = graphId;
    HCCL_INFO(
        "%s success, group[%s], graphId[%u], stream[%p], thread[0x%llx]", __func__, group.c_str(), graphId, stream,
        thread);
    return HCCL_SUCCESS;
}

void OrderLaunchThreadMgr::UpdateGroupContextMapping(const std::string& group, u64 currentContext)
{
    auto groupIt = groupCtxMap_.find(group);
    if (groupIt != groupCtxMap_.end() && groupIt->second != currentContext) {
        HCCL_INFO(
            "[OrderLaunchThreadMgr][%s] group[%s] context updated: [0x%llx] -> [0x%llx]", __func__, group.c_str(),
            groupIt->second, currentContext);
        if (groupIt->second != UINT64_MAX) {
            auto oldCtxIt = contextGroupsMap_.find(groupIt->second);
            if (oldCtxIt != contextGroupsMap_.end()) {
                oldCtxIt->second.erase(group);
                if (oldCtxIt->second.empty()) {
                    contextGroupsMap_.erase(oldCtxIt);
                }
            }
        }
        groupIt->second = currentContext;
        contextGroupsMap_[currentContext].insert(group);
    } else if (groupIt == groupCtxMap_.end()) {
        groupCtxMap_[group] = currentContext;
        contextGroupsMap_[currentContext].insert(group);
    }
}

bool OrderLaunchThreadMgr::IsOrderLaunchDisabled(u64 currentContext)
{
    if (blockNum_ == 0U) {
        blockNum_ = GetAicpuBlockNum(static_cast<s32>(Hccl::HrtGetDevice()));
    }
    if (blockNum_ == 0U) {
        return false;
    }
    auto ctxGroupsIt = contextGroupsMap_.find(currentContext);
    u32 groupCount = (ctxGroupsIt != contextGroupsMap_.end()) ? static_cast<u32>(ctxGroupsIt->second.size()) : 0U;
    HCCL_INFO(
        "[OrderLaunchThreadMgr][%s] blockNum[%u] groupCount[%u] context[0x%llx]", __func__, blockNum_, groupCount,
        currentContext);
    return groupCount <= blockNum_;
}

HcclResult OrderLaunchThreadMgr::EnsureOrderThread(
    OrderThreadMode mode, const std::string& group, uint32_t notifyNumPerThread, ThreadHandle& thread)
{
    std::unique_lock<std::mutex> lock(mutex_);
    thread = 0;

    u64 currentContext = UINT64_MAX;
    CHK_RET(GetCurrentContext(currentContext));
    CHK_RET(EnsureContextRes(currentContext));

    auto& ctxRes = contextResMap_[currentContext];

    UpdateGroupContextMapping(group, currentContext);

    if (IsOrderLaunchDisabled(currentContext)) {
        HCCL_INFO(
            "[OrderLaunchThreadMgr][%s] order launch disabled, group[%s], context[0x%llx]", __func__, group.c_str(),
            currentContext);
        thread = 0;
        return HCCL_SUCCESS;
    }

    ThreadHandle& targetThread = (mode == OrderThreadMode::ACLGRAPH) ? ctxRes.aclgraphThread : ctxRes.opbaseThread;

    if (targetThread == 0) {
        HCCL_INFO(
            "[OrderLaunchThreadMgr][%s] creating new order thread, context[0x%llx], mode[%u], notifyNumPerThread[%u]",
            __func__, currentContext, static_cast<u8>(mode), notifyNumPerThread);

        HcommResult ret = HcommThreadAlloc(COMM_ENGINE_CPU_TS, 1, &notifyNumPerThread, &targetThread);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR("[%s] HcommThreadAlloc failed, ret[%d], mode[%u]", __func__, ret, static_cast<u8>(mode)),
            static_cast<HcclResult>(ret));

        ctxRes.resValid = true;
        HCCL_INFO(
            "[OrderLaunchThreadMgr] Created new order thread[0x%llx], context[0x%llx], mode[%u]", targetThread,
            currentContext, static_cast<u8>(mode));
    } else {
        HCCL_INFO(
            "[OrderLaunchThreadMgr][%s] order thread already exists, context[0x%llx], thread[0x%llx]", __func__,
            currentContext, targetThread);
    }

    thread = targetThread;
    return HCCL_SUCCESS;
}

HcclResult OrderLaunchThreadMgr::EnsureDeviceOrderThread(
    CollComm* collComm, const std::string& group, uint32_t notifyNumPerThread, ThreadHandle& thread)
{
    std::unique_lock<std::mutex> lock(mutex_);
    thread = 0;

    u64 currentContext = UINT64_MAX;
    CHK_RET(GetCurrentContext(currentContext));
    UpdateGroupContextMapping(group, currentContext);
    if (IsOrderLaunchDisabled(currentContext)) {
        HCCL_INFO(
            "[OrderLaunchThreadMgr][%s] order launch disabled, group[%s], context[0x%llx]", __func__, group.c_str(),
            currentContext);
        thread = 0;
        return HCCL_SUCCESS;
    }

    CHK_PRT_RET(
        collComm == nullptr, HCCL_ERROR("[%s] collComm is null, group[%s]", __func__, group.c_str()), HCCL_E_PTR);
    CommEngineResMgr* engineResMgr = collComm->GetCommEngineResMgr();
    CHK_PRT_RET(
        engineResMgr == nullptr, HCCL_ERROR("[%s] engineResMgr is null, group[%s]", __func__, group.c_str()),
        HCCL_E_PTR);
    CHK_RET(engineResMgr->HcclDedicatedThreadAcquire(
        HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_DEVICE, notifyNumPerThread, &thread));

    HCCL_INFO("[OrderLaunchThreadMgr][%s] device order thread[0x%llx], group[%s]", __func__, thread, group.c_str());
    return HCCL_SUCCESS;
}

ThreadHandle OrderLaunchThreadMgr::GetHcomAttachedThreadByGroup(const std::string& group)
{
    std::unique_lock<std::mutex> lock(mutex_);

    u64 currentContext = UINT64_MAX;
    if (GetCurrentContext(currentContext) != HCCL_SUCCESS) {
        HCCL_WARNING("[%s] GetCurrentContext failed, group[%s]", __func__, group.c_str());
        return 0;
    }
    UpdateGroupContextMapping(group, currentContext);
    if (IsOrderLaunchDisabled(currentContext)) {
        HCCL_INFO(
            "[OrderLaunchThreadMgr][%s] order launch disabled, group[%s], context[0x%llx]", __func__, group.c_str(),
            currentContext);
        return 0;
    }

    auto graphIt = groupGraphMap_.find(group);
    if (graphIt == groupGraphMap_.end()) {
        HCCL_WARNING(
            "[%s] graphId not found for group[%s], please call HcomSetAttachedStream first", __func__, group.c_str());
        return 0;
    }
    u32 graphId = graphIt->second;

    auto attachedThreadIt = hcomAttachedThreadMap_.find(graphId);
    if (attachedThreadIt == hcomAttachedThreadMap_.end() || attachedThreadIt->second == 0) {
        HCCL_ERROR(
            "[%s] hcomThread not found for group[%s], graphId[%u], please call HcomSetAttachedStream first", __func__,
            group.c_str(), graphId);
        return 0;
    }
    HCCL_INFO(
        "[%s] success, group[%s], graphId[%u], thread[0x%llx]", __func__, group.c_str(), graphId,
        attachedThreadIt->second);
    return attachedThreadIt->second;
}

HcclResult OrderLaunchThreadMgr::RegisterThreadToComm(CollComm* collComm, ThreadHandle thread) const
{
    CommEngineResMgr* engineResMgr = collComm->GetCommEngineResMgr();
    if (engineResMgr != nullptr) {
        CHK_RET(engineResMgr->RegisterOrderLaunchThread(thread));
    }
    return HCCL_SUCCESS;
}

HcclResult OrderLaunchThreadMgr::RegisterDfx(
    CollComm* collComm, HcclDedicatedThreadType useType, ThreadHandle thread, u64 beginTime,
    const std::string& commId) const
{
    if (useType == HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_DEVICE) {
        HcclCommDfx* hcclCommDfx = collComm->GetHcclCommDfx();
        if (hcclCommDfx != nullptr) {
            const std::string kernelName = "RunAicpuIndOpThreadInit";
            CHK_RET(hcclCommDfx->ReportKernel(beginTime, commId, kernelName, SalGetTid(), false));
            HCCL_INFO("[%s] DEVICE order launch thread ReportKernel done, comm[%s]", __func__, commId.c_str());
        }
    } else {
        std::function<HcclResult(u32, u32, const Hccl::TaskParam&, u64)> dfxCallback
            = [](u32 streamId, u32 taskId, const Hccl::TaskParam& taskParam, u64 handle) {
                  (void)streamId;
                  (void)taskId;
                  (void)taskParam;
                  (void)handle;
                  return HCCL_SUCCESS;
              };
        int ret = HcommThreadRegisterDfx(thread, dfxCallback);
        if (ret != 0) {
            HCCL_WARNING("[%s] HcommThreadRegisterDfx failed, ret[%d], thread[0x%llx]", __func__, ret, thread);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult OrderLaunchThreadMgr::OrderLaunchThreadAcquire(
    HcclDedicatedThreadType useType, CollComm* collComm, const std::string& group, uint32_t notifyNumPerThread,
    ThreadHandle& thread)
{
    thread = 0;

    HCCL_INFO(
        "[%s] begin, useType[%d], group[%s], notifyNumPerThread[%u]", __func__, static_cast<s32>(useType),
        group.c_str(), notifyNumPerThread);

    u64 beginTime = Hccl::DfxDlProfFunction::GetInstance().dlMsprofSysCycleTime();

    switch (useType) {
        case HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_OPBASE: {
            HcclResult ret = EnsureOrderThread(OrderThreadMode::OPBASE, group, notifyNumPerThread, thread);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS, HCCL_ERROR("[%s] EnsureOrderThread OPBASE failed, ret[%d]", __func__, ret), ret);
            break;
        }
        case HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_ACLGRAPH: {
            HcclResult ret = EnsureOrderThread(OrderThreadMode::ACLGRAPH, group, notifyNumPerThread, thread);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS, HCCL_ERROR("[%s] EnsureOrderThread ACLGRAPH failed, ret[%d]", __func__, ret), ret);
            break;
        }
        case HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_GE: {
            ThreadHandle th = GetHcomAttachedThreadByGroup(group);
            thread = th;
            CHK_PRT_RET(
                th == 0,
                HCCL_WARNING(
                    "[%s] GetHcomAttachedThreadByGroup failed, group[%s], please call HcomSetAttachedStream first",
                    __func__, group.c_str()),
                HCCL_SUCCESS);
            break;
        }
        case HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_DEVICE: {
            HcclResult ret = EnsureDeviceOrderThread(collComm, group, notifyNumPerThread, thread);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS, HCCL_ERROR("[%s] EnsureDeviceOrderThread failed, ret[%d]", __func__, ret), ret);
            break;
        }
        default:
            HCCL_ERROR("[%s] invalid useType[%d] for order launch", __func__, static_cast<s32>(useType));
            return HCCL_E_PARA;
    }

    if (thread != 0 && collComm != nullptr) {
        if (useType != HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_DEVICE) {
            CHK_RET(RegisterThreadToComm(collComm, thread));
        }
        CHK_RET(RegisterDfx(collComm, useType, thread, beginTime, group));
    }

    HCCL_INFO("[%s] success, useType[%d], thread[0x%llx]", __func__, static_cast<s32>(useType), thread);
    return HCCL_SUCCESS;
}

} // namespace hccl
