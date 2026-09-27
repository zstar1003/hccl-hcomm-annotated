/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "order_launch.h"
#include "log.h"
#include "acl/acl_rt.h"
#include "config_log.h"
#include "hccl_types.h"
#include "adapter_rts_common.h"
#include "stream_utils.h"

namespace hccl {
OrderLaunch& OrderLaunch::GetInstance(s32 deviceLogicID)
{
    static OrderLaunch orderLaunch[MAX_MODULE_DEVICE_NUM];
    if (static_cast<u32>(deviceLogicID) >= MAX_MODULE_DEVICE_NUM) {
        HCCL_WARNING("[OrderLaunch][GetInstance]Invalid deviceLogicID[%d]", deviceLogicID);
        return orderLaunch[0];
    }
    HCCL_DEBUG("[OrderLaunch][GetInstance]Valid deviceLogicID[%d]", deviceLogicID);
    return orderLaunch[deviceLogicID];
}

OrderLaunch::OrderLaunch() : initialized_(true) {}

OrderLaunch::~OrderLaunch()
{
    std::unique_lock<std::mutex> mapLock(streamMutex_);
    initialized_ = false;
    groupCtxMap_.clear();
    DestroyRes();
}

void OrderLaunch::DestroyRes()
{
    for (auto& entry : contextResMgrMap_) {
        entry.second.DestroyResources();
    }
    contextResMgrMap_.clear();
    hcomStreamMap_.clear();
}

HcclResult OrderLaunch::RegisterOrderLaunch(const std::string& group)
{
    std::unique_lock<std::mutex> mapLock(streamMutex_);
    if (groupCtxMap_.find(group) != groupCtxMap_.end()) {
        HCCL_WARNING("%s skip, group[%s] has already been registered", __func__, group.c_str());
        return HCCL_SUCCESS;
    }
    // 只记录group，context暂不赋值，只在算子下发阶段对context赋值
    groupCtxMap_.insert({group, INVALID_U64});
    HCCL_INFO("%s success, group[%s]", __func__, group.c_str());
    return HCCL_SUCCESS;
}

/**
 * @brief 从order launch系统注销group
 * 注销group时，会清理group与context的映射关系。
 * 只有当context下没有其他group时，才会清理context对应的资源。
 */
HcclResult OrderLaunch::UnRegisterOrderLaunch(const std::string& group)
{
    CHK_PRT_RET(initialized_ == false, HCCL_WARNING("OrderLaunch has been destroyed"), HCCL_SUCCESS);
    std::unique_lock<std::mutex> mapLock(streamMutex_);
    auto it = groupCtxMap_.find(group);
    if (it == groupCtxMap_.end()) {
        HCCL_WARNING("%s skip, group[%s] has not been registered", __func__, group.c_str());
        return HCCL_SUCCESS;
    }

    u64 context = it->second;
    HCCL_INFO(
        "[OrderLaunch][UnRegisterOrderLaunch] group[%s] context[0x%llx], contextGroupsMap_.size[%zu]", group.c_str(),
        context, contextGroupsMap_.size());
    if (contextGroupsMap_.find(context) != contextGroupsMap_.end()) {
        contextGroupsMap_[context].erase(group);
        if (contextGroupsMap_[context].empty()) {
            contextGroupsMap_.erase(context);
            if (contextResMgrMap_.find(context) != contextResMgrMap_.end()) {
                contextResMgrMap_[context].DestroyResources();
                contextResMgrMap_.erase(context);
            }
            HCCL_INFO("%s contextGroupsMap_ erase context[0x%llx]", __func__, context);
        }
    }

    groupCtxMap_.erase(it);
    HCCL_INFO("%s success, group[%s]", __func__, group.c_str());
    return HCCL_SUCCESS;
}

/**
 * @brief 设置图模式使用的HCOM stream
 * 图模式下，通信使用的附属从流预先设置到hcomStreamMap_中
 */
HcclResult OrderLaunch::SetHcomStream(u32 graphId, const Stream& hcomAttachedStream)
{
    std::unique_lock<std::mutex> mapLock(streamMutex_);
    hcomStreamMap_[graphId] = hcomAttachedStream;
    HCCL_INFO("%s success, graphId[%u], hcomStreamId[%u]", __func__, graphId, hcomAttachedStream.id());
    return HCCL_SUCCESS;
}

/**
 * @brief 初始化group的context映射关系
 *
 * 获取当前线程的context，建立以下映射关系：
 *   - groupCtxMap_[group] = currentContext
 *   - contextGroupsMap_[currentContext] 包含 group
 *   - contextResMgrMap_[currentContext] 包含该context的资源管理器
 */
HcclResult OrderLaunch::InitGroupCtx(const std::string& group)
{
    u64 currentContext = INVALID_U64;
    HcclResult ret = GetCurrentContext(currentContext);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[%s]GetCurrentContext failed, ret[%d]", __func__, ret), ret);

    if (contextResMgrMap_.find(currentContext) == contextResMgrMap_.end()) {
        contextResMgrMap_[currentContext] = OrderLaunchResMgr();
        HCCL_INFO("[OrderLaunch][InitGroupCtx] created new OrderLaunchResMgr for context[0x%llx]", currentContext);
    }

    auto& resMgr = contextResMgrMap_[currentContext];
    if (!resMgr.contextInitialized) {
        resMgr.MarkContextInitialized(currentContext);
    }

    groupCtxMap_[group] = currentContext;
    contextGroupsMap_[currentContext].insert(group);

    HCCL_RUN_INFO("[%s]group[%s] init or update context[0x%llx]", __func__, group.c_str(), currentContext);
    return HCCL_SUCCESS;
}

// aclgraph模式下，先在kernel stream上写record，再在上order stream写wait；解order stream的wait
HcclResult OrderLaunch::AclgraphLaunchInOrderToOrderStream(
    std::string& group, const Stream& kernelStream, const Stream& mainStream, std::shared_ptr<LocalNotify> notify0,
    std::shared_ptr<LocalNotify> notify1, u32 timeOut, HcclRtEvent event)
{
    std::unique_lock<std::mutex> mapLock(streamMutex_);
    // group未注册过，或者未记录过算子下发阶段的线程context
    if (groupCtxMap_.find(group) == groupCtxMap_.end() || groupCtxMap_[group] == INVALID_U64) {
        CHK_RET(InitGroupCtx(group));
    }

    u64 context = groupCtxMap_[group];
    Stream& aclgraphStream = contextResMgrMap_[context].aclgraphStream;
    EnsureOrderStreamForGroup(group, context, aclgraphStream); // aclgraph控制流

    rtModel_t rtModel = nullptr;
    bool isCapture = false;
    CHK_RET(GetStreamCaptureInfo(mainStream.ptr(), rtModel, isCapture));
    CHK_RET(AddStreamToModel(aclgraphStream.ptr(), rtModel));

    aclError ret = ACL_SUCCESS;
    // kernelStream -> aclgraphStream
    ret = aclrtRecordEvent(event, kernelStream.ptr());
    CHK_PRT_RET(ret != ACL_SUCCESS, HCCL_ERROR("[%s]aclrtRecordEvent failed, ret[%d]", __func__, ret), HCCL_E_RUNTIME);
    HCCL_CONFIG_INFO(HCCL_TASK, "[%s]aclrtRecordEvent para: kernelStreamId[%d]", __func__, kernelStream.id());

    ret = aclrtStreamWaitEvent(aclgraphStream.ptr(), event);
    CHK_PRT_RET(
        ret != ACL_SUCCESS, HCCL_ERROR("[%s]aclrtStreamWaitEvent failed, ret[%d]", __func__, ret), HCCL_E_RUNTIME);
    HCCL_CONFIG_INFO(HCCL_TASK, "[%s]aclrtStreamWaitEvent para: orderStreamId[%d]", __func__, aclgraphStream.id());

    HCCL_INFO(
        "[%s] group[%s], kernelStreamId[%u], orderStreamId[%u], context[0x%llx]", __func__, group.c_str(),
        kernelStream.id(), aclgraphStream.id(), context);
    CHK_RET(LaunchInOrder(group, kernelStream, aclgraphStream, notify0, notify1, timeOut));
    return HCCL_SUCCESS;
}

/**
 * @brief ACLGRAPH模式第二步：在order stream上record事件并解kernel stream的wait
 * 执行流程：
 * 1. 在order stream上record事件
 * 2. 在kernel stream上wait该事件，解开kernel stream的阻塞
 */
HcclResult
OrderLaunch::AclgraphLaunchInOrderToKernelStream(std::string& group, const Stream& kernelStream, HcclRtEvent event)
{
    std::unique_lock<std::mutex> mapLock(streamMutex_);

    auto ctxIt = groupCtxMap_.find(group);
    CHK_PRT_RET(
        ctxIt == groupCtxMap_.end(), HCCL_ERROR("[%s]fail, group[%s] is not in groupCtxMap_", __func__, group.c_str()),
        HCCL_E_NOT_FOUND);

    u64 context = ctxIt->second;
    if (contextResMgrMap_.find(context) == contextResMgrMap_.end()) {
        HCCL_ERROR("[%s]fail, context[0x%llx] is not in contextResMgrMap_", __func__, context);
        return HCCL_E_NOT_FOUND;
    }

    Stream& aclgraphStream = contextResMgrMap_[context].aclgraphStream;

    aclError ret = ACL_SUCCESS;
    ret = aclrtRecordEvent(event, aclgraphStream.ptr());
    CHK_PRT_RET(ret != ACL_SUCCESS, HCCL_ERROR("[%s]aclrtRecordEvent failed, ret[%d]", __func__, ret), HCCL_E_RUNTIME);
    HCCL_CONFIG_INFO(HCCL_TASK, "[%s]aclrtRecordEvent para: orderStreamId[%d]", __func__, aclgraphStream.id());

    ret = aclrtStreamWaitEvent(kernelStream.ptr(), event);
    CHK_PRT_RET(
        ret != ACL_SUCCESS, HCCL_ERROR("[%s]aclrtStreamWaitEvent failed, ret[%d]", __func__, ret), HCCL_E_RUNTIME);
    HCCL_CONFIG_INFO(HCCL_TASK, "[%s]aclrtStreamWaitEvent para: kernelStreamId[%d]", __func__, kernelStream.id());

    HCCL_INFO(
        "[%s] group[%s], kernelStreamId[%u], orderStreamId[%u], context[0x%llx]", __func__, group.c_str(),
        kernelStream.id(), aclgraphStream.id(), ctxIt->second);
    return HCCL_SUCCESS;
}

HcclResult OrderLaunch::OpbaseLaunchInOrder(
    std::string& group, const Stream& kernelStream, std::shared_ptr<LocalNotify> notify0,
    std::shared_ptr<LocalNotify> notify1, u32 timeOut)
{
    std::unique_lock<std::mutex> mapLock(streamMutex_);
    HCCL_INFO(
        "[OrderLaunch][OpbaseLaunchInOrder] group[%s], kernelStreamId[%u], timeOut[%d ms]", group.c_str(),
        kernelStream.id(), timeOut);
    // group未注册过，或者未记录过算子下发阶段的线程context
    if (groupCtxMap_.find(group) == groupCtxMap_.end() || groupCtxMap_[group] == INVALID_U64) {
        CHK_RET(InitGroupCtx(group));
    }

    u64 context = groupCtxMap_[group];
    Stream& opbaseStream = contextResMgrMap_[context].opbaseStream;
    EnsureOrderStreamForGroup(group, context, opbaseStream); // 单算子控制流

    HCCL_INFO(
        "[%s] group[%s], kernelStreamId[%u], orderStreamId[%u], context[0x%llx]", __func__, group.c_str(),
        kernelStream.id(), opbaseStream.id(), context);
    CHK_RET(LaunchInOrder(group, kernelStream, opbaseStream, notify0, notify1, timeOut));
    return HCCL_SUCCESS;
}

/**
 * @brief 图模式下的按序下发
 *
 * 图模式下，使用预先设置的hcomAttachedStream作为order stream
 */
HcclResult OrderLaunch::HcomLaunchInOrder(
    std::string& group, const Stream& kernelStream, u32 graphId, std::shared_ptr<LocalNotify> notify0,
    std::shared_ptr<LocalNotify> notify1, u32 timeOut)
{
    std::unique_lock<std::mutex> mapLock(streamMutex_);
    Stream hostOrderStream;
    if (hcomStreamMap_.find(graphId) == hcomStreamMap_.end()) {
        HCCL_ERROR("[%s] graphId[%u] group[%s] stream not found", __func__, graphId, group.c_str());
        return HCCL_E_NOT_FOUND;
    }
    hostOrderStream = hcomStreamMap_[graphId];
    CHK_PTR_NULL(hostOrderStream.ptr());
    HCCL_INFO("[%s] group[%s], graphId[%u], streamId[%u]", __func__, group.c_str(), graphId, hostOrderStream.id());
    CHK_RET(LaunchInOrder(group, kernelStream, hostOrderStream, notify0, notify1, timeOut));
    return HCCL_SUCCESS;
}

/**
 * @brief 使用notify机制实现stream间的按序下发
 * 1. wait notify0 on kernelStream - 等待kernel stream上的算子完成
 * 2. record notify0 on hostOrderStream - 在order stream上record notify
 * 3. wait notify1 on hostOrderStream - 等待其他算子在order stream上完成
 */
HcclResult OrderLaunch::LaunchInOrder(
    [[maybe_unused]] const std::string& group, const Stream& kernelStream, const Stream& hostOrderStream,
    std::shared_ptr<LocalNotify> notify0, std::shared_ptr<LocalNotify> notify1, u32 timeOut)
{
    CHK_SMART_PTR_NULL(notify0);
    CHK_SMART_PTR_NULL(notify1);
    aclError ret = ACL_SUCCESS;
    ret = aclrtWaitAndResetNotify(notify0->ptr(), kernelStream.ptr(), timeOut);
    CHK_PRT_RET(
        ret != ACL_SUCCESS,
        HCCL_ERROR(
            "[%s] aclrtWaitAndResetNotify failed, ret[%d], notifyId[%u], streamId[%d], timeOut[%d s]", __func__, ret,
            notify0->notifyId_, kernelStream.id(), timeOut),
        HCCL_E_RUNTIME);
    HCCL_CONFIG_INFO(
        HCCL_TASK, "[%s] aclrtWaitAndResetNotify para: notifyId[%u], streamId[%d], timeOut[%d s]", __func__,
        notify0->notifyId_, kernelStream.id(), timeOut);

    ret = aclrtRecordNotify(notify0->ptr(), hostOrderStream.ptr());
    CHK_PRT_RET(
        ret != ACL_SUCCESS,
        HCCL_ERROR(
            "[%s] aclrtRecordNotify failed, ret[%d], notifyId[%u], streamId[%d]", __func__, ret, notify0->notifyId_,
            hostOrderStream.id()),
        HCCL_E_RUNTIME);
    HCCL_CONFIG_INFO(
        HCCL_TASK, "[%s] aclrtRecordNotify para: notifyId[%u], streamId[%d]", __func__, notify0->notifyId_,
        hostOrderStream.id());

    ret = aclrtWaitAndResetNotify(notify1->ptr(), hostOrderStream.ptr(), timeOut);
    CHK_PRT_RET(
        ret != ACL_SUCCESS,
        HCCL_ERROR(
            "[%s] aclrtWaitAndResetNotify failed, ret[%d], notifyId[%u], streamId[%d], timeOut[%d s]", __func__, ret,
            notify1->notifyId_, hostOrderStream.id(), timeOut),
        HCCL_E_RUNTIME);
    HCCL_CONFIG_INFO(
        HCCL_TASK, "[%s] aclrtWaitAndResetNotify para: notifyId[%u], streamId[%d], timeOut[%d s]", __func__,
        notify1->notifyId_, hostOrderStream.id(), timeOut);
    return HCCL_SUCCESS;
}

HcclResult OrderLaunch::EnsureOrderStreamForGroup(std::string& group, u64 context, Stream& orderStream)
{
    auto it = groupCtxMap_.find(group);
    if (it == groupCtxMap_.end()) {
        HCCL_ERROR("[%s] group[%s] not found", __func__, group.c_str());
        return HCCL_E_PARA;
    }

    if (contextResMgrMap_.find(context) == contextResMgrMap_.end()) {
        HCCL_ERROR("[%s] context[0x%llx] not found for group[%s]", __func__, context, group.c_str());
        return HCCL_E_PARA;
    }

    auto& groupCtxRes = contextResMgrMap_[context];

    if (orderStream.ptr() == nullptr) {
        HCCL_INFO(
            "[OrderLaunch][EnsureOrderStreamForGroup] creating new order stream for group[%s], context[0x%llx]",
            group.c_str(), context);
        // Stream 尚未创建，基于传入的 context 创建
        // 创建新的order stream; streamMode = 1 使能遇错即停，避免出错后流卡住不退
        constexpr u32 streamMode = 1;
        orderStream = Stream(StreamType::STREAM_TYPE_ONLINE);
        CHK_RET(hrtStreamSetMode(orderStream.ptr(), streamMode));
        HCCL_INFO(
            "[OrderLaunch] Created new order stream with id[%d] with context [0x%llx]", orderStream.id(), context);

        // 对group->contextResource的映射关系进行更新
        if (!groupCtxRes.contextInitialized) {
            groupCtxRes.MarkContextInitialized(context);
        } else {
            groupCtxRes.UpdateContext(context);
        }
        // 对context->group的映射关系进行更新
        contextGroupsMap_[context].insert(group);
        HCCL_INFO(
            "[OrderLaunch] Added group[%s] to context [0x%llx] with order stream id[%d]", group.c_str(), context,
            orderStream.id());
    } else {
        HCCL_INFO(
            "[OrderLaunch][EnsureOrderStreamForGroup] order stream already exists, group[%s], context[0x%llx], "
            "streamId[%d]",
            group.c_str(), context, orderStream.id());
    }
    return HCCL_SUCCESS;
}

/**
 * @brief 获取当前线程的context
 * 通过hrtCtxGetCurrent接口获取当前线程关联的runtime context
 */
HcclResult OrderLaunch::GetCurrentContext(u64& currentContext) const
{
    HcclRtContext rtCtx = nullptr;
    CHK_RET(hrtCtxGetCurrent(&rtCtx));
    currentContext = reinterpret_cast<u64>(rtCtx);

    if (currentContext == INVALID_U64) {
        HCCL_ERROR("[%s] GetCurrentContext failed", __func__);
        return HCCL_E_RUNTIME;
    }

    return HCCL_SUCCESS;
}
} // namespace hccl
