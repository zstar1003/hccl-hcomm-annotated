/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 日志染色: 模块 tag (须在 include sim_log.h 之前)
#define HCCL_VM_MODULE "NOTIFY_STUB"

#include <atomic>
#include <cstdint>
#include <iostream>
#include <unistd.h>
#include <unistd.h>
#include <vector>

#include "acl/acl_base.h"
#include "acl/acl_rt.h"
#include "store_sim_store_pub.h"
#include "sim_log.h"
#include "runtime/base.h"
#include "runtime/event.h"
#include "db_sim_runner_ops.h"
#include "db_sim_runner_common.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

void PrintTaskMetaData(const HcclTaskMetaData& taskMeta)
{
    pid_t pid = getpid();
    switch (taskMeta.taskType) {
        case HccLTaskMetaType::MEM_CPY:
            HCCL_VM_INFO(
                "pid[{}]: rankId[{}], streamId[{}], taskType[MEM_CPY], srcOffset[{}], dstOffset[{}], len[{}], "
                "srcRankId[{}], dstRankId[{}]",
                pid, taskMeta.rankId, taskMeta.streamId, taskMeta.taskData.transMem.srcOffset,
                taskMeta.taskData.transMem.dstOffset, taskMeta.taskData.transMem.len,
                taskMeta.taskData.transMem.srcRankId, taskMeta.taskData.transMem.dstRankId);
            break;
        case HccLTaskMetaType::REDUCE:
            HCCL_VM_INFO(
                "pid[{}]: rankId[{}], streamId[{}], taskType[REDUCE], srcOffset[{}], dstOffset[{}], len[{}], "
                "srcRankId[{}], dstRankId[{}], reduceOp[{}], dataType[{}]",
                pid, taskMeta.rankId, taskMeta.streamId, taskMeta.taskData.reduce.srcOffset,
                taskMeta.taskData.reduce.dstOffset, taskMeta.taskData.reduce.dataCount,
                taskMeta.taskData.reduce.srcRankId, taskMeta.taskData.reduce.dstRankId,
                taskMeta.taskData.reduce.reduceOp, taskMeta.taskData.reduce.dataType);
            break;
        case HccLTaskMetaType::NOTIFY_WAIT:
            HCCL_VM_INFO(
                "pid[{}]: rankId[{}], streamId[{}], taskType[NOTIFY_WAIT], notifyId[{}], srcRankId[{}], dstRankId[{}]",
                pid, taskMeta.rankId, taskMeta.streamId, taskMeta.taskData.notify.notifyId,
                taskMeta.taskData.notify.srcRankId, taskMeta.taskData.notify.dstRankId);
            break;
        case HccLTaskMetaType::NOTIFY_RECORD:
            HCCL_VM_INFO(
                "pid[{}]: rankId[{}], streamId[{}], taskType[NOTIFY_RECORD], notifyId[{}], srcRankId[{}], "
                "dstRankId[{}]",
                pid, taskMeta.rankId, taskMeta.streamId, taskMeta.taskData.notify.notifyId,
                taskMeta.taskData.notify.srcRankId, taskMeta.taskData.notify.dstRankId);
            break;
        default:
            break;
    }
}

aclError aclrtCreateNotify(aclrtNotify* notify, uint64_t flag)
{
    (void)flag;
    auto serverId = sim::GetCurServerId();
    if (serverId == 0) {
        return ACL_ERROR_INVALID_PARAM;
    }
    sim::Runner runner;
    if (!sim::GetCurrRunnerTls(serverId, runner)) {
        return ACL_ERROR_INVALID_PARAM;
    }
    if (runner.current_ctx_id == 0) {
        HCCL_VM_ERROR("invalid param");
        return ACL_ERROR_INVALID_PARAM;
    }
    auto currCtx = RunnerDB::GetById<sim::Context>(runner.current_ctx_id);
    if (!currCtx.has_value()) {
        HCCL_VM_ERROR("CurrContext not found:{:d}", runner.current_ctx_id);
        return ACL_ERROR_INVALID_PARAM;
    }

    auto& currCtxId = currCtx->id;

    sim::Notify tmp{};
    tmp.create_ctx_id = currCtxId;
    auto res = RunnerDB::Add<sim::Notify>(tmp);

    *notify = (aclrtNotify)res;
    HCCL_VM_DEBUG("id:{:d}", res);
    return ACL_SUCCESS;
}

aclError aclrtDestroyNotify(aclrtNotify notify)
{
    uint64_t notifyId = (uint32_t)(uintptr_t)notify;
    HCCL_VM_DEBUG("id:{:d}", notifyId);
    RunnerDB::Delete<sim::Notify>(notifyId);
    return ACL_SUCCESS;
}

aclError aclrtGetNotifyId(aclrtNotify notify, uint32_t* notifyId)
{
    *notifyId = (uint32_t)(uintptr_t)notify;
    HCCL_VM_DEBUG("id:{:d}", *notifyId);
    return ACL_SUCCESS;
}

aclError aclrtRecordNotify(aclrtNotify notify, aclrtStream stream)
{
    uint64_t streamId = sim::GetCurrentStreamId((uint64_t)(uintptr_t)stream);
    uint64_t notifyId = (uint32_t)(uintptr_t)notify;

    uint32_t curRank = (uint32_t)sim::GetCurrRankId();

    HcclTaskMetaData taskMetaData;
    taskMetaData.taskType = HccLTaskMetaType::NOTIFY_RECORD;
    taskMetaData.commId = 0;
    taskMetaData.rankId = curRank;
    taskMetaData.streamId = streamId;
    taskMetaData.taskData.notify.notifyId = notifyId;
    taskMetaData.taskData.notify.notifyCount = 0; // notify value
    taskMetaData.taskData.notify.srcRankId = curRank;
    taskMetaData.taskData.notify.dstRankId = curRank;
    PrintTaskMetaData(taskMetaData);

    uint32_t index{0};
    HCCL_VM_DEBUG("Get notify task, id={:d}, streamId={:d}", notifyId, streamId);
    auto ret = InsertTaskToCollection(&taskMetaData, &index);
    if (ret != HcclSim::HcclVmResult::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("InsertTaskToCollection fail");
        return ACL_ERROR_INTERNAL_ERROR;
    }

    return ACL_SUCCESS;
}

aclError aclrtWaitAndResetNotify(aclrtNotify notify, aclrtStream stream, uint32_t timeout)
{
    (void)timeout;
    uint64_t streamId = sim::GetCurrentStreamId((uint64_t)(uintptr_t)stream);
    uint64_t notifyId = (uint32_t)(uintptr_t)notify;

    // reset notify
    auto res = RunnerDB::Update<sim::Notify>(notifyId, [](sim::Notify& notify) {
        notify.value = 0;
    });
    if (!res) {
        HCCL_VM_ERROR("notify not found:{:d}", notifyId);
        return ACL_ERROR_INVALID_PARAM;
    }

    uint32_t curRank = (uint32_t)sim::GetCurrRankId();

    HcclTaskMetaData taskMetaData;
    taskMetaData.taskType = HccLTaskMetaType::NOTIFY_WAIT;
    taskMetaData.commId = 0;
    taskMetaData.rankId = curRank;
    taskMetaData.streamId = streamId;
    taskMetaData.taskData.notify.notifyId = notifyId;
    taskMetaData.taskData.notify.notifyCount = 0; // notify value
    taskMetaData.taskData.notify.srcRankId = curRank;
    taskMetaData.taskData.notify.dstRankId = curRank;
    PrintTaskMetaData(taskMetaData);

    uint32_t index{0};
    HCCL_VM_DEBUG("Get notify task, id={:d}, streamId={:d}", notifyId, streamId);
    auto ret = InsertTaskToCollection(&taskMetaData, &index);
    if (ret != HcclSim::HcclVmResult::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("InsertTaskToCollection fail");
        return ACL_ERROR_INTERNAL_ERROR;
    }

    return ACL_SUCCESS;
}

aclError aclrtNotifyBatchReset(aclrtNotify* notifies, size_t num)
{
    for (int i = 0; i < num; i++) {
        uint64_t notifyId = (uint32_t)(uintptr_t)notifies[i];

        // reset notify
        auto res = RunnerDB::Update<sim::Notify>(notifyId, [](sim::Notify& notify) {
            notify.value = 0;
        });
        if (!res) {
            HCCL_VM_ERROR("notify not found:{:d}", notifyId);
            return ACL_ERROR_INVALID_PARAM;
        }
    }
    return ACL_SUCCESS;
}

aclError aclrtNotifyGetExportKey(aclrtNotify notify, char* key, size_t len, uint64_t flags)
{
    (void)len;
    (void)flags;
    uint64_t notifyId = (uint32_t)(uintptr_t)notify;

    std::string notifyKeyStr = std::to_string(notifyId);

    sim::IpcNotify tmp{};
    tmp.notify_id = notifyId;
    // tmp.flag = (uint8_t)flags;
    memcpy(tmp.name_or_key, notifyKeyStr.data(), notifyKeyStr.length());
    memcpy(key, notifyKeyStr.data(), notifyKeyStr.length());
    tmp.create_pid = getpid();
    auto res = RunnerDB::Add<sim::IpcNotify>(tmp);
    HCCL_VM_DEBUG("id:{:d}", notifyId);
    return ACL_SUCCESS;
}

aclError aclrtNotifySetImportPid(aclrtNotify notify, int32_t* pid, size_t num)
{
    uint64_t notifyId = (uint32_t)(uintptr_t)notify;
    HCCL_VM_DEBUG("id:{:d}", notifyId);

    auto ipcNotify = RunnerDB::GetOneByPred<sim::IpcNotify>([notifyId](const sim::IpcNotify& ipc) {
        return ipc.notify_id == notifyId;
    });
    if (!ipcNotify.second) {
        HCCL_VM_ERROR("notify in ipc notify not found:{:d}", notifyId);
        return ACL_ERROR_INVALID_PARAM;
    }

    for (int i = 0; i < num; i++) {
        sim::IpcNotifyVistorList vistor;
        vistor.ipc_id = ipcNotify.first.id;
        vistor.visitor_pid = pid[i];
        RunnerDB::Add<sim::IpcNotifyVistorList>(vistor);
    }

    return ACL_SUCCESS;
}

aclError aclrtNotifyImportByKey(aclrtNotify* notify, const char* key, uint64_t flags)
{
    (void)flags;
    auto ipcNotify = RunnerDB::GetOneByPred<sim::IpcNotify>([key](const sim::IpcNotify& ipc) {
        return memcmp(key, ipc.name_or_key, strlen(key)) == 0;
    });
    if (!ipcNotify.second) {
        HCCL_VM_ERROR("notify in ipc notify key not found:{}", key);
        return ACL_ERROR_INVALID_PARAM;
    }
    *notify = (aclrtNotify)ipcNotify.first.notify_id;
    HCCL_VM_DEBUG("id:{:d}", ipcNotify.first.notify_id);
    return ACL_SUCCESS;
}

////////////////////////////rt 接口/////////////////////////////////
rtError_t rtsNotifyCreate(rtNotify_t* notify, uint64_t flag) { return aclrtCreateNotify(notify, flag); }

rtError_t rtNotifyWait(rtNotify_t notify, rtStream_t stm) { return aclrtWaitAndResetNotify(notify, stm, 0); }

rtError_t rtGetNotifyAddress(rtNotify_t notify, uint64_t* const notifyAddres)
{
    uint64_t notifyId = (uint32_t)(uintptr_t)notify;
    *notifyAddres = notifyId;
    return RT_ERROR_NONE;
}

rtError_t rtNotifyGetPhyInfo(rtNotify_t notify, uint32_t* phyDevId, uint32_t* tsId)
{
    uint64_t notifyId = (uint32_t)(uintptr_t)notify;
    // reset notify
    auto res = RunnerDB::GetById<sim::Notify>(notifyId);
    if (!res.has_value()) {
        HCCL_VM_ERROR("notify not found:{:d}", notifyId);
        return ACL_ERROR_INVALID_PARAM;
    }

    auto createCtx = RunnerDB::GetById<sim::Context>(res->create_ctx_id);
    if (!createCtx.has_value()) {
        HCCL_VM_ERROR("create ctx not found:{:d}", res->create_ctx_id);
        return ACL_ERROR_INVALID_PARAM;
    }

    auto devRes = RunnerDB::GetById<sim::Device>(createCtx->device_id);
    if (!devRes.has_value()) {
        HCCL_VM_ERROR("device id not found:{:d}", createCtx->device_id);
        return ACL_ERROR_INVALID_PARAM;
    }
    *phyDevId = devRes->physical_id;
    *tsId = 0;
    return RT_ERROR_NONE;
}

rtError_t rtNotifyGetPhyInfoExt(rtNotify_t notify, rtNotifyPhyInfo* notifyInfo)
{
    uint64_t notifyId = (uint32_t)(uintptr_t)notify;
    // reset notify
    auto res = RunnerDB::GetById<sim::Notify>(notifyId);
    if (!res.has_value()) {
        HCCL_VM_ERROR("notify not found:{:d}", notifyId);
        return ACL_ERROR_INVALID_PARAM;
    }

    auto createCtx = RunnerDB::GetById<sim::Context>(res->create_ctx_id);
    if (!createCtx.has_value()) {
        HCCL_VM_ERROR("create ctx not found:{:d}", res->create_ctx_id);
        return ACL_ERROR_INVALID_PARAM;
    }

    auto devRes = RunnerDB::GetById<sim::Device>(createCtx->device_id);
    if (!devRes.has_value()) {
        HCCL_VM_ERROR("device id not found:{:d}", createCtx->device_id);
        return ACL_ERROR_INVALID_PARAM;
    }
    notifyInfo->phyId = devRes->physical_id;
    notifyInfo->tsId = 0;
    notifyInfo->idType = 0;
    notifyInfo->shrId = (uint32_t)notifyId;
    notifyInfo->flag = 0;
    return RT_ERROR_NONE;
}

rtError_t rtNotifyGetAddrOffset(rtNotify_t notify, uint64_t* devAddrOffset)
{
    (void)notify;
    (void)devAddrOffset;
    return RT_ERROR_NONE;
}

#ifdef __cplusplus
}
#endif // __cplusplus
