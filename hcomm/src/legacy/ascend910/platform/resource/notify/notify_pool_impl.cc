/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include "device_capacity.h"
#include "sal_pub.h"
#include "adapter_hal.h"
#include "dlhal_function.h"
#include "adapter_rts.h"
#include "notify_pool_impl.h"

namespace hccl {
constexpr u32 NOTIFY_NORMAL = 0; // 常规算子
constexpr u32 NOTIFY_A2A = 1;    // alltoall
constexpr u32 NOTIFY_ALIGN = 2;  // atomic write，需要8字节对齐的notify，暂不支持alltoall

const std::string HCCL_ALLTOALL = "ALLTOALL";

// notify的offset对齐标准
constexpr u32 NOTIFY_OFFSET_ALIGN_EIGHT = 8;
constexpr u32 NOTIFY_OFFSET_ALIGN_FOUR = 4;

NotifyPoolImpl::NotifyPoolImpl(const s32 devicePhyId) : devicePhyId_(devicePhyId) {}

NotifyPoolImpl::~NotifyPoolImpl()
{
    HcclResult ret = Destroy();
    if (ret != HCCL_SUCCESS) {
        HCCL_WARNING("destroy NotifyPoolImpl resources failed, ret[%d]", ret);
    }
}

HcclResult NotifyPoolImpl::Init()
{
#ifndef HCCD
    CHK_RET(SalGetBareTgid(&pid_)); // 当前进程id
#else
    s32 psPid = 0;
    hrtDrvDeviceGetBareTgid(psPid);
    pid_ = psPid;
#endif
    return HCCL_SUCCESS;
}

HcclResult NotifyPoolImpl::Destroy()
{
    HCCL_INFO("NotifyPoolImpl Destroy.");
    for (u32 index = 0; index < NOTIFY_RES_MGR_NUM; ++index) {
        CHK_RET(DestroyRegisteredOpMap(index));
        CHK_RET(DestroyNotifyPoolIPCAsignedMap(index));
        CHK_RET(DestroyNotifyPoolDevIPCAsignedMap(index));
        CHK_RET(DestroyNotifyPoolNoIPCAsignedMap(index));
        CHK_RET(DestroyNotifyPoolDevNoIPCAsignedMap(index));
    }
    HCCL_INFO("NotifyPoolImpl Destroy success.");
    return HCCL_SUCCESS;
}

HcclResult NotifyPoolImpl::CreateNotify(
    std::shared_ptr<LocalIpcNotify>& localNotify, const s32 localDeviceId, const s32 remoteDeviceId,
    const NotifyLoadType type, bool withIpc, s64 recvId, u32 offsetAlignSize)
{
    std::vector<std::shared_ptr<LocalIpcNotify>> tmpNotifys;
    HcclResult ret = HCCL_SUCCESS;
    bool errorFlag = false;
    while (true) {
        std::shared_ptr<LocalIpcNotify> tmpNotify;
        EXCEPTION_CATCH((tmpNotify = std::make_shared<LocalIpcNotify>()), errorFlag = true);
        CHK_PRT_BREAK(
            !(tmpNotify) || errorFlag,
            HCCL_ERROR(
                "[NotifyPoolImpl][CreateNotify]create notify failed, "
                "errorFlag[%d]",
                errorFlag),
            errorFlag = true);

        ret = tmpNotify->Init(localDeviceId, remoteDeviceId, type);
        CHK_PRT_BREAK(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[NotifyPoolImpl][CreateNotify]localNotify init failed, "
                "ret[%d]",
                ret),
            errorFlag = true);

        // 申请到的notify offset不满足要求，再次申请
        bool isAligned = false;
        ret = IsNotifyOffsetAligned(tmpNotify, offsetAlignSize, isAligned);
        CHK_PRT_BREAK(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[NotifyPoolImpl][CreateNotify]IsNotifyOffsetAligned failed, "
                "ret[%d]",
                ret),
            errorFlag = true);
        if (!isAligned) {
            tmpNotifys.push_back(tmpNotify);
            HCCL_DEBUG(
                "CreateNotify id[%u] offset[%llu] is not support atomic write, create again", tmpNotify->notifyId_,
                tmpNotify->offset);
            continue;
        }

        localNotify = tmpNotify;
        HCCL_DEBUG("withIpc[%d], Is310PDevice[%d] recvId[%lld]", withIpc, Is310PDevice(), recvId);
        if (withIpc || Is310PDevice()) {
            ret = localNotify->SetIpc();
            CHK_PRT_BREAK(
                ret != HCCL_SUCCESS,
                HCCL_ERROR(
                    "[NotifyPoolImpl][CreateNotify]localNotify set ipc failed, "
                    "ret[%d]",
                    ret),
                errorFlag = true);
        }

        if (withIpc) {
            HCCL_DEBUG("withIpc[%d], Is310PDevice[%d] recvId[%lld]", withIpc, Is310PDevice(), recvId);
            ret = localNotify->Grant(recvId);
            CHK_PRT_BREAK(
                ret != HCCL_SUCCESS,
                HCCL_ERROR(
                    "[NotifyPoolImpl][CreateNotify]localNotify grant failed, "
                    "ret[%d]",
                    ret),
                errorFlag = true);
        }
        break;
    }

    tmpNotifys.clear(); // 释放不满足要求的notify
    if (errorFlag) {
        HCCL_ERROR("[NotifyPoolImpl][CreateNotify]localNotify create failed ,ret[%d]", ret);
        localNotify = nullptr;
        return ret;
    }

    return HCCL_SUCCESS;
}

HcclResult NotifyPoolImpl::DestroyNotifyPoolIPCAsignedMap(u32 index)
{
    std::unique_lock<std::mutex> lock(notifyResMgr_[index].notifyPoolIPCAsignedMutex);
    auto& notifyPoolIPCAsignedMap = notifyResMgr_[index].notifyPoolIPCAsignedMap;
    for (auto iter = notifyPoolIPCAsignedMap.begin(); iter != notifyPoolIPCAsignedMap.end(); iter++) {
        for (auto& it : iter->second) {
            if (DestroyNotify(it) != HCCL_SUCCESS) {
                HCCL_ERROR("[%s]DestroyNotify failed", __func__);
            }
        }
    }
    notifyPoolIPCAsignedMap.clear();
    HCCL_INFO("%s index[%u] success", __func__, index);
    return HCCL_SUCCESS;
}

HcclResult NotifyPoolImpl::DestroyNotifyPoolDevIPCAsignedMap(u32 index)
{
    std::unique_lock<std::mutex> lock(notifyResMgr_[index].notifyPoolIPCAsignedMutex);
    auto& notifyPoolDevIPCAsignedMap = notifyResMgr_[index].notifyPoolDevIPCAsignedMap;
    for (auto iter = notifyPoolDevIPCAsignedMap.begin(); iter != notifyPoolDevIPCAsignedMap.end(); iter++) {
        for (auto& it : iter->second) {
            if (DestroyNotify(it) != HCCL_SUCCESS) {
                HCCL_ERROR("[%s]DestroyNotify failed", __func__);
            }
        }
    }
    notifyPoolDevIPCAsignedMap.clear();
    HCCL_INFO("%s index[%u] success", __func__, index);
    return HCCL_SUCCESS;
}

HcclResult NotifyPoolImpl::DestroyNotifyPoolNoIPCAsignedMap(u32 index)
{
    std::unique_lock<std::mutex> lock(notifyResMgr_[index].notifyPoolNoIPCAsignedMutex);
    auto& notifyPoolNoIPCAsignedMap = notifyResMgr_[index].notifyPoolNoIPCAsignedMap;
    for (auto iter = notifyPoolNoIPCAsignedMap.begin(); iter != notifyPoolNoIPCAsignedMap.end(); iter++) {
        for (auto& it : iter->second) {
            if (DestroyNotify(it) != HCCL_SUCCESS) {
                HCCL_ERROR("[%s]DestroyNotify failed", __func__);
            }
        }
    }
    notifyPoolNoIPCAsignedMap.clear();
    HCCL_INFO("%s index[%u] success", __func__, index);
    return HCCL_SUCCESS;
}

HcclResult NotifyPoolImpl::DestroyNotifyPoolDevNoIPCAsignedMap(u32 index)
{
    std::unique_lock<std::mutex> lock(notifyResMgr_[index].notifyPoolNoIPCAsignedMutex);
    auto& notifyPoolDevNoIPCAsignedMap = notifyResMgr_[index].notifyPoolDevNoIPCAsignedMap;
    for (auto iter = notifyPoolDevNoIPCAsignedMap.begin(); iter != notifyPoolDevNoIPCAsignedMap.end(); iter++) {
        for (auto& it : iter->second) {
            if (DestroyNotify(it) != HCCL_SUCCESS) {
                HCCL_ERROR("[%s]DestroyNotify failed", __func__);
            }
        }
    }
    notifyPoolDevNoIPCAsignedMap.clear();
    HCCL_INFO("%s index[%u] success", __func__, index);
    return HCCL_SUCCESS;
}

HcclResult NotifyPoolImpl::DestroyNotify(std::shared_ptr<LocalIpcNotify>& localNotify)
{
    CHK_PTR_NULL(localNotify);
    CHK_RET(localNotify->Destroy());
    return HCCL_SUCCESS;
}
HcclResult
NotifyPoolImpl::RegisterOpMap(const std::string& tag, std::map<std::string, NotifyPoolIndicator>& registeredOpMap)
{
    auto iterTag = registeredOpMap.find(tag);
    if (iterTag == registeredOpMap.end()) {
        NotifyPoolIndicator indicator;
        registeredOpMap.insert(std::make_pair(tag, indicator));
    } else {
        HCCL_ERROR(
            "[NotifyPoolImpl][RegisterOp]register op to the notify pool failed, tag[%s] has existed", tag.c_str());
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

HcclResult NotifyPoolImpl::RegisterOp(const std::string& tag)
{
    std::string upTag = tag;
    std::transform(upTag.begin(), upTag.end(), upTag.begin(), ::toupper);
    bool hasAlltoAll = upTag.find(HCCL_ALLTOALL) != std::string::npos;
    HCCL_INFO("RegisterOp hasAlltoAll[%d]", hasAlltoAll);
    u32 notifyResIdx = hasAlltoAll ? NOTIFY_A2A : NOTIFY_NORMAL;
    auto tagDev = "Dev_" + tag;

    /* 此处可能会与并发，加锁 */
    std::unique_lock<std::mutex> lock1(notifyResMgr_[notifyResIdx].registeredOpMapMutex);
    CHK_RET(RegisterOpMap(tag, notifyResMgr_[notifyResIdx].registeredOpMap));
    CHK_RET(RegisterOpMap(tagDev, notifyResMgr_[notifyResIdx].registeredOpMap));
    lock1.unlock();

    // ALIGN的资源池不支持alltoall算子
    if (!hasAlltoAll) {
        std::unique_lock<std::mutex> lock2(notifyResMgr_[NOTIFY_ALIGN].registeredOpMapMutex);
        CHK_RET(RegisterOpMap(tag, notifyResMgr_[NOTIFY_ALIGN].registeredOpMap));
        CHK_RET(RegisterOpMap(tagDev, notifyResMgr_[NOTIFY_ALIGN].registeredOpMap));
        lock2.unlock();
    }
    HCCL_INFO("register op[%s] to the notify pool success.", tag.c_str());
    return HCCL_SUCCESS;
}

HcclResult
NotifyPoolImpl::UnregisterOpMap(const std::string& tag, std::map<std::string, NotifyPoolIndicator>& registeredOpMap)
{
    auto iterTag = registeredOpMap.find(tag);
    if (iterTag == registeredOpMap.end()) {
        HCCL_ERROR(
            "[NotifyPoolImpl][UnregisterOp]unregister op from the notify pool failed, tag[%s] has unregistered",
            tag.c_str());
        return HCCL_E_PARA;
    } else {
        registeredOpMap.erase(tag);
    }
    return HCCL_SUCCESS;
}

HcclResult NotifyPoolImpl::UnregisterOp(const std::string& tag)
{
    std::string upTag = tag;
    std::transform(upTag.begin(), upTag.end(), upTag.begin(), ::toupper);
    bool hasAlltoAll = upTag.find(HCCL_ALLTOALL) != std::string::npos;
    HCCL_INFO("UnregisterOp hasAlltoAll[%d]", hasAlltoAll);
    u32 notifyResIdx = hasAlltoAll ? NOTIFY_A2A : NOTIFY_NORMAL;
    auto tagDev = "Dev_" + tag;

    /* 此处可能会与并发，加锁 */
    std::unique_lock<std::mutex> lock1(notifyResMgr_[notifyResIdx].registeredOpMapMutex);
    CHK_RET(UnregisterOpMap(tag, notifyResMgr_[notifyResIdx].registeredOpMap));
    CHK_RET(UnregisterOpMap(tagDev, notifyResMgr_[notifyResIdx].registeredOpMap));
    lock1.unlock();

    if (!hasAlltoAll) {
        std::unique_lock<std::mutex> lock2(notifyResMgr_[NOTIFY_ALIGN].registeredOpMapMutex);
        CHK_RET(UnregisterOpMap(tag, notifyResMgr_[NOTIFY_ALIGN].registeredOpMap));
        CHK_RET(UnregisterOpMap(tagDev, notifyResMgr_[NOTIFY_ALIGN].registeredOpMap));
        lock2.unlock();
    }
    HCCL_INFO("unregister op[%s] from the notify pool success.", tag.c_str());
    return HCCL_SUCCESS;
}

HcclResult NotifyPoolImpl::DestroyRegisteredOpMap(u32 index)
{
    std::unique_lock<std::mutex> lock(notifyResMgr_[index].registeredOpMapMutex);
    notifyResMgr_[index].registeredOpMap.clear();
    HCCL_INFO("%s index[%u] success", __func__, index);
    return HCCL_SUCCESS;
}

HcclResult NotifyPoolImpl::IsNotifyOffsetAligned(
    std::shared_ptr<LocalIpcNotify>& localNotify, u32 offsetAlignSize, bool& isAligned)
{
    if (offsetAlignSize == INVALID_UINT) {
        isAligned = true;
    } else if (offsetAlignSize == NOTIFY_OFFSET_ALIGN_EIGHT) { // offset按照8byte对齐
        isAligned = (localNotify->offset % NOTIFY_OFFSET_ALIGN_EIGHT == 0);
    } else if (offsetAlignSize == NOTIFY_OFFSET_ALIGN_FOUR) { // offset按照4byte对齐
        isAligned = (localNotify->offset % NOTIFY_OFFSET_ALIGN_FOUR == 0);
    } else {
        HCCL_ERROR("IsNotifyOffsetAligned offsetAlignSize[%u] is invalid, only support 4 or 8", offsetAlignSize);
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

HcclResult NotifyPoolImpl::AllocIpc(
    const std::string& tag, s64 remote, s64 recvId, const s32 localDeviceId, const s32 remoteDeviceId,
    const NotifyLoadType type, std::shared_ptr<LocalIpcNotify>& localNotify, std::mutex& registeredOpMapMutex,
    std::map<std::string, NotifyPoolIndicator>& registeredOpMap, std::mutex& notifyPoolIPCAsignedMapMutex,
    std::map<s64, NotifyPoolIPCSub>& notifyPoolIPCAsignedMap, u32 offsetAlignSize)
{
    /* 此处可能会与并发，加锁 */
    std::unique_lock<std::mutex> lock(registeredOpMapMutex);
    auto iterTag = registeredOpMap.find(tag);
    CHK_PRT_RET(
        iterTag == registeredOpMap.end(), HCCL_ERROR("[NotifyPoolImpl][Alloc]tag[%s] is not registered.", tag.c_str()),
        HCCL_E_PARA);
    auto iterIdx = iterTag->second.notifyPoolIPC.find(remote);
    if (iterIdx == iterTag->second.notifyPoolIPC.end()) {
        iterTag->second.notifyPoolIPC.insert({remote, 0});
        iterIdx = iterTag->second.notifyPoolIPC.find(remote);
        CHK_PRT_RET(
            iterIdx == iterTag->second.notifyPoolIPC.end(),
            HCCL_ERROR(
                "[NotifyPoolImpl][Alloc]remote[%d] "
                "is not found.",
                remote),
            HCCL_E_PARA);
    }

    std::unique_lock<std::mutex> lockIPC(notifyPoolIPCAsignedMapMutex);
    auto iterRemoteDev = notifyPoolIPCAsignedMap.find(remote);
    if (iterRemoteDev != notifyPoolIPCAsignedMap.end()) {
        // 从资源池中遍历获取offset按照要求对齐的notify
        while (iterIdx->second < iterRemoteDev->second.size()) {
            bool isAligned = false;
            CHK_RET(IsNotifyOffsetAligned(iterRemoteDev->second[iterIdx->second], offsetAlignSize, isAligned));
            if (isAligned) {
                break; // notify满足对齐要求，退出循环
            }
            iterIdx->second++; // 继续遍历
        }

        if (iterIdx->second >= iterRemoteDev->second.size()) {
            CHK_RET(CreateNotify(localNotify, localDeviceId, remoteDeviceId, type, true, recvId, offsetAlignSize));

            iterRemoteDev->second.push_back(localNotify);
            HCCL_INFO(
                "create one notify in notify pool(ipc):tag[%s] remote[%d] total[%zu].", tag.c_str(), remote,
                iterRemoteDev->second.size());
        } else {
            localNotify = iterRemoteDev->second[iterIdx->second];
            CHK_SMART_PTR_NULL(localNotify);
            CHK_RET(localNotify->Grant(recvId));
            HCCL_INFO(
                "create one notify in notify pool(ipc):tag[%s] remote[%d] total[%zu].", tag.c_str(), remote,
                iterRemoteDev->second.size());
        }
    } else {
        CHK_RET(CreateNotify(localNotify, localDeviceId, remoteDeviceId, type, true, recvId, offsetAlignSize));

        NotifyPoolIPCSub tmpVec{{localNotify}};
        notifyPoolIPCAsignedMap.insert(std::make_pair(remote, tmpVec));
        HCCL_INFO(
            "create one notify in notify pool(ipc):tag[%s] remote[%d] total[%zu].", tag.c_str(), remote, tmpVec.size());
    }
    iterIdx->second++;

    HCCL_INFO("notify pool ipc alloc: tag[%s] remote[%d] used[%u]", tag.c_str(), remote, iterIdx->second);
    return HCCL_SUCCESS;
}

u32 NotifyPoolImpl::GetNotifyResIdx(const std::string& tag, u32 offsetAlignSize)
{
    std::string upTag = tag;
    std::transform(upTag.begin(), upTag.end(), upTag.begin(), ::toupper);
    bool hasAlltoAll = upTag.find(HCCL_ALLTOALL) != std::string::npos;
    u32 res = 0;
    // alltoall算子不需要使用atomic write，因此固定使用A2A的资源池
    if (hasAlltoAll) {
        res = NOTIFY_A2A;
    } else if (offsetAlignSize == NOTIFY_OFFSET_ALIGN_EIGHT) {
        res = NOTIFY_ALIGN;
    } else {
        res = NOTIFY_NORMAL;
    }
    HCCL_INFO("%s tag[%s], offsetAlignSize[%u], res[%u]", __func__, tag.c_str(), offsetAlignSize, res);
    return res;
}

HcclResult NotifyPoolImpl::Alloc(
    const std::string& tag, s64 remote, s64 recvId, const s32 localDeviceId, const s32 remoteDeviceId,
    const NotifyLoadType type, std::shared_ptr<LocalIpcNotify>& localNotify, u32 offsetAlignSize)
{
    u32 notifyResIdx = GetNotifyResIdx(tag, offsetAlignSize);
    HCCL_INFO(
        "[NotifyPoolImpl][Alloc]notifyResIdx[%u], tag[%s], remote[%lld], recvId[%lld], "
        "localDeviceId[%d], remoteDeviceId[%d], type[%d], offsetAlignSize[%u]",
        notifyResIdx, tag.c_str(), remote, recvId, localDeviceId, remoteDeviceId, type, offsetAlignSize);

    std::mutex& registeredOpMapMutex = notifyResMgr_[notifyResIdx].registeredOpMapMutex;
    std::mutex& notifyPoolIPCAsignedMapMutex = notifyResMgr_[notifyResIdx].notifyPoolIPCAsignedMutex;
    std::map<std::string, NotifyPoolIndicator>& registeredOpMap = notifyResMgr_[notifyResIdx].registeredOpMap;
    if (type == NotifyLoadType::HOST_NOTIFY) {
        std::map<s64, NotifyPoolIPCSub>& notifyPoolIPCAsignedMap = notifyResMgr_[notifyResIdx].notifyPoolIPCAsignedMap;
        CHK_RET(AllocIpc(
            tag, remote, recvId, localDeviceId, remoteDeviceId, type, localNotify, registeredOpMapMutex,
            registeredOpMap, notifyPoolIPCAsignedMapMutex, notifyPoolIPCAsignedMap, offsetAlignSize));
    } else if (type == NotifyLoadType::DEVICE_NOTIFY) { // 申请device上使用的notify资源
        std::map<s64, NotifyPoolIPCSub>& notifyPoolIPCAsignedMap
            = notifyResMgr_[notifyResIdx].notifyPoolDevIPCAsignedMap;
        auto tagDev = "Dev_" + tag;
        CHK_RET(AllocIpc(
            tagDev, remote, recvId, localDeviceId, remoteDeviceId, type, localNotify, registeredOpMapMutex,
            registeredOpMap, notifyPoolIPCAsignedMapMutex, notifyPoolIPCAsignedMap, offsetAlignSize));
    }
    return HCCL_SUCCESS;
}

HcclResult NotifyPoolImpl::AllocNoIpc(
    const std::string& tag, s64 remote, const s32 deviceId, const NotifyLoadType type,
    std::shared_ptr<LocalIpcNotify>& localNotify, std::mutex& registeredOpMapMutex,
    std::map<std::string, NotifyPoolIndicator>& registeredOpMap, std::mutex& notifyPoolNoIPCAsignedMapMutex,
    std::map<s64, NotifyPoolNoIPCSub>& notifyPoolNoIPCAsignedMap, u32 offsetAlignSize)
{
    /* 此处可能会与并发，加锁 */
    std::unique_lock<std::mutex> lock(registeredOpMapMutex);

    auto iterTag = registeredOpMap.find(tag);
    CHK_PRT_RET(
        iterTag == registeredOpMap.end(), HCCL_ERROR("[NotifyPool][Alloc]tag[%s] is not registered.", tag.c_str()),
        HCCL_E_PARA);

    auto iterIdx = iterTag->second.notifyPoolNoIPC.find(remote);
    if (iterIdx == iterTag->second.notifyPoolNoIPC.end()) {
        iterTag->second.notifyPoolNoIPC.insert({remote, 0});
        iterIdx = iterTag->second.notifyPoolNoIPC.find(remote);
        CHK_PRT_RET(
            iterIdx == iterTag->second.notifyPoolNoIPC.end(),
            HCCL_ERROR(
                "[NotifyPool][Alloc]remote[%lld] is "
                "not found.",
                remote),
            HCCL_E_PARA);
    }

    std::unique_lock<std::mutex> lockNoIPC(notifyPoolNoIPCAsignedMapMutex);
    auto iterRemoteDev = notifyPoolNoIPCAsignedMap.find(remote);
    if (iterRemoteDev != notifyPoolNoIPCAsignedMap.end()) {
        // 从资源池中获取offset按照要求对齐的notify
        while (iterIdx->second < iterRemoteDev->second.size()) {
            bool isAligned = false;
            CHK_RET(IsNotifyOffsetAligned(iterRemoteDev->second[iterIdx->second], offsetAlignSize, isAligned));
            if (isAligned) {
                break; // notify满足对齐要求，退出循环
            }
            iterIdx->second++; // 继续遍历
        }

        if (iterIdx->second >= iterRemoteDev->second.size()) {
            CHK_RET(CreateNotify(localNotify, deviceId, deviceId, type, false, -1, offsetAlignSize));
            iterRemoteDev->second.push_back(localNotify);
            HCCL_INFO(
                "create one notify in notify pool(no ipc):tag[%s] remote[%lld] total[%zu].", tag.c_str(), remote,
                iterRemoteDev->second.size());
        } else {
            localNotify = iterRemoteDev->second[iterIdx->second];
            CHK_SMART_PTR_NULL(localNotify);
        }
    } else {
        CHK_RET(CreateNotify(localNotify, deviceId, deviceId, type, false, -1, offsetAlignSize));

        NotifyPoolNoIPCSub tmpVec{localNotify};
        notifyPoolNoIPCAsignedMap.insert(std::make_pair(remote, tmpVec));
        HCCL_INFO(
            "create one notify in notify pool(no ipc):tag[%s] remote[%lld] total[%zu].", tag.c_str(), remote,
            tmpVec.size());
    }
    iterIdx->second++;

    HCCL_INFO("notify pool no ipc alloc: tag[%s] remote[%lld] used[%u]", tag.c_str(), remote, iterIdx->second);
    return HCCL_SUCCESS;
}

HcclResult NotifyPoolImpl::Alloc(
    const std::string& tag, s64 remote, const s32 deviceId, const NotifyLoadType type,
    std::shared_ptr<LocalIpcNotify>& localNotify, u32 offsetAlignSize)
{
    u32 notifyResIdx = GetNotifyResIdx(tag, offsetAlignSize);
    HCCL_INFO(
        "[NotifyPoolImpl][Alloc]notifyResIdx[%u], tag[%s], remote[%lld], deviceId[%d], type[%d], "
        "offsetAlignSize[%u]",
        notifyResIdx, tag.c_str(), remote, deviceId, type, offsetAlignSize);

    std::mutex& registeredOpMapMutex = notifyResMgr_[notifyResIdx].registeredOpMapMutex;
    std::mutex& notifyPoolNoIPCAsignedMapMutex = notifyResMgr_[notifyResIdx].notifyPoolNoIPCAsignedMutex;
    std::map<std::string, NotifyPoolIndicator>& registeredOpMap = notifyResMgr_[notifyResIdx].registeredOpMap;
    if (type == NotifyLoadType::HOST_NOTIFY) {
        std::map<s64, NotifyPoolNoIPCSub>& notifyPoolNoIPCAsignedMap
            = notifyResMgr_[notifyResIdx].notifyPoolNoIPCAsignedMap;
        CHK_RET(AllocNoIpc(
            tag, remote, deviceId, type, localNotify, registeredOpMapMutex, registeredOpMap,
            notifyPoolNoIPCAsignedMapMutex, notifyPoolNoIPCAsignedMap, offsetAlignSize));
    } else if (type == NotifyLoadType::DEVICE_NOTIFY) { // 申请device上使用的notify资源
        std::map<s64, NotifyPoolNoIPCSub>& notifyPoolNoIPCAsignedMap
            = notifyResMgr_[notifyResIdx].notifyPoolDevNoIPCAsignedMap;
        auto tagDev = "Dev_" + tag;
        CHK_RET(AllocNoIpc(
            tagDev, remote, deviceId, type, localNotify, registeredOpMapMutex, registeredOpMap,
            notifyPoolNoIPCAsignedMapMutex, notifyPoolNoIPCAsignedMap, offsetAlignSize));
    }
    return HCCL_SUCCESS;
}

HcclResult NotifyPoolImpl::Alloc(
    const std::string& tag, const RemoteRankInfo& info, const NotifyLoadType type,
    std::shared_ptr<LocalIpcNotify>& localNotify, u32 offsetAlignSize)
{
    HCCL_DEBUG(
        "[Alloc][IpcNotify]localPid[%016llx], remotePid[%016llx], localDeviceId[%d], remoteDeviceId[%d], "
        "remoteSdid[%x], offsetAlignSize[%u]",
        pid_, info.remotePid, devicePhyId_, info.remoteDeviceId, info.remoteSdid, offsetAlignSize);
    // 主从流下标为-1、rdma notify下标是remoteRank
    if (pid_ == info.remotePid && devicePhyId_ == info.remoteDeviceId && info.remoteSdid == INVALID_INT) {
        CHK_RET(Alloc(tag, info.remoteRank, info.remoteDeviceId, type, localNotify, offsetAlignSize));
    } else {
        // 统一使用remoteRank作为notifyPoolMap下标区分
        s64 remoteRank = static_cast<s64>(info.remoteRank);
        // 将(s32)SDID和(s32)Pid拼接成s64作为标志位, 高32位为SDID, 低32位为pid
        s64 recvId = ((static_cast<s64>(info.remoteSdid) & 0xFFFFFFFF) << 32)
                     | (static_cast<s64>(info.remotePid) & 0xFFFFFFFF);
        HCCL_INFO(
            "[Alloc][IpcNotify]recvSdid[%016llx], recvPid[%016llx], remoteRank[%u]", info.remoteSdid, info.remotePid,
            info.remoteRank);
        CHK_RET(Alloc(tag, remoteRank, recvId, devicePhyId_, info.remoteDeviceId, type, localNotify, offsetAlignSize));
    }

    return HCCL_SUCCESS;
}

HcclResult NotifyPoolImpl::ResetNotifyForDestRank(s64 destRank)
{
    // send/recv场景需要单点重置和对端的notify
    std::vector<u32> notifyResIdxs = {NOTIFY_NORMAL, NOTIFY_ALIGN};
    for (u32 notifyResIdx : notifyResIdxs) {
        std::unique_lock<std::mutex> lockIPC(notifyResMgr_[notifyResIdx].notifyPoolIPCAsignedMutex);
        const auto& notifyPoolDevIPCAsignedIt = notifyResMgr_[notifyResIdx].notifyPoolDevIPCAsignedMap.find(destRank);
        if (notifyPoolDevIPCAsignedIt == notifyResMgr_[notifyResIdx].notifyPoolDevIPCAsignedMap.end()) {
            HCCL_RUN_INFO(
                "[ResetNotify]remoteRank[%d] is not in notifyPoolDevIPCAsignedMap, notifyResIdx[%u]", destRank,
                notifyResIdx);
        } else {
            HCCL_RUN_INFO(
                "[ResetNotify]reset notifyPoolDevIPCAsignedIt remoteRank[%d], notifyResIdx[%u]", destRank,
                notifyResIdx);
            for (auto& it : notifyPoolDevIPCAsignedIt->second) {
                CHK_RET(hrtNotifyReset(it->ptr()));
            }
        }
        lockIPC.unlock();

        std::unique_lock<std::mutex> lockNoIPC(notifyResMgr_[notifyResIdx].notifyPoolNoIPCAsignedMutex);
        const auto& notifyPoolDevNoIPCAsignedIt
            = notifyResMgr_[notifyResIdx].notifyPoolDevNoIPCAsignedMap.find(destRank);
        if (notifyPoolDevNoIPCAsignedIt == notifyResMgr_[notifyResIdx].notifyPoolDevNoIPCAsignedMap.end()) {
            HCCL_RUN_INFO(
                "[ResetNotify]remoteRank[%d] is not in notifyPoolDevNoIPCAsignedMap, notifyResIdx[%u]", destRank,
                notifyResIdx);
        } else {
            HCCL_RUN_INFO(
                "[ResetNotify]reset notifyPoolDevNoIPCAsignedIt remoteRank[%d], notifyResIdx[%u]", destRank,
                notifyResIdx);
            for (auto& it : notifyPoolDevNoIPCAsignedIt->second) {
                CHK_RET(hrtNotifyReset(it->ptr()));
            }
        }
        lockNoIPC.unlock();
    }
    return HCCL_SUCCESS;
}

HcclResult NotifyPoolImpl::ResetNotify()
{
    HCCL_DEBUG("NotifyPoolImpl ResetNotify");
    for (u32 notifyResIdx = 0; notifyResIdx < NOTIFY_RES_MGR_NUM; ++notifyResIdx) {
        HCCL_INFO("NotifyPoolImpl ResetNotify, notifyResIdx[%u]", notifyResIdx);
        std::unique_lock<std::mutex> lockIPC(notifyResMgr_[notifyResIdx].notifyPoolIPCAsignedMutex);
        auto& notifyPoolDevIPCAsignedMap = notifyResMgr_[notifyResIdx].notifyPoolDevIPCAsignedMap;
        for (auto iter = notifyPoolDevIPCAsignedMap.begin(); iter != notifyPoolDevIPCAsignedMap.end(); iter++) {
            for (auto& it : iter->second) {
                CHK_RET(hrtNotifyReset(it->ptr()));
            }
        }
        lockIPC.unlock();

        std::unique_lock<std::mutex> lockNoIPC(notifyResMgr_[notifyResIdx].notifyPoolNoIPCAsignedMutex);
        auto& notifyPoolDevNoIPCAsignedMap = notifyResMgr_[notifyResIdx].notifyPoolDevNoIPCAsignedMap;
        for (auto iter = notifyPoolDevNoIPCAsignedMap.begin(); iter != notifyPoolDevNoIPCAsignedMap.end(); iter++) {
            for (auto& it : iter->second) {
                CHK_RET(hrtNotifyReset(it->ptr()));
            }
        }
        lockNoIPC.unlock();
    }
    return HCCL_SUCCESS;
}
} // namespace hccl
