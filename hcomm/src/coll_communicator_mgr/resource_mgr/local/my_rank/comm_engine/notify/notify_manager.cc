/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "notify_manager.h"
#include "adapter_hal_pub.h"
#include "adapter_rts_common.h"
#include "device_capacity.h"
#include "aicpu_launch_manager.h"

namespace hccl {
#ifndef CCL_KERNEL_AICPU
NotifyManager::NotifyManager(std::string commId, aclrtBinHandle binHandle, const ManagerCallbacks& callbacks)
    : commId_(commId),
      binHandle_(binHandle),
      callbacks_(callbacks)
{}
#endif

HcclResult NotifyManager::InitNotifys(
    std::istringstream& iss, size_t notifyNum, std::vector<std::unique_ptr<LocalNotify>>& newNotifys)
{
    newNotifys.reserve(newNotifys.size() + notifyNum);
    for (u32 idx = 0; idx < notifyNum; idx++) {
        std::unique_ptr<LocalNotify> notify;
        EXCEPTION_CATCH(notify = std::make_unique<LocalNotify>(), return HCCL_E_PTR);
        HcclSignalInfo notifyInfo;
        iss.read(reinterpret_cast<char_t*>(&notifyInfo), sizeof(notifyInfo));
        CHK_RET(notify->Init(notifyInfo, NotifyLoadType::DEVICE_NOTIFY));
        newNotifys.emplace_back(std::move(notify));
        HCCL_INFO(
            "[NotifyManager][Init]local notify init success, resId[%u], tsId:%d, devId[%u]", notifyInfo.resId,
            notifyInfo.tsId, notifyInfo.devId);
    }
    return HCCL_SUCCESS;
}

HcclResult
NotifyManager::ParseBinNotifys(const std::string& uniqueIdStr, std::vector<std::unique_ptr<LocalNotify>>& newNotifys)
{
    bool isDeviceSid = false;
    CHK_RET(GetRunSideIsDevice(isDeviceSid));
    NotifyLoadType loadType;
    size_t notifyNum = 0;
    if (!isDeviceSid) {
        HCCL_ERROR("[NotifyManager][%s] not in deviceSide", __func__);
        return HCCL_E_NOT_SUPPORT;
    } else {
        CHK_PRT_RET(uniqueIdStr.empty(), HCCL_ERROR("[HcclThread][%s] uniqueIdStr is empty"), HCCL_E_INTERNAL);
        std::istringstream iss(uniqueIdStr);
        iss.read(reinterpret_cast<char_t*>(&loadType), sizeof(loadType));
        iss.read(reinterpret_cast<char_t*>(&notifyNum), sizeof(notifyNum));
        CHK_RET(InitNotifys(iss, notifyNum, newNotifys));
    }
    HCCL_RUN_INFO(
        "[NotifyManager][%s] recover success, notifyNum[%zu], notifyType[%d], uniqueIdSize[%zu]", __func__, notifyNum,
        loadType, uniqueIdStr.size());
    return HCCL_SUCCESS;
}

#ifndef CCL_KERNEL_AICPU
std::string
NotifyManager::GetBinNotifys(std::vector<std::unique_ptr<LocalNotify>>& newNotifys, const NotifyLoadType notifyType)
{
    std::string uniqueIdStr;
    std::ostringstream oss;
    size_t notifyNum = newNotifys.size();
    oss.write(reinterpret_cast<const char_t*>(&notifyType), sizeof(notifyType));
    oss.write(reinterpret_cast<const char_t*>(&notifyNum), sizeof(notifyNum));
    HcclResult ret = HCCL_SUCCESS;
    for (u32 idx = 0; idx < notifyNum; idx++) {
        HcclSignalInfo notifyInfo;
        ret = newNotifys[idx]->GetNotifyData(notifyInfo);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("[NotifyManager][%s] GetNotifyData failed, ret[%d]", __func__, ret);
            std::string temp = std::string();
            return temp;
        }
        HCCL_INFO(
            "[NotifyManager][%s] get local notify data success, resId[%u], tsId:%d, devId[%u]", __func__,
            notifyInfo.resId, notifyInfo.tsId, notifyInfo.devId);
        oss.write(reinterpret_cast<const char_t*>(&notifyInfo), sizeof(notifyInfo));
    }
    HCCL_RUN_INFO(
        "[NotifyManager][%s] GetUniqueId success, notifyNum[%zu], notifyType[%u], uniqueId[%s]", __func__, notifyNum,
        notifyType, oss.str().c_str());
    uniqueIdStr = oss.str();
    return uniqueIdStr;
}

HcclResult NotifyManager::NotifyTypeToNotifyLoadType(::NotifyType notifyType, NotifyLoadType& notifyLoadType)
{
    switch (notifyType) {
        case ::NOTIFY_TYPE_RTS_NOTIFY:
        case ::NOTIFY_TYPE_RTS_EVENT:
            notifyLoadType = NotifyLoadType::HOST_NOTIFY;
            break;
        case ::NOTIFY_TYPE_DEVICE_MEM:
            notifyLoadType = NotifyLoadType::DEVICE_NOTIFY;
            break;
        default:
            HCCL_ERROR("[NotifyManager] Unknown comm notifyType notifyLoadType: %d", notifyType);
            return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

HcclResult NotifyManager::HcclAllocNotify(
    CommEngine commEngine, ::NotifyType notifyType, uint32_t notifyNum, NotifyHandle** notifyHandleList)
{
    std::lock_guard<std::mutex> lock(notifyMutex_);
    notifys_.reserve(notifys_.size() + notifyNum);

    std::vector<std::unique_ptr<LocalNotify>> newNotifys;
    newNotifys.reserve(notifyNum);
    NotifyLoadType notifyLoadType;
    CHK_PRT(NotifyTypeToNotifyLoadType(notifyType, notifyLoadType));
    bool isAicpu = (commEngine == CommEngine::COMM_ENGINE_AICPU || commEngine == CommEngine::COMM_ENGINE_AICPU_TS);

    // 构建 LocalNotify
    for (uint32_t i = 0; i < notifyNum; ++i) {
        std::unique_ptr<LocalNotify> notify;
        EXCEPTION_CATCH(notify = std::make_unique<LocalNotify>(), return HCCL_E_PTR);
        CHK_RET(notify->Init(notifyLoadType));
        if (Is310PDevice()) {
            CHK_RET(notify->SetIpc());
        }
        newNotifys.emplace_back(std::move(notify));
    }

    std::unique_ptr<NotifyHandle[]> handles;
    EXCEPTION_CATCH(handles = std::make_unique<NotifyHandle[]>(notifyNum), return HCCL_E_PTR);
    if (isAicpu) {
        if (!callbacks_.getAicpuCommState()) {
            HcclResult ret = callbacks_.kernelLaunchAicpuCommInit();
            CHK_PRT_RET(
                ret != HCCL_SUCCESS, HCCL_ERROR("[%s] kernelLaunchAicpuCommInit failed, return [%d].", __func__, ret),
                ret);
            callbacks_.setAicpuCommState(true);
        }
        CHK_RET(AicpuLaunchMgr::NotifyKernelLaunchAlloc(newNotifys, commId_, handles, binHandle_));
        for (uint32_t i = 0; i < notifyNum; ++i) {
            HCCL_INFO("[NotifyManager][%s] aicpu handles[%u] = [%llu]", __func__, i, handles[i]);
        }
    } else {
        for (uint32_t i = 0; i < notifyNum; ++i) {
            handles[i] = reinterpret_cast<NotifyHandle>(newNotifys[i].get());
            HCCL_INFO("[NotifyManager][%s] host handles[%u] = [%llu]", __func__, i, handles[i]);
        }
    }
    for (uint32_t i = 0; i < notifyNum; ++i) {
        LocalNotify* local = newNotifys[i].get();
        NotifyInfo info{commEngine, notifyType, isAicpu, handles[i]};
        notifysInfo_[local] = info;
    }
    // 插入到 notifys_ 尾部
    notifys_.insert(
        notifys_.end(), std::make_move_iterator(newNotifys.begin()), std::make_move_iterator(newNotifys.end()));

    handleBlocks_.push_back(std::move(handles));
    *notifyHandleList = handleBlocks_.back().get();
    return HCCL_SUCCESS;
}

HcclResult NotifyManager::HcommFreeNotify(uint32_t notifyNum, NotifyHandle* notifyHandleList)
{
    std::lock_guard<std::mutex> lock(notifyMutex_);

    std::vector<LocalNotify*> localNotifys;
    localNotifys.reserve(notifyNum);
    std::vector<NotifyHandle> aicpuNotifys;

    // 1. 预扫描，判断是否为 AICPU，并收集 LocalNotify 指针
    for (uint32_t i = 0; i < notifyNum; ++i) {
        NotifyHandle handle = notifyHandleList[i];
        HCCL_INFO("[NotifyManager][%s] handles[%u] = [%llu]", __func__, i, handle);
        auto itInfo = std::find_if(notifysInfo_.begin(), notifysInfo_.end(), [handle](const auto& pair) {
            return pair.second.notifyHandle == handle;
        });
        if (itInfo == notifysInfo_.end()) {
            HCCL_RUN_WARNING("[NotifyManager][%s] handle[%llu] not found in notifysInfo_", __func__, handle);
            continue;
        }
        LocalNotify* localNotify = itInfo->first;
        const NotifyInfo& info = itInfo->second;
        if (info.isAicpu) {
            aicpuNotifys.push_back(handle);
        }
        localNotifys.push_back(localNotify);
    }

    // 2. 先释放 Device 侧（若失败则直接返回，不动 Host）
    bool hasAicpu = !aicpuNotifys.empty();
    if (hasAicpu) {
        HcclResult ret = AicpuLaunchMgr::NotifyKernelLaunchFree(aicpuNotifys, aicpuNotifys.size(), commId_, binHandle_);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR(
                "[NotifyManager][%s] NotifyKernelLaunchFree failed ret[%d], num[%zu], skip host erase", __func__, ret,
                aicpuNotifys.size());
            return ret; // 保留 Host 状态以便恢复
        }
    }

    // 3. 成功后再移除 Host 侧
    for (auto* localNotify : localNotifys) {
        notifysInfo_.erase(localNotify);
        auto it
            = std::find_if(notifys_.begin(), notifys_.end(), [localNotify](const std::unique_ptr<LocalNotify>& ptr) {
                  return ptr.get() == localNotify;
              });
        if (it != notifys_.end()) {
            notifys_.erase(it);
        }
    }

    // 4. 删除对应的 handle block
    auto itBlock = std::find_if(
        handleBlocks_.begin(), handleBlocks_.end(), [notifyHandleList](const std::unique_ptr<NotifyHandle[]>& block) {
            return block.get() == notifyHandleList;
        });
    if (itBlock == handleBlocks_.end()) {
        HCCL_RUN_WARNING("[NotifyManager][%s] itBlock not found for notifyHandleList[%p]", __func__, notifyHandleList);
    } else {
        handleBlocks_.erase(itBlock);
    }
    return HCCL_SUCCESS;
}

HcclResult NotifyManager::ResetAllocLocalNotifies()
{
    std::lock_guard<std::mutex> lock(notifyMutex_);
    HCCL_INFO(
        "[NotifyManager][ResetAllocLocalNotifies] start, Hcom[%s], notifyNum[%zu]", commId_.c_str(), notifys_.size());
    for (size_t i = 0; i < notifys_.size(); ++i) {
        const auto& notifyPtr = notifys_[i];
        if (notifyPtr == nullptr) {
            continue;
        }
        HcclRtNotify rtNotify = notifyPtr->ptr();
        if (rtNotify == nullptr) {
            continue;
        }
        HcclResult ret = hrtNotifyReset(rtNotify);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR(
                "[NotifyManager][ResetAllocLocalNotifies] hrtNotifyReset failed, Hcom[%s], notifyIdx[%zu], "
                "notifyId[%u], "
                "ret[0x%016llx]",
                commId_.c_str(), i, notifyPtr->notifyId_, HCCL_ERROR_CODE(ret));
            return ret;
        }
        HCCL_INFO(
            "[NotifyManager][ResetAllocLocalNotifies] reset notify success, Hcom[%s], notifyIdx[%zu], notifyId[%u], "
            "notifyPtr[%p]",
            commId_.c_str(), i, notifyPtr->notifyId_, rtNotify);
    }
    HCCL_INFO("[NotifyManager][ResetAllocLocalNotifies] finish, Hcom[%s]", commId_.c_str());
    return HCCL_SUCCESS;
}
#endif
} // namespace hccl
