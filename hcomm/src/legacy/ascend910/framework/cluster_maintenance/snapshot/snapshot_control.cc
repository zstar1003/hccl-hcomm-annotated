/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "snapshot_control.h"
#include "adapter_rts_common.h"
#include "adapter_hccp_common.h"
#include "externalinput.h"
#include "transport_pub.h"
#include "rt_external.h"

namespace hccl {

bool SnapshotControl::registered = false;

uint32_t PreProcessCallback(int32_t devId, [[maybe_unused]] void* args)
{
    HCCL_RUN_INFO("[Snapshot] PreProcess callback, devId[%d]", devId);
    HcclResult ret = SnapshotControl::GetInstance(devId).PreProcess();
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Snapshot] PreProcess fail, devId[%d].", devId), ret);
    HCCL_RUN_INFO("[Snapshot] PreProcess success, devId[%d]", devId);
    return 0;
}

uint32_t PostProcessCallback(int32_t devId, [[maybe_unused]] void* args)
{
    HCCL_RUN_INFO("[Snapshot] PostProcess callback, devId[%d]", devId);
    HcclResult ret = SnapshotControl::GetInstance(devId).PostProcess();
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Snapshot] PostProcess fail, devId[%d].", devId), ret);
    HCCL_RUN_INFO("[Snapshot] PostProcess success, devId[%d]", devId);
    return 0;
}

uint32_t RecoveryCallback(int32_t devId, [[maybe_unused]] void* args)
{
    HCCL_RUN_INFO("[Snapshot] Recovery callback, devId[%d]", devId);
    HcclResult ret = SnapshotControl::GetInstance(devId).Recovery();
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Snapshot] Recovery fail, devId[%d].", devId), ret);
    HCCL_RUN_INFO("[Snapshot] Recovery success, devId[%d]", devId);
    return 0;
}

HcclResult ResgisterSnapshotCallback()
{
    rtError_t ret = aclrtSnapShotCallbackRegister(ACL_RT_SNAPSHOT_LOCK_PRE, PreProcessCallback, nullptr);
    CHK_PRT_RET(
        ret != ACL_SUCCESS,
        HCCL_ERROR(
            "[SnapshotControl]errNo[0x%016llx] register preprocess callback fail, ret[%d]",
            HCCL_ERROR_CODE(HCCL_E_RUNTIME), ret),
        HCCL_E_RUNTIME);
    ret = aclrtSnapShotCallbackRegister(ACL_RT_SNAPSHOT_UNLOCK_POST, PostProcessCallback, nullptr);
    CHK_PRT_RET(
        ret != ACL_SUCCESS,
        HCCL_ERROR(
            "[SnapshotControl]errNo[0x%016llx] register postprocess callback fail, ret[%d]",
            HCCL_ERROR_CODE(HCCL_E_RUNTIME), ret),
        HCCL_E_RUNTIME);
    ret = aclrtSnapShotCallbackRegister(ACL_RT_SNAPSHOT_RESTORE_POST, RecoveryCallback, nullptr);
    CHK_PRT_RET(
        ret != ACL_SUCCESS,
        HCCL_ERROR(
            "[SnapshotControl]errNo[0x%016llx] register recovery callback fail, ret[%d]",
            HCCL_ERROR_CODE(HCCL_E_RUNTIME), ret),
        HCCL_E_RUNTIME);
    return HCCL_SUCCESS;
}

HcclResult UnResgisterSnapshotCallback()
{
    rtError_t ret = aclrtSnapShotCallbackUnregister(ACL_RT_SNAPSHOT_LOCK_PRE, PreProcessCallback);
    CHK_PRT_RET(
        ret != ACL_SUCCESS,
        HCCL_ERROR(
            "[SnapshotControl]errNo[0x%016llx] unregister preprocess callback fail, ret[%d]",
            HCCL_ERROR_CODE(HCCL_E_RUNTIME), ret),
        HCCL_E_RUNTIME);
    ret = aclrtSnapShotCallbackUnregister(ACL_RT_SNAPSHOT_UNLOCK_POST, PostProcessCallback);
    CHK_PRT_RET(
        ret != ACL_SUCCESS,
        HCCL_ERROR(
            "[SnapshotControl]errNo[0x%016llx] unregister postprocess callback fail, ret[%d]",
            HCCL_ERROR_CODE(HCCL_E_RUNTIME), ret),
        HCCL_E_RUNTIME);
    ret = aclrtSnapShotCallbackUnregister(ACL_RT_SNAPSHOT_RESTORE_POST, RecoveryCallback);
    CHK_PRT_RET(
        ret != ACL_SUCCESS,
        HCCL_ERROR(
            "[SnapshotControl]errNo[0x%016llx] unregister recovery callback fail, ret[%d]",
            HCCL_ERROR_CODE(HCCL_E_RUNTIME), ret),
        HCCL_E_RUNTIME);
    return HCCL_SUCCESS;
}

SnapshotControl& SnapshotControl::GetInstance(s32 deviceLogicId)
{
    static SnapshotControl instances[MAX_MODULE_DEVICE_NUM];
    if (static_cast<u32>(deviceLogicId) >= MAX_MODULE_DEVICE_NUM) {
        return instances[0];
    }
    instances[deviceLogicId].deviceLogicId_ = deviceLogicId;
    return instances[deviceLogicId];
}

SnapshotControl::SnapshotControl()
{
    if (!registered) {
        DevType devType = DevType::DEV_TYPE_COUNT;
        HcclResult ret = hrtGetDeviceType(devType);
        CHK_PRT_CONT(ret != HCCL_SUCCESS, HCCL_ERROR("[SnapshotControl] Get device type fail, ret[%u]", ret));
        if (devType == DevType::DEV_TYPE_910B || devType == DevType::DEV_TYPE_910_93) {
            (void)ResgisterSnapshotCallback();
            registered = true;
        }
    }
}

SnapshotControl::~SnapshotControl()
{
    if (registered) {
        (void)UnResgisterSnapshotCallback();
        registered = false;
    }

    std::lock_guard<std::mutex> lock(commMutex_);
    commCallbacks_.clear();
}

HcclResult SnapshotControl::SetStatus(SnapshotStatus status)
{
    std::lock_guard<std::mutex> lock(statusMutex_);
    CHK_PRT_RET(
        status_ == status,
        HCCL_DEBUG(
            "[SnapshotControl][SetStatus]status has already been set to [%u], deviceLogicId[%d]", status_,
            deviceLogicId_),
        HCCL_SUCCESS);
    status_ = status;
    HCCL_RUN_INFO("[SnapshotControl][SetStatus]set status to [%u], deviceLogicId[%d]", status_, deviceLogicId_);
    return HCCL_SUCCESS;
}

SnapshotStatus SnapshotControl::GetStatus()
{
    std::lock_guard<std::mutex> lock(statusMutex_);
    return status_;
}

HcclResult SnapshotControl::RegisterComm(
    const std::string& identifier, SnapshotSetInvalidComm setInvalidCommCallback,
    SnapshotCheckPreProcess preProcessCallback, SnapshotCheckPostProcess postProcessCallback)
{
    std::lock_guard<std::mutex> lock(commMutex_);
    if (commCallbacks_.find(identifier) != commCallbacks_.end()) {
        HCCL_WARNING(
            "[SnapshotControl][RegisterComm] comm[%s] has already registered, devId[%d].", identifier.c_str(),
            deviceLogicId_);
        return HCCL_SUCCESS;
    }
    SnapshotCallbacks callbacks = {setInvalidCommCallback, preProcessCallback, postProcessCallback};
    commCallbacks_.emplace(identifier, callbacks);
    HCCL_RUN_INFO(
        "[SnapshotControl][RegisterComm] comm[%s] register to snapshot control, devId[%d].", identifier.c_str(),
        deviceLogicId_);
    return HCCL_SUCCESS;
}

HcclResult SnapshotControl::RegisterBackup(const std::string& identifier, u32 backupDevicePhyId)
{
    std::lock_guard<std::mutex> lock(commMutex_);
    backupDeviceCount_[backupDevicePhyId].Ref();
    HCCL_RUN_INFO(
        "[SnapshotControl][RegisterBackup] comm[%s] register backup device to snapshot control, "
        "devId[%d], backupDevPhyId[%u].",
        identifier.c_str(), deviceLogicId_, backupDevicePhyId);
    return HCCL_SUCCESS;
}

HcclResult SnapshotControl::UnRegisterComm(const std::string& identifier)
{
    std::lock_guard<std::mutex> lock(commMutex_);
    auto callbackIter = commCallbacks_.find(identifier);
    if (callbackIter == commCallbacks_.end()) {
        HCCL_RUN_WARNING(
            "[SnapshotControl][UnRegisterComm] "
            "comm[%s] has not registered and cannot be unregistered, devId[%d].",
            identifier.c_str(), deviceLogicId_);
        return HCCL_SUCCESS;
    }
    commCallbacks_.erase(callbackIter);
    HCCL_RUN_INFO(
        "[SnapshotControl][UnRegisterComm] comm[%s] unregister from snapshot control, devId[%d].", identifier.c_str(),
        deviceLogicId_);
    if (commCallbacks_.empty()) {
        HCCL_RUN_INFO(
            "[SnapshotControl][UnRegisterComm] all comms have unregistered from snapshot control, devId[%d].",
            deviceLogicId_);
    }
    return HCCL_SUCCESS;
}

HcclResult SnapshotControl::UnRegisterBackup(const std::string& identifier, u32 backupDevicePhyId)
{
    std::lock_guard<std::mutex> lock(commMutex_);
    auto backupIter = backupDeviceCount_.find(backupDevicePhyId);
    if (backupIter == backupDeviceCount_.end()) {
        HCCL_WARNING(
            "[SnapshotControl][UnRegisterBackup] comm[%s] backupDevicePhyId[%u] has not been registered, "
            "devId[%d]",
            identifier.c_str(), backupDevicePhyId, deviceLogicId_);
        return HCCL_SUCCESS;
    }
    int count = backupDeviceCount_[backupDevicePhyId].Unref();
    if (count < 0) {
        HCCL_WARNING(
            "[SnapshotControl][UnRegisterBackup] comm[%s] unregister backup device exceed, "
            "devId[%d], count[%d]",
            identifier.c_str(), deviceLogicId_, count);
        return HCCL_SUCCESS;
    }
    HCCL_RUN_INFO(
        "[SnapshotControl][UnRegisterBackup] release backup device phydId[%u], comm[%s], "
        "devId[%d], count[%d]",
        backupDevicePhyId, identifier.c_str(), deviceLogicId_, count);
    if (count == 0) {
        backupDeviceCount_.erase(backupIter);
        HCCL_RUN_INFO(
            "[SnapshotControl][UnRegisterBackup] backup device phydId[%u] is totally released, "
            "devId[%d], count[%d]",
            backupDevicePhyId, deviceLogicId_, count);
    }
    return HCCL_SUCCESS;
}

HcclResult SnapshotControl::CheckCommsPreProcess()
{
    std::lock_guard<std::mutex> lock(commMutex_);
    for (auto callbackIter : commCallbacks_) {
        CHK_RET(callbackIter.second.preProcessCallback());
        HCCL_RUN_INFO(
            "[SnapshotControl][CheckCommsPreProcess] comm[%s] check pre-process success, devId[%d].",
            callbackIter.first.c_str(), deviceLogicId_);
    }
    HCCL_INFO("[SnapshotControl][CheckCommsPreProcess] devId[%d], check pre-process success finish.", deviceLogicId_);
    return HCCL_SUCCESS;
}

HcclResult SnapshotControl::DevicePreProcess()
{
    std::lock_guard<std::mutex> lock(commMutex_);
    CHK_RET(hrtGetDevicePhyIdByIndex(static_cast<u32>(deviceLogicId_), devicePhyId_, true));

    HcclResult ret = SnapShotSaveAction(
        static_cast<s32>(NICDeployment::NIC_DEPLOYMENT_DEVICE), devicePhyId_,
        HcclSaveSnapShotAction::HCCL_SAVE_SNAPSHOT_ACTION_PRE_PROCESSING);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[SnapshotControl][DevicePreProcess] call SnapShotSaveAction fail, devicePhyId[%u], action[%u]",
            devicePhyId_, HcclSaveSnapShotAction::HCCL_SAVE_SNAPSHOT_ACTION_PRE_PROCESSING),
        ret);
    HCCL_INFO(
        "[SnapshotControl][DevicePreProcess] device[%u] do device pre-process success, devId[%d].", devicePhyId_,
        deviceLogicId_);

    for (auto backupIter : backupDeviceCount_) {
        ret = SnapShotSaveAction(
            static_cast<s32>(NICDeployment::NIC_DEPLOYMENT_DEVICE), backupIter.first,
            HcclSaveSnapShotAction::HCCL_SAVE_SNAPSHOT_ACTION_PRE_PROCESSING);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[SnapshotControl][DevicePreProcess] call SnapShotSaveAction fail, backup devicePhyId[%u], "
                "action[%u]",
                backupIter.first, HcclSaveSnapShotAction::HCCL_SAVE_SNAPSHOT_ACTION_PRE_PROCESSING),
            ret);
        HCCL_INFO(
            "[SnapshotControl][DevicePreProcess] backup device[%u] do device pre-process success, devId[%d].",
            backupIter.first, deviceLogicId_);
    }
    return HCCL_SUCCESS;
}

HcclResult SnapshotControl::PreProcess()
{
    CHK_RET(SetStatus(SnapshotStatus::PRE_SNAPSHOT));
    CHK_RET(CheckCommsPreProcess());

    if (commCallbacks_.size() != 0) {
        CHK_RET(DevicePreProcess());
    }

    HCCL_INFO(
        "[SnapshotControl][PreProcess] snapshot pre-process success, devId[%d], devPhyId[%u].", deviceLogicId_,
        devicePhyId_);
    return HCCL_SUCCESS;
}

HcclResult SnapshotControl::CheckCommsPostProcess()
{
    std::lock_guard<std::mutex> lock(commMutex_);
    for (auto callbackIter : commCallbacks_) {
        CHK_RET(callbackIter.second.postProcessCallback());
        HCCL_RUN_INFO(
            "[SnapshotControl][CheckCommsPostProcess] comm[%s] check post-process success, devId[%d].",
            callbackIter.first.c_str(), deviceLogicId_);
    }
    HCCL_INFO("[SnapshotControl][CheckCommsPostProcess] devId[%d], check post-process finish.", deviceLogicId_);
    return HCCL_SUCCESS;
}

HcclResult SnapshotControl::DevicePostProcess()
{
    std::lock_guard<std::mutex> lock(commMutex_);
    CHK_RET(hrtGetDevicePhyIdByIndex(static_cast<u32>(deviceLogicId_), devicePhyId_, true));

    HcclResult ret = SnapShotSaveAction(
        static_cast<s32>(NICDeployment::NIC_DEPLOYMENT_DEVICE), devicePhyId_,
        HcclSaveSnapShotAction::HCCL_SAVE_SNAPSHOT_ACTION_POST_PROCESSING);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[SnapshotControl][DevicePostProcess] call SnapShotSaveAction fail, devicePhyId[%u], action[%u]",
            devicePhyId_, HcclSaveSnapShotAction::HCCL_SAVE_SNAPSHOT_ACTION_POST_PROCESSING),
        ret);
    HCCL_INFO(
        "[SnapshotControl][DevicePostProcess] device[%u] do device post-process success, devId[%d].", devicePhyId_,
        deviceLogicId_);

    for (auto backupIter : backupDeviceCount_) {
        ret = SnapShotSaveAction(
            static_cast<s32>(NICDeployment::NIC_DEPLOYMENT_DEVICE), backupIter.first,
            HcclSaveSnapShotAction::HCCL_SAVE_SNAPSHOT_ACTION_POST_PROCESSING);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[SnapshotControl][DevicePostProcess] call SnapShotSaveAction fail, backup devicePhyId[%u], "
                "action[%u]",
                backupIter.first, HcclSaveSnapShotAction::HCCL_SAVE_SNAPSHOT_ACTION_POST_PROCESSING),
            ret);
        HCCL_INFO(
            "[SnapshotControl][DevicePostProcess] backup device[%u] do device post-process success, devId[%d].",
            backupIter.first, deviceLogicId_);
    }
    return HCCL_SUCCESS;
}

HcclResult SnapshotControl::PostProcess()
{
    if (commCallbacks_.size() != 0) {
        CHK_RET(DevicePostProcess());
    }

    if (GetStatus() != SnapshotStatus::RESTORE_SNAPSHOT) {
        CHK_RET(SetStatus(SnapshotStatus::POST_SNAPSHOT));
        CHK_RET(CheckCommsPostProcess());
    }
    CHK_RET(SetStatus(SnapshotStatus::DEFAULT));

    HCCL_INFO(
        "[SnapshotControl][PostProcess] snapshot post-process success, devId[%d], devPhyId[%u].", deviceLogicId_,
        devicePhyId_);
    return HCCL_SUCCESS;
}

HcclResult SnapshotControl::MarkInvalidComms()
{
    std::lock_guard<std::mutex> lock(commMutex_);
    for (auto callbackIter : commCallbacks_) {
        CHK_RET(callbackIter.second.setInvalidCommCallback(true));
        HCCL_RUN_INFO(
            "[SnapshotControl][MarkInvalidComms] comm[%s] has been marked as invalid comm, devId[%d].",
            callbackIter.first.c_str(), deviceLogicId_);
    }
    HCCL_INFO("[SnapshotControl][MarkInvalidComms] devId[%d], mark invalid comms finish.", deviceLogicId_);
    return HCCL_SUCCESS;
}

HcclResult SnapshotControl::DeviceRestore()
{
    std::lock_guard<std::mutex> lock(commMutex_);
    CHK_RET(hrtGetDevicePhyIdByIndex(static_cast<u32>(deviceLogicId_), devicePhyId_, true));

    HcclResult ret = SnapShotRestoreAction(static_cast<s32>(NICDeployment::NIC_DEPLOYMENT_DEVICE), devicePhyId_);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[SnapshotControl][DeviceRestore] call SnapShotRestoreAction fail, "
            "devicePhyId[%u]",
            devicePhyId_),
        ret);
    HCCL_INFO(
        "[SnapshotControl][DeviceRestore] device[%u] do SnapShotRestoreAction success, devId[%d].", devicePhyId_,
        deviceLogicId_);

    for (auto backupIter : backupDeviceCount_) {
        ret = SnapShotRestoreAction(static_cast<s32>(NICDeployment::NIC_DEPLOYMENT_DEVICE), backupIter.first);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[SnapshotControl][DeviceRestore] call SnapShotRestoreAction fail, "
                "backup devicePhyId[%u]",
                backupIter.first),
            ret);
        HCCL_INFO(
            "[SnapshotControl][DeviceRestore] backup device[%u] do SnapShotRestoreAction success, devId[%d].",
            backupIter.first, deviceLogicId_);
    }
    return HCCL_SUCCESS;
}

HcclResult SnapshotControl::Recovery()
{
    HCCL_ERROR("-------------------- THE ABOVE AND THIS ERROR LOG CAN BE IGNORED. --------------------");

    if (commCallbacks_.size() != 0) {
        CHK_RET(DeviceRestore());
    }

    // set device status to stopped, need to skip device operations
    CHK_RET(MarkInvalidComms());
    CHK_RET(SetStatus(SnapshotStatus::RESTORE_SNAPSHOT));
    CHK_RET(Transport::SetDeviceUnavailable(deviceLogicId_));
    CHK_RET(ResetInitState());
    HCCL_INFO(
        "[SnapshotControl][PostProcess] snapshot recovery success, devId[%d], devPhyId[%u].", deviceLogicId_,
        devicePhyId_);
    return HCCL_SUCCESS;
}
} // namespace hccl
