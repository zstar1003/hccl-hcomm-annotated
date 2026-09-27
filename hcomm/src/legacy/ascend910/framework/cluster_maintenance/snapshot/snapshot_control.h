/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_SNAPSHOT_CONTROL_H
#define HCCL_SNAPSHOT_CONTROL_H

#include <mutex>
#include <functional>
#include "hccl_common.h"
#include "referenced.h"

namespace hccl {
using SnapshotSetInvalidComm = std::function<HcclResult(bool)>;
using SnapshotCheckPreProcess = std::function<HcclResult()>;
using SnapshotCheckPostProcess = std::function<HcclResult()>;

enum class SnapshotStatus {
    DEFAULT = 0,
    PRE_SNAPSHOT = 1,
    POST_SNAPSHOT = 2,
    RESTORE_SNAPSHOT = 3,
};

struct SnapshotCallbacks {
    SnapshotSetInvalidComm setInvalidCommCallback;
    SnapshotCheckPreProcess preProcessCallback;
    SnapshotCheckPostProcess postProcessCallback;
};

class SnapshotControl {
public:
    static SnapshotControl& GetInstance(s32 deviceLogicId);
    SnapshotStatus GetStatus();
    HcclResult RegisterComm(
        const std::string& identifier, SnapshotSetInvalidComm setInvalidCommCallback,
        SnapshotCheckPreProcess preProcessCallback, SnapshotCheckPostProcess postProcessCallback);
    HcclResult RegisterBackup(const std::string& identifier, u32 backupDevicePhyId);
    HcclResult UnRegisterComm(const std::string& identifier);
    HcclResult UnRegisterBackup(const std::string& identifier, u32 backupDevicePhyId);
    HcclResult PreProcess();
    HcclResult PostProcess();
    HcclResult Recovery();

private:
    SnapshotControl();
    ~SnapshotControl();
    HcclResult SetStatus(SnapshotStatus status);
    HcclResult CheckCommsPreProcess();
    HcclResult CheckCommsPostProcess();
    HcclResult MarkInvalidComms();

    HcclResult DevicePreProcess();
    HcclResult DevicePostProcess();
    HcclResult DeviceRestore();

    static bool registered;
    std::mutex statusMutex_;
    SnapshotStatus status_{SnapshotStatus::DEFAULT};
    std::mutex commMutex_;
    std::map<const std::string, SnapshotCallbacks> commCallbacks_;
    s32 deviceLogicId_{INVALID_INT};
    u32 devicePhyId_{INVALID_UINT};
    std::map<u32, Referenced> backupDeviceCount_; // key是backupDevPhyId
};
} // namespace hccl
#endif // HCCL_SNAPSHOT_CONTROL_H
