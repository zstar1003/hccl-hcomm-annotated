/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_INC_ADAPTER_HAL_PUB_H
#define HCCL_INC_ADAPTER_HAL_PUB_H

#include "hccl_common.h"
#include "driver/ascend_hal.h"

enum class HcclGeneralEventType {
    HCCL_GENERAL_EVENT_TYPE_RESUME = 0,
    HCCL_GENERAL_EVENT_TYPE_OCCUR = 1,
    HCCL_GENERAL_EVENT_TYPE_ONE_TIME = 2,
    HCCL_GENERAL_EVENT_TYPE_MAX
};

using mc2Funcs = void (*)(void*);

HcclResult hrtHalGetDeviceType(const uint32_t devId, DevType& devType);
HcclResult hrtHalGetDeviceInfo(uint32_t devId, int32_t moduleType, int32_t infoType, int64_t* value);
HcclResult HrtHalDrvQueryProcessHostPid(
    int pid, unsigned int* chipId, unsigned int* vfid, unsigned int* hostPid, unsigned int* cpType);
HcclResult HrtHalDrvGetDevIDByLocalDevID(uint32_t localDevId, uint32_t* devId);

HcclResult hrtHalSensorNodeRegister(uint32_t devId, uint64_t* handle);
HcclResult hrtHalSensorNodeUnregister(uint32_t devId, uint64_t handle);
HcclResult hrtHalSensorNodeUpdateState(uint32_t devId, uint64_t handle, int val, HcclGeneralEventType assertion);

HcclResult hrtDrvGetLocalDevIDByHostDevID(u32 hostUdevid, u32* localDevid);
HcclResult hrtDrvMemSmmuQuery(uint32_t localDevid, uint32_t* SSID);
bool IsSupportStartMC2MaintenanceThread();
HcclResult hrtHalStartMC2MaintenanceThread(mc2Funcs f1, void* p1, mc2Funcs f2, void* p2);
// custom进程notify资源同步，在调用halResourceIdCheck前调用
HcclResult hrtHalResourceIdRestore(u32 devId, u32 tsId, drvIdType_t resType, u32 resId, u32 flag);
HcclResult GetRunSideIsDevice(bool& isDeviceSide);
#endif // HCCL_INC_ADAPTER_HAL_PUB_H
