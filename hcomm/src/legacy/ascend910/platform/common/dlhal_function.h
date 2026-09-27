/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_SRC_DLHALFUNCTION_H
#define HCCL_SRC_DLHALFUNCTION_H

#include <functional>
#include <mutex>
#include <hccl/hccl_types.h>

#include "hccl/base.h"
#include "ascend_hal.h"
#include "ascend_hal_define.h"

namespace hccl {
class DlHalFunction {
public:
    virtual ~DlHalFunction();
    static DlHalFunction& GetInstance();
    HcclResult DlHalFunctionInit();
    std::function<drvError_t(unsigned int, struct event_summary*)> dlHalEschedSubmitEvent;
    std::function<drvError_t(unsigned int)> dlHalEschedAttachDevice;
    std::function<drvError_t(int*)> dlHalGetAPIVersion;
    std::function<drvError_t(unsigned int, unsigned int, GROUP_TYPE)> dlHalEschedCreateGrp;
    std::function<drvError_t(
        unsigned int, unsigned int, void (*ackFunc)(unsigned int, unsigned int, u8*, unsigned int))>
        dlHalEschedRegisterAckFunc;
    std::function<drvError_t(int, unsigned int*, unsigned int*, unsigned int*, unsigned int*)>
        dlHalDrvQueryProcessHostPid;
    std::function<drvError_t(unsigned int, unsigned int*)> dlHalDrvGetDevIDByLocalDevID;
    std::function<DVresult(
        unsigned long long dst, unsigned long long destMax, unsigned long long src, unsigned long long count)>
        dlDrvMemCpy;
    std::function<drvError_t(uint32_t*)> dlHalDrvGetDevNum;
    std::function<drvError_t(uint32_t, int32_t, int32_t, int64_t*)> dlHalGetDeviceInfo;
    std::function<pid_t(void)> dlDrvDeviceGetBareTgid;
    std::function<drvError_t(GroupQueryCmdType, void*, unsigned int, void*, unsigned int*)> dlHalGrpQuery;
    std::function<drvError_t(unsigned int devId)> dlHalEschedDettachDevice;
    std::function<drvError_t(uint32_t*)> dlHalDrvGetPlatformInfo;
    std::function<drvError_t(uint32_t, halChipInfo*)> dlHalGetChipInfo;
    std::function<drvError_t(BIND_CGROUP_TYPE)> dlHalBindCgroup;
    std::function<drvError_t(void*, uint64_t, uint32_t, uint32_t, void**)> dlHalHostRegister;
    std::function<drvError_t(void*, uint32_t)> dlHalHostUnregister;
    std::function<drvError_t(void*, uint32_t, uint32_t)> dlHalHostUnregisterEx;
    std::function<drvError_t(int, void*, size_t, void*, size_t*)> dlHalMemCtl;
    std::function<drvError_t(uint32_t, struct halSensorNodeCfg*, uint64_t*)> dlHalSensorNodeRegister;
    std::function<drvError_t(uint32_t, uint64_t)> dlHalSensorNodeUnregister;
    std::function<drvError_t(uint32_t, uint64_t, int, halGeneralEventType_t)> dlHalSensorNodeUpdateState;
    bool DlHalFunctionIsInit()
    {
        std::lock_guard<std::mutex> lock(handleMutex_);
        return (handle_ != nullptr);
    }

protected:
private:
    void* handle_;
    std::mutex handleMutex_;
    DlHalFunction(const DlHalFunction&);
    DlHalFunction& operator=(const DlHalFunction&);
    DlHalFunction();
    HcclResult DlHalFunctionEschedInit();
};
} // namespace hccl

#endif // HCCL_SRC_DLHALFUNCTION_H
