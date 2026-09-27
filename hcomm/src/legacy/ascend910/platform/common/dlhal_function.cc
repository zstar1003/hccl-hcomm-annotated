/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dlhal_function.h"

#include <string>
#include <map>
#include "log.h"
#include "hccl_dl.h"

namespace hccl {
DlHalFunction& DlHalFunction::GetInstance()
{
    static DlHalFunction hcclDlHalFunction;
    return hcclDlHalFunction;
}

DlHalFunction::DlHalFunction() : handle_(nullptr) {}

DlHalFunction::~DlHalFunction()
{
    if (handle_ != nullptr) {
        (void)HcclDlclose(handle_);
        handle_ = nullptr;
    }
}

HcclResult DlHalFunction::DlHalFunctionEschedInit()
{
    dlHalEschedSubmitEvent
        = (drvError_t(*)(unsigned int, struct event_summary*))HcclDlsym(handle_, "halEschedSubmitEvent");
    CHK_SMART_PTR_NULL(dlHalEschedSubmitEvent);

    dlHalEschedAttachDevice = (drvError_t(*)(unsigned int))HcclDlsym(handle_, "halEschedAttachDevice");
    CHK_SMART_PTR_NULL(dlHalEschedAttachDevice);

    dlHalGetAPIVersion = (drvError_t(*)(int*))HcclDlsym(handle_, "halGetAPIVersion");
    CHK_SMART_PTR_NULL(dlHalGetAPIVersion);

    dlHalEschedDettachDevice = (drvError_t(*)(unsigned int))HcclDlsym(handle_, "halEschedDettachDevice");
    CHK_SMART_PTR_NULL(dlHalEschedDettachDevice);

    dlHalEschedCreateGrp
        = (drvError_t(*)(unsigned int, unsigned int, GROUP_TYPE))HcclDlsym(handle_, "halEschedCreateGrp");
    CHK_SMART_PTR_NULL(dlHalEschedCreateGrp);

    dlHalEschedRegisterAckFunc
        = (drvError_t(*)(unsigned int, unsigned int, void (*ackFunc)(unsigned int, unsigned int, u8*, unsigned int)))
            HcclDlsym(handle_, "halEschedRegisterAckFunc");
    CHK_SMART_PTR_NULL(dlHalEschedRegisterAckFunc);

    dlHalDrvQueryProcessHostPid = (drvError_t(*)(
        int, unsigned int*, unsigned int*, unsigned int*, unsigned int*))HcclDlsym(handle_, "drvQueryProcessHostPid");
    CHK_SMART_PTR_NULL(dlHalDrvQueryProcessHostPid);

    dlHalDrvGetDevIDByLocalDevID
        = (drvError_t(*)(unsigned int, unsigned int*))HcclDlsym(handle_, "drvGetDevIDByLocalDevID");
    CHK_SMART_PTR_NULL(dlHalDrvGetDevIDByLocalDevID);

    dlDrvMemCpy = (DVresult(*)(
        unsigned long long dst, unsigned long long destMax, unsigned long long src,
        unsigned long long count))HcclDlsym(handle_, "drvMemcpy");
    CHK_SMART_PTR_NULL(dlDrvMemCpy);

    dlHalDrvGetDevNum = (drvError_t(*)(uint32_t*))HcclDlsym(handle_, "drvGetDevNum");
    CHK_SMART_PTR_NULL(dlHalDrvGetDevNum);

    dlDrvDeviceGetBareTgid = (pid_t(*)(void))HcclDlsym(handle_, "drvDeviceGetBareTgid");
    CHK_SMART_PTR_NULL(dlDrvDeviceGetBareTgid);

    dlHalGetDeviceInfo = (drvError_t(*)(uint32_t, int32_t, int32_t, int64_t*))HcclDlsym(handle_, "halGetDeviceInfo");
    CHK_SMART_PTR_NULL(dlHalGetDeviceInfo);

    dlHalGrpQuery = (drvError_t(*)(GroupQueryCmdType, void*, unsigned int, void*, unsigned int*))HcclDlsym(
        handle_, "halGrpQuery");
    CHK_SMART_PTR_NULL(dlHalGrpQuery);

    dlHalDrvGetPlatformInfo = (drvError_t(*)(uint32_t*))HcclDlsym(handle_, "drvGetPlatformInfo");
    CHK_SMART_PTR_NULL(dlHalDrvGetPlatformInfo);

    dlHalGetChipInfo = (drvError_t(*)(uint32_t, halChipInfo*))HcclDlsym(handle_, "halGetChipInfo");
    if (dlHalGetChipInfo == nullptr) {
        HCCL_WARNING("dlHalGetChipInfo is nullptr, can not use halGetChipInfo");
    }

    dlHalBindCgroup = (drvError_t(*)(BIND_CGROUP_TYPE))HcclDlsym(handle_, "halBindCgroup");
    if (dlHalBindCgroup == nullptr) {
        HCCL_WARNING("dlHalBindCgroup is nullptr, can not use dlHalBindCgroup");
    }

    dlHalHostRegister
        = (drvError_t(*)(void*, uint64_t, uint32_t, uint32_t, void**))HcclDlsym(handle_, "halHostRegister");
    if (dlHalHostRegister == nullptr) {
        HCCL_WARNING("dlHalHostRegister is nullptr, can not use dlHalHostRegister");
    }

    dlHalHostUnregister = (drvError_t(*)(void*, uint32_t))HcclDlsym(handle_, "halHostUnregister");
    if (dlHalHostUnregister == nullptr) {
        HCCL_WARNING("dlHalHostUnregister is nullptr, can not use dlHalHostUnregister");
    }

    dlHalHostUnregisterEx = (drvError_t(*)(void*, uint32_t, uint32_t))HcclDlsym(handle_, "halHostUnregisterEx");
    if (dlHalHostUnregisterEx == nullptr) {
        HCCL_WARNING("dlHalHostUnregisterEx is nullptr, can not use dlHalHostUnregisterEx");
    }

    dlHalMemCtl = (drvError_t(*)(int, void*, size_t, void*, size_t*))HcclDlsym(handle_, "halMemCtl");
    CHK_SMART_PTR_NULL(dlHalMemCtl);
#ifdef CCL_KERNEL
    dlHalSensorNodeRegister
        = (drvError_t(*)(uint32_t, struct halSensorNodeCfg*, uint64_t*))HcclDlsym(handle_, "halSensorNodeRegister");
    CHK_SMART_PTR_NULL(dlHalSensorNodeRegister);

    dlHalSensorNodeUnregister = (drvError_t(*)(uint32_t, uint64_t))HcclDlsym(handle_, "halSensorNodeUnregister");
    CHK_SMART_PTR_NULL(dlHalSensorNodeUnregister);

    dlHalSensorNodeUpdateState
        = (drvError_t(*)(uint32_t, uint64_t, int, halGeneralEventType_t))HcclDlsym(handle_, "halSensorNodeUpdateState");
    CHK_SMART_PTR_NULL(dlHalSensorNodeUpdateState);
#endif
    return HCCL_SUCCESS;
}

HcclResult DlHalFunction::DlHalFunctionInit()
{
    std::lock_guard<std::mutex> lock(handleMutex_);
    if (handle_ == nullptr) {
        handle_ = HcclDlopen("libascend_hal.so", RTLD_NOW);
        const char* errMsg = dlerror();
        CHK_PRT_RET(
            handle_ == nullptr,
            HCCL_ERROR(
                "dlopen [%s] failed, %s", "libascend_hal.so",
                (errMsg == nullptr) ? "please check the file exist or permission denied." : errMsg),
            HCCL_E_OPEN_FILE_FAILURE);
    }

    CHK_RET(DlHalFunctionEschedInit());
    return HCCL_SUCCESS;
}
} // namespace hccl
