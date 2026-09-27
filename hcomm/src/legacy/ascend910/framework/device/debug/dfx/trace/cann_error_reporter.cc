/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cann_error_reporter.h"
#include <cstdio>
#include "log.h"

namespace dfx {
constexpr uint64_t DEFAULT_SENSOR_HANDLE = 0xFFFFFFFF;

CannErrorReporter::CannErrorReporter() : sensorHandle_(DEFAULT_SENSOR_HANDLE), devId_(0) {}

CannErrorReporter::~CannErrorReporter()
{
    status2Event_.clear();
    void Clear();
}

CannErrorReporter& CannErrorReporter::GetInstance()
{
    static CannErrorReporter instance;
    return instance;
}

HcclResult CannErrorReporter::Init(uint32_t deviceId)
{
    if (sensorHandle_ != DEFAULT_SENSOR_HANDLE && devId_ == deviceId) {
        return HCCL_SUCCESS;
    }

    if (sensorHandle_ != DEFAULT_SENSOR_HANDLE) {
        void Clear();
    }

    CHK_PRT_RET(
        RegisterSensorNode(deviceId, &sensorHandle_) != HCCL_SUCCESS,
        HCCL_RUN_WARNING("[CannErrorReporter][Init] init sensor node fail, deviceId[%u].", deviceId), HCCL_E_INTERNAL);
    devId_ = deviceId;
    return HCCL_SUCCESS;
}

HcclResult CannErrorReporter::Clear()
{
    if (sensorHandle_ != DEFAULT_SENSOR_HANDLE) {
        HcclResult ret = UnRegisterSensorNode(devId_, sensorHandle_);
        if (ret != HCCL_SUCCESS) {
            HCCL_RUN_WARNING(
                "[CannErrorReporter][Clear] unregister sensor node fail, handle[%llu], deviceId[%u].", sensorHandle_,
                devId_);
        } else {
            HCCL_INFO(
                "[CannErrorReporter][Clear] unregister sensor node success, handle[%llu], devId[%u].", sensorHandle_,
                devId_);
        }
    }
    sensorHandle_ = DEFAULT_SENSOR_HANDLE;
    devId_ = 0;
    return HCCL_SUCCESS;
}

HcclResult CannErrorReporter::UpdateSensorNode(uint32_t deviceId, ReportStatus reportStatus)
{
    if (sensorHandle_ == DEFAULT_SENSOR_HANDLE || devId_ != deviceId) {
        CHK_PRT_RET(
            Init(deviceId) != HCCL_SUCCESS,
            HCCL_RUN_WARNING("[CannErrorReporter][UpdateSensorNode] fail to get a valid node sensor."),
            HCCL_E_INTERNAL);
    }
    uint64_t handle = sensorHandle_;

    auto iter = status2Event_.find(reportStatus);
    if (iter == status2Event_.end()) {
        HCCL_RUN_WARNING(
            "[CannErrorReporter][UpdateSensorNode] "
            "unknown reportStatus[%d], fail to find corresponding event type, update node sensor fail.",
            static_cast<int>(reportStatus));
        return HCCL_E_NOT_SUPPORT;
    }
    int32_t eventType = static_cast<int32_t>(iter->second.first);
    HcclGeneralEventType eventAssert = iter->second.second;

    HCCL_INFO(
        "[CannErrorReporter][UpdateSensorNode]updating sensor, "
        "deviceId[%u], reportStatus[%d], event val[%d], event assert[%d].",
        deviceId, static_cast<int>(reportStatus), eventType, static_cast<int>(eventAssert));

    CHK_PRT_RET(
        hrtHalSensorNodeUpdateState(deviceId, handle, eventType, eventAssert) != HCCL_SUCCESS,
        HCCL_RUN_WARNING(
            "[CannErrorReporter][UpdateSensorNode] "
            "update sensor node fail, deviceId[%u], reportStatus[%d], val[%d], handle[%llu].",
            deviceId, static_cast<int>(reportStatus), eventType, handle),
        HCCL_E_INTERNAL);

    HCCL_RUN_INFO(
        "[CannErrorReporter][UpdateSensorNode] send report success, "
        "deviceId[%u], reportStatus[%d], val[%d], handle[%llu].",
        deviceId, static_cast<int>(reportStatus), eventType, handle);
    return HCCL_SUCCESS;
}

HcclResult CannErrorReporter::RegisterSensorNode(uint32_t deviceId, uint64_t* handle)
{
    HcclResult ret = hrtHalSensorNodeRegister(deviceId, handle);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_RUN_WARNING("[CannErrorReporter][RegisterSensorNode] register sensor node fail, device id[%u].", deviceId),
        HCCL_E_INTERNAL);

    HCCL_INFO(
        "[CannErrorReporter][RegisterSensorNode] register node sensor success, deviceId[%u], handle[%llu].", deviceId,
        *handle);
    return HCCL_SUCCESS;
}

HcclResult CannErrorReporter::UnRegisterSensorNode(uint32_t deviceId, uint64_t handle)
{
    HcclResult ret = hrtHalSensorNodeUnregister(deviceId, handle);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_RUN_WARNING(
            "[CannErrorReporter][UnRegisterSensorNode] "
            "unregister sensor node fail, device id[%u], handle[%llu].",
            deviceId, handle),
        HCCL_E_INTERNAL);
    HCCL_INFO(
        "[CannErrorReporter][UnRegisterSensorNode] unregister sensor success, device id[%u], handle[%llu].", deviceId,
        handle);

    return HCCL_SUCCESS;
}

} // namespace dfx
