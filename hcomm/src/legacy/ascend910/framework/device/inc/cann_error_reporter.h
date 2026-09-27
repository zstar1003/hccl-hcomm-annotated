/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RAS_REPORTER_H
#define RAS_REPORTER_H

#include <map>
#include "adapter_hal_pub.h"

namespace dfx {
enum class HcclReportEvent {
    HCCL_OP_RETRY_SUCCESS = 0x09,
    HCCL_OP_USE_BACKUP_LINK = 0x0A,
    HCCL_OP_RETRY_FAIL = 0X0B,
    HCCL_REPORT_RESERVE
};

enum class ReportStatus : int64_t {
    kDefault = 0,
    kRetrySuccess,
    kRetryWithBackupLink,
    kRetryFail,
};

class CannErrorReporter {
public:
    // 单例
    static CannErrorReporter& GetInstance();

    // 解注册所有保存的 Sensor Node，并清空用于缓存 Sensor Node 句柄的 map
    HcclResult Clear();

    // 向 Sensor Node 报告故障状态更新
    HcclResult UpdateSensorNode(uint32_t deviceId, ReportStatus reportStatus);

private:
    CannErrorReporter();
    ~CannErrorReporter();

    // 初始化并注册 Sensor Node
    HcclResult Init(uint32_t deviceId);

    // 注册新的 Sensor Node，并返回该 Sensor 的句柄
    HcclResult RegisterSensorNode(uint32_t deviceId, uint64_t* handle);

    // 解注册指定的 Sensor Node
    HcclResult UnRegisterSensorNode(uint32_t deviceId, uint64_t handle);

    // Sensor Node 的句柄
    uint64_t sensorHandle_;
    uint32_t devId_;

    // 故障状态对应的上报事件类型
    std::map<ReportStatus, std::pair<HcclReportEvent, HcclGeneralEventType>> status2Event_ = {
        {ReportStatus::kRetrySuccess,
         {HcclReportEvent::HCCL_OP_RETRY_SUCCESS, HcclGeneralEventType::HCCL_GENERAL_EVENT_TYPE_ONE_TIME}},
        {ReportStatus::kRetryWithBackupLink,
         {HcclReportEvent::HCCL_OP_USE_BACKUP_LINK, HcclGeneralEventType::HCCL_GENERAL_EVENT_TYPE_ONE_TIME}},
        {ReportStatus::kRetryFail,
         {HcclReportEvent::HCCL_OP_RETRY_FAIL, HcclGeneralEventType::HCCL_GENERAL_EVENT_TYPE_ONE_TIME}},
    };
};

} // namespace dfx
#endif // RAS_REPORTER_H
