/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_OP_TASK_CONFIG_H
#define HCCLV2_OP_TASK_CONFIG_H

#include <hccl/base.h>

namespace Hccl {

constexpr u32 INVALID_QOSCFG = 0xFFFFFFFF;
constexpr s32 NOTIFY_DEFAULT_WAIT_TIME = 27 * 68; // notifywait默认1800等待时长

class OpTaskConfig {
public:
    u32 GetQosCfg() const { return qosCfg; }
    void SetQosCfg(u32 qos) { OpTaskConfig::qosCfg = qos; }
    u32 GetNotifyWaitTime() const { return notifyWaitTime; }
    void SetNotifyWaitTime(u32 seconds) { OpTaskConfig::notifyWaitTime = seconds; }

private:
    u32 qosCfg{INVALID_QOSCFG};
    u32 notifyWaitTime{NOTIFY_DEFAULT_WAIT_TIME};
};
} // namespace Hccl

#endif
