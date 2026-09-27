/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_NOTIFY_TIMEOUT_CFG_H
#define HCCLV2_NOTIFY_TIMEOUT_CFG_H

#include "types.h"
#include "env_config/env_config_v2.h"

namespace Hccl {
class NotifyTimeoutCfg {
public:
    static constexpr Timeout NOTIFY_DEFAULT_WAIT_TIME = 27 * 68;
    static constexpr Timeout NOTIFY_MAX_WAIT_TIME = 255 * 68;

    void Init()
    {
        notifyTimeout = EnvConfig::GetInstance().GetRtsConfig().GetExecTimeOut();
        HCCL_INFO("[NotifyTimeoutCfg][Init] set notifyTimeout[%u]s", notifyTimeout);
    }

    Timeout GetBarrierTimeout() const { return barrierTimeout; }

    Timeout GetNotifyTimeout() const { return notifyTimeout; }

private:
    Timeout barrierTimeout{NOTIFY_MAX_WAIT_TIME};
    Timeout notifyTimeout{NOTIFY_DEFAULT_WAIT_TIME};
};

} // namespace Hccl

#endif // HCCLV2_NOTIFY_TIMEOUT_CFG_H
