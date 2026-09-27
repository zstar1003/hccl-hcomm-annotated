/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "env_config_v2.h"

namespace Hccl {

EnvConfig::EnvConfig() { Parse(); }

EnvConfig& EnvConfig::GetInstance()
{
    static EnvConfig envConfig;
    return envConfig;
}

void EnvConfig::Parse()
{
    hostNicCfg.Parse();
    socketCfg.Parse();
    rtsCfg.Parse();
    rdmaCfg.Parse();
    algoCfg.Parse();
    logCfg.Parse();
    detourCfg.Parse();
}

const EnvHostNicConfig& EnvConfig::GetHostNicConfig() { return hostNicCfg; }

const EnvSocketConfig& EnvConfig::GetSocketConfig() { return socketCfg; }

const EnvRtsConfig& EnvConfig::GetRtsConfig() { return rtsCfg; }

const EnvRdmaConfig& EnvConfig::GetRdmaConfig() { return rdmaCfg; }

const EnvAlgoConfig& EnvConfig::GetAlgoConfig() { return algoCfg; }

const EnvLogConfig& EnvConfig::GetLogConfig() { return logCfg; }

const EnvDetourConfig& EnvConfig::GetDetourConfig() { return detourCfg; }

} // namespace Hccl
