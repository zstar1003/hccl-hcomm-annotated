/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "env_config_stub.h"

namespace Hccl {

EnvConfigStub::EnvConfigStub() { Parse(); }

void EnvConfigStub::Parse()
{
    hostNicCfg.Parse();
    socketCfg.Parse();
    rtsCfg.Parse();
    rdmaCfg.Parse();
    algoCfg.Parse();
    logCfg.Parse();
    detourCfg.Parse();
}

const EnvHostNicConfig& EnvConfigStub::GetHostNicConfig() { return hostNicCfg; }

const EnvSocketConfig& EnvConfigStub::GetSocketConfig() { return socketCfg; }

const EnvRtsConfig& EnvConfigStub::GetRtsConfig() { return rtsCfg; }

const EnvRdmaConfig& EnvConfigStub::GetRdmaConfig() { return rdmaCfg; }

const EnvAlgoConfig& EnvConfigStub::GetAlgoConfig() { return algoCfg; }

const EnvLogConfig& EnvConfigStub::GetLogConfig() { return logCfg; }
using namespace std;
const EnvDetourConfig& EnvConfigStub::GetDetourConfig() { return detourCfg; }

} // namespace Hccl
