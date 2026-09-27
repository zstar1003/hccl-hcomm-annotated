/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_VM_LOG_COMMAND_H
#define HCCL_VM_LOG_COMMAND_H

#include <map>
#include <optional>
#include <string>

#include "cmd_base.h"
#include "sim_common_defs.h"

namespace HcclSim {
class LogCommand : public CommandBase {
public:
    static std::string StaticName() { return "log"; }
    void Setup(CLI::App& app) override;

private:
    void Execute();

    std::optional<int> level_;
    std::optional<int> consoleLevel_;
    std::optional<int> fileLevel_;

    const std::map<std::string, int> levelMap_{
        {"trace", 0}, {"debug", 1}, {"info", 2}, {"warn", 3}, {"warning", 3}, {"error", 4}, {"err", 4}, {"critical", 5},
    };
};
} // namespace HcclSim

#endif
