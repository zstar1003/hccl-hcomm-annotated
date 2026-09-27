/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_VM_RUN_COMMAND_H
#define HCCL_VM_RUN_COMMAND_H

#include <string>

#include "cmd_base.h"
#include "sim_common_defs.h"

namespace HcclSim {
class RunCommand : public CommandBase {
public:
    static std::string StaticName() { return "run"; }
    void Setup(CLI::App& app) override;

private:
    void Execute(CLI::App& app);

    std::string configClusterName;
    std::vector<std::string> leftargvs;
    bool checkOnlyMode = false;
};
} // namespace HcclSim

#endif
