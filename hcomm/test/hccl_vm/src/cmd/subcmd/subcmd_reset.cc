/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <string>

#include "cmd_base_utils.h"
#include "sim_common_defs.h"
#include "sim_log.h"
#include "subcmd_reset.h"

namespace HcclSim {
void ResetCommand::Setup(CLI::App& app)
{
    auto sub_reset = app.add_subcommand("reset", "reset: 清除数据库表数据，重置仿真环境");

    sub_reset->callback([this]() {
        Execute();
    });
}

void ResetCommand::Execute()
{
    HCCL_VM_INFO("Resetting: Clearing database tables...");
    auto ret = ClearDbTables();
    if (ret != HcclVmResult::HCCL_SIM_HOST_SUCCESS_CMD) {
        HCCL_VM_ERROR("Failed to clear database tables.");
        return;
    }
    HCCL_VM_INFO("Database tables cleared successfully.");
}

static inline CommandAutoRegister<ResetCommand> g_reset_cmd_reg{};
} // namespace HcclSim
