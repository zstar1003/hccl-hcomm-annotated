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
#include "subcmd_model.h"

namespace HcclSim {
void ModelCommand::Setup(CLI::App& app)
{
    auto sub_model = app.add_subcommand("model", "管理建模文件");

    sub_model->require_subcommand(1);
    auto model_list = sub_model->add_subcommand("list", "展示建模文件");

    model_list->callback([this]() {
        Execute();
    });
}

void ModelCommand::Execute() { ShowModel(); }

static inline CommandAutoRegister<ModelCommand> g_model_cmd_reg{};
} // namespace HcclSim
