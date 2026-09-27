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

#include "cmd_table_utils.h"
#include "sim_common_defs.h"
#include "sim_log.h"
#include "db_sim_runner_db.h"
#include "subcmd_table.h"

namespace HcclSim {
void TableCommand::Setup(CLI::App& app)
{
    auto tableCmd = app.add_subcommand("table", "table opt");
    tableCmd->require_subcommand(1);
    auto showCmd = tableCmd->add_subcommand("show", "show table");

    showCmd->add_option("name", showStr, "show table all")->required();
    showCmd->callback([this]() {
        if (showStr == "all") {
            std::vector<std::string> tables;
            tables = RunnerDB::GetAllTableName();
            for (auto& tbl : tables) {
                HCCL_VM_INFO("{}", tbl);
            }
        } else {
            CmdTableShow(showStr);
        }
    });

    auto updateCmd = tableCmd->add_subcommand("update", "update table");

    updateCmd->add_option("table", tableName, "Table name (e.g., Device, Server, Host)")->required();
    updateCmd->add_option("id", rowId, "Row ID to update")->required();
    updateCmd->add_option("column", columnName, "Column to update (e.g., soc_version)")->required();
    updateCmd->add_option("value", newValue, "New value (as string)")->required();

    updateCmd->callback([&]() {
        CmdTableUpdate(tableName, rowId, columnName, newValue);
    });
}

static inline CommandAutoRegister<TableCommand> g_table_cmd_reg{};
} // namespace HcclSim
