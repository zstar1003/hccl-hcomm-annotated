/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdlib>
#include <string>

#include "topo_ascend_cluster_parser.h"
#include "cmd_base_utils.h"
#include "sim_common_defs.h"
#include "sim_log.h"
#include "subcmd_mock_comm.h"

namespace HcclSim {
void MockCommCommand::Setup(CLI::App& app)
{
    auto sub_run = app.add_subcommand("mock-comm", "mock-comm: 启动一个算子的通信域");

    sub_run->add_option("configFile", configFileName, "加载昇腾集群组网配置目录")->required()->check(FileInModelDir);
    sub_run->add_option("--level", g_hcclVmLevel, "设置模拟等级, 当前支持等级为 1 和 2, 默认模拟等级 2 ");

    sub_run->callback([this]() {
        Execute();
    });
}

void MockCommCommand::Execute()
{
    HCCL_VM_INFO("Initializing mock communication domain: Config={}, Level={}", configFileName, g_hcclVmLevel);

    // 首先清理动态数据库表项
    auto ret = ClearDbTables();
    if (ret != HcclVmResult::HCCL_SIM_HOST_SUCCESS_CMD) {
        HCCL_VM_ERROR("Failed to clear database tables.");
        return;
    }

    // 拷贝topo.json文件到执行目录
    ret = CopyFile(g_configClusterDir);
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("copy topo.json file failed");
        return;
    }

    // 重置通信域
    ret = HcclVmResetCommDomain();
    if (ret != HcclVmResult::HCCL_SIM_HOST_SUCCESS_CMD) {
        HCCL_VM_ERROR("Failed to reset mock communication domain.");
        return;
    }

    if (configFileName != "ranktable") {
        if (!ParseYamlTopo(configFileName, topoMeta)) {
            return;
        }
    }

    ret = InitHvmCommEnv(topoMeta, configFileName, g_hcclVmLevel);
    if (ret != HcclVmResult::HCCL_SIM_HOST_SUCCESS_CMD) {
        HCCL_VM_ERROR("Failed to initialize mock communication environment. Cleaning up environment.");
        return;
    }
    return;
}

static inline CommandAutoRegister<MockCommCommand> g_mock_comm_cmd_reg{};
} // namespace HcclSim
