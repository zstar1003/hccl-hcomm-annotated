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
#include "db_sim_runner_db.h"
#include "sim_common_api.h"
#include "sim_common_defs.h"
#include "sim_log.h"
#include "sim_models.h"
#include "subcmd_run.h"

namespace HcclSim {
void RunCommand::Setup(CLI::App& app)
{
    auto sub_oneShot
        = app.add_subcommand("run", "算例one_shot运行模式, 请勿在子bash中重复启用,命令后必须接有算例执行指令");

    sub_oneShot->add_option("configFile", configClusterName, "加载昇腾集群组网配置目录")
        ->required()
        ->check(GenerateClusterTopo);
    sub_oneShot->add_option("--level", g_hcclVmLevel, "设置模拟等级, 当前支持等级为 1 和 2, 默认模拟等级 2 ");
    sub_oneShot->add_flag(
        "--check-only", checkOnlyMode,
        "仅校验模式:大块(200MB-4GB)内存申请复用同一块 4GB 共享区,内容不保证正确,换取内存节省");
    sub_oneShot->allow_extras(true);

    sub_oneShot->callback([this, &app]() {
        Execute(app);
    });
}

void RunCommand::Execute(CLI::App& app)
{
    if (g_hcclVmBashFlag) {
        HCCL_VM_WARN("hccl-vm is already running. Please do not run examples in one_shot mode within a sub-bash. Exit "
                     "the sub-bash and try again.");
        return;
    }
    CLI::App* tmp_cmd = app.get_subcommand("run");
    std::vector<std::string> leftargvs = tmp_cmd->remaining();
    if (leftargvs.empty()) {
        HCCL_VM_ERROR("In one_shot mode, an example execution command must be provided.");
        return;
    }
    HCCL_VM_INFO("Initializing: Model={}, Level={}", configClusterName, g_hcclVmLevel);
    auto clusterDir = InstallPath::ResolveToInstallRoot("config/network/cluster/" + configClusterName);

    sim::RunModeConfig runMode{};
    runMode.mode = checkOnlyMode ? 1 : 0;
    RunnerDB::DeleteAll<sim::RunModeConfig>();
    RunnerDB::Add<sim::RunModeConfig>(runMode);
    HCCL_VM_INFO("run mode: {}", checkOnlyMode ? "check-only" : "normal");

    auto ret = InitHvmEnv(clusterDir, g_hcclVmLevel, checkOnlyMode);
    if (ret != HcclVmResult::HCCL_SIM_HOST_SUCCESS_CMD) {
        HCCL_VM_ERROR("Failed to initialize simulation environment. Cleaning up environment.");
        auto cleanRet = HcclVmExit();
        if (cleanRet != HcclVmResult::HCCL_SIM_HOST_SUCCESS_CMD) {
            HCCL_VM_ERROR("Failed to clean up environment. Please check for residual environment artifacts.");
        }
        return;
    }
    CstyleCmd syscmd(leftargvs);
    HCCL_VM_INFO("one_shot mode, executing: {}", syscmd.cmd());
    std::string libDir = "lib/" + GetArchStr() + "/";
    std::string proxyPathL0 = InstallPath::ResolveToInstallRoot(libDir + "libhccl_proxy_level0.so");
    std::string proxyPathL2 = InstallPath::ResolveToInstallRoot(libDir + "libhccl_proxy_level2.so");
    std::string preload = proxyPathL0 + ":" + proxyPathL2;
    setenv("LD_PRELOAD", preload.c_str(), 1);
    int sysRet = std::system(syscmd.cmd().c_str()); // system() 会阻塞当前进程直到子命令结束
    if (sysRet != 0) {
        HCCL_VM_ERROR("Example execution failed: {}", sysRet);
    }
    RemoveFromLDPreload(proxyPathL0);
    RemoveFromLDPreload(proxyPathL2);
    auto cleanRet = HcclVmExit();
    return;
}

static inline CommandAutoRegister<RunCommand> g_run_cmd_reg{};
} // namespace HcclSim
