/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdio>
#include <cstdlib>
#include <string>

#include "cmd_base.h"
#include "cmd_base_utils.h"
#include "sim_common_api.h"
#include "sim_log.h"
#include "db_sim_op_db_ops.h"

void envInit()
{
    setenv("HCCL_VM_INSTALL_ROOT", GetBinLocation().c_str(), 1);
    std::string dbPrefix = InstallPath::ResolveToInstallRoot("data/hccl_vm_data.db");
    int ret3 = system(("sudo rm -fr " + dbPrefix + " " + dbPrefix + "-wal " + dbPrefix + "-shm 2>/dev/null").c_str());
    int ret1 = system("sudo rm -fr /dev/shm/* 2>/dev/null");
    std::string dataDir = InstallPath::ResolveToInstallRoot("data") + "/";
    int ret2 = system(("sudo rm -fr " + dataDir + "* 2>/dev/null").c_str());
    std::string logsDir = InstallPath::ResolveToInstallRoot("logs") + "/";
    int ret4 = system(("sudo rm -fr " + logsDir + "* 2>/dev/null").c_str());

    if (ret1 != 0 || ret2 != 0 || ret4 != 0) {
        printf("envInit failed\n");
    }
    printf("envInit success\n");
}

int main(int argc, char* argv[])
{
    const char* envCheck = std::getenv(HVM_BASH_ENV_KEY.c_str());

    if (envCheck != nullptr) {
        StartHostClient(argc, argv);
    } else {
        envInit();
        LogConfig config = LoadLogConfig("hccl_vm");
        InitLogger(config);
        sim::InitOpDataDb();
        std::string cmd = ArgvToString(argc, argv);
        if (argc == 1) {
            cmd += " --help";
        }
        ParseCommand(cmd);
    }
    return 0;
}
