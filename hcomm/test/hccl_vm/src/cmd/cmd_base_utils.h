/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_VM_CMD_UTILS_H
#define HCCL_VM_CMD_UTILS_H

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <exception>
#include <fstream>
#include <libgen.h>
#include <sstream>
#include <cstdlib>
#include <stdlib.h>
#include <string>
#include <unistd.h>
#include <vector>

#include "cmd_cluster_model_utils.h"
#include "sim_common_defs.h"
#include "sim_plugin_manager.h"

extern const std::string HVM_BASH_ENV_KEY;
extern bool g_hcclVmBashFlag;
extern std::string g_configClusterDir;
extern std::uint32_t g_hcclVmLevel;

class CstyleCmd {
public:
    explicit CstyleCmd(const std::vector<std::string>& args)
    {
        storage_ = args;
        rebuild();
    }
    int argc() const { return static_cast<int>(pointers_.size()); }
    char** argv() { return pointers_.data(); }
    std::string cmd() { return cmd_; }

private:
    void rebuild()
    {
        std::stringstream ss;
        pointers_.clear();
        pointers_.reserve(storage_.size() + 1);
        for (auto& s : storage_) {
            pointers_.push_back(const_cast<char*>(s.c_str()));
            ss << s << " ";
        }
        cmd_ = ss.str();
    }
    std::vector<std::string> storage_;
    std::vector<char*> pointers_;
    std::string cmd_;
};

// 通用工具方法
std::string GetBinLocation();
std::string ArgvToString(int argc, char* argv[]);
std::string FileInModelDir(const std::string& fileName);
std::string CheckClusterConfigFile(const std::string& topoFileName);
std::string GenerateClusterTopo(const std::string& topoFileName);
void RemoveFromLDPreload(const std::string& targetValue);
void ShowModel();

// init hvm env
HcclVmResult InitHvmEnv(const std::string& configClusterDir, uint32_t level, bool checkOnlyMode);
HcclVmResult InitHvmCommEnv(const TopoMeta& topoMeta, const std::string& configFileName, uint32_t level);
HcclVmResult HcclVmExit();
HcclVmResult CopyFile(const std::string& clusterDir);

// plugin manager
HcclVmResult InstallUserPlugin(std::string argStr);
HcclVmResult RunUserPlugin(std::string argStr);
HcclVmResult UninstallUserPlugin(std::string argStr);
void ShowUserPlugin();

// config
HcclVmResult SetConsoleLogLevel(int level);
HcclVmResult SetFileLogLevel(int level);
HcclVmResult ShowCurrentLogLevel();

// hccl-vm 流程类方法
HcclVmResult StartHvmCmd();

void StartHostClient(int argc, char* argv[]);
void ServerListen();
void RunnerListen();
void ParseCommand(std::string& cmd);

HcclVmResult ClearDbTables();
HcclVmResult HcclVmResetCommDomain();

#endif
