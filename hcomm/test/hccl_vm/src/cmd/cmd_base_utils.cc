/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cmd_base_utils.h"
#include <algorithm>
#include <cerrno>
#include <atomic>
#include <chrono>
#include <CLI11.hpp>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <mqueue.h>
#include <nlohmann_json/json.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>

#include "topo_ascend_cluster_parser.h"
#include "cmd_base.h"
#include "store_dump_shm_data.h"
#include "store_sim_comm_pool_policy.h"
#include "store_sim_run_mode.h"
#include "sim_data_dump.h"
#include "sim_log.h"
#include "runnerdb/modeldb/db_hccl_op_db_ops.h"
#include "store_sim_memory_manager.h"
#include "store_sim_device_memory_manager.h"
#include "sim_process_syncer.h"
#include "db_sim_runner_common.h"
#include "sim_common_api.h"
#include "db_sim_runner_db.h"
#include "yaml-cpp/yaml.h"

using namespace HcclSim;
namespace fs = std::filesystem;

static std::atomic<bool> g_serverListenFlag{false};
static std::atomic<bool> g_runnerListenFlag{false};
static std::thread* g_serverThread = nullptr;
static std::thread* g_runnerThread = nullptr;

static std::string MakeDataBackupTimestamp();
static void BackupAivTaskFiles(const fs::path& backupDir);

const std::string HVM_BASH_ENV_KEY = "_HVM_BASH_ENV_PATH";
const std::string g_binDir = GetBinLocation();
static void StopListenThreads();
static void ArchiveLogsAndData();

// 定义队列名称和大小配置
const char* MQ_REQ_NAME = "host_mq_request";
const char* MQ_RESP_NAME = "host_mq_response";
const int MAX_MSG_SIZE = 4096; // 单条命令最大长度
const int MAX_MSG_COUNT = 5;   // 队列深度

static const int HOST_CLIENT_RESP_TIMEOUT_MS = 5000;

bool g_hcclVmBashFlag{false};
std::uint32_t g_hcclVmLevel{2};
std::string g_configClusterDir{""};

static std::string ToPosixMqName(const char* name)
{
    if (name == nullptr || name[0] == '\0') {
        throw std::invalid_argument("message queue name is empty");
    }
    if (name[0] == '/') {
        return name;
    }
    return "/" + std::string(name);
}

static std::string MakeMqErrorMessage(const std::string& operation, const std::string& name, int err)
{
    return operation + " message queue " + name + " failed: " + std::strerror(err);
}

static void RemoveMessageQueue(const char* name)
{
    const std::string mqName = ToPosixMqName(name);
    if (mq_unlink(mqName.c_str()) == -1 && errno != ENOENT) {
        HCCL_VM_WARN("{}", MakeMqErrorMessage("remove", mqName, errno));
    }
}

static mq_attr MakeMessageQueueAttr()
{
    mq_attr attr{};
    attr.mq_maxmsg = MAX_MSG_COUNT;
    attr.mq_msgsize = MAX_MSG_SIZE;
    return attr;
}

static timespec MakeDeadlineAfterMs(int timeoutMs)
{
    timespec deadline{};
    if (clock_gettime(CLOCK_REALTIME, &deadline) == -1) {
        throw std::runtime_error(MakeMqErrorMessage("get deadline for", "host response", errno));
    }

    deadline.tv_sec += timeoutMs / 1000;
    deadline.tv_nsec += static_cast<long>(timeoutMs % 1000) * 1000L * 1000L;
    if (deadline.tv_nsec >= 1000L * 1000L * 1000L) {
        deadline.tv_sec += 1;
        deadline.tv_nsec -= 1000L * 1000L * 1000L;
    }
    return deadline;
}

class PosixMessageQueue {
public:
    PosixMessageQueue(const char* name, int flags, const mq_attr* attr = nullptr) : name_(ToPosixMqName(name))
    {
        if ((flags & O_CREAT) != 0) {
            mq_ = mq_open(name_.c_str(), flags, S_IRUSR | S_IWUSR, const_cast<mq_attr*>(attr));
        } else {
            mq_ = mq_open(name_.c_str(), flags);
        }
        if (mq_ == static_cast<mqd_t>(-1)) {
            throw std::runtime_error(MakeMqErrorMessage("open", name_, errno));
        }
    }

    ~PosixMessageQueue()
    {
        if (mq_ != static_cast<mqd_t>(-1)) {
            mq_close(mq_);
        }
    }

    PosixMessageQueue(const PosixMessageQueue&) = delete;
    PosixMessageQueue& operator=(const PosixMessageQueue&) = delete;

    void Send(const void* data, size_t size, unsigned int priority) const
    {
        while (mq_send(mq_, static_cast<const char*>(data), size, priority) == -1) {
            if (errno == EINTR) {
                continue;
            }
            throw std::runtime_error(MakeMqErrorMessage("send to", name_, errno));
        }
    }

    ssize_t Receive(void* buffer, size_t size, unsigned int* priority) const
    {
        while (true) {
            ssize_t recvdSize = mq_receive(mq_, static_cast<char*>(buffer), size, priority);
            if (recvdSize >= 0) {
                return recvdSize;
            }
            if (errno != EINTR) {
                throw std::runtime_error(MakeMqErrorMessage("receive from", name_, errno));
            }
        }
    }

    bool
    TimedReceive(void* buffer, size_t size, unsigned int* priority, const timespec& deadline, ssize_t& recvdSize) const
    {
        while (true) {
            recvdSize = mq_timedreceive(mq_, static_cast<char*>(buffer), size, priority, &deadline);
            if (recvdSize >= 0) {
                return true;
            }
            if (errno == EINTR) {
                continue;
            }
            if (errno == ETIMEDOUT) {
                return false;
            }
            throw std::runtime_error(MakeMqErrorMessage("receive from", name_, errno));
        }
    }

private:
    std::string name_;
    mqd_t mq_{static_cast<mqd_t>(-1)};
};

static bool IsBinDumpDisabled()
{
    const char* env = std::getenv("HCCL_VM_SKIP_BIN_DUMP");
    return env != nullptr && std::string(env) == "1";
}

static bool StartsWith(const std::string& value, const std::string& prefix) { return value.rfind(prefix, 0) == 0; }

std::string GetBinLocation()
{
    std::error_code ec;
    fs::path exePath = fs::read_symlink("/proc/self/exe", ec);
    if (ec) {
        throw std::runtime_error("read_symlink failed: " + ec.message());
    }
    fs::path binDir = exePath.parent_path();
    if (binDir.filename() == "bin") {
        return binDir.parent_path().string();
    }
    return binDir.string();
}

std::string ArgvToString(int argc, char* argv[])
{
    std::string cmd;
    for (int i = 0; i < argc; ++i) {
        std::string s = argv[i];
        // 如果包含空格，两头加引号
        if (s.find(' ') != std::string::npos) {
            cmd += "\"" + s + "\"";
        } else {
            cmd += s;
        }
        if (i < argc - 1) {
            cmd += " ";
        }
    }
    return cmd;
}

void RemoveFromLDPreload(const std::string& targetValue)
{
    HCCL_VM_DEBUG("clean LD_PRELOAD env: {}", targetValue);
    const char* curVal = std::getenv("LD_PRELOAD");

    if (curVal == nullptr) {
        return;
    }
    std::string envStr(curVal);

    // 如果当前值就是目标值，直接 unset
    if (envStr == targetValue) {
        unsetenv("LD_PRELOAD");
        return;
    }

    std::stringstream ss(envStr);
    std::string item;
    std::string envStrNew;
    bool first = true;

    while (std::getline(ss, item, ':')) {
        // 过滤空项（双冒号情况）和目标项
        if (item.empty() || item == targetValue) {
            continue;
        }
        if (!first) {
            envStrNew += ":";
        }
        envStrNew += item;
        first = false;
    }
    if (envStrNew.empty()) {
        // 如果结果为空，说明只包含要删除的项，直接 unset
        unsetenv("LD_PRELOAD");
    } else {
        // 覆盖原变量 (overwrite = 1)
        setenv("LD_PRELOAD", envStrNew.c_str(), 1);
    }
}

std::string FileInModelDir(const std::string& fileName)
{
    if (fileName == "ranktable") {
        std::string clusterInfo = InstallPath::ResolveToInstallRoot(fileName + ".json");
        std::ifstream f(clusterInfo.c_str());
        if (!f.good()) {
            return "config ranktable, but rank table file not found: " + clusterInfo;
        }
        return "";
    }
    std::string filePath = InstallPath::ResolveToInstallRoot("config/topo_meta/" + fileName + ".yaml");
    auto fileExistd = [&]() -> bool {
        std::ifstream f(filePath.c_str());
        return f.good();
    };
    if (fileExistd()) {
        return "";
    } else {
        return "[HVM] model File not found: " + filePath;
    }
}

std::string CheckClusterConfigFile(const std::string& topoFileName)
{
    if (topoFileName.empty()) {
        return "[HVM] topoFileName is empty";
    }
    // 检查topoFileName是否带.yaml后缀, 有则删除
    auto topoName = topoFileName;
    if (topoName.find(".yaml") != std::string::npos) {
        topoName.erase(topoName.find(".yaml"), 5);
    }

    std::string generateShellPath = InstallPath::ResolveToInstallRoot("script/generate_cluster_topo.sh");
    if (!fs::exists(generateShellPath)) {
        HCCL_VM_ERROR("generate_cluster_topo.sh not found: {}", generateShellPath);
        return "[HVM] generate_cluster_topo.sh not found: " + generateShellPath;
    }
    std::string clusterConfigFilePath = InstallPath::ResolveToInstallRoot("config/cluster/" + topoName + ".yaml");
    if (!fs::exists(clusterConfigFilePath)) {
        HCCL_VM_ERROR("cluster config file not found: {}", clusterConfigFilePath);
        return "[HVM] cluster config file not found: " + clusterConfigFilePath;
    }
    return "";
}

std::string GenerateClusterTopo(const std::string& topoFileName)
{
    std::string generateShellPath = InstallPath::ResolveToInstallRoot("script/generate_cluster_topo.sh");
    std::string clusterConfigFilePath = InstallPath::ResolveToInstallRoot("config/cluster/" + topoFileName + ".yaml");
    // 执行generate_cluster_topo.sh脚本（默认路径），生成集群组网
    std::string cmd = "bash " + generateShellPath + " " + clusterConfigFilePath;
    if (std::system(cmd.c_str()) != 0) {
        HCCL_VM_ERROR("generate_cluster_topo.sh failed: {}", cmd);
        return "[HVM] generate_cluster_topo.sh failed: " + cmd;
    }
    return "";
}

void ShowModel()
{
    std::string modelPath = InstallPath::ResolveToInstallRoot("cluster_model");
    if (!fs::exists(modelPath)) {
        HCCL_VM_ERROR("path not exist -> {}", modelPath);
        return;
    }
    if (!fs::is_directory(modelPath)) {
        HCCL_VM_ERROR("path not a dict -> {}", modelPath);
        return;
    }
    bool hasFiles = false;
    HCCL_VM_INFO("model : ");
    for (const auto& entry : fs::directory_iterator(modelPath)) {
        // 过滤：只关心"常规文件"，忽略子文件夹
        if (entry.is_regular_file()) {
            hasFiles = true;
            fs::path filePath = entry.path();
            HCCL_VM_INFO("  {}  [description] : ", filePath.stem().string());
            YAML::Node root = YAML::LoadFile(filePath);
            uint32_t podNum = root["meta"]["podNum"].as<uint32_t>();
            uint32_t serNum = root["meta"]["serNum"].as<uint32_t>();
            uint32_t rankNum = root["meta"]["rankNum"].as<uint32_t>();
            HCCL_VM_INFO("podNum: {}, serNum: {}, rankNum: {}", podNum, serNum, rankNum);
        }
    }
    if (!hasFiles) {
        HCCL_VM_WARN("there is no model");
    }
    return;
}

HcclVmResult InitHvmEnv(const std::string& configClusterDir, uint32_t level, bool checkOnlyMode)
{
    HCCL_VM_INFO("Enter InitHvmEnv: {}", configClusterDir);
    // 启动仿真环境
    // 创建用于Host-Device通信的共享内存
    // 仅校验模式才创建大块复用区 HcclCommPool，须在其它共享内存创建前，仅一次。
    if (checkOnlyMode) {
        void* commPool = sim::MemoryManager::GetInstance().AllocMemByName(
            sim::CommPoolPolicy::kPoolName, sim::CommPoolPolicy::kPoolSize);
        if (commPool == nullptr) {
            HCCL_VM_ERROR("create HcclCommPool fail");
            return HcclVmResult::HCCL_SIM_HOST_ERROR_CMD;
        }
    }

    // 解析集群拓扑，并初始化静态数据模型数据
    HCCL_VM_INFO("Initializing: Cluster Topo");
    AscendClusterTopoParser::GetInstance().InitClusterTopo(configClusterDir);

    std::string checkerTag = "@checker";
    auto chkInstallRet = InstallUserPlugin(checkerTag);
    if (chkInstallRet != HCCL_SIM_HOST_SUCCESS_CMD) {
        HCCL_VM_ERROR("default plugin install fail, please check your plugin path");
    }

    HCCL_VM_INFO("====================================");
    ShowUserPlugin();

    HCCL_VM_INFO("======================================");

    return HcclVmResult::HCCL_SIM_HOST_SUCCESS_CMD;
}

HcclVmResult InitHvmCommEnv(const TopoMeta& topoMeta, const std::string& configFileName, uint32_t level)
{
    // 通信域已初始化，则无需重复初始化
    if (AscendClusterTopoParser::GetInstance().GetClusterStatus() == HvmClusterStatus::COMM_DOMAIN_INIT_DONE) {
        HCCL_VM_ERROR("communication domain already initialized");
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }
    HcclVmResult ret;
    if (configFileName != "ranktable") {
        HCCL_VM_INFO("mock-comm cmd config topo yaml file, config by {}.yaml", configFileName);
        // 初始化本次算子通信域相关的表项
        ret = AscendClusterTopoParser::GetInstance().InitCommunicationDomain(topoMeta, false);
    } else {
        HCCL_VM_INFO("mock-comm cmd config rank table file, config by ranktable.json");
        std::string clusterInfo = InstallPath::ResolveToInstallRoot("data/ranktable.json");
        std::ifstream f(clusterInfo.c_str());
        if (!f.good()) {
            HCCL_VM_ERROR("ranktable.json file not found: {}", clusterInfo);
            return HcclVmResult::HCCL_SIM_E_INTERNAL;
        }
        // 从ranktable文件中读取通信域相关表项
        ret = AscendClusterTopoParser::GetInstance().ParseRanktableAndInitCommDomain(clusterInfo);
    }
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("InitHvmCommEnv failed");
        return ret;
    }
    setenv("RANK_TABLE_FILE", InstallPath::ResolveToInstallRoot("data/ranktable.json").c_str(), 1);

    return HcclVmResult::HCCL_SIM_HOST_SUCCESS_CMD;
}

static void StopListenThreads()
{
    g_serverListenFlag.store(false);
    g_runnerListenFlag.store(false);

    try {
        PosixMessageQueue wakeup(MQ_REQ_NAME, O_WRONLY | O_NONBLOCK);
        wakeup.Send("", 1, 0);
    } catch (...) {
    }

    if (g_serverThread && g_serverThread->joinable()) {
        g_serverThread->join();
    }
    if (g_runnerThread && g_runnerThread->joinable()) {
        g_runnerThread->join();
    }

    delete g_serverThread;
    delete g_runnerThread;
    g_serverThread = nullptr;
    g_runnerThread = nullptr;
}

static void ArchiveLogsAndData()
{
    const std::string& installRoot = InstallPath::GetHcclVmInstallAbsPath();
    if (installRoot.empty()) {
        return;
    }
    std::string archiveScript = installRoot + "/script/archive.sh";
    if (access(archiveScript.c_str(), F_OK) != 0) {
        return;
    }
    int status = system(("bash " + archiveScript).c_str());
    if (status != 0) {
        HCCL_VM_WARN("archive.sh exited with code {:d}", status);
    }
}

HcclVmResult HcclVmExit()
{
    // hccl-vm 退出清除所有使用资源
    StopListenThreads();
    HCCL_VM_INFO("start Destroy SharedMemory.");

    RemoveMessageQueue(MQ_REQ_NAME);
    RemoveMessageQueue(MQ_RESP_NAME);
    HcclPluginManager& pluginManager = HcclPluginManager::GetInstance();
    auto ret = pluginManager.StopAllPlugins();

    const fs::path backupDir = fs::path(InstallPath::ResolveToInstallRoot("data/" + MakeDataBackupTimestamp()));
    BackupAivTaskFiles(backupDir);

    HCCL_VM_INFO("start Destroy ALL Resources.");
    if (sim::IsCheckOnlyMode()) {
        sim::MemoryManager::GetInstance().FreeMemByName(sim::CommPoolPolicy::kPoolName);
    }
    int ret1 = system("sudo rm -fr /dev/shm/* 2>/dev/null");
    FlushLog();
    ArchiveLogsAndData();
    return ret;
}

HcclVmResult InstallUserPlugin(std::string argStr)
{
    // 处理插件tag和路径
    HcclVmResult ret{HcclVmResult::HCCL_SIM_HOST_ERROR_CMD};
    if (argStr.empty() || argStr[0] != '@') {
        HCCL_VM_ERROR("plugin tag should start with '@', invalid tag: {}", argStr);
        return ret;
    }
    argStr.erase(argStr.begin());

    // 注册插件
    HcclPluginManager& pluginManager = HcclPluginManager::GetInstance();
    ret = pluginManager.RegisterPlugin(argStr);
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("Install plugin [{}] failed", argStr);
        return ret;
    }

    // 装 Runner 时若仅校验模式开着，复用池仍在、大块仍会引流，可能覆盖 Runner 真实数据，告警提示。
    if (argStr == "runner" && sim::IsCheckOnlyMode()) {
        HCCL_VM_WARN("check-only mode on while runner installed; big-block contents are not guaranteed");
    }

    return HcclVmResult::HCCL_SIM_HOST_SUCCESS_CMD;
}

HcclVmResult RunUserPlugin(std::string argStr)
{
    nlohmann::json j;
    HcclVmResult ret = DumpData(j);

    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("DumpData failed");
        return ret;
    }
    HcclPluginManager& pluginManager = HcclPluginManager::GetInstance();
    ret = pluginManager.SendMessageToPlugin(argStr.substr(1), "status", j);
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("Run Plugin failed");
        return ret;
    }
    return HcclVmResult::HCCL_SIM_HOST_SUCCESS_CMD;
}

HcclVmResult UninstallUserPlugin(std::string argStr)
{
    std::vector<std::string> pluginTags{};

    size_t start = 0;
    size_t end = argStr.find(',');
    while (end != std::string::npos) {
        std::string tag = argStr.substr(start, end - start);
        tag.erase(tag.begin());
        pluginTags.push_back(tag);
        start = end + 1;
        end = argStr.find(',', start);
    }
    std::string lastTag = argStr.substr(start);
    lastTag.erase(lastTag.begin());
    pluginTags.push_back(lastTag);

    HcclPluginManager& pluginManager = HcclPluginManager::GetInstance();
    auto rets = pluginManager.StopPlugins(pluginTags);

    for (int i = 0; i < pluginTags.size(); ++i) {
        if (rets[i] != HcclVmResult::HCCL_SIM_SUCCESS) {
            HCCL_VM_ERROR("plugin Uninstall fail : {}", pluginTags[i]);
            return rets[i];
        }
    }

    return HcclVmResult::HCCL_SIM_HOST_SUCCESS_CMD;
}

void ShowUserPlugin()
{
    std::vector<std::string> listPlugins{};
    HcclPluginManager& pluginManager = HcclPluginManager::GetInstance();
    listPlugins = pluginManager.GetPluginStatus();

    if (listPlugins.empty()) {
        HCCL_VM_INFO("no plugin installed in hccl_vm");
    } else {
        for (auto& plugin : listPlugins) {
            HCCL_VM_INFO("{}", plugin);
        }
    }
    return;
}

HcclVmResult SetConsoleLogLevel(int level)
{
    (void)level;
    HCCL_VM_WARN("set console log level is disabled because ProxyConfig shared memory is removed");
    return HcclVmResult::HCCL_SIM_HOST_ERROR_CMD;
}

HcclVmResult SetFileLogLevel(int level)
{
    (void)level;
    HCCL_VM_WARN("set file log level is disabled because ProxyConfig shared memory is removed");
    return HcclVmResult::HCCL_SIM_HOST_ERROR_CMD;
}

HcclVmResult ShowCurrentLogLevel()
{
    HCCL_VM_WARN("show log level is disabled because ProxyConfig shared memory is removed");
    return HcclVmResult::HCCL_SIM_HOST_ERROR_CMD;
}

HcclVmResult StartHvmCmd()
{
    // 初始化HOST通信队列
    RemoveMessageQueue(MQ_REQ_NAME);
    RemoveMessageQueue(MQ_RESP_NAME);
    // 启动监听线程
    g_serverThread = new std::thread(ServerListen);
    g_runnerThread = new std::thread(RunnerListen);

    // Child Process(Bash)
    // 劫持库存在性判断
    std::string hcclVmbin = InstallPath::ResolveToInstallRoot("bin/hccl-vm");
    std::string libDir = "lib/" + GetArchStr() + "/";
    std::string proxyPathL0 = InstallPath::ResolveToInstallRoot(libDir + "libhccl_proxy_level0.so");
    std::string proxyPathL2
        = InstallPath::ResolveToInstallRoot(libDir + "libhccl_proxy_level" + std::to_string(g_hcclVmLevel) + ".so");
    if (!fs::exists(proxyPathL0) || !fs::exists(proxyPathL2)) {
        HCCL_VM_ERROR(
            "proxy hacking .so not found: l0={}, l2={}. please check your proxy hacking .so:"
            "1. Whether the hook library has been successfully built and installed. 2. Whether the simulation level "
            "matches the proxy hook library version. Current simulation level: {}, Default simulation level: 2",
            proxyPathL0, proxyPathL2, g_hcclVmLevel);
        return HCCL_SIM_HOST_ERROR_CMD;
    }
    std::string preload = proxyPathL0 + ":" + proxyPathL2;

    // 管道处理的用途是隔绝进程终端在std::cout中的残留
    int pipefds[2] = {-1, -1};
    if (pipe(pipefds) == -1) {
        HCCL_VM_ERROR("pipe failed");
        return HCCL_SIM_HOST_ERROR_CMD;
    }

    std::string safeBin = "'" + hcclVmbin + "'";
    std::string fdNum = std::to_string(pipefds[0]);
    std::string bashrcHack = "__HVM_SAVED_PATH=\"$PATH\";\n"
                             "__HVM_SAVED_LD_PRELOAD=\"$LD_PRELOAD\";\n"
                             "if [ -f ~/.bashrc ]; then . ~/.bashrc; fi;\n"
                             "export PATH=\"$__HVM_SAVED_PATH:$PATH\";\n"
                             "export LD_PRELOAD=\"$__HVM_SAVED_LD_PRELOAD\";\n"
                             "alias hccl-vm=\""
                             + safeBin
                             + "\";\n"
                               "PS1='(hvm)$> ';\n"
                               "unset __HVM_SAVED_PATH __HVM_SAVED_LD_PRELOAD;\n"
                               "exec "
                             + fdNum + "<&-;\n";
    ssize_t written = write(pipefds[1], bashrcHack.c_str(), bashrcHack.size());
    if (written != static_cast<ssize_t>(bashrcHack.size())) {
        HCCL_VM_ERROR("Failed to write full script to pipe");
        close(pipefds[0]);
        close(pipefds[1]);
        return HCCL_SIM_HOST_ERROR_CMD;
    }
    close(pipefds[1]);
    g_hcclVmBashFlag = true;

    std::string devFdPath = "/dev/fd/" + std::to_string(pipefds[0]);

    char* bashArgv[]
        = {const_cast<char*>("bash"), const_cast<char*>("--rcfile"), const_cast<char*>(devFdPath.c_str()),
           const_cast<char*>("-i"), nullptr};

    // Fork Bash
    pid_t pid = fork();
    if (pid == -1) {
        HCCL_VM_ERROR("fork failed: {}", std::strerror(errno));
        auto ret = HcclVmExit();
        close(pipefds[0]);
        return (ret == HcclVmResult::HCCL_SIM_HOST_SUCCESS_CMD) ? HcclVmResult::HCCL_SIM_HOST_ERROR_CMD : ret;
    } else if (pid == 0) {
        setenv(HVM_BASH_ENV_KEY.c_str(), g_binDir.c_str(), 1);
        setenv("LD_PRELOAD", preload.c_str(), 1);
        setenv("HCCL_VM_INSTALL_ROOT", g_binDir.c_str(), 1);
        setenv("RANK_TABLE_FILE", InstallPath::ResolveToInstallRoot("data/ranktable.json").c_str(), 1);
        execv("/bin/bash", bashArgv);
        exit(1);
    } else {
        // Parent Process (Host)
        // 等待 bash 结束 (阻塞等待，保持Host存活)
        int status;
        waitpid(pid, &status, 0);
        HCCL_VM_INFO("Shell exited. Host shutting down.");
        auto ret = HcclVmExit();
    }
    return HcclVmResult::HCCL_SIM_HOST_SUCCESS_CMD;
}

void StartHostClient(int argc, char* argv[])
{
    // 打开队列
    PosixMessageQueue mqReq(MQ_REQ_NAME, O_RDWR);
    PosixMessageQueue mqResp(MQ_RESP_NAME, O_RDWR);

    std::string cmd = ArgvToString(argc, argv);
    if (argc == 1) {
        cmd += " --help";
    }
    // 向主host发生命令行
    mqReq.Send(cmd.data(), cmd.size(), 0);
    // 等待回执，简单回执
    timespec deadline = MakeDeadlineAfterMs(HOST_CLIENT_RESP_TIMEOUT_MS);
    char buffer[MAX_MSG_SIZE];
    ssize_t recvdSize = 0;
    unsigned int priority = 0;
    bool hasMessage = mqResp.TimedReceive(&buffer, MAX_MSG_SIZE, &priority, deadline, recvdSize);
    if (hasMessage) {
        HCCL_VM_DEBUG("Client : Command parsing response received: {}", std::string(buffer, recvdSize));
    } else {
        HCCL_VM_WARN("Client : Command parsing out of time, Client quit");
    }
}

void ServerListen()
{
    // 创建队列
    g_serverListenFlag.store(true);
    mq_attr attr = MakeMessageQueueAttr();
    PosixMessageQueue mqReq(MQ_REQ_NAME, O_CREAT | O_EXCL | O_RDWR, &attr);
    PosixMessageQueue mqResp(MQ_RESP_NAME, O_CREAT | O_EXCL | O_RDWR, &attr);
    HCCL_VM_INFO("HOST Server listening...");

    while (g_serverListenFlag.load()) {
        char buffer[MAX_MSG_SIZE];
        unsigned int priority = 0;
        ssize_t recvdSize = -1;
        timespec deadline = MakeDeadlineAfterMs(500);
        if (!mqReq.TimedReceive(buffer, MAX_MSG_SIZE, &priority, deadline, recvdSize)) {
            continue;
        }

        std::string cmd(buffer, recvdSize);
        HCCL_VM_DEBUG("HOST : Command parsing request received: {}", cmd);

        // 执行接收到的命令逻辑
        ParseCommand(cmd);
        std::string resp = "Success SubCommand Received";
        // 发送回执, 简易回执
        mqResp.Send(resp.data(), resp.size(), 0);
    }
    HCCL_VM_INFO("HOST Server listening thread exit...");
}

// 监听proxy和runner进程，中转进程状态
void RunnerListen()
{
    HCCL_VM_INFO("Runner listening...");
    g_runnerListenFlag.store(true);
    sim::ProcessSyncer syncer;
    syncer.Init();
    while (g_runnerListenFlag.load()) {
        // 1. 监听DeviceStatus，等待所有proxy进程都进入等待状态
        auto allStatus = RunnerDB::GetByPred<sim::DeviceStatus>([](const sim::DeviceStatus& d) {
            return d.synchronize_strategy == 1;
        });
        auto allRank = RunnerDB::GetByPred<sim::Rank>([](const sim::Rank& d) {
            return true;
        });

        if (allStatus.size() != allRank.size() || allRank.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }
        auto ret = HcclVmResult::HCCL_SIM_SUCCESS;
        // 单算子场景才dump数据
        if (syncer.getCurrentRound() == 0) {
            if (IsBinDumpDisabled()) {
                HCCL_VM_INFO("skip dumping bin files (HCCL_VM_SKIP_BIN_DUMP=1)");
            } else {
                ret = DumpDataToFile("runner");
                if (ret == HcclVmResult::HCCL_SIM_E_SKIP) {
                    HCCL_VM_INFO("skip dumping data (multi-op scenario)");
                } else if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
                    HCCL_VM_WARN("dump data to file failed. ret: {:d}", static_cast<int>(ret));
                }
            }
        } else if (syncer.getCurrentRound() >= 1) {
            HCCL_VM_INFO("skip dumping data for round {:d}", syncer.getCurrentRound());
            std::string dataDir = InstallPath::ResolveToInstallRoot("data") + "/";
            std::remove((dataDir + "runner_hcclvm_instr_data.bin").c_str());
            std::remove((dataDir + "runner_hcclvm_syn_data.bin").c_str());
            std::remove((dataDir + "runner_hcclvm_task_data.bin").c_str());
        }

        bool isRunnerRunning = true;
        {
            int fd = open("/tmp/hccl_vm_runner.lock", O_RDONLY);
            if (fd < 0) {
                isRunnerRunning = false;
            } else if (flock(fd, LOCK_EX | LOCK_NB) == 0) {
                flock(fd, LOCK_UN);
                close(fd);
                isRunnerRunning = false;
            } else {
                close(fd);
            }
        }
        // runner 未安装, 也会barrier proxy进程
        if (!isRunnerRunning) {
            uint32_t targetRound = syncer.getCurrentRound() + 1;
            HCCL_VM_INFO("{:d} rank ready, barrier-only sync completed. targetRound={}", allStatus.size(), targetRound);
            for (auto& ds : allStatus) {
                auto dsId = ds.id;
                RunnerDB::Update<sim::DeviceStatus>(dsId, [](sim::DeviceStatus& ds) {
                    ds.synchronize_strategy = 0;
                });
            }
            syncer.notifyProxyToContinue(targetRound);
            continue;
        }

        HCCL_VM_INFO("{:d} rank ready, start runner...", allStatus.size());

        // 2. 清除DeviceStatus表项
        for (auto& ds : allStatus) {
            auto dsId = ds.id;
            RunnerDB::Update<sim::DeviceStatus>(dsId, [](sim::DeviceStatus& ds) {
                ds.synchronize_strategy = 0;
            });
        }

        {
            int fd = open("/tmp/hccl_vm_runner.lock", O_RDONLY);
            if (fd < 0) {
                close(fd);
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }
            if (flock(fd, LOCK_EX | LOCK_NB) == 0) {
                // 成功获得锁，说明runner进程已经结束，清理状态继续监听
                flock(fd, LOCK_UN);
                close(fd);
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }
            close(fd);
        }

        uint32_t targetRound = syncer.getCurrentRound() + 1;
        syncer.notifyRunnerAndWaitAcknowledge(targetRound);
        HCCL_VM_INFO("The task is ready, notify runner to execute. targetRound={}", targetRound);
    }
}

void ParseCommand(std::string& cmd)
{
    CLI::App app{"Hccl Virtual Simulator"};
    app.set_help_all_flag("--help-all", "show all the sub command help");

    auto commands = CommandRegistry::CreateAll();

    for (auto& command : commands) {
        command->Setup(app);
    }
    try {
        app.parse(cmd, true);            // true 表示命令第一个字段为程序名
    } catch (const CLI::ParseError& e) { // 非help请求，强行打印help
        if (e.get_exit_code() != 0) {
            std::cerr << "[HVM] [ERROR] para error, please check:\n\n" << app.help() << std::endl;
        }
        if (g_hcclVmBashFlag) {
            app.exit(e);
        } else {
            std::exit(app.exit(e));
        }
    }
}

static void BackupDatabase(const std::string& srcPath)
{
    if (!fs::exists(srcPath)) {
        HCCL_VM_INFO("Source database {} does not exist, skipping backup.", srcPath);
        return;
    }

    fs::path dataDir = fs::path(InstallPath::ResolveToInstallRoot("data"));
    fs::create_directories(dataDir);

    auto now = std::chrono::system_clock::now();
    auto timeT = std::chrono::system_clock::to_time_t(now);
    std::tm tmBuf{};
    localtime_r(&timeT, &tmBuf);
    std::ostringstream oss;
    oss << std::put_time(&tmBuf, "%Y%m%d_%H%M%S");

    std::string destPath = (dataDir / ("hccl_vm_data_backup_" + oss.str() + ".db")).string();

    auto ret = HcclSim::DB::OpDbOps::Instance().Backup(destPath);
    if (ret != HcclSim::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("Failed to backup database to {}", destPath);
    } else {
        HCCL_VM_INFO("Database backed up to: {}", destPath);
    }
}

static std::string MakeDataBackupTimestamp()
{
    auto now = std::chrono::system_clock::now();
    auto timeT = std::chrono::system_clock::to_time_t(now);
    std::tm tmBuf{};
    localtime_r(&timeT, &tmBuf);
    std::ostringstream oss;
    oss << std::put_time(&tmBuf, "%Y%m%d%H%M%S");
    return oss.str();
}

static bool EnsureBackupDir(const fs::path& backupDir)
{
    std::error_code ec;
    fs::create_directories(backupDir, ec);
    if (ec) {
        HCCL_VM_WARN("failed to create backup directory {}: {}", backupDir.string(), ec.message());
        return false;
    }
    return true;
}

static void BackupBinFiles(const fs::path& backupDir)
{
    fs::path dataDir = fs::path(InstallPath::ResolveToInstallRoot("data"));
    std::vector<fs::path> binFiles
        = {dataDir / "runner_hcclvm_instr_data.bin", dataDir / "runner_hcclvm_syn_data.bin",
           dataDir / "runner_hcclvm_task_data.bin"};

    bool hasBinFile = std::any_of(binFiles.begin(), binFiles.end(), [](const fs::path& f) {
        return fs::exists(f);
    });
    if (!hasBinFile) {
        return;
    }

    if (!EnsureBackupDir(backupDir)) {
        return;
    }

    std::error_code ec;
    for (const auto& f : binFiles) {
        if (!fs::exists(f)) {
            continue;
        }
        fs::rename(f, backupDir / f.filename(), ec);
        if (ec) {
            HCCL_VM_WARN("failed to move {} to {}: {}", f.string(), (backupDir / f.filename()).string(), ec.message());
        }
    }
}

static void BackupAivTaskFiles(const fs::path& backupDir)
{
    fs::path dataDir = fs::path(InstallPath::ResolveToInstallRoot("data"));
    std::error_code ec;
    if (!fs::exists(dataDir, ec) || ec || !fs::is_directory(dataDir, ec)) {
        if (ec) {
            HCCL_VM_WARN("failed to access data directory {}: {}", dataDir.string(), ec.message());
        }
        return;
    }

    std::vector<fs::path> aivTaskFiles;
    static const std::string aivFilePrefix = "hcclvm_aiv_";
    fs::directory_iterator iter(dataDir, ec);
    if (ec) {
        HCCL_VM_WARN("failed to iterate data directory {}: {}", dataDir.string(), ec.message());
        return;
    }
    for (const fs::directory_iterator end; iter != end; iter.increment(ec)) {
        if (ec) {
            HCCL_VM_WARN("failed to iterate data directory {}: {}", dataDir.string(), ec.message());
            return;
        }

        const fs::path filePath = iter->path();
        std::error_code statusEc;
        const fs::file_status status = fs::symlink_status(filePath, statusEc);
        if (statusEc || !fs::is_regular_file(status)) {
            continue;
        }

        const std::string fileName = filePath.filename().string();
        if (StartsWith(fileName, aivFilePrefix)) {
            aivTaskFiles.emplace_back(filePath);
        }
    }

    if (aivTaskFiles.empty()) {
        return;
    }
    if (!EnsureBackupDir(backupDir)) {
        return;
    }

    for (const auto& f : aivTaskFiles) {
        ec.clear();
        fs::rename(f, backupDir / f.filename(), ec);
        if (ec) {
            HCCL_VM_WARN("failed to move {} to {}: {}", f.string(), (backupDir / f.filename()).string(), ec.message());
        }
    }
}

static void CleanShmFilesByPrefix(const std::vector<std::string>& prefixes)
{
    const fs::path shmDir = "/dev/shm";
    std::error_code ec;
    if (!fs::exists(shmDir, ec) || ec || !fs::is_directory(shmDir, ec)) {
        if (ec) {
            HCCL_VM_WARN("failed to access shared memory directory {}: {}", shmDir.string(), ec.message());
        }
        return;
    }

    uint32_t removedCount = 0;
    uint32_t failedCount = 0;
    fs::directory_iterator iter(shmDir, ec);
    if (ec) {
        HCCL_VM_WARN("failed to iterate shared memory directory {}: {}", shmDir.string(), ec.message());
        return;
    }
    for (const fs::directory_iterator end; iter != end; iter.increment(ec)) {
        if (ec) {
            HCCL_VM_WARN("failed to iterate shared memory directory {}: {}", shmDir.string(), ec.message());
            return;
        }

        const fs::path filePath = iter->path();
        const std::string fileName = filePath.filename().string();
        const bool matched = std::any_of(prefixes.begin(), prefixes.end(), [&fileName](const std::string& prefix) {
            return StartsWith(fileName, prefix);
        });
        if (!matched) {
            continue;
        }

        std::error_code statusEc;
        const fs::file_status status = fs::symlink_status(filePath, statusEc);
        if (statusEc || !fs::is_regular_file(status)) {
            continue;
        }

        std::error_code removeEc;
        if (!fs::remove(filePath, removeEc) || removeEc) {
            ++failedCount;
            HCCL_VM_WARN("failed to remove shared memory file {}: {}", filePath.string(), removeEc.message());
            continue;
        }
        ++removedCount;
    }
    HCCL_VM_INFO("cleaned {} shared memory files in {}, failed {}", removedCount, shmDir.string(), failedCount);
}

HcclVmResult ClearDbTables()
{
    const fs::path backupDir = fs::path(InstallPath::ResolveToInstallRoot("data/" + MakeDataBackupTimestamp()));
    BackupBinFiles(backupDir);
    BackupAivTaskFiles(backupDir);
    BackupDatabase(InstallPath::ResolveToInstallRoot("data/hccl_vm_data.db"));

    // 1. 清空 OpDbOps (SQLite) 中的静态表数据
    std::vector<std::string> staticTables
        = {"opDetails", "opMemInfo", "syncRecords", "ccuChannels", "JettyMaps", "ccuInstrRes", "ccuInstr"};
    for (const auto& table : staticTables) {
        HcclSim::DB::OpDbOps::Instance().ExecUpdate("DELETE FROM " + table, {});
    }

    // 2. 删除 OpDbOps (SQLite) 中的动态表 (opTask_P_*)
    std::vector<std::vector<std::string>> rows;
    std::string querySql = "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'opTask_P_%'";
    if (HcclSim::DB::OpDbOps::Instance().ExecQuery(querySql, {}, rows) == 0) {
        for (const auto& row : rows) {
            if (!row.empty()) {
                HcclSim::DB::OpDbOps::Instance().ExecUpdate("DROP TABLE IF EXISTS " + row[0], {});
            }
        }
    }

    // 3. 清空 RunnerDB (内存/SHM) 表
    RunnerDB::DeleteAll<sim::PhyMemBlock>();
    RunnerDB::DeleteAll<sim::VirtualMemBlock>();
    RunnerDB::DeleteAll<sim::RaDevice>();
    RunnerDB::DeleteAll<sim::RaContext>();
    RunnerDB::DeleteAll<sim::RaQP>();
    RunnerDB::DeleteAll<sim::RaJetty>();
    RunnerDB::DeleteAll<sim::EndPointPair>();
    RunnerDB::DeleteAll<sim::MemoryLayout>();
    RunnerDB::DeleteAll<sim::SimModelData>();
    auto ccuRealTab = RunnerDB::GetByPred<sim::CcuResource>([](auto&&) {
        return true;
    });
    for (auto& tmp : ccuRealTab) {
        RunnerDB::Update<sim::CcuResource>(tmp.id, [](sim::CcuResource& ccuRes) {
            ccuRes.instr_cnt = 0;
            ccuRes.state = 0;
            memset(ccuRes.instr_space, 0, sizeof(ccuRes.instr_space));
        });
    }
    RunnerDB::DeleteAll<sim::CcuChannel>();
    RunnerDB::DeleteAll<sim::DeviceStatus>();
    RunnerDB::DeleteAll<sim::Task>();
    RunnerDB::DeleteAll<sim::Runner>();
    RunnerDB::DeleteAll<sim::RaSocket>();
    RunnerDB::DeleteAll<sim::RaSocketPair>();

    CleanShmFilesByPrefix({"DEV", "ra_sock_"});

    sim::ProcessSyncer syncer;
    syncer.Reset();

    return HcclVmResult::HCCL_SIM_HOST_SUCCESS_CMD;
}

extern uint64_t g_cur_server_key;

HcclVmResult HcclVmResetCommDomain()
{
    AscendClusterTopoParser::GetInstance().SetClusterStatus(HvmClusterStatus::COMM_DOMAIN_UNINIT);
    // 重置 Host 进程中缓存的 server key，防止跨用例残留
    g_cur_server_key = 0;
    // 重置device的逻辑ID
    auto ret2 = sim::ResetAllDeviceLogicId();
    if (!ret2) {
        HCCL_VM_ERROR("reset all device logic id failed.");
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }
    // 清除Rank/CcuResource表
    RunnerDB::DeleteAll<sim::Rank>();
    RunnerDB::DeleteAll<sim::CcuResource>();
    return ClearDbTables();
}

HcclVmResult CopyFile(const std::string& clusterDir)
{
    // 1. 拼接源文件和目标文件路径
    auto srcPath = clusterDir + "/superpod0/server0/topo.json";
    fs::create_directories(fs::path(InstallPath::ResolveToInstallRoot("data")));
    auto destPath = InstallPath::ResolveToInstallRoot("data/topo.json");

    // 显式提示：文件已存在，将覆盖
    if (fs::exists(destPath)) {
        HCCL_VM_WARN("target file already exists, will overwrite: topo.json");
    }

    // 2. 二进制模式打开文件（显式指定 trunc 覆盖，语义更清晰）
    std::ifstream srcFile(srcPath, std::ios::binary);
    std::ofstream destFile(destPath, std::ios::binary | std::ios::trunc);

    // 打开失败检查
    if (!srcFile.is_open() || !destFile.is_open()) {
        HCCL_VM_ERROR("open file failed. src: {}, dest: {}", srcPath, destPath);
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    // 3. 执行文件拷贝
    destFile << srcFile.rdbuf();

    // 4. 检查拷贝是否失败
    if (srcFile.bad() || destFile.bad()) {
        HCCL_VM_ERROR("copy file failed. src: {}, dest: {}", srcPath, destPath);
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    // 5. 手动关闭文件（确保数据写入磁盘）
    srcFile.close();
    destFile.close();

    HCCL_VM_INFO("✅ topo.json copy success!");

    return HcclVmResult::HCCL_SIM_SUCCESS;
}
