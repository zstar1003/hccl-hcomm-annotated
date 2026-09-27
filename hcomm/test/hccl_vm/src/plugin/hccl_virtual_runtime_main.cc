/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <filesystem>
#include <iostream>
#include <nlohmann_json/json.hpp>
#include <string>
#include <sys/file.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>

#include "ccu_resource_manager.h"
#include "device_resource_manager.h"
#include "sim_models.h"
#include "store_dump_shm_data.h"
#include "sim_common_defs.h"
#include "sim_common_macro.h"
#include "hccl_task_sequential_execute.h"
#include "sim_log.h"
#include "sim_loader.h"
#include "sim_process_syncer.h"
#include "storage_manager.h"

using json = nlohmann::json;

namespace {
using RuntimeClock = std::chrono::steady_clock;
constexpr uint64_t RUNTIME_NS_PER_MS = 1000000ULL;

static std::vector<std::map<uint32_t, sim::CompositeOpDetail>>
TransposeCompositeOpMap(const std::map<uint32_t, std::vector<sim::CompositeOpDetail>>& compositeDataMap)
{
    size_t maxOps = 0;
    for (const auto& entry : compositeDataMap) {
        maxOps = std::max(maxOps, entry.second.size());
    }
    std::vector<std::map<uint32_t, sim::CompositeOpDetail>> opGroups(maxOps);
    for (const auto& entry : compositeDataMap) {
        uint32_t rankId = entry.first;
        const auto& ops = entry.second;
        for (size_t i = 0; i < ops.size(); i++) {
            opGroups[i][rankId] = ops[i];
        }
    }
    return opGroups;
}

uint64_t ElapsedRuntimeMs(RuntimeClock::time_point start, RuntimeClock::time_point end)
{
    const auto elapsedNs = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    return static_cast<uint64_t>(elapsedNs) / RUNTIME_NS_PER_MS;
}

struct RunVirtualRuntimeTimeStats {
    uint64_t runCount{0};
    uint64_t totalCostMs{0};
    uint64_t lastCostMs{0};
    uint64_t maxCostMs{0};
};

void RecordRunVirtualRuntimeTime(RunVirtualRuntimeTimeStats& stats, uint64_t costMs)
{
    ++stats.runCount;
    stats.totalCostMs += costMs;
    stats.lastCostMs = costMs;
    if (costMs > stats.maxCostMs) {
        stats.maxCostMs = costMs;
    }
}

void DumpRunVirtualRuntimeTimeStats(const RunVirtualRuntimeTimeStats& stats)
{
    if (stats.runCount == 0) {
        HCCL_VM_INFO("RunVirtualRuntime no execution, runCount=0");
        return;
    }
    HCCL_VM_INFO(
        "RunVirtualRuntime summary, runCount={}, totalCostMs={}, lastCostMs={}, "
        "maxCostMs={}",
        stats.runCount, stats.totalCostMs, stats.lastCostMs, stats.maxCostMs);
}

bool IsAivOpExpansionMode(uint32_t opExpansionMode)
{
    return opExpansionMode == static_cast<uint32_t>(sim::SimOpExpansionMode::SIM_OP_EXPANSION_MODE_AIV);
}
} // namespace

static const char* HcclCmdTypeToString(HcclCMDType t)
{
    switch (t) {
        case HCCL_CMD_BROADCAST:
            return "Broadcast";
        case HCCL_CMD_ALLREDUCE:
            return "AllReduce";
        case HCCL_CMD_REDUCE:
            return "Reduce";
        case HCCL_CMD_SEND:
            return "Send";
        case HCCL_CMD_RECEIVE:
            return "Recv";
        case HCCL_CMD_ALLGATHER:
            return "AllGather";
        case HCCL_CMD_REDUCE_SCATTER:
            return "ReduceScatter";
        case HCCL_CMD_ALLTOALLV:
            return "AllToAllV";
        case HCCL_CMD_ALLTOALLVC:
            return "AllToAllVC";
        case HCCL_CMD_ALLTOALL:
            return "AllToAll";
        case HCCL_CMD_SCATTER:
            return "Scatter";
        case HCCL_CMD_BATCH_SEND_RECV:
            return "BatchSendRecv";
        case HCCL_CMD_ALLGATHER_V:
            return "AllGatherV";
        case HCCL_CMD_REDUCE_SCATTER_V:
            return "ReduceScatterV";
        default:
            return "Unknown";
    }
}

static const char* HcclDataTypeToString(HcclDataType t)
{
    switch (t) {
        case HCCL_DATA_TYPE_INT8:
            return "INT8";
        case HCCL_DATA_TYPE_INT16:
            return "INT16";
        case HCCL_DATA_TYPE_INT32:
            return "INT32";
        case HCCL_DATA_TYPE_FP16:
            return "FP16";
        case HCCL_DATA_TYPE_FP32:
            return "FP32";
        case HCCL_DATA_TYPE_INT64:
            return "INT64";
        case HCCL_DATA_TYPE_UINT64:
            return "UINT64";
        case HCCL_DATA_TYPE_UINT8:
            return "UINT8";
        case HCCL_DATA_TYPE_UINT16:
            return "UINT16";
        case HCCL_DATA_TYPE_UINT32:
            return "UINT32";
        case HCCL_DATA_TYPE_FP64:
            return "FP64";
        case HCCL_DATA_TYPE_BFP16:
            return "BFP16";
        default:
            return "Unknown";
    }
}

static const char* HcclReduceOpToString(HcclReduceOp t)
{
    switch (t) {
        case HCCL_REDUCE_SUM:
            return "SUM";
        case HCCL_REDUCE_PROD:
            return "PROD";
        case HCCL_REDUCE_MAX:
            return "MAX";
        case HCCL_REDUCE_MIN:
            return "MIN";
        default:
            return "Unknown";
    }
}

// 使用原子变量控制程序生命周期
std::atomic<bool> g_keep_running{true};
loader::Loader g_loader;

// --- stdin 命令处理（与 checker 保持一致） ---
void ProcessCommand(const std::string& line)
{
    try {
        auto j = json::parse(line);
        std::string action = j.value("action", "");

        if (action == "stop") {
            HCCL_VM_INFO("Stop signal received. Initiating shutdown...");
            g_keep_running.store(false);
        }
        // 未来可扩展其他 action...
    } catch (const std::exception& e) {
        HCCL_VM_ERROR("Command processing failed: {}", e.what());
    }
}

// --- 业务函数 ---
void RunVirtualRuntime(HcclSim::StorageManager& storage)
{
    std::vector<sim::CcuChannelTab> channels;
    g_loader.GetCcuChannelInfo(channels);

    std::vector<sim::SyncRecordTab> records;
    g_loader.GetSyncRecordsByStatus(0, records);
    if (records.size() == 0) {
        HCCL_VM_ERROR("can not get one effective sync iter.");
        return;
    }

    std::map<uint32_t, std::vector<sim::CompositeOpDetail>> compositeDataMap;
    g_loader.LoadRunnerSingleSync(records[0].syncIter, compositeDataMap);

    // 清理CCU资源管理器状态，防止跨sync iter状态污染（KN/XN/MS/simulators）
    // 注意：instrSpace_ 也会被清理，但会在后续的 InitCcuResource 中重新设置
    CcuResourceManager::GetInstance().Reset();
    storage.ResetAivResource();

    auto opTasks = TransposeCompositeOpMap(compositeDataMap);
    for (auto& rankTask : opTasks) {
        for (auto& it : rankTask) {
            auto& opDetail = it.second;
            storage.LoadHcclVmSynthesisData(opDetail.detail, channels);
            storage.LoadHcclVmTaskMetaData(opDetail.tasks);
            if (IsAivOpExpansionMode(opDetail.detail.opExpansionMode)) {
                auto ret = storage.InitAivResourceFromCompositeOpDetail(opDetail);
                if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
                    storage.ResetAivResource();
                    HCCL_VM_ERROR("Aiv resource init fail.");
                    return;
                }
            }
        }
    }

    std::vector<sim::CcuInstrResTab> instrRes;
    g_loader.GetInstrResInfo(instrRes);

    // 初始化CCU资源
    storage.InitCcuResource(instrRes);

    HcclSim::CheckerParam param = storage.GetCheckerParam();
    HcclCMDType cmdType = param.cmdType;
    HCCL_VM_INFO(
        "start check semantic ....: cmdType = {}, reduceOp = {}", static_cast<uint32_t>(cmdType),
        static_cast<uint32_t>(param.reduceType));

    HCCL_VM_INFO(
        "syncIter = {}, op = {}, rankSize = {}, dataType = {}, dataCount = {}, "
        "reduceType = {}, srcRank = {}, dstRank = {}, root = {}",
        records[0].syncIter, HcclCmdTypeToString(cmdType), param.rankSize, HcclDataTypeToString(param.dataType),
        param.dataCount, HcclReduceOpToString(param.reduceType), param.srcRank, param.dstRank, param.root);

    // 获取所有rank的任务队列
    auto rootPath = storage.FindRootPath();
    HcclSim::AllRankTaskQueues& allRankTaskQueues = storage.GetAllRankTaskQueues();
    VirtualRunTime::SqeuentialExecutor executor(allRankTaskQueues, rootPath);
    auto allTaskSize = allRankTaskQueues.size();
    executor.Execute();
    // 查看input/output的buffer数据
    storage.DumpAllRankInputOutput(opTasks);
    // 释放内存
    storage.ReleasePhyMem();
    storage.Reset();
    HCCL_VM_INFO(
        "==============Runner Success Iter :{:d} tasks:{:d}============================", records[0].syncIter,
        allTaskSize);
}

void init_lock()
{
    int fd = open("/tmp/hccl_vm_runner.lock", O_CREAT | O_RDWR, 0666);
    if (fd == -1) {
        HCCL_VM_ERROR("Failed to open lock file: {}", strerror(errno));
        exit(1);
    }

    // open() 的创建权限会受 umask 影响，显式恢复为所有用户可读写。
    if (fchmod(fd, 0666) == -1 && errno != EPERM) {
        HCCL_VM_ERROR("Failed to set lock file permissions: {}", strerror(errno));
        close(fd);
        exit(1);
    }

    if (flock(fd, LOCK_EX | LOCK_NB) == -1) {
        if (errno == EWOULDBLOCK) {
            HCCL_VM_ERROR("Another instance of runner is already running. Exiting.");
        } else {
            HCCL_VM_ERROR("Failed to acquire lock: {}", strerror(errno));
        }
        close(fd);
        exit(1);
    }
}

int main(int argc, char* argv[])
{
    (void)argc;
    (void)argv;
    LogConfig config = LoadLogConfig("runner");
    InitLogger(config);
    init_lock();
    HcclSim::StorageManager& storage = HcclSim::StorageManager::GetInstance();
    storage.SetDataId("runner");

    sim::ProcessSyncer syncer;

    g_loader.LoadOpTaskFile();
    // 启动 stdin 监听线程（与 checker 保持一致）
    // 读取 Host 通过 pipe 发送的控制命令（如 "stop"）
    std::thread([]() {
        std::string line;
        while (g_keep_running.load() && std::getline(std::cin, line)) {
            if (line.empty()) {
                continue;
            }
            ProcessCommand(line);
        }
        // EOF 或 stop 命令后，确保标志被设置
        g_keep_running.store(false);
        HCCL_VM_INFO("stdin reader thread exiting.");
    }).detach();

    while (g_keep_running.load()) {
        uint32_t targetRound = syncer.checkProxyReadySignal();
        if (targetRound == 0) {
            // 无任务，继续轮询
            std::this_thread::sleep_for(std::chrono::seconds(1));
            continue;
        }
        if (targetRound == sim::kRunnerExitSignal) {
            HCCL_VM_INFO("Exit signal received from hccl-vm.");
            break;
        }
        HCCL_VM_INFO("Task received, targetRound={}", targetRound);
        RunVirtualRuntime(storage);
        syncer.notifyProxyToContinue(targetRound);
        HCCL_VM_INFO("Round {} completed.", targetRound);
        FlushLog(); // 将本轮完整日志落盘
    }

    HCCL_VM_INFO("Exiting...");
    return 0;
}
