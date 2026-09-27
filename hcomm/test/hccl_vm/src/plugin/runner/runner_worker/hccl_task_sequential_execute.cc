/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl_task_sequential_execute.h"

#include <cstdint>
#include <cstdlib>
#include <execinfo.h>
#include <iostream>
#include <signal.h>
#include <unistd.h>

#include "device_resource_manager.h"
#include "sim_log.h"
#include "trace/ccu_trace_collector.h"
#include "trace/ccu_trace_serializer.h"
#include "trace/ccu_trace_types.h"

using namespace HcclSim;

// ===== 信号处理：崩溃时紧急 dump trace 数据 =====
// 使用全局变量传递崩溃输出路径（signal handler 参数有限制）
static const char* g_crashDumpPath = nullptr;
static std::string g_crashDumpPathStorage;

static void CrashDumpHandler(int sig)
{
    // 1. 恢复默认信号处理，防止 dump 过程中再次触发信号导致无限递归
    signal(sig, SIG_DFL);

    // 2. 紧急 dump 已采集的 trace 数据
    auto& collector = CcuTrace::CcuTraceCollector::GetInstance();
    if (collector.IsEnabled() && g_crashDumpPath != nullptr) {
        collector.EndRun();
        auto traceRun = collector.GetTraceRun();
        CcuTrace::CcuTraceSerializer::DumpToFile(traceRun, std::string(g_crashDumpPath));
    }

    // 3. 打印调用栈辅助定位
    void* frames[64];
    int n = backtrace(frames, 64);
    backtrace_symbols_fd(frames, n, STDERR_FILENO);

    // 4. 重新触发默认信号处理（产生 core dump）
    raise(sig);
}

namespace VirtualRunTime {
// 增量落盘的默认间隔（指令条数），可通过环境变量 HCCLVM_TRACE_FLUSH_INTERVAL 覆盖
static constexpr uint32_t DEFAULT_TRACE_FLUSH_INTERVAL = 100;

static uint32_t GetFlushInterval()
{
    const char* env = std::getenv("HCCLVM_TRACE_FLUSH_INTERVAL");
    if (env != nullptr) {
        uint32_t val = static_cast<uint32_t>(std::strtoul(env, nullptr, 10));
        if (val > 0) {
            return val;
        }
    }
    return DEFAULT_TRACE_FLUSH_INTERVAL;
}

SqeuentialExecutor::SqeuentialExecutor(AllRankTaskQueues& allRankTaskQueues, const std::string& rootPath)
{
    allRankTaskQueues_ = allRankTaskQueues;
    rootPath_ = rootPath;
}

const std::map<HccLTaskMetaType, const std::string> SqeuentialExecutor::taskNames_
    = {{HccLTaskMetaType::REDUCE, "reduce"},        {HccLTaskMetaType::MEM_CPY, "mem_cpy"},
       {HccLTaskMetaType::NOTIFY_RECORD, "record"}, {HccLTaskMetaType::CCU_GRAPH, "ccu_graph"},
       {HccLTaskMetaType::AIV_GRAPH, "aiv_graph"},  {HccLTaskMetaType::NOTIFY_WAIT, "wait"}};

HcclVmResult SqeuentialExecutor::Execute()
{
    auto rankSize = allRankTaskQueues_.size();
    auto& devResMgr = DeviceResourceManager::GetInstance();
    devResMgr.Init(rankSize);

    // ===== Trace 初始化 =====
    auto& traceCollector = CcuTrace::CcuTraceCollector::GetInstance();
    const char* envEnableTrace = std::getenv("HCCLVM_ENABLE_TRACE");
    if (envEnableTrace != nullptr && std::strcmp(envEnableTrace, "1") == 0) {
        HCCL_VM_INFO("Enable trace collection");
        traceCollector.SetEnabled(true);
        CcuTrace::CcuRunMetadata metadata;
        metadata.traceFormatVersion = 1;
        metadata.rankSize = static_cast<uint32_t>(rankSize);
        traceCollector.StartRun(metadata);

        // 设置输出路径（用于增量落盘和崩溃 dump）
        std::string outputPath = rootPath_ + "/hccl_trace_output.json";
        traceCollector.SetOutputPath(outputPath);

        // 注册信号处理函数：段错误/abort 时紧急 dump trace
        g_crashDumpPathStorage = rootPath_ + "/hccl_trace_crash_dump.json";
        g_crashDumpPath = g_crashDumpPathStorage.c_str();
        signal(SIGSEGV, CrashDumpHandler);
        signal(SIGABRT, CrashDumpHandler);
        HCCL_VM_INFO("Crash dump path: {}", g_crashDumpPathStorage);
    }

    const uint32_t flushInterval = GetFlushInterval();
    uint32_t instrCountSinceFlush = 0;

    uint32_t execRound = 0;
    while (HasTask()) {
        traceCollector.BeginRound(execRound++);
        uint32_t rankId = 0;
        for (auto& rankTasks : allRankTaskQueues_) { // rank
            devResMgr.InitRankRes(rankId++, rankTasks.size());
            for (auto& streamTasks : rankTasks) { // stream
                while (!streamTasks.empty()) {
                    auto task = streamTasks.front();
                    auto ret = ExecuteOneTask(task);
                    if (ret == HcclVmResult::HCCL_SIM_VRT_HOLD_CMD) {
                        break;
                    } else if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
                        HCCL_VM_ERROR(
                            "ExecuteOneTask failed, ret: {}, rankId = {}, type= {}", static_cast<int>(ret), rankId,
                            taskNames_.at(task.taskType));
                        // 异常退出前也尝试 dump 已采集的 trace
                        if (traceCollector.IsEnabled()) {
                            traceCollector.EndRun();
                            CcuTrace::CcuTraceSerializer::DumpToFile(
                                traceCollector.GetTraceRun(), g_crashDumpPathStorage);
                            HCCL_VM_INFO("Dumped trace on error exit");
                        }
                        return ret;
                    }
                    streamTasks.pop();

                    // 增量落盘：定期将已采集的 trace 数据写入磁盘
                    if (traceCollector.IsEnabled()) {
                        instrCountSinceFlush++;
                        if (instrCountSinceFlush >= flushInterval) {
                            traceCollector.IncrementalDump();
                            instrCountSinceFlush = 0;
                        }
                    }
                }
            }
        }
    }

    // ===== Trace 落盘（最终完整 dump） =====
    if (traceCollector.IsEnabled()) {
        traceCollector.EndRun();
        auto traceRun = traceCollector.GetTraceRun();
        std::string outputPath = rootPath_ + "/hccl_trace_output.json";
        CcuTrace::CcuTraceSerializer::DumpToFile(traceRun, outputPath);
        HCCL_VM_INFO("Dump trace to: {}", outputPath);

        // 清理崩溃 dump 文件（正常退出时不需要）
        std::remove(g_crashDumpPathStorage.c_str());
    }

    // 恢复默认信号处理
    signal(SIGSEGV, SIG_DFL);
    signal(SIGABRT, SIG_DFL);
    g_crashDumpPath = nullptr;

    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult SqeuentialExecutor::ExecuteOneTask(HcclTaskMetaData& task)
{
    switch (task.taskType) {
        case HccLTaskMetaType::REDUCE:
            return TaskReduce(task);
        case HccLTaskMetaType::MEM_CPY:
            return TaskMemcpy(task);
        case HccLTaskMetaType::NOTIFY_RECORD:
            return TaskNotifyRecord(task);
        case HccLTaskMetaType::NOTIFY_WAIT:
            return TaskNotifyWait(task);
        case HccLTaskMetaType::CCU_GRAPH:
            return TaskCcuGraph(task);
        case HccLTaskMetaType::AIV_GRAPH:
            return TaskAivGraph(task);
        default:
            break;
    }
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

bool SqeuentialExecutor::HasTask()
{
    for (auto& rankTasks : allRankTaskQueues_) {
        for (auto& streamTasks : rankTasks) {
            if (!streamTasks.empty()) {
                return true;
            }
        }
    }
    return false;
}
} // namespace VirtualRunTime
