/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "db_sim_runner_ops.h"

#include <cstdint>
#include <iostream>
#include <map>

#include "sim_log.h"
#include "sim_models.h"
#include "db_sim_runner_common.h"
#include "db_sim_runner_db.h"

uint32_t streamCnt = 0;
std::map<uint64_t, uint32_t> stream2checkerStream;

extern uint64_t g_cur_server_key;
namespace sim {
thread_local Runner g_runner;

thread_local uint64_t g_last_streamId;
thread_local uint64_t g_last_taskId;
thread_local int g_tsId = 0;

bool InsertRunner(uint64_t serverKey)
{
    try {
        auto curServer = RunnerDB::GetById<sim::Server>(serverKey);
        if (!curServer.has_value()) {
            HCCL_VM_ERROR("can not get server by key: {:d}", serverKey);
            return false;
        }

        auto ret = RunnerDB::GetOneByPred<sim::Host>([serverKey](const sim::Host& d) {
            return d.server_id == serverKey;
        });
        if (!ret.second) {
            HCCL_VM_ERROR("cannot find host by server key: {:d}", serverKey);
            return false;
        }
        auto hostKey = ret.first.id;

        // 插入runner
        sim::Runner runner{};
        runner.host_id = hostKey;
        runner.pid = getpid();
        runner.thread_id = pthread_self();

        // todo: runner未实际插入数据库，这里需要重新赋值
        g_runner.host_id = hostKey;
        g_runner.pid = getpid();
        g_runner.thread_id = pthread_self();

        g_cur_server_key = serverKey;
        HCCL_VM_INFO("Init host ip= {}, server key= {}.", ret.first.ip_addr, g_cur_server_key);

        g_runner.id = RunnerDB::Add<sim::Runner>(runner);
        return true;
    } catch (const std::exception& e) {
        HCCL_VM_ERROR("SQLite exception: {}", e.what());
        return false;
    }
}

bool GetCurrRunnerTls(uint64_t serverKey, Runner& runner)
{
    if (g_runner.id == 0 && serverKey == 0) {
        HCCL_VM_ERROR("can not get runner by server key: {:d}", serverKey);
        return false;
    }
    if (g_runner.id == 0) {
        InsertRunner(serverKey);
    }
    runner = g_runner;
    return true;
}

bool SetCurrCtxTls(uint64_t ctx)
{
    try {
        auto& currRunnerId = g_runner.id;
        g_runner.current_ctx_id = ctx;
        RunnerDB::Update<sim::Runner>(currRunnerId, [currRunnerId, ctx](sim::Runner& runner) {
            runner.current_ctx_id = ctx;
        });
        return true;
    } catch (const std::exception& e) {
        HCCL_VM_ERROR("SQLite exception: {}", e.what());
        return false;
    }
}

void SetLastStreamIdTls(uint64_t streamId) { g_last_streamId = streamId; }

void SetLastTaskIdTls(uint64_t taskId) { g_last_taskId = taskId; }

uint64_t GetLastStreamIdTls() { return g_last_streamId; }

uint64_t GetLastTaskIdTls() { return g_last_taskId; }

uint64_t GetCurrRankId()
{
    auto currCtx = RunnerDB::GetById<sim::Context>(g_runner.current_ctx_id);
    if (!currCtx.has_value()) {
        // not find
        HCCL_VM_ERROR("can not get CurrContext: {:d}", g_runner.current_ctx_id);
        return 0;
    }

    auto dev = RunnerDB::GetById<sim::Device>(currCtx->device_id);
    if (!dev.has_value()) {
        // not find
        HCCL_VM_ERROR("can not get device: {:d}", currCtx->device_id);
        return 0;
    }

    auto devKey = currCtx->device_id;
    auto rank = RunnerDB::GetOneByPred<sim::Rank>([devKey](const sim::Rank& r) {
        return r.device_id == devKey;
    });
    if (!rank.second) {
        HCCL_VM_ERROR("can not find any rank");
        return 0;
    }

    return rank.first.rank_id;
}

uint64_t GetCurrDeviceKey()
{
    auto currCtx = RunnerDB::GetById<sim::Context>(g_runner.current_ctx_id);
    if (!currCtx.has_value()) {
        // not find
        HCCL_VM_ERROR("can not get CurrContext: {:d}", g_runner.current_ctx_id);
        return 0;
    }

    return currCtx->device_id;
}

uint64_t GetRankIdByCtxId(uint64_t ctxId)
{
    auto currCtx = RunnerDB::GetById<sim::Context>(ctxId);
    if (!currCtx.has_value()) {
        // not find
        HCCL_VM_ERROR("can not get CurrContext: {:d}", ctxId);
        return 0;
    }

    auto dev = RunnerDB::GetById<sim::Device>(currCtx->device_id);
    if (!dev.has_value()) {
        // not find
        HCCL_VM_ERROR("can not get device: {:d}", currCtx->device_id);
        return 0;
    }

    return dev->logic_id;
}

void SetTsDevice(int tsId) { g_tsId = tsId; }

uint32_t GetRankSize()
{
    auto allRank = RunnerDB::GetByPred<sim::Rank>([](const sim::Rank& r) {
        return true;
    });
    return allRank.size();
}

uint32_t GetHostSize()
{
    auto allHost = RunnerDB::GetByPred<sim::Host>([](const sim::Host& r) {
        return true;
    });
    return allHost.size();
}

uint32_t GetCurrentStreamId(uint64_t streamKey)
{
    auto cs = stream2checkerStream.find(streamKey);
    if (cs != stream2checkerStream.end()) {
        return cs->second;
    } else {
        auto stremId = streamCnt++;
        stream2checkerStream[streamKey] = stremId;
        return stremId;
    }
}

uint64_t GetServerKeyById(uint32_t superPodIdx, uint32_t serverIdx)
{
    auto serverRet = RunnerDB::GetOneByPred<sim::Server>([superPodIdx, serverIdx](const sim::Server& r) {
        return r.pod_id == superPodIdx && r.server_id == serverIdx;
    });
    if (!serverRet.second) {
        HCCL_VM_ERROR("can not find server by id: {:d}, {:d}", superPodIdx, serverIdx);
        return 0;
    }
    return serverRet.first.id;
}
} // namespace sim
