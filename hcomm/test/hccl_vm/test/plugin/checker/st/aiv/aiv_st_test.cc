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
#include <cerrno>
#include <cstdlib>
#include <dirent.h>
#include <fstream>
#include <functional>
#include <gtest/gtest.h>
#include <map>
#include <memory>
#include <nlohmann_json/json.hpp>
#include <queue>
#include <set>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>
#include <vector>

#include "framework/task_graph_generator_v3/task_graph_generator_v3.h"
#include "framework/task_graph_generator_v3/task_graph_mem_conflict_v3.h"
#include "framework/task_graph_generator_v3/task_graph_single_task_check_v3.h"
// sim_common_defs.h and checker-internal sim_common.h both define these global
// types. This test only needs AivCommInfoLayout from sim_common_defs.h.
#define BufferType AivStLayoutBufferType
#define DataSlice AivStLayoutDataSlice
#define INPUT AivStLayoutInput
#define OUTPUT AivStLayoutOutput
#define CCL AivStLayoutCcl
#define RESERVED AivStLayoutReserved
#include "sim_common_defs.h"
#undef RESERVED
#undef CCL
#undef OUTPUT
#undef INPUT
#undef DataSlice
#undef BufferType
#include "storage_manager.h"

std::map<RankId, std::map<u32, HcclSim::ChannelsPerDie>> g_allRankChannelInfo;

namespace {
using Json = nlohmann::json;
namespace V3 = HcclSim::TaskGraphGeneratorV3;

constexpr uint32_t TASK_MEM_COPY = 0;
constexpr uint32_t TASK_REDUCE = 1;
constexpr uint32_t TASK_SET_FLAG = 2;
constexpr uint32_t TASK_WAIT_FLAG = 3;
constexpr uint32_t TASK_PIPE_BARRIER = 4;
constexpr uint32_t TASK_SYNC_ALL = 5;
constexpr uint32_t TASK_SEND_FLAG = 6;
constexpr uint32_t TASK_RECV_FLAG = 7;
constexpr uint32_t BUFFER_OUTPUT = 1;
constexpr uint32_t BUFFER_INPUT = 0;
constexpr uint32_t BUFFER_CCL = 2;
constexpr uint32_t BUFFER_UB = 3;
constexpr uint32_t BUFFER_AIV_COMM_INFO = 4;
constexpr uint32_t PIPE_MTE2 = 1;
constexpr uint32_t PIPE_MTE3 = 2;
constexpr uint32_t PIPE_ALL = 3;
constexpr uint64_t DEFAULT_BUFFER_SIZE = 1ULL << 20;
constexpr uint64_t AIV_COMM_INFO_SIZE = AivCommInfoLayout::SIZE_BYTES;
constexpr uint64_t AIV_COMM_INFO_FLAG1_OFFSET = AivCommInfoLayout::FLAG1_OFFSET;
constexpr uint64_t AIV_COMM_INFO_SYNC_CELL_BYTES = AivCommInfoLayout::SYNC_CELL_BYTES;

uint64_t AivCommInfoFlagOffset(uint64_t cellIndex)
{
    return AIV_COMM_INFO_FLAG1_OFFSET + cellIndex * AIV_COMM_INFO_SYNC_CELL_BYTES;
}

void EnsureDir(const std::string& path)
{
    if (mkdir(path.c_str(), 0700) != 0 && errno != EEXIST) {
        throw std::runtime_error("failed to create dir: " + path);
    }
}

void RemoveTree(const std::string& path)
{
    DIR* dir = opendir(path.c_str());
    if (dir == nullptr) {
        unlink(path.c_str());
        return;
    }
    while (dirent* entry = readdir(dir)) {
        const std::string name = entry->d_name;
        if (name == "." || name == "..") {
            continue;
        }
        const std::string child = path + "/" + name;
        struct stat st {};
        if (lstat(child.c_str(), &st) != 0) {
            continue;
        }
        if (S_ISDIR(st.st_mode)) {
            RemoveTree(child);
        } else {
            unlink(child.c_str());
        }
    }
    closedir(dir);
    rmdir(path.c_str());
}

class TempAivInstallDir {
public:
    explicit TempAivInstallDir(const std::string& caseName)
    {
        char tmpl[] = "/tmp/checkerl2_aiv_st_XXXXXX";
        char* created = mkdtemp(tmpl);
        if (created == nullptr) {
            throw std::runtime_error("mkdtemp failed");
        }
        root_ = std::string(created) + "_" + caseName;
        if (rename(created, root_.c_str()) != 0) {
            RemoveTree(created);
            throw std::runtime_error("failed to rename temp dir: " + root_);
        }
        dataDir_ = root_ + "/data";
        EnsureDir(dataDir_);

        const char* oldEnv = std::getenv("HCCL_VM_INSTALL_ROOT");
        if (oldEnv != nullptr) {
            hadOldEnv_ = true;
            oldEnv_ = oldEnv;
        }
        if (setenv("HCCL_VM_INSTALL_ROOT", root_.c_str(), 1) != 0) {
            throw std::runtime_error("failed to set HCCL_VM_INSTALL_ROOT");
        }
    }

    ~TempAivInstallDir()
    {
        if (hadOldEnv_) {
            setenv("HCCL_VM_INSTALL_ROOT", oldEnv_.c_str(), 1);
        } else {
            unsetenv("HCCL_VM_INSTALL_ROOT");
        }
        if (!root_.empty()) {
            RemoveTree(root_);
        }
    }

    std::string TaskJsonPath(uint32_t rank, uint64_t launch) const
    {
        return dataDir_ + "/hcclvm_aiv_rank" + std::to_string(rank) + "_launch" + std::to_string(launch) + "_task.json";
    }

    void ClearData()
    {
        RemoveTree(dataDir_);
        EnsureDir(dataDir_);
    }

private:
    std::string root_;
    std::string dataDir_;
    bool hadOldEnv_{false};
    std::string oldEnv_;
};

Json Slice(uint32_t bufferType, uint64_t offset, uint64_t size)
{
    return Json{{"bufferType", bufferType}, {"offset", offset}, {"size", size}};
}

Json RuntimeTask(
    uint32_t taskType, const std::string& name, uint32_t taskId, uint32_t rank, uint32_t block, uint32_t pipe,
    Json payload)
{
    return Json{{"taskType", taskType},
                {"taskTypeName", name},
                {"taskId", taskId},
                {"rankId", rank},
                {"blockId", block},
                {"curPipe", pipe},
                {"payload", std::move(payload)}};
}

Json MemCopy(
    uint32_t taskId, uint32_t rank, uint32_t block, uint32_t pipe, uint32_t srcRank, uint32_t dstRank, uint32_t srcType,
    uint64_t srcOffset, uint32_t dstType, uint64_t dstOffset, uint64_t size = 64)
{
    return RuntimeTask(
        TASK_MEM_COPY, "MemCopy", taskId, rank, block, pipe,
        Json{
            {"srcRank", srcRank},
            {"dstRank", dstRank},
            {"src", Slice(srcType, srcOffset, size)},
            {"dst", Slice(dstType, dstOffset, size)}});
}

Json Reduce(
    uint32_t taskId, uint32_t rank, uint32_t block, uint32_t pipe, uint32_t srcRank, uint32_t dstRank, uint32_t srcType,
    uint64_t srcOffset, uint32_t dstType, uint64_t dstOffset, uint64_t size = 64, uint32_t dataType = 0,
    uint32_t reduceOp = 0)
{
    return RuntimeTask(
        TASK_REDUCE, "Reduce", taskId, rank, block, pipe,
        Json{
            {"srcRank", srcRank},
            {"dstRank", dstRank},
            {"src", Slice(srcType, srcOffset, size)},
            {"dst", Slice(dstType, dstOffset, size)},
            {"dataType", dataType},
            {"reduceOp", reduceOp}});
}

Json SetFlag(
    uint32_t taskId, uint32_t rank, uint32_t block, uint32_t pipe, uint32_t srcPipe, uint32_t dstPipe, int32_t eventId)
{
    return RuntimeTask(
        TASK_SET_FLAG, "SetFlag", taskId, rank, block, pipe,
        Json{{"srcPipe", srcPipe}, {"dstPipe", dstPipe}, {"eventId", eventId}});
}

Json WaitFlag(
    uint32_t taskId, uint32_t rank, uint32_t block, uint32_t pipe, uint32_t srcPipe, uint32_t dstPipe, int32_t eventId)
{
    return RuntimeTask(
        TASK_WAIT_FLAG, "WaitFlag", taskId, rank, block, pipe,
        Json{{"srcPipe", srcPipe}, {"dstPipe", dstPipe}, {"eventId", eventId}});
}

Json NormalTask(uint32_t taskId, uint32_t rank, uint32_t block, uint32_t pipe)
{
    const uint64_t base = 0x20000ULL + static_cast<uint64_t>(rank) * 0x4000ULL + static_cast<uint64_t>(block) * 0x400ULL
                          + static_cast<uint64_t>(pipe) * 0x80ULL;
    return MemCopy(taskId, rank, block, pipe, rank, rank, BUFFER_UB, base + 0x2000ULL, BUFFER_UB, base, 32);
}

Json PipeBarrier(uint32_t taskId, uint32_t rank, uint32_t block, uint32_t pipe, const std::vector<uint32_t>& members)
{
    return RuntimeTask(
        TASK_PIPE_BARRIER, "PipeBarrier", taskId, rank, block, pipe,
        Json{{"pipeType", PIPE_ALL}, {"barrierGroupTaskIds", members}});
}

Json SyncAll(uint32_t taskId, uint32_t rank, uint32_t block, uint32_t pipe, uint32_t round)
{
    return RuntimeTask(TASK_SYNC_ALL, "SyncAll", taskId, rank, block, pipe, Json{{"syncRound", round}});
}

Json SendFlag(
    uint32_t taskId, uint32_t rank, uint32_t block, uint32_t pipe, uint32_t ownerRank, uint64_t offset, int32_t value)
{
    return RuntimeTask(
        TASK_SEND_FLAG, "SendFlag", taskId, rank, block, pipe,
        Json{{"rank", ownerRank}, {"commInfoOffset", offset}, {"flagValue", value}});
}

Json RecvFlag(
    uint32_t taskId, uint32_t rank, uint32_t block, uint32_t pipe, uint32_t ownerRank, uint64_t offset, int32_t value)
{
    return RuntimeTask(
        TASK_RECV_FLAG, "RecvFlag", taskId, rank, block, pipe,
        Json{{"rank", ownerRank}, {"commInfoOffset", offset}, {"targetValue", value}});
}

Json Block(uint32_t blockIdx, std::vector<Json> pipe0, std::vector<Json> pipe1, std::vector<Json> pipe2)
{
    return Json{
        {"blockIdx", blockIdx},
        {"scalarTasks", std::move(pipe0)},
        {"mte2Tasks", std::move(pipe1)},
        {"mte3Tasks", std::move(pipe2)}};
}

Json Snapshot(uint32_t rank, uint32_t rankSize, uint64_t launch, std::vector<Json> blocks)
{
    return Json{
        {"rank", rank},
        {"rankSize", rankSize},
        {"launchIndex", launch},
        {"inBufferSize", DEFAULT_BUFFER_SIZE},
        {"outBufferSize", DEFAULT_BUFFER_SIZE},
        {"cclBufferSize", DEFAULT_BUFFER_SIZE},
        {"aivCommInfoSize", AIV_COMM_INFO_SIZE},
        {"ubBufferSize", DEFAULT_BUFFER_SIZE},
        {"aivCores", std::move(blocks)}};
}

void WriteJson(const std::string& path, const Json& json)
{
    std::ofstream ofs(path);
    if (!ofs.is_open()) {
        throw std::runtime_error("failed to open " + path);
    }
    ofs << json.dump(2);
}

Json NormalSnapshot(uint32_t rank, uint32_t rankSize, uint64_t launch, uint32_t blockCount = 1)
{
    std::vector<Json> blocks;
    for (uint32_t block = 0; block < blockCount; ++block) {
        const uint32_t base = 1000U + block * 10U;
        blocks.push_back(Block(
            block, {NormalTask(base + 0U, rank, block, 0)}, {NormalTask(base + 1U, rank, block, 1)},
            {NormalTask(base + 2U, rank, block, 2)}));
    }
    return Snapshot(rank, rankSize, launch, std::move(blocks));
}

Json PipeBarrierSnapshot(uint32_t rank, uint32_t rankSize, uint64_t launch, uint32_t rounds = 1)
{
    std::vector<Json> pipe0;
    std::vector<Json> pipe1;
    std::vector<Json> pipe2;
    for (uint32_t round = 0; round < rounds; ++round) {
        const uint32_t base = 100U + round * 100U;
        pipe0.push_back(PipeBarrier(base + 0U, rank, 0, 0, {base + 1U, base + 2U}));
        pipe1.push_back(PipeBarrier(base + 1U, rank, 0, 1, {base + 0U, base + 2U}));
        pipe2.push_back(PipeBarrier(base + 2U, rank, 0, 2, {base + 0U, base + 1U}));
    }
    return Snapshot(rank, rankSize, launch, {Block(0, std::move(pipe0), std::move(pipe1), std::move(pipe2))});
}

Json PipeBarrierMissingMemberSnapshot(uint32_t rank, uint32_t rankSize, uint64_t launch)
{
    std::vector<Json> pipe0{PipeBarrier(100, rank, 0, 0, {101})};
    std::vector<Json> pipe1{PipeBarrier(101, rank, 0, 1, {100})};
    std::vector<Json> pipe2{NormalTask(102, rank, 0, 2)};
    return Snapshot(rank, rankSize, launch, {Block(0, std::move(pipe0), std::move(pipe1), std::move(pipe2))});
}

Json SyncAllSnapshot(uint32_t rank, uint32_t rankSize, uint64_t launch, uint32_t rounds = 1)
{
    std::vector<Json> blocks;
    for (uint32_t block = 0; block < 2; ++block) {
        std::vector<Json> pipe0;
        std::vector<Json> pipe1;
        std::vector<Json> pipe2;
        for (uint32_t round = 0; round < rounds; ++round) {
            const uint32_t base = 200U + round * 100U + block * 10U;
            pipe0.push_back(SyncAll(base + 0U, rank, block, 0, round));
            pipe1.push_back(SyncAll(base + 1U, rank, block, 1, round));
            pipe2.push_back(SyncAll(base + 2U, rank, block, 2, round));
        }
        blocks.push_back(Block(block, std::move(pipe0), std::move(pipe1), std::move(pipe2)));
    }
    return Snapshot(rank, rankSize, launch, std::move(blocks));
}

Json SyncAllMissingMemberSnapshot(uint32_t rank, uint32_t rankSize, uint64_t launch)
{
    std::vector<Json> blocks;
    blocks.push_back(
        Block(0, {SyncAll(200, rank, 0, 0, 0)}, {SyncAll(201, rank, 0, 1, 0)}, {NormalTask(202, rank, 0, 2)}));
    blocks.push_back(
        Block(1, {SyncAll(210, rank, 1, 0, 0)}, {SyncAll(211, rank, 1, 1, 0)}, {SyncAll(212, rank, 1, 2, 0)}));
    return Snapshot(rank, rankSize, launch, std::move(blocks));
}

Json CombinationSnapshot(uint32_t rank, uint32_t rankSize, uint64_t launch)
{
    std::vector<Json> blocks;
    blocks.push_back(Block(
        0, {PipeBarrier(100, rank, 0, 0, {101, 102}), SyncAll(200, rank, 0, 0, 0), NormalTask(300, rank, 0, 0)},
        {PipeBarrier(101, rank, 0, 1, {100, 102}), SyncAll(201, rank, 0, 1, 0), NormalTask(301, rank, 0, 1)},
        {PipeBarrier(102, rank, 0, 2, {100, 101}), SyncAll(202, rank, 0, 2, 0), NormalTask(302, rank, 0, 2)}));
    blocks.push_back(Block(
        1, {SyncAll(210, rank, 1, 0, 0), NormalTask(310, rank, 1, 0)},
        {SyncAll(211, rank, 1, 1, 0), NormalTask(311, rank, 1, 1)},
        {SyncAll(212, rank, 1, 2, 0), NormalTask(312, rank, 1, 2)}));
    return Snapshot(rank, rankSize, launch, std::move(blocks));
}

enum class UbMode {
    NORMAL,
    PARALLEL_CONFLICT,
    PARALLEL_NO_CONFLICT,
    ORDERED_OVERLAP,
};

Json UbSnapshot(uint32_t rank, uint32_t rankSize, uint64_t launch, UbMode mode)
{
    if (mode == UbMode::PARALLEL_CONFLICT) {
        return Snapshot(
            rank, rankSize, launch,
            {Block(
                0, {NormalTask(10, rank, 0, 0)},
                {MemCopy(20, rank, 0, 1, rank, rank, BUFFER_UB, 0x4000, BUFFER_UB, 0x1000)},
                {MemCopy(21, rank, 0, 2, rank, rank, BUFFER_UB, 0x5000, BUFFER_UB, 0x1000)})});
    }
    if (mode == UbMode::PARALLEL_NO_CONFLICT) {
        return Snapshot(
            rank, rankSize, launch,
            {Block(
                0, {NormalTask(10, rank, 0, 0)},
                {MemCopy(20, rank, 0, 1, rank, rank, BUFFER_UB, 0x4000, BUFFER_UB, 0x1000)},
                {MemCopy(21, rank, 0, 2, rank, rank, BUFFER_UB, 0x5000, BUFFER_UB, 0x1080)})});
    }
    if (mode == UbMode::ORDERED_OVERLAP) {
        return Snapshot(
            rank, rankSize, launch,
            {Block(
                0, {NormalTask(10, rank, 0, 0)},
                {MemCopy(20, rank, 0, 1, rank, rank, BUFFER_UB, 0x4000, BUFFER_UB, 0x1000),
                 MemCopy(21, rank, 0, 1, rank, rank, BUFFER_UB, 0x5000, BUFFER_UB, 0x1000)},
                {NormalTask(12, rank, 0, 2)})});
    }
    return NormalSnapshot(rank, rankSize, launch);
}

enum class CclMode {
    NORMAL,
    WRITER_LOW,
    WRITER_SAME_RANGE,
    WRITER_DISJOINT,
};

Json CclSnapshot(uint32_t rank, uint32_t rankSize, uint64_t launch, CclMode mode)
{
    if (mode == CclMode::WRITER_LOW) {
        return Snapshot(
            rank, rankSize, launch,
            {Block(
                0, {NormalTask(10, rank, 0, 0)},
                {MemCopy(20, rank, 0, 1, rank, 0, BUFFER_UB, 0x4000, BUFFER_CCL, 0x1000)},
                {NormalTask(12, rank, 0, 2)})});
    }
    if (mode == CclMode::WRITER_SAME_RANGE) {
        return Snapshot(
            rank, rankSize, launch,
            {Block(
                0, {NormalTask(10, rank, 0, 0)},
                {MemCopy(20, rank, 0, 1, rank, 0, BUFFER_UB, 0x5000, BUFFER_CCL, 0x1000)},
                {NormalTask(12, rank, 0, 2)})});
    }
    if (mode == CclMode::WRITER_DISJOINT) {
        return Snapshot(
            rank, rankSize, launch,
            {Block(
                0, {NormalTask(10, rank, 0, 0)},
                {MemCopy(20, rank, 0, 1, rank, 0, BUFFER_UB, 0x5000, BUFFER_CCL, 0x1080)},
                {NormalTask(12, rank, 0, 2)})});
    }
    return NormalSnapshot(rank, rankSize, launch);
}

Json SendRecvSnapshot(uint32_t rank, uint32_t rankSize, uint64_t launch)
{
    if (rank == 0) {
        return Snapshot(
            rank, rankSize, launch,
            {Block(
                0,
                {SendFlag(50, rank, 0, 0, 1, AivCommInfoFlagOffset(1), 1),
                 SendFlag(51, rank, 0, 0, 2, AivCommInfoFlagOffset(2), 1),
                 SendFlag(52, rank, 0, 0, 3, AivCommInfoFlagOffset(3), 1)},
                {NormalTask(11, rank, 0, 1)}, {NormalTask(12, rank, 0, 2)})});
    }
    return Snapshot(
        rank, rankSize, launch,
        {Block(
            0, {RecvFlag(50 + rank, rank, 0, 0, rank, AivCommInfoFlagOffset(rank), 1)}, {NormalTask(11, rank, 0, 1)},
            {NormalTask(12, rank, 0, 2)})});
}

Json CpGm2GMSnapshot(
    uint32_t rank, uint32_t rankSize, uint64_t launch, uint32_t iterationCount, bool reduceOut, bool flagMem = false,
    bool externalGap = false)
{
    std::vector<Json> mte2;
    std::vector<Json> mte3;
    const uint64_t sliceSize = 64;
    const uint32_t extSrcType = flagMem ? BUFFER_AIV_COMM_INFO : BUFFER_INPUT;
    const uint32_t extDstType = flagMem ? BUFFER_AIV_COMM_INFO : BUFFER_OUTPUT;
    const uint64_t externalOffsetBase = flagMem ? AIV_COMM_INFO_FLAG1_OFFSET : 0;
    for (uint32_t index = 0; index < iterationCount; ++index) {
        const uint32_t base = 100 + index * 6;
        const uint64_t offset
            = externalOffsetBase + static_cast<uint64_t>(index) * (sliceSize + (externalGap ? sliceSize : 0));
        mte2.push_back(MemCopy(base + 0, rank, 0, PIPE_MTE2, rank, rank, extSrcType, offset, BUFFER_UB, 0, sliceSize));
        mte2.push_back(SetFlag(base + 1, rank, 0, PIPE_MTE2, PIPE_MTE2, PIPE_MTE3, 0));
        mte2.push_back(WaitFlag(base + 5, rank, 0, PIPE_MTE2, PIPE_MTE3, PIPE_MTE2, 1));

        mte3.push_back(WaitFlag(base + 2, rank, 0, PIPE_MTE3, PIPE_MTE2, PIPE_MTE3, 0));
        if (reduceOut) {
            mte3.push_back(
                Reduce(base + 3, rank, 0, PIPE_MTE3, rank, rank, BUFFER_UB, 0, extDstType, offset, sliceSize));
        } else {
            mte3.push_back(
                MemCopy(base + 3, rank, 0, PIPE_MTE3, rank, rank, BUFFER_UB, 0, extDstType, offset, sliceSize));
        }
        mte3.push_back(SetFlag(base + 4, rank, 0, PIPE_MTE3, PIPE_MTE3, PIPE_MTE2, 1));
    }
    return Snapshot(rank, rankSize, launch, {Block(0, {}, std::move(mte2), std::move(mte3))});
}

Json CpGm2GMPipeBoundarySnapshot(uint32_t rank, uint32_t rankSize, uint64_t launch)
{
    std::vector<Json> mte2{NormalTask(90, rank, 0, PIPE_MTE2)};
    std::vector<Json> mte3{NormalTask(91, rank, 0, PIPE_MTE3)};
    const uint64_t sliceSize = 64;
    for (uint32_t index = 0; index < 2; ++index) {
        const uint32_t base = 100 + index * 6;
        const uint64_t offset = static_cast<uint64_t>(index) * sliceSize;
        mte2.push_back(
            MemCopy(base + 0, rank, 0, PIPE_MTE2, rank, rank, BUFFER_INPUT, offset, BUFFER_UB, 0, sliceSize));
        mte2.push_back(SetFlag(base + 1, rank, 0, PIPE_MTE2, PIPE_MTE2, PIPE_MTE3, 0));
        mte2.push_back(WaitFlag(base + 5, rank, 0, PIPE_MTE2, PIPE_MTE3, PIPE_MTE2, 1));

        mte3.push_back(WaitFlag(base + 2, rank, 0, PIPE_MTE3, PIPE_MTE2, PIPE_MTE3, 0));
        mte3.push_back(
            MemCopy(base + 3, rank, 0, PIPE_MTE3, rank, rank, BUFFER_UB, 0, BUFFER_OUTPUT, offset, sliceSize));
        mte3.push_back(SetFlag(base + 4, rank, 0, PIPE_MTE3, PIPE_MTE3, PIPE_MTE2, 1));
    }
    mte2.push_back(NormalTask(1000, rank, 0, PIPE_MTE2));
    mte3.push_back(NormalTask(1001, rank, 0, PIPE_MTE3));
    return Snapshot(rank, rankSize, launch, {Block(0, {}, std::move(mte2), std::move(mte3))});
}

struct CaseConfig {
    std::string name;
    uint32_t rankSize{2};
    std::vector<uint64_t> launches{0};
    std::function<Json(uint32_t, uint64_t)> snapshotFactory;
    std::set<std::pair<uint32_t, uint64_t>> missingSnapshots;
};

struct RunResult {
    HcclResult genRet{HCCL_SUCCESS};
    std::unique_ptr<V3::TaskGraphGeneratorV3> graph;
    HcclResult memRet{HCCL_SUCCESS};
    V3::MemConflictCheckStats memStats;
};

RunResult RunAivCase(const CaseConfig& config, bool runMemConflict = false)
{
    HcclSim::StorageManager::GetInstance().Reset();
    static auto install = std::make_unique<TempAivInstallDir>("all_cases");
    install->ClearData();
    for (uint32_t rank = 0; rank < config.rankSize; ++rank) {
        for (uint64_t launch : config.launches) {
            if (config.missingSnapshots.count({rank, launch}) != 0) {
                continue;
            }
            WriteJson(install->TaskJsonPath(rank, launch), config.snapshotFactory(rank, launch));
        }
    }

    std::vector<std::unique_ptr<V3::TaskNode>> nodes;
    V3::AllRankNodeQueues queues;
    for (uint32_t rank = 0; rank < config.rankSize; ++rank) {
        auto& rankQueues = queues[rank];
        rankQueues.resize(1);
        for (uint64_t launch : config.launches) {
            auto node = std::make_unique<V3::TaskAivGraph>(rank, launch, 0);
            V3::TaskPosition pos;
            pos.rankId = rank;
            pos.streamId = 0;
            pos.launchIdx = launch;
            node->SetPosition(pos);
            const V3::NodeId nodeId = static_cast<V3::NodeId>(nodes.size());
            node->SetNodeId(nodeId);
            rankQueues[0].push_back(nodeId);
            nodes.push_back(std::move(node));
        }
    }

    RunResult result;
    const HcclResult queueRet = V3::CheckSlaveTaskQueue(nodes, queues);
    if (queueRet != HCCL_SUCCESS) {
        result.genRet = queueRet;
        HcclSim::StorageManager::GetInstance().Reset();
        return result;
    }

    result.graph = std::make_unique<V3::TaskGraphGeneratorV3>();
    result.genRet = result.graph->GenGraph(std::move(nodes), std::move(queues));
    if (result.genRet == HCCL_SUCCESS && runMemConflict) {
        result.memRet = V3::CheckDataMoveTaskMemConflict(result.graph->GetMainStartNode(), &result.memStats);
    }
    HcclSim::StorageManager::GetInstance().Reset();
    return result;
}

size_t CountNodes(const V3::TaskGraphGeneratorV3& graph, V3::TaskType type)
{
    size_t count = 0;
    for (const auto& node : graph.GetNodes()) {
        if (node != nullptr && node->GetType() == type) {
            ++count;
        }
    }
    return count;
}

size_t CountMergedPipeBarrier(const V3::TaskGraphGeneratorV3& graph)
{
    size_t count = 0;
    for (const auto& node : graph.GetNodes()) {
        if (node == nullptr || node->GetType() != V3::TaskType::AIV_PIPE_BARRIER) {
            continue;
        }
        const auto* barrier = dynamic_cast<const V3::TaskAivPipeBarrier*>(node.get());
        if (barrier != nullptr && barrier->GetInfo().merged) {
            EXPECT_EQ(barrier->GetInfo().memberPipes.size(), 3U);
            ++count;
        }
    }
    return count;
}

size_t CountMergedSyncAll(const V3::TaskGraphGeneratorV3& graph, size_t expectedMembers)
{
    size_t count = 0;
    for (const auto& node : graph.GetNodes()) {
        if (node == nullptr || node->GetType() != V3::TaskType::AIV_SYNC_ALL) {
            continue;
        }
        const auto* syncAll = dynamic_cast<const V3::TaskAivSyncAll*>(node.get());
        if (syncAll != nullptr && syncAll->GetInfo().merged) {
            EXPECT_EQ(syncAll->GetInfo().memberNodeIds.size(), expectedMembers);
            ++count;
        }
    }
    return count;
}

size_t CountDirectEdges(const V3::TaskGraphGeneratorV3& graph, V3::TaskType parentType, V3::TaskType childType)
{
    size_t count = 0;
    for (const auto& node : graph.GetNodes()) {
        if (node == nullptr || node->GetType() != parentType) {
            continue;
        }
        for (const V3::TaskNode* child : node->GetChildren()) {
            if (child != nullptr && child->GetType() == childType) {
                ++count;
            }
        }
    }
    return count;
}

std::vector<const V3::TaskNode*> CollectNodesByType(const V3::TaskGraphGeneratorV3& graph, V3::TaskType type)
{
    std::vector<const V3::TaskNode*> result;
    for (const auto& node : graph.GetNodes()) {
        if (node != nullptr && node->GetType() == type) {
            result.push_back(node.get());
        }
    }
    return result;
}

const V3::TaskNode*
FindNodeByTaskId(const V3::TaskGraphGeneratorV3& graph, uint32_t rank, uint32_t pipe, uint32_t taskId)
{
    for (const auto& node : graph.GetNodes()) {
        if (node == nullptr) {
            continue;
        }
        const V3::TaskPosition& position = node->GetPosition();
        if (position.rankId == rank && position.launchIdx == 0 && position.blockId == 0 && position.pipe == pipe
            && position.taskId == taskId) {
            return node.get();
        }
    }
    return nullptr;
}

size_t CountLinkedNodesByTypeAndPipe(
    const std::vector<V3::TaskNode*>& nodes, V3::TaskType type, uint32_t pipe, uint32_t minTaskId)
{
    size_t count = 0;
    for (const V3::TaskNode* node : nodes) {
        if (node == nullptr) {
            continue;
        }
        const V3::TaskPosition& position = node->GetPosition();
        if (node->GetType() == type && position.pipe == pipe && position.taskId > minTaskId) {
            ++count;
        }
    }
    return count;
}

class AivStTest : public testing::Test {
protected:
    void SetUp() override
    {
        HcclSim::StorageManager::GetInstance().Reset();
        g_allRankChannelInfo.clear();
    }

    void TearDown() override
    {
        HcclSim::StorageManager::GetInstance().Reset();
        g_allRankChannelInfo.clear();
    }
};
} // namespace

TEST_F(AivStTest, AIV_ST_2P_001_PipeBarrierMergePositive)
{
    RunResult result = RunAivCase({"AIV_ST_2P_001", 2, {0}, [](uint32_t rank, uint64_t launch) {
                                       return PipeBarrierSnapshot(rank, 2, launch);
                                   }});
    ASSERT_EQ(result.genRet, HCCL_SUCCESS);
    EXPECT_EQ(result.graph->GetAivExpandStats().pipeBarrierMergeCount, 2U);
    EXPECT_EQ(CountMergedPipeBarrier(*result.graph), 2U);
}

TEST_F(AivStTest, AIV_ST_2P_002_PipeBarrierMissingMemberNegative)
{
    RunResult result = RunAivCase({"AIV_ST_2P_002", 2, {0}, [](uint32_t rank, uint64_t launch) {
                                       return rank == 0 ? PipeBarrierMissingMemberSnapshot(rank, 2, launch) :
                                                          PipeBarrierSnapshot(rank, 2, launch);
                                   }});
    EXPECT_NE(result.genRet, HCCL_SUCCESS);
}

TEST_F(AivStTest, AIV_ST_2P_003_SyncAllMergePositive)
{
    RunResult result = RunAivCase({"AIV_ST_2P_003", 2, {0}, [](uint32_t rank, uint64_t launch) {
                                       return SyncAllSnapshot(rank, 2, launch);
                                   }});
    ASSERT_EQ(result.genRet, HCCL_SUCCESS);
    EXPECT_EQ(result.graph->GetAivExpandStats().syncAllMergeCount, 2U);
    EXPECT_EQ(CountMergedSyncAll(*result.graph, 6U), 2U);
}

TEST_F(AivStTest, AIV_ST_2P_004_SyncAllMissingMemberNegative)
{
    RunResult result = RunAivCase({"AIV_ST_2P_004", 2, {0}, [](uint32_t rank, uint64_t launch) {
                                       return rank == 0 ? SyncAllMissingMemberSnapshot(rank, 2, launch) :
                                                          SyncAllSnapshot(rank, 2, launch);
                                   }});
    EXPECT_NE(result.genRet, HCCL_SUCCESS);
}

TEST_F(AivStTest, AIV_ST_2P_005_PipeBarrierTwoRoundMergePositive)
{
    RunResult result = RunAivCase({"AIV_ST_2P_005", 2, {0}, [](uint32_t rank, uint64_t launch) {
                                       return PipeBarrierSnapshot(rank, 2, launch, 2);
                                   }});
    ASSERT_EQ(result.genRet, HCCL_SUCCESS);
    EXPECT_EQ(result.graph->GetAivExpandStats().pipeBarrierMergeCount, 4U);
    EXPECT_EQ(CountMergedPipeBarrier(*result.graph), 4U);
}

TEST_F(AivStTest, AIV_ST_2P_006_SyncAllTwoRoundMergePositive)
{
    RunResult result = RunAivCase({"AIV_ST_2P_006", 2, {0}, [](uint32_t rank, uint64_t launch) {
                                       return SyncAllSnapshot(rank, 2, launch, 2);
                                   }});
    ASSERT_EQ(result.genRet, HCCL_SUCCESS);
    EXPECT_EQ(result.graph->GetAivExpandStats().syncAllMergeCount, 4U);
    EXPECT_EQ(CountMergedSyncAll(*result.graph, 6U), 4U);
}

TEST_F(AivStTest, AIV_ST_2P_007_MultiLaunchPositive)
{
    RunResult result = RunAivCase({"AIV_ST_2P_007", 2, {0, 1}, [](uint32_t rank, uint64_t launch) {
                                       return NormalSnapshot(rank, 2, launch);
                                   }});
    ASSERT_EQ(result.genRet, HCCL_SUCCESS);
    EXPECT_EQ(result.graph->GetAivExpandStats().graphCount, 4U);
}

TEST_F(AivStTest, AIV_ST_2P_008_MultiLaunchMissingFileNegative)
{
    CaseConfig config{
        "AIV_ST_2P_008",
        2,
        {0, 1},
        [](uint32_t rank, uint64_t launch) {
            return NormalSnapshot(rank, 2, launch);
        },
        {{1, 1}}};
    RunResult result = RunAivCase(config);
    EXPECT_NE(result.genRet, HCCL_SUCCESS);
}

TEST_F(AivStTest, AIV_ST_2P_009_UbParallelWriteConflict)
{
    RunResult result = RunAivCase(
        {"AIV_ST_2P_009",
         2,
         {0},
         [](uint32_t rank, uint64_t launch) {
             return UbSnapshot(rank, 2, launch, rank == 0 ? UbMode::PARALLEL_CONFLICT : UbMode::NORMAL);
         }},
        true);
    ASSERT_EQ(result.genRet, HCCL_SUCCESS);
    EXPECT_EQ(result.memRet, HCCL_E_MEMORY);
    EXPECT_GT(result.memStats.parallelCandidatePairCount, 0U);
}

TEST_F(AivStTest, AIV_ST_2P_010_UbParallelWriteNoConflict)
{
    RunResult result = RunAivCase(
        {"AIV_ST_2P_010",
         2,
         {0},
         [](uint32_t rank, uint64_t launch) {
             return UbSnapshot(rank, 2, launch, rank == 0 ? UbMode::PARALLEL_NO_CONFLICT : UbMode::NORMAL);
         }},
        true);
    ASSERT_EQ(result.genRet, HCCL_SUCCESS);
    EXPECT_EQ(result.memRet, HCCL_SUCCESS);
}

TEST_F(AivStTest, AIV_ST_2P_011_UbOrderedOverlapNoConflict)
{
    RunResult result = RunAivCase(
        {"AIV_ST_2P_011",
         2,
         {0},
         [](uint32_t rank, uint64_t launch) {
             return UbSnapshot(rank, 2, launch, rank == 0 ? UbMode::ORDERED_OVERLAP : UbMode::NORMAL);
         }},
        true);
    ASSERT_EQ(result.genRet, HCCL_SUCCESS);
    EXPECT_EQ(result.memRet, HCCL_SUCCESS);
    EXPECT_GT(result.memStats.orderedCandidatePairCount, 0U);
}

TEST_F(AivStTest, AIV_ST_4P_001_PipeBarrierMergePositive)
{
    RunResult result = RunAivCase({"AIV_ST_4P_001", 4, {0}, [](uint32_t rank, uint64_t launch) {
                                       return PipeBarrierSnapshot(rank, 4, launch);
                                   }});
    ASSERT_EQ(result.genRet, HCCL_SUCCESS);
    EXPECT_EQ(result.graph->GetAivExpandStats().pipeBarrierMergeCount, 4U);
    EXPECT_EQ(CountMergedPipeBarrier(*result.graph), 4U);
}

TEST_F(AivStTest, AIV_ST_4P_002_MultiLaunchPositive)
{
    RunResult result = RunAivCase({"AIV_ST_4P_002", 4, {0, 1}, [](uint32_t rank, uint64_t launch) {
                                       return NormalSnapshot(rank, 4, launch);
                                   }});
    ASSERT_EQ(result.genRet, HCCL_SUCCESS);
    EXPECT_EQ(result.graph->GetAivExpandStats().graphCount, 8U);
}

TEST_F(AivStTest, AIV_ST_4P_003_SendRecvFlagCrossRankPositive)
{
    RunResult result = RunAivCase({"AIV_ST_4P_003", 4, {0}, [](uint32_t rank, uint64_t launch) {
                                       return SendRecvSnapshot(rank, 4, launch);
                                   }});
    ASSERT_EQ(result.genRet, HCCL_SUCCESS);
    EXPECT_EQ(result.graph->GetAivExpandStats().sendRecvEdgeCount, 3U);
    EXPECT_EQ(CountDirectEdges(*result.graph, V3::TaskType::AIV_SEND_FLAG, V3::TaskType::AIV_RECV_FLAG), 3U);
}

TEST_F(AivStTest, AIV_ST_4P_004_UbParallelWriteConflict)
{
    RunResult result = RunAivCase(
        {"AIV_ST_4P_004",
         4,
         {0},
         [](uint32_t rank, uint64_t launch) {
             return UbSnapshot(rank, 4, launch, rank == 0 ? UbMode::PARALLEL_CONFLICT : UbMode::NORMAL);
         }},
        true);
    ASSERT_EQ(result.genRet, HCCL_SUCCESS);
    EXPECT_EQ(result.memRet, HCCL_E_MEMORY);
    EXPECT_GT(result.memStats.parallelCandidatePairCount, 0U);
}

TEST_F(AivStTest, AIV_ST_4P_005_UbParallelWriteNoConflict)
{
    RunResult result = RunAivCase(
        {"AIV_ST_4P_005",
         4,
         {0},
         [](uint32_t rank, uint64_t launch) {
             return UbSnapshot(rank, 4, launch, rank == 0 ? UbMode::PARALLEL_NO_CONFLICT : UbMode::NORMAL);
         }},
        true);
    ASSERT_EQ(result.genRet, HCCL_SUCCESS);
    EXPECT_EQ(result.memRet, HCCL_SUCCESS);
}

TEST_F(AivStTest, AIV_ST_4P_006_UbOrderedOverlapNoConflict)
{
    RunResult result = RunAivCase(
        {"AIV_ST_4P_006",
         4,
         {0},
         [](uint32_t rank, uint64_t launch) {
             return UbSnapshot(rank, 4, launch, rank == 0 ? UbMode::ORDERED_OVERLAP : UbMode::NORMAL);
         }},
        true);
    ASSERT_EQ(result.genRet, HCCL_SUCCESS);
    EXPECT_EQ(result.memRet, HCCL_SUCCESS);
    EXPECT_GT(result.memStats.orderedCandidatePairCount, 0U);
}

TEST_F(AivStTest, AIV_ST_4P_007_CombinationPositive)
{
    RunResult result = RunAivCase(
        {"AIV_ST_4P_007",
         4,
         {0, 1},
         [](uint32_t rank, uint64_t launch) {
             return CombinationSnapshot(rank, 4, launch);
         }},
        true);
    ASSERT_EQ(result.genRet, HCCL_SUCCESS);
    EXPECT_EQ(result.memRet, HCCL_SUCCESS);
    EXPECT_EQ(result.graph->GetAivExpandStats().graphCount, 8U);
    EXPECT_EQ(result.graph->GetAivExpandStats().pipeBarrierMergeCount, 8U);
    EXPECT_EQ(result.graph->GetAivExpandStats().syncAllMergeCount, 8U);
}

TEST_F(AivStTest, AIV_ST_2P_012_CclCrossRankParallelWriteConflict)
{
    RunResult result = RunAivCase(
        {"AIV_ST_2P_012",
         2,
         {0},
         [](uint32_t rank, uint64_t launch) {
             return CclSnapshot(rank, 2, launch, rank == 0 ? CclMode::WRITER_LOW : CclMode::WRITER_SAME_RANGE);
         }},
        true);
    ASSERT_EQ(result.genRet, HCCL_SUCCESS);
    EXPECT_EQ(result.memRet, HCCL_E_MEMORY);
    EXPECT_GT(result.memStats.parallelCandidatePairCount, 0U);
}

TEST_F(AivStTest, AIV_ST_2P_013_CclCrossRankParallelWriteNoConflict)
{
    RunResult result = RunAivCase(
        {"AIV_ST_2P_013",
         2,
         {0},
         [](uint32_t rank, uint64_t launch) {
             return CclSnapshot(rank, 2, launch, rank == 0 ? CclMode::WRITER_LOW : CclMode::WRITER_DISJOINT);
         }},
        true);
    ASSERT_EQ(result.genRet, HCCL_SUCCESS);
    EXPECT_EQ(result.memRet, HCCL_SUCCESS);
}

TEST_F(AivStTest, AIV_ST_4P_008_CclCrossRankParallelWriteConflict)
{
    RunResult result = RunAivCase(
        {"AIV_ST_4P_008",
         4,
         {0},
         [](uint32_t rank, uint64_t launch) {
             if (rank == 0) {
                 return CclSnapshot(rank, 4, launch, CclMode::WRITER_LOW);
             }
             if (rank == 1) {
                 return CclSnapshot(rank, 4, launch, CclMode::WRITER_SAME_RANGE);
             }
             return CclSnapshot(rank, 4, launch, CclMode::NORMAL);
         }},
        true);
    ASSERT_EQ(result.genRet, HCCL_SUCCESS);
    EXPECT_EQ(result.memRet, HCCL_E_MEMORY);
    EXPECT_GT(result.memStats.parallelCandidatePairCount, 0U);
}

TEST_F(AivStTest, AIV_ST_4P_009_CclCrossRankParallelWriteNoConflict)
{
    RunResult result = RunAivCase(
        {"AIV_ST_4P_009",
         4,
         {0},
         [](uint32_t rank, uint64_t launch) {
             if (rank == 0) {
                 return CclSnapshot(rank, 4, launch, CclMode::WRITER_LOW);
             }
             if (rank == 1) {
                 return CclSnapshot(rank, 4, launch, CclMode::WRITER_DISJOINT);
             }
             return CclSnapshot(rank, 4, launch, CclMode::NORMAL);
         }},
        true);
    ASSERT_EQ(result.genRet, HCCL_SUCCESS);
    EXPECT_EQ(result.memRet, HCCL_SUCCESS);
}

TEST_F(AivStTest, AIV_ST_2P_014_CpGm2GMMemCopyLoopMergePositive)
{
    RunResult result = RunAivCase(
        {"AIV_ST_2P_014",
         2,
         {0},
         [](uint32_t rank, uint64_t launch) {
             return CpGm2GMSnapshot(rank, 2, launch, 3, false);
         }},
        true);
    ASSERT_EQ(result.genRet, HCCL_SUCCESS);
    EXPECT_EQ(result.memRet, HCCL_SUCCESS);
    EXPECT_EQ(result.graph->GetAivExpandStats().cpGmLoopMergeCount, 2U);
    EXPECT_EQ(result.graph->GetAivExpandStats().cpGmMergedIterationCount, 6U);
    EXPECT_EQ(result.graph->GetAivExpandStats().cpGmMergedOriginalNodeCount, 36U);
    EXPECT_EQ(result.graph->GetAivExpandStats().cpGmGeneratedNodeCount, 12U);
    EXPECT_LT(
        result.graph->GetAivExpandStats().dagNodeCountAfterCpGmMerge,
        result.graph->GetAivExpandStats().dagNodeCountBeforeCpGmMerge);
    EXPECT_EQ(CollectNodesByType(*result.graph, V3::TaskType::BATCH_TRANS_MEM).size(), 4U);
}

TEST_F(AivStTest, AIV_ST_2P_015_CpGm2GMReduceLoopMergePositive)
{
    RunResult result = RunAivCase(
        {"AIV_ST_2P_015",
         2,
         {0},
         [](uint32_t rank, uint64_t launch) {
             return CpGm2GMSnapshot(rank, 2, launch, 2, true);
         }},
        true);
    ASSERT_EQ(result.genRet, HCCL_SUCCESS);
    EXPECT_EQ(result.memRet, HCCL_SUCCESS);
    EXPECT_EQ(result.graph->GetAivExpandStats().cpGmLoopMergeCount, 2U);
    EXPECT_EQ(result.graph->GetAivExpandStats().cpGmMergedIterationCount, 4U);
    EXPECT_EQ(CollectNodesByType(*result.graph, V3::TaskType::BATCH_TRANS_MEM).size(), 2U);
    EXPECT_EQ(CollectNodesByType(*result.graph, V3::TaskType::BATCH_REDUCE).size(), 2U);
}

TEST_F(AivStTest, AIV_ST_2P_016_CpGm2GMSingleIterationNoMerge)
{
    RunResult result = RunAivCase({"AIV_ST_2P_016", 2, {0}, [](uint32_t rank, uint64_t launch) {
                                       return CpGm2GMSnapshot(rank, 2, launch, 1, false);
                                   }});
    ASSERT_EQ(result.genRet, HCCL_SUCCESS);
    EXPECT_EQ(result.graph->GetAivExpandStats().cpGmLoopMergeCount, 0U);
    EXPECT_EQ(CollectNodesByType(*result.graph, V3::TaskType::BATCH_TRANS_MEM).size(), 0U);
}

TEST_F(AivStTest, AIV_ST_2P_017_CpGm2GMFlagMemLoopMergePositive)
{
    RunResult result = RunAivCase({"AIV_ST_2P_017", 2, {0}, [](uint32_t rank, uint64_t launch) {
                                       return CpGm2GMSnapshot(rank, 2, launch, 3, false, true);
                                   }});
    ASSERT_EQ(result.genRet, HCCL_SUCCESS);
    EXPECT_EQ(result.graph->GetAivExpandStats().cpGmLoopMergeCount, 2U);
    EXPECT_EQ(result.graph->GetAivExpandStats().cpGmMergedIterationCount, 6U);
    EXPECT_EQ(CollectNodesByType(*result.graph, V3::TaskType::BATCH_TRANS_MEM).size(), 4U);
}

TEST_F(AivStTest, AIV_ST_2P_018_CpGm2GMExternalGapNoMerge)
{
    RunResult result = RunAivCase({"AIV_ST_2P_018", 2, {0}, [](uint32_t rank, uint64_t launch) {
                                       return CpGm2GMSnapshot(rank, 2, launch, 2, false, false, true);
                                   }});
    ASSERT_EQ(result.genRet, HCCL_SUCCESS);
    EXPECT_EQ(result.graph->GetAivExpandStats().cpGmLoopMergeCount, 0U);
    EXPECT_EQ(CollectNodesByType(*result.graph, V3::TaskType::BATCH_TRANS_MEM).size(), 0U);
}

TEST_F(AivStTest, AIV_ST_2P_019_CpGm2GMMte3BoundaryKeepPipe)
{
    RunResult result = RunAivCase({"AIV_ST_2P_019", 2, {0}, [](uint32_t rank, uint64_t launch) {
                                       return CpGm2GMPipeBoundarySnapshot(rank, 2, launch);
                                   }});
    ASSERT_EQ(result.genRet, HCCL_SUCCESS);
    EXPECT_EQ(result.graph->GetAivExpandStats().cpGmLoopMergeCount, 2U);

    constexpr uint32_t minSyntheticTaskId = 1001;
    for (uint32_t rank = 0; rank < 2; ++rank) {
        const V3::TaskNode* mte3Before = FindNodeByTaskId(*result.graph, rank, PIPE_MTE3, 91);
        const V3::TaskNode* mte3After = FindNodeByTaskId(*result.graph, rank, PIPE_MTE3, 1001);
        ASSERT_NE(mte3Before, nullptr);
        ASSERT_NE(mte3After, nullptr);
        EXPECT_EQ(
            CountLinkedNodesByTypeAndPipe(
                mte3Before->GetChildren(), V3::TaskType::AIV_WAIT_FLAG, PIPE_MTE3, minSyntheticTaskId),
            1U);
        EXPECT_EQ(
            CountLinkedNodesByTypeAndPipe(
                mte3Before->GetChildren(), V3::TaskType::BATCH_TRANS_MEM, PIPE_MTE2, minSyntheticTaskId),
            0U);
        EXPECT_EQ(
            CountLinkedNodesByTypeAndPipe(
                mte3After->GetParents(), V3::TaskType::AIV_SET_FLAG, PIPE_MTE3, minSyntheticTaskId),
            1U);
        EXPECT_EQ(
            CountLinkedNodesByTypeAndPipe(
                mte3After->GetParents(), V3::TaskType::AIV_WAIT_FLAG, PIPE_MTE2, minSyntheticTaskId),
            0U);
    }
}
