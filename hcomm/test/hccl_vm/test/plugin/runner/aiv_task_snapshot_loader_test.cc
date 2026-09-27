/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <nlohmann_json/json.hpp>

#include "aiv_task_snapshot_loader.h"

namespace fs = std::filesystem;
using json = nlohmann::json;

class AivTaskSnapshotLoaderTest : public testing::Test {
protected:
    void SetUp() override
    {
        testDir_ = fs::temp_directory_path() / ("aiv_test_" + std::to_string(::getpid()));
        fs::create_directories(testDir_ / "data");
        setenv("HCCL_VM_INSTALL_ROOT", testDir_.c_str(), 1);
    }

    void TearDown() override
    {
        unsetenv("HCCL_VM_INSTALL_ROOT");
        fs::remove_all(testDir_);
    }

    std::string WriteTaskFile(uint32_t rankId, uint32_t launchIndex, const json& content)
    {
        std::string fileName
            = "hcclvm_aiv_rank" + std::to_string(rankId) + "_launch" + std::to_string(launchIndex) + "_task.json";
        fs::path filePath = testDir_ / "data" / fileName;
        std::ofstream ofs(filePath);
        ofs << content.dump();
        return filePath.string();
    }

    static json MakeDataSlice(uint32_t bufferType, uint64_t offset, uint64_t size)
    {
        return {{"bufferType", bufferType}, {"offset", offset}, {"size", size}};
    }

    static json MakeMemCopyPayload(uint32_t srcRank, uint32_t dstRank, const json& src, const json& dst)
    {
        return {{"srcRank", srcRank}, {"dstRank", dstRank}, {"src", src}, {"dst", dst}};
    }

    static json MakeReducePayload(
        uint32_t srcRank, uint32_t dstRank, const json& src, const json& dst, uint32_t dataType = 0,
        uint32_t reduceOp = 0)
    {
        return {{"srcRank", srcRank}, {"dstRank", dstRank},   {"src", src},
                {"dst", dst},         {"dataType", dataType}, {"reduceOp", reduceOp}};
    }

    static json MakeSetFlagPayload(uint32_t srcPipe, uint32_t dstPipe, int32_t eventId)
    {
        return {{"srcPipe", srcPipe}, {"dstPipe", dstPipe}, {"eventId", eventId}};
    }

    static json MakeWaitFlagPayload(uint32_t srcPipe, uint32_t dstPipe, int32_t eventId)
    {
        return {{"srcPipe", srcPipe}, {"dstPipe", dstPipe}, {"eventId", eventId}};
    }

    static json MakePipeBarrierPayload(uint32_t pipeType, const std::vector<uint32_t>& barrierGroupTaskIds)
    {
        return {{"pipeType", pipeType}, {"barrierGroupTaskIds", barrierGroupTaskIds}};
    }

    static json MakeSendFlagPayload(uint32_t rank, uint64_t commInfoOffset, int32_t flagValue)
    {
        return {{"rank", rank}, {"commInfoOffset", commInfoOffset}, {"flagValue", flagValue}};
    }

    static json MakeRecvFlagPayload(uint32_t rank, uint64_t commInfoOffset, int32_t targetValue)
    {
        return {{"rank", rank}, {"commInfoOffset", commInfoOffset}, {"targetValue", targetValue}};
    }

    static json MakeTaskJson(
        uint32_t taskType, uint32_t taskId, uint32_t rankId, uint32_t blockId, uint32_t curPipe, const json& payload)
    {
        return {{"taskType", taskType}, {"taskId", taskId},   {"rankId", rankId},
                {"blockId", blockId},   {"curPipe", curPipe}, {"payload", payload}};
    }

    static json MakeBlockJson(
        uint32_t blockIdx, const json& scalarTasks = json::array(), const json& mte2Tasks = json::array(),
        const json& mte3Tasks = json::array())
    {
        return {
            {"blockIdx", blockIdx}, {"scalarTasks", scalarTasks}, {"mte2Tasks", mte2Tasks}, {"mte3Tasks", mte3Tasks}};
    }

    fs::path testDir_;
};

// ==================== IsAivExpansionModeEnabled Tests ====================

TEST_F(AivTaskSnapshotLoaderTest, IsAivExpansionModeEnabled_EnvNotSet_ReturnsFalse)
{
    unsetenv("HCCL_OP_EXPANSION_MODE");
    EXPECT_FALSE(IsAivExpansionModeEnabled());
}

TEST_F(AivTaskSnapshotLoaderTest, IsAivExpansionModeEnabled_EnvAiv_ReturnsTrue)
{
    setenv("HCCL_OP_EXPANSION_MODE", "AIV", 1);
    EXPECT_TRUE(IsAivExpansionModeEnabled());
    unsetenv("HCCL_OP_EXPANSION_MODE");
}

TEST_F(AivTaskSnapshotLoaderTest, IsAivExpansionModeEnabled_EnvOther_ReturnsFalse)
{
    setenv("HCCL_OP_EXPANSION_MODE", "OTHER", 1);
    EXPECT_FALSE(IsAivExpansionModeEnabled());
    unsetenv("HCCL_OP_EXPANSION_MODE");
}

TEST_F(AivTaskSnapshotLoaderTest, IsAivExpansionModeEnabled_EnvEmpty_ReturnsFalse)
{
    setenv("HCCL_OP_EXPANSION_MODE", "", 1);
    EXPECT_FALSE(IsAivExpansionModeEnabled());
    unsetenv("HCCL_OP_EXPANSION_MODE");
}

// ==================== LoadRuntimeTaskSnapshotByLaunchDirect Success Tests ====================

TEST_F(AivTaskSnapshotLoaderTest, LoadByLaunchDirect_MemCopy_Success)
{
    auto dsS = MakeDataSlice(0, 0, 128);
    auto dsD = MakeDataSlice(1, 0, 128);
    auto payload = MakeMemCopyPayload(0, 1, dsS, dsD);
    auto task = MakeTaskJson(0, 1, 0, 0, 0, payload);
    auto block = MakeBlockJson(0, json::array({task}), json::array(), json::array());
    json content = {{"rank", 0}, {"rankSize", 4}, {"launchIndex", 0}, {"aivCores", json::array({block})}};
    WriteTaskFile(0, 0, content);

    AivRuntimeTaskSnapshot snapshot;
    std::string errorMsg;
    EXPECT_TRUE(AivTaskSnapshotLoader::LoadRuntimeTaskSnapshotByLaunchDirect(0, 0, snapshot, &errorMsg));
    EXPECT_EQ(snapshot.rankId, 0u);
    EXPECT_EQ(snapshot.rankSize, 4u);
    EXPECT_EQ(snapshot.launchIndex, 0u);
    ASSERT_EQ(snapshot.blocks.size(), 1u);
    ASSERT_EQ(snapshot.blocks[0].scalarTasks.size(), 1u);
    EXPECT_EQ(snapshot.blocks[0].scalarTasks[0]->GetTaskType(), AivSim::AivTaskType::MEM_COPY);
}

TEST_F(AivTaskSnapshotLoaderTest, LoadByLaunchDirect_Reduce_Success)
{
    auto dsS = MakeDataSlice(0, 0, 256);
    auto dsD = MakeDataSlice(1, 0, 256);
    auto payload = MakeReducePayload(0, 2, dsS, dsD, 1, 2);
    auto task = MakeTaskJson(1, 2, 1, 1, 1, payload);
    auto block = MakeBlockJson(0, json::array(), json::array({task}), json::array());
    json content = {{"rank", 1}, {"rankSize", 2}, {"launchIndex", 1}, {"aivCores", json::array({block})}};
    WriteTaskFile(1, 1, content);

    AivRuntimeTaskSnapshot snapshot;
    std::string errorMsg;
    EXPECT_TRUE(AivTaskSnapshotLoader::LoadRuntimeTaskSnapshotByLaunchDirect(1, 1, snapshot, &errorMsg));
    ASSERT_EQ(snapshot.blocks.size(), 1u);
    ASSERT_EQ(snapshot.blocks[0].mte2Tasks.size(), 1u);
    EXPECT_EQ(snapshot.blocks[0].mte2Tasks[0]->GetTaskType(), AivSim::AivTaskType::REDUCE);
}

TEST_F(AivTaskSnapshotLoaderTest, LoadByLaunchDirect_SetFlag_Success)
{
    auto payload = MakeSetFlagPayload(0, 1, 42);
    auto task = MakeTaskJson(2, 3, 0, 0, 0, payload);
    auto block = MakeBlockJson(0, json::array({task}), json::array(), json::array());
    json content = {{"rank", 0}, {"rankSize", 1}, {"launchIndex", 0}, {"aivCores", json::array({block})}};
    WriteTaskFile(0, 0, content);

    AivRuntimeTaskSnapshot snapshot;
    EXPECT_TRUE(AivTaskSnapshotLoader::LoadRuntimeTaskSnapshotByLaunchDirect(0, 0, snapshot));
    ASSERT_EQ(snapshot.blocks[0].scalarTasks.size(), 1u);
    EXPECT_EQ(snapshot.blocks[0].scalarTasks[0]->GetTaskType(), AivSim::AivTaskType::SET_FLAG);
}

TEST_F(AivTaskSnapshotLoaderTest, LoadByLaunchDirect_WaitFlag_Success)
{
    auto payload = MakeWaitFlagPayload(1, 2, 99);
    auto task = MakeTaskJson(3, 4, 0, 0, 0, payload);
    auto block = MakeBlockJson(0, json::array({task}));
    json content = {{"rank", 0}, {"rankSize", 1}, {"launchIndex", 0}, {"aivCores", json::array({block})}};
    WriteTaskFile(0, 0, content);

    AivRuntimeTaskSnapshot snapshot;
    EXPECT_TRUE(AivTaskSnapshotLoader::LoadRuntimeTaskSnapshotByLaunchDirect(0, 0, snapshot));
    ASSERT_EQ(snapshot.blocks[0].scalarTasks.size(), 1u);
    EXPECT_EQ(snapshot.blocks[0].scalarTasks[0]->GetTaskType(), AivSim::AivTaskType::WAIT_FLAG);
}

TEST_F(AivTaskSnapshotLoaderTest, LoadByLaunchDirect_PipeBarrier_Success)
{
    auto payload = MakePipeBarrierPayload(0, {5});
    auto task = MakeTaskJson(4, 5, 0, 0, 0, payload);
    auto block = MakeBlockJson(0, json::array({task}));
    json content = {{"rank", 0}, {"rankSize", 1}, {"launchIndex", 0}, {"aivCores", json::array({block})}};
    WriteTaskFile(0, 0, content);

    AivRuntimeTaskSnapshot snapshot;
    EXPECT_TRUE(AivTaskSnapshotLoader::LoadRuntimeTaskSnapshotByLaunchDirect(0, 0, snapshot));
    ASSERT_EQ(snapshot.blocks[0].scalarTasks.size(), 1u);
    EXPECT_EQ(snapshot.blocks[0].scalarTasks[0]->GetTaskType(), AivSim::AivTaskType::PIPE_BARRIER);
}

TEST_F(AivTaskSnapshotLoaderTest, LoadByLaunchDirect_SendFlag_Success)
{
    auto payload = MakeSendFlagPayload(3, 0x1000, 1);
    auto task = MakeTaskJson(6, 6, 1, 0, 0, payload);
    auto block = MakeBlockJson(0, json::array({task}));
    json content = {{"rank", 1}, {"rankSize", 4}, {"launchIndex", 0}, {"aivCores", json::array({block})}};
    WriteTaskFile(1, 0, content);

    AivRuntimeTaskSnapshot snapshot;
    EXPECT_TRUE(AivTaskSnapshotLoader::LoadRuntimeTaskSnapshotByLaunchDirect(1, 0, snapshot));
    ASSERT_EQ(snapshot.blocks[0].scalarTasks.size(), 1u);
    EXPECT_EQ(snapshot.blocks[0].scalarTasks[0]->GetTaskType(), AivSim::AivTaskType::SEND_FLAG);
}

TEST_F(AivTaskSnapshotLoaderTest, LoadByLaunchDirect_RecvFlag_Success)
{
    auto payload = MakeRecvFlagPayload(2, 0x2000, 1);
    auto task = MakeTaskJson(7, 7, 0, 1, 0, payload);
    auto block = MakeBlockJson(1, json::array(), json::array(), json::array({task}));
    json content = {{"rank", 0}, {"rankSize", 3}, {"launchIndex", 2}, {"aivCores", json::array({block})}};
    WriteTaskFile(0, 2, content);

    AivRuntimeTaskSnapshot snapshot;
    EXPECT_TRUE(AivTaskSnapshotLoader::LoadRuntimeTaskSnapshotByLaunchDirect(0, 2, snapshot));
    ASSERT_EQ(snapshot.blocks[0].mte3Tasks.size(), 1u);
    EXPECT_EQ(snapshot.blocks[0].mte3Tasks[0]->GetTaskType(), AivSim::AivTaskType::RECV_FLAG);
}

TEST_F(AivTaskSnapshotLoaderTest, LoadByLaunchDirect_AllTaskTypes_Success)
{
    auto ds = MakeDataSlice(0, 0, 64);
    auto memCopy = MakeTaskJson(0, 1, 0, 0, 0, MakeMemCopyPayload(0, 1, ds, ds));
    auto reduce = MakeTaskJson(1, 2, 0, 0, 1, MakeReducePayload(0, 1, ds, ds));
    auto setFlag = MakeTaskJson(2, 3, 0, 0, 0, MakeSetFlagPayload(0, 1, 0));
    auto waitFlag = MakeTaskJson(3, 4, 0, 0, 0, MakeWaitFlagPayload(0, 1, 0));
    auto barrier = MakeTaskJson(4, 5, 0, 0, 0, MakePipeBarrierPayload(0, {5}));
    auto sendFlag = MakeTaskJson(6, 6, 0, 0, 0, MakeSendFlagPayload(1, 0, 0));
    auto recvFlag = MakeTaskJson(7, 7, 0, 0, 0, MakeRecvFlagPayload(2, 0, 0));
    auto block = MakeBlockJson(
        0, json::array({memCopy, setFlag, waitFlag, barrier, sendFlag, recvFlag}), json::array({reduce}),
        json::array());
    json content = {{"rank", 0}, {"rankSize", 8}, {"launchIndex", 0}, {"aivCores", json::array({block})}};
    WriteTaskFile(0, 0, content);

    AivRuntimeTaskSnapshot snapshot;
    EXPECT_TRUE(AivTaskSnapshotLoader::LoadRuntimeTaskSnapshotByLaunchDirect(0, 0, snapshot));
    EXPECT_EQ(snapshot.rankSize, 8u);
    ASSERT_EQ(snapshot.blocks.size(), 1u);
    EXPECT_EQ(snapshot.blocks[0].scalarTasks.size(), 6u);
    EXPECT_EQ(snapshot.blocks[0].mte2Tasks.size(), 1u);
    EXPECT_EQ(snapshot.blocks[0].mte3Tasks.size(), 0u);
}

TEST_F(AivTaskSnapshotLoaderTest, LoadByLaunchDirect_MultiBlock_Success)
{
    auto task = MakeTaskJson(0, 1, 0, 0, 0, MakeMemCopyPayload(0, 1, MakeDataSlice(0, 0, 8), MakeDataSlice(1, 0, 8)));
    auto block0 = MakeBlockJson(0, json::array({task}), json::array(), json::array());
    auto block1 = MakeBlockJson(1, json::array(), json::array({task}), json::array());
    auto block2 = MakeBlockJson(2, json::array(), json::array(), json::array({task}));
    json content
        = {{"rank", 0}, {"rankSize", 1}, {"launchIndex", 0}, {"aivCores", json::array({block0, block1, block2})}};
    WriteTaskFile(0, 0, content);

    AivRuntimeTaskSnapshot snapshot;
    EXPECT_TRUE(AivTaskSnapshotLoader::LoadRuntimeTaskSnapshotByLaunchDirect(0, 0, snapshot));
    ASSERT_EQ(snapshot.blocks.size(), 3u);
    EXPECT_EQ(snapshot.blocks[0].blockIdx, 0u);
    EXPECT_EQ(snapshot.blocks[1].blockIdx, 1u);
    EXPECT_EQ(snapshot.blocks[2].blockIdx, 2u);
}

TEST_F(AivTaskSnapshotLoaderTest, LoadByLaunchDirect_BarrierGroup_Success)
{
    auto barrier1 = MakeTaskJson(4, 10, 0, 0, 0, MakePipeBarrierPayload(0, {11}));
    auto barrier2 = MakeTaskJson(4, 11, 0, 0, 0, MakePipeBarrierPayload(0, {10}));
    auto block = MakeBlockJson(0, json::array({barrier1, barrier2}));
    json content = {{"rank", 0}, {"rankSize", 1}, {"launchIndex", 0}, {"aivCores", json::array({block})}};
    WriteTaskFile(0, 0, content);

    AivRuntimeTaskSnapshot snapshot;
    EXPECT_TRUE(AivTaskSnapshotLoader::LoadRuntimeTaskSnapshotByLaunchDirect(0, 0, snapshot));
    ASSERT_EQ(snapshot.blocks[0].scalarTasks.size(), 2u);
}

TEST_F(AivTaskSnapshotLoaderTest, LoadByLaunchDirect_ErrorNullPointer_Succeeds)
{
    auto task = MakeTaskJson(0, 1, 0, 0, 0, MakeMemCopyPayload(0, 1, MakeDataSlice(0, 0, 8), MakeDataSlice(1, 0, 8)));
    json content
        = {{"rank", 0},
           {"rankSize", 1},
           {"launchIndex", 0},
           {"aivCores", json::array({MakeBlockJson(0, json::array({task}))})}};
    WriteTaskFile(0, 0, content);

    AivRuntimeTaskSnapshot snapshot;
    EXPECT_TRUE(AivTaskSnapshotLoader::LoadRuntimeTaskSnapshotByLaunchDirect(0, 0, snapshot, nullptr));
}

// ==================== LoadRuntimeTaskSnapshotByLaunchDirect Error Tests ====================

TEST_F(AivTaskSnapshotLoaderTest, LoadByLaunchDirect_FileNotExist_ReturnsFalse)
{
    AivRuntimeTaskSnapshot snapshot;
    std::string errorMsg;
    EXPECT_FALSE(AivTaskSnapshotLoader::LoadRuntimeTaskSnapshotByLaunchDirect(99, 0, snapshot, &errorMsg));
    EXPECT_FALSE(errorMsg.empty());
}

TEST_F(AivTaskSnapshotLoaderTest, LoadByLaunchDirect_InvalidJson_ReturnsFalse)
{
    std::ofstream ofs(testDir_ / "data" / "hcclvm_aiv_rank0_launch0_task.json");
    ofs << "not valid json";
    ofs.close();

    AivRuntimeTaskSnapshot snapshot;
    std::string errorMsg;
    EXPECT_FALSE(AivTaskSnapshotLoader::LoadRuntimeTaskSnapshotByLaunchDirect(0, 0, snapshot, &errorMsg));
    EXPECT_FALSE(errorMsg.empty());
}

TEST_F(AivTaskSnapshotLoaderTest, LoadByLaunchDirect_RankIdMismatch_ReturnsFalse)
{
    json content = {{"rank", 1}, {"rankSize", 1}, {"launchIndex", 0}, {"aivCores", json::array()}};
    WriteTaskFile(0, 0, content);

    AivRuntimeTaskSnapshot snapshot;
    std::string errorMsg;
    EXPECT_FALSE(AivTaskSnapshotLoader::LoadRuntimeTaskSnapshotByLaunchDirect(0, 0, snapshot, &errorMsg));
    EXPECT_TRUE(errorMsg.find("rank id mismatch") != std::string::npos);
}

TEST_F(AivTaskSnapshotLoaderTest, LoadByLaunchDirect_LaunchIndexMismatch_ReturnsFalse)
{
    json content = {{"rank", 0}, {"rankSize", 1}, {"launchIndex", 5}, {"aivCores", json::array()}};
    WriteTaskFile(0, 3, content);

    AivRuntimeTaskSnapshot snapshot;
    std::string errorMsg;
    EXPECT_FALSE(AivTaskSnapshotLoader::LoadRuntimeTaskSnapshotByLaunchDirect(0, 3, snapshot, &errorMsg));
    EXPECT_TRUE(errorMsg.find("launchIndex mismatch") != std::string::npos);
}

TEST_F(AivTaskSnapshotLoaderTest, LoadByLaunchDirect_MissingAivCores_ReturnsFalse)
{
    json content = {{"rank", 0}, {"rankSize", 1}, {"launchIndex", 0}};
    WriteTaskFile(0, 0, content);

    AivRuntimeTaskSnapshot snapshot;
    std::string errorMsg;
    EXPECT_FALSE(AivTaskSnapshotLoader::LoadRuntimeTaskSnapshotByLaunchDirect(0, 0, snapshot, &errorMsg));
    EXPECT_TRUE(errorMsg.find("missing aivCores") != std::string::npos);
}

TEST_F(AivTaskSnapshotLoaderTest, LoadByLaunchDirect_AivCoresNotArray_ReturnsFalse)
{
    json content = {{"rank", 0}, {"rankSize", 1}, {"launchIndex", 0}, {"aivCores", "not_array"}};
    WriteTaskFile(0, 0, content);

    AivRuntimeTaskSnapshot snapshot;
    std::string errorMsg;
    EXPECT_FALSE(AivTaskSnapshotLoader::LoadRuntimeTaskSnapshotByLaunchDirect(0, 0, snapshot, &errorMsg));
    EXPECT_TRUE(errorMsg.find("missing aivCores") != std::string::npos);
}

TEST_F(AivTaskSnapshotLoaderTest, LoadByLaunchDirect_TaskArrayNotArray_ReturnsFalse)
{
    auto blockJson = json::object();
    blockJson["blockIdx"] = 0;
    blockJson["scalarTasks"] = "not_array";
    blockJson["mte2Tasks"] = json::array();
    blockJson["mte3Tasks"] = json::array();
    json content = {{"rank", 0}, {"rankSize", 1}, {"launchIndex", 0}, {"aivCores", json::array({blockJson})}};
    WriteTaskFile(0, 0, content);

    AivRuntimeTaskSnapshot snapshot;
    std::string errorMsg;
    EXPECT_FALSE(AivTaskSnapshotLoader::LoadRuntimeTaskSnapshotByLaunchDirect(0, 0, snapshot, &errorMsg));
    EXPECT_TRUE(errorMsg.find("not array") != std::string::npos);
}

TEST_F(AivTaskSnapshotLoaderTest, LoadByLaunchDirect_UnsupportedTaskType_ReturnsFalse)
{
    auto task = MakeTaskJson(99, 1, 0, 0, 0, json::object());
    auto block = MakeBlockJson(0, json::array({task}));
    json content = {{"rank", 0}, {"rankSize", 1}, {"launchIndex", 0}, {"aivCores", json::array({block})}};
    WriteTaskFile(0, 0, content);

    AivRuntimeTaskSnapshot snapshot;
    std::string errorMsg;
    EXPECT_FALSE(AivTaskSnapshotLoader::LoadRuntimeTaskSnapshotByLaunchDirect(0, 0, snapshot, &errorMsg));
    EXPECT_TRUE(errorMsg.find("unsupported AIV runtime task type") != std::string::npos);
}

TEST_F(AivTaskSnapshotLoaderTest, LoadByLaunchDirect_MemCopyMissingSrcSlice_ReturnsFalse)
{
    auto payload = json::object();
    payload["srcRank"] = 0;
    payload["dstRank"] = 1;
    payload["dst"] = MakeDataSlice(1, 0, 64);
    auto task = MakeTaskJson(0, 1, 0, 0, 0, payload);
    json content
        = {{"rank", 0},
           {"rankSize", 1},
           {"launchIndex", 0},
           {"aivCores", json::array({MakeBlockJson(0, json::array({task}))})}};
    WriteTaskFile(0, 0, content);

    AivRuntimeTaskSnapshot snapshot;
    std::string errorMsg;
    EXPECT_FALSE(AivTaskSnapshotLoader::LoadRuntimeTaskSnapshotByLaunchDirect(0, 0, snapshot, &errorMsg));
    EXPECT_TRUE(errorMsg.find("missing data slice pipe") != std::string::npos);
}

TEST_F(AivTaskSnapshotLoaderTest, LoadByLaunchDirect_ReduceMissingDstSlice_ReturnsFalse)
{
    auto dsS = MakeDataSlice(0, 0, 64);
    auto payload = json::object();
    payload["srcRank"] = 0;
    payload["dstRank"] = 1;
    payload["src"] = dsS;
    // dst missing
    auto task = MakeTaskJson(1, 2, 0, 0, 0, payload);
    json content
        = {{"rank", 0},
           {"rankSize", 1},
           {"launchIndex", 0},
           {"aivCores", json::array({MakeBlockJson(0, json::array({task}))})}};
    WriteTaskFile(0, 0, content);

    AivRuntimeTaskSnapshot snapshot;
    std::string errorMsg;
    EXPECT_FALSE(AivTaskSnapshotLoader::LoadRuntimeTaskSnapshotByLaunchDirect(0, 0, snapshot, &errorMsg));
    EXPECT_TRUE(errorMsg.find("missing data slice pipe") != std::string::npos);
}

TEST_F(AivTaskSnapshotLoaderTest, LoadByLaunchDirect_BarrierGroupUnresolvable_ReturnsFalse)
{
    // barrier1 references task 999 which doesn't exist
    auto barrier1 = MakeTaskJson(4, 10, 0, 0, 0, MakePipeBarrierPayload(0, {999}));
    json content
        = {{"rank", 0},
           {"rankSize", 1},
           {"launchIndex", 0},
           {"aivCores", json::array({MakeBlockJson(0, json::array({barrier1}))})}};
    WriteTaskFile(0, 0, content);

    AivRuntimeTaskSnapshot snapshot;
    std::string errorMsg;
    EXPECT_FALSE(AivTaskSnapshotLoader::LoadRuntimeTaskSnapshotByLaunchDirect(0, 0, snapshot, &errorMsg));
    EXPECT_TRUE(errorMsg.find("failed to resolve barrier group taskId") != std::string::npos);
}

TEST_F(AivTaskSnapshotLoaderTest, LoadByLaunchDirect_EmptyFile_ReturnsFalse)
{
    std::ofstream ofs(testDir_ / "data" / "hcclvm_aiv_rank0_launch0_task.json");
    ofs << "";
    ofs.close();

    AivRuntimeTaskSnapshot snapshot;
    std::string errorMsg;
    EXPECT_FALSE(AivTaskSnapshotLoader::LoadRuntimeTaskSnapshotByLaunchDirect(0, 0, snapshot, &errorMsg));
    EXPECT_TRUE(errorMsg.find("failed to parse file") != std::string::npos);
}

TEST_F(AivTaskSnapshotLoaderTest, LoadByLaunchDirect_EmptyAivCores_Succeeds)
{
    json content = {{"rank", 0}, {"rankSize", 1}, {"launchIndex", 0}, {"aivCores", json::array()}};
    WriteTaskFile(0, 0, content);

    AivRuntimeTaskSnapshot snapshot;
    std::string errorMsg;
    EXPECT_TRUE(AivTaskSnapshotLoader::LoadRuntimeTaskSnapshotByLaunchDirect(0, 0, snapshot, &errorMsg));
    EXPECT_TRUE(snapshot.blocks.empty());
}

TEST_F(AivTaskSnapshotLoaderTest, LoadByLaunchDirect_DataSliceDefaultValues)
{
    auto payload = json::object();
    payload["srcRank"] = 0;
    payload["dstRank"] = 1;
    payload["src"] = json::object();
    payload["dst"] = json::object();
    auto task = MakeTaskJson(0, 1, 0, 0, 0, payload);
    json content
        = {{"rank", 0},
           {"rankSize", 1},
           {"launchIndex", 0},
           {"aivCores", json::array({MakeBlockJson(0, json::array({task}))})}};
    WriteTaskFile(0, 0, content);

    AivRuntimeTaskSnapshot snapshot;
    EXPECT_TRUE(AivTaskSnapshotLoader::LoadRuntimeTaskSnapshotByLaunchDirect(0, 0, snapshot));
    ASSERT_EQ(snapshot.blocks[0].scalarTasks.size(), 1u);
    EXPECT_EQ(snapshot.blocks[0].scalarTasks[0]->GetTaskType(), AivSim::AivTaskType::MEM_COPY);
}

TEST_F(AivTaskSnapshotLoaderTest, LoadByLaunchDirect_TaskDefaultValues_Success)
{
    // task json with missing optional fields; defaults should be applied
    auto task = json::object();
    task["taskType"] = 2; // SET_FLAG, simplest payload
    task["payload"] = json::object();
    json content
        = {{"rank", 0},
           {"rankSize", 1},
           {"launchIndex", 0},
           {"aivCores", json::array({MakeBlockJson(0, json::array({task}))})}};
    WriteTaskFile(0, 0, content);

    AivRuntimeTaskSnapshot snapshot;
    EXPECT_TRUE(AivTaskSnapshotLoader::LoadRuntimeTaskSnapshotByLaunchDirect(0, 0, snapshot));
    ASSERT_EQ(snapshot.blocks[0].scalarTasks.size(), 1u);
    EXPECT_EQ(snapshot.blocks[0].scalarTasks[0]->GetTaskType(), AivSim::AivTaskType::SET_FLAG);
    EXPECT_EQ(snapshot.blocks[0].scalarTasks[0]->GetTaskId(), 0u);
    EXPECT_EQ(snapshot.blocks[0].scalarTasks[0]->GetRankId(), 0u);
    EXPECT_EQ(snapshot.blocks[0].scalarTasks[0]->GetBlockId(), 0u);
}
