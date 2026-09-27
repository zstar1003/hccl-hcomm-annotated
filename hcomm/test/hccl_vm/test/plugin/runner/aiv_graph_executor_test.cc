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

#include "aiv_graph_executor.h"
#include "aiv_mode_stub_base.h"
#include "aiv_task.h"
#include "aiv_task_snapshot_loader.h"
#include "aiv_resource_manager.h"
#include "ascendc_base_stub.h"
#include "sim_common_defs.h"

namespace fs = std::filesystem;
using json = nlohmann::json;

class AivGraphExecutorTest : public testing::Test {
protected:
    void SetUp() override
    {
        testDir_ = fs::temp_directory_path() / ("aiv_exec_test_" + std::to_string(::getpid()));
        fs::create_directories(testDir_ / "data");
        setenv("HCCL_VM_INSTALL_ROOT", testDir_.c_str(), 1);
        AivResourceManager::GetInstance().Reset();
        ownedBuffers_.clear();
    }

    void TearDown() override
    {
        unsetenv("HCCL_VM_INSTALL_ROOT");
        AivResourceManager::GetInstance().Reset();
        ownedBuffers_.clear();
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

    void SetupRankResource(uint32_t rankId, uint64_t inputSize, uint64_t outputSize, uint64_t cclSize)
    {
        auto& resMgr = AivResourceManager::GetInstance();
        auto& resources = const_cast<std::vector<AivRankResource>&>(resMgr.GetAllRankResources());
        if (rankId >= resources.size()) {
            resources.resize(rankId + 1);
        }
        auto inputBuf = std::make_unique<uint8_t[]>(inputSize);
        auto outputBuf = std::make_unique<uint8_t[]>(outputSize);
        auto cclBuf = std::make_unique<uint8_t[]>(cclSize);
        auto aivCommInfoBuf = std::make_unique<uint8_t[]>(AivCommInfoLayout::SIZE_BYTES);

        resources[rankId].inputBuffer.realAddr = inputBuf.get();
        resources[rankId].inputBuffer.size = inputSize;
        resources[rankId].outputBuffer.realAddr = outputBuf.get();
        resources[rankId].outputBuffer.size = outputSize;
        resources[rankId].cclBuffer.realAddr = cclBuf.get();
        resources[rankId].cclBuffer.size = cclSize;
        resources[rankId].aivCommInfoBuffer.realAddr = aivCommInfoBuf.get();
        resources[rankId].aivCommInfoBuffer.size = AivCommInfoLayout::SIZE_BYTES;

        ownedBuffers_.emplace_back(std::move(inputBuf));
        ownedBuffers_.emplace_back(std::move(outputBuf));
        ownedBuffers_.emplace_back(std::move(cclBuf));
        ownedBuffers_.emplace_back(std::move(aivCommInfoBuf));
    }

    fs::path testDir_;
    std::vector<std::unique_ptr<uint8_t[]>> ownedBuffers_;
};

TEST_F(AivGraphExecutorTest, Constructor_Default_NotInitialized)
{
    AivGraphExecutor executor(0);
    EXPECT_FALSE(executor.IsInitialized());
}

TEST_F(AivGraphExecutorTest, Constructor_WithLaunchIdx)
{
    AivGraphExecutor executor(42);
    EXPECT_FALSE(executor.IsInitialized());
}

TEST_F(AivGraphExecutorTest, Init_LoadSnapshot_Success)
{
    auto ds = MakeDataSlice(3, 0, 128);
    auto task = MakeTaskJson(0, 1, 0, 0, 0, MakeMemCopyPayload(0, 1, ds, ds));
    auto block = MakeBlockJson(0, json::array({task}));
    json content = {{"rank", 0}, {"rankSize", 2}, {"launchIndex", 0}, {"aivCores", json::array({block})}};
    WriteTaskFile(0, 0, content);

    AivGraphExecutor executor(0);
    EXPECT_TRUE(executor.Init(0, 0));
    EXPECT_TRUE(executor.IsInitialized());
}

TEST_F(AivGraphExecutorTest, Init_FileNotExist_ReturnsFalse)
{
    AivGraphExecutor executor(0);
    EXPECT_FALSE(executor.Init(99, 99));
    EXPECT_FALSE(executor.IsInitialized());
}

TEST_F(AivGraphExecutorTest, Init_RankIdMismatch_ReturnsFalse)
{
    json content = {{"rank", 5}, {"rankSize", 2}, {"launchIndex", 0}, {"aivCores", json::array()}};
    WriteTaskFile(0, 0, content);

    AivGraphExecutor executor(0);
    EXPECT_FALSE(executor.Init(0, 0));
}

TEST_F(AivGraphExecutorTest, Execute_EmptyTasks_Success)
{
    json content = {{"rank", 0}, {"rankSize", 1}, {"launchIndex", 0}, {"aivCores", json::array()}};
    WriteTaskFile(0, 0, content);

    AivGraphExecutor executor(0);
    ASSERT_TRUE(executor.Init(0, 0));
    EXPECT_EQ(executor.Execute(), HcclSim::HcclVmResult::HCCL_SIM_SUCCESS);
}

TEST_F(AivGraphExecutorTest, Execute_MemCopyUB_Success)
{
    auto ds = MakeDataSlice(3, 0, 128);
    auto task = MakeTaskJson(0, 1, 0, 0, 0, MakeMemCopyPayload(0, 0, ds, ds));
    auto block = MakeBlockJson(0, json::array({task}));
    json content = {{"rank", 0}, {"rankSize", 1}, {"launchIndex", 0}, {"aivCores", json::array({block})}};
    WriteTaskFile(0, 0, content);

    AivGraphExecutor executor(0);
    ASSERT_TRUE(executor.Init(0, 0));
    EXPECT_EQ(executor.Execute(), HcclSim::HcclVmResult::HCCL_SIM_SUCCESS);
}

TEST_F(AivGraphExecutorTest, Execute_MemCopyInputToOutput_Success)
{
    SetupRankResource(0, 4096, 4096, 4096);
    auto srcDs = MakeDataSlice(0, 0, 128);
    auto dstDs = MakeDataSlice(1, 0, 128);
    auto task = MakeTaskJson(0, 1, 0, 0, 0, MakeMemCopyPayload(0, 0, srcDs, dstDs));
    auto block = MakeBlockJson(0, json::array({task}));
    json content = {{"rank", 0}, {"rankSize", 1}, {"launchIndex", 0}, {"aivCores", json::array({block})}};
    WriteTaskFile(0, 0, content);

    AivGraphExecutor executor(0);
    ASSERT_TRUE(executor.Init(0, 0));
    EXPECT_EQ(executor.Execute(), HcclSim::HcclVmResult::HCCL_SIM_SUCCESS);
}

TEST_F(AivGraphExecutorTest, Execute_MemCopyLenMismatch_ReturnsError)
{
    auto srcDs = MakeDataSlice(3, 0, 128);
    auto dstDs = MakeDataSlice(3, 0, 256);
    auto task = MakeTaskJson(0, 1, 0, 0, 0, MakeMemCopyPayload(0, 0, srcDs, dstDs));
    auto block = MakeBlockJson(0, json::array({task}));
    json content = {{"rank", 0}, {"rankSize", 1}, {"launchIndex", 0}, {"aivCores", json::array({block})}};
    WriteTaskFile(0, 0, content);

    AivGraphExecutor executor(0);
    ASSERT_TRUE(executor.Init(0, 0));
    EXPECT_EQ(executor.Execute(), HcclSim::HcclVmResult::HCCL_SIM_VRT_ERROR_CMD);
}

TEST_F(AivGraphExecutorTest, Execute_MemCopyUBOutOfBounds_ReturnsError)
{
    auto srcDs = MakeDataSlice(3, 0, 200 * 1024);
    auto dstDs = MakeDataSlice(3, 0, 200 * 1024);
    auto task = MakeTaskJson(0, 1, 0, 0, 0, MakeMemCopyPayload(0, 0, srcDs, dstDs));
    auto block = MakeBlockJson(0, json::array({task}));
    json content = {{"rank", 0}, {"rankSize", 1}, {"launchIndex", 0}, {"aivCores", json::array({block})}};
    WriteTaskFile(0, 0, content);

    AivGraphExecutor executor(0);
    ASSERT_TRUE(executor.Init(0, 0));
    EXPECT_EQ(executor.Execute(), HcclSim::HcclVmResult::HCCL_SIM_VRT_ERROR_CMD);
}

TEST_F(AivGraphExecutorTest, Execute_MemCopyRankResourceNull_ReturnsError)
{
    auto srcDs = MakeDataSlice(0, 0, 128);
    auto dstDs = MakeDataSlice(1, 0, 128);
    auto task = MakeTaskJson(0, 1, 0, 0, 0, MakeMemCopyPayload(0, 99, srcDs, dstDs));
    auto block = MakeBlockJson(0, json::array({task}));
    json content = {{"rank", 0}, {"rankSize", 1}, {"launchIndex", 0}, {"aivCores", json::array({block})}};
    WriteTaskFile(0, 0, content);

    AivGraphExecutor executor(0);
    ASSERT_TRUE(executor.Init(0, 0));
    EXPECT_EQ(executor.Execute(), HcclSim::HcclVmResult::HCCL_SIM_VRT_ERROR_CMD);
}

TEST_F(AivGraphExecutorTest, Execute_MemCopyAivCommInfo_Success)
{
    SetupRankResource(0, 4096, 4096, 4096);
    auto* resource = const_cast<AivRankResource*>(AivResourceManager::GetInstance().GetRankResource(0));
    ASSERT_NE(resource, nullptr);
    auto* commInfo = static_cast<uint8_t*>(resource->aivCommInfoBuffer.realAddr);
    constexpr uint64_t pongOffset = AivCommInfoLayout::PONG_OFFSET;
    commInfo[pongOffset] = 0x5a;
    auto srcDs = MakeDataSlice(4, pongOffset, 1);
    auto dstDs = MakeDataSlice(1, 0, 1);
    auto task = MakeTaskJson(0, 1, 0, 0, 0, MakeMemCopyPayload(0, 0, srcDs, dstDs));
    auto block = MakeBlockJson(0, json::array({task}));
    json content = {{"rank", 0}, {"rankSize", 1}, {"launchIndex", 0}, {"aivCores", json::array({block})}};
    WriteTaskFile(0, 0, content);

    AivGraphExecutor executor(0);
    ASSERT_TRUE(executor.Init(0, 0));
    EXPECT_EQ(executor.Execute(), HcclSim::HcclVmResult::HCCL_SIM_SUCCESS);
    EXPECT_EQ(static_cast<uint8_t*>(resource->outputBuffer.realAddr)[0], 0x5a);
}

TEST_F(AivGraphExecutorTest, Execute_MemCopyRankMemOutOfBounds_ReturnsError)
{
    SetupRankResource(0, 64, 64, 64);
    auto srcDs = MakeDataSlice(0, 0, 4096);
    auto dstDs = MakeDataSlice(1, 0, 4096);
    auto task = MakeTaskJson(0, 1, 0, 0, 0, MakeMemCopyPayload(0, 0, srcDs, dstDs));
    auto block = MakeBlockJson(0, json::array({task}));
    json content = {{"rank", 0}, {"rankSize", 1}, {"launchIndex", 0}, {"aivCores", json::array({block})}};
    WriteTaskFile(0, 0, content);

    AivGraphExecutor executor(0);
    ASSERT_TRUE(executor.Init(0, 0));
    EXPECT_EQ(executor.Execute(), HcclSim::HcclVmResult::HCCL_SIM_VRT_ERROR_CMD);
}

TEST_F(AivGraphExecutorTest, Execute_MemCopyCCLBuffer_Success)
{
    SetupRankResource(0, 4096, 4096, 4096);
    auto srcDs = MakeDataSlice(2, 0, 128);
    auto dstDs = MakeDataSlice(3, 0, 128);
    auto task = MakeTaskJson(0, 1, 0, 0, 0, MakeMemCopyPayload(0, 0, srcDs, dstDs));
    auto block = MakeBlockJson(0, json::array({task}));
    json content = {{"rank", 0}, {"rankSize", 1}, {"launchIndex", 0}, {"aivCores", json::array({block})}};
    WriteTaskFile(0, 0, content);

    AivGraphExecutor executor(0);
    ASSERT_TRUE(executor.Init(0, 0));
    EXPECT_EQ(executor.Execute(), HcclSim::HcclVmResult::HCCL_SIM_SUCCESS);
}

TEST_F(AivGraphExecutorTest, Execute_ReduceFP32Sum_Success)
{
    SetupRankResource(0, 4096, 4096, 4096);
    auto srcDs = MakeDataSlice(0, 0, 16);
    auto dstDs = MakeDataSlice(1, 0, 16);
    auto task = MakeTaskJson(1, 1, 0, 0, 1, MakeReducePayload(0, 0, srcDs, dstDs, 4, 0));
    auto block = MakeBlockJson(0, json::array(), json::array({task}));
    json content = {{"rank", 0}, {"rankSize", 1}, {"launchIndex", 0}, {"aivCores", json::array({block})}};
    WriteTaskFile(0, 0, content);

    AivGraphExecutor executor(0);
    ASSERT_TRUE(executor.Init(0, 0));
    EXPECT_EQ(executor.Execute(), HcclSim::HcclVmResult::HCCL_SIM_SUCCESS);
}

TEST_F(AivGraphExecutorTest, Execute_ReduceInt32Max_Success)
{
    SetupRankResource(0, 4096, 4096, 4096);
    auto srcDs = MakeDataSlice(0, 0, 16);
    auto dstDs = MakeDataSlice(1, 0, 16);
    auto task = MakeTaskJson(1, 1, 0, 0, 1, MakeReducePayload(0, 0, srcDs, dstDs, 2, 2));
    auto block = MakeBlockJson(0, json::array(), json::array({task}));
    json content = {{"rank", 0}, {"rankSize", 1}, {"launchIndex", 0}, {"aivCores", json::array({block})}};
    WriteTaskFile(0, 0, content);

    AivGraphExecutor executor(0);
    ASSERT_TRUE(executor.Init(0, 0));
    EXPECT_EQ(executor.Execute(), HcclSim::HcclVmResult::HCCL_SIM_SUCCESS);
}

TEST_F(AivGraphExecutorTest, Execute_ReduceInt32Min_Success)
{
    SetupRankResource(0, 4096, 4096, 4096);
    auto srcDs = MakeDataSlice(0, 0, 16);
    auto dstDs = MakeDataSlice(1, 0, 16);
    auto task = MakeTaskJson(1, 1, 0, 0, 1, MakeReducePayload(0, 0, srcDs, dstDs, 5, 2));
    auto block = MakeBlockJson(0, json::array(), json::array({task}));
    json content = {{"rank", 0}, {"rankSize", 1}, {"launchIndex", 0}, {"aivCores", json::array({block})}};
    WriteTaskFile(0, 0, content);

    AivGraphExecutor executor(0);
    ASSERT_TRUE(executor.Init(0, 0));
    EXPECT_EQ(executor.Execute(), HcclSim::HcclVmResult::HCCL_SIM_SUCCESS);
}

TEST_F(AivGraphExecutorTest, Execute_ReduceHIF8Sum_Success)
{
    SetupRankResource(0, 4096, 4096, 4096);
    auto srcDs = MakeDataSlice(0, 0, 16);
    auto dstDs = MakeDataSlice(1, 0, 16);
    auto task = MakeTaskJson(1, 1, 0, 0, 1, MakeReducePayload(0, 0, srcDs, dstDs, 14, 0));
    auto block = MakeBlockJson(0, json::array(), json::array({task}));
    json content = {{"rank", 0}, {"rankSize", 1}, {"launchIndex", 0}, {"aivCores", json::array({block})}};
    WriteTaskFile(0, 0, content);

    AivGraphExecutor executor(0);
    ASSERT_TRUE(executor.Init(0, 0));
    EXPECT_EQ(executor.Execute(), HcclSim::HcclVmResult::HCCL_SIM_SUCCESS);
}

TEST_F(AivGraphExecutorTest, Execute_ReduceInt16Sum_Success)
{
    SetupRankResource(0, 4096, 4096, 4096);
    auto srcDs = MakeDataSlice(0, 0, 16);
    auto dstDs = MakeDataSlice(1, 0, 16);
    auto task = MakeTaskJson(1, 1, 0, 0, 1, MakeReducePayload(0, 0, srcDs, dstDs, 1, 0));
    auto block = MakeBlockJson(0, json::array(), json::array({task}));
    json content = {{"rank", 0}, {"rankSize", 1}, {"launchIndex", 0}, {"aivCores", json::array({block})}};
    WriteTaskFile(0, 0, content);

    AivGraphExecutor executor(0);
    ASSERT_TRUE(executor.Init(0, 0));
    EXPECT_EQ(executor.Execute(), HcclSim::HcclVmResult::HCCL_SIM_SUCCESS);
}

TEST_F(AivGraphExecutorTest, Execute_ReduceUint16Sum_Success)
{
    SetupRankResource(0, 4096, 4096, 4096);
    auto srcDs = MakeDataSlice(0, 0, 16);
    auto dstDs = MakeDataSlice(1, 0, 16);
    auto task = MakeTaskJson(1, 1, 0, 0, 1, MakeReducePayload(0, 0, srcDs, dstDs, 2, 0));
    auto block = MakeBlockJson(0, json::array(), json::array({task}));
    json content = {{"rank", 0}, {"rankSize", 1}, {"launchIndex", 0}, {"aivCores", json::array({block})}};
    WriteTaskFile(0, 0, content);

    AivGraphExecutor executor(0);
    ASSERT_TRUE(executor.Init(0, 0));
    EXPECT_EQ(executor.Execute(), HcclSim::HcclVmResult::HCCL_SIM_SUCCESS);
}

TEST_F(AivGraphExecutorTest, Execute_ReduceUint32Sum_Success)
{
    SetupRankResource(0, 4096, 4096, 4096);
    auto srcDs = MakeDataSlice(0, 0, 16);
    auto dstDs = MakeDataSlice(1, 0, 16);
    auto task = MakeTaskJson(1, 1, 0, 0, 1, MakeReducePayload(0, 0, srcDs, dstDs, 6, 0));
    auto block = MakeBlockJson(0, json::array(), json::array({task}));
    json content = {{"rank", 0}, {"rankSize", 1}, {"launchIndex", 0}, {"aivCores", json::array({block})}};
    WriteTaskFile(0, 0, content);

    AivGraphExecutor executor(0);
    ASSERT_TRUE(executor.Init(0, 0));
    EXPECT_EQ(executor.Execute(), HcclSim::HcclVmResult::HCCL_SIM_SUCCESS);
}

TEST_F(AivGraphExecutorTest, Execute_ReduceInt8Sum_Success)
{
    SetupRankResource(0, 4096, 4096, 4096);
    auto srcDs = MakeDataSlice(0, 0, 16);
    auto dstDs = MakeDataSlice(1, 0, 16);
    auto task = MakeTaskJson(1, 1, 0, 0, 1, MakeReducePayload(0, 0, srcDs, dstDs, 0, 0));
    auto block = MakeBlockJson(0, json::array(), json::array({task}));
    json content = {{"rank", 0}, {"rankSize", 1}, {"launchIndex", 0}, {"aivCores", json::array({block})}};
    WriteTaskFile(0, 0, content);

    AivGraphExecutor executor(0);
    ASSERT_TRUE(executor.Init(0, 0));
    EXPECT_EQ(executor.Execute(), HcclSim::HcclVmResult::HCCL_SIM_SUCCESS);
}

TEST_F(AivGraphExecutorTest, Execute_ReduceUint8Sum_Success)
{
    SetupRankResource(0, 4096, 4096, 4096);
    auto srcDs = MakeDataSlice(0, 0, 16);
    auto dstDs = MakeDataSlice(1, 0, 16);
    auto task = MakeTaskJson(1, 1, 0, 0, 1, MakeReducePayload(0, 0, srcDs, dstDs, 7, 0));
    auto block = MakeBlockJson(0, json::array(), json::array({task}));
    json content = {{"rank", 0}, {"rankSize", 1}, {"launchIndex", 0}, {"aivCores", json::array({block})}};
    WriteTaskFile(0, 0, content);

    AivGraphExecutor executor(0);
    ASSERT_TRUE(executor.Init(0, 0));
    EXPECT_EQ(executor.Execute(), HcclSim::HcclVmResult::HCCL_SIM_SUCCESS);
}

TEST_F(AivGraphExecutorTest, Execute_ReduceBFP16Sum_Success)
{
    SetupRankResource(0, 4096, 4096, 4096);
    auto srcDs = MakeDataSlice(0, 0, 16);
    auto dstDs = MakeDataSlice(1, 0, 16);
    auto task = MakeTaskJson(1, 1, 0, 0, 1, MakeReducePayload(0, 0, srcDs, dstDs, 11, 0));
    auto block = MakeBlockJson(0, json::array(), json::array({task}));
    json content = {{"rank", 0}, {"rankSize", 1}, {"launchIndex", 0}, {"aivCores", json::array({block})}};
    WriteTaskFile(0, 0, content);

    AivGraphExecutor executor(0);
    ASSERT_TRUE(executor.Init(0, 0));
    EXPECT_EQ(executor.Execute(), HcclSim::HcclVmResult::HCCL_SIM_SUCCESS);
}

TEST_F(AivGraphExecutorTest, Execute_ReduceInt64Sum_Success)
{
    SetupRankResource(0, 4096, 4096, 4096);
    auto srcDs = MakeDataSlice(0, 0, 32);
    auto dstDs = MakeDataSlice(1, 0, 32);
    auto task = MakeTaskJson(1, 1, 0, 0, 1, MakeReducePayload(0, 0, srcDs, dstDs, 9, 0));
    auto block = MakeBlockJson(0, json::array(), json::array({task}));
    json content = {{"rank", 0}, {"rankSize", 1}, {"launchIndex", 0}, {"aivCores", json::array({block})}};
    WriteTaskFile(0, 0, content);

    AivGraphExecutor executor(0);
    ASSERT_TRUE(executor.Init(0, 0));
    EXPECT_EQ(executor.Execute(), HcclSim::HcclVmResult::HCCL_SIM_SUCCESS);
}

TEST_F(AivGraphExecutorTest, Execute_ReduceUint64Sum_Success)
{
    SetupRankResource(0, 4096, 4096, 4096);
    auto srcDs = MakeDataSlice(0, 0, 32);
    auto dstDs = MakeDataSlice(1, 0, 32);
    auto task = MakeTaskJson(1, 1, 0, 0, 1, MakeReducePayload(0, 0, srcDs, dstDs, 6, 0));
    auto block = MakeBlockJson(0, json::array(), json::array({task}));
    json content = {{"rank", 0}, {"rankSize", 1}, {"launchIndex", 0}, {"aivCores", json::array({block})}};
    WriteTaskFile(0, 0, content);

    AivGraphExecutor executor(0);
    ASSERT_TRUE(executor.Init(0, 0));
    EXPECT_EQ(executor.Execute(), HcclSim::HcclVmResult::HCCL_SIM_SUCCESS);
}

TEST_F(AivGraphExecutorTest, Execute_ReduceLenMismatch_ReturnsError)
{
    SetupRankResource(0, 4096, 4096, 4096);
    auto srcDs = MakeDataSlice(0, 0, 128);
    auto dstDs = MakeDataSlice(1, 0, 256);
    auto task = MakeTaskJson(1, 1, 0, 0, 1, MakeReducePayload(0, 0, srcDs, dstDs, 4, 0));
    auto block = MakeBlockJson(0, json::array(), json::array({task}));
    json content = {{"rank", 0}, {"rankSize", 1}, {"launchIndex", 0}, {"aivCores", json::array({block})}};
    WriteTaskFile(0, 0, content);

    AivGraphExecutor executor(0);
    ASSERT_TRUE(executor.Init(0, 0));
    EXPECT_EQ(executor.Execute(), HcclSim::HcclVmResult::HCCL_SIM_VRT_ERROR_CMD);
}

TEST_F(AivGraphExecutorTest, Execute_ReduceUnsupportedDataType_ReturnsError)
{
    SetupRankResource(0, 4096, 4096, 4096);
    auto srcDs = MakeDataSlice(0, 0, 16);
    auto dstDs = MakeDataSlice(1, 0, 16);
    auto task = MakeTaskJson(1, 1, 0, 0, 1, MakeReducePayload(0, 0, srcDs, dstDs, 99, 0));
    auto block = MakeBlockJson(0, json::array(), json::array({task}));
    json content = {{"rank", 0}, {"rankSize", 1}, {"launchIndex", 0}, {"aivCores", json::array({block})}};
    WriteTaskFile(0, 0, content);

    AivGraphExecutor executor(0);
    ASSERT_TRUE(executor.Init(0, 0));
    EXPECT_EQ(executor.Execute(), HcclSim::HcclVmResult::HCCL_SIM_VRT_ERROR_CMD);
}

TEST_F(AivGraphExecutorTest, Execute_ReduceUnsupportedOp_ReturnsError)
{
    SetupRankResource(0, 4096, 4096, 4096);
    auto srcDs = MakeDataSlice(0, 0, 16);
    auto dstDs = MakeDataSlice(1, 0, 16);
    auto task = MakeTaskJson(1, 1, 0, 0, 1, MakeReducePayload(0, 0, srcDs, dstDs, 4, 99));
    auto block = MakeBlockJson(0, json::array(), json::array({task}));
    json content = {{"rank", 0}, {"rankSize", 1}, {"launchIndex", 0}, {"aivCores", json::array({block})}};
    WriteTaskFile(0, 0, content);

    AivGraphExecutor executor(0);
    ASSERT_TRUE(executor.Init(0, 0));
    EXPECT_EQ(executor.Execute(), HcclSim::HcclVmResult::HCCL_SIM_VRT_ERROR_CMD);
}

TEST_F(AivGraphExecutorTest, Execute_ReduceLengthNotAligned_ReturnsError)
{
    SetupRankResource(0, 4096, 4096, 4096);
    auto srcDs = MakeDataSlice(0, 0, 3);
    auto dstDs = MakeDataSlice(1, 0, 3);
    auto task = MakeTaskJson(1, 1, 0, 0, 1, MakeReducePayload(0, 0, srcDs, dstDs, 4, 0));
    auto block = MakeBlockJson(0, json::array(), json::array({task}));
    json content = {{"rank", 0}, {"rankSize", 1}, {"launchIndex", 0}, {"aivCores", json::array({block})}};
    WriteTaskFile(0, 0, content);

    AivGraphExecutor executor(0);
    ASSERT_TRUE(executor.Init(0, 0));
    EXPECT_EQ(executor.Execute(), HcclSim::HcclVmResult::HCCL_SIM_VRT_ERROR_CMD);
}

TEST_F(AivGraphExecutorTest, Execute_ReduceSrcNull_ReturnsError)
{
    auto srcDs = MakeDataSlice(0, 0, 128);
    auto dstDs = MakeDataSlice(1, 0, 128);
    auto task = MakeTaskJson(1, 1, 0, 0, 1, MakeReducePayload(0, 99, srcDs, dstDs, 4, 0));
    auto block = MakeBlockJson(0, json::array(), json::array({task}));
    json content = {{"rank", 0}, {"rankSize", 1}, {"launchIndex", 0}, {"aivCores", json::array({block})}};
    WriteTaskFile(0, 0, content);

    AivGraphExecutor executor(0);
    ASSERT_TRUE(executor.Init(0, 0));
    EXPECT_EQ(executor.Execute(), HcclSim::HcclVmResult::HCCL_SIM_VRT_ERROR_CMD);
}

TEST_F(AivGraphExecutorTest, Execute_SetFlagWaitFlag_Success)
{
    auto setFlag = MakeTaskJson(2, 1, 0, 0, 0, MakeSetFlagPayload(0, 1, 5));
    auto waitFlag = MakeTaskJson(3, 2, 0, 0, 0, MakeWaitFlagPayload(1, 2, 5));
    auto block = MakeBlockJson(0, json::array({setFlag, waitFlag}));
    json content = {{"rank", 0}, {"rankSize", 1}, {"launchIndex", 0}, {"aivCores", json::array({block})}};
    WriteTaskFile(0, 0, content);

    AivGraphExecutor executor(0);
    ASSERT_TRUE(executor.Init(0, 0));
    EXPECT_EQ(executor.Execute(), HcclSim::HcclVmResult::HCCL_SIM_SUCCESS);
}

TEST_F(AivGraphExecutorTest, DISABLED_Execute_PipeBarrierAllPass_Success)
{
    auto barrier1 = MakeTaskJson(4, 10, 0, 0, 0, MakePipeBarrierPayload(0, {11}));
    auto barrier2 = MakeTaskJson(4, 11, 0, 0, 0, MakePipeBarrierPayload(0, {10}));
    auto block = MakeBlockJson(0, json::array({barrier1, barrier2}));
    json content = {{"rank", 0}, {"rankSize", 1}, {"launchIndex", 0}, {"aivCores", json::array({block})}};
    WriteTaskFile(0, 0, content);

    AivGraphExecutor executor(0);
    ASSERT_TRUE(executor.Init(0, 0));
    EXPECT_EQ(executor.Execute(), HcclSim::HcclVmResult::HCCL_SIM_SUCCESS);
}

TEST_F(AivGraphExecutorTest, Execute_PipeBarrierSingleTask_Success)
{
    auto barrier = MakeTaskJson(4, 5, 0, 0, 0, MakePipeBarrierPayload(0, json::array()));
    auto block = MakeBlockJson(0, json::array({barrier}));
    json content = {{"rank", 0}, {"rankSize", 1}, {"launchIndex", 0}, {"aivCores", json::array({block})}};
    WriteTaskFile(0, 0, content);

    AivGraphExecutor executor(0);
    ASSERT_TRUE(executor.Init(0, 0));
    EXPECT_EQ(executor.Execute(), HcclSim::HcclVmResult::HCCL_SIM_SUCCESS);
}

TEST_F(AivGraphExecutorTest, Execute_SendFlagSuccess)
{
    SetupRankResource(0, 4096, 4096, 4096);
    auto sendFlag = MakeTaskJson(6, 1, 0, 0, 0, MakeSendFlagPayload(0, 0, 42));
    auto block = MakeBlockJson(0, json::array({sendFlag}));
    json content = {{"rank", 0}, {"rankSize", 1}, {"launchIndex", 0}, {"aivCores", json::array({block})}};
    WriteTaskFile(0, 0, content);

    AivGraphExecutor executor(0);
    ASSERT_TRUE(executor.Init(0, 0));
    EXPECT_EQ(executor.Execute(), HcclSim::HcclVmResult::HCCL_SIM_SUCCESS);
}

TEST_F(AivGraphExecutorTest, Execute_SendFlagRankNotExist_ReturnsError)
{
    auto sendFlag = MakeTaskJson(6, 1, 0, 0, 0, MakeSendFlagPayload(99, 0, 1));
    auto block = MakeBlockJson(0, json::array({sendFlag}));
    json content = {{"rank", 0}, {"rankSize", 1}, {"launchIndex", 0}, {"aivCores", json::array({block})}};
    WriteTaskFile(0, 0, content);

    AivGraphExecutor executor(0);
    ASSERT_TRUE(executor.Init(0, 0));
    EXPECT_EQ(executor.Execute(), HcclSim::HcclVmResult::HCCL_SIM_VRT_ERROR_CMD);
}

TEST_F(AivGraphExecutorTest, Execute_SendFlagOffsetOutOfBounds_ReturnsError)
{
    SetupRankResource(0, 4096, 4096, 4096);
    auto sendFlag = MakeTaskJson(6, 1, 0, 0, 0, MakeSendFlagPayload(0, AivCommInfoLayout::SIZE_BYTES, 1));
    auto block = MakeBlockJson(0, json::array({sendFlag}));
    json content = {{"rank", 0}, {"rankSize", 1}, {"launchIndex", 0}, {"aivCores", json::array({block})}};
    WriteTaskFile(0, 0, content);

    AivGraphExecutor executor(0);
    ASSERT_TRUE(executor.Init(0, 0));
    EXPECT_EQ(executor.Execute(), HcclSim::HcclVmResult::HCCL_SIM_VRT_ERROR_CMD);
}

TEST_F(AivGraphExecutorTest, Execute_RecvFlagMatchSuccess)
{
    SetupRankResource(0, 4096, 4096, 4096);
    auto sendFlag = MakeTaskJson(6, 1, 0, 0, 0, MakeSendFlagPayload(0, 0, 7));
    auto recvFlag = MakeTaskJson(7, 2, 0, 0, 0, MakeRecvFlagPayload(0, 0, 7));
    auto block = MakeBlockJson(0, json::array({sendFlag, recvFlag}));
    json content = {{"rank", 0}, {"rankSize", 1}, {"launchIndex", 0}, {"aivCores", json::array({block})}};
    WriteTaskFile(0, 0, content);

    AivGraphExecutor executor(0);
    ASSERT_TRUE(executor.Init(0, 0));
    EXPECT_EQ(executor.Execute(), HcclSim::HcclVmResult::HCCL_SIM_SUCCESS);
}

TEST_F(AivGraphExecutorTest, DISABLED_Execute_RecvFlagNotMatch_ReturnsHold)
{
    SetupRankResource(0, 4096, 4096, 4096);
    auto sendFlag = MakeTaskJson(6, 1, 0, 0, 0, MakeSendFlagPayload(0, 0, 7));
    auto recvFlag = MakeTaskJson(7, 2, 0, 0, 0, MakeRecvFlagPayload(0, 0, 99));
    auto block = MakeBlockJson(0, json::array({sendFlag, recvFlag}));
    json content = {{"rank", 0}, {"rankSize", 1}, {"launchIndex", 0}, {"aivCores", json::array({block})}};
    WriteTaskFile(0, 0, content);

    AivGraphExecutor executor(0);
    ASSERT_TRUE(executor.Init(0, 0));
    EXPECT_EQ(executor.Execute(), HcclSim::HcclVmResult::HCCL_SIM_SUCCESS);
}

TEST_F(AivGraphExecutorTest, Execute_ReduceUint32Min_Success)
{
    SetupRankResource(0, 4096, 4096, 4096);
    auto srcDs = MakeDataSlice(0, 0, 16);
    auto dstDs = MakeDataSlice(1, 0, 16);
    auto task = MakeTaskJson(1, 1, 0, 0, 1, MakeReducePayload(0, 0, srcDs, dstDs, 7, 2));
    auto block = MakeBlockJson(0, json::array(), json::array({task}));
    json content = {{"rank", 0}, {"rankSize", 1}, {"launchIndex", 0}, {"aivCores", json::array({block})}};
    WriteTaskFile(0, 0, content);

    AivGraphExecutor executor(0);
    ASSERT_TRUE(executor.Init(0, 0));
    EXPECT_EQ(executor.Execute(), HcclSim::HcclVmResult::HCCL_SIM_SUCCESS);
}

TEST_F(AivGraphExecutorTest, Init_TaskWithSetFlag_AppendPipe)
{
    auto ds = MakeDataSlice(3, 0, 64);
    auto task = MakeTaskJson(2, 1, 0, 0, 0, MakeSetFlagPayload(0, 0, 1));
    auto block = MakeBlockJson(0, json::array({task}));
    json content = {{"rank", 0}, {"rankSize", 1}, {"launchIndex", 0}, {"aivCores", json::array({block})}};
    WriteTaskFile(0, 0, content);

    AivGraphExecutor executor(0);
    ASSERT_TRUE(executor.Init(0, 0));
    EXPECT_TRUE(executor.IsInitialized());
}

TEST_F(AivGraphExecutorTest, Init_TaskWithWaitFlag_AppendPipe)
{
    auto ds = MakeDataSlice(3, 0, 64);
    auto task = MakeTaskJson(3, 2, 0, 0, 0, MakeWaitFlagPayload(0, 0, 2));
    auto block = MakeBlockJson(0, json::array({task}));
    json content = {{"rank", 0}, {"rankSize", 1}, {"launchIndex", 0}, {"aivCores", json::array({block})}};
    WriteTaskFile(0, 0, content);

    AivGraphExecutor executor(0);
    ASSERT_TRUE(executor.Init(0, 0));
    EXPECT_TRUE(executor.IsInitialized());
}

TEST_F(AivGraphExecutorTest, Execute_SendRecvFlag_Hold)
{
    SetupRankResource(0, 4096, 4096, 4096);
    SetupRankResource(1, 4096, 4096, 4096);
    auto recvTask = MakeTaskJson(7, 1, 0, 0, 0, MakeRecvFlagPayload(1, 0, -1));
    auto block = MakeBlockJson(0, json::array(), json::array({recvTask}));
    json content = {{"rank", 0}, {"rankSize", 2}, {"launchIndex", 0}, {"aivCores", json::array({block})}};
    WriteTaskFile(0, 0, content);

    AivGraphExecutor executor(0);
    ASSERT_TRUE(executor.Init(0, 0));
    EXPECT_EQ(executor.Execute(), HcclSim::HcclVmResult::HCCL_SIM_VRT_HOLD_CMD);
}

TEST_F(AivGraphExecutorTest, Execute_MixedTasks_LengthOtherThan255)
{
    SetupRankResource(0, 4096, 4096, 4096);
    SetupRankResource(1, 4096, 4096, 4096);
    auto memCpy
        = MakeTaskJson(0, 1, 0, 0, 0, MakeMemCopyPayload(0, 1, MakeDataSlice(0, 0, 128), MakeDataSlice(1, 0, 128)));
    auto reduce
        = MakeTaskJson(1, 2, 1, 0, 0, MakeReducePayload(1, 0, MakeDataSlice(0, 0, 16), MakeDataSlice(1, 0, 16), 5, 0));
    auto block = MakeBlockJson(0, json::array({memCpy}), json::array({reduce}));
    json content = {{"rank", 0}, {"rankSize", 2}, {"launchIndex", 0}, {"aivCores", json::array({block})}};
    WriteTaskFile(0, 0, content);

    AivGraphExecutor executor(0);
    ASSERT_TRUE(executor.Init(0, 0));
    EXPECT_EQ(executor.Execute(), HcclSim::HcclVmResult::HCCL_SIM_SUCCESS);
}

TEST_F(AivGraphExecutorTest, Execute_RecvFlagRankNotExist_ReturnsError)
{
    auto recvFlag = MakeTaskJson(7, 1, 0, 0, 0, MakeRecvFlagPayload(99, 0, 1));
    auto block = MakeBlockJson(0, json::array({recvFlag}));
    json content = {{"rank", 0}, {"rankSize", 1}, {"launchIndex", 0}, {"aivCores", json::array({block})}};
    WriteTaskFile(0, 0, content);

    AivGraphExecutor executor(0);
    ASSERT_TRUE(executor.Init(0, 0));
    EXPECT_EQ(executor.Execute(), HcclSim::HcclVmResult::HCCL_SIM_VRT_ERROR_CMD);
}

TEST_F(AivGraphExecutorTest, Execute_RecvFlagOffsetOutOfBounds_ReturnsError)
{
    SetupRankResource(0, 4096, 4096, 4096);
    auto recvFlag = MakeTaskJson(7, 1, 0, 0, 0, MakeRecvFlagPayload(0, AivCommInfoLayout::SIZE_BYTES, 1));
    auto block = MakeBlockJson(0, json::array({recvFlag}));
    json content = {{"rank", 0}, {"rankSize", 1}, {"launchIndex", 0}, {"aivCores", json::array({block})}};
    WriteTaskFile(0, 0, content);

    AivGraphExecutor executor(0);
    ASSERT_TRUE(executor.Init(0, 0));
    EXPECT_EQ(executor.Execute(), HcclSim::HcclVmResult::HCCL_SIM_VRT_ERROR_CMD);
}

TEST_F(AivGraphExecutorTest, DISABLED_Execute_UnsupportedTaskType_ReturnsError)
{
    auto task = MakeTaskJson(99, 1, 0, 0, 0, json::object());
    auto block = MakeBlockJson(0, json::array({task}));
    json content = {{"rank", 0}, {"rankSize", 1}, {"launchIndex", 0}, {"aivCores", json::array({block})}};
    WriteTaskFile(0, 0, content);

    AivGraphExecutor executor(0);
    ASSERT_TRUE(executor.Init(0, 0));
    EXPECT_EQ(executor.Execute(), HcclSim::HcclVmResult::HCCL_SIM_VRT_ERROR_CMD);
}

TEST_F(AivGraphExecutorTest, Execute_MixedTasks_Success)
{
    SetupRankResource(0, 4096, 4096, 4096);
    auto ds = MakeDataSlice(3, 0, 64);
    auto memCopy = MakeTaskJson(0, 1, 0, 0, 0, MakeMemCopyPayload(0, 0, ds, ds));
    auto setFlag = MakeTaskJson(2, 2, 0, 0, 0, MakeSetFlagPayload(0, 1, 0));
    auto waitFlag = MakeTaskJson(3, 3, 0, 0, 0, MakeWaitFlagPayload(1, 2, 0));
    auto sendFlag = MakeTaskJson(6, 4, 0, 0, 0, MakeSendFlagPayload(0, 0, 1));
    auto recvFlag = MakeTaskJson(7, 5, 0, 0, 0, MakeRecvFlagPayload(0, 0, 1));
    auto block = MakeBlockJson(0, json::array({memCopy, setFlag, waitFlag, sendFlag, recvFlag}));
    json content = {{"rank", 0}, {"rankSize", 1}, {"launchIndex", 0}, {"aivCores", json::array({block})}};
    WriteTaskFile(0, 0, content);

    AivGraphExecutor executor(0);
    ASSERT_TRUE(executor.Init(0, 0));
    EXPECT_EQ(executor.Execute(), HcclSim::HcclVmResult::HCCL_SIM_SUCCESS);
}

TEST_F(AivGraphExecutorTest, Execute_MultiBlockTasks_Success)
{
    SetupRankResource(0, 4096, 4096, 4096);
    auto ds = MakeDataSlice(3, 0, 64);
    auto task0 = MakeTaskJson(0, 1, 0, 0, 0, MakeMemCopyPayload(0, 0, ds, ds));
    auto task1 = MakeTaskJson(0, 2, 0, 1, 1, MakeMemCopyPayload(0, 0, ds, ds));
    auto block0 = MakeBlockJson(0, json::array({task0}));
    auto block1 = MakeBlockJson(1, json::array(), json::array({task1}));
    json content = {{"rank", 0}, {"rankSize", 1}, {"launchIndex", 0}, {"aivCores", json::array({block0, block1})}};
    WriteTaskFile(0, 0, content);

    AivGraphExecutor executor(0);
    ASSERT_TRUE(executor.Init(0, 0));
    EXPECT_EQ(executor.Execute(), HcclSim::HcclVmResult::HCCL_SIM_SUCCESS);
}

TEST_F(AivGraphExecutorTest, Execute_ReduceWithUBBuffer_Success)
{
    auto srcDs = MakeDataSlice(3, 0, 16);
    auto dstDs = MakeDataSlice(3, 0, 16);
    auto reduce = MakeTaskJson(1, 1, 0, 0, 0, MakeReducePayload(0, 0, srcDs, dstDs, 4, 0));
    auto block = MakeBlockJson(0, json::array({reduce}));
    json content = {{"rank", 0}, {"rankSize", 1}, {"launchIndex", 0}, {"aivCores", json::array({block})}};
    WriteTaskFile(0, 0, content);

    AivGraphExecutor executor(0);
    ASSERT_TRUE(executor.Init(0, 0));
    EXPECT_EQ(executor.Execute(), HcclSim::HcclVmResult::HCCL_SIM_SUCCESS);
}

TEST_F(AivGraphExecutorTest, Execute_MemCopyVerifyData)
{
    SetupRankResource(0, 4096, 4096, 4096);
    auto* inputBuf = static_cast<uint8_t*>(
        const_cast<AivRankResource*>(AivResourceManager::GetInstance().GetRankResource(0))->inputBuffer.realAddr);
    auto* outputBuf = static_cast<uint8_t*>(
        const_cast<AivRankResource*>(AivResourceManager::GetInstance().GetRankResource(0))->outputBuffer.realAddr);
    for (int i = 0; i < 128; ++i) {
        inputBuf[i] = static_cast<uint8_t>(i);
    }

    auto srcDs = MakeDataSlice(0, 0, 128);
    auto dstDs = MakeDataSlice(1, 0, 128);
    auto task = MakeTaskJson(0, 1, 0, 0, 0, MakeMemCopyPayload(0, 0, srcDs, dstDs));
    auto block = MakeBlockJson(0, json::array({task}));
    json content = {{"rank", 0}, {"rankSize", 1}, {"launchIndex", 0}, {"aivCores", json::array({block})}};
    WriteTaskFile(0, 0, content);

    AivGraphExecutor executor(0);
    ASSERT_TRUE(executor.Init(0, 0));
    EXPECT_EQ(executor.Execute(), HcclSim::HcclVmResult::HCCL_SIM_SUCCESS);
    for (int i = 0; i < 128; ++i) {
        EXPECT_EQ(outputBuf[i], static_cast<uint8_t>(i));
    }
}

TEST_F(AivGraphExecutorTest, Execute_ReduceVerifyData)
{
    SetupRankResource(0, 4096, 4096, 4096);
    auto* inputBuf = static_cast<float*>(
        const_cast<AivRankResource*>(AivResourceManager::GetInstance().GetRankResource(0))->inputBuffer.realAddr);
    auto* outputBuf = static_cast<float*>(
        const_cast<AivRankResource*>(AivResourceManager::GetInstance().GetRankResource(0))->outputBuffer.realAddr);
    inputBuf[0] = 1.0f;
    inputBuf[1] = 2.0f;
    outputBuf[0] = 10.0f;
    outputBuf[1] = 20.0f;

    auto srcDs = MakeDataSlice(0, 0, 8);
    auto dstDs = MakeDataSlice(1, 0, 8);
    auto task = MakeTaskJson(1, 1, 0, 0, 0, MakeReducePayload(0, 0, srcDs, dstDs, 4, 0));
    auto block = MakeBlockJson(0, json::array({task}));
    json content = {{"rank", 0}, {"rankSize", 1}, {"launchIndex", 0}, {"aivCores", json::array({block})}};
    WriteTaskFile(0, 0, content);

    AivGraphExecutor executor(0);
    ASSERT_TRUE(executor.Init(0, 0));
    EXPECT_EQ(executor.Execute(), HcclSim::HcclVmResult::HCCL_SIM_SUCCESS);
    EXPECT_FLOAT_EQ(outputBuf[0], 11.0f);
    EXPECT_FLOAT_EQ(outputBuf[1], 22.0f);
}

TEST_F(AivGraphExecutorTest, Init_MultiBlockWithSetWaitFlag)
{
    auto setFlag0 = MakeTaskJson(2, 1, 0, 0, 0, MakeSetFlagPayload(0, 1, 0));
    auto setFlag1 = MakeTaskJson(2, 2, 0, 1, 0, MakeSetFlagPayload(0, 1, 1));
    auto waitFlag0 = MakeTaskJson(3, 3, 0, 0, 0, MakeWaitFlagPayload(1, 2, 0));
    auto waitFlag1 = MakeTaskJson(3, 4, 0, 1, 0, MakeWaitFlagPayload(1, 2, 1));
    auto block0 = MakeBlockJson(0, json::array({setFlag0, waitFlag0}));
    auto block1 = MakeBlockJson(1, json::array({setFlag1, waitFlag1}));
    json content = {{"rank", 0}, {"rankSize", 1}, {"launchIndex", 0}, {"aivCores", json::array({block0, block1})}};
    WriteTaskFile(0, 0, content);

    AivGraphExecutor executor(0);
    ASSERT_TRUE(executor.Init(0, 0));
    EXPECT_EQ(executor.Execute(), HcclSim::HcclVmResult::HCCL_SIM_SUCCESS);
}

TEST_F(AivGraphExecutorTest, Destructor_AfterInit_NoThrow)
{
    auto* executor = new AivGraphExecutor(0);
    json content = {{"rank", 0}, {"rankSize", 1}, {"launchIndex", 0}, {"aivCores", json::array()}};
    WriteTaskFile(0, 0, content);
    executor->Init(0, 0);
    EXPECT_NO_THROW(delete executor);
}

TEST_F(AivGraphExecutorTest, Execute_ReduceMaxVerifyData)
{
    SetupRankResource(0, 4096, 4096, 4096);
    auto* inputBuf = static_cast<int32_t*>(
        const_cast<AivRankResource*>(AivResourceManager::GetInstance().GetRankResource(0))->inputBuffer.realAddr);
    auto* outputBuf = static_cast<int32_t*>(
        const_cast<AivRankResource*>(AivResourceManager::GetInstance().GetRankResource(0))->outputBuffer.realAddr);
    inputBuf[0] = 5;
    outputBuf[0] = 10;

    auto srcDs = MakeDataSlice(0, 0, 4);
    auto dstDs = MakeDataSlice(1, 0, 4);
    auto task = MakeTaskJson(1, 1, 0, 0, 0, MakeReducePayload(0, 0, srcDs, dstDs, 2, 2));
    auto block = MakeBlockJson(0, json::array({task}));
    json content = {{"rank", 0}, {"rankSize", 1}, {"launchIndex", 0}, {"aivCores", json::array({block})}};
    WriteTaskFile(0, 0, content);

    AivGraphExecutor executor(0);
    ASSERT_TRUE(executor.Init(0, 0));
    EXPECT_EQ(executor.Execute(), HcclSim::HcclVmResult::HCCL_SIM_SUCCESS);
    EXPECT_EQ(outputBuf[0], 10);
}

TEST_F(AivGraphExecutorTest, Execute_ReduceMinVerifyData)
{
    SetupRankResource(0, 4096, 4096, 4096);
    auto* inputBuf = static_cast<int32_t*>(
        const_cast<AivRankResource*>(AivResourceManager::GetInstance().GetRankResource(0))->inputBuffer.realAddr);
    auto* outputBuf = static_cast<int32_t*>(
        const_cast<AivRankResource*>(AivResourceManager::GetInstance().GetRankResource(0))->outputBuffer.realAddr);
    inputBuf[0] = 3;
    outputBuf[0] = 7;

    auto srcDs = MakeDataSlice(0, 0, 4);
    auto dstDs = MakeDataSlice(1, 0, 4);
    auto task = MakeTaskJson(1, 1, 0, 0, 0, MakeReducePayload(0, 0, srcDs, dstDs, 2, 3));
    auto block = MakeBlockJson(0, json::array({task}));
    json content = {{"rank", 0}, {"rankSize", 1}, {"launchIndex", 0}, {"aivCores", json::array({block})}};
    WriteTaskFile(0, 0, content);

    AivGraphExecutor executor(0);
    ASSERT_TRUE(executor.Init(0, 0));
    EXPECT_EQ(executor.Execute(), HcclSim::HcclVmResult::HCCL_SIM_SUCCESS);
    EXPECT_EQ(outputBuf[0], 3);
}

TEST_F(AivGraphExecutorTest, Execute_SendRecvFlagVerify)
{
    SetupRankResource(0, 4096, 4096, 4096);
    auto* commInfoBytes = static_cast<uint8_t*>(
        const_cast<AivRankResource*>(AivResourceManager::GetInstance().GetRankResource(0))->aivCommInfoBuffer.realAddr);
    constexpr uint64_t commInfoOffset = AivCommInfoLayout::FLAG2_OFFSET + 5ULL * AivCommInfoLayout::SYNC_CELL_BYTES;

    auto sendFlag = MakeTaskJson(6, 1, 0, 0, 0, MakeSendFlagPayload(0, commInfoOffset, 42));
    auto recvFlag = MakeTaskJson(7, 2, 0, 0, 0, MakeRecvFlagPayload(0, commInfoOffset, 42));
    auto block = MakeBlockJson(0, json::array({sendFlag, recvFlag}));
    json content = {{"rank", 0}, {"rankSize", 1}, {"launchIndex", 0}, {"aivCores", json::array({block})}};
    WriteTaskFile(0, 0, content);

    AivGraphExecutor executor(0);
    ASSERT_TRUE(executor.Init(0, 0));
    EXPECT_EQ(executor.Execute(), HcclSim::HcclVmResult::HCCL_SIM_SUCCESS);
    EXPECT_EQ(*reinterpret_cast<AivSim::flag_t*>(commInfoBytes + commInfoOffset), 42);
}

TEST_F(AivGraphExecutorTest, Execute_MemCopyDstNull_ReturnsError)
{
    SetupRankResource(0, 4096, 4096, 4096);
    auto srcDs = MakeDataSlice(0, 0, 128);
    auto dstDs = MakeDataSlice(0, 0, 128);
    auto task = MakeTaskJson(0, 1, 0, 0, 0, MakeMemCopyPayload(0, 99, srcDs, dstDs));
    auto block = MakeBlockJson(0, json::array({task}));
    json content = {{"rank", 0}, {"rankSize", 1}, {"launchIndex", 0}, {"aivCores", json::array({block})}};
    WriteTaskFile(0, 0, content);

    AivGraphExecutor executor(0);
    ASSERT_TRUE(executor.Init(0, 0));
    EXPECT_EQ(executor.Execute(), HcclSim::HcclVmResult::HCCL_SIM_VRT_ERROR_CMD);
}

TEST_F(AivGraphExecutorTest, Execute_ReduceDstNull_ReturnsError)
{
    SetupRankResource(0, 4096, 4096, 4096);
    auto srcDs = MakeDataSlice(0, 0, 16);
    auto dstDs = MakeDataSlice(0, 0, 16);
    auto task = MakeTaskJson(1, 1, 0, 0, 0, MakeReducePayload(0, 99, srcDs, dstDs, 4, 0));
    auto block = MakeBlockJson(0, json::array({task}));
    json content = {{"rank", 0}, {"rankSize", 1}, {"launchIndex", 0}, {"aivCores", json::array({block})}};
    WriteTaskFile(0, 0, content);

    AivGraphExecutor executor(0);
    ASSERT_TRUE(executor.Init(0, 0));
    EXPECT_EQ(executor.Execute(), HcclSim::HcclVmResult::HCCL_SIM_VRT_ERROR_CMD);
}

TEST_F(AivGraphExecutorTest, Execute_MemCopyWithOffset_Success)
{
    SetupRankResource(0, 4096, 4096, 4096);
    auto* inputBuf = static_cast<uint8_t*>(
        const_cast<AivRankResource*>(AivResourceManager::GetInstance().GetRankResource(0))->inputBuffer.realAddr);
    auto* outputBuf = static_cast<uint8_t*>(
        const_cast<AivRankResource*>(AivResourceManager::GetInstance().GetRankResource(0))->outputBuffer.realAddr);
    inputBuf[100] = 0xAA;

    auto srcDs = MakeDataSlice(0, 100, 1);
    auto dstDs = MakeDataSlice(1, 200, 1);
    auto task = MakeTaskJson(0, 1, 0, 0, 0, MakeMemCopyPayload(0, 0, srcDs, dstDs));
    auto block = MakeBlockJson(0, json::array({task}));
    json content = {{"rank", 0}, {"rankSize", 1}, {"launchIndex", 0}, {"aivCores", json::array({block})}};
    WriteTaskFile(0, 0, content);

    AivGraphExecutor executor(0);
    ASSERT_TRUE(executor.Init(0, 0));
    EXPECT_EQ(executor.Execute(), HcclSim::HcclVmResult::HCCL_SIM_SUCCESS);
    EXPECT_EQ(outputBuf[200], 0xAA);
}

TEST_F(AivGraphExecutorTest, Execute_MemCopyMTE3PipeTasks)
{
    auto ds = MakeDataSlice(3, 0, 64);
    auto task = MakeTaskJson(0, 1, 0, 0, 0, MakeMemCopyPayload(0, 0, ds, ds));
    auto block = MakeBlockJson(0, json::array(), json::array(), json::array({task}));
    json content = {{"rank", 0}, {"rankSize", 1}, {"launchIndex", 0}, {"aivCores", json::array({block})}};
    WriteTaskFile(0, 0, content);

    AivGraphExecutor executor(0);
    ASSERT_TRUE(executor.Init(0, 0));
    EXPECT_EQ(executor.Execute(), HcclSim::HcclVmResult::HCCL_SIM_SUCCESS);
}

TEST_F(AivGraphExecutorTest, Execute_MemCopyMTE2PipeTasks)
{
    auto ds = MakeDataSlice(3, 0, 64);
    auto task = MakeTaskJson(0, 1, 0, 0, 0, MakeMemCopyPayload(0, 0, ds, ds));
    auto block = MakeBlockJson(0, json::array(), json::array({task}));
    json content = {{"rank", 0}, {"rankSize", 1}, {"launchIndex", 0}, {"aivCores", json::array({block})}};
    WriteTaskFile(0, 0, content);

    AivGraphExecutor executor(0);
    ASSERT_TRUE(executor.Init(0, 0));
    EXPECT_EQ(executor.Execute(), HcclSim::HcclVmResult::HCCL_SIM_SUCCESS);
}
