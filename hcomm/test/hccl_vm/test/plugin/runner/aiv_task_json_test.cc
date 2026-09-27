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
#include <cstring>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <memory>
#include <vector>

#include "ai_core_stub.h"
#include "aiv_mode_stub_base.h"
#include "aiv_task.h"
#include "aiv_task_json.h"
#include "ascendc_base_stub.h"

namespace AivSim {
extern nlohmann::json SerializeTaskBase(const AivTask& task);
extern nlohmann::json
SerializeTaskByDynamicType(const AivKernelExecutor& executor, const std::shared_ptr<AivTask>& task);
extern nlohmann::json
SerializeTaskArray(const AivKernelExecutor& executor, const std::vector<std::shared_ptr<AivTask>>& tasks);
extern nlohmann::json SerializeCore(const AivKernelExecutor& executor, const AivCore& core);
extern nlohmann::json SerializeDataSlice(const AivKernelExecutor& executor, const AivDataSlice& slice);
extern nlohmann::json SerializeOpParam(const AivOpParam& opParam);
extern std::string SerializeKernelName(const AivOpParam& opParam);
extern uint64_t ResolveSerializedDataSliceOffset(const AivKernelExecutor& executor, const AivDataSlice& slice);
extern uint64_t ResolveBufferSize(const AivKernelExecutor& executor, bool useCclBuffer);
extern bool ResolveExecutorJsonFilePath(const AivKernelExecutor& executor, uint32_t launchIndex, std::string& filePath);
extern bool WriteJsonAtomically(const nlohmann::json& content, const std::string& filePath);
} // namespace AivSim

using namespace AivSim;
using json = nlohmann::json;
namespace fs = std::filesystem;

static AivOpParam MakeOpParam(const char* kernelName = "test_kernel")
{
    AivOpParam opParam{};
    opParam.dataType = 1;
    opParam.len = 1024;
    opParam.reduceOp = 0;
    opParam.root = 0;
    opParam.sliceId = 1;
    opParam.inputStride = 64;
    opParam.outputStride = 128;
    if (kernelName != nullptr) {
        std::strncpy(opParam.kernelName, kernelName, AIV_OP_KERNEL_NAME_MAX_LEN - 1);
    }
    return opParam;
}

static void InitBasicExecutor(uint32_t rankId = 0)
{
    auto& executor = AivKernelExecutor::GetInstance();
    executor.Reset();
    executor.Init(rankId, 2, 4);

    auto opParam = MakeOpParam();
    executor.SetCurOp(opParam);

    executor.SetIoBuffer(0x1000, 0x100, 0x2000, 0x200, 0x10000, 0x20000);
    executor.SetCommBuffer(0, 0x3000, 0x400, 0x5000, 0x100);
    executor.SetCommBuffer(1, 0x3100, 0x410, 0x5100, 0x110);
}

class AivTaskJsonTest : public testing::Test {
protected:
    void SetUp() override { InitBasicExecutor(); }

    void TearDown() override { AivKernelExecutor::GetInstance().Reset(); }
};

// ========== SerializeKernelName ==========

TEST_F(AivTaskJsonTest, SerializeKernelName_Normal)
{
    auto opParam = MakeOpParam("hello_kernel");
    EXPECT_EQ(SerializeKernelName(opParam), "hello_kernel");
}

TEST_F(AivTaskJsonTest, SerializeKernelName_Empty)
{
    auto opParam = MakeOpParam("");
    EXPECT_EQ(SerializeKernelName(opParam), "");
}

TEST_F(AivTaskJsonTest, SerializeKernelName_Partial)
{
    auto opParam = MakeOpParam();
    opParam.kernelName[0] = 'a';
    opParam.kernelName[1] = 'b';
    opParam.kernelName[2] = '\0';
    opParam.kernelName[3] = 'c';
    EXPECT_EQ(SerializeKernelName(opParam), "ab");
}

// ========== SerializeOpParam ==========

TEST_F(AivTaskJsonTest, SerializeOpParam_AllFields)
{
    auto opParam = MakeOpParam("my_kernel");
    json j = SerializeOpParam(opParam);
    EXPECT_EQ(j["dataType"], 1);
    EXPECT_EQ(j["len"], 1024);
    EXPECT_EQ(j["reduceOp"], 0);
    EXPECT_EQ(j["root"], 0);
    EXPECT_EQ(j["sliceId"], 1);
    EXPECT_EQ(j["inputStride"], 64);
    EXPECT_EQ(j["outputStride"], 128);
    EXPECT_EQ(j["kernelName"], "my_kernel");
}

// ========== ResolveSerializedDataSliceOffset ==========

TEST_F(AivTaskJsonTest, ResolveOffset_InputType)
{
    auto& executor = AivKernelExecutor::GetInstance();
    AivDataSlice slice(AivBufferType::INPUT, 0x50, 0x100);
    uint64_t offset = ResolveSerializedDataSliceOffset(executor, slice);
    EXPECT_EQ(offset, 0x50 + 0x10000);
}

TEST_F(AivTaskJsonTest, ResolveOffset_OutputType)
{
    auto& executor = AivKernelExecutor::GetInstance();
    AivDataSlice slice(AivBufferType::OUTPUT, 0x80, 0x200);
    uint64_t offset = ResolveSerializedDataSliceOffset(executor, slice);
    EXPECT_EQ(offset, 0x80 + 0x20000);
}

TEST_F(AivTaskJsonTest, ResolveOffset_OtherType)
{
    auto& executor = AivKernelExecutor::GetInstance();
    AivDataSlice slice(AivBufferType::CCL, 0xA0, 0x100);
    uint64_t offset = ResolveSerializedDataSliceOffset(executor, slice);
    EXPECT_EQ(offset, 0xA0);
}

// ========== SerializeDataSlice ==========

TEST_F(AivTaskJsonTest, SerializeDataSlice_Input)
{
    auto& executor = AivKernelExecutor::GetInstance();
    AivDataSlice slice(AivBufferType::INPUT, 0x10, 0x80);
    json j = SerializeDataSlice(executor, slice);
    EXPECT_EQ(j["bufferType"], 0);
    EXPECT_EQ(j["bufferTypeName"], "INPUT");
    EXPECT_EQ(j["offset"], 0x10 + 0x10000);
    EXPECT_EQ(j["size"], 0x80);
}

TEST_F(AivTaskJsonTest, SerializeDataSlice_Output)
{
    auto& executor = AivKernelExecutor::GetInstance();
    AivDataSlice slice(AivBufferType::OUTPUT, 0x20, 0x100);
    json j = SerializeDataSlice(executor, slice);
    EXPECT_EQ(j["bufferTypeName"], "OUTPUT");
    EXPECT_EQ(j["offset"], 0x20 + 0x20000);
    EXPECT_EQ(j["size"], 0x100);
}

TEST_F(AivTaskJsonTest, SerializeDataSlice_Ccl)
{
    auto& executor = AivKernelExecutor::GetInstance();
    AivDataSlice slice(AivBufferType::CCL, 0x40, 0x200);
    json j = SerializeDataSlice(executor, slice);
    EXPECT_EQ(j["bufferTypeName"], "CCL");
    EXPECT_EQ(j["offset"], 0x40);
    EXPECT_EQ(j["size"], 0x200);
}

// ========== SerializeTaskBase ==========

TEST_F(AivTaskJsonTest, SerializeTaskBase_AllFields)
{
    AivTask task(AivTaskType::MEM_COPY, 42, 0, 1, AscendC::PIPE_MTE2);
    json j = SerializeTaskBase(task);
    EXPECT_EQ(j["taskType"], 0);
    EXPECT_EQ(j["taskTypeName"], "MemCopy");
    EXPECT_EQ(j["taskId"], 42);
    EXPECT_EQ(j["rankId"], 0);
    EXPECT_EQ(j["blockId"], 1);
    EXPECT_EQ(j["curPipe"], 1);
    EXPECT_EQ(j["curPipeName"], "PIPE_MTE2");
}

TEST_F(AivTaskJsonTest, SerializeTaskBase_ReduceType)
{
    AivTask task(AivTaskType::REDUCE, 10, 1, 3, AscendC::PIPE_S);
    json j = SerializeTaskBase(task);
    EXPECT_EQ(j["taskTypeName"], "Reduce");
    EXPECT_EQ(j["curPipeName"], "PIPE_SCALAR");
}

TEST_F(AivTaskJsonTest, SerializeTaskBase_DefaultCtor)
{
    AivTask task(AivTaskType::INVALID_TYPE);
    json j = SerializeTaskBase(task);
    EXPECT_EQ(j["taskType"], static_cast<uint32_t>(AivTaskType::INVALID_TYPE));
    EXPECT_EQ(j["taskTypeName"], "UnknownTask");
    EXPECT_EQ(j["taskId"], UINT32_MAX);
}

// ========== SerializeTaskByDynamicType ==========

TEST_F(AivTaskJsonTest, DynamicType_NullTask)
{
    auto& executor = AivKernelExecutor::GetInstance();
    json j = SerializeTaskByDynamicType(executor, nullptr);
    EXPECT_TRUE(j.is_null());
}

TEST_F(AivTaskJsonTest, DynamicType_MemCopy)
{
    auto& executor = AivKernelExecutor::GetInstance();
    AivDataSlice src(AivBufferType::INPUT, 0x0, 0x100);
    AivDataSlice dst(AivBufferType::OUTPUT, 0x0, 0x100);
    auto task = std::make_shared<AivTaskMemCopy>(0, src, 1, dst);

    json j = SerializeTaskByDynamicType(executor, task);
    EXPECT_EQ(j["taskTypeName"], "MemCopy");
    EXPECT_EQ(j["payload"]["srcRank"], 0);
    EXPECT_EQ(j["payload"]["dstRank"], 1);
    EXPECT_EQ(j["payload"]["src"]["bufferTypeName"], "INPUT");
    EXPECT_EQ(j["payload"]["dst"]["bufferTypeName"], "OUTPUT");
}

TEST_F(AivTaskJsonTest, DynamicType_Reduce)
{
    auto& executor = AivKernelExecutor::GetInstance();
    AivDataSlice src(AivBufferType::INPUT, 0x0, 0x80);
    AivDataSlice dst(AivBufferType::OUTPUT, 0x0, 0x80);
    auto task = std::make_shared<AivTaskReduce>(0, src, 1, dst, 1, 0);

    json j = SerializeTaskByDynamicType(executor, task);
    EXPECT_EQ(j["taskTypeName"], "Reduce");
    EXPECT_EQ(j["payload"]["dataType"], 1);
    EXPECT_EQ(j["payload"]["reduceOp"], 0);
    EXPECT_EQ(j["payload"]["reduceOpName"], "SUM");
}

TEST_F(AivTaskJsonTest, DynamicType_SetFlag)
{
    auto& executor = AivKernelExecutor::GetInstance();
    auto task = std::make_shared<AivTaskSetFlag>(AscendC::PIPE_S, AscendC::PIPE_MTE2, 5);

    json j = SerializeTaskByDynamicType(executor, task);
    EXPECT_EQ(j["taskTypeName"], "SetFlag");
    EXPECT_EQ(j["payload"]["srcPipe"], 0);
    EXPECT_EQ(j["payload"]["srcPipeName"], "PIPE_SCALAR");
    EXPECT_EQ(j["payload"]["dstPipe"], 1);
    EXPECT_EQ(j["payload"]["dstPipeName"], "PIPE_MTE2");
    EXPECT_EQ(j["payload"]["eventId"], 5);
}

TEST_F(AivTaskJsonTest, DynamicType_WaitFlag)
{
    auto& executor = AivKernelExecutor::GetInstance();
    auto task = std::make_shared<AivTaskWaitFlag>(AscendC::PIPE_MTE2, AscendC::PIPE_MTE3, 3);

    json j = SerializeTaskByDynamicType(executor, task);
    EXPECT_EQ(j["taskTypeName"], "WaitFlag");
    EXPECT_EQ(j["payload"]["srcPipe"], 1);
    EXPECT_EQ(j["payload"]["dstPipe"], 2);
    EXPECT_EQ(j["payload"]["eventId"], 3);
}

TEST_F(AivTaskJsonTest, DynamicType_PipeBarrier)
{
    auto& executor = AivKernelExecutor::GetInstance();
    auto task = std::make_shared<AivTaskPipeBarrier>(AscendC::PIPE_MTE2);

    json j = SerializeTaskByDynamicType(executor, task);
    EXPECT_EQ(j["taskTypeName"], "PipeBarrier");
    EXPECT_EQ(j["payload"]["pipeType"], 1);
    EXPECT_EQ(j["payload"]["pipeTypeName"], "PIPE_MTE2");
    EXPECT_TRUE(j["payload"]["barrierGroupTaskIds"].is_array());
}

TEST_F(AivTaskJsonTest, DynamicType_SyncAll)
{
    auto& executor = AivKernelExecutor::GetInstance();
    auto task = std::make_shared<AivTaskSyncAll>(3);

    json j = SerializeTaskByDynamicType(executor, task);
    EXPECT_EQ(j["taskTypeName"], "SyncAll");
    EXPECT_EQ(j["payload"]["syncRound"], 3);
}

TEST_F(AivTaskJsonTest, DynamicType_SendFlag)
{
    auto& executor = AivKernelExecutor::GetInstance();
    auto task = std::make_shared<AivTaskSendFlag>(1, 0x200, 99);

    json j = SerializeTaskByDynamicType(executor, task);
    EXPECT_EQ(j["taskTypeName"], "SendFlag");
    EXPECT_EQ(j["payload"]["rank"], 1);
    EXPECT_EQ(j["payload"]["commInfoOffset"], 0x200);
    EXPECT_EQ(j["payload"]["flagValue"], 99);
}

TEST_F(AivTaskJsonTest, DynamicType_RecvFlag)
{
    auto& executor = AivKernelExecutor::GetInstance();
    auto task = std::make_shared<AivTaskRecvFlag>(2, 0x300, 88);

    json j = SerializeTaskByDynamicType(executor, task);
    EXPECT_EQ(j["taskTypeName"], "RecvFlag");
    EXPECT_EQ(j["payload"]["rank"], 2);
    EXPECT_EQ(j["payload"]["commInfoOffset"], 0x300);
    EXPECT_EQ(j["payload"]["targetValue"], 88);
}

// ========== SerializeTaskArray ==========

TEST_F(AivTaskJsonTest, SerializeTaskArray_Empty)
{
    auto& executor = AivKernelExecutor::GetInstance();
    std::vector<std::shared_ptr<AivTask>> tasks;
    json j = SerializeTaskArray(executor, tasks);
    EXPECT_TRUE(j.is_array());
    EXPECT_EQ(j.size(), 0);
}

TEST_F(AivTaskJsonTest, SerializeTaskArray_WithNullsAndTasks)
{
    auto& executor = AivKernelExecutor::GetInstance();
    std::vector<std::shared_ptr<AivTask>> tasks;
    tasks.push_back(std::make_shared<AivTaskSendFlag>(1, 0x100, 11));
    tasks.push_back(nullptr);
    tasks.push_back(std::make_shared<AivTaskRecvFlag>(2, 0x200, 22));

    json j = SerializeTaskArray(executor, tasks);
    EXPECT_EQ(j.size(), 3);
    EXPECT_EQ(j[0]["taskTypeName"], "SendFlag");
    EXPECT_EQ(j[0]["payload"]["flagValue"], 11);
    EXPECT_TRUE(j[1].is_null()); // null task produces null
    EXPECT_EQ(j[2]["taskTypeName"], "RecvFlag");
    EXPECT_EQ(j[2]["payload"]["targetValue"], 22);
}

// ========== SerializeCore ==========

TEST_F(AivTaskJsonTest, SerializeCore_AllFields)
{
    auto& executor = AivKernelExecutor::GetInstance();
    AivCore core(3);

    auto scalarTask = std::make_shared<AivTaskSetFlag>(AscendC::PIPE_S, AscendC::PIPE_MTE2, 10);
    core.AppendScalar(scalarTask);

    auto mte2Task = std::make_shared<AivTaskMemCopy>(
        0, AivDataSlice(AivBufferType::INPUT, 0, 16), 1, AivDataSlice(AivBufferType::OUTPUT, 0, 16));
    core.AppendMTE2(mte2Task);

    json j = SerializeCore(executor, core);
    EXPECT_EQ(j["blockIdx"], 3);
    EXPECT_EQ(j["scalarTasks"].size(), 1);
    EXPECT_EQ(j["scalarTasks"][0]["taskTypeName"], "SetFlag");
    EXPECT_EQ(j["mte2Tasks"].size(), 1);
    EXPECT_EQ(j["mte2Tasks"][0]["taskTypeName"], "MemCopy");
    EXPECT_EQ(j["mte3Tasks"].size(), 0);
}

TEST_F(AivTaskJsonTest, SerializeCore_NoTasks)
{
    auto& executor = AivKernelExecutor::GetInstance();
    AivCore core(0);
    json j = SerializeCore(executor, core);
    EXPECT_EQ(j["blockIdx"], 0);
    EXPECT_EQ(j["scalarTasks"].size(), 0);
    EXPECT_EQ(j["mte2Tasks"].size(), 0);
    EXPECT_EQ(j["mte3Tasks"].size(), 0);
}

// ========== ResolveBufferSize ==========

TEST_F(AivTaskJsonTest, ResolveBufferSize_CclFound)
{
    auto& executor = AivKernelExecutor::GetInstance();
    uint64_t size = ResolveBufferSize(executor, true);
    EXPECT_EQ(size, 0x400);
}

TEST_F(AivTaskJsonTest, ResolveBufferSize_AivCommInfoFound)
{
    auto& executor = AivKernelExecutor::GetInstance();
    uint64_t size = ResolveBufferSize(executor, false);
    EXPECT_EQ(size, 0x100);
}

TEST_F(AivTaskJsonTest, ResolveBufferSize_CclZero)
{
    auto& executor = AivKernelExecutor::GetInstance();
    executor.Reset();
    executor.Init(1, 1, 1);
    executor.SetCommBuffer(0, 0, 0, 0, 0);
    uint64_t size = ResolveBufferSize(executor, true);
    EXPECT_EQ(size, 0);
}

TEST_F(AivTaskJsonTest, ResolveBufferSize_AivCommInfoZero)
{
    auto& executor = AivKernelExecutor::GetInstance();
    executor.Reset();
    executor.Init(0, 1, 1);
    executor.SetCommBuffer(0, 0x1000, 0, 0x1000, 0);
    uint64_t size = ResolveBufferSize(executor, false);
    EXPECT_EQ(size, 0);
}

// ========== SerializeExecutor ==========

TEST_F(AivTaskJsonTest, SerializeExecutor_Basic)
{
    auto& executor = AivKernelExecutor::GetInstance();
    json j = SerializeExecutor(executor, 0);
    EXPECT_EQ(j["mode"], "aiv");
    EXPECT_EQ(j["rank"], 0);
    EXPECT_EQ(j["launchIndex"], 0);
    EXPECT_EQ(j["rankSize"], 4);
    EXPECT_EQ(j["inputGlobalOffsetBase"], 0x10000);
    EXPECT_EQ(j["outputGlobalOffsetBase"], 0x20000);
    EXPECT_EQ(j["inBufferSize"], 0x100);
    EXPECT_EQ(j["outBufferSize"], 0x200);
    EXPECT_EQ(j["cclBufferSize"], 0x400);
    EXPECT_EQ(j["aivCommInfoSize"], 0x100);
    EXPECT_EQ(j["ubBufferSize"], 190 * 1024);
    EXPECT_EQ(j["curOp"]["kernelName"], "test_kernel");
    EXPECT_EQ(j["aivCores"].size(), 2);
}

TEST_F(AivTaskJsonTest, SerializeExecutor_WithTasks)
{
    auto& executor = AivKernelExecutor::GetInstance();
    auto core = executor.GetAivCore(0);
    ASSERT_NE(core, nullptr);

    core->AppendScalar(std::make_shared<AivTaskSetFlag>(AscendC::PIPE_S, AscendC::PIPE_MTE2, 1));
    core->AppendMTE2(std::make_shared<AivTaskMemCopy>(
        0, AivDataSlice(AivBufferType::INPUT, 0, 64), 1, AivDataSlice(AivBufferType::OUTPUT, 0, 64)));

    core = executor.GetAivCore(1);
    ASSERT_NE(core, nullptr);
    core->AppendMTE3(std::make_shared<AivTaskSendFlag>(2, 0x200, 7));

    json j = SerializeExecutor(executor, 5);
    EXPECT_EQ(j["launchIndex"], 5);
    EXPECT_EQ(j["aivCores"].size(), 2);
    EXPECT_EQ(j["aivCores"][0]["scalarTasks"].size(), 1);
    EXPECT_EQ(j["aivCores"][0]["mte2Tasks"].size(), 1);
    EXPECT_EQ(j["aivCores"][1]["mte3Tasks"].size(), 1);
    EXPECT_EQ(j["aivCores"][1]["mte3Tasks"][0]["taskTypeName"], "SendFlag");
}

// ========== ResolveExecutorJsonFilePath ==========

TEST_F(AivTaskJsonTest, ResolveExecutorJsonFilePath_Success)
{
    setenv("HCCL_VM_INSTALL_ROOT", "/tmp/hccl_test", 1);
    auto& executor = AivKernelExecutor::GetInstance();
    std::string filePath;
    bool result = ResolveExecutorJsonFilePath(executor, 3, filePath);
    EXPECT_TRUE(result);
    EXPECT_NE(filePath.find("hcclvm_aiv_rank0_launch3_task.json"), std::string::npos);
    unsetenv("HCCL_VM_INSTALL_ROOT");
}

// ========== WriteJsonAtomically ==========

TEST_F(AivTaskJsonTest, WriteJsonAtomically_Success)
{
    std::error_code ec;
    json content = {{"test", "value"}};
    std::string filePath = "/tmp/aiv_json_test_output.json";
    fs::remove(filePath, ec);

    bool result = WriteJsonAtomically(content, filePath);
    EXPECT_TRUE(result);
    EXPECT_TRUE(fs::exists(filePath));

    std::ifstream ifs(filePath);
    std::string readContent((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    EXPECT_NE(readContent.find("\"test\""), std::string::npos);
    EXPECT_NE(readContent.find("\"value\""), std::string::npos);

    fs::remove(filePath, ec);
}

TEST_F(AivTaskJsonTest, WriteJsonAtomically_NestedDir)
{
    std::error_code ec;
    json content = {{"nested", true}};
    std::string filePath = "/tmp/aiv_test_dir/sub/dir/output.json";
    fs::remove_all("/tmp/aiv_test_dir", ec);

    bool result = WriteJsonAtomically(content, filePath);
    EXPECT_TRUE(result);

    fs::remove_all("/tmp/aiv_test_dir", ec);
}

TEST_F(AivTaskJsonTest, WriteJsonAtomically_RenameFallback)
{
    std::error_code ec;
    json content = {{"fallback", "test"}};
    // Ensure target exists so rename may fail (platform-dependent)
    std::string filePath = "/tmp/aiv_json_rename_to_existing.json";
    std::string tempPath = filePath + ".tmp";
    fs::remove(filePath, ec);
    fs::remove(tempPath, ec);

    json firstContent = {{"first", "write"}};
    WriteJsonAtomically(firstContent, filePath);
    EXPECT_TRUE(fs::exists(filePath));

    // Write again - renames over existing
    bool result = WriteJsonAtomically(content, filePath);
    EXPECT_TRUE(result);

    fs::remove(filePath, ec);
    fs::remove(tempPath, ec);
}

// ========== DumpExecutorToJsonFile ==========

TEST_F(AivTaskJsonTest, DumpExecutorToJsonFile_Success)
{
    std::error_code ec;
    auto& executor = AivKernelExecutor::GetInstance();
    auto core = executor.GetAivCore(0);
    ASSERT_NE(core, nullptr);
    core->AppendScalar(std::make_shared<AivTaskSetFlag>(AscendC::PIPE_S, AscendC::PIPE_MTE2, 42));

    std::string expectedPath;
    ASSERT_TRUE(ResolveExecutorJsonFilePath(executor, 0, expectedPath));
    fs::remove(expectedPath, ec);

    bool result = DumpExecutorToJsonFile(executor, 0);
    EXPECT_TRUE(result);

    EXPECT_TRUE(fs::exists(expectedPath));

    std::ifstream ifs(expectedPath);
    std::string content((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    EXPECT_NE(content.find("\"mode\""), std::string::npos);
    EXPECT_NE(content.find("\"aiv\""), std::string::npos);

    fs::remove(expectedPath, ec);
}

// ========== aiv_task.h coverage: AivTask base ==========

TEST_F(AivTaskJsonTest, AivTask_GetUUID)
{
    AivTask task(AivTaskType::MEM_COPY, 0xABCD, 0x12, 0, AscendC::PIPE_S);
    uint64_t uuid = task.GetUUID();
    EXPECT_EQ(uuid, (static_cast<uint64_t>(0x12) << 32) | 0xABCD);
}

TEST_F(AivTaskJsonTest, AivTask_GetUUID_Zero)
{
    AivTask task(AivTaskType::SET_FLAG, 0, 0, 0, AscendC::PIPE_MTE2);
    uint64_t uuid = task.GetUUID();
    EXPECT_EQ(uuid, 0);
}

TEST_F(AivTaskJsonTest, AivTask_Describe)
{
    AivTask task(AivTaskType::MEM_COPY, 42, 0, 1, AscendC::PIPE_MTE2);
    std::string desc = task.Describe();
    EXPECT_NE(desc.find("[MemCopy]"), std::string::npos);
    EXPECT_NE(desc.find("RankId=0"), std::string::npos);
    EXPECT_NE(desc.find("TaskId=42"), std::string::npos);
    EXPECT_NE(desc.find("BlockId=1"), std::string::npos);
    EXPECT_NE(desc.find("PIPE_MTE2"), std::string::npos);
}

TEST_F(AivTaskJsonTest, AivTask_SetTaskType)
{
    AivTask task(AivTaskType::INVALID_TYPE);
    EXPECT_EQ(task.GetTaskType(), AivTaskType::INVALID_TYPE);
    task.SetTaskType(AivTaskType::REDUCE);
    EXPECT_EQ(task.GetTaskType(), AivTaskType::REDUCE);
}

TEST_F(AivTaskJsonTest, AivTask_SetTaskId)
{
    AivTask task(AivTaskType::SET_FLAG);
    task.SetTaskId(99);
    EXPECT_EQ(task.GetTaskId(), 99);
}

TEST_F(AivTaskJsonTest, AivTask_SetRankId)
{
    AivTask task(AivTaskType::SET_FLAG);
    task.SetRankId(3);
    EXPECT_EQ(task.GetRankId(), 3);
}

TEST_F(AivTaskJsonTest, AivTask_SetBlockId)
{
    AivTask task(AivTaskType::SET_FLAG);
    task.SetBlockId(5);
    EXPECT_EQ(task.GetBlockId(), 5);
}

TEST_F(AivTaskJsonTest, AivTask_SetCurPipe)
{
    AivTask task(AivTaskType::SET_FLAG);
    task.SetCurPipe(AscendC::PIPE_MTE3);
    EXPECT_EQ(task.GetCurPipe(), AscendC::PIPE_MTE3);
}

// ========== aiv_task.h coverage: AivTaskMemCopy ==========

TEST_F(AivTaskJsonTest, MemCopy_Describe)
{
    AivDataSlice src(AivBufferType::INPUT, 0, 64);
    AivDataSlice dst(AivBufferType::OUTPUT, 0, 64);
    AivTaskMemCopy task(0, src, 1, dst);
    task.SetTaskId(10);
    std::string desc = task.Describe();
    EXPECT_NE(desc.find("[MemCopy]"), std::string::npos);
    EXPECT_NE(desc.find("SrcRank=0"), std::string::npos);
    EXPECT_NE(desc.find("DstRank=1"), std::string::npos);
}

TEST_F(AivTaskJsonTest, MemCopy_SetSrcRank)
{
    AivDataSlice src, dst;
    AivTaskMemCopy task(0, src, 0, dst);
    task.SetSrcRank(5);
    EXPECT_EQ(task.GetSrcRank(), 5);
}

TEST_F(AivTaskJsonTest, MemCopy_SetSrc)
{
    AivDataSlice src, dst;
    AivTaskMemCopy task(0, src, 0, dst);
    AivDataSlice newSrc(AivBufferType::CCL, 0x100, 0x80);
    task.SetSrc(newSrc);
    EXPECT_EQ(task.GetSrc().GetType(), AivBufferType::CCL);
    EXPECT_EQ(task.GetSrc().GetOffset(), 0x100);
}

TEST_F(AivTaskJsonTest, MemCopy_SetDstRank)
{
    AivDataSlice src, dst;
    AivTaskMemCopy task(0, src, 0, dst);
    task.SetDstRank(3);
    EXPECT_EQ(task.GetDstRank(), 3);
}

TEST_F(AivTaskJsonTest, MemCopy_SetDst)
{
    AivDataSlice src, dst;
    AivTaskMemCopy task(0, src, 0, dst);
    AivDataSlice newDst(AivBufferType::UB, 0x200, 0x40);
    task.SetDst(newDst);
    EXPECT_EQ(task.GetDst().GetType(), AivBufferType::UB);
    EXPECT_EQ(task.GetDst().GetOffset(), 0x200);
}

// ========== aiv_task.h coverage: AivTaskReduce ==========

TEST_F(AivTaskJsonTest, Reduce_Describe)
{
    AivDataSlice src(AivBufferType::INPUT, 0, 32);
    AivDataSlice dst(AivBufferType::OUTPUT, 0, 32);
    AivTaskReduce task(0, src, 1, dst, 2, 1); // dataType=2, reduceOp=1(PROD)
    task.SetTaskId(20);
    std::string desc = task.Describe();
    EXPECT_NE(desc.find("[Reduce]"), std::string::npos);
    EXPECT_NE(desc.find("DataType=2"), std::string::npos);
    EXPECT_NE(desc.find("PROD"), std::string::npos);
}

TEST_F(AivTaskJsonTest, Reduce_SetSrcRank)
{
    AivDataSlice src, dst;
    AivTaskReduce task(0, src, 0, dst, 0, 0);
    task.SetSrcRank(7);
    EXPECT_EQ(task.GetSrcRank(), 7);
}

TEST_F(AivTaskJsonTest, Reduce_SetSrc)
{
    AivDataSlice src, dst;
    AivTaskReduce task(0, src, 0, dst, 0, 0);
    AivDataSlice newSrc(AivBufferType::AIV_COMM, 0x500, 0x10);
    task.SetSrc(newSrc);
    EXPECT_EQ(task.GetSrc().GetType(), AivBufferType::AIV_COMM);
}

TEST_F(AivTaskJsonTest, Reduce_SetDstRank)
{
    AivDataSlice src, dst;
    AivTaskReduce task(0, src, 0, dst, 0, 0);
    task.SetDstRank(6);
    EXPECT_EQ(task.GetDstRank(), 6);
}

TEST_F(AivTaskJsonTest, Reduce_SetDst)
{
    AivDataSlice src, dst;
    AivTaskReduce task(0, src, 0, dst, 0, 0);
    AivDataSlice newDst(AivBufferType::CCL, 0x300, 0x60);
    task.SetDst(newDst);
    EXPECT_EQ(task.GetDst().GetType(), AivBufferType::CCL);
}

TEST_F(AivTaskJsonTest, Reduce_SetDataType)
{
    AivDataSlice src, dst;
    AivTaskReduce task(0, src, 0, dst, 0, 0);
    task.SetDataType(3);
    EXPECT_EQ(task.GetDataType(), 3);
}

TEST_F(AivTaskJsonTest, Reduce_SetReduceOp)
{
    AivDataSlice src, dst;
    AivTaskReduce task(0, src, 0, dst, 0, 0);
    task.SetReduceOp(2); // MAX
    EXPECT_EQ(task.GetReduceOp(), 2);
}

// ========== aiv_task.h coverage: AivTaskSetFlag ==========

TEST_F(AivTaskJsonTest, SetFlag_Describe)
{
    AivTaskSetFlag task(AscendC::PIPE_MTE2, AscendC::PIPE_MTE3, 7);
    task.SetTaskId(30);
    std::string desc = task.Describe();
    EXPECT_NE(desc.find("[SetFlag]"), std::string::npos);
    EXPECT_NE(desc.find("PIPE_MTE2"), std::string::npos);
    EXPECT_NE(desc.find("PIPE_MTE3"), std::string::npos);
    EXPECT_NE(desc.find("EventId=7"), std::string::npos);
}

TEST_F(AivTaskJsonTest, SetFlag_SetSrcPipe)
{
    AivTaskSetFlag task(AscendC::PIPE_S, AscendC::PIPE_S, 0);
    task.SetSrcPipe(AscendC::PIPE_MTE3);
    EXPECT_EQ(task.GetSrcPipe(), AscendC::PIPE_MTE3);
}

TEST_F(AivTaskJsonTest, SetFlag_SetDstPipe)
{
    AivTaskSetFlag task(AscendC::PIPE_S, AscendC::PIPE_S, 0);
    task.SetDstPipe(AscendC::PIPE_MTE2);
    EXPECT_EQ(task.GetDstPipe(), AscendC::PIPE_MTE2);
}

TEST_F(AivTaskJsonTest, SetFlag_SetEventId)
{
    AivTaskSetFlag task(AscendC::PIPE_S, AscendC::PIPE_S, 0);
    task.SetEventId(42);
    EXPECT_EQ(task.GetEventId(), 42);
}

// ========== aiv_task.h coverage: AivTaskWaitFlag ==========

TEST_F(AivTaskJsonTest, WaitFlag_Describe)
{
    AivTaskWaitFlag task(AscendC::PIPE_MTE3, AscendC::PIPE_S, 4);
    task.SetTaskId(40);
    std::string desc = task.Describe();
    EXPECT_NE(desc.find("[WaitFlag]"), std::string::npos);
    EXPECT_NE(desc.find("PIPE_MTE3"), std::string::npos);
    EXPECT_NE(desc.find("PIPE_SCALAR"), std::string::npos);
    EXPECT_NE(desc.find("EventId=4"), std::string::npos);
}

TEST_F(AivTaskJsonTest, WaitFlag_SetSrcPipe)
{
    AivTaskWaitFlag task(AscendC::PIPE_S, AscendC::PIPE_S, 0);
    task.SetSrcPipe(AscendC::PIPE_MTE2);
    EXPECT_EQ(task.GetSrcPipe(), AscendC::PIPE_MTE2);
}

TEST_F(AivTaskJsonTest, WaitFlag_SetDstPipe)
{
    AivTaskWaitFlag task(AscendC::PIPE_S, AscendC::PIPE_S, 0);
    task.SetDstPipe(AscendC::PIPE_MTE3);
    EXPECT_EQ(task.GetDstPipe(), AscendC::PIPE_MTE3);
}

TEST_F(AivTaskJsonTest, WaitFlag_SetEventId)
{
    AivTaskWaitFlag task(AscendC::PIPE_S, AscendC::PIPE_S, 0);
    task.SetEventId(88);
    EXPECT_EQ(task.GetEventId(), 88);
}

// ========== aiv_task.h coverage: AivTaskPipeBarrier ==========

TEST_F(AivTaskJsonTest, PipeBarrier_Describe)
{
    AivTaskPipeBarrier task(AscendC::PIPE_MTE2);
    task.SetTaskId(50);

    auto sub1 = std::make_shared<AivTaskPipeBarrier>(AscendC::PIPE_MTE2);
    sub1->SetTaskId(100);
    auto sub2 = std::make_shared<AivTaskPipeBarrier>(AscendC::PIPE_MTE2);
    sub2->SetTaskId(200);
    task.AddBarrierGroup(sub1);
    task.AddBarrierGroup(sub2);

    std::string desc = task.Describe();
    EXPECT_NE(desc.find("[PipeBarrier]"), std::string::npos);
    EXPECT_NE(desc.find("PIPE_MTE2"), std::string::npos);
    EXPECT_NE(desc.find("BarrierGroup=[100,200]"), std::string::npos);
}

TEST_F(AivTaskJsonTest, PipeBarrier_SetPipeType)
{
    AivTaskPipeBarrier task(AscendC::PIPE_S);
    task.SetPipeType(AscendC::PIPE_MTE3);
    EXPECT_EQ(task.GetPipeType(), AscendC::PIPE_MTE3);
}

TEST_F(AivTaskJsonTest, PipeBarrier_AddBarrierGroup)
{
    AivTaskPipeBarrier task(AscendC::PIPE_MTE3);
    auto sub = std::make_shared<AivTaskPipeBarrier>(AscendC::PIPE_MTE3);
    sub->SetTaskId(77);
    task.AddBarrierGroup(sub);
    EXPECT_EQ(task.GetBarrierGroup().size(), 1);
    EXPECT_EQ(task.GetBarrierGroup()[0]->GetTaskId(), 77);
}

// ========== aiv_task.h coverage: AivTaskSyncAll ==========

TEST_F(AivTaskJsonTest, SyncAll_Describe)
{
    AivTaskSyncAll task(2);
    task.SetTaskId(60);
    std::string desc = task.Describe();
    EXPECT_NE(desc.find("[SyncAll]"), std::string::npos);
    EXPECT_NE(desc.find("SyncRound=2"), std::string::npos);
}

// ========== aiv_task.h coverage: AivTaskSendFlag ==========

TEST_F(AivTaskJsonTest, SendFlag_Describe)
{
    AivTaskSendFlag task(2, 0x400, -1);
    task.SetTaskId(90);
    std::string desc = task.Describe();
    EXPECT_NE(desc.find("[SendFlag]"), std::string::npos);
    EXPECT_NE(desc.find("Rank=2"), std::string::npos);
    EXPECT_NE(desc.find("CommInfoOffset=1024"), std::string::npos);
    EXPECT_NE(desc.find("Value=-1"), std::string::npos);
}

TEST_F(AivTaskJsonTest, SendFlag_SetRank)
{
    AivTaskSendFlag task(0, 0, 0);
    task.SetRank(4);
    EXPECT_EQ(task.GetRank(), 4);
}

TEST_F(AivTaskJsonTest, SendFlag_SetCommInfoOffset)
{
    AivTaskSendFlag task(0, 0, 0);
    task.SetCommInfoOffset(0x800);
    EXPECT_EQ(task.GetCommInfoOffset(), 0x800);
}

TEST_F(AivTaskJsonTest, SendFlag_SetFlagValue)
{
    AivTaskSendFlag task(0, 0, 0);
    task.SetFlagValue(42);
    EXPECT_EQ(task.GetFlagValue(), 42);
}

// ========== aiv_task.h coverage: AivTaskRecvFlag ==========

TEST_F(AivTaskJsonTest, RecvFlag_Describe)
{
    AivTaskRecvFlag task(3, 0x600, 77);
    task.SetTaskId(100);
    std::string desc = task.Describe();
    EXPECT_NE(desc.find("[RecvFlag]"), std::string::npos);
    EXPECT_NE(desc.find("Rank=3"), std::string::npos);
    EXPECT_NE(desc.find("CommInfoOffset=1536"), std::string::npos);
    EXPECT_NE(desc.find("Value=77"), std::string::npos);
}

TEST_F(AivTaskJsonTest, RecvFlag_SetRank)
{
    AivTaskRecvFlag task(0, 0, 0);
    task.SetRank(5);
    EXPECT_EQ(task.GetRank(), 5);
}

TEST_F(AivTaskJsonTest, RecvFlag_SetCommInfoOffset)
{
    AivTaskRecvFlag task(0, 0, 0);
    task.SetCommInfoOffset(0x900);
    EXPECT_EQ(task.GetCommInfoOffset(), 0x900);
}

TEST_F(AivTaskJsonTest, RecvFlag_SetTargetValue)
{
    AivTaskRecvFlag task(0, 0, 0);
    task.SetTargetValue(33);
    EXPECT_EQ(task.GetTargetValue(), 33);
}
