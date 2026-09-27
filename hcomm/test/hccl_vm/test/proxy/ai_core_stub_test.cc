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
#include <gtest/gtest.h>

#include "ai_core_stub.h"
#include "aiv_task.h"
#include "ascendc_base_stub.h"

using namespace AivSim;

namespace AivSim {
extern bool AivBufferContains(const Mem& buffer, uint64_t addr, uint64_t size);
extern bool AivBufferMatch(
    uint64_t addr, uint64_t size, const Mem& buffer, AivBufferType type, RankId rank, AivDataSlice& slice,
    RankId* matchedRank);
} // namespace AivSim

class AiCoreStubTest : public testing::Test {
protected:
    void SetUp() override { AivKernelExecutor::GetInstance().Reset(); }

    void TearDown() override { AivKernelExecutor::GetInstance().Reset(); }
};

TEST_F(AiCoreStubTest, GetInstance_ReturnsSameInstance)
{
    auto& inst1 = AivKernelExecutor::GetInstance();
    auto& inst2 = AivKernelExecutor::GetInstance();
    EXPECT_EQ(&inst1, &inst2);
}

TEST_F(AiCoreStubTest, Init_NormalParams_SetsFields)
{
    AivKernelExecutor::GetInstance().Init(0, 4, 8);
    EXPECT_EQ(AivKernelExecutor::GetInstance().GetRankId(), 0u);
    EXPECT_EQ(AivKernelExecutor::GetInstance().GetRankSize(), 8u);
    EXPECT_EQ(AivKernelExecutor::GetInstance().GetBlockNum(), 4);
}

TEST_F(AiCoreStubTest, Init_RankSizeExceedsMax_Clamped)
{
    AivKernelExecutor::GetInstance().Init(1, 2, MAX_RANK_NUM + 1);
    EXPECT_EQ(AivKernelExecutor::GetInstance().GetRankSize(), MAX_RANK_NUM);
}

TEST_F(AiCoreStubTest, Init_ZeroRankSize_NoClamp)
{
    AivKernelExecutor::GetInstance().Init(0, 2, 0);
    EXPECT_EQ(AivKernelExecutor::GetInstance().GetRankSize(), 0u);
}

TEST_F(AiCoreStubTest, Init_BlockNumZero_NoCores)
{
    AivKernelExecutor::GetInstance().Init(0, 0, 4);
    EXPECT_EQ(AivKernelExecutor::GetInstance().GetBlockNum(), 0);
}

TEST_F(AiCoreStubTest, Reset_ClearsAllFields)
{
    AivKernelExecutor::GetInstance().Init(3, 4, 8);
    AivKernelExecutor::GetInstance().SetIoBuffer(0x1000, 1024, 0x2000, 2048, 0, 0);
    AivKernelExecutor::GetInstance().Reset();
    EXPECT_EQ(AivKernelExecutor::GetInstance().GetRankId(), UINT32_MAX);
    EXPECT_EQ(AivKernelExecutor::GetInstance().GetRankSize(), 0u);
    EXPECT_EQ(AivKernelExecutor::GetInstance().GetBlockNum(), 0);
    EXPECT_EQ(AivKernelExecutor::GetInstance().GetInBuffer().addr, 0u);
    EXPECT_EQ(AivKernelExecutor::GetInstance().GetOutBuffer().addr, 0u);
}

TEST_F(AiCoreStubTest, NextTaskId_Increments)
{
    AivKernelExecutor::GetInstance().Init(0, 1, 1);
    TaskId id0 = AivKernelExecutor::GetInstance().NextTaskId();
    TaskId id1 = AivKernelExecutor::GetInstance().NextTaskId();
    EXPECT_EQ(id1 - id0, 1u);
}

TEST_F(AiCoreStubTest, GetAivCore_ValidIndex_ReturnsCore)
{
    AivKernelExecutor::GetInstance().Init(0, 4, 2);
    auto core = AivKernelExecutor::GetInstance().GetAivCore(0);
    ASSERT_NE(core, nullptr);
    EXPECT_EQ(core->GetBlockIdx(), 0);
}

TEST_F(AiCoreStubTest, GetAivCore_IndexOutOfBounds_ReturnsNull)
{
    AivKernelExecutor::GetInstance().Init(0, 2, 2);
    EXPECT_EQ(AivKernelExecutor::GetInstance().GetAivCore(-1), nullptr);
    EXPECT_EQ(AivKernelExecutor::GetInstance().GetAivCore(2), nullptr);
}

TEST_F(AiCoreStubTest, SetIoBuffer_SetsBuffers)
{
    AivKernelExecutor::GetInstance().Init(0, 1, 1);
    AivKernelExecutor::GetInstance().SetIoBuffer(0x1000, 512, 0x2000, 1024, 0x10, 0x20);
    EXPECT_EQ(AivKernelExecutor::GetInstance().GetInBuffer().addr, 0x1000u);
    EXPECT_EQ(AivKernelExecutor::GetInstance().GetInBuffer().size, 512u);
    EXPECT_EQ(AivKernelExecutor::GetInstance().GetOutBuffer().addr, 0x2000u);
    EXPECT_EQ(AivKernelExecutor::GetInstance().GetOutBuffer().size, 1024u);
    EXPECT_EQ(AivKernelExecutor::GetInstance().GetInputGlobalOffsetBase(), 0x10u);
    EXPECT_EQ(AivKernelExecutor::GetInstance().GetOutputGlobalOffsetBase(), 0x20u);
}

TEST_F(AiCoreStubTest, SetCommBuffer_ValidRankId_SetsBuffers)
{
    AivKernelExecutor::GetInstance().Init(0, 1, 1);
    AivKernelExecutor::GetInstance().SetCommBuffer(0, 0x3000, 2048, 0x4000, 256);
    EXPECT_EQ(AivKernelExecutor::GetInstance().GetCclBuffer(0).addr, 0x3000u);
    EXPECT_EQ(AivKernelExecutor::GetInstance().GetCclBuffer(0).size, 2048u);
    EXPECT_EQ(AivKernelExecutor::GetInstance().GetAivCommInfoBuffer(0).addr, 0x4000u);
    EXPECT_EQ(AivKernelExecutor::GetInstance().GetAivCommInfoBuffer(0).size, 256u);
}

TEST_F(AiCoreStubTest, SetCommBuffer_RankIdOutOfRange_Skipped)
{
    AivKernelExecutor::GetInstance().Init(0, 1, 1);
    AivKernelExecutor::GetInstance().SetCommBuffer(MAX_RANK_NUM, 0x3000, 2048, 0x4000, 256);
    EXPECT_EQ(AivKernelExecutor::GetInstance().GetCclBuffer(MAX_RANK_NUM).addr, 0u);
    EXPECT_EQ(AivKernelExecutor::GetInstance().GetAivCommInfoBuffer(MAX_RANK_NUM).addr, 0u);
}

TEST_F(AiCoreStubTest, GetCclBuffer_RankIdOutOfRange_ReturnsEmpty)
{
    AivKernelExecutor::GetInstance().Init(0, 1, 1);
    auto mem = AivKernelExecutor::GetInstance().GetCclBuffer(MAX_RANK_NUM);
    EXPECT_EQ(mem.addr, 0u);
    EXPECT_EQ(mem.size, 0u);
}

TEST_F(AiCoreStubTest, GetAivCommInfoBuffer_RankIdOutOfRange_ReturnsEmpty)
{
    AivKernelExecutor::GetInstance().Init(0, 1, 1);
    auto mem = AivKernelExecutor::GetInstance().GetAivCommInfoBuffer(MAX_RANK_NUM);
    EXPECT_EQ(mem.addr, 0u);
    EXPECT_EQ(mem.size, 0u);
}

TEST_F(AiCoreStubTest, AivBufferContains_NullBuffer_ReturnsFalse)
{
    Mem buf{0, 0};
    EXPECT_FALSE(AivBufferContains(buf, 0x100, 10));
}

TEST_F(AiCoreStubTest, AivBufferContains_AddrBeforeBuffer_ReturnsFalse)
{
    Mem buf{0x1000, 1024};
    EXPECT_FALSE(AivBufferContains(buf, 0x500, 10));
}

TEST_F(AiCoreStubTest, AivBufferContains_OffsetExceedsSize_ReturnsFalse)
{
    Mem buf{0x1000, 1024};
    EXPECT_FALSE(AivBufferContains(buf, 0x1400, 10));
}

TEST_F(AiCoreStubTest, AivBufferContains_SizeExceedsRemaining_ReturnsFalse)
{
    Mem buf{0x1000, 1024};
    EXPECT_FALSE(AivBufferContains(buf, 0x1F00, 256));
}

TEST_F(AiCoreStubTest, AivBufferContains_WithinBuffer_ReturnsTrue)
{
    Mem buf{0x1000, 1024};
    EXPECT_TRUE(AivBufferContains(buf, 0x1000, 1024));
    EXPECT_TRUE(AivBufferContains(buf, 0x1100, 512));
}

TEST_F(AiCoreStubTest, AivBufferMatch_NotInBuffer_ReturnsFalse)
{
    Mem buf{0x1000, 1024};
    AivDataSlice slice;
    RankId rank;
    EXPECT_FALSE(AivBufferMatch(0x500, 10, buf, AivBufferType::INPUT, 0, slice, &rank));
}

TEST_F(AiCoreStubTest, AivBufferMatch_WithinBuffer_SetsSliceAndRank)
{
    Mem buf{0x1000, 1024};
    AivDataSlice slice;
    RankId rank = UINT32_MAX;
    EXPECT_TRUE(AivBufferMatch(0x1100, 256, buf, AivBufferType::OUTPUT, 2, slice, &rank));
    EXPECT_EQ(slice.GetType(), AivBufferType::OUTPUT);
    EXPECT_EQ(slice.GetOffset(), 0x100u);
    EXPECT_EQ(slice.GetSize(), 256u);
    EXPECT_EQ(rank, 2u);
}

TEST_F(AiCoreStubTest, AivBufferMatch_NullMatchedRank_DoesNotSet)
{
    Mem buf{0x1000, 1024};
    AivDataSlice slice;
    EXPECT_TRUE(AivBufferMatch(0x1000, 10, buf, AivBufferType::INPUT, 5, slice, nullptr));
}

TEST_F(AiCoreStubTest, ResolveGlobalDataSlice_AddrInInputBuffer_ReturnsInputSlice)
{
    AivKernelExecutor::GetInstance().Init(0, 1, 1);
    AivKernelExecutor::GetInstance().SetIoBuffer(0x1000, 1024, 0x2000, 2048, 0, 0);
    RankId rank;
    auto slice = AivKernelExecutor::GetInstance().ResolveGlobalDataSlice(0x1100, 256, &rank);
    EXPECT_EQ(slice.GetType(), AivBufferType::INPUT);
    EXPECT_EQ(slice.GetOffset(), 0x100u);
    EXPECT_EQ(rank, 0u);
}

TEST_F(AiCoreStubTest, ResolveGlobalDataSlice_AddrInOutputBuffer_ReturnsOutputSlice)
{
    AivKernelExecutor::GetInstance().Init(1, 1, 1);
    AivKernelExecutor::GetInstance().SetIoBuffer(0x1000, 1024, 0x2000, 2048, 0, 0);
    RankId rank;
    auto slice = AivKernelExecutor::GetInstance().ResolveGlobalDataSlice(0x2200, 512, &rank);
    EXPECT_EQ(slice.GetType(), AivBufferType::OUTPUT);
    EXPECT_EQ(rank, 1u);
}

TEST_F(AiCoreStubTest, ResolveGlobalDataSlice_AddrInCclBuffer_ReturnsCclSlice)
{
    AivKernelExecutor::GetInstance().Init(0, 1, 2);
    AivKernelExecutor::GetInstance().SetCommBuffer(1, 0x5000, 2048, 0x6000, 256);
    RankId rank;
    auto slice = AivKernelExecutor::GetInstance().ResolveGlobalDataSlice(0x5100, 128, &rank);
    EXPECT_EQ(slice.GetType(), AivBufferType::CCL);
    EXPECT_EQ(rank, 1u);
}

TEST_F(AiCoreStubTest, ResolveGlobalDataSlice_AddrInAivCommInfo_ReturnsCommInfoSlice)
{
    AivKernelExecutor::GetInstance().Init(0, 1, 2);
    AivKernelExecutor::GetInstance().SetCommBuffer(0, 0x5000, 2048, 0x6000, 256);
    RankId rank;
    auto slice = AivKernelExecutor::GetInstance().ResolveGlobalDataSlice(0x6010, 32, &rank);
    EXPECT_EQ(slice.GetType(), AivBufferType::AIV_COMM);
    EXPECT_EQ(rank, 0u);
    EXPECT_EQ(slice.GetOffset(), 0x10u);
}

TEST_F(AiCoreStubTest, ResolveGlobalDataSlice_NoMatch_ReturnsEmptySlice)
{
    AivKernelExecutor::GetInstance().Init(0, 1, 1);
    AivKernelExecutor::GetInstance().SetIoBuffer(0x1000, 1024, 0x2000, 2048, 0, 0);
    RankId rank;
    auto slice = AivKernelExecutor::GetInstance().ResolveGlobalDataSlice(0x9000, 10, &rank);
    EXPECT_EQ(slice.GetSize(), 0u);
    EXPECT_EQ(rank, UINT32_MAX);
}

TEST_F(AiCoreStubTest, ResolveGlobalDataSlice_NullRankId_DoesNotCrash)
{
    AivKernelExecutor::GetInstance().Init(0, 1, 1);
    AivKernelExecutor::GetInstance().SetIoBuffer(0x1000, 1024, 0x2000, 2048, 0, 0);
    auto slice = AivKernelExecutor::GetInstance().ResolveGlobalDataSlice(0x9000, 10, nullptr);
    EXPECT_EQ(slice.GetSize(), 0u);
}

TEST_F(AiCoreStubTest, ResolveGlobalDataSlice_InputPriorityOverOutput)
{
    AivKernelExecutor::GetInstance().Init(0, 1, 1);
    AivKernelExecutor::GetInstance().SetIoBuffer(0x1000, 1024, 0x1000, 1024, 0, 0);
    RankId rank;
    auto slice = AivKernelExecutor::GetInstance().ResolveGlobalDataSlice(0x1000, 10, &rank);
    EXPECT_EQ(slice.GetType(), AivBufferType::INPUT);
}

TEST_F(AiCoreStubTest, DumpAllTasks_DoesNotCrash)
{
    AivKernelExecutor::GetInstance().Init(0, 2, 1);
    EXPECT_NO_THROW(AivKernelExecutor::GetInstance().DumpAllTasks());
}

TEST_F(AiCoreStubTest, AivCore_GetBlockIdx_ReturnsCorrectIndex)
{
    AivCore core(3);
    EXPECT_EQ(core.GetBlockIdx(), 3);
}

TEST_F(AiCoreStubTest, AivCore_SetAndGetAtomicOp)
{
    AivCore core(0);
    EXPECT_EQ(core.GetAtomicOp(), ReduceOp::REDUCE_RESERVED);
    core.SetAtomicOp(ReduceOp::REDUCE_SUM);
    EXPECT_EQ(core.GetAtomicOp(), ReduceOp::REDUCE_SUM);
}

TEST_F(AiCoreStubTest, AivCore_AppendScalar_IncreasesPipeSize)
{
    AivKernelExecutor::GetInstance().Init(0, 1, 1);
    auto core = AivKernelExecutor::GetInstance().GetAivCore(0);
    ASSERT_NE(core, nullptr);
    auto task = std::make_shared<AivTask>(AivTaskType::MEM_COPY);
    core->AppendScalar(task);
    EXPECT_EQ(core->GetScalarPipe().size(), 1u);
    EXPECT_EQ(core->GetScalarPipe()[0]->GetCurPipe(), AscendC::pipe_t::PIPE_S);
}

TEST_F(AiCoreStubTest, AivCore_AppendMTE2_IncreasesPipeSize)
{
    AivKernelExecutor::GetInstance().Init(0, 1, 1);
    auto core = AivKernelExecutor::GetInstance().GetAivCore(0);
    ASSERT_NE(core, nullptr);
    auto task = std::make_shared<AivTask>(AivTaskType::MEM_COPY);
    core->AppendMTE2(task);
    EXPECT_EQ(core->GetMTE2Pipe().size(), 1u);
    EXPECT_EQ(core->GetMTE2Pipe()[0]->GetCurPipe(), AscendC::pipe_t::PIPE_MTE2);
}

TEST_F(AiCoreStubTest, AivCore_AppendMTE3_IncreasesPipeSize)
{
    AivKernelExecutor::GetInstance().Init(0, 1, 1);
    auto core = AivKernelExecutor::GetInstance().GetAivCore(0);
    ASSERT_NE(core, nullptr);
    auto task = std::make_shared<AivTask>(AivTaskType::MEM_COPY);
    core->AppendMTE3(task);
    EXPECT_EQ(core->GetMTE3Pipe().size(), 1u);
    EXPECT_EQ(core->GetMTE3Pipe()[0]->GetCurPipe(), AscendC::pipe_t::PIPE_MTE3);
}

TEST_F(AiCoreStubTest, AivCore_AppendTask_SetsRankAndBlockId)
{
    AivKernelExecutor::GetInstance().Init(5, 1, 2);
    auto core = AivKernelExecutor::GetInstance().GetAivCore(0);
    ASSERT_NE(core, nullptr);
    auto task = std::make_shared<AivTask>(AivTaskType::MEM_COPY);
    core->AppendScalar(task);
    EXPECT_EQ(task->GetRankId(), 5u);
    EXPECT_EQ(task->GetBlockId(), 0u);
}

TEST_F(AiCoreStubTest, AivCore_DumpAllTasks_DoesNotCrash)
{
    AivKernelExecutor::GetInstance().Init(0, 1, 1);
    auto core = AivKernelExecutor::GetInstance().GetAivCore(0);
    ASSERT_NE(core, nullptr);
    auto task = std::make_shared<AivTask>(AivTaskType::MEM_COPY);
    core->AppendScalar(task);
    EXPECT_NO_THROW(core->DumpAllTasks());
}

TEST_F(AiCoreStubTest, SetCurOp_GetCurOp_RoundTrip)
{
    AivKernelExecutor::GetInstance().Init(0, 1, 1);
    AivOpParam op{};
    op.dataType = 1;
    op.len = 1024;
    op.reduceOp = 0;
    op.root = 3;
    op.sliceId = 2;
    AivKernelExecutor::GetInstance().SetCurOp(op);
    EXPECT_EQ(AivKernelExecutor::GetInstance().GetCurOp().dataType, 1u);
    EXPECT_EQ(AivKernelExecutor::GetInstance().GetCurOp().len, 1024u);
    EXPECT_EQ(AivKernelExecutor::GetInstance().GetCurOp().root, 3u);
}

TEST_F(AiCoreStubTest, GetUbBufferSize_ReturnsExpectedValue)
{
    EXPECT_EQ(AivKernelExecutor::GetInstance().GetUbBufferSize(), AIV_UB_SIZE);
}

TEST_F(AiCoreStubTest, Reset_ClearsIoAndCommBuffers)
{
    AivKernelExecutor::GetInstance().Init(0, 1, 1);
    AivKernelExecutor::GetInstance().SetIoBuffer(0x1000, 512, 0x2000, 1024, 0, 0);
    AivKernelExecutor::GetInstance().SetCommBuffer(0, 0x3000, 2048, 0x4000, 256);
    AivKernelExecutor::GetInstance().Reset();
    EXPECT_EQ(AivKernelExecutor::GetInstance().GetInBuffer().addr, 0u);
    EXPECT_EQ(AivKernelExecutor::GetInstance().GetOutBuffer().addr, 0u);
    EXPECT_EQ(AivKernelExecutor::GetInstance().GetInputGlobalOffsetBase(), 0u);
    EXPECT_EQ(AivKernelExecutor::GetInstance().GetOutputGlobalOffsetBase(), 0u);
    EXPECT_EQ(AivKernelExecutor::GetInstance().GetCclBuffer(0).addr, 0u);
    EXPECT_EQ(AivKernelExecutor::GetInstance().GetAivCommInfoBuffer(0).addr, 0u);
}

class AivTaskTest : public testing::Test {
protected:
    void SetUp() override { AivKernelExecutor::GetInstance().Reset(); }
    void TearDown() override { AivKernelExecutor::GetInstance().Reset(); }
};

TEST_F(AivTaskTest, AivTask_GetUUID_ReturnsExpectedValue)
{
    AivTask task(AivTaskType::MEM_COPY, 10, 2, 0, AscendC::pipe_t::PIPE_S);
    uint64_t expected = (static_cast<uint64_t>(2) << 32) | 10u;
    EXPECT_EQ(task.GetUUID(), expected);
}

TEST_F(AivTaskTest, AivTask_Describe_ReturnsNonEmpty)
{
    AivTask task(AivTaskType::MEM_COPY, 10, 2, 0, AscendC::pipe_t::PIPE_S);
    std::string desc = task.Describe();
    EXPECT_FALSE(desc.empty());
    EXPECT_NE(desc.find("MemCopy"), std::string::npos);
    EXPECT_NE(desc.find("RankId=2"), std::string::npos);
    EXPECT_NE(desc.find("TaskId=10"), std::string::npos);
    EXPECT_NE(desc.find("BlockId=0"), std::string::npos);
}

TEST_F(AivTaskTest, AivTask_SetTaskType)
{
    AivTask task(AivTaskType::MEM_COPY);
    task.SetTaskType(AivTaskType::REDUCE);
    EXPECT_EQ(task.GetTaskType(), AivTaskType::REDUCE);
}

TEST_F(AivTaskTest, AivTaskMemCopy_Describe_ContainsSrcAndDst)
{
    AivDataSlice src(AivBufferType::INPUT, 0, 100);
    AivDataSlice dst(AivBufferType::OUTPUT, 0, 100);
    AivTaskMemCopy task(0, src, 1, dst);
    std::string desc = task.Describe();
    EXPECT_FALSE(desc.empty());
    EXPECT_NE(desc.find("SrcRank=0"), std::string::npos);
    EXPECT_NE(desc.find("DstRank=1"), std::string::npos);
    EXPECT_NE(desc.find("INPUT"), std::string::npos);
    EXPECT_NE(desc.find("OUTPUT"), std::string::npos);
}

TEST_F(AivTaskTest, AivTaskMemCopy_SetSrcAndDst)
{
    AivDataSlice src(AivBufferType::INPUT, 0, 100);
    AivDataSlice dst(AivBufferType::OUTPUT, 0, 100);
    AivTaskMemCopy task(0, src, 1, dst);
    AivDataSlice newSrc(AivBufferType::CCL, 10, 50);
    AivDataSlice newDst(AivBufferType::AIV_COMM, 20, 60);
    task.SetSrc(newSrc);
    task.SetDst(newDst);
    task.SetSrcRank(5);
    task.SetDstRank(6);
    EXPECT_EQ(task.GetSrc().GetType(), AivBufferType::CCL);
    EXPECT_EQ(task.GetDst().GetType(), AivBufferType::AIV_COMM);
    EXPECT_EQ(task.GetSrcRank(), 5u);
    EXPECT_EQ(task.GetDstRank(), 6u);
}

TEST_F(AivTaskTest, AivTaskReduce_Describe_ContainsAllFields)
{
    AivDataSlice src(AivBufferType::INPUT, 0, 100);
    AivDataSlice dst(AivBufferType::OUTPUT, 0, 100);
    AivTaskReduce task(0, src, 1, dst, 1, static_cast<uint32_t>(ReduceOp::REDUCE_SUM));
    std::string desc = task.Describe();
    EXPECT_NE(desc.find("Reduce"), std::string::npos);
    EXPECT_NE(desc.find("SrcRank=0"), std::string::npos);
    EXPECT_NE(desc.find("DstRank=1"), std::string::npos);
    EXPECT_NE(desc.find("DataType=1"), std::string::npos);
    EXPECT_NE(desc.find("SUM"), std::string::npos);
}

TEST_F(AivTaskTest, AivTaskReduce_Setters)
{
    AivDataSlice src(AivBufferType::INPUT, 0, 100);
    AivDataSlice dst(AivBufferType::OUTPUT, 0, 100);
    AivTaskReduce task(0, src, 1, dst, 1, 0);
    AivDataSlice newSrc(AivBufferType::CCL, 10, 50);
    AivDataSlice newDst(AivBufferType::AIV_COMM, 20, 60);
    task.SetSrc(newSrc);
    task.SetDst(newDst);
    task.SetSrcRank(3);
    task.SetDstRank(4);
    task.SetDataType(2);
    task.SetReduceOp(static_cast<uint32_t>(ReduceOp::REDUCE_MAX));
    EXPECT_EQ(task.GetSrc().GetType(), AivBufferType::CCL);
    EXPECT_EQ(task.GetDst().GetType(), AivBufferType::AIV_COMM);
    EXPECT_EQ(task.GetSrcRank(), 3u);
    EXPECT_EQ(task.GetDstRank(), 4u);
    EXPECT_EQ(task.GetDataType(), 2u);
    EXPECT_EQ(task.GetReduceOp(), static_cast<uint32_t>(ReduceOp::REDUCE_MAX));
}

TEST_F(AivTaskTest, AivTaskSetFlag_Describe_ContainsPipeAndEvent)
{
    AivTaskSetFlag task(AscendC::pipe_t::PIPE_S, AscendC::pipe_t::PIPE_MTE2, 7);
    std::string desc = task.Describe();
    EXPECT_NE(desc.find("SetFlag"), std::string::npos);
    EXPECT_NE(desc.find("EventId=7"), std::string::npos);
    EXPECT_NE(desc.find("PIPE_SCALAR"), std::string::npos);
    EXPECT_NE(desc.find("PIPE_MTE2"), std::string::npos);
}

TEST_F(AivTaskTest, AivTaskSetFlag_Setters)
{
    AivTaskSetFlag task(AscendC::pipe_t::PIPE_S, AscendC::pipe_t::PIPE_MTE2, 7);
    task.SetSrcPipe(AscendC::pipe_t::PIPE_MTE3);
    task.SetDstPipe(AscendC::pipe_t::PIPE_S);
    task.SetEventId(99);
    EXPECT_EQ(task.GetSrcPipe(), AscendC::pipe_t::PIPE_MTE3);
    EXPECT_EQ(task.GetDstPipe(), AscendC::pipe_t::PIPE_S);
    EXPECT_EQ(task.GetEventId(), 99);
}

TEST_F(AivTaskTest, AivTaskWaitFlag_Describe_ContainsPipeAndEvent)
{
    AivTaskWaitFlag task(AscendC::pipe_t::PIPE_MTE2, AscendC::pipe_t::PIPE_S, 3);
    std::string desc = task.Describe();
    EXPECT_NE(desc.find("WaitFlag"), std::string::npos);
    EXPECT_NE(desc.find("EventId=3"), std::string::npos);
}

TEST_F(AivTaskTest, AivTaskWaitFlag_Setters)
{
    AivTaskWaitFlag task(AscendC::pipe_t::PIPE_S, AscendC::pipe_t::PIPE_MTE2, 1);
    task.SetSrcPipe(AscendC::pipe_t::PIPE_MTE3);
    task.SetDstPipe(AscendC::pipe_t::PIPE_MTE2);
    task.SetEventId(42);
    EXPECT_EQ(task.GetSrcPipe(), AscendC::pipe_t::PIPE_MTE3);
    EXPECT_EQ(task.GetDstPipe(), AscendC::pipe_t::PIPE_MTE2);
    EXPECT_EQ(task.GetEventId(), 42);
}

TEST_F(AivTaskTest, AivTaskPipeBarrier_Describe_ContainsPipeType)
{
    AivTaskPipeBarrier task(AscendC::pipe_t::PIPE_MTE2);
    std::string desc = task.Describe();
    EXPECT_NE(desc.find("PipeBarrier"), std::string::npos);
    EXPECT_NE(desc.find("PIPE_MTE2"), std::string::npos);
}

TEST_F(AivTaskTest, AivTaskPipeBarrier_Describe_WithBarrierGroup)
{
    auto task1 = std::make_shared<AivTaskPipeBarrier>(AscendC::pipe_t::PIPE_S);
    auto task2 = std::make_shared<AivTaskPipeBarrier>(AscendC::pipe_t::PIPE_S);
    AivKernelExecutor::GetInstance().Init(0, 1, 1);
    task1->SetTaskId(AivKernelExecutor::GetInstance().NextTaskId());
    task2->SetTaskId(AivKernelExecutor::GetInstance().NextTaskId());
    task1->AddBarrierGroup(task2);
    std::string desc = task1->Describe();
    EXPECT_NE(desc.find("BarrierGroup"), std::string::npos);
}

TEST_F(AivTaskTest, AivTaskPipeBarrier_AddBarrierGroup_IncreasesSize)
{
    auto task1 = std::make_shared<AivTaskPipeBarrier>(AscendC::pipe_t::PIPE_S);
    auto task2 = std::make_shared<AivTaskPipeBarrier>(AscendC::pipe_t::PIPE_S);
    EXPECT_EQ(task1->GetBarrierGroup().size(), 0u);
    task1->AddBarrierGroup(task2);
    EXPECT_EQ(task1->GetBarrierGroup().size(), 1u);
}

TEST_F(AivTaskTest, AivTaskPipeBarrier_SetPipeType)
{
    AivTaskPipeBarrier task(AscendC::pipe_t::PIPE_S);
    task.SetPipeType(AscendC::pipe_t::PIPE_MTE3);
    EXPECT_EQ(task.GetPipeType(), AscendC::pipe_t::PIPE_MTE3);
}

TEST_F(AivTaskTest, AivTaskSyncAll_Describe_ReturnsNonEmpty)
{
    AivTaskSyncAll task(0);
    std::string desc = task.Describe();
    EXPECT_NE(desc.find("SyncRound=0"), std::string::npos);
}

TEST_F(AivTaskTest, AivTaskSendFlag_Describe_ContainsRankOffsetValue)
{
    AivTaskSendFlag task(3, 0x100, 1);
    std::string desc = task.Describe();
    EXPECT_NE(desc.find("SendFlag"), std::string::npos);
    EXPECT_NE(desc.find("Rank=3"), std::string::npos);
    EXPECT_NE(desc.find("CommInfoOffset=256"), std::string::npos);
    EXPECT_NE(desc.find("Value=1"), std::string::npos);
}

TEST_F(AivTaskTest, AivTaskSendFlag_Setters)
{
    AivTaskSendFlag task(0, 0, 0);
    task.SetRank(5);
    task.SetCommInfoOffset(0x200);
    task.SetFlagValue(2);
    EXPECT_EQ(task.GetRank(), 5u);
    EXPECT_EQ(task.GetCommInfoOffset(), 0x200u);
    EXPECT_EQ(task.GetFlagValue(), 2);
}

TEST_F(AivTaskTest, AivTaskRecvFlag_Describe_ContainsRankOffsetValue)
{
    AivTaskRecvFlag task(2, 0x80, 1);
    std::string desc = task.Describe();
    EXPECT_NE(desc.find("RecvFlag"), std::string::npos);
    EXPECT_NE(desc.find("Rank=2"), std::string::npos);
}

TEST_F(AivTaskTest, AivTaskRecvFlag_Setters)
{
    AivTaskRecvFlag task(0, 0, 0);
    task.SetRank(4);
    task.SetCommInfoOffset(0x400);
    task.SetTargetValue(3);
    EXPECT_EQ(task.GetRank(), 4u);
    EXPECT_EQ(task.GetCommInfoOffset(), 0x400u);
    EXPECT_EQ(task.GetTargetValue(), 3);
}

TEST_F(AivTaskTest, GetTypeName_AllTaskTypes)
{
    EXPECT_EQ(GetTypeName(AivTaskType::MEM_COPY), "MemCopy");
    EXPECT_EQ(GetTypeName(AivTaskType::REDUCE), "Reduce");
    EXPECT_EQ(GetTypeName(AivTaskType::SET_FLAG), "SetFlag");
    EXPECT_EQ(GetTypeName(AivTaskType::WAIT_FLAG), "WaitFlag");
    EXPECT_EQ(GetTypeName(AivTaskType::PIPE_BARRIER), "PipeBarrier");
    EXPECT_EQ(GetTypeName(AivTaskType::SYNC_ALL), "SyncAll");
    EXPECT_EQ(GetTypeName(AivTaskType::SEND_FLAG), "SendFlag");
    EXPECT_EQ(GetTypeName(AivTaskType::RECV_FLAG), "RecvFlag");
    EXPECT_EQ(GetTypeName(AivTaskType::INVALID_TYPE), "UnknownTask");
}
