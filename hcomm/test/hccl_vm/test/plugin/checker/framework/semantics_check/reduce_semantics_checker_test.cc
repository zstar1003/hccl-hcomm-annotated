/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <map>
#include <set>

#include "check_utils.h"
#include "reduce_semantics_checker.h"

namespace HcclSim {
class ReduceSemanticsCheckerTest : public testing::Test {
protected:
    void SetUp() override {}

    void TearDown() override {}

    // Helper function to create valid Reduce semantics
    void CreateValidReduceSemantics(
        std::map<RankId, RankMemorySemantics>& allRankMemSemantics, u32 rankSize, u64 dataSize, HcclReduceOp reduceType,
        RankId root)
    {
        // Only root has output with reduce result
        RankMemorySemantics rootMemSemantics;
        std::set<BufferSemantic> outputSemantics;

        BufferSemantic bufSem(0, dataSize, true, reduceType);
        for (RankId srcRank = 0; srcRank < rankSize; srcRank++) {
            bufSem.srcBufs.insert(SrcBufDes(srcRank, BufferType::INPUT, 0));
        }
        outputSemantics.insert(bufSem);

        rootMemSemantics[BufferType::OUTPUT] = outputSemantics;
        allRankMemSemantics[root] = rootMemSemantics;

        // Other ranks have empty output semantics
        for (RankId rankId = 0; rankId < rankSize; rankId++) {
            if (rankId != root) {
                RankMemorySemantics rankMemSemantics;
                allRankMemSemantics[rankId] = rankMemSemantics;
            }
        }
    }
};

// Test normal case: valid Reduce semantics with root 0
TEST_F(ReduceSemanticsCheckerTest, ValidSemantics_RootZero)
{
    std::map<RankId, RankMemorySemantics> allRankMemSemantics;
    u32 rankSize = 4;
    u64 dataSize = 1024;
    RankId root = 0;

    CreateValidReduceSemantics(allRankMemSemantics, rankSize, dataSize, HcclReduceOp::HCCL_REDUCE_SUM, root);

    HcclResult result = TaskCheckReduceSemantics(allRankMemSemantics, dataSize, HcclReduceOp::HCCL_REDUCE_SUM, root);
    EXPECT_EQ(result, HcclResult::HCCL_SUCCESS);
}

// Test normal case: valid Reduce semantics with non-zero root
TEST_F(ReduceSemanticsCheckerTest, ValidSemantics_NonZeroRoot)
{
    std::map<RankId, RankMemorySemantics> allRankMemSemantics;
    u32 rankSize = 4;
    u64 dataSize = 1024;
    RankId root = 2;

    CreateValidReduceSemantics(allRankMemSemantics, rankSize, dataSize, HcclReduceOp::HCCL_REDUCE_SUM, root);

    HcclResult result = TaskCheckReduceSemantics(allRankMemSemantics, dataSize, HcclReduceOp::HCCL_REDUCE_SUM, root);
    EXPECT_EQ(result, HcclResult::HCCL_SUCCESS);
}

// Test normal case: different reduce operations
TEST_F(ReduceSemanticsCheckerTest, ValidSemantics_ReduceProd)
{
    std::map<RankId, RankMemorySemantics> allRankMemSemantics;
    u32 rankSize = 4;
    u64 dataSize = 1024;
    RankId root = 0;

    CreateValidReduceSemantics(allRankMemSemantics, rankSize, dataSize, HcclReduceOp::HCCL_REDUCE_PROD, root);

    HcclResult result = TaskCheckReduceSemantics(allRankMemSemantics, dataSize, HcclReduceOp::HCCL_REDUCE_PROD, root);
    EXPECT_EQ(result, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceSemanticsCheckerTest, ValidSemantics_ReduceMax)
{
    std::map<RankId, RankMemorySemantics> allRankMemSemantics;
    u32 rankSize = 4;
    u64 dataSize = 1024;
    RankId root = 0;

    CreateValidReduceSemantics(allRankMemSemantics, rankSize, dataSize, HcclReduceOp::HCCL_REDUCE_MAX, root);

    HcclResult result = TaskCheckReduceSemantics(allRankMemSemantics, dataSize, HcclReduceOp::HCCL_REDUCE_MAX, root);
    EXPECT_EQ(result, HcclResult::HCCL_SUCCESS);
}

// Test boundary case: single rank
TEST_F(ReduceSemanticsCheckerTest, ValidSemantics_SingleRank)
{
    std::map<RankId, RankMemorySemantics> allRankMemSemantics;
    u32 rankSize = 1;
    u64 dataSize = 1024;
    RankId root = 0;

    CreateValidReduceSemantics(allRankMemSemantics, rankSize, dataSize, HcclReduceOp::HCCL_REDUCE_SUM, root);

    HcclResult result = TaskCheckReduceSemantics(allRankMemSemantics, dataSize, HcclReduceOp::HCCL_REDUCE_SUM, root);
    EXPECT_EQ(result, HcclResult::HCCL_SUCCESS);
}

// Test boundary case: zero data size
TEST_F(ReduceSemanticsCheckerTest, ValidSemantics_ZeroDataSize)
{
    std::map<RankId, RankMemorySemantics> allRankMemSemantics;
    u32 rankSize = 4;
    u64 dataSize = 0;
    RankId root = 0;

    CreateValidReduceSemantics(allRankMemSemantics, rankSize, dataSize, HcclReduceOp::HCCL_REDUCE_SUM, root);

    HcclResult result = TaskCheckReduceSemantics(allRankMemSemantics, dataSize, HcclReduceOp::HCCL_REDUCE_SUM, root);
    EXPECT_EQ(result, HcclResult::HCCL_SUCCESS);
}

// Test boundary case: empty rank size
TEST_F(ReduceSemanticsCheckerTest, ValidSemantics_EmptyRankSize)
{
    std::map<RankId, RankMemorySemantics> allRankMemSemantics;
    u64 dataSize = 1024;
    RankId root = 0;

    HcclResult result = TaskCheckReduceSemantics(allRankMemSemantics, dataSize, HcclReduceOp::HCCL_REDUCE_SUM, root);
    EXPECT_EQ(result, HcclResult::HCCL_SUCCESS);
}

// Test abnormal case: missing root rank
TEST_F(ReduceSemanticsCheckerTest, Abnormal_MissingRootRank)
{
    std::map<RankId, RankMemorySemantics> allRankMemSemantics;
    u32 rankSize = 2;
    u64 dataSize = 1024;
    RankId root = 1;

    // Only add rank 0, missing root rank 1
    RankMemorySemantics rankMemSemantics;
    allRankMemSemantics[0] = rankMemSemantics;

    HcclResult result = TaskCheckReduceSemantics(allRankMemSemantics, dataSize, HcclReduceOp::HCCL_REDUCE_SUM, root);
    EXPECT_EQ(result, HcclResult::HCCL_E_PARA);
}

// Test abnormal case: wrong reduce type
TEST_F(ReduceSemanticsCheckerTest, Abnormal_WrongReduceType)
{
    std::map<RankId, RankMemorySemantics> allRankMemSemantics;
    u32 rankSize = 2;
    u64 dataSize = 1024;
    RankId root = 0;

    CreateValidReduceSemantics(allRankMemSemantics, rankSize, dataSize, HcclReduceOp::HCCL_REDUCE_SUM, root);

    HcclResult result = TaskCheckReduceSemantics(allRankMemSemantics, dataSize, HcclReduceOp::HCCL_REDUCE_PROD, root);
    EXPECT_EQ(result, HcclResult::HCCL_E_PARA);
}

// Test abnormal case: insufficient source buffers
TEST_F(ReduceSemanticsCheckerTest, Abnormal_InsufficientSourceBuffers)
{
    std::map<RankId, RankMemorySemantics> allRankMemSemantics;
    u32 rankSize = 4;
    u64 dataSize = 1024;
    RankId root = 0;

    RankMemorySemantics rootMemSemantics;
    std::set<BufferSemantic> outputSemantics;

    BufferSemantic bufSem(0, dataSize, true, HcclReduceOp::HCCL_REDUCE_SUM);
    // Only add 2 source buffers instead of 4
    bufSem.srcBufs.insert(SrcBufDes(0, BufferType::INPUT, 0));
    bufSem.srcBufs.insert(SrcBufDes(1, BufferType::INPUT, 0));
    outputSemantics.insert(bufSem);

    rootMemSemantics[BufferType::OUTPUT] = outputSemantics;
    allRankMemSemantics[root] = rootMemSemantics;

    for (RankId rankId = 1; rankId < rankSize; rankId++) {
        RankMemorySemantics rankMemSemantics;
        allRankMemSemantics[rankId] = rankMemSemantics;
    }

    HcclResult result = TaskCheckReduceSemantics(allRankMemSemantics, dataSize, HcclReduceOp::HCCL_REDUCE_SUM, root);
    EXPECT_EQ(result, HcclResult::HCCL_E_PARA);
}

// Test abnormal case: wrong buffer type
TEST_F(ReduceSemanticsCheckerTest, Abnormal_WrongBufferType)
{
    std::map<RankId, RankMemorySemantics> allRankMemSemantics;
    u32 rankSize = 2;
    u64 dataSize = 1024;
    RankId root = 0;

    RankMemorySemantics rootMemSemantics;
    std::set<BufferSemantic> outputSemantics;

    BufferSemantic bufSem(0, dataSize, true, HcclReduceOp::HCCL_REDUCE_SUM);
    bufSem.srcBufs.insert(SrcBufDes(0, BufferType::OUTPUT, 0)); // Wrong buffer type
    bufSem.srcBufs.insert(SrcBufDes(1, BufferType::INPUT, 0));
    outputSemantics.insert(bufSem);

    rootMemSemantics[BufferType::OUTPUT] = outputSemantics;
    allRankMemSemantics[root] = rootMemSemantics;

    RankMemorySemantics otherMemSemantics;
    allRankMemSemantics[1] = otherMemSemantics;

    HcclResult result = TaskCheckReduceSemantics(allRankMemSemantics, dataSize, HcclReduceOp::HCCL_REDUCE_SUM, root);
    EXPECT_EQ(result, HcclResult::HCCL_E_PARA);
}

// Test abnormal case: wrong total size
TEST_F(ReduceSemanticsCheckerTest, Abnormal_WrongTotalSize)
{
    std::map<RankId, RankMemorySemantics> allRankMemSemantics;
    u32 rankSize = 2;
    u64 dataSize = 1024;
    RankId root = 0;

    RankMemorySemantics rootMemSemantics;
    std::set<BufferSemantic> outputSemantics;

    BufferSemantic bufSem(0, 512, true, HcclReduceOp::HCCL_REDUCE_SUM); // Wrong size
    bufSem.srcBufs.insert(SrcBufDes(0, BufferType::INPUT, 0));
    bufSem.srcBufs.insert(SrcBufDes(1, BufferType::INPUT, 0));
    outputSemantics.insert(bufSem);

    rootMemSemantics[BufferType::OUTPUT] = outputSemantics;
    allRankMemSemantics[root] = rootMemSemantics;

    RankMemorySemantics otherMemSemantics;
    allRankMemSemantics[1] = otherMemSemantics;

    HcclResult result = TaskCheckReduceSemantics(allRankMemSemantics, dataSize, HcclReduceOp::HCCL_REDUCE_SUM, root);
    EXPECT_EQ(result, HcclResult::HCCL_E_PARA);
}

// Test boundary case: large number of ranks
TEST_F(ReduceSemanticsCheckerTest, ValidSemantics_LargeRankSize)
{
    std::map<RankId, RankMemorySemantics> allRankMemSemantics;
    u32 rankSize = 128;
    u64 dataSize = 1024;
    RankId root = 64;

    CreateValidReduceSemantics(allRankMemSemantics, rankSize, dataSize, HcclReduceOp::HCCL_REDUCE_SUM, root);

    HcclResult result = TaskCheckReduceSemantics(allRankMemSemantics, dataSize, HcclReduceOp::HCCL_REDUCE_SUM, root);
    EXPECT_EQ(result, HcclResult::HCCL_SUCCESS);
}
} // namespace HcclSim
