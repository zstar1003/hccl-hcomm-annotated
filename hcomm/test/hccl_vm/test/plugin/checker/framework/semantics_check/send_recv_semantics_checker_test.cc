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
#include "send_recv_semantics_checker.h"

namespace HcclSim {
class SendRecvSemanticsCheckerTest : public testing::Test {
protected:
    void SetUp() override {}

    void TearDown() override {}

    // Helper function to create valid Send/Recv semantics
    void CreateValidSendRecvSemantics(
        std::map<RankId, RankMemorySemantics>& allRankMemSemantics, u64 dataSize, RankId srcRank, RankId dstRank)
    {
        // Only destination rank has output with data from source rank
        RankMemorySemantics dstMemSemantics;
        std::set<BufferSemantic> outputSemantics;

        BufferSemantic bufSem(0, dataSize);
        bufSem.srcBufs.insert(SrcBufDes(srcRank, BufferType::INPUT, 0));
        outputSemantics.insert(bufSem);

        dstMemSemantics[BufferType::OUTPUT] = outputSemantics;
        allRankMemSemantics[dstRank] = dstMemSemantics;

        // Source rank has empty semantics
        RankMemorySemantics srcMemSemantics;
        allRankMemSemantics[srcRank] = srcMemSemantics;
    }
};

// Test normal case: valid Send/Recv semantics
TEST_F(SendRecvSemanticsCheckerTest, ValidSemantics_Basic)
{
    std::map<RankId, RankMemorySemantics> allRankMemSemantics;
    u64 dataSize = 1024;
    RankId srcRank = 0;
    RankId dstRank = 1;

    CreateValidSendRecvSemantics(allRankMemSemantics, dataSize, srcRank, dstRank);

    HcclResult result = TaskCheckSendRecvSemantics(allRankMemSemantics, dataSize, srcRank, dstRank);
    EXPECT_EQ(result, HcclResult::HCCL_SUCCESS);
}

// Test normal case: reversed ranks
TEST_F(SendRecvSemanticsCheckerTest, ValidSemantics_ReversedRanks)
{
    std::map<RankId, RankMemorySemantics> allRankMemSemantics;
    u64 dataSize = 2048;
    RankId srcRank = 1;
    RankId dstRank = 0;

    CreateValidSendRecvSemantics(allRankMemSemantics, dataSize, srcRank, dstRank);

    HcclResult result = TaskCheckSendRecvSemantics(allRankMemSemantics, dataSize, srcRank, dstRank);
    EXPECT_EQ(result, HcclResult::HCCL_SUCCESS);
}

// Test boundary case: zero data size
TEST_F(SendRecvSemanticsCheckerTest, ValidSemantics_ZeroDataSize)
{
    std::map<RankId, RankMemorySemantics> allRankMemSemantics;
    u64 dataSize = 0;
    RankId srcRank = 0;
    RankId dstRank = 1;

    CreateValidSendRecvSemantics(allRankMemSemantics, dataSize, srcRank, dstRank);

    HcclResult result = TaskCheckSendRecvSemantics(allRankMemSemantics, dataSize, srcRank, dstRank);
    EXPECT_EQ(result, HcclResult::HCCL_SUCCESS);
}

// Test abnormal case: not exactly 2 ranks
TEST_F(SendRecvSemanticsCheckerTest, Abnormal_NotTwoRanks)
{
    std::map<RankId, RankMemorySemantics> allRankMemSemantics;
    u64 dataSize = 1024;
    RankId srcRank = 0;
    RankId dstRank = 1;

    // Add 3 ranks instead of 2
    RankMemorySemantics rank0MemSemantics;
    allRankMemSemantics[0] = rank0MemSemantics;

    RankMemorySemantics rank1MemSemantics;
    allRankMemSemantics[1] = rank1MemSemantics;

    RankMemorySemantics rank2MemSemantics;
    allRankMemSemantics[2] = rank2MemSemantics;

    HcclResult result = TaskCheckSendRecvSemantics(allRankMemSemantics, dataSize, srcRank, dstRank);
    EXPECT_EQ(result, HcclResult::HCCL_E_PARA);
}

// Test abnormal case: single rank
TEST_F(SendRecvSemanticsCheckerTest, Abnormal_SingleRank)
{
    std::map<RankId, RankMemorySemantics> allRankMemSemantics;
    u64 dataSize = 1024;
    RankId srcRank = 0;
    RankId dstRank = 0;

    RankMemorySemantics rankMemSemantics;
    allRankMemSemantics[0] = rankMemSemantics;

    HcclResult result = TaskCheckSendRecvSemantics(allRankMemSemantics, dataSize, srcRank, dstRank);
    EXPECT_EQ(result, HcclResult::HCCL_E_PARA);
}

// Test abnormal case: wrong source rank in semantics
TEST_F(SendRecvSemanticsCheckerTest, Abnormal_WrongSourceRank)
{
    std::map<RankId, RankMemorySemantics> allRankMemSemantics;
    u64 dataSize = 1024;
    RankId srcRank = 0;
    RankId dstRank = 1;

    RankMemorySemantics dstMemSemantics;
    std::set<BufferSemantic> outputSemantics;

    BufferSemantic bufSem(0, dataSize);
    bufSem.srcBufs.insert(SrcBufDes(2, BufferType::INPUT, 0)); // Wrong source rank
    outputSemantics.insert(bufSem);

    dstMemSemantics[BufferType::OUTPUT] = outputSemantics;
    allRankMemSemantics[dstRank] = dstMemSemantics;

    RankMemorySemantics srcMemSemantics;
    allRankMemSemantics[srcRank] = srcMemSemantics;

    HcclResult result = TaskCheckSendRecvSemantics(allRankMemSemantics, dataSize, srcRank, dstRank);
    EXPECT_EQ(result, HcclResult::HCCL_E_PARA);
}

// Test abnormal case: wrong buffer type
TEST_F(SendRecvSemanticsCheckerTest, Abnormal_WrongBufferType)
{
    std::map<RankId, RankMemorySemantics> allRankMemSemantics;
    u64 dataSize = 1024;
    RankId srcRank = 0;
    RankId dstRank = 1;

    RankMemorySemantics dstMemSemantics;
    std::set<BufferSemantic> outputSemantics;

    BufferSemantic bufSem(0, dataSize);
    bufSem.srcBufs.insert(SrcBufDes(srcRank, BufferType::OUTPUT, 0)); // Wrong buffer type
    outputSemantics.insert(bufSem);

    dstMemSemantics[BufferType::OUTPUT] = outputSemantics;
    allRankMemSemantics[dstRank] = dstMemSemantics;

    RankMemorySemantics srcMemSemantics;
    allRankMemSemantics[srcRank] = srcMemSemantics;

    HcclResult result = TaskCheckSendRecvSemantics(allRankMemSemantics, dataSize, srcRank, dstRank);
    EXPECT_EQ(result, HcclResult::HCCL_E_PARA);
}

// Test abnormal case: wrong source addr
TEST_F(SendRecvSemanticsCheckerTest, Abnormal_WrongSourceAddr)
{
    std::map<RankId, RankMemorySemantics> allRankMemSemantics;
    u64 dataSize = 1024;
    RankId srcRank = 0;
    RankId dstRank = 1;

    RankMemorySemantics dstMemSemantics;
    std::set<BufferSemantic> outputSemantics;

    BufferSemantic bufSem(0, dataSize);
    bufSem.srcBufs.insert(SrcBufDes(srcRank, BufferType::INPUT, 100)); // Wrong source addr
    outputSemantics.insert(bufSem);

    dstMemSemantics[BufferType::OUTPUT] = outputSemantics;
    allRankMemSemantics[dstRank] = dstMemSemantics;

    RankMemorySemantics srcMemSemantics;
    allRankMemSemantics[srcRank] = srcMemSemantics;

    HcclResult result = TaskCheckSendRecvSemantics(allRankMemSemantics, dataSize, srcRank, dstRank);
    EXPECT_EQ(result, HcclResult::HCCL_E_PARA);
}

// Test abnormal case: incomplete total size
TEST_F(SendRecvSemanticsCheckerTest, Abnormal_IncompleteTotalSize)
{
    std::map<RankId, RankMemorySemantics> allRankMemSemantics;
    u64 dataSize = 1024;
    RankId srcRank = 0;
    RankId dstRank = 1;

    RankMemorySemantics dstMemSemantics;
    std::set<BufferSemantic> outputSemantics;

    BufferSemantic bufSem(0, dataSize / 2); // Only half the size
    bufSem.srcBufs.insert(SrcBufDes(srcRank, BufferType::INPUT, 0));
    outputSemantics.insert(bufSem);

    dstMemSemantics[BufferType::OUTPUT] = outputSemantics;
    allRankMemSemantics[dstRank] = dstMemSemantics;

    RankMemorySemantics srcMemSemantics;
    allRankMemSemantics[srcRank] = srcMemSemantics;

    HcclResult result = TaskCheckSendRecvSemantics(allRankMemSemantics, dataSize, srcRank, dstRank);
    EXPECT_EQ(result, HcclResult::HCCL_E_PARA);
}

// Test boundary case: large data size
TEST_F(SendRecvSemanticsCheckerTest, ValidSemantics_LargeDataSize)
{
    std::map<RankId, RankMemorySemantics> allRankMemSemantics;
    u64 dataSize = 1024 * 1024 * 1024; // 1GB
    RankId srcRank = 0;
    RankId dstRank = 1;

    CreateValidSendRecvSemantics(allRankMemSemantics, dataSize, srcRank, dstRank);

    HcclResult result = TaskCheckSendRecvSemantics(allRankMemSemantics, dataSize, srcRank, dstRank);
    EXPECT_EQ(result, HcclResult::HCCL_SUCCESS);
}
} // namespace HcclSim
