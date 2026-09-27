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
#include <vector>

#include "all2all_semantics_checker.h"
#include "check_utils.h"
#include "checker_def.h"

namespace HcclSim {
class All2AllSemanticsCheckerTest : public testing::Test {
protected:
    void SetUp() override {}

    void TearDown() override {}

    // Helper function to create valid All2All semantics
    void CreateValidAll2AllSemantics(
        std::map<RankId, RankMemorySemantics>& allRankMemSemantics, All2AllDataDesTagInner& all2AllDataDes,
        u32 rankSize, u64 countPerRank)
    {
        all2AllDataDes.recvType = HcclDataType::HCCL_DATA_TYPE_INT32;
        all2AllDataDes.sendCountMatrix.clear();
        all2AllDataDes.sendCountMatrix.resize(rankSize * rankSize, countPerRank);

        u64 dataSize = countPerRank * CHECK_SIZE_TABLE[all2AllDataDes.recvType];

        for (RankId rankId = 0; rankId < rankSize; rankId++) {
            RankMemorySemantics rankMemSemantics;
            std::set<BufferSemantic> outputSemantics;

            u64 totalSize = 0;
            for (RankId srcRank = 0; srcRank < rankSize; srcRank++) {
                BufferSemantic bufSem(totalSize, dataSize);
                bufSem.srcBufs.insert(SrcBufDes(srcRank, BufferType::INPUT, rankId * dataSize));
                outputSemantics.insert(bufSem);
                totalSize += dataSize;
            }

            rankMemSemantics[BufferType::OUTPUT] = outputSemantics;
            allRankMemSemantics[rankId] = rankMemSemantics;
        }
    }
};

// Test normal case: valid All2All semantics with 2 ranks
TEST_F(All2AllSemanticsCheckerTest, ValidSemantics_TwoRanks)
{
    std::map<RankId, RankMemorySemantics> allRankMemSemantics;
    All2AllDataDesTagInner all2AllDataDes;
    u32 rankSize = 2;
    u64 countPerRank = 100;

    CreateValidAll2AllSemantics(allRankMemSemantics, all2AllDataDes, rankSize, countPerRank);

    HcclResult result = TaskCheckAll2AllSemantics(allRankMemSemantics, all2AllDataDes);
    EXPECT_EQ(result, HcclResult::HCCL_SUCCESS);
}

// Test normal case: valid All2All semantics with 4 ranks
TEST_F(All2AllSemanticsCheckerTest, ValidSemantics_FourRanks)
{
    std::map<RankId, RankMemorySemantics> allRankMemSemantics;
    All2AllDataDesTagInner all2AllDataDes;
    u32 rankSize = 4;
    u64 countPerRank = 100;

    CreateValidAll2AllSemantics(allRankMemSemantics, all2AllDataDes, rankSize, countPerRank);

    HcclResult result = TaskCheckAll2AllSemantics(allRankMemSemantics, all2AllDataDes);
    EXPECT_EQ(result, HcclResult::HCCL_SUCCESS);
}

// Test boundary case: single rank
TEST_F(All2AllSemanticsCheckerTest, ValidSemantics_SingleRank)
{
    std::map<RankId, RankMemorySemantics> allRankMemSemantics;
    All2AllDataDesTagInner all2AllDataDes;
    u32 rankSize = 1;
    u64 countPerRank = 100;

    CreateValidAll2AllSemantics(allRankMemSemantics, all2AllDataDes, rankSize, countPerRank);

    HcclResult result = TaskCheckAll2AllSemantics(allRankMemSemantics, all2AllDataDes);
    EXPECT_EQ(result, HcclResult::HCCL_SUCCESS);
}

// Test boundary case: zero count - no data transferred, buffer semantics should be empty
TEST_F(All2AllSemanticsCheckerTest, ValidSemantics_ZeroCount)
{
    std::map<RankId, RankMemorySemantics> allRankMemSemantics;
    All2AllDataDesTagInner all2AllDataDes;
    u32 rankSize = 2;
    u64 countPerRank = 0;

    all2AllDataDes.recvType = HcclDataType::HCCL_DATA_TYPE_INT32;
    all2AllDataDes.sendCountMatrix.clear();
    all2AllDataDes.sendCountMatrix.resize(rankSize * rankSize, countPerRank);

    for (RankId rankId = 0; rankId < rankSize; rankId++) {
        RankMemorySemantics rankMemSemantics;
        allRankMemSemantics[rankId] = rankMemSemantics;
    }

    HcclResult result = TaskCheckAll2AllSemantics(allRankMemSemantics, all2AllDataDes);
    EXPECT_EQ(result, HcclResult::HCCL_SUCCESS);
}

// Test abnormal case: missing rank
TEST_F(All2AllSemanticsCheckerTest, Abnormal_MissingRank)
{
    std::map<RankId, RankMemorySemantics> allRankMemSemantics;
    All2AllDataDesTagInner all2AllDataDes;
    u32 rankSize = 2;
    u64 countPerRank = 100;

    all2AllDataDes.recvType = HcclDataType::HCCL_DATA_TYPE_INT32;
    all2AllDataDes.sendCountMatrix.resize(rankSize * rankSize, countPerRank);

    // Only add rank 0
    u64 dataSize = countPerRank * CHECK_SIZE_TABLE[all2AllDataDes.recvType];
    RankMemorySemantics rankMemSemantics;
    std::set<BufferSemantic> outputSemantics;

    for (RankId srcRank = 0; srcRank < rankSize; srcRank++) {
        BufferSemantic bufSem(srcRank * dataSize, dataSize);
        bufSem.srcBufs.insert(SrcBufDes(srcRank, BufferType::INPUT, 0));
        outputSemantics.insert(bufSem);
    }

    rankMemSemantics[BufferType::OUTPUT] = outputSemantics;
    allRankMemSemantics[0] = rankMemSemantics;

    HcclResult result = TaskCheckAll2AllSemantics(allRankMemSemantics, all2AllDataDes);
    EXPECT_EQ(result, HcclResult::HCCL_E_PARA);
}

// Test abnormal case: wrong source rank
TEST_F(All2AllSemanticsCheckerTest, Abnormal_WrongSourceRank)
{
    std::map<RankId, RankMemorySemantics> allRankMemSemantics;
    All2AllDataDesTagInner all2AllDataDes;
    u32 rankSize = 2;
    u64 countPerRank = 100;

    all2AllDataDes.recvType = HcclDataType::HCCL_DATA_TYPE_INT32;
    all2AllDataDes.sendCountMatrix.resize(rankSize * rankSize, countPerRank);

    u64 dataSize = countPerRank * CHECK_SIZE_TABLE[all2AllDataDes.recvType];

    RankMemorySemantics rankMemSemantics;
    std::set<BufferSemantic> outputSemantics;

    BufferSemantic bufSem0(0, dataSize);
    bufSem0.srcBufs.insert(SrcBufDes(1, BufferType::INPUT, 0)); // Wrong source rank
    outputSemantics.insert(bufSem0);

    BufferSemantic bufSem1(dataSize, dataSize);
    bufSem1.srcBufs.insert(SrcBufDes(0, BufferType::INPUT, dataSize)); // Wrong source rank
    outputSemantics.insert(bufSem1);

    rankMemSemantics[BufferType::OUTPUT] = outputSemantics;
    allRankMemSemantics[0] = rankMemSemantics;
    allRankMemSemantics[1] = rankMemSemantics;

    HcclResult result = TaskCheckAll2AllSemantics(allRankMemSemantics, all2AllDataDes);
    EXPECT_EQ(result, HcclResult::HCCL_E_PARA);
}

// Test boundary case: large rank size
TEST_F(All2AllSemanticsCheckerTest, ValidSemantics_LargeRankSize)
{
    std::map<RankId, RankMemorySemantics> allRankMemSemantics;
    All2AllDataDesTagInner all2AllDataDes;
    u32 rankSize = 16;
    u64 countPerRank = 100;

    CreateValidAll2AllSemantics(allRankMemSemantics, all2AllDataDes, rankSize, countPerRank);

    HcclResult result = TaskCheckAll2AllSemantics(allRankMemSemantics, all2AllDataDes);
    EXPECT_EQ(result, HcclResult::HCCL_SUCCESS);
}
} // namespace HcclSim
