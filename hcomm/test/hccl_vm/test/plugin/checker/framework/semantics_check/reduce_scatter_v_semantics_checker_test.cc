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
#include <map>
#include <set>
#include <vector>

#include "check_utils.h"
#include "checker_def.h"
#include "reduce_scatter_v_semantics_checker.h"

namespace HcclSim {
class ReduceScatterVSemanticsCheckerTest : public testing::Test {
protected:
    void SetUp() override {}

    void TearDown() override {}

    // Helper function to create valid ReduceScatterV semantics
    void CreateValidReduceScatterVSemantics(
        std::map<RankId, RankMemorySemantics>& allRankMemSemantics, VDataDesTagInner& vDataDes, u32 rankSize,
        const std::vector<uint64_t>& counts, HcclReduceOp reduceType)
    {
        vDataDes.dataType = HcclDataType::HCCL_DATA_TYPE_INT32;
        vDataDes.counts.clear();
        vDataDes.displs.clear();

        u64 displsCount = 0;
        for (auto count : counts) {
            vDataDes.counts.push_back(count);
            vDataDes.displs.push_back(displsCount);
            displsCount += count;
        }

        for (RankId rankId = 0; rankId < rankSize; rankId++) {
            RankMemorySemantics rankMemSemantics;
            std::set<BufferSemantic> outputSemantics;

            u64 curDataSize = counts[rankId] * CHECK_SIZE_TABLE[vDataDes.dataType];
            if (curDataSize == 0) {
                allRankMemSemantics[rankId] = rankMemSemantics;
                continue;
            }

            BufferSemantic bufSem(0, curDataSize, true, reduceType);
            for (RankId srcRank = 0; srcRank < rankSize; srcRank++) {
                bufSem.srcBufs.insert(SrcBufDes(
                    srcRank, BufferType::INPUT, vDataDes.displs[rankId] * CHECK_SIZE_TABLE[vDataDes.dataType]));
            }
            outputSemantics.insert(bufSem);

            rankMemSemantics[BufferType::OUTPUT] = outputSemantics;
            allRankMemSemantics[rankId] = rankMemSemantics;
        }
    }
};

// Test normal case: valid ReduceScatterV semantics with equal counts
TEST_F(ReduceScatterVSemanticsCheckerTest, ValidSemantics_EqualCounts)
{
    std::map<RankId, RankMemorySemantics> allRankMemSemantics;
    VDataDesTagInner vDataDes;
    u32 rankSize = 4;
    std::vector<uint64_t> counts = {100, 100, 100, 100};

    CreateValidReduceScatterVSemantics(allRankMemSemantics, vDataDes, rankSize, counts, HcclReduceOp::HCCL_REDUCE_SUM);

    HcclResult result = TaskCheckReduceScatterVSemantics(allRankMemSemantics, HcclReduceOp::HCCL_REDUCE_SUM, vDataDes);
    EXPECT_EQ(result, HcclResult::HCCL_SUCCESS);
}

// Test normal case: valid ReduceScatterV semantics with different counts
TEST_F(ReduceScatterVSemanticsCheckerTest, ValidSemantics_DifferentCounts)
{
    std::map<RankId, RankMemorySemantics> allRankMemSemantics;
    VDataDesTagInner vDataDes;
    u32 rankSize = 4;
    std::vector<uint64_t> counts = {50, 100, 150, 200};

    CreateValidReduceScatterVSemantics(allRankMemSemantics, vDataDes, rankSize, counts, HcclReduceOp::HCCL_REDUCE_SUM);

    HcclResult result = TaskCheckReduceScatterVSemantics(allRankMemSemantics, HcclReduceOp::HCCL_REDUCE_SUM, vDataDes);
    EXPECT_EQ(result, HcclResult::HCCL_SUCCESS);
}

// Test boundary case: single rank
TEST_F(ReduceScatterVSemanticsCheckerTest, ValidSemantics_SingleRank)
{
    std::map<RankId, RankMemorySemantics> allRankMemSemantics;
    VDataDesTagInner vDataDes;
    u32 rankSize = 1;
    std::vector<uint64_t> counts = {100};

    CreateValidReduceScatterVSemantics(allRankMemSemantics, vDataDes, rankSize, counts, HcclReduceOp::HCCL_REDUCE_SUM);

    HcclResult result = TaskCheckReduceScatterVSemantics(allRankMemSemantics, HcclReduceOp::HCCL_REDUCE_SUM, vDataDes);
    EXPECT_EQ(result, HcclResult::HCCL_SUCCESS);
}

// Test boundary case: zero counts for some ranks
TEST_F(ReduceScatterVSemanticsCheckerTest, ValidSemantics_ZeroCounts)
{
    std::map<RankId, RankMemorySemantics> allRankMemSemantics;
    VDataDesTagInner vDataDes;
    u32 rankSize = 4;
    std::vector<uint64_t> counts = {0, 100, 0, 200};

    CreateValidReduceScatterVSemantics(allRankMemSemantics, vDataDes, rankSize, counts, HcclReduceOp::HCCL_REDUCE_SUM);

    HcclResult result = TaskCheckReduceScatterVSemantics(allRankMemSemantics, HcclReduceOp::HCCL_REDUCE_SUM, vDataDes);
    EXPECT_EQ(result, HcclResult::HCCL_SUCCESS);
}

// Test abnormal case: missing rank
TEST_F(ReduceScatterVSemanticsCheckerTest, Abnormal_MissingRank)
{
    std::map<RankId, RankMemorySemantics> allRankMemSemantics;
    VDataDesTagInner vDataDes;
    u32 rankSize = 2;
    std::vector<uint64_t> counts = {100, 100};

    vDataDes.dataType = HcclDataType::HCCL_DATA_TYPE_INT32;
    vDataDes.counts = counts;
    vDataDes.displs = {0, 100};

    // Only add rank 0
    RankMemorySemantics rankMemSemantics;
    std::set<BufferSemantic> outputSemantics;

    u64 dataSize = counts[0] * CHECK_SIZE_TABLE[vDataDes.dataType];
    BufferSemantic bufSem(0, dataSize, true, HcclReduceOp::HCCL_REDUCE_SUM);
    bufSem.srcBufs.insert(SrcBufDes(0, BufferType::INPUT, 0));
    bufSem.srcBufs.insert(SrcBufDes(1, BufferType::INPUT, 0));
    outputSemantics.insert(bufSem);

    rankMemSemantics[BufferType::OUTPUT] = outputSemantics;
    allRankMemSemantics[0] = rankMemSemantics;

    HcclResult result = TaskCheckReduceScatterVSemantics(allRankMemSemantics, HcclReduceOp::HCCL_REDUCE_SUM, vDataDes);
    EXPECT_EQ(result, HcclResult::HCCL_E_PARA);
}

// Test abnormal case: wrong reduce type
TEST_F(ReduceScatterVSemanticsCheckerTest, Abnormal_WrongReduceType)
{
    std::map<RankId, RankMemorySemantics> allRankMemSemantics;
    VDataDesTagInner vDataDes;
    u32 rankSize = 2;
    std::vector<uint64_t> counts = {100, 100};

    CreateValidReduceScatterVSemantics(allRankMemSemantics, vDataDes, rankSize, counts, HcclReduceOp::HCCL_REDUCE_SUM);

    HcclResult result = TaskCheckReduceScatterVSemantics(allRankMemSemantics, HcclReduceOp::HCCL_REDUCE_PROD, vDataDes);
    EXPECT_EQ(result, HcclResult::HCCL_E_PARA);
}
} // namespace HcclSim
