/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include <string>

#include "topoinfo_ranktable_partition.h"

using namespace hccl;

namespace {
RankInfo_t MakeRank(u32 rankId, const std::string& superPodId)
{
    RankInfo_t info;
    info.rankId = rankId;
    info.superPodId = superPodId;
    info.originalSuperPodId = superPodId;
    return info;
}
} // namespace

class TopoinfoRanktablePartitionTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "TopoinfoRanktablePartitionTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "TopoinfoRanktablePartitionTest TearDown" << std::endl; }

    virtual void SetUp() {}

    virtual void TearDown() {}

    HcclCommParams params_;
    RankTable_t globalRankTable_;
};

// 同一原始超节点内 rankId 连续，不触发分裂
TEST_F(TopoinfoRanktablePartitionTest, GenerateSubSuperPodId_WhenRankIdsConsecutive_ExpectNoSplit)
{
    TopoinfoRanktablePartition partition(params_, globalRankTable_);

    RankTable_t subRankTable;
    subRankTable.rankList = {MakeRank(0, "SP0"), MakeRank(1, "SP0"), MakeRank(2, "SP0")};

    HcclResult ret = partition.GenerateSubSuperPodId(subRankTable);

    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(subRankTable.superPodNum, 1u);
    for (const auto& rank : subRankTable.rankList) {
        EXPECT_EQ(rank.superPodId, "SP0");
    }
}

// 同一原始超节点内 rankId 不连续，断点处分裂出新逻辑超节点
TEST_F(TopoinfoRanktablePartitionTest, GenerateSubSuperPodId_WhenRankIdsNotConsecutive_ExpectSplit)
{
    TopoinfoRanktablePartition partition(params_, globalRankTable_);

    RankTable_t subRankTable;
    subRankTable.rankList = {MakeRank(0, "SP0"), MakeRank(1, "SP0"), MakeRank(5, "SP0"), MakeRank(6, "SP0")};

    HcclResult ret = partition.GenerateSubSuperPodId(subRankTable);

    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(subRankTable.superPodNum, 2u);
    EXPECT_EQ(subRankTable.rankList[0].superPodId, "SP0");
    EXPECT_EQ(subRankTable.rankList[1].superPodId, "SP0");
    EXPECT_EQ(subRankTable.rankList[2].superPodId, "SP0_HCCLSPLIT_0");
    EXPECT_EQ(subRankTable.rankList[3].superPodId, "SP0_HCCLSPLIT_0");
}
