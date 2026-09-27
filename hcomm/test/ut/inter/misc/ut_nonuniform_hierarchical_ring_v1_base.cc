/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include <mockcpp/mockcpp.hpp>
#include <vector>
#include <algorithm>
#include <iterator>
#include "nonuniform_hierarchical_ring_v1_base_pub.h"

using namespace hccl;

class RingInfoTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "RingInfoTest SetUP" << std::endl; }
    static void TearDownTestCase() { std::cout << "RingInfoTest TearDown" << std::endl; }
    // Some expensive resource shared by all tests.
    virtual void SetUp()
    {
        s32 portNum = 7;
        MOCKER(hrtGetHccsPortNum).stubs().with(mockcpp::any(), outBound(portNum)).will(returnValue(HCCL_SUCCESS));
        std::cout << "A Test SetUP" << std::endl;
    }
    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test TearDown" << std::endl;
    }
};

class RefRingInfo {
public:
    RefRingInfo(const std::vector<std::vector<s32>> refMatrix) : refMatrix_(refMatrix) {}
    ~RefRingInfo() {}

    s32 GetVIndex(u32 rank)
    {
        for (u32 vIndex = 0; vIndex < refMatrix_.size(); vIndex++) {
            const std::vector<s32>& refRow = refMatrix_[vIndex];
            if (std::find(refRow.begin(), refRow.end(), rank) != refRow.end())
                return vIndex;
        }
        return -1;
    }

    s32 GetHIndex(u32 rank)
    {
        for (auto& refRow : refMatrix_) {
            std::vector<s32>::const_iterator it = std::find(refRow.begin(), refRow.end(), rank);
            if (it != refRow.end())
                return std::distance(refRow.begin(), it);
        }
        return -1;
    }

    s32 GetVSizeByRank(u32 rank)
    {
        s32 hIndex = GetHIndex(rank);
        return (hIndex == -1) ? -1 : GetVSizeByHIndex(hIndex);
    }

    s32 GetVSizeByHIndex(u32 hIndex)
    {
        for (u32 vIndex = 0; vIndex < refMatrix_.size(); vIndex++) {
            if (refMatrix_[vIndex][hIndex] == -1)
                return vIndex;
        }
        return refMatrix_.size();
    }

    s32 GetHSizeByRank(u32 rank)
    {
        s32 vIndex = GetVIndex(rank);
        return (vIndex == -1) ? -1 : GetHSizeByVIndex(vIndex);
    }

    s32 GetHSizeByVIndex(u32 vIndex)
    {
        for (u32 hIndex = 0; hIndex < refMatrix_[vIndex].size(); hIndex++) {
            if (refMatrix_[vIndex][hIndex] == -1)
                return hIndex;
        }
        return refMatrix_[vIndex].size();
    }

    s32 GetRank(u32 vIndex, u32 hIndex) { return refMatrix_[vIndex][hIndex]; }

private:
    const std::vector<std::vector<s32>> refMatrix_;
};

void testRingInfo(u32 rankSize, const std::vector<std::vector<s32>> refMatrix)
{
    RingInfo info = RingInfo(rankSize);
    RefRingInfo refInfo = RefRingInfo(refMatrix);

    for (u32 rank = 0; rank < rankSize; rank++) {
        EXPECT_EQ(info.GetVIndex(rank), refInfo.GetVIndex(rank));
        EXPECT_EQ(info.GetHIndex(rank), refInfo.GetHIndex(rank));
        EXPECT_EQ(info.GetVSizeByRank(rank), refInfo.GetVSizeByRank(rank));
        EXPECT_EQ(info.GetHSizeByRank(rank), refInfo.GetHSizeByRank(rank));
        EXPECT_EQ(info.GetRank(refInfo.GetVIndex(rank), refInfo.GetHIndex(rank)), rank);
    }
}

TEST_F(RingInfoTest, ringInfo_8p)
{
    const std::vector<std::vector<s32>> refMatrix = {{0, 1, 2, 3}, {4, 5, 6, 7}};
    testRingInfo(/*rankSize=*/8, refMatrix);
}

TEST_F(RingInfoTest, ringInfo_36p)
{
    const std::vector<std::vector<s32>> refMatrix
        = {{0, 1, 2, 3, 4, 5},       {6, 7, 8, 9, 10, 11},     {12, 13, 14, 15, 16, 17},
           {18, 19, 20, 21, 22, 23}, {24, 25, 26, 27, 28, 29}, {30, 31, 32, 33, 34, 35}};
    testRingInfo(/*rankSize=*/36, refMatrix);
}

TEST_F(RingInfoTest, ringInfo_37p)
{
    const std::vector<std::vector<s32>> refMatrix
        = {{0, 1, 2, 3, 4, 5, 6},        {7, 8, 9, 10, 11, 12, -1},    {13, 14, 15, 16, 17, 18, -1},
           {19, 20, 21, 22, 23, 24, -1}, {25, 26, 27, 28, 29, 30, -1}, {31, 32, 33, 34, 35, 36, -1}};
    testRingInfo(/*rankSize=*/37, refMatrix);
}

TEST_F(RingInfoTest, ringInfo_42p)
{
    const std::vector<std::vector<s32>> refMatrix
        = {{0, 1, 2, 3, 4, 5, 6},        {7, 8, 9, 10, 11, 12, 13},    {14, 15, 16, 17, 18, 19, 20},
           {21, 22, 23, 24, 25, 26, 27}, {28, 29, 30, 31, 32, 33, 34}, {35, 36, 37, 38, 39, 40, 41}};
    testRingInfo(/*rankSize=*/42, refMatrix);
}

TEST_F(RingInfoTest, ringInfo_43p)
{
    const std::vector<std::vector<s32>> refMatrix
        = {{0, 1, 2, 3, 4, 5, 6},        {7, 8, 9, 10, 11, 12, -1},    {13, 14, 15, 16, 17, 18, -1},
           {19, 20, 21, 22, 23, 24, -1}, {25, 26, 27, 28, 29, 30, -1}, {31, 32, 33, 34, 35, 36, -1},
           {37, 38, 39, 40, 41, 42, -1}};
    testRingInfo(/*rankSize=*/43, refMatrix);
}

TEST_F(RingInfoTest, ringInfo_48p)
{
    const std::vector<std::vector<s32>> refMatrix
        = {{0, 1, 2, 3, 4, 5, 6, 7},         {8, 9, 10, 11, 12, 13, 14, 15},   {16, 17, 18, 19, 20, 21, 22, 23},
           {24, 25, 26, 27, 28, 29, 30, 31}, {32, 33, 34, 35, 36, 37, 38, 39}, {40, 41, 42, 43, 44, 45, 46, 47}};
    testRingInfo(/*rankSize=*/48, refMatrix);
}
