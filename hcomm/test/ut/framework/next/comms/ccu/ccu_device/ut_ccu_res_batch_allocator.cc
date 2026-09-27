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
#include <mockcpp/mockcpp.hpp>

#include "src/base_comm/resources/ccu/ccu_device/ccu_res_batch_allocator.h"
#include "ccu_comp.h"
#include "hccl_common.h"

#define private public
#define protected public

using namespace hcomm;

static void InitTestMissionReq(MissionReq& missionReq, MissionReqType reqType)
{
    missionReq.reqType = reqType;
    missionReq.req[0] = 0;
    missionReq.req[1] = 0;
}

class CcuResBatchAllocatorTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        mockcpp::GlobalMockObject::verify();
        mockcpp::GlobalMockObject::reset();
    }
    static void TearDownTestCase()
    {
        mockcpp::GlobalMockObject::verify();
        mockcpp::GlobalMockObject::reset();
    }
    void SetUp() override
    {
        mockcpp::GlobalMockObject::verify();
        mockcpp::GlobalMockObject::reset();
    }
    void TearDown() override
    {
        mockcpp::GlobalMockObject::verify();
        mockcpp::GlobalMockObject::reset();
    }
};

TEST_F(CcuResBatchAllocatorTest, Ut_Alloc_When_ReqTypeNotDefault_Expect_ForceSetToDefault)
{
    hcomm::CcuResBatchAllocator allocat{};
    uintptr_t handleKey;
    MissionReq missionReq;
    MissionResInfo missionInfos;

    // 调用公共函数
    InitTestMissionReq(missionReq, MissionReqType::COMM_ENGINE_RESERVED);

    auto ret = allocat.missionMgr_.Alloc(handleKey, missionReq, missionInfos);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(CcuResBatchAllocatorTest, Ut_Alloc_When_ReqNumIs0_Expect_ReturnSuccessDirectly)
{
    hcomm::CcuResBatchAllocator allocat{};
    uintptr_t handleKey;
    MissionReq missionReq;
    MissionResInfo missionInfos;

    // 调用公共函数
    InitTestMissionReq(missionReq, MissionReqType::FUSION_MULTIPLE_DIE);

    auto ret = allocat.missionMgr_.Alloc(handleKey, missionReq, missionInfos);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(CcuResBatchAllocatorTest, Ut_QueryRemainRes_When_NoAllocations_Expect_MaxGapEqualsTotalCapacity)
{
    CcuResBatchAllocator allocator;
    allocator.dieEnableFlags_[0] = true;
    allocator.userDevId_ = 0;

    allocator.resStrategies_[0].loopNum = 8;
    allocator.maxResBlockNums_.loopNum = 4; // poolSize = 4 * 8 = 32
    allocator.resBlocks_[0][ResType::LOOP] = {
        {0, 0, 8, 0, false},
        {1, 8, 8, 0, false},
        {2, 16, 8, 0, false},
        {3, 24, 8, 0, false},
    };

    uint32_t remainNum = 0;
    HcclResult ret = allocator.QueryRemainRes(0, ResType::LOOP, remainNum);

    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(remainNum, 32u) << "no allocations -> max gap = pool size 32";
}

TEST_F(CcuResBatchAllocatorTest, Ut_QueryRemainRes_When_WithAllocations_Expect_MaxContiguousGap)
{
    CcuResBatchAllocator allocator;
    allocator.dieEnableFlags_[0] = true;
    allocator.userDevId_ = 0;

    allocator.maxResBlockNums_.loopNum = 8; // poolSize = 8 * 8 = 64
    allocator.resBlocks_[0][ResType::LOOP] = {
        {0, 0, 8, 0, true},   // allocated
        {1, 8, 8, 0, true},   // allocated
        {2, 16, 8, 0, false}, // free
        {3, 24, 8, 0, true},  // allocated
        {4, 32, 8, 0, false}, // free
        {5, 40, 8, 0, false}, // free
        {6, 48, 8, 0, false}, // free
        {7, 56, 8, 0, false}, // free
    };

    uint32_t remainNum = 0;
    HcclResult ret = allocator.QueryRemainRes(0, ResType::LOOP, remainNum);

    EXPECT_EQ(ret, HCCL_SUCCESS);
    // blocks 4-7 连续空闲: 4 × 8 = 32
    EXPECT_EQ(remainNum, 32u);
}

TEST_F(CcuResBatchAllocatorTest, Ut_QueryRemainRes_When_AllAllocated_Expect_Zero)
{
    CcuResBatchAllocator allocator;
    allocator.dieEnableFlags_[0] = true;
    allocator.userDevId_ = 0;

    allocator.maxResBlockNums_.loopNum = 4; // poolSize = 4 * 8 = 32
    allocator.resBlocks_[0][ResType::LOOP] = {
        {0, 0, 8, 0, true},
        {1, 8, 8, 0, true},
        {2, 16, 8, 0, true},
        {3, 24, 8, 0, true},
    };

    uint32_t remainNum = 0;
    HcclResult ret = allocator.QueryRemainRes(0, ResType::LOOP, remainNum);

    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(remainNum, 0u) << "all allocated -> zero remaining";
}

TEST_F(CcuResBatchAllocatorTest, Ut_QueryRemainRes_When_HoleAtStart_Expect_InitialGap)
{
    CcuResBatchAllocator allocator;
    allocator.dieEnableFlags_[0] = true;
    allocator.userDevId_ = 0;

    allocator.maxResBlockNums_.loopNum = 6;
    allocator.resBlocks_[0][ResType::LOOP] = {
        {0, 0, 8, 0, true},   {1, 8, 8, 0, true},   {2, 16, 8, 0, false},
        {3, 24, 8, 0, false}, {4, 32, 8, 0, false}, {5, 40, 8, 0, false},
    };

    uint32_t remainNum = 0;
    HcclResult ret = allocator.QueryRemainRes(0, ResType::LOOP, remainNum);

    EXPECT_EQ(ret, HCCL_SUCCESS);
    // 4 free blocks × 8 = 32
    EXPECT_EQ(remainNum, 32u);
}

TEST_F(CcuResBatchAllocatorTest, Ut_QueryRemainRes_When_HoleAtEnd_Expect_TailGap)
{
    CcuResBatchAllocator allocator;
    allocator.dieEnableFlags_[0] = true;
    allocator.userDevId_ = 0;

    allocator.maxResBlockNums_.loopNum = 6;
    allocator.resBlocks_[0][ResType::LOOP] = {
        {0, 0, 8, 0, false},  {1, 8, 8, 0, false}, {2, 16, 8, 0, false},
        {3, 24, 8, 0, false}, {4, 32, 8, 0, true}, {5, 40, 8, 0, true},
    };

    uint32_t remainNum = 0;
    HcclResult ret = allocator.QueryRemainRes(0, ResType::LOOP, remainNum);

    EXPECT_EQ(ret, HCCL_SUCCESS);
    // 4 free blocks × 8 = 32
    EXPECT_EQ(remainNum, 32u);
}

TEST_F(CcuResBatchAllocatorTest, Ut_QueryRemainRes_When_PreReservedMSBlocks_Expect_MaxGapReducedByReserved)
{
    CcuResBatchAllocator allocator;
    allocator.dieEnableFlags_[0] = true;
    allocator.userDevId_ = 0;

    allocator.resStrategies_[0].msNum = 64;
    allocator.maxResBlockNums_.msNum = 4; // poolSize = 4 * 64 = 256

    // Block0: 预预留(allocated), Block1: 空闲, Block2: 已分配, Block3: 空闲
    allocator.resBlocks_[0][ResType::MS] = {
        {0, 0, 64, 0, true},
        {1, 64, 64, 0, false},
        {2, 128, 64, 0, true},
        {3, 192, 64, 0, false},
    };

    uint32_t remainNum = 0;
    HcclResult ret = allocator.QueryRemainRes(0, ResType::MS, remainNum);

    EXPECT_EQ(ret, HCCL_SUCCESS);
    // 最大连续空闲: 单 block = 64
    EXPECT_EQ(remainNum, 64u) << "pre-reserved + allocated block leave max gap 64";
}

TEST_F(CcuResBatchAllocatorTest, Ut_QueryRemainRes_When_PoolStartsAtNonZero_Expect_GapWithinPoolOnly)
{
    CcuResBatchAllocator allocator;
    allocator.dieEnableFlags_[0] = true;
    allocator.userDevId_ = 0;

    // 模拟 mission 预分配池 [100, 356), 4 blocks of 64
    allocator.missionMgr_.blocks_ = {
        {0, 100, 64, 0, true},  // 已分配
        {1, 164, 64, 0, false}, // 空闲
        {2, 228, 64, 0, true},  // 已分配
        {3, 292, 64, 0, false}, // 空闲
    };

    uint32_t remainNum = 0;
    HcclResult ret = allocator.QueryRemainRes(0, ResType::MISSION, remainNum);

    EXPECT_EQ(ret, HCCL_SUCCESS);
    // 最大连续空闲: 单 block = 64
    EXPECT_EQ(remainNum, 64u) << "max gap 64 within pool only";
}

TEST_F(CcuResBatchAllocatorTest, Ut_GetConsecutiveRemainSize_When_PartialAlloc_Expect_MaxGap)
{
    CcuResIdAllocator allocator(100);
    allocator.resInfos_ = {{0, 20}, {70, 20}};

    EXPECT_EQ(allocator.GetConsecutiveRemainSize(), 50u);
}

TEST_F(CcuResBatchAllocatorTest, Ut_GetConsecutiveRemainSize_When_Full_Expect_Zero)
{
    CcuResIdAllocator allocator(50);
    std::vector<ResInfo> resInfos;
    allocator.Alloc(50, true, resInfos);

    EXPECT_EQ(allocator.GetConsecutiveRemainSize(), 0u);
}
