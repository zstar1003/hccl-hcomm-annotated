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

#include "ccu_res_allocator.h"
#include "ccu_res_specs.h"
#include "hccl_common.h"

#define private public
#define protected public

using namespace hcomm;

class CcuResIdAllocatorTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(CcuResIdAllocatorTest, Ut_Alloc_When_NumIsZero_Expect_ReturnHCCL_E_PARA)
{
    CcuResIdAllocator allocator(100);
    std::vector<ResInfo> resInfos;
    EXPECT_EQ(allocator.Alloc(0, false, resInfos), HcclResult::HCCL_E_PARA);
}

TEST_F(CcuResIdAllocatorTest, Ut_Alloc_When_ExceedCapacity_Expect_ReturnHCCL_E_UNAVAIL)
{
    CcuResIdAllocator allocator(100);
    std::vector<ResInfo> resInfos;
    EXPECT_EQ(allocator.Alloc(101, false, resInfos), HcclResult::HCCL_E_UNAVAIL);
}

TEST_F(CcuResIdAllocatorTest, Ut_Alloc_When_NonConsecutiveNormal_Expect_ReturnSuccess)
{
    CcuResIdAllocator allocator(100);
    std::vector<ResInfo> resInfos;
    EXPECT_EQ(allocator.Alloc(10, false, resInfos), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(resInfos.size(), 1U);
    EXPECT_EQ(resInfos[0].startId, 0U);
    EXPECT_EQ(resInfos[0].num, 10U);
    EXPECT_EQ(allocator.allocatedSize_, 10U);
}

TEST_F(CcuResIdAllocatorTest, Ut_Alloc_When_ConsecutiveNormal_Expect_ReturnSuccess)
{
    CcuResIdAllocator allocator(100);
    std::vector<ResInfo> resInfos;
    EXPECT_EQ(allocator.Alloc(10, true, resInfos), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(resInfos.size(), 1U);
    EXPECT_EQ(resInfos[0].startId, 0U);
    EXPECT_EQ(resInfos[0].num, 10U);
}

TEST_F(CcuResIdAllocatorTest, Ut_Alloc_When_ConsecutiveButFragmented_Expect_ReturnHCCL_E_UNAVAIL)
{
    CcuResIdAllocator allocator(20);
    std::vector<ResInfo> resInfos;
    ASSERT_EQ(allocator.Alloc(10, false, resInfos), HcclResult::HCCL_SUCCESS);
    ASSERT_EQ(allocator.Release(0, 5), HcclResult::HCCL_SUCCESS);
    std::vector<ResInfo> resInfos2;
    EXPECT_EQ(allocator.Alloc(15, true, resInfos2), HcclResult::HCCL_E_UNAVAIL);
}

TEST_F(CcuResIdAllocatorTest, Ut_Alloc_When_MultipleNonConsecutive_Expect_ReturnSuccess)
{
    CcuResIdAllocator allocator(100);
    std::vector<ResInfo> resInfos1;
    ASSERT_EQ(allocator.Alloc(10, false, resInfos1), HcclResult::HCCL_SUCCESS);
    std::vector<ResInfo> resInfos2;
    ASSERT_EQ(allocator.Alloc(10, false, resInfos2), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(resInfos2[0].startId, 10U);
    std::vector<ResInfo> resInfos3;
    ASSERT_EQ(allocator.Alloc(80, false, resInfos3), HcclResult::HCCL_SUCCESS);
}

TEST_F(CcuResIdAllocatorTest, Ut_Alloc_When_PartialFragmentThenConsecutive_Expect_ReturnSuccess)
{
    CcuResIdAllocator allocator(100);
    std::vector<ResInfo> resInfos1;
    ASSERT_EQ(allocator.Alloc(10, false, resInfos1), HcclResult::HCCL_SUCCESS);
    ASSERT_EQ(allocator.Release(0, 5), HcclResult::HCCL_SUCCESS);
    std::vector<ResInfo> resInfos2;
    EXPECT_EQ(allocator.Alloc(5, true, resInfos2), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(resInfos2[0].startId, 0U);
}

TEST_F(CcuResIdAllocatorTest, Ut_Release_When_NumIsZero_Expect_ReturnHCCL_E_PARA)
{
    CcuResIdAllocator allocator(100);
    EXPECT_EQ(allocator.Release(0, 0), HcclResult::HCCL_E_PARA);
}

TEST_F(CcuResIdAllocatorTest, Ut_Release_When_NumExceedsCapacity_Expect_ReturnHCCL_E_PARA)
{
    CcuResIdAllocator allocator(100);
    EXPECT_EQ(allocator.Release(0, 101), HcclResult::HCCL_E_PARA);
}

TEST_F(CcuResIdAllocatorTest, Ut_Release_When_StartIdOutOfRange_Expect_ReturnHCCL_E_PARA)
{
    CcuResIdAllocator allocator(100);
    EXPECT_EQ(allocator.Release(99, 2), HcclResult::HCCL_E_PARA);
}

TEST_F(CcuResIdAllocatorTest, Ut_Release_When_NotAllocated_Expect_ReturnHCCL_E_PARA)
{
    CcuResIdAllocator allocator(100);
    EXPECT_EQ(allocator.Release(0, 10), HcclResult::HCCL_E_PARA);
}

TEST_F(CcuResIdAllocatorTest, Ut_Release_When_NumExceedsAllocated_Expect_ReturnHCCL_E_PARA)
{
    CcuResIdAllocator allocator(100);
    std::vector<ResInfo> resInfos;
    ASSERT_EQ(allocator.Alloc(10, false, resInfos), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(allocator.Release(0, 20), HcclResult::HCCL_E_PARA);
}

TEST_F(CcuResIdAllocatorTest, Ut_Release_When_WholeBlock_Expect_ReturnSuccess)
{
    CcuResIdAllocator allocator(100);
    std::vector<ResInfo> resInfos;
    ASSERT_EQ(allocator.Alloc(10, false, resInfos), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(allocator.Release(0, 10), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(allocator.allocatedSize_, 0U);
    EXPECT_TRUE(allocator.resInfos_.empty());
}

TEST_F(CcuResIdAllocatorTest, Ut_Release_When_HeadOfBlock_Expect_ReturnSuccess)
{
    CcuResIdAllocator allocator(100);
    std::vector<ResInfo> resInfos;
    ASSERT_EQ(allocator.Alloc(10, false, resInfos), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(allocator.Release(0, 3), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(allocator.resInfos_[0].startId, 3U);
    EXPECT_EQ(allocator.resInfos_[0].num, 7U);
}

TEST_F(CcuResIdAllocatorTest, Ut_Release_When_TailOfBlock_Expect_ReturnSuccess)
{
    CcuResIdAllocator allocator(100);
    std::vector<ResInfo> resInfos;
    ASSERT_EQ(allocator.Alloc(10, false, resInfos), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(allocator.Release(7, 3), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(allocator.resInfos_[0].startId, 0U);
    EXPECT_EQ(allocator.resInfos_[0].num, 7U);
}

TEST_F(CcuResIdAllocatorTest, Ut_Release_When_MiddleOfBlock_Expect_SplitBlock)
{
    CcuResIdAllocator allocator(100);
    std::vector<ResInfo> resInfos;
    ASSERT_EQ(allocator.Alloc(10, false, resInfos), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(allocator.Release(3, 4), HcclResult::HCCL_SUCCESS);
    ASSERT_EQ(allocator.resInfos_.size(), 2U);
    EXPECT_EQ(allocator.resInfos_[0].startId, 0U);
    EXPECT_EQ(allocator.resInfos_[0].num, 3U);
    EXPECT_EQ(allocator.resInfos_[1].startId, 7U);
    EXPECT_EQ(allocator.resInfos_[1].num, 3U);
}

TEST_F(CcuResIdAllocatorTest, Ut_AllocReleaseReuse_Expect_ReturnSuccess)
{
    CcuResIdAllocator allocator(100);
    std::vector<ResInfo> resInfos;
    ASSERT_EQ(allocator.Alloc(10, false, resInfos), HcclResult::HCCL_SUCCESS);
    ASSERT_EQ(allocator.Release(0, 10), HcclResult::HCCL_SUCCESS);
    std::vector<ResInfo> resInfos2;
    EXPECT_EQ(allocator.Alloc(10, false, resInfos2), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(resInfos2[0].startId, 0U);
}

TEST_F(CcuResIdAllocatorTest, Ut_Describe_Expect_ReturnNonEmpty)
{
    CcuResIdAllocator allocator(100);
    std::string desc = allocator.Describe();
    EXPECT_FALSE(desc.empty());
}

class CcuResAllocatorTest : public testing::Test {
protected:
    void SetUp() override
    {
        auto& specs = CcuResSpecifications::GetInstance(0);
        specs.initFlag_ = true;
        specs.ccuVersion_ = CcuVersion::CCU_V2;
        specs.dieEnableFlags_[0] = true;
        specs.resSpecs_[0].loopEngineNum = 200;
        specs.resSpecs_[0].msNum = 1536;
        specs.resSpecs_[0].ckeNum = 1024;
        specs.resSpecs_[0].xnNum = 3072;
        specs.resSpecs_[0].countXnNum = 4096;
        specs.resSpecs_[0].gsaNum = 3072;
        specs.resSpecs_[0].instructionNum = 32768;
        specs.resSpecs_[0].missionNum = 16;
    }
    void TearDown() override {}
};

TEST_F(CcuResAllocatorTest, Ut_Init_When_Normal_Expect_ReturnSuccess)
{
    CcuResAllocator allocator(0, 0);
    EXPECT_EQ(allocator.Init(), HcclResult::HCCL_SUCCESS);
    EXPECT_FALSE(allocator.idAllocatorMap_.empty());
}

TEST_F(CcuResAllocatorTest, Ut_Alloc_When_Normal_Expect_ReturnSuccess)
{
    CcuResAllocator allocator(0, 0);
    ASSERT_EQ(allocator.Init(), HcclResult::HCCL_SUCCESS);
    std::vector<ResInfo> resInfos;
    EXPECT_EQ(allocator.Alloc(ResType::XN, 10, false, resInfos), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(resInfos.size(), 1U);
    EXPECT_EQ(resInfos[0].startId, 0U);
    EXPECT_EQ(resInfos[0].num, 10U);
}

TEST_F(CcuResAllocatorTest, Ut_Alloc_When_InvalidResType_Expect_ReturnHCCL_E_PARA)
{
    CcuResAllocator allocator(0, 0);
    ASSERT_EQ(allocator.Init(), HcclResult::HCCL_SUCCESS);
    std::vector<ResInfo> resInfos;
    EXPECT_EQ(allocator.Alloc(ResType::INVALID, 10, false, resInfos), HcclResult::HCCL_E_PARA);
}

TEST_F(CcuResAllocatorTest, Ut_Release_When_Normal_Expect_ReturnSuccess)
{
    CcuResAllocator allocator(0, 0);
    ASSERT_EQ(allocator.Init(), HcclResult::HCCL_SUCCESS);
    std::vector<ResInfo> resInfos;
    ASSERT_EQ(allocator.Alloc(ResType::XN, 10, false, resInfos), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(allocator.Release(ResType::XN, 0, 10), HcclResult::HCCL_SUCCESS);
}

TEST_F(CcuResAllocatorTest, Ut_Release_When_InvalidResType_Expect_ReturnHCCL_E_PARA)
{
    CcuResAllocator allocator(0, 0);
    ASSERT_EQ(allocator.Init(), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(allocator.Release(ResType::INVALID, 0, 10), HcclResult::HCCL_E_PARA);
}

TEST_F(CcuResAllocatorTest, Ut_AllocCountXn_When_Normal_Expect_ReturnSuccess)
{
    CcuResAllocator allocator(0, 0);
    ASSERT_EQ(allocator.Init(), HcclResult::HCCL_SUCCESS);
    ResInfo resInfo;
    EXPECT_EQ(allocator.AllocCountXn(10, resInfo), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(resInfo.num, 10U);
}

TEST_F(CcuResAllocatorTest, Ut_AllocCountXn_When_FallbackToXn_Expect_ReturnSuccess)
{
    CcuResAllocator allocator(0, 0);
    ASSERT_EQ(allocator.Init(), HcclResult::HCCL_SUCCESS);
    std::vector<ResInfo> allRes;
    ASSERT_EQ(allocator.Alloc(ResType::COUNT_XN, 4096, true, allRes), HcclResult::HCCL_SUCCESS);
    ResInfo resInfo;
    EXPECT_EQ(allocator.AllocCountXn(10, resInfo), HcclResult::HCCL_SUCCESS);
}

TEST_F(CcuResAllocatorTest, Ut_ReleaseCountXn_When_NormalXn_Expect_ReturnSuccess)
{
    CcuResAllocator allocator(0, 0);
    ASSERT_EQ(allocator.Init(), HcclResult::HCCL_SUCCESS);
    ResInfo resInfo;
    ASSERT_EQ(allocator.AllocCountXn(10, resInfo), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(allocator.ReleaseCountXn(resInfo.startId, resInfo.num), HcclResult::HCCL_SUCCESS);
}

TEST_F(CcuResAllocatorTest, Ut_ReleaseCountXn_When_BelowXnSpecNum_Expect_DelegateToXn)
{
    CcuResAllocator allocator(0, 0);
    ASSERT_EQ(allocator.Init(), HcclResult::HCCL_SUCCESS);
    std::vector<ResInfo> resInfos;
    ASSERT_EQ(allocator.Alloc(ResType::XN, 10, false, resInfos), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(allocator.ReleaseCountXn(0, 10), HcclResult::HCCL_SUCCESS);
}

TEST_F(CcuResAllocatorTest, Ut_Describe_Expect_ReturnNonEmpty)
{
    CcuResAllocator allocator(0, 0);
    ASSERT_EQ(allocator.Init(), HcclResult::HCCL_SUCCESS);
    std::string desc = allocator.Describe();
    EXPECT_FALSE(desc.empty());
}
