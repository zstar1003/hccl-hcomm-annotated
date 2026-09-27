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

#include "aiv_resource_manager.h"
#include "sim_op_db_types.h"
#include "sim_common_defs.h"

namespace {
constexpr uint64_t EXPECTED_AIV_COMM_INFO_SIZE = 65ULL * 1024ULL * 1024ULL;
static_assert(
    AivCommInfoLayout::SIZE_BYTES == EXPECTED_AIV_COMM_INFO_SIZE,
    "AIV commInfo size must match the HCCL ping-pong protocol.");
static_assert(
    AivCommInfoLayout::PING_OFFSET < AivCommInfoLayout::PONG_OFFSET, "AIV ping buffer must precede the pong buffer.");
static_assert(AivCommInfoLayout::PONG_OFFSET < AivCommInfoLayout::SIZE_BYTES, "AIV pong buffer must fit in commInfo.");

sim::OpMemInfoTab EmptyOpMemInfo() { return sim::OpMemInfoTab{}; }
} // namespace

class AivResourceManagerTest : public testing::Test {
protected:
    void SetUp() override { AivResourceManager::GetInstance().Reset(); }

    void TearDown() override { AivResourceManager::GetInstance().Reset(); }
};

TEST_F(AivResourceManagerTest, GetInstance_ReturnsSameSingleton)
{
    AivResourceManager& mgr1 = AivResourceManager::GetInstance();
    AivResourceManager& mgr2 = AivResourceManager::GetInstance();
    EXPECT_EQ(&mgr1, &mgr2);
}

TEST_F(AivResourceManagerTest, Init_RankSizeZero_ReturnsError)
{
    auto ret = AivResourceManager::GetInstance().Init(0, EmptyOpMemInfo(), 0);
    EXPECT_EQ(ret, HcclSim::HcclVmResult::HCCL_SIM_E_PARA);
}

TEST_F(AivResourceManagerTest, Init_EmptyMemInfo_CreatesBaseResources)
{
    auto ret = AivResourceManager::GetInstance().Init(0, EmptyOpMemInfo(), 2);
    EXPECT_EQ(ret, HcclSim::HcclVmResult::HCCL_SIM_SUCCESS);

    const auto& resources = AivResourceManager::GetInstance().GetAllRankResources();
    EXPECT_EQ(resources.size(), 2);
}

TEST_F(AivResourceManagerTest, GetAllRankResources_AfterInit_ReturnsCorrectSize)
{
    AivResourceManager::GetInstance().Init(0, EmptyOpMemInfo(), 4);

    const auto& resources = AivResourceManager::GetInstance().GetAllRankResources();
    EXPECT_EQ(resources.size(), 4);
}

TEST_F(AivResourceManagerTest, GetRankResource_ValidRankId_ReturnsNonNull)
{
    AivResourceManager::GetInstance().Init(0, EmptyOpMemInfo(), 4);

    const AivRankResource* resource = AivResourceManager::GetInstance().GetRankResource(2);
    EXPECT_NE(resource, nullptr);
}

TEST_F(AivResourceManagerTest, GetRankResource_OutOfRange_ReturnsNull)
{
    AivResourceManager::GetInstance().Init(0, EmptyOpMemInfo(), 4);

    const AivRankResource* resource = AivResourceManager::GetInstance().GetRankResource(10);
    EXPECT_EQ(resource, nullptr);
}

TEST_F(AivResourceManagerTest, GetRankResource_EqualToSize_ReturnsNull)
{
    AivResourceManager::GetInstance().Init(0, EmptyOpMemInfo(), 4);

    const AivRankResource* resource = AivResourceManager::GetInstance().GetRankResource(4);
    EXPECT_EQ(resource, nullptr);
}

TEST_F(AivResourceManagerTest, Reset_ClearsResources)
{
    AivResourceManager::GetInstance().Init(0, EmptyOpMemInfo(), 4);
    AivResourceManager::GetInstance().Reset();

    const auto& resources = AivResourceManager::GetInstance().GetAllRankResources();
    EXPECT_EQ(resources.size(), 0);
}

TEST_F(AivResourceManagerTest, GetRankResource_AfterReset_ReturnsNull)
{
    AivResourceManager::GetInstance().Init(0, EmptyOpMemInfo(), 4);
    AivResourceManager::GetInstance().Reset();

    const AivRankResource* resource = AivResourceManager::GetInstance().GetRankResource(0);
    EXPECT_EQ(resource, nullptr);
}

TEST_F(AivResourceManagerTest, Init_RankIdOutOfRange_ReturnsError)
{
    auto ret = AivResourceManager::GetInstance().Init(5, EmptyOpMemInfo(), 2);
    EXPECT_EQ(ret, HcclSim::HcclVmResult::HCCL_SIM_E_PARA);
}

TEST_F(AivResourceManagerTest, Init_MultipleTimesWithSameRankSize_Succeeds)
{
    auto ret = AivResourceManager::GetInstance().Init(0, EmptyOpMemInfo(), 2);
    EXPECT_EQ(ret, HcclSim::HcclVmResult::HCCL_SIM_SUCCESS);
    EXPECT_EQ(AivResourceManager::GetInstance().GetAllRankResources().size(), 2);

    ret = AivResourceManager::GetInstance().Init(1, EmptyOpMemInfo(), 2);
    EXPECT_EQ(ret, HcclSim::HcclVmResult::HCCL_SIM_SUCCESS);
    EXPECT_EQ(AivResourceManager::GetInstance().GetAllRankResources().size(), 2);
}

TEST_F(AivResourceManagerTest, Init_RankSizeMismatchAfterInit_ReturnsError)
{
    auto ret = AivResourceManager::GetInstance().Init(0, EmptyOpMemInfo(), 2);
    EXPECT_EQ(ret, HcclSim::HcclVmResult::HCCL_SIM_SUCCESS);

    ret = AivResourceManager::GetInstance().Init(0, EmptyOpMemInfo(), 5);
    EXPECT_EQ(ret, HcclSim::HcclVmResult::HCCL_SIM_E_PARA);
}

TEST_F(AivResourceManagerTest, RankResource_DefaultValues_AfterInit)
{
    AivResourceManager::GetInstance().Init(0, EmptyOpMemInfo(), 2);

    const AivRankResource* resource = AivResourceManager::GetInstance().GetRankResource(0);
    ASSERT_NE(resource, nullptr);
    EXPECT_EQ(resource->inputBuffer.realAddr, nullptr);
    EXPECT_EQ(resource->inputBuffer.size, 0);
    EXPECT_EQ(resource->outputBuffer.realAddr, nullptr);
    EXPECT_EQ(resource->outputBuffer.size, 0);
    EXPECT_EQ(resource->cclBuffer.realAddr, nullptr);
    EXPECT_EQ(resource->cclBuffer.size, 0);
    EXPECT_NE(resource->aivCommInfoBuffer.realAddr, nullptr);
    EXPECT_EQ(resource->aivCommInfoBuffer.size, EXPECTED_AIV_COMM_INFO_SIZE);
}

TEST_F(AivResourceManagerTest, Init_IncompleteBufferInfo_SkipsAndSucceeds)
{
    sim::OpMemInfoTab opMemInfo{};
    opMemInfo.inputAddr = 0x1000;
    opMemInfo.inputSize = 0;
    opMemInfo.outputAddr = 0;
    opMemInfo.outputSize = 4096;

    auto ret = AivResourceManager::GetInstance().Init(0, opMemInfo, 2);
    EXPECT_EQ(ret, HcclSim::HcclVmResult::HCCL_SIM_SUCCESS);

    const AivRankResource* resource = AivResourceManager::GetInstance().GetRankResource(0);
    ASSERT_NE(resource, nullptr);
    EXPECT_EQ(resource->inputBuffer.realAddr, nullptr);
    EXPECT_EQ(resource->outputBuffer.realAddr, nullptr);
}

TEST_F(AivResourceManagerTest, Init_InputBufferMappingAttempt_DoesNotReturnParameterError)
{
    sim::OpMemInfoTab opMemInfo{};
    opMemInfo.inputAddr = 0x1000;
    opMemInfo.inputSize = 4096;

    auto ret = AivResourceManager::GetInstance().Init(0, opMemInfo, 2);
    EXPECT_NE(ret, HcclSim::HcclVmResult::HCCL_SIM_E_PARA);
}

TEST_F(AivResourceManagerTest, Init_AllSupportedBuffersMappingAttempt_DoesNotReturnParameterError)
{
    sim::OpMemInfoTab opMemInfo{};
    opMemInfo.inputAddr = 0x1000;
    opMemInfo.inputSize = 4096;
    opMemInfo.outputAddr = 0x2000;
    opMemInfo.outputSize = 4096;
    opMemInfo.cclAddr = 0x3000;
    opMemInfo.cclSize = 4096;

    auto ret = AivResourceManager::GetInstance().Init(0, opMemInfo, 2);
    EXPECT_NE(ret, HcclSim::HcclVmResult::HCCL_SIM_E_PARA);
}

TEST_F(AivResourceManagerTest, Destructor_CallsReset)
{
    AivResourceManager& mgr = AivResourceManager::GetInstance();
    EXPECT_EQ(&mgr, &AivResourceManager::GetInstance());
}

TEST_F(AivResourceManagerTest, GetRankResource_BoundaryZero_ReturnsNonNull)
{
    AivResourceManager::GetInstance().Init(0, EmptyOpMemInfo(), 2);

    const AivRankResource* resource = AivResourceManager::GetInstance().GetRankResource(0);
    EXPECT_NE(resource, nullptr);
}

TEST_F(AivResourceManagerTest, GetRankResource_BoundaryMax_ReturnsNonNull)
{
    AivResourceManager::GetInstance().Init(0, EmptyOpMemInfo(), 2);

    const AivRankResource* resource = AivResourceManager::GetInstance().GetRankResource(1);
    EXPECT_NE(resource, nullptr);
}

TEST_F(AivResourceManagerTest, GetRankResource_ExactSize_ReturnsNull)
{
    AivResourceManager::GetInstance().Init(0, EmptyOpMemInfo(), 2);

    const AivRankResource* resource = AivResourceManager::GetInstance().GetRankResource(2);
    EXPECT_EQ(resource, nullptr);
}

TEST_F(AivResourceManagerTest, Reset_MultipleTimes_Safe)
{
    AivResourceManager::GetInstance().Init(0, EmptyOpMemInfo(), 2);
    AivResourceManager::GetInstance().Reset();
    AivResourceManager::GetInstance().Reset();
    AivResourceManager::GetInstance().Reset();

    const auto& resources = AivResourceManager::GetInstance().GetAllRankResources();
    EXPECT_EQ(resources.size(), 0);
}

TEST_F(AivResourceManagerTest, Init_RankSizeOne_Success)
{
    auto ret = AivResourceManager::GetInstance().Init(0, EmptyOpMemInfo(), 1);
    EXPECT_EQ(ret, HcclSim::HcclVmResult::HCCL_SIM_SUCCESS);

    const auto& resources = AivResourceManager::GetInstance().GetAllRankResources();
    EXPECT_EQ(resources.size(), 1);
}

TEST_F(AivResourceManagerTest, Init_LargeRankSize_Success)
{
    auto ret = AivResourceManager::GetInstance().Init(0, EmptyOpMemInfo(), 8);
    EXPECT_EQ(ret, HcclSim::HcclVmResult::HCCL_SIM_SUCCESS);

    const auto& resources = AivResourceManager::GetInstance().GetAllRankResources();
    EXPECT_EQ(resources.size(), 8);
}

TEST_F(AivResourceManagerTest, Init_AivCommInfoBufferAllocatedForEachRank)
{
    AivResourceManager::GetInstance().Init(0, EmptyOpMemInfo(), 3);

    for (uint32_t i = 0; i < 3; ++i) {
        const AivRankResource* resource = AivResourceManager::GetInstance().GetRankResource(i);
        ASSERT_NE(resource, nullptr);
        EXPECT_NE(resource->aivCommInfoBuffer.realAddr, nullptr);
        EXPECT_EQ(resource->aivCommInfoBuffer.size, EXPECTED_AIV_COMM_INFO_SIZE);
    }
}

TEST_F(AivResourceManagerTest, Init_RankResourcesIndependent)
{
    AivResourceManager::GetInstance().Init(0, EmptyOpMemInfo(), 2);

    const AivRankResource* resource0 = AivResourceManager::GetInstance().GetRankResource(0);
    const AivRankResource* resource1 = AivResourceManager::GetInstance().GetRankResource(1);

    ASSERT_NE(resource0, nullptr);
    ASSERT_NE(resource1, nullptr);
    EXPECT_NE(resource0->aivCommInfoBuffer.realAddr, resource1->aivCommInfoBuffer.realAddr);
}

TEST_F(AivResourceManagerTest, Init_EmptyMemInfo_BufferDefaults)
{
    AivResourceManager::GetInstance().Init(0, EmptyOpMemInfo(), 1);

    const AivRankResource* resource = AivResourceManager::GetInstance().GetRankResource(0);
    ASSERT_NE(resource, nullptr);
    EXPECT_EQ(resource->inputBuffer.realAddr, nullptr);
    EXPECT_EQ(resource->inputBuffer.size, 0);
    EXPECT_EQ(resource->outputBuffer.realAddr, nullptr);
    EXPECT_EQ(resource->outputBuffer.size, 0);
    EXPECT_EQ(resource->cclBuffer.realAddr, nullptr);
    EXPECT_EQ(resource->cclBuffer.size, 0);
    EXPECT_NE(resource->aivCommInfoBuffer.realAddr, nullptr);
    EXPECT_EQ(resource->aivCommInfoBuffer.size, EXPECTED_AIV_COMM_INFO_SIZE);
}
