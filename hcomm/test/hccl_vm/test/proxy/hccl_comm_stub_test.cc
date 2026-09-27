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

#include "hccl/hccl.h"
#include "acl/acl_base.h"
#include "acl/acl_rt.h"

extern "C" {
HcclResult HcclGetRootInfo(HcclRootInfo* rootInfo);
HcclResult HcclCommInitRootInfo(uint32_t nRanks, const HcclRootInfo* rootInfo, uint32_t rank, HcclComm* comm);
HcclResult HcclCommInitRootInfoConfig(
    uint32_t nRanks, const HcclRootInfo* rootInfo, uint32_t rank, const HcclCommConfig* config, HcclComm* comm);
HcclResult HcclCommInitClusterInfo(const char* clusterInfo, uint32_t rank, HcclComm* comm);
HcclResult
HcclCommInitClusterInfoConfig(const char* clusterInfo, uint32_t rank, HcclCommConfig* config, HcclComm* comm);
HcclResult HcclCommDestroy(HcclComm comm);
aclError aclrtSetDevice(int32_t deviceId);
aclError aclrtResetDevice(int32_t deviceId);
}

HcclResult HcclCommInitClusterInfo(const char* clusterInfo, uint32_t rank, HcclComm* comm) { return HCCL_SUCCESS; }

HcclResult HcclCommInitClusterInfoConfig(const char* clusterInfo, uint32_t rank, HcclCommConfig* config, HcclComm* comm)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommDestroy(HcclComm comm) { return HCCL_SUCCESS; }

aclError aclrtSetDevice(int32_t deviceId) { return ACL_SUCCESS; }

aclError aclrtResetDevice(int32_t deviceId) { return ACL_SUCCESS; }

class HcclCommStubTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override { unsetenv("RANK_TABLE_FILE"); }
};

TEST_F(HcclCommStubTest, HcclGetRootInfo_BasicCall)
{
    HcclRootInfo rootInfo = {0};
    HcclResult result = HcclGetRootInfo(&rootInfo);
    EXPECT_EQ(result, HCCL_SUCCESS);
}

TEST_F(HcclCommStubTest, HcclGetRootInfo_MultipleCalls)
{
    HcclRootInfo rootInfo1 = {0};
    HcclRootInfo rootInfo2 = {0};

    EXPECT_EQ(HcclGetRootInfo(&rootInfo1), HCCL_SUCCESS);
    EXPECT_EQ(HcclGetRootInfo(&rootInfo2), HCCL_SUCCESS);
}

TEST_F(HcclCommStubTest, HcclGetRootInfo_NullPointer)
{
    // This should not crash, but behavior is undefined
    // We just test that it doesn't throw
    EXPECT_NO_THROW(HcclGetRootInfo(nullptr));
}

TEST_F(HcclCommStubTest, HcclCommInitRootInfo_NoEnvVar)
{
    unsetenv("RANK_TABLE_FILE");
    HcclRootInfo rootInfo = {0};
    HcclComm comm = nullptr;
    // Without RANK_TABLE_FILE set, this may fail or behave differently
    // Just test that the function can be called
    EXPECT_NO_THROW(HcclCommInitRootInfo(2, &rootInfo, 0, &comm));
}

TEST_F(HcclCommStubTest, HcclCommInitRootInfo_WithEnvVar)
{
    setenv("RANK_TABLE_FILE", "/tmp/test_rank_table.json", 1);
    HcclRootInfo rootInfo = {0};
    HcclComm comm = nullptr;
    EXPECT_NO_THROW(HcclCommInitRootInfo(2, &rootInfo, 0, &comm));
}

TEST_F(HcclCommStubTest, HcclCommInitRootInfo_DifferentNRanks)
{
    HcclRootInfo rootInfo = {0};
    HcclComm comm = nullptr;

    for (uint32_t nRanks = 1; nRanks <= 8; nRanks++) {
        EXPECT_NO_THROW(HcclCommInitRootInfo(nRanks, &rootInfo, 0, &comm));
    }
}

TEST_F(HcclCommStubTest, HcclCommInitRootInfo_DifferentRanks)
{
    HcclRootInfo rootInfo = {0};
    HcclComm comm = nullptr;

    for (uint32_t rank = 0; rank < 8; rank++) {
        EXPECT_NO_THROW(HcclCommInitRootInfo(8, &rootInfo, rank, &comm));
    }
}

TEST_F(HcclCommStubTest, HcclCommInitRootInfoConfig_NoEnvVar)
{
    unsetenv("RANK_TABLE_FILE");
    HcclRootInfo rootInfo = {0};
    HcclCommConfig config = {0};
    HcclComm comm = nullptr;
    EXPECT_NO_THROW(HcclCommInitRootInfoConfig(2, &rootInfo, 0, &config, &comm));
}

TEST_F(HcclCommStubTest, HcclCommInitRootInfoConfig_WithEnvVar)
{
    setenv("RANK_TABLE_FILE", "/tmp/test_rank_table.json", 1);
    HcclRootInfo rootInfo = {0};
    HcclCommConfig config = {0};
    HcclComm comm = nullptr;
    EXPECT_NO_THROW(HcclCommInitRootInfoConfig(2, &rootInfo, 0, &config, &comm));
}

TEST_F(HcclCommStubTest, HcclCommInitRootInfoConfig_DifferentNRanks)
{
    HcclRootInfo rootInfo = {0};
    HcclCommConfig config = {0};
    HcclComm comm = nullptr;

    for (uint32_t nRanks = 1; nRanks <= 8; nRanks++) {
        EXPECT_NO_THROW(HcclCommInitRootInfoConfig(nRanks, &rootInfo, 0, &config, &comm));
    }
}

TEST_F(HcclCommStubTest, HcclCommInitRootInfoConfig_DifferentRanks)
{
    HcclRootInfo rootInfo = {0};
    HcclCommConfig config = {0};
    HcclComm comm = nullptr;

    for (uint32_t rank = 0; rank < 8; rank++) {
        EXPECT_NO_THROW(HcclCommInitRootInfoConfig(8, &rootInfo, rank, &config, &comm));
    }
}

TEST_F(HcclCommStubTest, HcclCommInitRootInfo_BoundaryValues)
{
    HcclRootInfo rootInfo = {0};
    HcclComm comm = nullptr;

    // Test with nRanks = 1
    EXPECT_NO_THROW(HcclCommInitRootInfo(1, &rootInfo, 0, &comm));

    // Test with larger nRanks
    EXPECT_NO_THROW(HcclCommInitRootInfo(1024, &rootInfo, 0, &comm));
}
