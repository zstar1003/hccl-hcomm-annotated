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

#include "acl/acl_rt.h"
#include "runtime/base.h"

extern "C" {
aclError aclsysGetVersionStr(char* pkgName, char* versionStr);
rtError_t rtEnableP2P(uint32_t devIdDes, uint32_t phyIdSrc, uint32_t flag);
rtError_t rtDisableP2P(uint32_t devIdDes, uint32_t phyIdSrc);
}

class AclrtStubLevel2Test : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(AclrtStubLevel2Test, aclsysGetVersionStr_BasicCall)
{
    char pkgName[128] = {0};
    char versionStr[128] = {0};
    aclError result = aclsysGetVersionStr(pkgName, versionStr);
    EXPECT_EQ(result, ACL_SUCCESS);
    EXPECT_STREQ(versionStr, "9.0.0");
}

TEST_F(AclrtStubLevel2Test, aclsysGetVersionStr_VersionBufferValid)
{
    char pkgName[128] = {0};
    char versionStr[128] = {0};
    aclError result = aclsysGetVersionStr(pkgName, versionStr);
    EXPECT_EQ(result, ACL_SUCCESS);
    EXPECT_TRUE(strlen(versionStr) > 0);
}

TEST_F(AclrtStubLevel2Test, aclsysGetVersionStr_MultipleCalls)
{
    char pkgName[128] = {0};
    char versionStr[128] = {0};
    for (int i = 0; i < 10; i++) {
        memset(versionStr, 0, sizeof(versionStr));
        aclError result = aclsysGetVersionStr(pkgName, versionStr);
        EXPECT_EQ(result, ACL_SUCCESS);
        EXPECT_STREQ(versionStr, "9.0.0");
    }
}

TEST_F(AclrtStubLevel2Test, rtEnableP2P_BasicCall)
{
    rtError_t result = rtEnableP2P(0, 0, 0);
    EXPECT_EQ(result, ACL_SUCCESS);
}

TEST_F(AclrtStubLevel2Test, rtEnableP2P_DifferentDevices)
{
    EXPECT_EQ(rtEnableP2P(0, 0, 0), ACL_SUCCESS);
    EXPECT_EQ(rtEnableP2P(0, 1, 0), ACL_SUCCESS);
    EXPECT_EQ(rtEnableP2P(1, 0, 0), ACL_SUCCESS);
    EXPECT_EQ(rtEnableP2P(1, 1, 0), ACL_SUCCESS);
}

TEST_F(AclrtStubLevel2Test, rtEnableP2P_DifferentFlags)
{
    EXPECT_EQ(rtEnableP2P(0, 0, 0), ACL_SUCCESS);
    EXPECT_EQ(rtEnableP2P(0, 0, 1), ACL_SUCCESS);
    EXPECT_EQ(rtEnableP2P(0, 0, UINT32_MAX), ACL_SUCCESS);
}

TEST_F(AclrtStubLevel2Test, rtEnableP2P_BoundaryValues)
{
    EXPECT_EQ(rtEnableP2P(0, 0, 0), ACL_SUCCESS);
    EXPECT_EQ(rtEnableP2P(UINT32_MAX, UINT32_MAX, UINT32_MAX), ACL_SUCCESS);
}

TEST_F(AclrtStubLevel2Test, rtDisableP2P_BasicCall)
{
    rtError_t result = rtDisableP2P(0, 0);
    EXPECT_EQ(result, ACL_SUCCESS);
}

TEST_F(AclrtStubLevel2Test, rtDisableP2P_DifferentDevices)
{
    EXPECT_EQ(rtDisableP2P(0, 0), ACL_SUCCESS);
    EXPECT_EQ(rtDisableP2P(0, 1), ACL_SUCCESS);
    EXPECT_EQ(rtDisableP2P(1, 0), ACL_SUCCESS);
    EXPECT_EQ(rtDisableP2P(1, 1), ACL_SUCCESS);
}

TEST_F(AclrtStubLevel2Test, rtDisableP2P_BoundaryValues)
{
    EXPECT_EQ(rtDisableP2P(0, 0), ACL_SUCCESS);
    EXPECT_EQ(rtDisableP2P(UINT32_MAX, UINT32_MAX), ACL_SUCCESS);
}

TEST_F(AclrtStubLevel2Test, rtEnableDisableP2P_Consistency)
{
    // Enable and disable should both succeed
    EXPECT_EQ(rtEnableP2P(0, 1, 0), ACL_SUCCESS);
    EXPECT_EQ(rtDisableP2P(0, 1), ACL_SUCCESS);
}
