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

#include "aprof_pub.h"

extern "C" {
uint64_t MsprofStr2Id(const char* hashInfo, size_t length);
int32_t MsprofRegTypeInfo(uint16_t level, uint32_t typeId, const char* typeName);
uint64_t MsprofSysCycleTime();
int32_t MsprofRegisterCallback(uint32_t moduleId, ProfCommandHandle handle);
int32_t MsprofReportApi(uint32_t agingFlag, const MsprofApi* api);
int32_t MsprofReportAdditionalInfo(uint32_t agingFlag, const VOID_PTR data, uint32_t length);
int32_t MsprofReportCompactInfo(uint32_t agingFlag, const VOID_PTR data, uint32_t length);
}

class AprofilingStubTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(AprofilingStubTest, MsprofStr2Id_BasicCall)
{
    const char* hashInfo = "test_hash";
    uint64_t result = MsprofStr2Id(hashInfo, strlen(hashInfo));
    EXPECT_EQ(result, 1u);
}

TEST_F(AprofilingStubTest, MsprofStr2Id_EmptyString)
{
    uint64_t result = MsprofStr2Id("", 0);
    EXPECT_EQ(result, 1u);
}

TEST_F(AprofilingStubTest, MsprofStr2Id_NullPointer)
{
    uint64_t result = MsprofStr2Id(nullptr, 0);
    EXPECT_EQ(result, 1u);
}

TEST_F(AprofilingStubTest, MsprofStr2Id_LongString)
{
    std::string longStr(1000, 'a');
    uint64_t result = MsprofStr2Id(longStr.c_str(), longStr.size());
    EXPECT_EQ(result, 1u);
}

TEST_F(AprofilingStubTest, MsprofRegTypeInfo_BasicCall)
{
    int32_t result = MsprofRegTypeInfo(1, 100, "TestType");
    EXPECT_EQ(result, 0);
}

TEST_F(AprofilingStubTest, MsprofRegTypeInfo_DifferentLevels)
{
    EXPECT_EQ(MsprofRegTypeInfo(0, 100, "Type0"), 0);
    EXPECT_EQ(MsprofRegTypeInfo(1, 100, "Type1"), 0);
    EXPECT_EQ(MsprofRegTypeInfo(UINT16_MAX, 100, "TypeMax"), 0);
}

TEST_F(AprofilingStubTest, MsprofRegTypeInfo_DifferentTypeIds)
{
    EXPECT_EQ(MsprofRegTypeInfo(1, 0, "Type0"), 0);
    EXPECT_EQ(MsprofRegTypeInfo(1, 1, "Type1"), 0);
    EXPECT_EQ(MsprofRegTypeInfo(1, UINT32_MAX, "TypeMax"), 0);
}

TEST_F(AprofilingStubTest, MsprofRegTypeInfo_NullTypeName)
{
    int32_t result = MsprofRegTypeInfo(1, 100, nullptr);
    EXPECT_EQ(result, 0);
}

TEST_F(AprofilingStubTest, MsprofSysCycleTime_ReturnsOne)
{
    uint64_t result = MsprofSysCycleTime();
    EXPECT_EQ(result, 1u);
}

TEST_F(AprofilingStubTest, MsprofSysCycleTime_MultipleCalls)
{
    for (int i = 0; i < 10; i++) {
        EXPECT_EQ(MsprofSysCycleTime(), 1u);
    }
}

TEST_F(AprofilingStubTest, MsprofRegisterCallback_BasicCall)
{
    int32_t result = MsprofRegisterCallback(0, nullptr);
    EXPECT_EQ(result, 0);
}

TEST_F(AprofilingStubTest, MsprofRegisterCallback_DifferentModuleIds)
{
    EXPECT_EQ(MsprofRegisterCallback(0, nullptr), 0);
    EXPECT_EQ(MsprofRegisterCallback(1, nullptr), 0);
    EXPECT_EQ(MsprofRegisterCallback(UINT32_MAX, nullptr), 0);
}

TEST_F(AprofilingStubTest, MsprofReportApi_BasicCall)
{
    int32_t result = MsprofReportApi(0, nullptr);
    EXPECT_EQ(result, 0);
}

TEST_F(AprofilingStubTest, MsprofReportApi_DifferentAgingFlags)
{
    EXPECT_EQ(MsprofReportApi(0, nullptr), 0);
    EXPECT_EQ(MsprofReportApi(1, nullptr), 0);
    EXPECT_EQ(MsprofReportApi(UINT32_MAX, nullptr), 0);
}

TEST_F(AprofilingStubTest, MsprofReportAdditionalInfo_BasicCall)
{
    int32_t result = MsprofReportAdditionalInfo(0, nullptr, 0);
    EXPECT_EQ(result, 0);
}

TEST_F(AprofilingStubTest, MsprofReportAdditionalInfo_DifferentLengths)
{
    EXPECT_EQ(MsprofReportAdditionalInfo(0, nullptr, 0), 0);
    EXPECT_EQ(MsprofReportAdditionalInfo(0, nullptr, 100), 0);
    EXPECT_EQ(MsprofReportAdditionalInfo(0, nullptr, UINT32_MAX), 0);
}

TEST_F(AprofilingStubTest, MsprofReportCompactInfo_BasicCall)
{
    int32_t result = MsprofReportCompactInfo(0, nullptr, 0);
    EXPECT_EQ(result, 0);
}

TEST_F(AprofilingStubTest, MsprofReportCompactInfo_DifferentParams)
{
    EXPECT_EQ(MsprofReportCompactInfo(0, nullptr, 0), 0);
    EXPECT_EQ(MsprofReportCompactInfo(1, nullptr, 100), 0);
    EXPECT_EQ(MsprofReportCompactInfo(UINT32_MAX, nullptr, UINT32_MAX), 0);
}
