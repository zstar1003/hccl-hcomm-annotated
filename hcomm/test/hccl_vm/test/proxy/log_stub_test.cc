/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <gtest/gtest.h>
#include <map>
#include <string>

extern "C" {
int32_t CheckLogLevel(int32_t moduleId, int32_t logLevel);
void DlogPrintStub(int level, int moduleId, const char* msgBuffer);
void DlogRecord(int moduleId, int level, const char* fmt, ...);
}

class LogStubLevel2Test : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(LogStubLevel2Test, CheckLogLevel_ReturnsOne)
{
    EXPECT_EQ(CheckLogLevel(0, 0), 1);
    EXPECT_EQ(CheckLogLevel(1, 1), 1);
    EXPECT_EQ(CheckLogLevel(100, 100), 1);
}

TEST_F(LogStubLevel2Test, CheckLogLevel_DifferentModules)
{
    for (int32_t moduleId = 0; moduleId < 10; moduleId++) {
        for (int32_t logLevel = 0; logLevel < 5; logLevel++) {
            EXPECT_EQ(CheckLogLevel(moduleId, logLevel), 1);
        }
    }
}

TEST_F(LogStubLevel2Test, DlogPrintStub_DebugLevel) { EXPECT_NO_THROW(DlogPrintStub(0, 3, "Debug message")); }

TEST_F(LogStubLevel2Test, DlogPrintStub_InfoLevel) { EXPECT_NO_THROW(DlogPrintStub(1, 3, "Info message")); }

TEST_F(LogStubLevel2Test, DlogPrintStub_WarningLevel) { EXPECT_NO_THROW(DlogPrintStub(2, 3, "Warning message")); }

TEST_F(LogStubLevel2Test, DlogPrintStub_ErrorLevel) { EXPECT_NO_THROW(DlogPrintStub(3, 3, "Error message")); }

TEST_F(LogStubLevel2Test, DlogPrintStub_TraceLevel) { EXPECT_NO_THROW(DlogPrintStub(4, 3, "Trace message")); }

TEST_F(LogStubLevel2Test, DlogPrintStub_EmptyMessage) { EXPECT_NO_THROW(DlogPrintStub(1, 3, "")); }

TEST_F(LogStubLevel2Test, DlogPrintStub_LongMessage)
{
    std::string longMsg(500, 'a');
    EXPECT_NO_THROW(DlogPrintStub(1, 3, longMsg.c_str()));
}

TEST_F(LogStubLevel2Test, DlogPrintStub_MultipleCalls)
{
    for (int i = 0; i < 10; i++) {
        EXPECT_NO_THROW(DlogPrintStub(i % 5, 3, "Test message"));
    }
}

TEST_F(LogStubLevel2Test, DlogRecord_BasicCall) { EXPECT_NO_THROW(DlogRecord(0, 1, "Test record message")); }

TEST_F(LogStubLevel2Test, DlogRecord_WithFormat) { EXPECT_NO_THROW(DlogRecord(0, 1, "Test record: %d", 123)); }

TEST_F(LogStubLevel2Test, DlogRecord_WithMultipleArgs)
{
    EXPECT_NO_THROW(DlogRecord(0, 1, "Test: %d, %s, %f", 123, "str", 3.14));
}

TEST_F(LogStubLevel2Test, DlogRecord_DifferentLevels)
{
    EXPECT_NO_THROW(DlogRecord(0, 0, "Debug"));
    EXPECT_NO_THROW(DlogRecord(0, 1, "Info"));
    EXPECT_NO_THROW(DlogRecord(0, 2, "Warning"));
    EXPECT_NO_THROW(DlogRecord(0, 3, "Error"));
}

TEST_F(LogStubLevel2Test, DlogRecord_MultipleCalls)
{
    for (int i = 0; i < 20; i++) {
        EXPECT_NO_THROW(DlogRecord(0, i % 4, "Message %d", i));
    }
}
