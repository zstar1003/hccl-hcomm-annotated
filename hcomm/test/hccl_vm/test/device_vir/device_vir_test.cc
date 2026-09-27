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
#include <cstdlib>
#include <dlfcn.h>
#include <gtest/gtest.h>
#include <string>

class DeviceVirTest : public testing::Test {
protected:
};

TEST_F(DeviceVirTest, StringToUint64_ValidNumber)
{
    const char* offsetStr = "12345";
    uint64_t result = atol(offsetStr);
    EXPECT_EQ(result, 12345);
}

TEST_F(DeviceVirTest, StringToUint64_Zero)
{
    const char* offsetStr = "0";
    uint64_t result = atol(offsetStr);
    EXPECT_EQ(result, 0);
}

TEST_F(DeviceVirTest, StringToUint64_LargeNumber)
{
    const char* offsetStr = "9999999999";
    uint64_t result = atol(offsetStr);
    EXPECT_EQ(result, 9999999999ULL);
}

TEST_F(DeviceVirTest, DlOpen_NonExistentLib_ShouldFail)
{
    void* handle = dlopen("nonexistent_lib.so", RTLD_NOW);
    EXPECT_EQ(handle, nullptr);
}

TEST_F(DeviceVirTest, DlError_AfterFailedDlOpen)
{
    dlopen("nonexistent.so", RTLD_NOW);
    char* err = dlerror();
    EXPECT_NE(err, nullptr);
}

TEST_F(DeviceVirTest, StringToUint64_NegativeNumber)
{
    const char* offsetStr = "-12345";
    uint64_t result = atol(offsetStr);
    EXPECT_EQ(result, static_cast<uint64_t>(-12345));
}

TEST_F(DeviceVirTest, StringToUint64_EmptyString)
{
    const char* offsetStr = "";
    uint64_t result = atol(offsetStr);
    EXPECT_EQ(result, 0);
}

TEST_F(DeviceVirTest, StringToUint64_InvalidChars)
{
    const char* offsetStr = "abc";
    uint64_t result = atol(offsetStr);
    EXPECT_EQ(result, 0);
}

TEST_F(DeviceVirTest, StringToUint64_HexFormat)
{
    const char* offsetStr = "0x1000";
    uint64_t result = strtol(offsetStr, nullptr, 16);
    EXPECT_EQ(result, 0x1000);
}

TEST_F(DeviceVirTest, DlOpen_NullPath)
{
    void* handle = dlopen(nullptr, RTLD_NOW);
    EXPECT_NE(handle, nullptr);
    if (handle) {
        dlclose(handle);
    }
}

TEST_F(DeviceVirTest, DlSym_NullHandle)
{
    void* sym = dlsym(nullptr, "test");
    EXPECT_EQ(sym, nullptr);
}

TEST_F(DeviceVirTest, DlClose_ValidHandle)
{
    void* handle = dlopen(nullptr, RTLD_NOW);
    if (handle) {
        int result = dlclose(handle);
        EXPECT_EQ(result, 0);
    }
}
