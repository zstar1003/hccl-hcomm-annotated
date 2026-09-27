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
#include <iostream>
#include "hccl_dl.h"
#include "dlrts_function_v2.h"

namespace {
constexpr char LIBC_SO[] = "libc.so.6";
constexpr char BAD_SO[] = "nonexistent_dummy.so";
constexpr char FUNC_PRINTF[] = "printf";
constexpr char FUNC_NONEXISTENT[] = "nonexistent_dummy_symbol";
} // namespace

using Hccl::DlRtsFunctionV2;

class HcclDlV2Test : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "HcclDlV2Test setup." << std::endl; }

    static void TearDownTestCase() { std::cout << "HcclDlV2Test teardown." << std::endl; }
};

// 用例 1: HcclDlopen/HcclDlsym/HcclDlclose 基本流程
TEST_F(HcclDlV2Test, OpenLibcSo_QueryPrintf_Close_Success)
{
    void* handle = HcclDlopen(LIBC_SO, RTLD_NOW);
    EXPECT_NE(handle, nullptr);
    void* func = HcclDlsym(handle, FUNC_PRINTF);
    EXPECT_NE(func, nullptr);
    EXPECT_EQ(HcclDlclose(handle), 0);
}

// 用例 2: 查不存在符号返回 nullptr
TEST_F(HcclDlV2Test, Dlsym_NonexistentSymbol_ReturnsNull)
{
    void* handle = HcclDlopen(LIBC_SO, RTLD_NOW);
    ASSERT_NE(handle, nullptr);
    EXPECT_EQ(HcclDlsym(handle, FUNC_NONEXISTENT), nullptr);
    HcclDlclose(handle);
}

// 用例 3: 打开不存在的 .so 返回 nullptr
TEST_F(HcclDlV2Test, Dlopen_NonexistentSo_ReturnsNull) { EXPECT_EQ(HcclDlopen(BAD_SO, RTLD_NOW), nullptr); }

// 用例 4: DlRtsFunctionV2 模板 Handle 已知符号返回非空
TEST_F(HcclDlV2Test, DlRtsFunctionV2_HandleKnownSymbol_ReturnsNonNull)
{
    DlRtsFunctionV2<LIBC_SO> dlRts;
    EXPECT_NE(dlRts.Handle<FUNC_PRINTF>(), nullptr);
}

// 用例 5: DlRtsFunctionV2 Handle 不存在符号返回 nullptr
TEST_F(HcclDlV2Test, DlRtsFunctionV2_HandleNonexistentSymbol_ReturnsNull)
{
    DlRtsFunctionV2<LIBC_SO> dlRts;
    EXPECT_EQ(dlRts.Handle<FUNC_NONEXISTENT>(), nullptr);
}

// 用例 6: DlRtsFunctionV2 打开不存在的 .so 后 Handle 返回 nullptr
TEST_F(HcclDlV2Test, DlRtsFunctionV2_InitFailed_HandleReturnsNull)
{
    DlRtsFunctionV2<BAD_SO> dlRts;
    EXPECT_EQ(dlRts.Handle<FUNC_PRINTF>(), nullptr);
}

// 用例 7: DlRtsFunctionV2 析构不崩溃
TEST_F(HcclDlV2Test, DlRtsFunctionV2_Destructor_NoCrash)
{
    {
        DlRtsFunctionV2<LIBC_SO> dlRts;
        dlRts.Handle<FUNC_PRINTF>();
    }
    SUCCEED();
}
