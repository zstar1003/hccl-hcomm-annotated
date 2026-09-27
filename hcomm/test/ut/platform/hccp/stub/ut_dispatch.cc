/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>

#include "gtest/gtest.h"

#undef MOCKCPP_USE_MOCKABLE
#include <mockcpp/mockcpp.hpp>

extern "C" GTEST_API_ int ut_main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

#define TC_FAIL 0
extern "C" void ut_testcase_expect_result(int result) { EXPECT_NE(TC_FAIL, result); }
extern "C" void ut_testcase_assert_result(int result) { ASSERT_NE(TC_FAIL, result); }

extern "C" int ut_testcase_log(const char* szFormat, ...)
{
    int ret;
    va_list ap;
    va_start(ap, szFormat);
    ret = vprintf(szFormat, ap);
    va_end(ap);
    return ret;
}

#ifdef NO_SUPPORT_HUTAF_LLT
/* this is a stub for hutaf llt hotpatch interface, and all interface return fail */
extern "C" int KKSetStubULData(void* pFuncAddr, int ulData) { return 0; }
extern "C" int KKInstallStub(void* pOldFuncAddr, void* pUserAPIAddr, int iStubType) { return 0; }
extern "C" int KKInstallULDataStub(void* pFuncAddr, int ulData) { return 0; }
extern "C" int KKUninstallStub(void* pOldFuncAddr) { return 0; }
#endif

extern "C" void mocker_clean() { GlobalMockObject::verify(); }

#define STUB_TYPE_DEF(n) typedef int (*stub_fn_p##n##_t)(STUB_PARAM_P##n)

#define STUB_PARAM_P0 void
#define STUB_PARAM_P1 void* p1
#define STUB_PARAM_P2 STUB_PARAM_P1, void* p2
#define STUB_PARAM_P3 STUB_PARAM_P2, void* p3
#define STUB_PARAM_P4 STUB_PARAM_P3, void* p4
#define STUB_PARAM_P5 STUB_PARAM_P4, void* p5
#define STUB_PARAM_P6 STUB_PARAM_P5, void* p6
#define STUB_PARAM_P7 STUB_PARAM_P6, void* p7
#define STUB_PARAM_P8 STUB_PARAM_P7, void* p8
#define STUB_PARAM_P9 STUB_PARAM_P8, void* p9
#define STUB_PARAM_P10 STUB_PARAM_P9, void* p10
#define STUB_PARAM_P11 STUB_PARAM_P10, void* p11
#define STUB_PARAM_P12 STUB_PARAM_P11, void* p12

STUB_TYPE_DEF(0);
STUB_TYPE_DEF(1);
STUB_TYPE_DEF(2);
STUB_TYPE_DEF(3);
STUB_TYPE_DEF(4);
STUB_TYPE_DEF(5);
STUB_TYPE_DEF(6);
STUB_TYPE_DEF(7);
STUB_TYPE_DEF(8);
STUB_TYPE_DEF(9);
STUB_TYPE_DEF(10);
STUB_TYPE_DEF(11);
STUB_TYPE_DEF(12);

#define MOCKER_APIs_DEFINE(n)                                                                                        \
    extern "C" void mocker_p##n(char* nameOfCaller, stub_fn_p##n##_t addrOfCaller, int most, int ret)                \
    {                                                                                                                \
        MOCKER(addrOfCaller).expects(atMost(most)).will(returnValue(ret));                                           \
    }                                                                                                                \
                                                                                                                     \
    extern "C" void mocker_ret_p##n(char* nameOfCaller, stub_fn_p##n##_t addrOfCaller, int ret0, int ret1, int ret2) \
    {                                                                                                                \
        MOCKER(addrOfCaller).stubs().will(returnValue(ret0)).then(returnValue(ret1)).then(returnValue(ret2));        \
    }                                                                                                                \
                                                                                                                     \
    extern "C" void mocker_invoke_p##n(                                                                              \
        char* nameOfCaller, char* nameOfstub, stub_fn_p##n##_t addrOfCaller, stub_fn_p##n##_t addrOfStub, int most)  \
    {                                                                                                                \
        MOCKER(addrOfCaller).expects(atMost(most)).will(invoke(addrOfStub, nameOfstub));                             \
    }

MOCKER_APIs_DEFINE(0) MOCKER_APIs_DEFINE(1) MOCKER_APIs_DEFINE(2) MOCKER_APIs_DEFINE(3) MOCKER_APIs_DEFINE(4)
    MOCKER_APIs_DEFINE(5) MOCKER_APIs_DEFINE(6) MOCKER_APIs_DEFINE(7) MOCKER_APIs_DEFINE(8) MOCKER_APIs_DEFINE(9)
        MOCKER_APIs_DEFINE(10) MOCKER_APIs_DEFINE(11) MOCKER_APIs_DEFINE(12)

            typedef int (*stub_fn_t)(
                long unsigned int data0, long unsigned int data1, long unsigned int data2, long unsigned int data3,
                long unsigned int data4, long unsigned int data5);

typedef long unsigned int (*stub_u64_fn_t)(
    long unsigned int data0, long unsigned int data1, long unsigned int data2, long unsigned int data3,
    long unsigned int data4, long unsigned int data5);
#define EXPECT_EQ_PRINT(orig, value, file, line)    \
    do {                                            \
        if (!(orig == value)) {                     \
            printf("%s:%d: Failure\n", file, line); \
        }                                           \
    } while (0);

#define EXPECT_NE_PRINT(orig, value, file, line)    \
    do {                                            \
        if (!(orig != value))                       \
            printf("%s:%d: Failure\n", file, line); \
    } while (0);

#define EXPECT_LT_PRINT(orig, value, file, line)    \
    do {                                            \
        if (!(orig < value))                        \
            printf("%s:%d: Failure\n", file, line); \
    } while (0);

#define EXPECT_LE_PRINT(orig, value, file, line)    \
    do {                                            \
        if (!(orig <= value))                       \
            printf("%s:%d: Failure\n", file, line); \
    } while (0);

#define EXPECT_GT_PRINT(orig, value, file, line)    \
    do {                                            \
        if (!(orig > value))                        \
            printf("%s:%d: Failure\n", file, line); \
    } while (0);

#define EXPECT_GE_PRINT(orig, value, file, line)    \
    do {                                            \
        if (!(orig >= value))                       \
            printf("%s:%d: Failure\n", file, line); \
    } while (0);

#define EXPECT_PRINT(file, line)                \
    do {                                        \
        printf("%s:%d: Failure\n", file, line); \
    } while (0);

extern "C" void ut_assert_int_true(int orig, const char* file, int line)
{
    EXPECT_NE_PRINT(orig, 0, file, line);
    ASSERT_TRUE(orig);
}

extern "C" void ut_assert_int_eq(int orig, int value, const char* file, int line)
{
    EXPECT_EQ_PRINT(orig, value, file, line);
    ASSERT_EQ(orig, value);
}

extern "C" void ut_assert_int_ne(int orig, int value, const char* file, int line)
{
    EXPECT_NE_PRINT(orig, value, file, line);
    ASSERT_NE(orig, value);
}

extern "C" void ut_assert_addr_eq(void* orig, void* value, const char* file, int line)
{
    EXPECT_EQ_PRINT(orig, value, file, line);
    ASSERT_EQ(orig, value);
}

extern "C" void ut_assert_addr_ne(void* orig, void* value, const char* file, int line)
{
    EXPECT_NE_PRINT(orig, value, file, line);
    ASSERT_NE(orig, value);
}

extern "C" void ut_expect_int_eq(int orig, int value, const char* file, int line)
{
    EXPECT_EQ_PRINT(orig, value, file, line);
    EXPECT_EQ(orig, value);
}

extern "C" void ut_expect_int_ne(int orig, int value, const char* file, int line)
{
    EXPECT_NE_PRINT(orig, value, file, line);
    EXPECT_NE(orig, value);
}

extern "C" void ut_expect_int_lt(int orig, int value, const char* file, int line)
{
    EXPECT_LT_PRINT(orig, value, file, line);
    EXPECT_LT(orig, value);
}

extern "C" void ut_expect_int_le(int orig, int value, const char* file, int line)
{
    EXPECT_LE_PRINT(orig, value, file, line);
    EXPECT_LE(orig, value);
}

extern "C" void ut_expect_int_gt(int orig, int value, const char* file, int line)
{
    EXPECT_GT_PRINT(orig, value, file, line);
    EXPECT_GT(orig, value);
}

extern "C" void ut_expect_int_ge(int orig, int value, const char* file, int line)
{
    EXPECT_GE_PRINT(orig, value, file, line);
    EXPECT_GE(orig, value);
}

extern "C" void ut_expect_long_eq(long orig, long value, const char* file, int line)
{
    EXPECT_EQ_PRINT(orig, value, file, line);
    EXPECT_EQ(orig, value);
}

extern "C" void ut_expect_long_ne(long orig, long value, const char* file, int line)
{
    EXPECT_NE_PRINT(orig, value, file, line);
    EXPECT_NE(orig, value);
}

extern "C" void ut_expect_mem_eq(void* a, void* b, int s, const char* file, int line)
{
    if (memcmp(a, b, s) != 0) {
        EXPECT_PRINT(file, line);
        EXPECT_EQ(0, 1);
    }
}

extern "C" void ut_expect_str_eq(char* a, char* b, const char* file, int line)
{
    if (strcmp(a, b) != 0) {
        EXPECT_PRINT(file, line);
        EXPECT_EQ(0, 1);
    }
}

extern "C" void ut_expect_str_ne(char* a, char* b, const char* file, int line)
{
    if (strcmp(a, b) == 0) {
        EXPECT_PRINT(file, line);
        EXPECT_EQ(0, 1);
    }
}

extern "C" void ut_expect_addr_eq(void* orig, void* value, const char* file, int line)
{
    EXPECT_EQ_PRINT(orig, value, file, line);
    EXPECT_EQ(orig, value);
}

extern "C" void ut_expect_addr_ne(void* orig, void* value, const char* file, int line)
{
    EXPECT_NE_PRINT(orig, value, file, line);
    EXPECT_NE(orig, value);
}

extern "C" void mocker(stub_fn_t h, int most, int ret) { MOCKER(h).expects(atMost(most)).will(returnValue(ret)); }

extern "C" void mocker_u64(stub_u64_fn_t h, int most, long unsigned int ret)
{
    MOCKER(h).expects(atMost(most)).will(returnValue(ret));
}

extern "C" void mocker_ret(stub_fn_t h, int ret0, int ret1, int ret2)
{
    MOCKER(h).stubs().will(returnValue(ret0)).then(returnValue(ret1)).then(returnValue(ret2));
}

extern "C" void mocker_ret_2(stub_fn_t h, int ret0, int ret1, int ret2, int ret3, int ret4)
{
    MOCKER(h)
        .stubs()
        .will(returnValue(ret0))
        .then(returnValue(ret1))
        .then(returnValue(ret2))
        .then(returnValue(ret3))
        .then(returnValue(ret4));
}

extern "C" void mocker_u64_ret(stub_u64_fn_t h, int most, long unsigned int ret0, long unsigned int ret1)
{
    MOCKER(h).expects(atMost(most)).will(returnValue(ret0)).then(returnValue(ret1));
}
extern "C" void mocker_invoke(stub_fn_t sh, stub_fn_t th, int most)
{
    MOCKER(sh).expects(atMost(most)).will(invoke(th));
}

extern "C" void mocker_u64_invoke(stub_u64_fn_t sh, stub_u64_fn_t th, int most)
{
    MOCKER(sh).expects(atMost(most)).will(invoke(th));
}
