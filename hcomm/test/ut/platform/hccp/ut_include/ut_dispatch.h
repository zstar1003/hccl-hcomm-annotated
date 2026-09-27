/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __UT_DISPATCH_H
#define __UT_DISPATCH_H

#include <stdio.h>
#include <stdint.h>
typedef int (*stub_fn_t)(
    long unsigned int data0, long unsigned int data1, long unsigned int data2, long unsigned int data3,
    long unsigned int data4, long unsigned int data5);

typedef long unsigned int (*stub_u64_fn_t)(
    long unsigned int data0, long unsigned int data1, long unsigned int data2, long unsigned int data3,
    long unsigned int data4, long unsigned int data5);
#ifdef USE_DISPATCH_EXPECT
void ut_expect_int_eq(int orig, int value, const char* file, int line);
void ut_expect_int_ne(int orig, int value, const char* file, int line);
void ut_expect_int_lt(int orig, int value, const char* file, int line);
void ut_expect_int_le(int orig, int value, const char* file, int line);
void ut_expect_int_gt(int orig, int value, const char* file, int line);
void ut_expect_int_ge(int orig, int value, const char* file, int line);
void ut_expect_long_eq(long orig, long value, const char* file, int line);
void ut_expect_long_ne(long orig, long value, const char* file, int line);
void ut_expect_str_eq(char* a, char* b, const char* file, int line);
void ut_expect_str_ne(char* a, char* b, const char* file, int line);
void ut_expect_mem_eq(void* a, void* b, int s, const char* file, int line);
void ut_expect_addr_eq(void* orig, void* value, const char* file, int line);
void ut_expect_addr_ne(void* orig, void* value, const char* file, int line);
void ut_assert_int_eq(int orig, int value, const char* file, int line);
void ut_assert_int_ne(int orig, int value, const char* file, int line);
void ut_assert_addr_eq(void* orig, void* value, const char* file, int line);
void ut_assert_addr_ne(void* orig, void* value, const char* file, int line);
void ut_assert_int_true(int orig, const char* file, int line);

#define ASSERT_INT_EQ(orig, value) ut_assert_int_eq(orig, value, __FILE__, __LINE__)
#define ASSERT_INT_NE(orig, value) ut_assert_int_ne(orig, value, __FILE__, __LINE__)
#define ASSERT_ADDR_EQ(orig, value) ut_assert_addr_eq((void*)(orig), (void*)(value), __FILE__, __LINE__)
#define ASSERT_ADDR_NE(orig, value) ut_assert_addr_ne((void*)(orig), (void*)(value), __FILE__, __LINE__)
#define ASSERT_TRUE(orig) ut_assert_int_true(orig, __FILE__, __LINE__)

#define EXPECT_INT_EQ(orig, value) ut_expect_int_eq(orig, value, __FILE__, __LINE__)
#define EXPECT_INT_NE(orig, value) ut_expect_int_ne(orig, value, __FILE__, __LINE__)
#define EXPECT_INT_LT(orig, value) ut_expect_int_lt(orig, value, __FILE__, __LINE__)
#define EXPECT_INT_LE(orig, value) ut_expect_int_le(orig, value, __FILE__, __LINE__)
#define EXPECT_INT_GT(orig, value) ut_expect_int_gt(orig, value, __FILE__, __LINE__)
#define EXPECT_INT_GE(orig, value) ut_expect_int_ge(orig, value, __FILE__, __LINE__)
#define EXPECT_LONG_EQ(orig, value) ut_expect_long_eq(orig, value, __FILE__, __LINE__)
#define EXPECT_LONG_NE(orig, value) ut_expect_long_ne(orig, value, __FILE__, __LINE__)
#define EXPECT_STR_EQ(orig, value) ut_expect_str_eq(orig, value, __FILE__, __LINE__)
#define EXPECT_STR_NE(orig, value) ut_expect_str_ne(orig, value, __FILE__, __LINE__)
#define EXPECT_MEM_EQ(orig, value, size) ut_expect_mem_eq(orig, value, size, __FILE__, __LINE__)
#define EXPECT_ADDR_EQ(orig, value) ut_expect_addr_eq((void*)orig, (void*)value, __FILE__, __LINE__)
#define EXPECT_ADDR_NE(orig, value) ut_expect_addr_ne((void*)orig, (void*)value, __FILE__, __LINE__)
#endif

#define ASSERT_INT_EQ(orig, value) ut_assert_int_eq(orig, value, __FILE__, __LINE__)
#define ASSERT_INT_NE(orig, value) ut_assert_int_ne(orig, value, __FILE__, __LINE__)
#define ASSERT_ADDR_EQ(orig, value) ut_assert_addr_eq((void*)(orig), (void*)(value), __FILE__, __LINE__)
#define ASSERT_ADDR_NE(orig, value) ut_assert_addr_ne((void*)(orig), (void*)(value), __FILE__, __LINE__)
#define ASSERT_TRUE(orig) ut_assert_int_true(orig, __FILE__, __LINE__)

#define MOCKER_APIs_DECLARE(n)                                                                  \
    void mocker_p##n(char* nameOfCaller, void* addrOfCaller, int most, int ret);                \
    void mocker_ret_p##n(char* nameOfCaller, void* addrOfCaller, int ret0, int ret1, int ret2); \
    void mocker_invoke_p##n(char* nameOfCaller, char* nameOfstub, void* addrOfCaller, void* addrOfStub, int most)

MOCKER_APIs_DECLARE(0);
MOCKER_APIs_DECLARE(1);
MOCKER_APIs_DECLARE(2);
MOCKER_APIs_DECLARE(3);
MOCKER_APIs_DECLARE(4);
MOCKER_APIs_DECLARE(5);
MOCKER_APIs_DECLARE(6);
MOCKER_APIs_DECLARE(7);
MOCKER_APIs_DECLARE(8);
MOCKER_APIs_DECLARE(9);
MOCKER_APIs_DECLARE(10);
MOCKER_APIs_DECLARE(11);
MOCKER_APIs_DECLARE(12);

/* mocker default interface */
#define mocker(fn, most_cnt, ret_val) mocker_p4(#fn, (void*)(stub_fn_t)(fn), most_cnt, (int)(intptr_t)(ret_val))
#define mocker_ret(fn, ret0, ret1, ret2) mocker_ret_p4(#fn, (void*)(stub_fn_t)(fn), ret0, ret1, ret2)
#define mocker_invoke(fn, stub, most) \
    mocker_invoke_p4(#fn, #stub, (void*)(stub_fn_t)(fn), (void*)(stub_fn_t)(stub), most)

void mocker_clean();

#define TEST_M(T, f) \
    TEST_F(T, ut_##f) { f(); }

#endif
