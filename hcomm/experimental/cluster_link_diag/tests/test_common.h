/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <cmath>
#include <exception>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace test {
inline int failures = 0;

inline void fail(const char* file, int line, const std::string& message)
{
    ++failures;
    std::cerr << file << ':' << line << ": " << message << std::endl;
}

template <typename Left, typename Right>
void expect_equal(
    const Left& left, const Right& right, const char* left_expr, const char* right_expr, const char* file, int line)
{
    if (!(left == right)) {
        std::ostringstream oss;
        oss << "expected " << left_expr << " == " << right_expr << ", actual " << left << " vs " << right;
        fail(file, line, oss.str());
    }
}

inline void expect_near(
    double left, double right, double tolerance, const char* left_expr, const char* right_expr, const char* file,
    int line)
{
    if (std::isnan(left) || std::isnan(right) || std::fabs(left - right) > tolerance) {
        std::ostringstream oss;
        oss << "expected " << left_expr << " ~= " << right_expr << " within " << tolerance << ", actual " << left
            << " vs " << right;
        fail(file, line, oss.str());
    }
}

inline int finish(const std::string& name)
{
    if (failures == 0) {
        std::cout << "[PASS] " << name << std::endl;
        return 0;
    }
    std::cerr << "[FAIL] " << name << ": " << failures << " assertion(s)" << std::endl;
    return 1;
}
} // namespace test

#define EXPECT_TRUE(expr)                                            \
    do {                                                             \
        if (!(expr))                                                 \
            test::fail(__FILE__, __LINE__, "expected true: " #expr); \
    } while (false)

#define EXPECT_FALSE(expr) EXPECT_TRUE(!(expr))
#define EXPECT_EQ(left, right) test::expect_equal((left), (right), #left, #right, __FILE__, __LINE__)
#define EXPECT_NEAR(left, right, tolerance) \
    test::expect_near((left), (right), (tolerance), #left, #right, __FILE__, __LINE__)
#define EXPECT_THROW(statement)                                                \
    do {                                                                       \
        bool threw = false;                                                    \
        try {                                                                  \
            statement;                                                         \
        } catch (const std::exception&) {                                      \
            threw = true;                                                      \
        }                                                                      \
        if (!threw)                                                            \
            test::fail(__FILE__, __LINE__, "expected exception: " #statement); \
    } while (false)
