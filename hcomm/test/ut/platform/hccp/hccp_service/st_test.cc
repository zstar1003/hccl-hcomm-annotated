/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

extern "C" {
#include "ut_dispatch.h"
#include "tc_hccp.h"
}

#include "gtest/gtest.h"
#include <stdio.h>

using namespace std;

class Hccp : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "\033[36m--RoCE Hccp SetUP--\033[0m" << std::endl; }
    static void TearDownTestCase() { std::cout << "\033[36m--RoCE Hccp TearDown--\033[0m" << std::endl; }
    virtual void SetUp() {}
    virtual void TearDown() {}
};

TEST_M(Hccp, TcNormal);
TEST_M(Hccp, TcAbnormal);
