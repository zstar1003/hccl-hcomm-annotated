/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include <mockcpp/mokc.h>
#include <mockcpp/mockcpp.hpp>
#include "detour_service.h"
#include "detour_rules.h"
using namespace Hccl;

class DetourServiceTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "DetourServiceTest SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "DetourServiceTest TearDown" << std::endl; }

    virtual void SetUp() { std::cout << "A Test case in DetourServiceTest SetUP" << std::endl; }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in DetourServiceTest TearDown" << std::endl;
    }
};

TEST_F(DetourServiceTest, SetDetourTable4P)
{
    unordered_map<LocalId, unordered_map<LocalId, vector<LocalId>>> detourTable;
    std::set<u32> tableIdSet = {0, 1, 2, 3};
    SetDetourTable4P(tableIdSet, detourTable);
    EXPECT_EQ(detourTable, DETOUR_4P_TABLE_0123);
    std::set<u32> tableIdSet1 = {4, 5, 6, 7};
    SetDetourTable4P(tableIdSet1, detourTable);
    EXPECT_EQ(detourTable, DETOUR_4P_TABLE_4567);
    std::set<u32> tableIdSet2 = {0, 2, 4, 6};
    SetDetourTable4P(tableIdSet2, detourTable);
    EXPECT_EQ(detourTable, DETOUR_4P_TABLE_0246);
    std::set<u32> tableIdSet3 = {1, 3, 5, 7};
    SetDetourTable4P(tableIdSet3, detourTable);
    EXPECT_EQ(detourTable, DETOUR_4P_TABLE_1357);
}
