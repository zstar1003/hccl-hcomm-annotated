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

#include <set>
#include <unordered_map>
#include <vector>

#include "detour_rules.h"
#include "detour_service.h"

namespace Hccl {
void SetDetourTable2P(
    const std::set<u32>& tableIdSet,
    std::unordered_map<LocalId, std::unordered_map<LocalId, std::vector<LocalId>>>& detourTable);
}

using namespace Hccl;

class DetourServiceTest : public testing::Test {};

TEST_F(DetourServiceTest, Ut_SetDetourTable4P_When_MatchedTableIds_Expect_SelectExpectedTable)
{
    std::unordered_map<LocalId, std::unordered_map<LocalId, std::vector<LocalId>>> detourTable;

    SetDetourTable4P({0, 1, 2, 3}, detourTable);
    EXPECT_EQ(GetDetour4PTable0123(), detourTable);

    SetDetourTable4P({4, 5, 6, 7}, detourTable);
    EXPECT_EQ(GetDetour4PTable4567(), detourTable);

    SetDetourTable4P({0, 2, 4, 6}, detourTable);
    EXPECT_EQ(GetDetour4PTable0246(), detourTable);

    SetDetourTable4P({1, 3, 5, 7}, detourTable);
    EXPECT_EQ(GetDetour4PTable1357(), detourTable);
}

TEST_F(DetourServiceTest, Ut_SetDetourTable2P_When_MatchedTableIds_Expect_SelectExpectedTable)
{
    std::unordered_map<LocalId, std::unordered_map<LocalId, std::vector<LocalId>>> detourTable;

    SetDetourTable2P({0, 1}, detourTable);
    EXPECT_EQ(GetDetour2PTable01(), detourTable);

    SetDetourTable2P({0, 4}, detourTable);
    EXPECT_EQ(GetDetour2PTable04(), detourTable);
}
