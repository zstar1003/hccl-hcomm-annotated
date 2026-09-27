/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "detour_rules.h"

namespace Hccl {
namespace {

    /*
        2P节点间链路规划:

        0->2->1;   0->4->1;   0->6->1;
        1->3->0;   1->5->0;   1->7->0;
        2->0->3;   2->4->3;   2->6->3;
        3->1->2;   3->5->2;   3->7->2;
        4->0->5;   4->2->5;   4->6->5;
        5->1->4;   5->3->4;   5->7->4;
        6->0->7;   6->2->7;   6->4->7;
        7->1->6;   7->3->6;   7->5->6;
    */
    const std::unordered_map<LocalId, std::unordered_map<LocalId, std::vector<LocalId>>> DETOUR_2P_TABLE_01
        = {{0, {{1, {2, 4, 6}}}}, {1, {{0, {3, 5, 7}}}}, {2, {{3, {0, 4, 6}}}}, {3, {{2, {1, 5, 7}}}},
           {4, {{5, {0, 2, 6}}}}, {5, {{4, {1, 3, 7}}}}, {6, {{7, {0, 2, 4}}}}, {7, {{6, {1, 3, 5}}}}};

    /*
        0/1/2/3通信借道4/5/6/7节点间链路规划:

        0->4->1;   0->5->2;   0->6->3;
        1->4->0;   1->6->2;   1->7->3;
        2->5->0;   2->6->1;   2->4->3;
        3->6->0;   3->7->1;   3->4->2;
    */
    const std::unordered_map<LocalId, std::unordered_map<LocalId, std::vector<LocalId>>> DETOUR_2P_TABLE_04
        = {{0, {{4, {1, 2, 3}}}}, {4, {{0, {5, 6, 7}}}}, {1, {{5, {0, 2, 3}}}}, {5, {{1, {4, 6, 7}}}},
           {2, {{6, {0, 1, 3}}}}, {6, {{2, {4, 5, 7}}}}, {3, {{7, {0, 1, 2}}}}, {7, {{3, {4, 5, 6}}}}};

    const std::unordered_map<LocalId, std::unordered_map<LocalId, std::vector<LocalId>>> DETOUR_4P_TABLE_0123
        = {{0, {{1, {4}}}}, {0, {{2, {5}}}}, {0, {{3, {6}}}}, {1, {{0, {4}}}}, {1, {{2, {6}}}}, {1, {{3, {7}}}},
           {2, {{0, {5}}}}, {2, {{1, {6}}}}, {2, {{3, {4}}}}, {3, {{0, {6}}}}, {3, {{1, {7}}}}, {3, {{2, {4}}}}};
    /*
        4/5/6/7通信借道0/1/2/3节点间链路规划:

        4->0->5;   4->1->6;   4->2->7;
        5->0->4;   5->2->6;   5->3->7;
        6->1->4;   6->2->5;   6->0->7;
        7->2->4;   7->3->5;   7->0->6;
    */

    const std::unordered_map<LocalId, std::unordered_map<LocalId, std::vector<LocalId>>> DETOUR_4P_TABLE_4567
        = {{4, {{5, {0}}}}, {4, {{6, {1}}}}, {4, {{7, {2}}}}, {5, {{4, {0}}}}, {5, {{6, {2}}}}, {5, {{7, {3}}}},
           {6, {{4, {1}}}}, {6, {{5, {2}}}}, {6, {{7, {0}}}}, {7, {{4, {2}}}}, {7, {{5, {3}}}}, {7, {{6, {0}}}}};

    /*
        0/2/4/6通信借道1/3/5/7节点间链路规划:

        0->1->2;   0->3->4;   0->5->6;
        2->1->0;   2->5->4;   2->7->6;
        4->3->0;   4->5->2;   4->1->6;
        6->5->0;   6->7->2;   6->1->4;
    */

    const std::unordered_map<LocalId, std::unordered_map<LocalId, std::vector<LocalId>>> DETOUR_4P_TABLE_0246
        = {{0, {{2, {1}}}}, {0, {{4, {3}}}}, {0, {{6, {5}}}}, {2, {{1, {0}}}}, {2, {{4, {5}}}}, {2, {{6, {7}}}},
           {4, {{0, {3}}}}, {4, {{2, {5}}}}, {4, {{6, {1}}}}, {6, {{0, {5}}}}, {6, {{2, {7}}}}, {6, {{4, {1}}}}};

    /*
        1/3/5/7通信借道0/2/4/6节点间链路规划:

        1->0->3;   1->2->5;   1->4->7;
        3->0->1;   3->4->5;   3->6->7;
        5->2->1;   5->4->3;   5->0->7;
        7->4->1;   7->6->3;   7->0->5;
    */

    const std::unordered_map<LocalId, std::unordered_map<LocalId, std::vector<LocalId>>> DETOUR_4P_TABLE_1357
        = {{1, {{3, {0}}}}, {1, {{5, {2}}}}, {1, {{7, {4}}}}, {3, {{1, {0}}}}, {3, {{5, {4}}}}, {3, {{7, {6}}}},
           {5, {{1, {2}}}}, {5, {{3, {4}}}}, {5, {{7, {0}}}}, {7, {{1, {4}}}}, {7, {{3, {6}}}}, {7, {{5, {0}}}}};

} // namespace

const std::unordered_map<LocalId, std::unordered_map<LocalId, std::vector<LocalId>>>& GetDetour2PTable01()
{
    return DETOUR_2P_TABLE_01;
}

const std::unordered_map<LocalId, std::unordered_map<LocalId, std::vector<LocalId>>>& GetDetour2PTable04()
{
    return DETOUR_2P_TABLE_04;
}

const std::unordered_map<LocalId, std::unordered_map<LocalId, std::vector<LocalId>>>& GetDetour4PTable0123()
{
    return DETOUR_4P_TABLE_0123;
}

const std::unordered_map<LocalId, std::unordered_map<LocalId, std::vector<LocalId>>>& GetDetour4PTable4567()
{
    return DETOUR_4P_TABLE_4567;
}

const std::unordered_map<LocalId, std::unordered_map<LocalId, std::vector<LocalId>>>& GetDetour4PTable0246()
{
    return DETOUR_4P_TABLE_0246;
}

const std::unordered_map<LocalId, std::unordered_map<LocalId, std::vector<LocalId>>>& GetDetour4PTable1357()
{
    return DETOUR_4P_TABLE_1357;
}

} // namespace Hccl
