/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DETOUR_SERVICE_H
#define DETOUR_SERVICE_H

#include <unordered_map>
#include "phy_topo.h"
#include "rank_gph.h"

namespace Hccl {

class DetourService {
public:
    static DetourService& GetInstance();

    // 插入detourLinks
    void InsertDetourLinks(RankGraph* rankGraph, const RankTableInfo* rankTable);

private:
    explicit DetourService(const PhyTopo* phyTopo);
    ~DetourService() = default;
    DetourService(const DetourService& detourService) = delete;
    DetourService& operator=(const DetourService& detourService) = delete;

    const PhyTopo* phyTopo{nullptr};
};

void SetDetourTable4P(
    const std::set<u32>& tableIdSet, unordered_map<LocalId, unordered_map<LocalId, vector<LocalId>>>& detourTable);
} // namespace Hccl

#endif // DETOUR_SERVICE_H
