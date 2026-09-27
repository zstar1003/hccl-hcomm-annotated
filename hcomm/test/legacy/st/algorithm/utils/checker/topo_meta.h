/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RANKTABLE_FOR_LLT_H
#define RANKTABLE_FOR_LLT_H

#ifndef HCCL_ALG_ANALYZER_DAVID
#include "topoinfo_struct.h"
#endif

#include "hccl_types.h"
#include <vector>

namespace checker {

using u32 = unsigned int;
using PhyDeviceId = u32;
using ServerMeta = std::vector<PhyDeviceId>;
using SuperPodMeta = std::vector<ServerMeta>;
using TopoMeta = std::vector<SuperPodMeta>;

u32 GetRankNumFormTopoMeta(TopoMeta& topoMeta);
u32 GetServerNumFormTopoMeta(TopoMeta& topoMeta);

class RankTable_For_LLT {
public:
    explicit RankTable_For_LLT() = default;
    ~RankTable_For_LLT() = default;
    static HcclResult GenTopoMeta(TopoMeta& topoMate, int superPodNum, int serverNum, int rankNum);
};
} // namespace checker

#endif
