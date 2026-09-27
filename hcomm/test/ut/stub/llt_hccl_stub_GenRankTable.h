/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RankTable_For_LLT_H
#define RankTable_For_LLT_H

#include "topoinfo_struct.h"
#include "hccl_types.h"
#include <vector>

namespace hccl {

using u32 = unsigned int;
using PhyDeviceId = u32;
using ServerMate = std::vector<PhyDeviceId>;
using SuperPodMeta = std::vector<ServerMate>;
using TopoMeta = std::vector<SuperPodMeta>;

class RankTable_For_LLT {
public:
    explicit RankTable_For_LLT() = default;
    ~RankTable_For_LLT() = default;
    static HcclResult GenRankTable(TopoMeta topoMate, RankTable_t& rankTable);
    static HcclResult GenTopoMeta(TopoMeta& topoMate, int arg1 = 2, int arg2 = 2, int arg3 = 8);
};

} // namespace hccl

#endif
