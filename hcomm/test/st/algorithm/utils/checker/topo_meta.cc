/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "topo_meta.h"
#include "topoinfo_struct.h"
#include "hccl_types.h"
#include <string>
#include <arpa/inet.h>
#include <vector>
#include <algorithm>

using namespace hccl;

namespace checker {

HcclResult RankTable_For_LLT::GenTopoMeta(TopoMeta& topoMate, int superPodNum, int serverNum, int rankNum)
{
    for (u32 i = 0; i < superPodNum; i++) { // box
        SuperPodMeta superPodMeta;
        for (u32 j = 0; j < serverNum; j++) { // serverNumPerBox
            ServerMeta serverMate;
            for (u32 k = 0; k < rankNum; k++) {
                serverMate.push_back(k);
            }
            superPodMeta.push_back(serverMate);
        }
        topoMate.push_back(superPodMeta);
    }
    return HCCL_SUCCESS;
}

u32 GetRankNumFormTopoMeta(TopoMeta& topoMeta)
{
    u32 rankNum = 0;
    for (auto& podMeta : topoMeta) {
        for (auto& serverMeta : podMeta) {
            rankNum += serverMeta.size();
        }
    }
    return rankNum;
}

u32 GetServerNumFormTopoMeta(TopoMeta& topoMeta)
{
    u32 sererNum = 0;
    for (auto& podMeta : topoMeta) {
        for (auto& serverMeta : podMeta) {
            if (serverMeta.size()) {
                sererNum++;
            }
        }
    }
    return sererNum;
}

} // namespace checker
