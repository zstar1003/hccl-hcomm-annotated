/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_COLL_ALG_PARAMS
#define HCCLV2_COLL_ALG_PARAMS

#include <string>
#include <vector>
#include "op_mode.h"
#include "types.h"
#include "virtual_topo.h"

namespace Hccl {
using CollAlgParams = struct CollAlgParamsDef {
    OpMode opMode;
    u64 maxTmpMemSize;
    u32 maxQueue;
    u32 maxLink;
    u32 maxDepQueuePairs;
    u64 dataSize;
    bool isMc2{false};
    OpExecuteConfig opExecuteConfig;
    std::string algConfig;
};

using ResRequirement = struct ResRequirementDef {
    std::string algName;
    std::vector<LinkData> links;
    u32 primQueueNum;
    std::vector<std::tuple<QId, QId, u32>> queueNotifys;
};

// 算法拓扑
using AlgTopoInfo = struct AlgTopoInfoDef {
    std::vector<std::vector<RankId>> virtRanks;          // 各级通信内包含的Ranks
    std::vector<std::map<RankId, u32>> virtRankMap;      // 为不保序的ReduceScatter和AllGather预留
    std::vector<std::vector<std::vector<RankId>>> vTopo; // 各级通信域内并行的包含本rank的各个通信平面

    void UpdateSingleLevelTopo(
        std::vector<RankId> tmpVirtRank, std::map<RankId, u32> tmpVirtRankMap,
        std::vector<std::vector<RankId>> tmpVTopo)
    {
        virtRanks.push_back(tmpVirtRank);
        virtRankMap.push_back(tmpVirtRankMap);
        vTopo.push_back(tmpVTopo);
    }

    void UpdateMultiLevelTopo(
        std::vector<std::vector<RankId>> tmpVirtRank, std::vector<std::map<RankId, u32>> tmpVirtRankMap,
        std::vector<std::vector<std::vector<RankId>>> tmpVTopo)
    {
        virtRanks = tmpVirtRank;
        virtRankMap = tmpVirtRankMap;
        vTopo = tmpVTopo;
    }
};

using CollAlgResReq = struct CollAlgResReqDef {
    std::vector<LinkData> links;
    u32 primQueueNum;
    std::vector<std::tuple<QId, QId, u32>> queueNotifys;
    AlgTopoInfo topoInfo;
    std::vector<std::pair<QId, u32>> localWaitGroupCntNotify;
    std::vector<std::pair<QId, u32>> localBcastPostCntNotify;
    std::vector<std::pair<u32, RankId>> levelRankPairs;
};

using CollAlgOpReq = struct CollAlgOpReqDef {
    std::string algName;
    CollAlgResReq resReq;
};

} // namespace Hccl
#endif // !HCCLV2_COLL_ALG_PARAMS
