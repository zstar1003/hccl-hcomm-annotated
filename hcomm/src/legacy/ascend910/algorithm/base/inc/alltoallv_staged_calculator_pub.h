/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ALLTOALL_V_STAGED_CALCULATOR_PUB_H
#define ALLTOALL_V_STAGED_CALCULATOR_PUB_H

#include "alg_template_base_pub.h"
#include "coll_alg_utils.h"
#include "common.h"

namespace hccl {
struct AlltoAllUserRankInfo {
    u32 userRankSize;
    u32 userRank;
};

class AlltoAllVStagedCalculator {
public:
    static void CalcWorkSpaceMemSize(
        const AlltoAllUserRankInfo& userRankInfo, const std::vector<SendRecvInfo>& allMeshAggregationSendRecvInfo,
        u64& workspaceMemSize, u32 meshAggregationRankSize);

protected:
private:
    explicit AlltoAllVStagedCalculator();
    virtual ~AlltoAllVStagedCalculator();
};
} // namespace hccl
#endif /* ALLTOALL_V_STAGED_CALCULATOR_PUB_H */
