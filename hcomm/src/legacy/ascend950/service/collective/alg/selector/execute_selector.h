/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_COLL_ALG_SELECTOR_EXECUTION
#define HCCLV2_COLL_ALG_SELECTOR_EXECUTION

#include <string>

#include "types.h"
#include "dev_type.h"
#include "rank_gph.h"
#include "coll_alg_params.h"
#include "coll_operator.h"
#include "mc2_selector.h"

namespace Hccl {
class ExecuteSelector {
public:
    ExecuteSelector& SetVirtualTopo(RankGraph* rankGraph);
    ExecuteSelector& SetDevType(DevType devType);
    ExecuteSelector& SetMyRank(RankId myRank);
    ExecuteSelector& SetRankSize(u32 rankSize);
    ExecuteSelector& SetSeverId(std::string severId);
    ExecuteSelector& SetDeviceNumPerSever(u32 deviceNumPerSever);
    ExecuteSelector& SetServerNum(u32 serverNum);
    ExecuteSelector& SetOpConfig(OpExecuteConfig opConfig);
    HcclResult Run(const CollAlgOperator& op, CollAlgParams& params, std::string& primQueueGenName);
    AlgorithmType GetAlgorithmTypeForMC2CCU(const std::string& name) const;

protected:
    RankGraph* rankGraph_ = nullptr;
    OpExecuteConfig opConfig_;
    DevType devType_;
    u32 myRank_;
    u32 rankSize_;
    std::string severId_;
    u32 deviceNumPerSever_;
    u32 serverNum_;
};
} // namespace Hccl
#endif
