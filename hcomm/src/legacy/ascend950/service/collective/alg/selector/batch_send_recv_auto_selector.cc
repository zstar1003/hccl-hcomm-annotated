/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "batch_send_recv_auto_selector.h"
#include "selector_registry.h"
#include "coll_operator.h"

namespace Hccl {

SelectorStatus BatchSendRecvAutoSelector::SelectAicpuAlgo(
    const TopoInfo& topoInfo, const CollAlgOperator& op,
    const std::map<OpType, std::vector<HcclAlgoType>>& configAlgMap, std::string& primQueueGenName) const
{
    (void)topoInfo;
    std::vector<HcclAlgoType> algos
        = std::vector<HcclAlgoType>(HCCL_ALGO_LEVEL_NUM, HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT);
    auto it = configAlgMap.find(op.opType);
    if (it != configAlgMap.end()) {
        algos = it->second;
    }

    HCCL_INFO(
        "hccl algo op config: config opType:%s, level0:%u, level1:%u, level2:%u, level3:%u",
        op.opType.Describe().c_str(), algos[0], algos[1], algos[2], algos[3]);

    primQueueGenName = "InsBatchSendRecv";
    return SelectorStatus::MATCH;
}

SelectorStatus BatchSendRecvAutoSelector::SelectAivAlgo(
    const TopoInfo& topoInfo, const CollAlgOperator& op,
    const std::map<OpType, std::vector<HcclAlgoType>>& configAlgMap, std::string& primQueueGenName) const
{
    (void)topoInfo;
    std::vector<HcclAlgoType> algos
        = std::vector<HcclAlgoType>(HCCL_ALGO_LEVEL_NUM, HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT);
    auto it = configAlgMap.find(op.opType);
    if (it != configAlgMap.end()) {
        algos = it->second;
    }

    HCCL_INFO("[BatchSendRecvAutoSelector] select AIV algo for BatchSendRecv");

    primQueueGenName = "AivBatchSendRecv";
    return SelectorStatus::MATCH;
}

REGISTER_SELECTOR_BY_OPTYPE(OpType::BATCHSENDRECV, 18, BatchSendRecvAutoSelector);
} // namespace Hccl
