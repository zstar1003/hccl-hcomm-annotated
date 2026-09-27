/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>
#include <string>
#include <sstream>
#include <ios>
#include <iostream>
#include "ccu_ins.h"
#include "ccu_ctx_signature.h"
#include "ccu_context_utils.h"

namespace Hccl {

constexpr int COMM_LEVEL_SIZE_1 = 1;
constexpr int COMM_LEVEL_SIZE_2 = 2;

uint64_t CalcLGMaxTransSize()
{
    return LOC_CPY_LOOP_NUM * 4096 * 8192; // 单片MS搬4096B，每个loop循环最多8192次
}

HcclResult GenerateCcuCtxSignature(
    CcuCtxSignature& sig, CcuInstType instType, const CollAlgOperator& op,
    const std::vector<std::vector<RankId>>& tempVTopo)
{
    sig.Append<int>(int(instType));
    if (op.opType == OpType::REDUCESCATTER || op.opType == OpType::ALLREDUCE || op.opType == OpType::REDUCE
        || op.opType == OpType::REDUCESCATTERV) {
        sig.Append<int>(int(op.reduceOp));
        sig.Append<int>(int(op.dataType));
        sig.Append<int>(int(op.outputDataType));
    }
    if (op.opType == OpType::REDUCE || op.opType == OpType::BROADCAST || op.opType == OpType::GATHER
        || op.opType == OpType::SCATTER) {
        // 带有root属性的算子需要考虑自己与root的关系，暂定直接用root号做区分
        sig.Append<char>('R');
        sig.Append<int>(int(op.root));
        // sig.Append<std::string>("_");
    }
    if (tempVTopo.size() == COMM_LEVEL_SIZE_1) {
        sig.Append<uint32_t>(tempVTopo[0].size());
        sig.Append<char>('P');
    } else if (tempVTopo.size() == COMM_LEVEL_SIZE_2) {
        sig.Append<uint32_t>(tempVTopo[0].size());
        sig.Append<char>('X');
        sig.Append<uint32_t>(tempVTopo[1].size());
        sig.Append<char>('P');
    } else {
        THROW<InvalidParamsException>(
            StringFormat("GenerateCcuCtxSignature failed: unexpected tempVTopoSize[%zu]", tempVTopo.size()));
    }
    return HcclResult::HCCL_SUCCESS;
}
} // namespace Hccl
