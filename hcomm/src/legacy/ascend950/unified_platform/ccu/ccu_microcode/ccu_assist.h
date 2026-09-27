/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_CONTEXT_ASSIST_H
#define HCCL_CCU_CONTEXT_ASSIST_H

#include <string>

#include "data_type.h"
#include "reduce_op.h"

namespace Hccl {
namespace CcuRep {

    // 辅助函数
    uint64_t GetMaxLoopIterNum();
    uint64_t GetLoopParam(uint64_t loopCtxId, uint64_t gsaOffset, uint64_t loopIterNum);
    uint64_t GetParallelParam(uint64_t repeatNum, uint64_t repeatLoopIndex, uint64_t totalLoopNum);
    uint16_t ParseRepeatNumFromParallelParam(uint64_t parallelParam);
    uint64_t GetOffsetParam(uint64_t gsaOffset, uint64_t msOffset, uint64_t ckeOffset);
    uint64_t GetToken(uint64_t tokenId, uint64_t tokenValue, uint64_t tokenValid);
    uint64_t GetExpansionParam(uint64_t expansionNum);

    uint16_t GetCcuReduceType(ReduceOp reduceOp);
    uint16_t GetCcuDataType(DataType dataType, ReduceOp reduceOp);
    uint16_t GetUBReduceType(ReduceOp reduceOp);
    uint16_t GetUBDataType(DataType dataType);
    uint32_t GetReduceExpansionNum(ReduceOp reduceOp, DataType dataType, DataType outputDataType);
    std::string GetReduceTypeStr(DataType dataType, ReduceOp opType);

    uint64_t GetTokenInfo(uint64_t va, uint64_t size);

}; // namespace CcuRep
}; // namespace Hccl

#endif // HCCL_CCU_CONTEXT_ASSIST_H
