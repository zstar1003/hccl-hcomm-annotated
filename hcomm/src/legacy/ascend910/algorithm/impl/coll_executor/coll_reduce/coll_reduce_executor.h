/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COLL_REDUCE_EXECUTOR_H
#define COLL_REDUCE_EXECUTOR_H
#include "coll_comm_executor.h"

namespace hccl {
class CollReduceExecutor : public CollCommExecutor {
public:
    CollReduceExecutor(const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher);
    ~CollReduceExecutor() override = default;

    HcclResult Orchestrate(OpParam& param, AlgResourceResponse& algRes) override;

protected:
    /* *************** 算法编排 *************** */
    // Reduce Loop Executor公共接口
    virtual u64 CalcLoopMaxCount(const u32 unitSize, const AlgResourceResponse& algRes);
    virtual bool IsHugeData(const u64 curSize);
    HcclResult RunLoop(OpParam& param, AlgResourceResponse& algRes);
    HcclResult RetryPostSync(OpParam& param, ExecMem& execMem);

private:
    HcclResult RunLoopInner(OpParam& param, const ReduceType& reduceType, ExecMem& execMem);
};

} // namespace hccl

#endif
