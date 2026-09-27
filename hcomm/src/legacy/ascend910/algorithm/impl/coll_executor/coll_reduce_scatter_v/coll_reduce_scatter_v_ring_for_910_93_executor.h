/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COLL_REDUCESCATTERV_RING_FOR_910_93_EXECUTOR_H
#define COLL_REDUCESCATTERV_RING_FOR_910_93_EXECUTOR_H

#include "coll_reduce_scatter_ring_for_910_93_executor.h"

namespace hccl {
class CollReduceScatterVRingFor91093Executor : public CollReduceScatterRingFor91093Executor {
public:
    CollReduceScatterVRingFor91093Executor(const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher);
    ~CollReduceScatterVRingFor91093Executor() override = default;

private:
    /* *************** 算法编排 *************** */
    u64 CalcLoopMaxCount(const u32 unitSize) override;
    bool IsHugeData(const u64 curSize, OpParam* param = nullptr) override;
};

} // namespace hccl

#endif
