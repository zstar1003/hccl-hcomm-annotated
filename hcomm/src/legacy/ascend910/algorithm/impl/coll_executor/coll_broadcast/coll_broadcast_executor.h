/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COLL_BROADCAST_COMM_EXECUTOR_H
#define COLL_BROADCAST_COMM_EXECUTOR_H
#include "coll_comm_executor.h"
#include "coll_alg_operator.h"

namespace hccl {
class CollBroadcastExecutor : public CollCommExecutor {
public:
    CollBroadcastExecutor(const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher);
    ~CollBroadcastExecutor() override = default;

    HcclResult Orchestrate(OpParam& param, AlgResourceResponse& algRes) override;

protected:
    /* *************** 算法编排 *************** */
    // Broadcast Loop Executor公共接口
    HcclResult RunLoop(OpParam& param, AlgResourceResponse& algRes);
    HcclResult GetSliceNum(const u64 size, const bool isSmallData, u64& sliceNum);
    bool IsBroadcastSmallData(u64 size, u64 totalSize);
    HcclResult CalcTransportMemType(TransportMemType& inputType, TransportMemType& outputType);
    virtual u64 CalcLoopMaxCount(const u64 cclBuffSize, const u32 unitSize);
    HcclResult
    GetRankSliceSize(HcclDataType dataType, const u64 count, const u32 rankSize, std::vector<Slice>& sliceList) const;
    bool DMAReduceFlag_{false};  // 是否DMA消减
    bool scratchMemFlag_{false}; // 是否需要申请scratch memory，不需要申请则传入outputmem为scratchmem
    std::vector<Slice> l0SliceList_; // 零拷贝时l0通信域各个rank切片，用于正确计算runloop时userIn到cclIn的偏移的大小

private:
    HcclResult RunLoopInner(OpParam& param, ExecMem& execMem);
};
} // namespace hccl

#endif
