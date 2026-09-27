/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ALIGNED_REDUCE_SCATTER_DOUBLE_RING_WITH_SERIAL_LOCAL_COPY_PUB_H
#define ALIGNED_REDUCE_SCATTER_DOUBLE_RING_WITH_SERIAL_LOCAL_COPY_PUB_H

#include "aligned_reduce_scatter_double_ring_pub.h"

namespace hccl {
class AlignedReduceScatterDoubleRingWithSerialLocalCopy : public AlignedReduceScatterDoubleRing {
public:
    explicit AlignedReduceScatterDoubleRingWithSerialLocalCopy(const HcclDispatcher dispatcher);
    ~AlignedReduceScatterDoubleRingWithSerialLocalCopy() override;

    HcclResult RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK>& links) override;

protected:
private:
    HcclResult MemcpyInitSlices(
        u64 ringIndex, DeviceMem& dstInit, DeviceMem& srcInit, DeviceMem& dstSubInit, DeviceMem& srcSubInit) override;
    HcclResult RunMainRingSubStream(const u32 rank, const u32 rankSize);
    HcclResult RunMainInitStep(const u32 rank, const u32 rankSize);
    HcclResult RunSubInitStep(const u32 rank, const u32 rankSize);
    HcclResult PreSync(const u32 ringIndex) override;
    HcclResult LocalMemcpy(
        const u32 step, const u32 rankSize, const u32 ringIndex, DeviceMem& localSrcMem,
        DeviceMem& localDstMem) override;
    HcclResult RunAllStreams(
        const u32 step, const u32 rankSize, std::vector<SenderMemoryInfo>& mainTxReduceMems,
        std::vector<ReducerMemoryInfo>& mainRxReduceMems, std::vector<SenderMemoryInfo>& subTxReduceMems,
        std::vector<ReducerMemoryInfo>& subRxReduceMems, std::vector<DeviceMem>& mainLocalSrcMems,
        std::vector<DeviceMem>& mainLocalDstMems, std::vector<DeviceMem>& subLocalSrcMems,
        std::vector<DeviceMem>& subLocalDstMems) override;
    HcclResult RunReduceScatter(const u32 rank, const u32 rankSize) override;
    HcclResult GetActiveSubstreamNumWithSerial(u32& activeSubstreamNum);
    HcclResult ExecEmptyTasks();
    HcclResult MainRecordSub();
    HcclResult SubWaitMain();
    HcclResult SubRecordMain();
    HcclResult MainWaitSub();
};
} // namespace hccl

#endif /* REDUCE_SCATTER_RING_CONCURRENT_DIRECT_PUB_H */
