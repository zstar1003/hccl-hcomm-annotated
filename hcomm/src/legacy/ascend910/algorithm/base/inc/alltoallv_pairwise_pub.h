/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ALLTOALL_V_PAIRWISE_PUB_H
#define ALLTOALL_V_PAIRWISE_PUB_H

#include "alg_template_base_pub.h"

namespace hccl {

class AlltoAllVPairWise : public AlgTemplateBase {
public:
    explicit AlltoAllVPairWise(const HcclDispatcher dispatcher);

    ~AlltoAllVPairWise() override;

    /* 图模式使用该prepare */
    HcclResult Prepare(
        AlltoAllVBufferInfo& sendBuffer, AlltoAllVBufferInfo& recvBuffer, bool isAlltoAllZCopyMode,
        const Stream& stream, HcclWorkflowMode workMode, std::map<u32, std::vector<u64>>& rankSendDisplsMap,
        std::map<u32, std::vector<u64>>& rankRecvDisplsMap) override;

    /* 单算子使用该prepare */
    HcclResult Prepare(
        AlltoAllVBufferInfo& sendBuffer, AlltoAllVBufferInfo& recvBuffer, DeviceMem& scratchInputMem,
        DeviceMem& scratchOutputMem, bool isAlltoAllZCopyMode, const Stream& stream, HcclWorkflowMode workMode,
        std::map<u32, std::vector<u64>>& rankSendDisplsMap,
        std::map<u32, std::vector<u64>>& rankRecvDisplsMap) override;
    HcclResult RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK>& links) override;
    HcclResult
    GetNslbAdjInfo(const u32 rank, const u32 rankSize, const std::vector<LINK>& links, AdjInfo& nslbAdjInfo) override;

protected:
private:
    HcclResult LocalCopy(const u32 rank);
    // 单算子模式使用该函数
    HcclResult RunBCopyAlltoAll(const u32 rank, const u32 rankSize, const std::vector<LINK>& links);
    // 图模式使用该函数
    HcclResult RunZCopyAlltoAll(const u32 rank, const u32 rankSize, const std::vector<LINK>& links);
    HcclResult CalcSendRecvCounts(u32 times, u32 curTime, u64 totalBytes, u64& curBytes) const;

    // 单算子模式使用该SendRecv
    HcclResult SendRecv(
        u64 curSendBytes, u64 curRecvBytes, u8* sendAddr, u8* recvAddr, std::shared_ptr<Transport> prevTransport,
        std::shared_ptr<Transport> nextTransport);
    // 图模式使用该SendRecv
    HcclResult SendRecv(
        TxMemoryInfo txMemoryInfo, RxMemoryInfo rxMemoryInfo, std::shared_ptr<Transport> prevTransport,
        std::shared_ptr<Transport> nextTransport);

    AlltoAllVBufferInfo sendBuffer_;
    AlltoAllVBufferInfo recvBuffer_;
    // 约束： scratchInputMem scratchOutputMem 用于transport中转，
    // 这两块内存需要在transport注册，两块内存的大小需一致
    DeviceMem scratchInputMem_;
    DeviceMem scratchOutputMem_;
    Stream stream_;
    u64 scratchMemSize_;
    u32 sendDataUnitBytes_;
    u32 recvDataUnitBytes_;
    const std::map<u32, std::vector<u64>>* rankSendDisplsMapPtr_{nullptr};
    const std::map<u32, std::vector<u64>>* rankRecvDisplsMapPtr_{nullptr};
    HcclWorkflowMode workMode_;
    bool isAlltoAllZCopyMode_;
};
} // namespace hccl

#endif /* ALLTOALL_V_PAIRWISE_PUB_H */
