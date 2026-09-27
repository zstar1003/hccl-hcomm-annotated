/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ALLTOALL_V_STAGED_PAIRWISE_PUB_H
#define ALLTOALL_V_STAGED_PAIRWISE_PUB_H

#include "alltoallv_staged_base_pub.h"

namespace hccl {
class AlltoAllVStagedPairwise : public AlltoAllVStagedBase {
public:
    using AlgTemplateBase::Prepare;
    explicit AlltoAllVStagedPairwise(const HcclDispatcher dispatcher);

    ~AlltoAllVStagedPairwise() override;
    HcclResult Prepare(
        DeviceMem& sendMem, DeviceMem& recvMem, StageAlltoAllVAddrInfo& sendAddrInfo,
        StageAlltoAllVAddrInfo& recvAddrInfo, bool isAlltoAllZCopyMode, Stream& mainStream) override;
    HcclResult Prepare(
        DeviceMem& sendMem, DeviceMem& recvMem, DeviceMem& scratchInputMem, DeviceMem& scratchOutputMem,
        StageAlltoAllVAddrInfo& sendAddrInfo, StageAlltoAllVAddrInfo& recvAddrInfo, bool isAlltoAllZCopyMode,
        Stream& mainStream) override;
    HcclResult RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK>& links) override;

protected:
private:
    HcclResult RunZCopyAlltoAll(const u32 rank, const u32 rankSize, const std::vector<LINK>& links);
    HcclResult RunBCopyAlltoAll(const u32 rank, const u32 rankSize, const std::vector<LINK>& links);

    // 单算子执行相关接口 start
    void CalcSendRecvTimes(u64& sendTimes, u64& recvTimes, const u32 prevRank, const u32 nextRank);
    void LoadPolicies(
        const u32 rank, StageAlltoAllVAddrInfo& addrInfos, std::vector<std::list<OneSendRecvAddrInfo>>& policies) const;
    HcclResult CheckPolicies(const u64 times, const std::vector<std::list<OneSendRecvAddrInfo>>& policies) const;
    void SplitSendRecvAddrInfo(
        OneSendRecvAddrInfo& curLastInfo, OneSendRecvAddrInfo& addrInfo, const u64& curCCLBufSize) const;
    HcclResult SendRecv(
        const u64 curSendTime, const std::vector<std::list<OneSendRecvAddrInfo>>& sendPolicies, const u64 curRecvTime,
        const std::vector<std::list<OneSendRecvAddrInfo>>& recvPolicies, std::shared_ptr<Transport> prevTransport,
        std::shared_ptr<Transport> nextTransport);
    // 单算子执行相关接口 end

    HcclResult ExecuteBarrier(std::shared_ptr<Transport> preLink, std::shared_ptr<Transport> aftLink) const;
    HcclResult ExecuteBarrier(
        bool hasSend, bool hasRecv, std::shared_ptr<Transport> preLink, std::shared_ptr<Transport> aftLink) const;

    // 单算子CCLbuf
    DeviceMem scratchInputMem_;
    DeviceMem scratchOutputMem_;
    u64 scratchMemSize_ = 0; // 单算子CCLbuf大小
};
} // namespace hccl
#endif /* ALLTOALL_V_STAGED_PAIRWISE_PUB_H */
