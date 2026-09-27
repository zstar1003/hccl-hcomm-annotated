/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ALLTOALL_V_STAGED_MESH_PUB_H
#define ALLTOALL_V_STAGED_MESH_PUB_H

#include "alltoallv_staged_base_pub.h"

namespace hccl {
class AlltoAllVStagedMesh : public AlltoAllVStagedBase {
public:
    using AlgTemplateBase::Prepare;
    explicit AlltoAllVStagedMesh(const HcclDispatcher dispatcher);
    ~AlltoAllVStagedMesh() override;
    HcclResult Prepare(
        DeviceMem& sendMem, DeviceMem& recvMem, StageAlltoAllVAddrInfo& sendAddrInfo,
        StageAlltoAllVAddrInfo& recvAddrInfo, bool isAlltoAllZCopyMode, u32 userRank, Stream& mainStream,
        std::vector<Stream>& subStreams, std::vector<std::shared_ptr<LocalNotify>>& meshSignalMainToSub,
        std::vector<std::shared_ptr<LocalNotify>>& meshSignalSubToMain) override;
    HcclResult Prepare(
        DeviceMem& sendMem, DeviceMem& recvMem, DeviceMem& scratchInputMem, DeviceMem& scratchOutputMem,
        StageAlltoAllVAddrInfo& sendAddrInfo, StageAlltoAllVAddrInfo& recvAddrInfo, bool isAlltoAllZCopyMode,
        u32 userRank, Stream& mainStream, std::vector<Stream>& subStreams,
        std::vector<std::shared_ptr<LocalNotify>>& meshSignalMainToSub,
        std::vector<std::shared_ptr<LocalNotify>>& meshSignalSubToMain) override;
    HcclResult RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK>& links) override;

protected:
private:
    HcclResult RunZCopyMode(const u32 rank, const u32 rankSize, const std::vector<LINK>& links);
    HcclResult LoadTask(
        std::shared_ptr<Transport> destTransport, Stream& currentStream, std::vector<TxMemoryInfo>& txMems,
        std::vector<RxMemoryInfo>& rxMems) const;
    void BuildSendRecvMemoryInfo(std::vector<TxMemoryInfo>& txMems, std::vector<RxMemoryInfo>& rxMems, u32 destRank);
    const std::vector<std::shared_ptr<LocalNotify>>* meshSignalMainToSubPtr_{nullptr};
    const std::vector<std::shared_ptr<LocalNotify>>* meshSignalSubToMainPtr_{nullptr};
    u32 userRank_;
    std::vector<Stream>* subStreamsPtr_{nullptr};
};
} // namespace hccl
#endif /* ALLTOALL_V_STAGED_MESH_PUB_H */
