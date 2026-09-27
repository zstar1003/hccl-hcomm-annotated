/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ALL_GATHER_UNIFIED_MARCH_PUB_H
#define ALL_GATHER_UNIFIED_MARCH_PUB_H
#include "alg_template_base_pub.h"

namespace hccl {
class AllGatherUnifiedMarch : public AlgTemplateBase {
public:
    using AlgTemplateBase::Prepare;
    explicit AllGatherUnifiedMarch(const HcclDispatcher dispatcher);
    ~AllGatherUnifiedMarch() override;
    HcclResult Prepare(
        const Stream& mainStream, SubCommInfo& level0CommInfo, DeviceMem& userInput, DeviceMem& userOutput,
        DeviceMem& usrInMem, DeviceMem& usrOutMem, u64 blockDataByte, std::vector<Stream>& subStreams,
        const std::vector<std::shared_ptr<LocalNotify>>& meshSignalMainToSub,
        const std::vector<std::shared_ptr<LocalNotify>>& meshSignalSubToMain,
        const std::vector<std::vector<Slice>>& multRingsUserMemSlice, const u64 baseOffset = 0) override;
    HcclResult RunAsync() override;

protected:
private:
    std::string GetStreamIndexString();
    HcclResult NotifySubStreamStart(u32 streamSize);
    HcclResult WaitSubStreamFinish(u32 streamSize);
    HcclResult NotifyNeighborsStart(LINK& prevIntraLink, LINK& nextIntralLink, u32 neighbors);
    HcclResult NotifyNeighborsEnd(LINK& prevIntraLink, LINK& nextIntralLink, u32 neighbors);
    HcclResult DoSerialSDMA(
        void* remoteSrcAddr, u64 remoteOffsetByte, void* dstAddr, Stream& temStream, LINK& tmpLink, u64 memSize,
        u32 step = INVALID_UINT);
    HcclResult RunSingleStep(u32 ringPrevRank, u32 ringNextRank, u32 step, u32 totalStep);
    HcclResult RunLastStep(u32 ringPrevRank, u32 ringNextRank, u32 totalStep);

    Stream mainStream_;
    std::vector<Stream> subStreams_;
    std::vector<std::shared_ptr<LocalNotify>> meshSignalMainToSub_;
    std::vector<std::shared_ptr<LocalNotify>> meshSignalSubToMain_;
    u32 intraRank_;
    u32 intraRankSize_;
    u64 blockDataByte_;
    u32 notifyIdx_ = 0; // 新增notify资源索引
    std::vector<LINK> links_;
    DeviceMem userInput_;
    DeviceMem userOutput_;
    DeviceMem usrInMem_;
    DeviceMem usrOutMem_;
    std::vector<std::vector<Slice>> multRingsUserMemSlice_; // 记录server内的要发送的不连续数据块
};
} // namespace hccl
#endif /* ALLTOALL_V_MESH_READ_ONLY_PUB_H */
