/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BROADCAST_ONESHOT_PUB_H
#define BROADCAST_ONESHOT_PUB_H

#include "alg_template_base_pub.h"

namespace hccl {
class BroadcastHD : public AlgTemplateBase {
public:
    explicit BroadcastHD(const HcclDispatcher dispatcher);
    ~BroadcastHD() override;

    HcclResult Prepare(
        DeviceMem& inputMem, DeviceMem& outputMem, DeviceMem& scratchMem, const u64 count, const HcclDataType dataType,
        const Stream& stream, const HcclReduceOp reductionOp, const u32 root, std::vector<Stream>& meshStreams,
        const std::vector<std::shared_ptr<LocalNotify>>& meshSignal,
        const std::vector<std::shared_ptr<LocalNotify>>& meshSignalAux, u32 interRank,
        const HcomCollOpInfo* opInfo) override;
    HcclResult RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK>& links) override;

protected:
private:
    HcclResult MainRecordSub();
    HcclResult SubWaitMain();
    HcclResult MainWaitSub();
    HcclResult SubRecordMain();
    HcclResult PrepareStep(u32 rankSize);
    HcclResult RunFinalStep(u32 rank, u32 rankSize, const std::vector<LINK>& links);
    HcclResult RunSend(u32 rank, u32 step, u32 rankSize, const std::vector<LINK>& links);
    HcclResult RunSendFirst(u32 rank, u32 rankSize, const std::vector<LINK>& links);
    HcclResult RunReceive(u32 rank, u32 step, u32 rankSize, const std::vector<LINK>& links);
    HcclResult RunReceiveFirst(u32 rank, u32 rankSize, const std::vector<LINK>& links);
    u32 GetDstRank(u32 rank, u32 step, u32 rankSize);
    u32 localRank_ = 0;
    std::vector<Stream> meshStreams_;                                            /* * 多steam* */
    const std::vector<std::shared_ptr<LocalNotify>>* meshSignalPtr_{nullptr};    /* 每个ring创建一个signal */
    const std::vector<std::shared_ptr<LocalNotify>>* meshSignalAuxPtr_{nullptr}; /* 从stream wait，主steam record */
    const HcomCollOpInfo* opInfo_{nullptr};
    DeviceMem emptyMem_;
    std::map<u32, u32> stepMap_;
    u32 nSteps_ = 0;
    const u32 base = 2;
};
} // namespace hccl
#endif /* BROADCAST_ONESHOT_PUB_H */
