/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef REDUCE_SCATTER_HD_STAGE_PUB_H
#define REDUCE_SCATTER_HD_STAGE_PUB_H

#include "alg_template_base_pub.h"
#include "mem_host_pub.h"
#include "mem_device_pub.h"
#include "stream_pub.h"
namespace hccl {
class ReduceScatterHDStage : public AlgTemplateBase {
public:
    explicit ReduceScatterHDStage(const HcclDispatcher dispatcher);
    ~ReduceScatterHDStage() override;

    HcclResult Prepare(
        DeviceMem& inputMem, DeviceMem& outputMem, DeviceMem& scratchMem, const u64 count, const HcclDataType dataType,
        const Stream& stream, const HcclReduceOp reductionOp, const u32 root, const std::vector<Slice>& slices,
        const u64 baseOffset, const u64 reduceAttrBitMap, std::vector<Stream>& meshStreams,
        std::vector<std::shared_ptr<LocalNotify>>& meshSignal, std::vector<std::shared_ptr<LocalNotify>>& meshSignalAux,
        u32 userRank, const HcomCollOpInfo* opInfo = nullptr) override;
    HcclResult RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK>& links) override;

protected:
private:
    HcclResult MainRecordSub(u32 streamNum);
    HcclResult SubWaitMain(u32 streamNum);
    HcclResult MainWaitSub(u32 streamNum);
    HcclResult SubRecordMain(u32 streamNum);
    HcclResult PrepareSliceData(u32 rankSize);
    HcclResult RunReduceScatterStage(u32 rank, u32 rankSize, const std::vector<LINK>& links);
    HcclResult RunReduceScatterStage1st(u32 rank, u32 rankSize, const std::vector<LINK>& links);
    HcclResult RunBetweenStep(u32 rank, u32 neighCur, u32 neighNext, const std::vector<LINK>& links);
    HcclResult RunReduceScatterRead(u32 rank, u32 rankSize, const std::vector<LINK>& links);
    HcclResult RunReduceScatterStageFinal(u32 rank, u32 rankSize, u32 peer, const std::vector<LINK>& links);
    inline u32 BackwardRank(u32 rank, u32 rankSize, u32 step) const
    {
        if (rankSize == 0) {
            return 0;
        }
        return (rank + rankSize - step) % rankSize;
    }
    u64 reduceAttr_ = 0;
    const u32 base = 2;
    u32 userRank_ = 0;
    std::vector<Stream> meshStreams_;                                            /* * 多steam* */
    const std::vector<std::shared_ptr<LocalNotify>>* meshSignalPtr_{nullptr};    /* 每个ring创建一个signal */
    const std::vector<std::shared_ptr<LocalNotify>>* meshSignalAuxPtr_{nullptr}; /* 从stream wait，主steam record */
    const HcomCollOpInfo* opInfo_{nullptr};
    std::map<u32, std::vector<Slice>> sliceMap_;
    u32 nSteps_ = 0;
};
} // namespace hccl
#endif /* REDUCE_SCATTER_HD_STAGE_H */
