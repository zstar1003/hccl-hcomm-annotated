/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef REDUCE_SCATTER_MESH_DIRECT_H
#define REDUCE_SCATTER_MESH_DIRECT_H

#include "alg_template_base_pub.h"

namespace hccl {
class ReduceScatterMeshDirect : public AlgTemplateBase {
public:
    explicit ReduceScatterMeshDirect(const HcclDispatcher dispatcher);
    ~ReduceScatterMeshDirect() override;
    HcclResult Prepare(
        DeviceMem& inputMem, DeviceMem& outputMem, DeviceMem& scratchMem, const u64 count, const HcclDataType dataType,
        const Stream& stream, const HcclReduceOp reductionOp, const u32 root, const std::vector<Slice>& slices,
        const u64 baseOffset, const u64 reduceAttrBitMap, std::vector<Stream>& meshStreams,
        std::vector<std::shared_ptr<LocalNotify>>& meshSignal, std::vector<std::shared_ptr<LocalNotify>>& meshSignalAux,
        u32 userRank, const HcomCollOpInfo* opInfo = nullptr) override;
    HcclResult RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK>& links) override;

private:
    HcclResult MainRecordSub();
    HcclResult SubWaitMain();
    HcclResult MainWaitSub();
    HcclResult SubRecordMain();

    u64 reduceAttr_ = 0;
    u32 localRank_ = 0;
    u32 localRankSize_ = 0;
    u32 userRank_ = 0;
    std::vector<Stream> meshStreams_;                                            /* * 多steam* */
    const std::vector<std::shared_ptr<LocalNotify>>* meshSignalPtr_{nullptr};    /* 每个ring创建一个signal */
    const std::vector<std::shared_ptr<LocalNotify>>* meshSignalAuxPtr_{nullptr}; /* 从stream wait，主steam record */
    std::vector<Slice> scratchSlices_;
    const HcomCollOpInfo* opInfo_{nullptr};
};
} // namespace hccl

#endif
