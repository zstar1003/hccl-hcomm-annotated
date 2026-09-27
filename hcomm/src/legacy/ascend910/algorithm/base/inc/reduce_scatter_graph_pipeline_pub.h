/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef REDUCE_SCATTER_GRAPH_PIPELINE_PUB_H
#define REDUCE_SCATTER_GRAPH_PIPELINE_PUB_H

#include "alg_template_base_pub.h"
#include "reducer_pub.h"
#include "sender_pub.h"

namespace hccl {

class ReduceScatterGraphPipeline : public AlgTemplateBase {
public:
    explicit ReduceScatterGraphPipeline(const HcclDispatcher dispatcher);
    ~ReduceScatterGraphPipeline() override;

    // 适配新CollExecutor接口
    HcclResult Prepare(
        HcomCollOpInfo* opInfo, DeviceMem& cclBuffer, const u64 count, const u64 bufferSize, const u64 offset,
        const SubCommInfo& level0CommInfo, const SubCommInfo& level1CommInfo, Stream& mainStream,
        std::vector<Stream>& subStream, std::vector<std::shared_ptr<LocalNotify>>& notifyMain,
        std::vector<std::shared_ptr<LocalNotify>>& notifySub, u64 reduceAttrBitMap) override;

    HcclResult RunAsync() override;

protected:
private:
    HcclResult RunIntraServer(u64 blockIdx);
    HcclResult RunInterServer(u64 blockIdx, const LINK& prevInterLink, const LINK& nextInterLink);
    HcclResult MainWaitSub(u32 begin);
    HcclResult SubRecordMain(u32 begin);
    HcclResult MainRecordSub(u32 begin);
    HcclResult SubWaitMain(u32 begin);

    HcomCollOpInfo* opInfo_{nullptr};

    void* usrInMem_ = nullptr;
    void* usrOutMem_ = nullptr;
    u64 count_ = 0;
    u32 unitSize_ = 0;
    u64 memSliceSize_ = 0;
    HcclReduceOp reductionOp_ = HCCL_REDUCE_RESERVED;
    HcclDataType dataType_ = HCCL_DATA_TYPE_RESERVED;

    std::vector<Stream> subStream_;

    std::vector<std::shared_ptr<LocalNotify>> streamNotifyMain_;
    std::vector<std::shared_ptr<LocalNotify>> streamNotifySub_;

    std::unique_ptr<Sender> senderInfo_;
    std::unique_ptr<Reducer> reducerInfo_;
    u64 reduceAttr_ = 0; /* 0x1:表示data_type + reduce_type支持inlinereduce  */
    u32 intraRankSize_ = 0;
    u32 interRankSize_ = 0;
    u32 intraRankId_ = 0;
    u32 interRankId_ = 0;
    u32 rankId_ = 0;
    u64 offset_ = 0;

    std::vector<LINK> intraLinks_;
    std::vector<LINK> interLinks_;
};
} // namespace hccl

#endif /* REDUCE_SCATTER_GRAPH_PIPELINE_PUB_H */
