/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef REDUCE_SCATTER_PIPELINE_PUB_H
#define REDUCE_SCATTER_PIPELINE_PUB_H

#include <vector>
#include <memory>
#include <hccl/hccl_types.h>
#include "hccl/base.h"
#include "externalinput_pub.h"
#include "mem_device_pub.h"
#include "stream_pub.h"
#include "dispatcher.h"
#include "alg_template_base_pub.h"
#include "reducer_pub.h"
#include "sender_pub.h"

namespace hccl {
class ReduceScatterPipeline : public AlgTemplateBase {
public:
    explicit ReduceScatterPipeline(const HcclDispatcher dispatcher);
    ~ReduceScatterPipeline() override;

    // 适配新CollExecutor接口
    HcclResult Prepare(
        HcomCollOpInfo* opInfo, DeviceMem& cclBuffer, const u64 count, const u64 bufferSize, const u64 offset,
        const SubCommInfo& level0CommInfo, const SubCommInfo& level1CommInfo, Stream& mainStream,
        std::vector<Stream>& subStream, std::vector<std::shared_ptr<LocalNotify>>& notifyMain,
        std::vector<std::shared_ptr<LocalNotify>>& notifySub, u64 reduceAttrBitMap) override;

    HcclResult RunAsync() override;
    HcclResult
    GetNslbAdjInfo(const u32 rank, const u32 rankSize, const std::vector<LINK>& links, AdjInfo& nslbAdjInfo) override;

protected:
    virtual HcclResult RunIntraServer(u32 step, u64 remoteOffset);
    virtual HcclResult RunInterServer(u32 step, const LINK& prevInterLink, const LINK& nextInterLink);
    virtual HcclResult CopyToScratchBuffer(u32 step);
    HcclResult MainWaitSub(u32 begin);
    HcclResult SubRecordMain(u32 begin);
    HcclResult MainRecordSub(u32 begin);
    HcclResult SubWaitMain(u32 begin);

    HcomCollOpInfo* opInfo_{nullptr};

    void* usrInMem_ = nullptr;
    void* usrOutMem_ = nullptr;
    u64 count_ = 0;
    u32 unitSize_ = 0;
    u64 curSize_ = 0;
    u64 memSliceSize_ = 0;
    u64 blockSize_ = 0;
    u64 bufferSize_ = 0;
    HcclReduceOp reductionOp_;
    HcclDataType dataType_;

    DeviceMem cclBuffer_;
    std::vector<DeviceMem> dmaMem_;
    u32 pipDepth_ = PIPELINE_DEPTH; // 流水深度设置为3，即localCopy、SDMA、RDMA并发

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

#endif /* REDUCE_SCATTER_PIPELINE_PUB_H */
