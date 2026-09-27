/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ALL_REDUCE_GRAPH_PIPELINE_PUB_H
#define ALL_REDUCE_GRAPH_PIPELINE_PUB_H

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

class AllReduceGraphPipeline : public AlgTemplateBase {
public:
    using AlgTemplateBase::Prepare;
    explicit AllReduceGraphPipeline(const HcclDispatcher dispatcher);
    ~AllReduceGraphPipeline() override;

    // 新增的两段式构造函数，获取实例后要无脑调用实现构造函数功能，后续还要调用其它的基类Prepare函数实现其它成员变量初始化
    HcclResult Prepare(u64 reduceAttrBitMap, HcomCollOpInfo* opInfo = nullptr) override;

    HcclResult Prepare(
        const HcomCollOpInfo* opInfo, DeviceMem& cclBufferA, DeviceMem& cclBufferB, const u64 count,
        const SubCommInfo& level1CommInfo, const SubCommInfo& level0CommInfo, Stream& mainStream,
        std::vector<Stream>& subStream, std::vector<std::shared_ptr<LocalNotify>>& notifyMain,
        std::vector<std::shared_ptr<LocalNotify>>& notifySub) override;

    HcclResult RunAsync() override;

protected:
private:
    HcclResult RunReduceScatterIntraServer(u32 step);
    HcclResult RunReduceScatterInterServer(u32 step, const LINK& prevInterLink, const LINK& nextInterLink);
    HcclResult RunAllGatherIntraServer(u32 step);
    HcclResult RunAllGatherInterServer(u32 step, const LINK& prevInterLink, const LINK& nextInterLink);
    HcclResult MainWaitSub();
    HcclResult SubRecordMain();
    HcclResult MainRecordSub();
    HcclResult SubWaitMain();

    HcomCollOpInfo* opInfo_;

    void* usrInMem_ = nullptr;
    void* usrOutMem_ = nullptr;
    u32 unitSize_ = 0;
    u64 curSize_ = 0;
    u64 sliceCount_ = 0;
    u64 memSliceSize_ = 0;
    u64 lastSliceCount_ = 0;
    u64 lastSliceSize_ = 0;

    HcclReduceOp reductionOp_;
    HcclDataType dataType_;

    std::vector<Stream> subStreams_;

    std::vector<std::shared_ptr<LocalNotify>> streamNotifyMain_;
    std::vector<std::shared_ptr<LocalNotify>> streamNotifySub_;

    std::unique_ptr<Sender> senderInfo_;
    std::unique_ptr<Reducer> reducerInfo_;
    u64 reduceAttr_; /* 0x1:表示data_type + reduce_type支持inlinereduce  */
    u32 intraRankSize_ = 0;
    u32 interRankSize_ = 0;
    u32 intraRankId_ = 0;
    u32 interRankId_ = 0;
    u32 rankId_ = 0;

    std::vector<LINK> intraLinks_;
    std::vector<LINK> interLinks_;
};
} // namespace hccl

#endif /* ALL_REDUCE_GRAPH_PIPELINE_PUB_H */
