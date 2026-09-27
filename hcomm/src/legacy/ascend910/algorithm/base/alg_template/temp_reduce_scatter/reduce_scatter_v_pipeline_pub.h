/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef REDUCE_SCATTER_V_PIPELINE_PUB_H
#define REDUCE_SCATTER_V_PIPELINE_PUB_H

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
#include "reduce_scatter_pipeline_pub.h"
namespace hccl {
class ReduceScatterVPipeline : public ReduceScatterPipeline {
public:
    explicit ReduceScatterVPipeline(const HcclDispatcher dispatcher);
    ~ReduceScatterVPipeline() override;
    using AlgTemplateBase::Prepare;
    // 适配新CollExecutor接口
    HcclResult Prepare(
        HcomCollOpInfo* opInfo, DeviceMem& cclBuffer, const u64 bufferSize, const std::vector<Slice>& slices,
        const SubCommInfo& level0CommInfo, const SubCommInfo& level1CommInfo, Stream& mainStream,
        std::vector<Stream>& subStream, std::vector<std::shared_ptr<LocalNotify>>& notifyMain,
        std::vector<std::shared_ptr<LocalNotify>>& notifySub, u64 reduceAttrBitMap) override;

    HcclResult RunAsync() override;

protected:
    HcclResult RunIntraServer(u32 step, u64 remoteOffset) override;
    HcclResult RunInterServer(u32 step, const LINK& prevInterLink, const LINK& nextInterLink) override;
    HcclResult CopyToScratchBuffer(u32 step) override;
};
} // namespace hccl

#endif /* REDUCE_SCATTER_V_PIPELINE_PUB_H */
