/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BROADCAST_NHR_ONESHOT_PUB_H
#define BROADCAST_NHR_ONESHOT_PUB_H

#include "nonuniform_hierarchical_ring_base_pub.h"

#include "mem_host_pub.h"
#include "mem_device_pub.h"
#include "stream_pub.h"

namespace hccl {
class BroadcastNHROneshot : public NHRBase {
public:
    explicit BroadcastNHROneshot(const HcclDispatcher dispatcher);

    ~BroadcastNHROneshot() override;

    HcclResult RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK>& links) override;

    HcclResult RunAsyncForAllReduce(const u32 rank, const u32 rankSize, const std::vector<LINK>& links);

protected:
private:
    HcclResult RunBroadcastNHROneshot(u32 rank, u32 rankSize, const std::vector<LINK>& links);

    HcclResult SimpleCheck(const u32 rank, const u32 rankSize, const std::vector<LINK>& links);

    HcclResult SdmaRx(LINK& linkLeft, LINK& linkRight, InterServerAlgoStep& stepInfo, const std::vector<LINK>& links);

    HcclResult RdmaTxRx(LINK& linkLeft, LINK& linkRight, InterServerAlgoStep& stepInfo, const std::vector<LINK>& links);

    HcclResult GetStepInfo(u32 step, u32 nSteps, u32 rank, u32 rankSize, InterServerAlgoStep& stepInfo) override;

    u64 localBaseOffset_;
    bool isForAllReduce_;
};
} // namespace hccl

#endif /* BROADCAST_NHR_ONESHOT_PUB_H */
