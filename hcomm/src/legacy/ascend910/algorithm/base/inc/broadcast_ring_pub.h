/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BROADCAST_RING_PUB_H
#define BROADCAST_RING_PUB_H

#include "alg_template_base_pub.h"

namespace hccl {
class BroadcastRing : public AlgTemplateBase {
public:
    explicit BroadcastRing(const HcclDispatcher dispatcher);
    ~BroadcastRing() override;

    HcclResult
    RunAsync(const u32 rank, const u32 rankSize, const std::vector<std::shared_ptr<Transport>>& links) override;
    HcclResult
    GetNslbAdjInfo(const u32 rank, const u32 rankSize, const std::vector<LINK>& links, AdjInfo& nslbAdjInfo) override;

protected:
private:
    std::shared_ptr<Transport> linkLeft_;
    std::shared_ptr<Transport> linkRight_;

    DeviceMem scratch_; /* * 临时deviceMem */
};
} // namespace hccl

#endif /* BROADCAST_RING_PUB_H */
