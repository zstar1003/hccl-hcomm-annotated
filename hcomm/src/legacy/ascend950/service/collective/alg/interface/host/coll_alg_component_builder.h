/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_COLL_ALG_COMPONENT_BUILDER
#define HCCLV2_COLL_ALG_COMPONENT_BUILDER

#include "dev_type.h"
#include "base_config_legacy.h"
#include "rank_gph.h"
#include "coll_alg_component.h"

namespace Hccl {

constexpr u64 DEFAULT_ALLIGN_SIZE = 128;

class CollAlgComponentBuilder {
public:
    CollAlgComponentBuilder& SetRankGraph(RankGraph* rankGraph);
    CollAlgComponentBuilder& SetDevType(DevType devType);
    CollAlgComponentBuilder& SetMyRank(u32 myRank);
    CollAlgComponentBuilder& SetRankSize(u32 rankSize);
    CollAlgComponentBuilder& EnableDetour(bool enableDetour);
    CollAlgComponentBuilder& EnableDataAllign(bool enableAllign);
    CollAlgComponentBuilder& SetAllignSize(u64 allignSize);
    CollAlgComponentBuilder& SetMaxQueue(u32 maxQueue);
    CollAlgComponentBuilder& SetMaxLink(u32 maxLink);
    CollAlgComponentBuilder& SetMaxDepQueuePairs(u32 maxDepQueuePairs);
    CollAlgComponentBuilder& SetDmaMode(const DmaMode& dmaMode);
    CollAlgComponentBuilder& SetMainboardId(u64 mainBoardId); // Stub 方法
    CollAlgComponentPtr Build();

private:
    RankGraph* rankGraph_ = nullptr;
    DevType devType_;
    u32 myRank_ = INVALID_RANKID;
    u32 rankSize_ = 0;

    bool enableDetour_ = false;
    bool enableAllign_ = true;
    u64 allignSize_ = DEFAULT_ALLIGN_SIZE;

    u32 maxQueue_ = 0;
    u32 maxLink_ = 0;
    u32 maxDepQueuePairs_ = 0;
    DmaMode dmaMode_ = DmaMode::DEFAULT;
};
} // namespace Hccl

#endif
