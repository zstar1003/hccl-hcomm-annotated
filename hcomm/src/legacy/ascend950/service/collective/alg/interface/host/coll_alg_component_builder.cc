/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_alg_component_builder.h"
#include "rank_gph.h"
#include "host/coll_alg_component.h"

namespace Hccl {
CollAlgComponentBuilder& CollAlgComponentBuilder::SetRankGraph(RankGraph* rankGraph)
{
    rankGraph_ = rankGraph;
    return *this;
}

CollAlgComponentBuilder& CollAlgComponentBuilder::SetDevType(DevType devType)
{
    devType_ = devType;
    return *this;
}

CollAlgComponentBuilder& CollAlgComponentBuilder::SetMyRank(u32 myRank)
{
    myRank_ = myRank;
    return *this;
}

CollAlgComponentBuilder& CollAlgComponentBuilder::SetRankSize(u32 rankSize)
{
    rankSize_ = rankSize;
    return *this;
}

CollAlgComponentBuilder& CollAlgComponentBuilder::EnableDetour(bool enableDetour)
{
    enableDetour_ = enableDetour;
    return *this;
}

CollAlgComponentBuilder& CollAlgComponentBuilder::SetDmaMode(const DmaMode& dmaMode)
{
    dmaMode_ = dmaMode;
    return *this;
}

CollAlgComponentBuilder& CollAlgComponentBuilder::EnableDataAllign(bool enableAllign)
{
    enableAllign_ = enableAllign;
    return *this;
}

CollAlgComponentBuilder& CollAlgComponentBuilder::SetAllignSize(u64 allignSize)
{
    allignSize_ = allignSize;
    return *this;
}

CollAlgComponentBuilder& CollAlgComponentBuilder::SetMaxQueue(u32 maxQueue)
{
    maxQueue_ = maxQueue;
    return *this;
}

CollAlgComponentBuilder& CollAlgComponentBuilder::SetMaxLink(u32 maxLink)
{
    maxLink_ = maxLink;
    return *this;
}

CollAlgComponentBuilder& CollAlgComponentBuilder::SetMaxDepQueuePairs(u32 maxDepQueuePairs)
{
    maxDepQueuePairs_ = maxDepQueuePairs;
    return *this;
}

CollAlgComponentBuilder& CollAlgComponentBuilder::SetMainboardId(u64 mainBoardId)
{
    (void)mainBoardId;
    return *this;
}

CollAlgComponentPtr CollAlgComponentBuilder::Build()
{
    CollAlgComponentPtr component = std::make_shared<CollAlgComponent>(rankGraph_, devType_, myRank_, rankSize_);
    component->EnableDetour(enableDetour_);
    component->EnableDataAllign(enableAllign_);
    component->SetAllignSize(allignSize_);
    component->SetMaxQueue(maxQueue_);
    component->SetMaxLink(maxLink_);
    component->SetMaxDepQueuePairs(maxDepQueuePairs_);
    component->SetDmaMode(dmaMode_);
    return component;
};
} // namespace Hccl
