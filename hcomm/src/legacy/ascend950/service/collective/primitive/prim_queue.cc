/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "prim_queue.h"

namespace Hccl {

void PrimQueue::Append(unique_ptr<Primitive> prim)
{
    auto primPtr = prim.get();
    if (primPtr->GetType() == PrimType::POST_TO) {
        PrimPostTo& postTo = static_cast<PrimPostTo&>(*primPtr);
        postTo.SetParent(shared_from_this());
    } else if (primPtr->GetType() == PrimType::WAIT_FROM) {
        PrimWaitFrom& waitFrom = static_cast<PrimWaitFrom&>(*primPtr);
        waitFrom.SetParent(shared_from_this());
    } else if (primPtr->GetType() == PrimType::WAIT_GROUP) {
        PrimWaitGroup& waitFrom = static_cast<PrimWaitGroup&>(*primPtr);
        waitFrom.SetParent(shared_from_this());
    }
    HierarchicalQueue::Append(std::move(prim));
}
void PrintPrimQueue(const PrimQueue& queue)
{
    HCCL_INFO("Master queue: ");
    for (auto it = queue.Iter(); it.HasNext(); ++it) {
        HCCL_INFO("    %s", it->Describe().c_str());
    }
    for (auto slaveIt = queue.IterSlaves(); slaveIt.HasNext(); ++slaveIt) {
        HCCL_INFO("Slave queue %u: ", slaveIt->GetId());
        for (auto it = slaveIt->Iter(); it.HasNext(); ++it) {
            HCCL_INFO("    %s", it->Describe().c_str());
        }
    }
}
} // namespace Hccl
