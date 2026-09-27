/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_INS_QUEUE_H
#define HCCLV2_INS_QUEUE_H

#include "instruction.h"
#include "hierarchical_queue.h"

#include <unordered_set>
#include <vector>
namespace Hccl {
using namespace std;

class InsQueue : public HierarchicalQueue<Instruction, InsQueue>, public enable_shared_from_this<InsQueue> {
public:
    vector<LinkData> GetUniqueLinks()
    {
        unordered_set<LinkData> uniqueLinks;
        for (auto iter = Iter(); iter.HasNext(); ++iter) {
            auto linkPtr = iter->GetLink();
            if (linkPtr == nullptr) {
                continue;
            }
            uniqueLinks.insert(*linkPtr);
        }
        for (auto slaveIter = IterSlaves(); slaveIter.HasNext(); ++slaveIter) {
            for (auto iterSlave = slaveIter->Iter(); iterSlave.HasNext(); ++iterSlave) {
                auto linkPtrSlave = iterSlave->GetLink();
                if (linkPtrSlave == nullptr) {
                    continue;
                }
                uniqueLinks.insert(*linkPtrSlave);
            }
        }
        return {uniqueLinks.begin(), uniqueLinks.end()};
    };

    void Append(unique_ptr<Instruction> ins) override
    {
        auto insPtr = ins.get();
        if (insPtr->GetType() == InstructionType::LOCAL_POST_TO) {
            InsLocalPostTo& postTo = static_cast<InsLocalPostTo&>(*insPtr);
            postTo.SetPostQid(GetId());
        } else if (insPtr->GetType() == InstructionType::LOCAL_WAIT_FROM) {
            InsLocalWaitFrom& waitFrom = static_cast<InsLocalWaitFrom&>(*insPtr);
            waitFrom.SetWaitQid(GetId());
        } else if (insPtr->GetType() == InstructionType::LOCAL_WAIT_GROUP) {
            InsLocalWaitGroup& waitGroup = static_cast<InsLocalWaitGroup&>(*insPtr);
            waitGroup.SetWaitQid(GetId());
        } else if (insPtr->GetType() == InstructionType::LOCAL_BCAST_POST) {
            InsLocalBcastPost& bcastPost = static_cast<InsLocalBcastPost&>(*insPtr);
            bcastPost.SetPostQid(GetId());
        }
        HierarchicalQueue::Append(std::move(ins));
    }
};
} // namespace Hccl

#endif // !HCCLV2_INS_QUEUE_H
