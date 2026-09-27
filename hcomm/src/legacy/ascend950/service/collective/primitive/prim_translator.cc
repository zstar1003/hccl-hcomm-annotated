/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include "prim_translator.h"
#include "prim_rules.h"

namespace Hccl {
PrimTranslator::PrimTranslator()
    : primTranslateRuleMap(
          {{PrimType::POST_TO, GetRule<PrimPostTo>()},
           {PrimType::WAIT_FROM, GetRule<PrimWaitFrom>()},
           {PrimType::WAIT_GROUP, GetRule<PrimWaitGroup>()},
           {PrimType::LOCAL_COPY, GetRule<PrimLocalCopy>()},
           {PrimType::LOCAL_REDUCE, GetRule<PrimLocalReduce>()},
           {PrimType::SEND, GetRule<PrimSend>()},
           {PrimType::RECV, GetRule<PrimRecv>()},
           {PrimType::GROUP, GetRule<PrimGroup>()},
           {PrimType::SEND_REDUCE, GetRule<PrimSendReduce>()},
           {PrimType::RECV_REDUCE, GetRule<PrimRecvReduce>()}})
{}

void PrimTranslator::TranslateOnePrimQue(const PrimQueue& primQueue, shared_ptr<InsQueue> insQueue)
{
    for (auto iter = primQueue.Iter(); iter.HasNext(); ++iter) {
        HCCL_INFO("primitive being translated is %s", iter->Describe().c_str());
        vector<unique_ptr<Instruction>> instructions = primTranslateRuleMap.at(iter->GetType())(*iter);
        for (auto& instruction : instructions) {
            HCCL_INFO("instruction is %s", instruction->Describe().c_str());
            insQueue->Append(std::move(instruction));
        }
    }
}

shared_ptr<InsQueue> PrimTranslator::Translate(const PrimQueue& primQueue)
{
    auto masterInsQue = make_shared<InsQueue>();
    for (auto slaveIter = primQueue.IterSlaves(); slaveIter.HasNext(); ++slaveIter) {
        auto slaveInsQue = masterInsQue->Fork();
        TranslateOnePrimQue(*slaveIter, slaveInsQue);
    }
    TranslateOnePrimQue(primQueue, masterInsQue);
    return masterInsQue;
}
} // namespace Hccl
