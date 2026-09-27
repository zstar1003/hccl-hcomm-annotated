/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_rep.h"
#include "ccu_interface_assist.h"

#include "string_util.h"
#include "exception_util.h"
#include "ccu_api_exception.h"
namespace Hccl {
namespace CcuRep {

    void LoopGroupCall::Run(
        const std::vector<LoopCall>& loopVec, const std::vector<Variable>& loopCfg,
        const std::vector<Executor>& executors, Variable paraCfgIn, Variable offsetCfgIn) const
    {
        Variable var1, var2;
        auto ret1 = CreateVariable(context, var1);
        auto ret2 = CreateVariable(context, var2);
        if (ret1 != HcclResult::HCCL_SUCCESS || ret2 != HcclResult::HCCL_SUCCESS) {
            THROW<CcuApiException>("CreateVariable failed. ret1[%d], ret2[%d]", ret1, ret2);
        }

        if (executors.size() < loopVec.size() || loopCfg.size() < loopVec.size()) {
            THROW<CcuApiException>(
                "Executors size[%lu] or loopCfg size[%lu] is less than loopVec size", executors.size(), loopCfg.size());
        }

        auto loopGroup = std::make_shared<CcuRepLoopGroup>(var1, var2);

        std::vector<std::shared_ptr<CcuRepLoop>> loops;
        for (uint32_t index = 0; index < loopVec.size(); index++) {
            Variable repVar;
            auto ret3 = CreateVariable(context, repVar);
            if (ret3 != HcclResult::HCCL_SUCCESS) {
                THROW<CcuApiException>("CreateVariable failed. ret3[%d]", ret3);
            }
            auto repLoop = std::make_shared<CcuRepLoop>(loopVec[index].GetLabel(), repVar);
            AppendToContext(context, repLoop->SetLoopParam(executors[index], loopCfg[index]));
            loops.push_back(repLoop);
        }

        Variable hideLoopVar;
        auto ret4 = CreateVariable(context, hideLoopVar);
        if (ret4 != HcclResult::HCCL_SUCCESS) {
            THROW<CcuApiException>("CreateVariable failed. ret4[%d]", ret4);
        }
        auto hideLoop = std::make_shared<CcuRepJump>("hideLoop", hideLoopVar);
        auto hideLoopLabel = std::make_shared<CcuRepJumpLabel>("hideLoop");
        hideLoop->Reference(hideLoopLabel);

        AppendToContext(context, loopGroup->SetParallelParam(paraCfgIn));
        AppendToContext(context, loopGroup->SetOffsetParam(offsetCfgIn));
        AppendToContext(context, loopGroup);

        AppendToContext(context, hideLoop);
        for (auto loop : loops) {
            AppendToContext(context, loop);
        }
        AppendToContext(context, hideLoopLabel);
    }

}; // namespace CcuRep
}; // namespace Hccl
