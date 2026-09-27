/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "execute_selector.h"

#include "base_selector.h"
#include "selector_registry.h"

namespace Hccl {
ExecuteSelector& ExecuteSelector::SetVirtualTopo(RankGraph* rankGraph)
{
    rankGraph_ = rankGraph;
    return *this;
}

ExecuteSelector& ExecuteSelector::SetDevType(DevType devType)
{
    devType_ = devType;
    return *this;
}

ExecuteSelector& ExecuteSelector::SetMyRank(RankId myRank)
{
    myRank_ = myRank;
    return *this;
}

ExecuteSelector& ExecuteSelector::SetRankSize(u32 rankSize)
{
    rankSize_ = rankSize;
    return *this;
}

ExecuteSelector& ExecuteSelector::SetSeverId(std::string severId)
{
    severId_ = severId;
    return *this;
}

ExecuteSelector& ExecuteSelector::SetDeviceNumPerSever(u32 deviceNumPerSever)
{
    deviceNumPerSever_ = deviceNumPerSever;
    return *this;
}

ExecuteSelector& ExecuteSelector::SetServerNum(u32 serverNum)
{
    serverNum_ = serverNum;
    return *this;
}

ExecuteSelector& ExecuteSelector::SetOpConfig(OpExecuteConfig opConfig)
{
    opConfig_ = opConfig;
    return *this;
}

AlgorithmType ExecuteSelector::GetAlgorithmTypeForMC2CCU(const std::string& name) const
{
    Mc2Selector mc2Selector;
    return mc2Selector.GetAlgorithmTypeForMC2CCU(name);
}

HcclResult ExecuteSelector::Run(const CollAlgOperator& op, CollAlgParams& params, std::string& primQueueGenName)
{
    if (rankGraph_ == nullptr) {
        HCCL_ERROR("[Algo][ExecuteSelector] rankGraph_ is nullptr.");
        return HcclResult::HCCL_E_PTR;
    }
    std::map<u32, BaseSelector*> selectors = SelectorRegistry::Global()->GetAllSelectors();

    if (params.isMc2) {
        auto iter = selectors.find(18);
        if (iter == selectors.end()) {
            HCCL_ERROR("[Algo][Selector] CCU selector is not registered.");
            return HcclResult::HCCL_E_NOT_SUPPORT;
        }
        iter->second->SetVirtualTopo(rankGraph_)
            .SetDevType(devType_)
            .SetMyRank(myRank_)
            .SetRankSize(rankSize_)
            .SetSeverId(severId_)
            .SetDeviceNumPerSever(deviceNumPerSever_)
            .SetServerNum(serverNum_)
            .SetIsMc2(params.isMc2);
        if (iter->second->Select(op, params, primQueueGenName) == SelectorStatus::MATCH) {
            HCCL_INFO(
                "[Algo][Selector] The ccu selector[priority of %u] is matched, the selected algo type is %s",
                iter->first, primQueueGenName.c_str());
            return HcclResult::HCCL_SUCCESS;
        }
        HCCL_ERROR("[Algo][Selector] CCU selector can not match for optype[%d].", op.opType);
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }

    selectors = SelectorRegistry::Global()->GetSelectorsByOpType(op.opType);
    HCCL_INFO(
        "[Algo][Selector] The selector nums of optype[%s] is [%zu].", op.opType.Describe().c_str(), selectors.size());
    for (auto iter : selectors) {
        HCCL_DEBUG("[Algo][Selector] The selector[priority of %llu] is running.", iter.first);
        iter.second->SetVirtualTopo(rankGraph_)
            .SetDevType(devType_)
            .SetMyRank(myRank_)
            .SetRankSize(rankSize_)
            .SetSeverId(severId_)
            .SetDeviceNumPerSever(deviceNumPerSever_)
            .SetServerNum(serverNum_)
            .SetOpConfig(opConfig_);
        if (iter.second->Select(op, params, primQueueGenName) == SelectorStatus::MATCH) {
            HCCL_INFO(
                "[Algo][Selector] The selector[priority of %llu] is matched, the selected algo type is %s", iter.first,
                primQueueGenName.c_str());
            return HcclResult::HCCL_SUCCESS;
        }
    }

    HCCL_WARNING("[Algo][Selector] No selector is matched.");
    return HcclResult::HCCL_E_NOT_SUPPORT;
}

} // namespace Hccl
