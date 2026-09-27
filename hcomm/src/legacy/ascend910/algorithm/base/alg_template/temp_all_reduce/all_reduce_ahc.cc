/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "alg_template_register.h"
#include "all_reduce_ahc.h"

namespace hccl {
AllReduceAHC::AllReduceAHC(const HcclDispatcher dispatcher) : AllReduceAHCBase(dispatcher) {}

AllReduceAHC::~AllReduceAHC() {}

HcclResult AllReduceAHC::DisposeSubGroups(const u32 rank)
{
    CommAHCBaseInfo::DisposeSubGroups(rank, globalSubGroups_, level0SubGroups_, level1SubGroups_);
    return HCCL_SUCCESS;
}

HcclResult AllReduceAHC::CommAHCInfoInit()
{
    commAHCBaseInfo_.reset(new (std::nothrow) CommAHCAlignInfo(level0SubGroups_));
    CHK_SMART_PTR_NULL(commAHCBaseInfo_);
    CHK_RET(commAHCBaseInfo_->Init(AHCOpType::AHC_OP_TYPE_ALLREDUCE, ahcAlgOption_));
    return HCCL_SUCCESS;
}

HcclResult AllReduceAHC::RunInterAllReduce(
    const u32 rank, const std::vector<LINK>& links, const std::unique_ptr<CommAHCBaseInfo>& commAHCBaseInfo)
{
    HCCL_INFO("[AllReduceAHC][RunInterAllReduce] begin inter AllReduce rank[%u]", rank);

    // 获取当前rank的组间rank
    u32 interRank = commAHCBaseInfo->GetInterRank(0, rank);

    // 创建执行算子实列
    std::unique_ptr<AlgTemplateBase> tempAlg;
    CHK_RET(commAHCBaseInfo->GetInterAlgTemplateOpInstance(
        AHCOpType::AHC_OP_TYPE_ALLREDUCE, tempAlg, dispatcher_, reduceAttr_, extendFlag_, ahcExtendPreparePara_));

    std::vector<std::vector<Slice>> interSlicesVector;
    std::vector<std::vector<LINK>> interLinksVector;
    std::vector<u32> logicCardList;
    CHK_RET(commAHCBaseInfo->CalcInterSlicesAndLinks(
        rank, DataUnitSize(dataType_), count_, links, interLinksVector, interSlicesVector, logicCardList));

    if (interLinksVector.size() != interSlicesVector.size()) {
        HCCL_ERROR(
            "[AllReduceAHC][RunInterAllReduce]rank[%u] linksVector size[%llu] is not equal to slicesVector size[%u]",
            rank, interLinksVector.size(), interSlicesVector.size());
        return HCCL_E_INTERNAL;
    }

    for (u32 i = 0; i < logicCardList.size(); ++i) {
        HCCL_DEBUG(
            "[AllReduceAHC][RunInterAllReduce] run logicCard[%u] rank[%u] interRank=%u, interLinksSize=%u",
            logicCardList[i], rank, interRank, interLinksVector[i].size());
        CHK_RET(RunInstance(
            interRank, interLinksVector[i], interSlicesVector[i], tempAlg, AHCOpType::AHC_OP_TYPE_ALLREDUCE));
    }
    return HCCL_SUCCESS;
}
REGISTER_TEMPLATE(TemplateType::TEMPLATE_ALL_REDUCE_AHC, AllReduceAHC);
} // namespace hccl
