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
#include "all_reduce_ahc_broke.h"

namespace hccl {
AllReduceAHCBroke::AllReduceAHCBroke(const HcclDispatcher dispatcher) : AllReduceAHCBase(dispatcher) {}

AllReduceAHCBroke::~AllReduceAHCBroke() {}

HcclResult AllReduceAHCBroke::DisposeSubGroups(const u32 rank)
{
    CommAHCBaseInfo::DisposeSubGroups(rank, globalSubGroups_, level0SubGroups_, level1SubGroups_);
    return HCCL_SUCCESS;
}

HcclResult AllReduceAHCBroke::CommAHCInfoInit()
{
    commAHCBaseInfo_.reset(new (std::nothrow) CommBrokeAlignInfo(level0SubGroups_));
    CHK_SMART_PTR_NULL(commAHCBaseInfo_);
    CHK_RET(commAHCBaseInfo_->Init(AHCOpType::AHC_OP_TYPE_ALLREDUCE, ahcAlgOption_));
    return HCCL_SUCCESS;
}

HcclResult AllReduceAHCBroke::RunInterAllReduce(
    const u32 rank, const std::vector<LINK>& links, const std::unique_ptr<CommAHCBaseInfo>& commAHCBaseInfo)
{
    if (!commAHCBaseInfo->IsNeedInterProc(rank)) {
        HCCL_DEBUG("[AllReduceAHCBroke][RunInterAllReduceBrokeType] rank[%u] bypass inter proc", rank);
        return HCCL_SUCCESS;
    }

    HCCL_INFO("[AllReduceAHCBroke][RunInterAllReduceBrokeType] begin inter AllReduce rank[%u]", rank);

    // 获取当前rank的组间rank
    u32 interRank = commAHCBaseInfo->GetInterRank(0, rank);

    // 创建执行算子实列
    std::unique_ptr<AlgTemplateBase> tempAlg;
    CHK_RET(commAHCBaseInfo->GetInterAlgTemplateOpInstance(
        AHCOpType::AHC_OP_TYPE_ALLREDUCE, tempAlg, dispatcher_, reduceAttr_, extendFlag_, ahcExtendPreparePara_));

    HCCL_DEBUG("[AllReduceAHCBroke][perform] rank[%u] begin calc inter slices", rank);

    std::vector<std::vector<Slice>> interSlicesVector;
    std::vector<std::vector<LINK>> interLinksVector;
    std::vector<u32> logicCardList;
    CHK_RET(commAHCBaseInfo->CalcInterSlicesAndLinks(
        rank, DataUnitSize(dataType_), count_, links, interLinksVector, interSlicesVector, logicCardList));

    HCCL_DEBUG("[AllReduceAHCBroke][perform] rank[%u] end calc inter slices", rank);

    if (interLinksVector.size() != interSlicesVector.size()) {
        HCCL_ERROR(
            "[AllReduceAHCBroke][RunInterAllReduceBrokeType]rank[%u] linksVector size[%llu] is not equal to "
            "slicesVector size [%u]",
            rank, interLinksVector.size(), interSlicesVector.size());
        return HCCL_E_INTERNAL;
    }

    HCCL_DEBUG(
        "[AllReduceAHCBroke][RunInterAllReduceBrokeType] run inst rank[%u] interRank=%u, interLinksSize=%u", rank,
        interRank, interLinksVector[0].size());

    CHK_RET(
        RunInstance(interRank, interLinksVector[0], interSlicesVector[0], tempAlg, AHCOpType::AHC_OP_TYPE_ALLREDUCE));

    HCCL_DEBUG("[AllReduceAHCBroke][RunInterAllReduceBrokeType] end inter AllReduce rank[%u]", rank);

    return HCCL_SUCCESS;
}
REGISTER_TEMPLATE(TemplateType::TEMPLATE_ALL_REDUCE_AHC_BROKE, AllReduceAHCBroke);
} // namespace hccl
