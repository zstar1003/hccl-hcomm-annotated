/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "all_gather_ahc_broke.h"
#include "alg_template_register.h"

namespace hccl {

AllGatherAHCBroke::AllGatherAHCBroke(const HcclDispatcher dispatcher) : AllGatherAHCBase(dispatcher) {}

AllGatherAHCBroke::~AllGatherAHCBroke() {}

HcclResult AllGatherAHCBroke::DisposeSubGroups(const u32 rank)
{
    CommAHCBaseInfo::DisposeSubGroups(rank, globalSubGroups_, level0SubGroups_, level1SubGroups_);
    return HCCL_SUCCESS;
}

HcclResult AllGatherAHCBroke::CommAHCInfoInit()
{
    commAHCBaseInfo_.reset(new (std::nothrow) CommBrokeAlignInfo(level0SubGroups_));
    CHK_SMART_PTR_NULL(commAHCBaseInfo_);
    CHK_RET(commAHCBaseInfo_->Init(AHCOpType::AHC_OP_TYPE_ALLGATHER, ahcAlgOption_));
    return HCCL_SUCCESS;
}

HcclResult AllGatherAHCBroke::RunInterAllGather(
    const u32 rank, const std::vector<LINK>& links, const std::unique_ptr<CommAHCBaseInfo>& commAHCBaseInfo)
{
    HCCL_INFO("[AllGatherAHCBroke][RunInterAllGather] begin inter AllGather rank[%u]", rank);

    // 创建执行算子实列
    std::unique_ptr<AlgTemplateBase> tempAlg;
    commAHCBaseInfo->GetInterAlgTemplateOpInstance(
        AHCOpType::AHC_OP_TYPE_ALLGATHER, tempAlg, dispatcher_, reduceAttr_, extendFlag_, ahcExtendPreparePara_);

    std::vector<std::vector<Slice>> interSlicesVector;
    std::vector<std::vector<LINK>> interLinksVector;
    std::vector<u32> interRankList;
    CHK_RET(commAHCBaseInfo->CalcInterSlicesAndLinks(
        rank, DataUnitSize(dataType_), count_, links, interLinksVector, interSlicesVector, interRankList));

    HCCL_DEBUG("[AllGatherAHCBroke][RunInterAllGather] run inst rank[%u]", rank);

    for (u32 i = 0; i < interLinksVector.size(); i++) {
        std::vector<Slice> interSlices = interSlicesVector[i];
        std::vector<LINK> interLinks = interLinksVector[i];
        if (interLinks.size() <= 1) {
            continue;
        }
        HCCL_DEBUG("[AllGatherAHCBroke][AHCDEBUG] rank[%u] group[%u] interRank[%u]", rank, i, interRankList[i]);
        CHK_RET(RunInstance(interRankList[i], interLinks, interSlices, tempAlg, AHCOpType::AHC_OP_TYPE_ALLGATHER));
    }

    HCCL_DEBUG("[AllGatherAHCBroke][RunInterAllGather] end inter AllGather rank[%u]", rank);

    return HCCL_SUCCESS;
}
REGISTER_TEMPLATE(TemplateType::TEMPLATE_ALL_GATHER_AHC_BROKE, AllGatherAHCBroke);
} // namespace hccl
