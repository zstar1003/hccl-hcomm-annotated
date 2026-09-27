/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "reduce_scatter_ahc.h"
#include "alg_template_register.h"

namespace hccl {

ReduceScatterAHC::ReduceScatterAHC(const HcclDispatcher dispatcher) : ReduceScatterAHCBase(dispatcher) {}

ReduceScatterAHC::~ReduceScatterAHC() {}

HcclResult ReduceScatterAHC::DisposeSubGroups(const u32 rank)
{
    CommAHCBaseInfo::DisposeSubGroups(rank, globalSubGroups_, level0SubGroups_, level1SubGroups_);
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterAHC::CommAHCInfoInit()
{
    commAHCBaseInfo_.reset(new (std::nothrow) CommAHCAlignInfo(level0SubGroups_));
    CHK_SMART_PTR_NULL(commAHCBaseInfo_);
    CHK_RET(commAHCBaseInfo_->Init(AHCOpType::AHC_OP_TYPE_REDUCE_SCATTER, ahcAlgOption_));
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterAHC::RunInterReduceScatter(
    u32 rank, const std::vector<LINK>& links, const std::unique_ptr<CommAHCBaseInfo>& commAHCBaseInfo)
{
    // 获取当前rank的组间rank
    HCCL_INFO("[ReduceScatterAHC][RunInterReduceScatter] begin inter ReduceScatter rank[%u]", rank);

    u32 interRank = commAHCBaseInfo->GetInterRank(0, rank);

    // 创建执行算子实列
    std::unique_ptr<AlgTemplateBase> tempAlg;
    commAHCBaseInfo->GetInterAlgTemplateOpInstance(
        AHCOpType::AHC_OP_TYPE_REDUCE_SCATTER, tempAlg, dispatcher_, reduceAttr_, extendFlag_, ahcExtendPreparePara_);

    std::vector<std::vector<Slice>> interSlicesVector;
    std::vector<std::vector<LINK>> interLinksVector;
    std::vector<u32> logicCardList;
    CHK_RET(commAHCBaseInfo->CalcInterSlicesAndLinks(
        rank, DataUnitSize(dataType_), count_, links, interLinksVector, interSlicesVector, logicCardList));

    HCCL_DEBUG("[ReduceScatterAHC][RunInterReduceScatter] run inst rank[%u] interRank[%u]", rank, interRank);

    for (u32 i = 0; i < logicCardList.size(); i++) {
        std::vector<Slice> interSlices = interSlicesVector[i];
        std::vector<LINK> interLinks = interLinksVector[i];
        if (interLinks.size() <= 1) {
            continue;
        }
        HCCL_INFO("[ReduceScatterAHC][RunInterReduceScatter] rank[%u] logicCardList[%u]", rank, logicCardList[i]);
        for (u32 j = 0; j < interSlices.size(); ++j) {
            HCCL_INFO(
                "[ReduceScatterAHC][RunInterReduceScatter] interSlices[%u].size[%u] interSlices[%u].offset[%u]", j,
                interSlices[j].size, j, interSlices[j].offset);
        }
        CHK_RET(RunInstance(interRank, interLinks, interSlices, tempAlg, AHCOpType::AHC_OP_TYPE_REDUCE_SCATTER));
    }

    HCCL_DEBUG("[ReduceScatterAHC][RunInterReduceScatter] end inter ReduceScatter rank[%u]", rank);

    return HCCL_SUCCESS;
}
REGISTER_TEMPLATE(TemplateType::TEMPLATE_REDUCESCATTER_AHC, ReduceScatterAHC);
} // namespace hccl
