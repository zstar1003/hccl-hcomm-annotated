/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "all_gather_ahc.h"
#include "alg_template_register.h"

namespace hccl {

AllGatherAHC::AllGatherAHC(const HcclDispatcher dispatcher) : AllGatherAHCBase(dispatcher) {}

AllGatherAHC::~AllGatherAHC() {}

HcclResult AllGatherAHC::DisposeSubGroups(const u32 rank)
{
    CommAHCBaseInfo::DisposeSubGroups(rank, globalSubGroups_, level0SubGroups_, level1SubGroups_);
    return HCCL_SUCCESS;
}

HcclResult AllGatherAHC::CommAHCInfoInit()
{
    commAHCBaseInfo_.reset(new (std::nothrow) CommAHCAlignInfo(level0SubGroups_));
    CHK_SMART_PTR_NULL(commAHCBaseInfo_);
    CHK_RET(commAHCBaseInfo_->Init(AHCOpType::AHC_OP_TYPE_ALLGATHER, ahcAlgOption_));
    return HCCL_SUCCESS;
}

HcclResult AllGatherAHC::RunInterAllGather(
    u32 rank, const std::vector<LINK>& links, const std::unique_ptr<CommAHCBaseInfo>& commAHCBaseInfo)
{
    // 获取当前rank的组间rank
    HCCL_INFO("[AllGatherAHC][RunInterAllGather] begin inter AllGather rank[%u]", rank);

    u32 interRank = commAHCBaseInfo->GetInterRank(0, rank);

    // 创建执行算子实列
    std::unique_ptr<AlgTemplateBase> tempAlg;
    commAHCBaseInfo->GetInterAlgTemplateOpInstance(
        AHCOpType::AHC_OP_TYPE_ALLGATHER, tempAlg, dispatcher_, reduceAttr_, extendFlag_, ahcExtendPreparePara_);

    std::vector<std::vector<Slice>> interSlicesVector;
    std::vector<std::vector<LINK>> interLinksVector;
    std::vector<u32> logicCardList;
    CHK_RET(commAHCBaseInfo->CalcInterSlicesAndLinks(
        rank, DataUnitSize(dataType_), count_, links, interLinksVector, interSlicesVector, logicCardList));

    HCCL_DEBUG("[AllGatherAHC][RunInterAllGather] run inst rank[%u] interRank[%u]", rank, interRank);

    for (u32 i = 0; i < logicCardList.size(); i++) {
        std::vector<Slice> interSlices = interSlicesVector[i];
        std::vector<LINK> interLinks = interLinksVector[i];
        if (interLinks.size() <= 1) {
            continue;
        }
        CHK_RET(RunInstance(interRank, interLinks, interSlices, tempAlg, AHCOpType::AHC_OP_TYPE_ALLGATHER));
    }

    HCCL_DEBUG("[AllGatherAHC][RunInterAllGather] end inter AllGather rank[%u]", rank);

    return HCCL_SUCCESS;
}
REGISTER_TEMPLATE(TemplateType::TEMPLATE_ALL_GATHER_AHC, AllGatherAHC);
} // namespace hccl
