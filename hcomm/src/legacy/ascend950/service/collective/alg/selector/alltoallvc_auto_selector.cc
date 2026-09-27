/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "alltoallvc_auto_selector.h"
#include "selector_registry.h"
#include "coll_operator.h"

namespace Hccl {

SelectorStatus AlltoAllVCAutoSelector::SelectCcuScheduleAlgo(
    const TopoInfo& topoInfo, const CollAlgOperator& op,
    const std::map<OpType, std::vector<HcclAlgoType>>& configAlgMap, std::string& primQueueGenName) const
{
    (void)topoInfo;
    (void)op;
    (void)configAlgMap;
    (void)primQueueGenName;
    HCCL_WARNING("[Algo][AllToAllVCAutoSelector] algo is not supported yet for ccu_ms mode, reset to default.");
    return SelectorStatus::NOT_MATCH;
}

SelectorStatus AlltoAllVCAutoSelector::SelectAicpuAlgo(
    const TopoInfo& topoInfo, const CollAlgOperator& op,
    const std::map<OpType, std::vector<HcclAlgoType>>& configAlgMap, std::string& primQueueGenName) const
{
    (void)op;
    (void)configAlgMap;
    HCCL_DEBUG("[AlltoAllVCAutoSelector][%s] start, topoInfo levelNum[%u]", __func__, topoInfo.levelNum);

    // aiv 直接走打平 mesh
    primQueueGenName = "InsAlltoAllvcMesh";

    HCCL_INFO("[Algo][AlltoAllVCAutoSelector][%s] Algo match [%s]", __func__, primQueueGenName.c_str());
    return SelectorStatus::MATCH;
}

REGISTER_SELECTOR_BY_OPTYPE(OpType::ALLTOALLVC, 18, AlltoAllVCAutoSelector);
} // namespace Hccl
