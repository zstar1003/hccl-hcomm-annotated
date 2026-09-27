/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_ALLTOALL_AUTO_SELECTOR_H
#define HCCLV2_ALLTOALL_AUTO_SELECTOR_H

#include "auto_selector_base.h"
#include "coll_alg_params.h"
#include "coll_operator.h"
#include "virtual_topo.h"

namespace Hccl {

class AlltoAllAutoSelector : public AutoSelectorBase {
private:
    SelectorStatus SelectCcuMsAlgo(
        const TopoInfo& topoInfo, const CollAlgOperator& op,
        const std::map<OpType, std::vector<HcclAlgoType>>& configAlgMap, std::string& primQueueGenName) const override;
    SelectorStatus SelectCcuScheduleAlgo(
        const TopoInfo& topoInfo, const CollAlgOperator& op,
        const std::map<OpType, std::vector<HcclAlgoType>>& configAlgMap, std::string& primQueueGenName) const override;
    SelectorStatus SelectAicpuAlgo(
        const TopoInfo& topoInfo, const CollAlgOperator& op,
        const std::map<OpType, std::vector<HcclAlgoType>>& configAlgMap, std::string& primQueueGenName) const override;
    SelectorStatus SelectAivAlgo(
        const TopoInfo& topoInfo, const CollAlgOperator& op,
        const std::map<OpType, std::vector<HcclAlgoType>>& configAlgMap, std::string& primQueueGenName) const override;

    SelectorStatus SelectCcuScheduleAlgoLevel1(
        const TopoInfo& topoInfo, const CollAlgOperator& op, std::string& primQueueGenName) const;
};

} // namespace Hccl
#endif
