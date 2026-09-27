/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_CCU_ALL_RANK_PARAM_RECORD_H
#define HCCLV2_CCU_ALL_RANK_PARAM_RECORD_H

#include <map>
#include <set>
#include <vector>
#include <hccl/hccl_types.h>
#include "base.h"
#include "log.h"
#include "task_graph_generator.h"

namespace Hccl {

class AllRankParamRecorder {
public:
    static AllRankParamRecorder* Global();
    void InitParam();
    void Reset();
    HcclResult CheckAllPostMatch();

    HcclResult SetXn(uint32_t rankId, uint32_t dieId, uint16_t xnId, uint64_t xnValue);
    HcclResult SetGSA(uint32_t rankId, uint32_t dieId, uint16_t gsaId, uint64_t gsaValue);
    HcclResult SetCKE(uint32_t rankId, uint32_t dieId, uint16_t ckeId, uint16_t ckeValue);

    HcclResult GetXn(uint32_t rankId, uint32_t dieId, uint16_t xnId, uint64_t& xnValue);
    HcclResult GetGSA(uint32_t rankId, uint32_t dieId, uint16_t gsaId, uint64_t& gsaValue);
    HcclResult GetCKE(uint32_t rankId, uint32_t dieId, uint16_t ckeId, uint16_t& ckeValue);

    // rankId -> dieId -> 寄存器Id -> 寄存器value
    std::map<uint32_t, std::map<uint32_t, std::map<uint16_t, uint64_t>>> curXn;
    std::map<uint32_t, std::map<uint32_t, std::map<uint16_t, uint64_t>>> curGSA;
    std::map<uint32_t, std::map<uint32_t, std::map<uint16_t, uint16_t>>> curCKE;

    std::map<uint32_t, std::map<uint32_t, std::map<uint16_t, std::set<checker::TaskNode*>>>> seenPost;
};

} // namespace Hccl

#endif
