/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_CCU_CONTEXT_UTILS_H_
#define HCCLV2_CCU_CONTEXT_UTILS_H_

#include <memory>
#include <map>
#include <vector>
#include <queue>
#include "instruction.h"
#include "ins_queue.h"
#include "virtual_topo.h"
#include "ccu_ctx_arg.h"
#include "ccu_ins.h"
#include "ccu_ctx_signature.h"

namespace Hccl {
// namespace Ccu {

constexpr uint16_t LOC_CPY_LOOP_NUM = 8;
constexpr uint64_t UB_MAX_TRANS_SIZE = 256 * 1024 * 1024;         // UB单次最大传输量256*1024*1024 Byte
constexpr uint64_t MAX_LOOP_GROUP_TRANS_SIZE = 256 * 1024 * 1024; // 暂时为 256M
constexpr uint64_t MESH_2D_NUM = 2;

uint64_t CalcLGMaxTransSize();

HcclResult GenerateCcuCtxSignature(
    CcuCtxSignature& sig, CcuInstType instType, const CollAlgOperator& op,
    const std::vector<std::vector<RankId>>& tempVTopo);
// }
} // namespace Hccl
#endif // HCCLV2_CCU_CONTEXT_UTILS_H_
