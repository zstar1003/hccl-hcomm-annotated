/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TRANSFORM_FUNSUNCTION_H
#define TRANSFORM_FUNSUNCTION_H

#include <map>
#include <vector>
#include <memory>
#include <queue>
#include <unordered_map>

#include "type_conversion.h"
#include "coll_operator.h"
#include "coll_alg_params.h"
#include "instruction.h"
#include "checker_data_slice.h"
#include "coll_service_stub.h"

using namespace checker;

namespace Hccl {

HcclResult GenCollAlgOperator(CollAlgOperator& op, CheckerOpParam& checkerOpParam);
HcclResult GenCollAlgParams(CollAlgParams& params, CheckerOpParam& checkerOpParam, DavidAlgConfig& config);
HcclResult HcclDataSlice2CheckerDataSlice(Hccl::DataSlice& dataSlice, checker::DataSlice& checkerDataSlice);
HcclResult TransformIns2Task(const Instruction& ins, RankId rankId, QId qId);

} // namespace Hccl

#endif
