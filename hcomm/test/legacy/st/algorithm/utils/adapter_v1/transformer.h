/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_ADAPTER_V1_TRANSFORMER_H
#define HCCL_ADAPTER_V1_TRANSFORMER_H

#include <map>
#include <unordered_map>
#include <vector>

#include "checker_def.h"
#include "hccl_common.h"
#include "alg_cmd_type.h"

using namespace checker;

namespace hccl {

extern std::map<CheckerOpType, HcclCMDType> g_CheckerOpType2HcclCMDType;
extern std::map<HcclCMDType, CheckerOpType> g_HcclCMDType2CheckerOpType;
extern std::map<CheckerReduceOp, HcclReduceOp> g_CheckerReduceOp2HcclReduceOp;
extern std::map<HcclReduceOp, CheckerReduceOp> g_HcclReduceOp2CheckerReduceOp;
extern std::map<CheckerDataType, HcclDataType> g_CheckerDataType2HcclDataType;
extern std::map<HcclDataType, CheckerDataType> g_HcclDataType2CheckerDataType;
extern std::map<CheckerDevType, DevType> g_CheckerDevType2HcclDevType;
extern std::map<DevType, CheckerDevType> g_HcclDevType2CheckerDevType;

} // namespace hccl

#endif
