/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_CCU_TRANSFORM_TASK_COMMON_H
#define HCCLV2_CCU_TRANSFORM_TASK_COMMON_H

#include <cstdint>
#include <hccl_types.h>
#include <map>
#include <memory>
#include <queue>
#include <set>
#include <unordered_map>
#include <vector>

#include "base.h"
#include "ccu_instr_info.h"
#include "ccu_microcode_v1.h"
#include "data_slice.h"
#include "data_type.h"
#include "log.h"
#include "sim_task.h"
#include "task_ccu.h"
#include "task_def.h"
#include "task_graph_generator.h"

using namespace hcomm;
namespace HcclSim {
std::string ParseMSList(const CcuRep::CcuInstr* instr);
}

#endif
