/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_VM_DUMP_V3_DAG_H
#define HCCL_VM_DUMP_V3_DAG_H

#include <string>

#include "framework/task_graph_generator_v3/task_graph_generator_v3.h"
#include "hccl_types.h"

namespace HcclSim {
HcclResult
DumpV3Dag(const TaskGraphGeneratorV3::TaskGraphGeneratorV3& graph, const std::string& stage = "v3_final_graph");
} // namespace HcclSim

#endif // HCCL_VM_DUMP_V3_DAG_H
