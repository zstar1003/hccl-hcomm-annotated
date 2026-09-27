/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_ADAPTER_V2_INTERFACE_H
#define HCCL_ADAPTER_V2_INTERFACE_H

#include <unordered_map>

#include "task_def.h"

namespace HcclSim {
using TaskStubPtr = TaskStub*;
using Ori2NewNodeMap = std::map<TaskNodePtr, TaskNodePtr>;
using CcuOri2NewNodeMap = std::map<TaskStubPtr, Ori2NewNodeMap>;
extern std::unordered_map<TaskStubPtr, TaskStubPtr> g_ccuGraphTaskOri2New;

TaskNodePtr GetCcuTaskHead(TaskNodePtr node);
// 拷贝ccu子图节点
HcclResult CopyCcuSubGraphNode(
    TaskStub* originCcu, TaskStub** newCcu, std::vector<std::unordered_map<TaskStubPtr, TaskStubPtr>>& ccuGraphs,
    std::vector<CcuOri2NewNodeMap>& AllOri2NewNodeMap);
// 拷贝ccu子图连接关系
HcclResult CopyCcuSubGraphConnection(
    std::vector<std::unordered_map<TaskStubPtr, TaskStubPtr>>& ccuGraphs,
    const std::vector<CcuOri2NewNodeMap>& AllOri2NewNodeMap);
} // namespace HcclSim
#endif
