/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CHECKER_H
#define CHECKER_H
#include <cstdint>
#include "checker_def.h"
#include "hccl_types.h"
#include "task_check_op_semantics.h"
#include "task_def.h"

namespace HcclSim {
HcclResult GenAndCheckGraphV3();

class Checker {
public:
    Checker() = default;
    ~Checker();
    HcclResult GenAndCheckGraph(AllRankTaskQueues& allRankTaskQueues, TaskCheckOpSemantics& opSemanticsChcker);
    void CloseRankMemCheck();

private:
    void CopyTaskGraph(TaskNodePtr originNode, TaskNodePtr copyNode);
    HcclResult CopyCcuTaskGraph(TaskNodePtr originNode, TaskNodePtr copyNode, uint32_t rankNum);
    // 空实现
    vector<HcclSim::TaskStub*> toDeleteCopyTaskResource_;
    vector<TaskNodePtr> toDeleteCopyTaskNodeResource_;
    bool closeRankMemCheck_ = true;
};
} // namespace HcclSim
#endif
