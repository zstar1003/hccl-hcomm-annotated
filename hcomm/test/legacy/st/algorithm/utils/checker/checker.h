/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV1_CHECKER_H
#define HCCLV1_CHECKER_H

#include <map>
#include <vector>
#include <memory>
#include "llt_common.h"
#include "topo_meta.h"
#include "checker_def.h"

#ifdef HCCL_ALG_ANALYZER_DAVID
#include "alg_adapt_v2_interface.h"
#endif

using namespace std;
namespace checker {

struct TaskNode;
using TaskNodePtr = TaskNode*;
class TaskStub;

class Checker {
public:
    Checker();
    virtual ~Checker();
#ifndef HCCL_ALG_ANALYZER_DAVID // 编译2.0用不到
    HcclResult Check(CheckerOpParam& checkerOpParam, TopoMeta& topoMeta);
#endif

#ifdef HCCL_ALG_ANALYZER_DAVID // 编译2.0 checker
    HcclResult CheckA5Aicpu(CheckerOpParam& checkerOpParam, TopoMeta& topoMeta);
#endif
    void EnableTaskPrint();
    void EnableGraphicDump();
    void EnableGraphPrint();
    void CloseRankMemCheck();
    static void SetDumpFileName(const string& fileName);
    void setCheckerLogWarn();

private:
    void PrintTask();
    void PrintGraphRevamp(TaskNodePtr head);
    void PrintAivGraph(bool isCopy);
    void PrintAivTask();
    void CopyTaskGraph(TaskNodePtr originNode, TaskNodePtr copyNode);
    void CopyAivTaskGraph(TaskNodePtr originNode, TaskNodePtr copyNode);
    HcclResult CopyCcuTaskGraph(TaskNodePtr originNode, TaskNodePtr copyNode, uint32_t rankNum);
    HcclResult
    RankMemCheck(TaskNode& dummyStart, TaskNode& dummyStartCopy, CheckerOpParam& checkerOpParam, u32 rankNum);
    HcclResult CheckPrimGraphs(CheckerOpParam& checkerOpParam, u32 rankNum);

    vector<TaskStub*> toDeleteCopyTaskResource_;
    vector<TaskNodePtr> toDeleteCopyTaskNodeResource_;
    // aux
    bool enablePrimQuePrint_ = false;
    bool enableGraphicDump_ = false;
    bool enableGraphPrint_ = false;
    bool closeRankMemCheck_ = false;

    u64 allignSize_ = 128; // 128 bytes by default
};

} // namespace checker

#endif
