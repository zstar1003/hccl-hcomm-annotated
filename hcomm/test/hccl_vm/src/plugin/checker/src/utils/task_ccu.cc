/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "task_ccu.h"

#include <cstdint>
#include <map>
#include <set>

#include "sim_log.h"
#include "log.h"

namespace HcclSim {
TaskStubCcuGraph::TaskStubCcuGraph(
    hcomm::CcuRep::CcuInstrInfo& missionInstrs, const std::vector<HcclSim::CcuTaskParam>& ccuParam, RankId rank)
    : TaskStub(TaskTypeStub::CCU_GRAPH),
      rankId(rank)
{
    instrInfo.clear();
    missionInstrs.missionStartInstrId = ccuParam[0].instStartId;
    instrInfo.push_back(missionInstrs);
    des = StringFormat(
        "[CcuGraph] : [rank=%u, die=%u, mission=%u]", rankId, static_cast<uint32_t>(ccuParam[0].dieId),
        ccuParam[0].missionId);
    if (instrInfo.size() == 0) {
        HCCL_VM_WARN("Get no microcode instructions...");
    }

    ccuParams.clear();
    ccuParams.push_back(ccuParam);

    for (int i = 0; i < instrInfo.size(); i++) {
        microCodePosInQue.push_back(instrInfo[i].missionStartInstrId - instrInfo[i].startInstrId);
        startInstrIdInQue.push_back(instrInfo[i].startInstrId);
        instrQueStatus.push_back(false);
        tailNodes.push_back(nullptr);
        ccuParamIndexs.push_back(0);
    }
    ccuHeadTaskNode = new TaskNode(nullptr, -1, 0, 0);
    toDeleteTaskNode_.push_back(ccuHeadTaskNode);
}

TaskStubCcuGraph::~TaskStubCcuGraph()
{
    for (auto& task : toDeleteTask_) {
        delete task;
    }
    for (auto& node : toDeleteTaskNode_) {
        delete node;
    }
}

// [注意!!!]拷贝构造函数: 用于后面ccu子图的内存冲突改造(仅保留了内存冲突改造所需要的成员变量,
// 如果用作它途,可能成员变量须补充)
TaskStubCcuGraph::TaskStubCcuGraph(const TaskStubCcuGraph* ccuTask)
    : TaskStub(TaskTypeStub::CCU_GRAPH),
      rankId(ccuTask->rankId),
      des(ccuTask->des),
      queueNum_(ccuTask->queueNum_)
{
    SetTaskId(ccuTask->GetTaskId());
    ccuHeadTaskNode = new TaskNode(nullptr, -1, 0, 0);
    toDeleteTaskNode_.push_back(ccuHeadTaskNode);
}

void TaskStubCcuGraph::AddCcuParams(HcclSim::CcuTaskParam ccuParam)
{
    ccuParams[0].push_back(ccuParam);
    return;
}

void TaskStubCcuGraph::GetSqe(uint32_t queId, uint16_t sqeArgsId, uint64_t& argVal)
{
    argVal = ccuParams[queId][ccuParamIndexs[queId]].args[sqeArgsId];
    if (sqeArgsId == (HcclSim::CCU_SQE_ARGS_LEN - 1)) {
        ccuParamIndexs[queId] += 1;
    }
    return;
}

void TaskStubCcuGraph::GetDieId(uint32_t queId, uint32_t& dieId)
{
    dieId = ccuParams[queId][0].dieId;
    return;
}

RankId TaskStubCcuGraph::GetRankId() { return rankId; }

std::string TaskStubCcuGraph::Describe() const
{
    if (ccuParams.empty() || ccuParams[0].empty()) {
        return StringFormat("%s", (des + "sqeSize= 0").c_str());
    }

    std::string paramInfo = "sqeSize= " + std::to_string(ccuParams[0].size());
    for (auto& ccuParam : ccuParams[0]) {
        paramInfo += "; Sqe startId =" + std::to_string(ccuParam.instStartId);
    }
    return StringFormat("%s", (des + paramInfo).c_str());
}

bool TaskStubCcuGraph::IsSameCcuGraph(uint32_t startId)
{
    auto sqeSize = ccuParams[0].size();
    auto lastInstrId = ccuParams[0][sqeSize - 1].instStartId + ccuParams[0][sqeSize - 1].instCnt;
    return ccuParams[0][sqeSize - 1].instCnt == 13 && startId == lastInstrId;
}

uint32_t TaskStubCcuGraph::GetMissionEndInstrId(uint32_t queId)
{
    auto sqeSize = ccuParams[queId].size();
    return ccuParams[queId][sqeSize - 1].instStartId + ccuParams[queId][sqeSize - 1].instCnt;
}

TaskStubSubGraphEnd::TaskStubSubGraphEnd(TaskNodePtr node) : TaskStub(TaskTypeStub::SUB_GRAPH_END), subGraphNode(node)
{}

std::string TaskStubSubGraphEnd::Describe() const
{
    return StringFormat("end node of sub graph: %s", subGraphNode->task->Describe().c_str());
}

TaskNode* UpdateNodeForCcuGraph(TaskNode* node, std::set<TaskNode*>& simulatedNodes)
{
    TaskNode* retNode = node;
    if (node->task != nullptr && node->task->GetType() == TaskTypeStub::CCU_GRAPH) {
        // 首次进入子图
        TaskStubCcuGraph* curCcuTask = dynamic_cast<TaskStubCcuGraph*>(node->task);
        retNode = curCcuTask->ccuHeadTaskNode;
        simulatedNodes.insert(retNode);
    } else if (node->task != nullptr && node->task->GetType() == TaskTypeStub::SUB_GRAPH_END) {
        // 走到子图的最后一个子节点了，就回到整图
        TaskStubSubGraphEnd* subGraphEnd = dynamic_cast<TaskStubSubGraphEnd*>(node->task);
        retNode = subGraphEnd->subGraphNode;
    }
    return retNode;
}
} // namespace HcclSim
