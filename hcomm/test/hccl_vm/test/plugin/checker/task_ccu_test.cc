/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <gtest/gtest.h>
#include <memory>
#include <set>

#include "ccu_instr_info.h"
#include "ccu_task_param.h"
#include "task_ccu.h"

using namespace HcclSim;

namespace {
CcuTaskParam MakeCcuParam(uint32_t startId, uint32_t cnt, uint8_t dieId, uint32_t missionId = 1)
{
    CcuTaskParam p{};
    p.dieId = dieId;
    p.instStartId = startId;
    p.instCnt = cnt;
    p.missionId = missionId;
    p.timeout = 100;
    p.argSize = 3;
    for (int i = 0; i < 3; i++) {
        p.args[i] = static_cast<uint64_t>(i) * 100;
    }
    return p;
}
} // anonymous namespace

class TaskCcuGraphTest : public testing::Test {};

TEST_F(TaskCcuGraphTest, UtConstructor_Basic)
{
    TaskStubCcuGraph graph(7);
    EXPECT_EQ(graph.GetRankId(), 7);
    EXPECT_EQ(graph.GetType(), TaskTypeStub::CCU_GRAPH);
    ASSERT_NE(graph.ccuHeadTaskNode, nullptr);
    EXPECT_EQ(graph.ccuHeadTaskNode->rankIdx, -1);
}

TEST_F(TaskCcuGraphTest, FullConstructor)
{
    std::vector<CcuTaskParam> params;
    params.push_back(MakeCcuParam(0, 10, 2));

    hcomm::CcuRep::CcuInstrInfo instrInfo{};
    instrInfo.missionStartInstrId = 0;
    instrInfo.startInstrId = 0;

    TaskStubCcuGraph graph(instrInfo, params, 3);

    EXPECT_EQ(graph.GetRankId(), 3);
    EXPECT_EQ(graph.GetType(), TaskTypeStub::CCU_GRAPH);
    EXPECT_EQ(graph.instrInfo.size(), 1);
    EXPECT_EQ(graph.ccuParams.size(), 1);
    EXPECT_EQ(graph.ccuParams[0].size(), 1);
    ASSERT_NE(graph.ccuHeadTaskNode, nullptr);
}

TEST_F(TaskCcuGraphTest, Describe)
{
    std::vector<CcuTaskParam> params;
    params.push_back(MakeCcuParam(5, 20, 1));

    hcomm::CcuRep::CcuInstrInfo instrInfo{};
    instrInfo.missionStartInstrId = 5;
    instrInfo.startInstrId = 5;

    TaskStubCcuGraph graph(instrInfo, params, 0);
    auto desc = graph.Describe();
    EXPECT_NE(desc.find("[CcuGraph]"), std::string::npos);
}

TEST_F(TaskCcuGraphTest, GetRankId)
{
    TaskStubCcuGraph graph(42);
    EXPECT_EQ(graph.GetRankId(), 42);
}

TEST_F(TaskCcuGraphTest, AddCcuParams)
{
    std::vector<CcuTaskParam> params;
    params.push_back(MakeCcuParam(0, 10, 1));

    hcomm::CcuRep::CcuInstrInfo instrInfo{};
    instrInfo.missionStartInstrId = 0;
    instrInfo.startInstrId = 0;
    TaskStubCcuGraph graph(instrInfo, params, 0);

    EXPECT_EQ(graph.ccuParams[0].size(), 1);

    CcuTaskParam p2 = MakeCcuParam(20, 30, 1);
    graph.AddCcuParams(p2);
    EXPECT_EQ(graph.ccuParams[0].size(), 2);
}

TEST_F(TaskCcuGraphTest, IsSameCcuGraph_Matches)
{
    std::vector<CcuTaskParam> params;
    CcuTaskParam p = MakeCcuParam(0, 13, 1);
    params.push_back(p);

    hcomm::CcuRep::CcuInstrInfo instrInfo{};
    instrInfo.missionStartInstrId = 0;
    instrInfo.startInstrId = 0;
    TaskStubCcuGraph graph(instrInfo, params, 0);

    EXPECT_TRUE(graph.IsSameCcuGraph(13));
}

TEST_F(TaskCcuGraphTest, IsSameCcuGraph_NotMatch)
{
    std::vector<CcuTaskParam> params;
    params.push_back(MakeCcuParam(0, 13, 1));

    hcomm::CcuRep::CcuInstrInfo instrInfo{};
    instrInfo.missionStartInstrId = 0;
    instrInfo.startInstrId = 0;
    TaskStubCcuGraph graph(instrInfo, params, 0);

    EXPECT_FALSE(graph.IsSameCcuGraph(99));
    EXPECT_FALSE(graph.IsSameCcuGraph(0));
}

TEST_F(TaskCcuGraphTest, GetMissionEndInstrId)
{
    std::vector<CcuTaskParam> params;
    params.push_back(MakeCcuParam(10, 15, 1));

    hcomm::CcuRep::CcuInstrInfo instrInfo{};
    instrInfo.startInstrId = 0;
    TaskStubCcuGraph graph(instrInfo, params, 0);

    uint32_t endId = graph.GetMissionEndInstrId(0);
    EXPECT_EQ(endId, 25);
}

TEST_F(TaskCcuGraphTest, GetSqe_ReadFirstArg)
{
    std::vector<CcuTaskParam> params;
    CcuTaskParam p1 = MakeCcuParam(0, 1, 1);
    p1.args[0] = 42;
    params.push_back(p1);

    hcomm::CcuRep::CcuInstrInfo instrInfo{};
    instrInfo.startInstrId = 0;
    TaskStubCcuGraph graph(instrInfo, params, 0);

    uint64_t val = 0;
    graph.GetSqe(0, 0, val);
    EXPECT_EQ(val, 42);
    EXPECT_EQ(graph.ccuParamIndexs[0], 0);
}

TEST_F(TaskCcuGraphTest, GetSqe_ReadLastArg_AdvancesIndex)
{
    std::vector<CcuTaskParam> params;
    CcuTaskParam p1 = MakeCcuParam(0, 1, 1);
    p1.args[12] = 888; // CCU_SQE_ARGS_LEN - 1
    params.push_back(p1);

    hcomm::CcuRep::CcuInstrInfo instrInfo{};
    instrInfo.startInstrId = 0;
    TaskStubCcuGraph graph(instrInfo, params, 0);

    uint64_t val = 0;
    graph.GetSqe(0, 12, val);
    EXPECT_EQ(val, 888);
    EXPECT_EQ(graph.ccuParamIndexs[0], 1);
}

TEST_F(TaskCcuGraphTest, GetDieId)
{
    std::vector<CcuTaskParam> params;
    params.push_back(MakeCcuParam(0, 1, 5));

    hcomm::CcuRep::CcuInstrInfo instrInfo{};
    instrInfo.startInstrId = 0;
    TaskStubCcuGraph graph(instrInfo, params, 0);

    uint32_t dieId = 0;
    graph.GetDieId(0, dieId);
    EXPECT_EQ(dieId, 5);
}

TEST_F(TaskCcuGraphTest, Destructor_NoLeak)
{
    auto* raw = new TaskStubCcuGraph(RankId(0));
    delete raw;
    SUCCEED();
}

TEST_F(TaskCcuGraphTest, CopyConstructor)
{
    std::vector<CcuTaskParam> params;
    params.push_back(MakeCcuParam(0, 1, 1));

    hcomm::CcuRep::CcuInstrInfo instrInfo{};
    instrInfo.startInstrId = 0;
    TaskStubCcuGraph original(instrInfo, params, 5);

    TaskStubCcuGraph copy(&original);
    EXPECT_EQ(copy.GetRankId(), 5);
    EXPECT_EQ(copy.GetType(), TaskTypeStub::CCU_GRAPH);
}

TEST_F(TaskCcuGraphTest, SubGraphEnd_Describe)
{
    std::vector<CcuTaskParam> params;
    params.push_back(MakeCcuParam(0, 1, 1));
    hcomm::CcuRep::CcuInstrInfo instrInfo{};
    instrInfo.startInstrId = 0;
    TaskStubCcuGraph ccuTask(instrInfo, params, RankId(0));
    TaskNode node(&ccuTask, 0, 0, 0);
    TaskStubSubGraphEnd end(&node);
    auto desc = end.Describe();
    EXPECT_NE(desc.find("end node of sub graph"), std::string::npos);
}

TEST_F(TaskCcuGraphTest, UpdateNodeForCcuGraph_CcuType)
{
    std::set<TaskNode*> simulatedNodes;
    auto* ccuTask = new TaskStubCcuGraph(1);
    TaskNode node(ccuTask, 1, 0, 0);

    TaskNode* result = UpdateNodeForCcuGraph(&node, simulatedNodes);

    EXPECT_NE(result, &node);
    EXPECT_EQ(result, ccuTask->ccuHeadTaskNode);
    EXPECT_TRUE(simulatedNodes.count(result));

    delete ccuTask;
}

TEST_F(TaskCcuGraphTest, UpdateNodeForCcuGraph_SubGraphEndType)
{
    std::set<TaskNode*> simulatedNodes;
    TaskNode targetNode(nullptr, 2, 0, 0);
    TaskStubSubGraphEnd subEnd(&targetNode);
    TaskNode node(&subEnd, 2, 0, 0);

    TaskNode* result = UpdateNodeForCcuGraph(&node, simulatedNodes);

    EXPECT_EQ(result, &targetNode);
}

TEST_F(TaskCcuGraphTest, UpdateNodeForCcuGraph_OtherType)
{
    std::set<TaskNode*> simulatedNodes;
    TaskNode node(nullptr, 0, 0, 0);

    TaskNode* result = UpdateNodeForCcuGraph(&node, simulatedNodes);

    EXPECT_EQ(result, &node);
}
