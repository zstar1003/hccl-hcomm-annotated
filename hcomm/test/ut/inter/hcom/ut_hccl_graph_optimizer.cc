/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include <mockcpp/mockcpp.hpp>
#include <stdio.h>

#define private public
#define protected public
#include "hcom_all_reduce_fusion.h"
#include "hcom_broadcast_fusion.h"
#include "hvd_all_reduce_fusion.h"
#include "hcom_reduce_fusion.h"
#include "hcom_alltoallvc_fusion.h"
#include "hcom_allgather_fusion.h"
#include "hcom_reducescatter_fusion.h"
#include "plugin_manager.h"
#include "hcom_fusion_optimizer.h"
#include "offline_build_config_parse.h"
#include "hcom_graph_optimizer.h"
#include "hvd_graph_optimizer.h"
#include "hcom_op_utils.h"
#undef private
#undef protected

#include "hccl/base.h"
#include <hccl/hccl_types.h>

#include "stream_pub.h"
#include "mem_host_pub.h"
#include "mem_device_pub.h"
#include "hccl_comm_pub.h"
#include "sal.h"
#include "hccl_impl.h"
#include "llt_hccl_stub_pub.h"
#include "externalinput.h"
#include "config.h"
#include "topoinfo_ranktableParser_pub.h"
#include "ranktable/v80_rank_table.h"
#include "hcom_op_utils.h"
#include "hcom_ops_kernel_info_store.h"
#include "external/ge/ge_api_types.h"  // ge对内options
#include "framework/common/ge_types.h" // ge对外options
#include "graph/ge_local_context.h"
#include "hcom_pub.h"
#include "hcom.h"
#include "workflow.h"

#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/node.h"
#include "graph/op_desc.h"
#include "graph/utils/tensor_utils.h"
#include "graph/ge_tensor.h"

#include "evaluator.h"
#include "model.h"
#include "cluster.h"

#include <iostream>
#include <fstream>
#include "graph/ge_context.h"

using namespace std;
using namespace hccl;

class HcomGraphOptimizerTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "\033[36m--HcomGraphOptimizerTest SetUP--\033[0m" << std::endl; }
    static void TearDownTestCase() { std::cout << "\033[36m--HcomGraphOptimizerTest TearDown--\033[0m" << std::endl; }
    // Some expensive resource shared by all tests.
    virtual void SetUp()
    {
        s32 portNum = 7;
        MOCKER(hrtGetHccsPortNum).stubs().with(mockcpp::any(), outBound(portNum)).will(returnValue(HCCL_SUCCESS));
        MOCKER_CPP(&HcomGraphOptimizer::SetSuperKernelScopeAttr)
            .stubs()
            .with(mockcpp::any())
            .will(returnValue(HCCL_SUCCESS));
        std::cout << "A Test SetUP" << std::endl;
    }
    virtual void TearDown() { std::cout << "A Test TearDown" << std::endl; }
};

TEST_F(HcomGraphOptimizerTest, ut_optimize_graphprepare_SetHcomOpParallelLabel)
{
    ge ::Status ret;
    ge ::Status ge_ret;
    bool bRet;
    std::map<std::string, std::string> options;
    ge::ComputeGraph graph("test_graph");
    std::map<string, GraphOptimizerPtr> graphOptimizers;
    ge::OpDescPtr opDescPtr = nullptr;

    // 未设置 rank table：失败
    options.insert(pair<string, string>(ge::OPTION_EXEC_RANK_TABLE_FILE, "rank_table.json"));
    // 实验室场景 设置 rank id
    options.insert(pair<string, string>(ge::OPTION_EXEC_DEPLOY_MODE, "0"));
    options.insert(pair<string, string>(ge::OPTION_EXEC_RANK_ID, "1"));
    options.insert(pair<string, string>(ge::OPTION_EXEC_PROFILING_MODE, "0"));

    MOCKER(HcomInitByFile).stubs().will(returnValue(HCCL_SUCCESS));
    // 实验室场景 hcom_init成功：成功
    ret = Initialize(options);
    EXPECT_EQ(ret, ge::SUCCESS);

    GetGraphOptimizerObjs(graphOptimizers);

    ge_ret = graphOptimizers.at(HCCL_GRAPH_OPTIMIZER_NAME)->Initialize(options, nullptr);
    EXPECT_EQ(ge_ret, ge::SUCCESS);
    ge::OpDesc op;
    graph.AddNode(opDescPtr);
    EXPECT_EQ(ge_ret, ge::SUCCESS);

    HcomGraphOptimizer optimizer;
    std::string groupLabel = "hcom_op";
    for (auto nodePtr : graph.GetAllNodes()) {
        HcclResult hRet = optimizer.SetHcomOpParallelLabel(*nodePtr, groupLabel);
        EXPECT_EQ(hRet, HCCL_SUCCESS);
    }

    bRet = ge::AttrUtils::SetInt(opDescPtr, "tag", 1);
    EXPECT_EQ(bRet, true);

    for (auto nodePtr : graph.GetAllNodes()) {
        HcclResult hRet = optimizer.SetHcomOpParallelLabel(*nodePtr, groupLabel);
        EXPECT_EQ(hRet, HCCL_SUCCESS);
    }

    GlobalMockObject::verify();
}

TEST_F(HcomGraphOptimizerTest, ut_Initialize_to_Finalize)
{
    ge ::Status ret;
    std::map<std::string, std::string> options;

    // 未设置 rank table：失败
    options.insert(pair<string, string>(ge::OPTION_EXEC_RANK_TABLE_FILE, "rank_table.json"));
    // 实验室场景 设置 rank id
    options.insert(pair<string, string>(ge::OPTION_EXEC_DEPLOY_MODE, "0"));
    options.insert(pair<string, string>(ge::OPTION_EXEC_RANK_ID, "1"));
    options.insert(pair<string, string>(ge::OPTION_EXEC_PROFILING_MODE, "0"));
    MOCKER(HcomInitByFile).stubs().will(returnValue(HCCL_SUCCESS));
    // 实验室场景 hcom_init成功：成功
    ret = Initialize(options);
    EXPECT_EQ(ret, ge::SUCCESS);

    ge ::Status ge_ret;
    bool bRet;

    std::map<string, GraphOptimizerPtr> graphOptimizers;
    GetGraphOptimizerObjs(graphOptimizers);

    ge_ret = graphOptimizers.at(HCCL_GRAPH_OPTIMIZER_NAME)->Initialize(options, nullptr);
    EXPECT_EQ(ge_ret, ge::SUCCESS);

    MOCKER_CPP(&HcomAllReduceFusion::GetFusionOps).stubs().will(returnValue(HCCL_SUCCESS));

    ge::GraphOptimizerAttribute attrs;
    ge_ret = graphOptimizers.at(HCCL_GRAPH_OPTIMIZER_NAME)->GetAttributes(attrs);
    EXPECT_EQ(ge_ret, ge::SUCCESS);
    EXPECT_EQ(attrs.engineName, HCCL_OPS_ENGIN);
    EXPECT_EQ(attrs.scope, ge::UNIT);

    // ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test_graph");
    ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test_graph");
    ge::OpDesc op;
    const string type = HCCL_KERNEL_OP_TYPE_ALLREDUCE;
    op.SetType(type);

    auto descPtr0 = std::make_shared<ge::OpDesc>("Allreduce0", HCCL_KERNEL_OP_TYPE_ALLREDUCE);
    EXPECT_EQ(descPtr0->GetType(), HCCL_KERNEL_OP_TYPE_ALLREDUCE);
    auto addedNodePtr0 = graph->AddNode(descPtr0);
    EXPECT_NE(addedNodePtr0, nullptr);

    auto descPtr1 = std::make_shared<ge::OpDesc>("Allreduce1", HCCL_KERNEL_OP_TYPE_ALLREDUCE);
    EXPECT_EQ(descPtr1->GetType(), HCCL_KERNEL_OP_TYPE_ALLREDUCE);
    auto addedNodePtr1 = graph->AddNode(descPtr1);
    EXPECT_NE(addedNodePtr1, nullptr);

    ge::ComputeGraphPtr subgraph = std::make_shared<ge::ComputeGraph>("test_subgraph");
    auto descPtr2 = std::make_shared<ge::OpDesc>("Allreduce2", HCCL_KERNEL_OP_TYPE_ALLREDUCE);
    EXPECT_EQ(descPtr2->GetType(), HCCL_KERNEL_OP_TYPE_ALLREDUCE);
    auto addedNodePtr2 = subgraph->AddNode(descPtr2);
    EXPECT_NE(addedNodePtr2, nullptr);

    auto descPtr3 = std::make_shared<ge::OpDesc>("Allreduce3", HCCL_KERNEL_OP_TYPE_ALLREDUCE);
    EXPECT_EQ(descPtr3->GetType(), HCCL_KERNEL_OP_TYPE_ALLREDUCE);
    auto addedNodePtr3 = subgraph->AddNode(descPtr3);
    EXPECT_NE(addedNodePtr3, nullptr);

    graph->AddSubGraph(subgraph);

    HCCL_INFO("start OptimizeOriginalGraph");
    ge_ret = graphOptimizers.at(HCCL_GRAPH_OPTIMIZER_NAME)->OptimizeOriginalGraph(*graph);
    HCCL_INFO("end OptimizeOriginalGraph");
    EXPECT_EQ(ge_ret, ge::SUCCESS);
    u64 streamNumber = 4;
    MOCKER(HcomGetWorkspaceSubStreamNum)
        .stubs()
        .with(mockcpp::any(), outBound(streamNumber))
        .will(returnValue(HCCL_SUCCESS));
    u32 rankSize = 8;
    MOCKER(HcomGetRankSize).stubs().with(mockcpp::any(), outBound(&rankSize)).will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&HcomGraphOptimizer::GetTaskNumAndCheckForceUnknown).stubs().will(returnValue(HCCL_SUCCESS));

    ge_ret = graphOptimizers.at(HCCL_GRAPH_OPTIMIZER_NAME)->OptimizeFusedGraph(*graph);
    EXPECT_EQ(ge_ret, ge::SUCCESS);
    ge_ret = graphOptimizers.at(HCCL_GRAPH_OPTIMIZER_NAME)->OptimizeWholeGraph(*graph);
    EXPECT_EQ(ge_ret, ge::SUCCESS);

    ge_ret = graphOptimizers.at(HCCL_GRAPH_OPTIMIZER_NAME)->Finalize();
    EXPECT_EQ(ge_ret, ge::SUCCESS);

    MOCKER(HcomDestroy).expects(atMost(1)).will(returnValue(HCCL_SUCCESS));
    ret = Finalize();
    EXPECT_EQ(ret, ge::SUCCESS);
    GlobalMockObject::verify();
}

TEST_F(HcomGraphOptimizerTest, ut_Initialize_to_Finalize_51)
{
    ge ::Status ret;
    std::map<std::string, std::string> options;

    // 未设置 rank table：失败
    options.insert(pair<string, string>(ge::OPTION_EXEC_RANK_TABLE_FILE, "rank_table.json"));
    // 实验室场景 设置 rank id
    options.insert(pair<string, string>(ge::OPTION_EXEC_DEPLOY_MODE, "0"));
    options.insert(pair<string, string>(ge::OPTION_EXEC_RANK_ID, "1"));
    options.insert(pair<string, string>(ge::OPTION_EXEC_PROFILING_MODE, "0"));
    u32 numHccsLink = 0;
    MOCKER(HcomInitByFile).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(HcomGetHccsLinkNum).stubs().with(mockcpp::any(), outBound(numHccsLink)).will(returnValue(HCCL_SUCCESS));

    u32 rankSize = 2;
    MOCKER(HcomGetRankSize).stubs().with(mockcpp::any(), outBoundP(&rankSize)).will(returnValue(HCCL_SUCCESS));

    // 实验室场景 hcom_init成功：成功
    ret = Initialize(options);
    EXPECT_EQ(ret, ge::SUCCESS);

    ge ::Status ge_ret;
    bool bRet;

    std::map<string, GraphOptimizerPtr> graphOptimizers;
    GetGraphOptimizerObjs(graphOptimizers);

    ge_ret = graphOptimizers.at(HCCL_GRAPH_OPTIMIZER_NAME)->Initialize(options, nullptr);
    EXPECT_EQ(ge_ret, ge::SUCCESS);

    MOCKER_CPP(&HcomAllReduceFusion::GetFusionOps).stubs().will(returnValue(HCCL_SUCCESS));

    ge::GraphOptimizerAttribute attrs;
    ge_ret = graphOptimizers.at(HCCL_GRAPH_OPTIMIZER_NAME)->GetAttributes(attrs);
    EXPECT_EQ(ge_ret, ge::SUCCESS);
    EXPECT_EQ(attrs.engineName, HCCL_OPS_ENGIN);
    EXPECT_EQ(attrs.scope, ge::UNIT);

    // ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test_graph");
    ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test_graph");
    ge::OpDesc op;
    const string type = HCCL_KERNEL_OP_TYPE_ALLREDUCE;
    op.SetType(type);

    auto descPtr0 = std::make_shared<ge::OpDesc>("Allreduce0", HCCL_KERNEL_OP_TYPE_ALLREDUCE);
    EXPECT_EQ(descPtr0->GetType(), HCCL_KERNEL_OP_TYPE_ALLREDUCE);
    auto addedNodePtr0 = graph->AddNode(descPtr0);
    EXPECT_NE(addedNodePtr0, nullptr);

    auto descPtr1 = std::make_shared<ge::OpDesc>("Allreduce1", HCCL_KERNEL_OP_TYPE_ALLREDUCE);
    EXPECT_EQ(descPtr1->GetType(), HCCL_KERNEL_OP_TYPE_ALLREDUCE);
    auto addedNodePtr1 = graph->AddNode(descPtr1);
    EXPECT_NE(addedNodePtr1, nullptr);

    ge::ComputeGraphPtr subgraph = std::make_shared<ge::ComputeGraph>("test_subgraph");
    auto descPtr2 = std::make_shared<ge::OpDesc>("Allreduce2", HCCL_KERNEL_OP_TYPE_ALLREDUCE);
    EXPECT_EQ(descPtr2->GetType(), HCCL_KERNEL_OP_TYPE_ALLREDUCE);
    auto addedNodePtr2 = subgraph->AddNode(descPtr2);
    EXPECT_NE(addedNodePtr2, nullptr);

    auto descPtr3 = std::make_shared<ge::OpDesc>("Allreduce3", HCCL_KERNEL_OP_TYPE_ALLREDUCE);
    EXPECT_EQ(descPtr3->GetType(), HCCL_KERNEL_OP_TYPE_ALLREDUCE);
    auto addedNodePtr3 = subgraph->AddNode(descPtr3);
    EXPECT_NE(addedNodePtr3, nullptr);

    graph->AddSubGraph(subgraph);

    HCCL_INFO("start OptimizeOriginalGraph");
    ge_ret = graphOptimizers.at(HCCL_GRAPH_OPTIMIZER_NAME)->OptimizeOriginalGraph(*graph);
    HCCL_INFO("end OptimizeOriginalGraph");
    EXPECT_EQ(ge_ret, ge::SUCCESS);
    u64 streamNumber = 4;
    MOCKER(HcomGetWorkspaceSubStreamNum)
        .stubs()
        .with(mockcpp::any(), outBound(streamNumber))
        .will(returnValue(HCCL_SUCCESS));

    DevType type610 = DevType::DEV_TYPE_310P1;
    MOCKER(GetOffDeviceTypeWithoutDev).stubs().with(outBound(type610)).will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&HcomGraphOptimizer::GetTaskNumAndCheckForceUnknown).stubs().will(returnValue(HCCL_SUCCESS));

    ge_ret = graphOptimizers.at(HCCL_GRAPH_OPTIMIZER_NAME)->OptimizeFusedGraph(*graph);
    EXPECT_EQ(ge_ret, ge::SUCCESS);
    ge_ret = graphOptimizers.at(HCCL_GRAPH_OPTIMIZER_NAME)->OptimizeWholeGraph(*graph);
    EXPECT_EQ(ge_ret, ge::SUCCESS);

    ge_ret = graphOptimizers.at(HCCL_GRAPH_OPTIMIZER_NAME)->Finalize();
    EXPECT_EQ(ge_ret, ge::SUCCESS);

    MOCKER(HcomDestroy).expects(atMost(1)).will(returnValue(HCCL_SUCCESS));
    ret = Finalize();
    EXPECT_EQ(ret, ge::SUCCESS);
    GlobalMockObject::verify();
}

HcclResult GetOffDeviceTypeWithoutDevMockA2(DevType& devType)
{
    devType = DevType::DEV_TYPE_910B;
    HCCL_DEBUG("[offline] Get devtype[%u]....", devType);
    return HCCL_SUCCESS;
}

TEST_F(HcomGraphOptimizerTest, ut_OptimizeFusedGraph_GetDeterministic)
{
    MOCKER(GetOffDeviceTypeWithoutDev).stubs().will(invoke(GetOffDeviceTypeWithoutDevMockA2));
    setenv("HCCL_DETERMINISTIC", "STRICT", 1);

    u8 deterministic = 0;
    HcclResult ret = GetDeterministic(deterministic);

    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(deterministic, 2);

    unsetenv("HCCL_DETERMINISTIC");
    GlobalMockObject::verify();
}

HcclResult stub_GetAllInputsTensorMemSize(const ge::OpDescPtr& opDescPtr, uint64_t& tensorSize)
{
    tensorSize = 2048 * 1024 * 1024;
    return HCCL_SUCCESS;
}

HcclResult stub_HcomGetCCLBufferAvailableSize(u64& size)
{
    size = 1024 * 1024;
    return HCCL_SUCCESS;
}

TEST_F(HcomGraphOptimizerTest, ut_GetFusionOpInfo)
{
    HcclResult ret;
    ge::ComputeGraph graph("test_graph");
    HcomAllReduceFusion fusionHcomAllReduceOp;
    FusionInfos fusionOps;
    std::vector<ge::NodePtr> ops(7);

    std::string tempStrReduction = "sum";
    ge::AttrUtils::SetStr(ops[1]->GetOpDesc(), "reduction", tempStrReduction);

    ge::AttrUtils::HasAttr(ops[0]->GetOpDesc(), "DUMMY_SET_FALSE_GROUP");
    ret = fusionHcomAllReduceOp.GetFusionOpInfo(ops[0], fusionOps);
    EXPECT_EQ(ret, HCCL_E_PARA);

    ge::AttrUtils::HasAttr(ops[1]->GetOpDesc(), "DUMMY_SET_TRUE_GROUP");
    std::string tempStr = HCCL_WORLD_GROUP;
    ge::AttrUtils::SetStr(ops[1]->GetOpDesc(), "group", tempStr);
    ret = fusionHcomAllReduceOp.GetFusionOpInfo(ops[1], fusionOps);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(fusionOps.size(), 1);
    EXPECT_EQ(fusionOps["hccl_world_group"].size(), 0);
    // EXPECT_EQ(fusionOps["hccl_world_group"][HCOMALLREDUCE_ATTR_FUSION_ID_DEFAULT].size(), 1);

    tempStr = "test_group";
    ge::AttrUtils::SetStr(ops[2]->GetOpDesc(), "group", tempStr);
    ret = fusionHcomAllReduceOp.GetFusionOpInfo(ops[2], fusionOps);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(fusionOps.size(), 3);
    EXPECT_EQ(fusionOps["test_group"].size(), 0);
    // EXPECT_EQ(fusionOps["test_group"][HCOMALLREDUCE_ATTR_FUSION_ID_DEFAULT].size(), 1);

    tempStr = "test_group";
    ge::AttrUtils::SetStr(ops[3]->GetOpDesc(), "group", tempStr);
    ge::AttrUtils::SetInt(ops[3]->GetOpDesc(), "fusion", HCOM_ATTR_FUSION_BY_FUSION_ID);
    ge::AttrUtils::SetInt(ops[3]->GetOpDesc(), "fusion_id", 0);
    ret = fusionHcomAllReduceOp.GetFusionOpInfo(ops[3], fusionOps);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(fusionOps.size(), 5);
    EXPECT_EQ(fusionOps["test_group"].size(), 0);
    // EXPECT_EQ(fusionOps["test_group"][0].size(), 1);

    tempStr = HCCL_WORLD_GROUP;
    ge::AttrUtils::SetStr(ops[4]->GetOpDesc(), "group", tempStr);
    ge::AttrUtils::SetInt(ops[4]->GetOpDesc(), "fusion", 0);
    ge::AttrUtils::SetInt(ops[4]->GetOpDesc(), "fusion_id", HCOM_ATTR_FUSION_ID_DEFAULT);
    ret = fusionHcomAllReduceOp.GetFusionOpInfo(ops[4], fusionOps);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(fusionOps.size(), 5);
    EXPECT_EQ(fusionOps["hccl_world_group"].size(), 0);
    // EXPECT_EQ(fusionOps["hccl_world_group"][HCOMALLREDUCE_ATTR_FUSION_ID_DEFAULT].size(), 1);

    tempStr = HCCL_WORLD_GROUP;
    ge::AttrUtils::SetStr(ops[5]->GetOpDesc(), "group", tempStr);
    ge::AttrUtils::SetInt(ops[5]->GetOpDesc(), "fusion", 3);
    ge::AttrUtils::SetInt(ops[5]->GetOpDesc(), "fusion_id", HCOM_ATTR_FUSION_ID_DEFAULT);
    ret = fusionHcomAllReduceOp.GetFusionOpInfo(ops[5], fusionOps);
    EXPECT_EQ(ret, HCCL_E_PARA);
    EXPECT_EQ(fusionOps.size(), 5);
    EXPECT_EQ(fusionOps["hccl_world_group"].size(), 0);
    // EXPECT_EQ(fusionOps["hccl_world_group"][HCOMALLREDUCE_ATTR_FUSION_ID_DEFAULT].size(), 1);

    tempStr = "test_group";
    ge::AttrUtils::SetStr(ops[6]->GetOpDesc(), "group", tempStr);
    ge::AttrUtils::SetInt(ops[6]->GetOpDesc(), "fusion", HCOM_ATTR_FUSION_BY_FUSION_ID);
    ge::AttrUtils::SetInt(ops[6]->GetOpDesc(), "fusion_id", 0);
    ret = fusionHcomAllReduceOp.GetFusionOpInfo(ops[6], fusionOps);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ge::AttrUtils::HasAttr(ops[3]->GetOpDesc(), "DUMMY_SET_FALSE_GROUP");

    // MOCKER_CPP(&ge::ComputeGraph::GetDirectNode)
    // .stubs()
    // .with(mockcpp::any())
    // .will(returnValue(ops));
    // ret = fusionHcomAllReduceOp.GetFusionOps(graph, fusionOps);
    // ret = fusionHcomAllReduceOp.GetFusionOpInfo(nodePtr, fusionOps);
    // EXPECT_EQ(ret, HCCL_SUCCESS);
    // EXPECT_EQ(fusionOps.size(), 1);
    // GlobalMockObject::verify();

    MOCKER(HcomGetCCLBufferAvailableSize).stubs().with(mockcpp::any()).will(invoke(stub_HcomGetCCLBufferAvailableSize));

    MOCKER(&HcomOpUtils::GetAllInputsTensorMemSize).stubs().will(invoke(stub_GetAllInputsTensorMemSize));

    FusionInfos fusionOpsTemp;
    ret = fusionHcomAllReduceOp.GetFusionOpsSlices(fusionOps, fusionOpsTemp);
}

TEST_F(HcomGraphOptimizerTest, ut_FuseOps)
{
    HcclResult ret;
    bool bRet;
    ge::ComputeGraph graph("test_graph");
    ge::OpDescPtr nodeGroup = nullptr;
    ge::OpDescPtr nodeFusionId = 0;
    HcomAllReduceFusion fusionHcomAllReduceOp;
    std::map<std::string, std::vector<ge::NodePtr>> fusionOps;
    u32 segmentNum = 0;
    std::vector<u32> segmentIndex;

    std::vector<ge::NodePtr> ops(3);
    std::vector<ge::NodePtr> nodeVec_0;
    std::string group = HCCL_WORLD_GROUP;
    int64_t fusionid = HCOM_ATTR_FUSION_ID_DEFAULT;
    bRet = ge::AttrUtils::SetStr(nodeGroup, "group", group);
    EXPECT_EQ(bRet, true);
    bRet = ge::AttrUtils::SetInt(nodeFusionId, "fusion_id", fusionid);
    EXPECT_EQ(bRet, true);
    nodeVec_0.push_back(ops[0]);
    nodeVec_0.push_back(ops[1]);
    nodeVec_0.push_back(ops[2]);
    fusionOps["hccl_world_group"] = nodeVec_0;

    std::vector<u32> segments;
    segments.push_back(1);
    segments.push_back(2);
    MOCKER(HcomGetSplitStrategy)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), outBound(segments))
        .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&HcomAllReduceFusion::RunFusionOps).stubs().will(returnValue(HCCL_SUCCESS));
    ret = fusionHcomAllReduceOp.FuseOps(graph, fusionOps["hccl_world_group"]);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
}

TEST_F(HcomGraphOptimizerTest, ut_GetFusionStrategy)
{
    HcclResult ret;
    bool bRet;
    ge::ComputeGraph graph("test_graph");
    HcomAllReduceFusion fusionHcomAllReduceOp;
    ge::OpDescPtr nodeGroup = nullptr;
    ge::OpDescPtr nodeFusionId = 0;
    std::vector<ge::NodePtr> fusionOps(3);
    u32 segmentNum = 0;
    std::vector<u32> segmentIndex;

    u32 segment_num = 2; // fused to 2 allreduce node
    std::vector<u32> segments;
    bool configured = false;
    segments.push_back(1);
    segments.push_back(2);
    MOCKER(HcomGetSplitStrategy)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), outBound(segments), outBound(configured))
        .will(returnValue(HCCL_SUCCESS));
    std::string group = HCCL_WORLD_GROUP;
    int64_t fusionid = HCOM_ATTR_FUSION_ID_DEFAULT;
    ge::AttrUtils::HasAttr(nodeGroup, "DUMMY_SET_TRUE_GROUP");
    bRet = ge::AttrUtils::SetStr(nodeGroup, "group", group);
    EXPECT_EQ(bRet, true);
    bRet = ge::AttrUtils::SetInt(nodeFusionId, "fusion_id", fusionid);
    EXPECT_EQ(bRet, true);
    // FusionInfo option("hccl_world_group", HCOMALLREDUCE_ATTR_FUSION_ID_DEFAULT);
    ret = fusionHcomAllReduceOp.GetFusionStrategy(graph, fusionOps, segmentNum, segmentIndex);
    EXPECT_EQ(segmentNum, segment_num);
    EXPECT_EQ(segmentIndex[0], segments[0]);
    EXPECT_EQ(segmentIndex[1], segments[1]);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
}

TEST_F(HcomGraphOptimizerTest, ut_GetNodeUnknownShapeInfo_known)
{
    HcclResult ret;
    ge::ComputeGraph graph("test_graph");
    HcomAllReduceFusion fusionHcomAllReduceOp;
    std::vector<ge::NodePtr> fusionOps(3);
    u32 segmentNum = 0;
    std::vector<u32> segmentIndex;
    bool bUnknownShapeNodeStatus = false;

    ret = fusionHcomAllReduceOp.GetNodeUnknownShapeInfo(fusionOps[0], bUnknownShapeNodeStatus);
    EXPECT_EQ(bUnknownShapeNodeStatus, false);
    EXPECT_EQ(fusionHcomAllReduceOp.bHasUnknownShapeNodeGraph_, false);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = fusionHcomAllReduceOp.GetNodeUnknownShapeInfo(fusionOps[0], bUnknownShapeNodeStatus);
    EXPECT_EQ(bUnknownShapeNodeStatus, false);
    EXPECT_EQ(fusionHcomAllReduceOp.bHasUnknownShapeNodeGraph_, false);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    GlobalMockObject::verify();
}

TEST_F(HcomGraphOptimizerTest, ut_GetNodeUnknownShapeInfo_unknown)
{
    HcclResult ret;
    ge::ComputeGraph graph("test_graph");
    HcomAllReduceFusion fusionHcomAllReduceOp;
    std::vector<ge::NodePtr> fusionOps(3);
    u32 segmentNum = 0;
    std::vector<u32> segmentIndex;
    bool is_unknown = true;
    bool bUnknownShapeNodeStatus = false;

    MOCKER(&ge::NodeUtils::GetNodeUnknownShapeStatus)
        .stubs()
        .with(mockcpp::any(), outBound(is_unknown))
        .will(returnValue(ge::GRAPH_SUCCESS));

    ret = fusionHcomAllReduceOp.GetNodeUnknownShapeInfo(fusionOps[0], bUnknownShapeNodeStatus);
    EXPECT_EQ(bUnknownShapeNodeStatus, true);
    EXPECT_EQ(fusionHcomAllReduceOp.bHasUnknownShapeNodeGraph_, true);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = fusionHcomAllReduceOp.GetNodeUnknownShapeInfo(fusionOps[0], bUnknownShapeNodeStatus);
    EXPECT_EQ(bUnknownShapeNodeStatus, true);
    EXPECT_EQ(fusionHcomAllReduceOp.bHasUnknownShapeNodeGraph_, true);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    GlobalMockObject::verify();
}

class NodeTest : public ge::Node {
public:
    NodeTest() { ; };
    ~NodeTest() { ; };
};

TEST_F(HcomGraphOptimizerTest, ut_RunFusionOps)
{
    HcclResult ret;
    ge::ComputeGraph graph("test_graph");
    HcomAllReduceFusion fusionHcomAllReduceOp;
    std::vector<ge::NodePtr> fusionOps(3);
    std::vector<ge::NodePtr> InOps(3);
    std::vector<ge::NodePtr> OutOps(3);

    fusionOps[0] = std::make_shared<NodeTest>();
    fusionOps[0]->Init();
    fusionOps[1] = std::make_shared<NodeTest>();
    fusionOps[1]->Init();
    fusionOps[2] = std::make_shared<NodeTest>();
    fusionOps[2]->Init();

    InOps[0] = std::make_shared<NodeTest>();
    InOps[0]->Init();
    InOps[1] = std::make_shared<NodeTest>();
    InOps[1]->Init();
    InOps[2] = std::make_shared<NodeTest>();
    InOps[2]->Init();

    OutOps[0] = std::make_shared<NodeTest>();
    OutOps[0]->Init();
    OutOps[1] = std::make_shared<NodeTest>();
    OutOps[1]->Init();
    OutOps[2] = std::make_shared<NodeTest>();
    OutOps[2]->Init();

    InOps[0]->GetOutControlAnchor()->LinkTo(fusionOps[0]->GetInControlAnchor());
    InOps[0]->GetOutDataAnchor(0)->LinkTo(fusionOps[0]->GetInDataAnchor(0));
    // InOps[0]->GetOutDataAnchor(0)->LinkTo(fusionOps[0]->GetInControlAnchor());
    // InOps[1]->GetOutControlAnchor()->LinkTo(fusionOps[1]->GetInControlAnchor());
    // InOps[1]->GetOutDataAnchor(0)->LinkTo(fusionOps[1]->GetInDataAnchor(0));
    InOps[1]->GetOutDataAnchor(0)->LinkTo(fusionOps[1]->GetInControlAnchor());
    // InOps[2]->GetOutControlAnchor()->LinkTo(fusionOps[2]->GetInControlAnchor());
    InOps[2]->GetOutDataAnchor(0)->LinkTo(fusionOps[2]->GetInDataAnchor(0));
    // InOps[2]->GetOutDataAnchor(0)->LinkTo(fusionOps[2]->GetInControlAnchor());

    fusionOps[0]->GetOutControlAnchor()->LinkTo(OutOps[0]->GetInControlAnchor());
    fusionOps[0]->GetOutDataAnchor(0)->LinkTo(OutOps[0]->GetInDataAnchor(0));
    // fusionOps[0]->GetOutDataAnchor(0)->LinkTo(OutOps[0]->GetInControlAnchor());
    // fusionOps[1]->GetOutControlAnchor()->LinkTo(OutOps[1]->GetInControlAnchor());
    fusionOps[1]->GetOutDataAnchor(0)->LinkTo(OutOps[1]->GetInDataAnchor(0));
    fusionOps[1]->GetOutDataAnchor(0)->LinkTo(OutOps[1]->GetInControlAnchor());
    // fusionOps[2]->GetOutControlAnchor()->LinkTo(OutOps[2]->GetInControlAnchor());
    fusionOps[2]->GetOutDataAnchor(0)->LinkTo(OutOps[2]->GetInDataAnchor(0));
    // fusionOps[2]->GetOutDataAnchor(0)->LinkTo(OutOps[2]->GetInControlAnchor());

    u32 segmentNum = 2;
    std::vector<u32> segmentIndex;
    segmentIndex.push_back(0);
    segmentIndex.push_back(1);

    ret = fusionHcomAllReduceOp.RunFusionOps(graph, fusionOps, segmentNum, segmentIndex);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(HcomGraphOptimizerTest, ut_RunFusionOps_have_duplication)
{
    HcclResult ret;
    ge::ComputeGraph graph("test_graph");
    HcomAllReduceFusion fusionHcomAllReduceOp;
    std::vector<ge::NodePtr> fusionOps(3);
    std::vector<ge::NodePtr> InOps(3);
    std::vector<ge::NodePtr> OutOps(3);

    fusionOps[0] = std::make_shared<NodeTest>();
    fusionOps[0]->Init();
    fusionOps[1] = std::make_shared<NodeTest>();
    fusionOps[1]->Init();
    fusionOps[2] = std::make_shared<NodeTest>();
    fusionOps[2]->Init();

    InOps[0] = std::make_shared<NodeTest>();
    InOps[0]->Init();
    InOps[1] = std::make_shared<NodeTest>();
    InOps[1]->Init();
    InOps[2] = std::make_shared<NodeTest>();
    InOps[2]->Init();

    OutOps[0] = std::make_shared<NodeTest>();
    OutOps[0]->Init();
    OutOps[1] = std::make_shared<NodeTest>();
    OutOps[1]->Init();
    OutOps[2] = std::make_shared<NodeTest>();
    OutOps[2]->Init();

    InOps[0]->GetOutControlAnchor()->LinkTo(fusionOps[0]->GetInControlAnchor());
    // InOps_0 link to fusionOps_0
    InOps[0]->GetOutDataAnchor(0)->LinkTo(fusionOps[0]->GetInDataAnchor(0));
    // InOps_0 link to fusionOps_1
    InOps[0]->GetOutDataAnchor(0)->LinkTo(fusionOps[1]->GetInDataAnchor(0));
    InOps[2]->GetOutDataAnchor(0)->LinkTo(fusionOps[2]->GetInDataAnchor(0));

    fusionOps[0]->GetOutControlAnchor()->LinkTo(OutOps[0]->GetInControlAnchor());
    fusionOps[0]->GetOutDataAnchor(0)->LinkTo(OutOps[0]->GetInDataAnchor(0));
    fusionOps[1]->GetOutDataAnchor(0)->LinkTo(OutOps[1]->GetInDataAnchor(0));
    fusionOps[1]->GetOutDataAnchor(0)->LinkTo(OutOps[1]->GetInControlAnchor());
    fusionOps[2]->GetOutDataAnchor(0)->LinkTo(OutOps[2]->GetInDataAnchor(0));

    u32 segmentNum = 2;
    std::vector<u32> segmentIndex;
    segmentIndex.push_back(0);
    segmentIndex.push_back(1);

    ret = fusionHcomAllReduceOp.RunFusionOps(graph, fusionOps, segmentNum, segmentIndex);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(HcomGraphOptimizerTest, ut_RunFusionOps_bcast)
{
    HcclResult ret;
    ge::ComputeGraph graph("test_graph");
    HcomBroadcastFusion fusionHcomBroadcastOp;
    std::vector<ge::NodePtr> fusionOps(3);
    std::vector<ge::NodePtr> InOps(3);
    std::vector<ge::NodePtr> OutOps(3);

    fusionOps[0] = std::make_shared<NodeTest>();
    fusionOps[0]->Init();
    fusionOps[1] = std::make_shared<NodeTest>();
    fusionOps[1]->Init();
    fusionOps[2] = std::make_shared<NodeTest>();
    fusionOps[2]->Init();

    InOps[0] = std::make_shared<NodeTest>();
    InOps[0]->Init();
    InOps[1] = std::make_shared<NodeTest>();
    InOps[1]->Init();
    InOps[2] = std::make_shared<NodeTest>();
    InOps[2]->Init();

    OutOps[0] = std::make_shared<NodeTest>();
    OutOps[0]->Init();
    OutOps[1] = std::make_shared<NodeTest>();
    OutOps[1]->Init();
    OutOps[2] = std::make_shared<NodeTest>();
    OutOps[2]->Init();

    InOps[0]->GetOutControlAnchor()->LinkTo(fusionOps[0]->GetInControlAnchor());
    InOps[0]->GetOutDataAnchor(0)->LinkTo(fusionOps[0]->GetInDataAnchor(0));
    // InOps[0]->GetOutDataAnchor(0)->LinkTo(fusionOps[0]->GetInControlAnchor());
    // InOps[1]->GetOutControlAnchor()->LinkTo(fusionOps[1]->GetInControlAnchor());
    // InOps[1]->GetOutDataAnchor(0)->LinkTo(fusionOps[1]->GetInDataAnchor(0));
    InOps[1]->GetOutDataAnchor(0)->LinkTo(fusionOps[1]->GetInControlAnchor());
    // InOps[2]->GetOutControlAnchor()->LinkTo(fusionOps[2]->GetInControlAnchor());
    InOps[2]->GetOutDataAnchor(0)->LinkTo(fusionOps[2]->GetInDataAnchor(0));
    // InOps[2]->GetOutDataAnchor(0)->LinkTo(fusionOps[2]->GetInControlAnchor());

    fusionOps[0]->GetOutControlAnchor()->LinkTo(OutOps[0]->GetInControlAnchor());
    fusionOps[0]->GetOutDataAnchor(0)->LinkTo(OutOps[0]->GetInDataAnchor(0));
    // fusionOps[0]->GetOutDataAnchor(0)->LinkTo(OutOps[0]->GetInControlAnchor());
    // fusionOps[1]->GetOutControlAnchor()->LinkTo(OutOps[1]->GetInControlAnchor());
    fusionOps[1]->GetOutDataAnchor(0)->LinkTo(OutOps[1]->GetInDataAnchor(0));
    fusionOps[1]->GetOutDataAnchor(0)->LinkTo(OutOps[1]->GetInControlAnchor());
    // fusionOps[2]->GetOutControlAnchor()->LinkTo(OutOps[2]->GetInControlAnchor());
    fusionOps[2]->GetOutDataAnchor(0)->LinkTo(OutOps[2]->GetInDataAnchor(0));
    // fusionOps[2]->GetOutDataAnchor(0)->LinkTo(OutOps[2]->GetInControlAnchor());

    u32 segmentNum = 2;
    std::vector<u32> segmentIndex;
    segmentIndex.push_back(0);
    segmentIndex.push_back(1);

    ret = fusionHcomBroadcastOp.RunFusionOps(graph, fusionOps, segmentNum, segmentIndex);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

class HvdGraphOptimizerTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "\033[36m--HvdGraphOptimizerTest SetUP--\033[0m" << std::endl; }
    static void TearDownTestCase() { std::cout << "\033[36m--HvdGraphOptimizerTest TearDown--\033[0m" << std::endl; }
    // Some expensive resource shared by all tests.
    virtual void SetUp()
    {
        s32 portNum = 7;
        MOCKER(hrtGetHccsPortNum).stubs().with(mockcpp::any(), outBound(portNum)).will(returnValue(HCCL_SUCCESS));

        std::cout << "A Test SetUP" << std::endl;
    }
    virtual void TearDown() { std::cout << "A Test TearDown" << std::endl; }
};

TEST_F(HvdGraphOptimizerTest, ut_HvdGetSplitStrategy)
{
    HcclResult ret;
    ge::ComputeGraph graph("test_graph");
    HvdAllReduceFusion fusionHvdAllReduceOp;
    std::vector<ge::NodePtr> fusionOps(10);
    std::vector<u32> segmentIndex;

    std::vector<u32> segments;
    segments.push_back(6);
    segments.push_back(9);
    ret = fusionHvdAllReduceOp.GetSplitStrategy(graph, fusionOps, segmentIndex);
    EXPECT_EQ(segmentIndex[0], segments[0]);
    EXPECT_EQ(segmentIndex[1], segments[1]);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
}

TEST_F(HvdGraphOptimizerTest, ut_HvdRunFusionOps)
{
    HcclResult ret;
    ge::ComputeGraph graph("test_graph");
    HvdAllReduceFusion fusionHvdAllReduceOp;
    std::vector<ge::NodePtr> fusionOps(3);
    std::vector<ge::NodePtr> InOps(3);
    std::vector<ge::NodePtr> OutOps(3);

    fusionOps[0] = std::make_shared<NodeTest>();
    fusionOps[0]->Init();
    fusionOps[1] = std::make_shared<NodeTest>();
    fusionOps[1]->Init();
    fusionOps[2] = std::make_shared<NodeTest>();
    fusionOps[2]->Init();

    InOps[0] = std::make_shared<NodeTest>();
    InOps[0]->Init();
    InOps[1] = std::make_shared<NodeTest>();
    InOps[1]->Init();
    InOps[2] = std::make_shared<NodeTest>();
    InOps[2]->Init();

    OutOps[0] = std::make_shared<NodeTest>();
    OutOps[0]->Init();
    OutOps[1] = std::make_shared<NodeTest>();
    OutOps[1]->Init();
    OutOps[2] = std::make_shared<NodeTest>();
    OutOps[2]->Init();

    InOps[0]->GetOutControlAnchor()->LinkTo(fusionOps[0]->GetInControlAnchor());
    InOps[0]->GetOutDataAnchor(0)->LinkTo(fusionOps[0]->GetInDataAnchor(0));

    InOps[1]->GetOutDataAnchor(0)->LinkTo(fusionOps[1]->GetInControlAnchor());
    InOps[2]->GetOutDataAnchor(0)->LinkTo(fusionOps[2]->GetInDataAnchor(0));

    fusionOps[0]->GetOutControlAnchor()->LinkTo(OutOps[0]->GetInControlAnchor());
    fusionOps[0]->GetOutDataAnchor(0)->LinkTo(OutOps[0]->GetInDataAnchor(0));

    fusionOps[1]->GetOutDataAnchor(0)->LinkTo(OutOps[1]->GetInDataAnchor(0));
    fusionOps[1]->GetOutDataAnchor(0)->LinkTo(OutOps[1]->GetInControlAnchor());

    fusionOps[2]->GetOutDataAnchor(0)->LinkTo(OutOps[2]->GetInDataAnchor(0));

    u32 segmentNum = 2;
    std::vector<u32> segmentIndex;
    segmentIndex.push_back(1);
    segmentIndex.push_back(2);
    ret = fusionHvdAllReduceOp.RunFusionOps(graph, fusionOps, segmentNum, segmentIndex);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(HcomGraphOptimizerTest, ut_HcomReduceScatter_OptimizeFusedGraph)
{
    ge ::Status ret;
    std::map<std::string, std::string> options;

    // 未设置 rank table：失败
    options.insert(pair<string, string>(ge::OPTION_EXEC_RANK_TABLE_FILE, "rank_table.json"));
    // 实验室场景 设置 rank id
    options.insert(pair<string, string>(ge::OPTION_EXEC_DEPLOY_MODE, "0"));
    options.insert(pair<string, string>(ge::OPTION_EXEC_RANK_ID, "1"));
    options.insert(pair<string, string>(ge::OPTION_EXEC_PROFILING_MODE, "0"));

    MOCKER(HcomInitByFile).stubs().will(returnValue(HCCL_SUCCESS));
    ret = Initialize(options);
    EXPECT_EQ(ret, ge::SUCCESS);

    ge ::Status ge_ret;
    bool bRet;

    std::map<string, GraphOptimizerPtr> graphOptimizers;
    GetGraphOptimizerObjs(graphOptimizers);

    ge_ret = graphOptimizers.at(HCCL_GRAPH_OPTIMIZER_NAME)->Initialize(options, nullptr);
    EXPECT_EQ(ge_ret, ge::SUCCESS);

    ge::GraphOptimizerAttribute attrs;
    ge_ret = graphOptimizers.at(HCCL_GRAPH_OPTIMIZER_NAME)->GetAttributes(attrs);
    EXPECT_EQ(ge_ret, ge::SUCCESS);
    EXPECT_EQ(attrs.engineName, HCCL_OPS_ENGIN);
    EXPECT_EQ(attrs.scope, ge::UNIT);

    ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test_graph");
    ge::OpDesc op;
    const string type = HCCL_KERNEL_OP_TYPE_REDUCESCATTER;
    op.SetType(type);

    auto descPtr0 = std::make_shared<ge::OpDesc>("reducescatter0", HCCL_KERNEL_OP_TYPE_REDUCESCATTER);
    EXPECT_EQ(descPtr0->GetType(), HCCL_KERNEL_OP_TYPE_REDUCESCATTER);
    bRet = ge::AttrUtils::SetInt(descPtr0, HCOM_ATTR_RANK_SIZE, 8);
    EXPECT_EQ(bRet, true);
    auto addedNodePtr0 = graph->AddNode(descPtr0);
    EXPECT_NE(addedNodePtr0, nullptr);

    HCCL_INFO("start hccl OptimizeOriginalGraph");
    ge_ret = graphOptimizers.at(HCCL_GRAPH_OPTIMIZER_NAME)->OptimizeOriginalGraph(*graph);
    HCCL_INFO("end hccl OptimizeOriginalGraph");
    EXPECT_EQ(ge_ret, ge::SUCCESS);
    u64 streamNumber = 4;
    MOCKER(HcomGetWorkspaceSubStreamNum)
        .stubs()
        .with(mockcpp::any(), outBound(streamNumber))
        .will(returnValue(HCCL_SUCCESS));
    u32 rankSize = 8;
    MOCKER(HcomGetRankSize).stubs().with(mockcpp::any(), outBound(&rankSize)).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HcomGraphOptimizer::GetTaskNumAndCheckForceUnknown).stubs().will(returnValue(HCCL_SUCCESS));

    ge_ret = graphOptimizers.at(HCCL_GRAPH_OPTIMIZER_NAME)->OptimizeFusedGraph(*graph);
    EXPECT_EQ(ge_ret, ge::SUCCESS);
    ge_ret = graphOptimizers.at(HCCL_GRAPH_OPTIMIZER_NAME)->OptimizeWholeGraph(*graph);
    EXPECT_EQ(ge_ret, ge::SUCCESS);

    ge_ret = graphOptimizers.at(HCCL_GRAPH_OPTIMIZER_NAME)->Finalize();
    EXPECT_EQ(ge_ret, ge::SUCCESS);

    GlobalMockObject::verify();
}

TEST_F(HcomGraphOptimizerTest, ut_HcomAllGather_OptimizeFusedGraph)
{
    ge ::Status ret;
    std::map<std::string, std::string> options;

    // 未设置 rank table：失败
    options.insert(pair<string, string>(ge::OPTION_EXEC_RANK_TABLE_FILE, "rank_table.json"));
    // 实验室场景 设置 rank id
    options.insert(pair<string, string>(ge::OPTION_EXEC_DEPLOY_MODE, "0"));
    options.insert(pair<string, string>(ge::OPTION_EXEC_RANK_ID, "1"));
    options.insert(pair<string, string>(ge::OPTION_EXEC_PROFILING_MODE, "0"));

    MOCKER(HcomInitByFile).stubs().will(returnValue(HCCL_SUCCESS));
    ret = Initialize(options);
    EXPECT_EQ(ret, ge::SUCCESS);

    ge ::Status ge_ret;
    bool bRet;

    std::map<string, GraphOptimizerPtr> graphOptimizers;
    GetGraphOptimizerObjs(graphOptimizers);

    ge_ret = graphOptimizers.at(HCCL_GRAPH_OPTIMIZER_NAME)->Initialize(options, nullptr);
    EXPECT_EQ(ge_ret, ge::SUCCESS);

    ge::GraphOptimizerAttribute attrs;
    ge_ret = graphOptimizers.at(HCCL_GRAPH_OPTIMIZER_NAME)->GetAttributes(attrs);
    EXPECT_EQ(ge_ret, ge::SUCCESS);
    EXPECT_EQ(attrs.engineName, HCCL_OPS_ENGIN);
    EXPECT_EQ(attrs.scope, ge::UNIT);

    ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test_graph");
    ge::OpDesc op;
    const string type = HCCL_KERNEL_OP_TYPE_ALLGATHER;
    op.SetType(type);

    auto descPtr0 = std::make_shared<ge::OpDesc>("allgather0", HCCL_KERNEL_OP_TYPE_ALLGATHER);
    EXPECT_EQ(descPtr0->GetType(), HCCL_KERNEL_OP_TYPE_ALLGATHER);
    bRet = ge::AttrUtils::SetInt(descPtr0, HCOM_ATTR_RANK_SIZE, 8);
    EXPECT_EQ(bRet, true);
    auto addedNodePtr0 = graph->AddNode(descPtr0);
    EXPECT_NE(addedNodePtr0, nullptr);

    HCCL_INFO("start hccl OptimizeOriginalGraph");
    ge_ret = graphOptimizers.at(HCCL_GRAPH_OPTIMIZER_NAME)->OptimizeOriginalGraph(*graph);
    HCCL_INFO("end hccl OptimizeOriginalGraph");
    EXPECT_EQ(ge_ret, ge::SUCCESS);
    u64 streamNumber = 4;
    MOCKER(HcomGetWorkspaceSubStreamNum)
        .stubs()
        .with(mockcpp::any(), outBound(streamNumber))
        .will(returnValue(HCCL_SUCCESS));
    u32 rankSize = 8;
    MOCKER(HcomGetRankSize).stubs().with(mockcpp::any(), outBound(&rankSize)).will(returnValue(HCCL_SUCCESS));
    MOCKER(HcomGetDeviceType).stubs().will(returnValue(DevType::DEV_TYPE_310P3));
    MOCKER_CPP(&HcomGraphOptimizer::GetTaskNumAndCheckForceUnknown).stubs().will(returnValue(HCCL_SUCCESS));

    ge_ret = graphOptimizers.at(HCCL_GRAPH_OPTIMIZER_NAME)->OptimizeFusedGraph(*graph);
    EXPECT_EQ(ge_ret, ge::SUCCESS);
    ge_ret = graphOptimizers.at(HCCL_GRAPH_OPTIMIZER_NAME)->OptimizeWholeGraph(*graph);
    EXPECT_EQ(ge_ret, ge::SUCCESS);

    ge_ret = graphOptimizers.at(HCCL_GRAPH_OPTIMIZER_NAME)->Finalize();
    EXPECT_EQ(ge_ret, ge::SUCCESS);

    GlobalMockObject::verify();
}

TEST_F(HcomGraphOptimizerTest, ut_HcomRemoteRead_OptimizeFusedGraph)
{
    ge ::Status ret;
    std::map<std::string, std::string> options;

    // 未设置 rank table：失败
    options.insert(pair<string, string>(ge::OPTION_EXEC_RANK_TABLE_FILE, "rank_table.json"));
    // 实验室场景 设置 rank id
    options.insert(pair<string, string>(ge::OPTION_EXEC_DEPLOY_MODE, "0"));
    options.insert(pair<string, string>(ge::OPTION_EXEC_RANK_ID, "1"));
    options.insert(pair<string, string>(ge::OPTION_EXEC_PROFILING_MODE, "0"));

    MOCKER(HcomInitByFile).stubs().will(returnValue(HCCL_SUCCESS));
    ret = Initialize(options);
    EXPECT_EQ(ret, ge::SUCCESS);

    ge ::Status ge_ret;
    bool bRet;

    std::map<string, GraphOptimizerPtr> graphOptimizers;
    GetGraphOptimizerObjs(graphOptimizers);

    ge_ret = graphOptimizers.at(HCCL_GRAPH_OPTIMIZER_NAME)->Initialize(options, nullptr);
    EXPECT_EQ(ge_ret, ge::SUCCESS);

    ge::GraphOptimizerAttribute attrs;
    ge_ret = graphOptimizers.at(HCCL_GRAPH_OPTIMIZER_NAME)->GetAttributes(attrs);
    EXPECT_EQ(ge_ret, ge::SUCCESS);
    EXPECT_EQ(attrs.engineName, HCCL_OPS_ENGIN);
    EXPECT_EQ(attrs.scope, ge::UNIT);

    ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test_graph");
    ge::OpDesc op;
    const string type = HCCL_KERNEL_OP_TYPE_REMOTE_READ;
    op.SetType(type);

    auto descPtr0 = std::make_shared<ge::OpDesc>("remoteread0", HCCL_KERNEL_OP_TYPE_REMOTE_READ);
    EXPECT_EQ(descPtr0->GetType(), HCCL_KERNEL_OP_TYPE_REMOTE_READ);
    bRet = ge::AttrUtils::SetInt(descPtr0, HCOM_ATTR_RANK_SIZE, 8);
    EXPECT_EQ(bRet, true);
    auto addedNodePtr0 = graph->AddNode(descPtr0);
    EXPECT_NE(addedNodePtr0, nullptr);

    HCCL_INFO("start hccl OptimizeOriginalGraph");
    ge_ret = graphOptimizers.at(HCCL_GRAPH_OPTIMIZER_NAME)->OptimizeOriginalGraph(*graph);
    HCCL_INFO("end hccl OptimizeOriginalGraph");
    EXPECT_EQ(ge_ret, ge::SUCCESS);
    u64 streamNumber = 4;
    MOCKER(HcomGetWorkspaceSubStreamNum)
        .stubs()
        .with(mockcpp::any(), outBound(streamNumber))
        .will(returnValue(HCCL_SUCCESS));
    u32 rankSize = 8;
    MOCKER(HcomGetRankSize).stubs().with(mockcpp::any(), outBound(&rankSize)).will(returnValue(HCCL_SUCCESS));
    ge_ret = graphOptimizers.at(HCCL_GRAPH_OPTIMIZER_NAME)->OptimizeFusedGraph(*graph);
    EXPECT_EQ(ge_ret, ge::SUCCESS);
    ge_ret = graphOptimizers.at(HCCL_GRAPH_OPTIMIZER_NAME)->OptimizeWholeGraph(*graph);
    EXPECT_EQ(ge_ret, ge::SUCCESS);

    ge_ret = graphOptimizers.at(HCCL_GRAPH_OPTIMIZER_NAME)->Finalize();
    EXPECT_EQ(ge_ret, ge::SUCCESS);

    GlobalMockObject::verify();
}

TEST_F(HcomGraphOptimizerTest, ut_HcomReveive_OptimizeFusedGraph)
{
    ge ::Status ret;
    std::map<std::string, std::string> options;

    // 未设置 rank table：失败
    options.insert(pair<string, string>(ge::OPTION_EXEC_RANK_TABLE_FILE, "rank_table.json"));
    // 实验室场景 设置 rank id
    options.insert(pair<string, string>(ge::OPTION_EXEC_DEPLOY_MODE, "0"));
    options.insert(pair<string, string>(ge::OPTION_EXEC_RANK_ID, "1"));
    options.insert(pair<string, string>(ge::OPTION_EXEC_PROFILING_MODE, "0"));

    MOCKER(HcomInitByFile).stubs().will(returnValue(HCCL_SUCCESS));
    ret = Initialize(options);
    EXPECT_EQ(ret, ge::SUCCESS);

    ge ::Status ge_ret;
    bool bRet;

    std::map<string, GraphOptimizerPtr> graphOptimizers;
    GetGraphOptimizerObjs(graphOptimizers);

    ge_ret = graphOptimizers.at(HCCL_GRAPH_OPTIMIZER_NAME)->Initialize(options, nullptr);
    EXPECT_EQ(ge_ret, ge::SUCCESS);

    ge::GraphOptimizerAttribute attrs;
    ge_ret = graphOptimizers.at(HCCL_GRAPH_OPTIMIZER_NAME)->GetAttributes(attrs);
    EXPECT_EQ(ge_ret, ge::SUCCESS);
    EXPECT_EQ(attrs.engineName, HCCL_OPS_ENGIN);
    EXPECT_EQ(attrs.scope, ge::UNIT);

    ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test_graph");
    ge::OpDesc op;
    const string type = HCCL_KERNEL_OP_TYPE_RECEIVE;
    op.SetType(type);

    auto descPtr0 = std::make_shared<ge::OpDesc>("remoteread0", HCCL_KERNEL_OP_TYPE_RECEIVE);
    EXPECT_EQ(descPtr0->GetType(), HCCL_KERNEL_OP_TYPE_RECEIVE);
    bRet = ge::AttrUtils::SetInt(descPtr0, HCOM_ATTR_RANK_SIZE, 8);
    EXPECT_EQ(bRet, true);
    auto addedNodePtr0 = graph->AddNode(descPtr0);
    EXPECT_NE(addedNodePtr0, nullptr);

    HCCL_INFO("start hccl OptimizeOriginalGraph");
    ge_ret = graphOptimizers.at(HCCL_GRAPH_OPTIMIZER_NAME)->OptimizeOriginalGraph(*graph);
    HCCL_INFO("end hccl OptimizeOriginalGraph");
    EXPECT_EQ(ge_ret, ge::SUCCESS);
    u64 streamNumber = 4;
    MOCKER(HcomGetWorkspaceSubStreamNum)
        .stubs()
        .with(mockcpp::any(), outBound(streamNumber))
        .will(returnValue(HCCL_SUCCESS));
    u32 rankSize = 8;
    MOCKER(HcomGetRankSize).stubs().with(mockcpp::any(), outBound(&rankSize)).will(returnValue(HCCL_SUCCESS));
    ge::AttrUtils::HasAttr(*descPtr0, "DUMMY_SET_TRUE_DTYPE");
    ge::AttrUtils::HasAttr(*descPtr0, "DUMMY_SET_TRUE_SHAPE");
    ge_ret = graphOptimizers.at(HCCL_GRAPH_OPTIMIZER_NAME)->OptimizeFusedGraph(*graph);
    EXPECT_EQ(ge_ret, ge::SUCCESS);
    ge_ret = graphOptimizers.at(HCCL_GRAPH_OPTIMIZER_NAME)->OptimizeWholeGraph(*graph);
    EXPECT_EQ(ge_ret, ge::SUCCESS);

    ge_ret = graphOptimizers.at(HCCL_GRAPH_OPTIMIZER_NAME)->Finalize();
    EXPECT_EQ(ge_ret, ge::SUCCESS);

    GlobalMockObject::verify();
}

TEST_F(HcomGraphOptimizerTest, ut_OriginalGraphShapeTypeCfg)
{
    ge::Status ge_ret;
    MOCKER(&ge::NodeUtils::GetNodeUnknownShapeStatus)
        .stubs()
        .with(mockcpp::any(), outBound(true))
        .will(returnValue(ge::GRAPH_SUCCESS));
    HcomGraphOptimizer graphOptimizer;
    ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test_graph");
    const string type = "HcomAllReduce";
    ge::OpDescPtr opPtr_ = std::make_shared<ge::OpDesc>();
    opPtr_->SetType(type);
    graph->AddNode(opPtr_);
    ge_ret = graphOptimizer.OptimizeOriginalGraph(*graph);
    EXPECT_EQ(ge_ret, ge::SUCCESS);
    GlobalMockObject::verify();
}

ge::graphStatus GetOption1(ge::GEContext* that, const std::string& optionExec, std::string& dumpDebugValue)
{
    return ge::GRAPH_FAILED;
}

TEST_F(HcomGraphOptimizerTest, ut_SetUnknownShapAttr)
{
    HcclResult ret;
    int64_t memSize = GetExternalInputCCLBuffSize() + 1;
    HcomGraphOptimizer graphOptimizer;
    ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test_graph");
    const string type = "HcomAllReduce";
    ge::OpDescPtr opPtr_ = std::make_shared<ge::OpDesc>();
    opPtr_->SetType(type);
    graph->AddNode(opPtr_);

    MOCKER(&ge::NodeUtils::GetNodeUnknownShapeStatus)
        .stubs()
        .with(mockcpp::any(), outBound(false))
        .will(returnValue(ge::GRAPH_SUCCESS));

    MOCKER_CPP(&ge::GEContext::GetOption).stubs().will(invoke(GetOption1));

    ret = graphOptimizer.SetUnknownShapeAttr(*graph, true);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    const string hcclType = "HcomAllGather";
    opPtr_->SetType(hcclType);
    graph->AddNode(opPtr_);
    ret = graphOptimizer.SetUnknownShapeAttr(*graph, true);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    MOCKER(HcomGetDeviceType).stubs().will(returnValue(DevType::DEV_TYPE_310P3));

    ret = graphOptimizer.SetUnknownShapeAttr(*graph, true);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
}

TEST_F(HcomGraphOptimizerTest, ut_SetUnknownShapAttr_AlltoAllv)
{
    HcclResult ret;
    int64_t memSize = GetExternalInputCCLBuffSize() + 1;
    HcomGraphOptimizer graphOptimizer;
    ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test_graph");
    const string type = "HcomAllToAllV";
    ge::OpDescPtr opPtr_ = std::make_shared<ge::OpDesc>();
    opPtr_->SetType(type);
    graph->AddNode(opPtr_);

    MOCKER(&ge::NodeUtils::GetNodeUnknownShapeStatus)
        .stubs()
        .with(mockcpp::any(), outBound(false))
        .will(returnValue(ge::GRAPH_SUCCESS));

    MOCKER_CPP(&ge::GEContext::GetOption).stubs().will(invoke(GetOption1));

    ret = graphOptimizer.SetUnknownShapeAttr(*graph, true);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
}

TEST_F(HcomGraphOptimizerTest, ut_GetFusionOpInfo_Bcast)
{
    HcclResult ret;
    ge::ComputeGraph graph("test_graph");
    HcomBroadcastFusion fusionHcomBroadcastOp;
    FusionInfos fusionOps;
    std::vector<ge::NodePtr> ops(5);

    ge::AttrUtils::HasAttr(ops[0]->GetOpDesc(), "DUMMY_SET_FALSE_GROUP");
    ret = fusionHcomBroadcastOp.GetFusionOpInfo(ops[0], fusionOps);
    EXPECT_EQ(ret, HCCL_E_PARA);

    ge::AttrUtils::HasAttr(ops[1]->GetOpDesc(), "DUMMY_SET_TRUE_GROUP");
    std::string tempStr = HCCL_WORLD_GROUP;
    ret = fusionHcomBroadcastOp.GetFusionOpInfo(ops[1], fusionOps);

    ge::AttrUtils::SetInt(ops[1]->GetOpDesc(), "fusion", 2);
    ret = fusionHcomBroadcastOp.GetFusionOpInfo(ops[1], fusionOps);

    ge::AttrUtils::SetStr(ops[1]->GetOpDesc(), "group", tempStr);
    ge::AttrUtils::SetInt(ops[1]->GetOpDesc(), "fusion", 2);
    ret = fusionHcomBroadcastOp.GetFusionOpInfo(ops[1], fusionOps);

    ge::AttrUtils::SetStr(ops[1]->GetOpDesc(), "group", tempStr);
    ge::AttrUtils::SetInt(ops[1]->GetOpDesc(), "fusion_id", 2);
    ge::AttrUtils::SetInt(ops[1]->GetOpDesc(), "fusion", 2);
    ge::AttrUtils::SetInt(ops[1]->GetOpDesc(), "root_rank", 0);
    ret = fusionHcomBroadcastOp.GetFusionOpInfo(ops[1], fusionOps);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(fusionOps.size(), 1);

    ge::AttrUtils::SetStr(ops[2]->GetOpDesc(), "group", tempStr);
    ge::AttrUtils::SetInt(ops[2]->GetOpDesc(), "fusion_id", 3);
    ge::AttrUtils::SetInt(ops[2]->GetOpDesc(), "fusion", 2);
    ge::AttrUtils::SetInt(ops[2]->GetOpDesc(), "root_rank", 0);
    ret = fusionHcomBroadcastOp.GetFusionOpInfo(ops[2], fusionOps);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ge::AttrUtils::SetStr(ops[3]->GetOpDesc(), "group", tempStr);
    ge::AttrUtils::SetInt(ops[3]->GetOpDesc(), "fusion_id", -1);
    ge::AttrUtils::SetInt(ops[3]->GetOpDesc(), "fusion", 2);
    ge::AttrUtils::SetInt(ops[3]->GetOpDesc(), "root_rank", 0);
    ret = fusionHcomBroadcastOp.GetFusionOpInfo(ops[3], fusionOps);
    EXPECT_EQ(ret, HCCL_E_PARA);

    ge::AttrUtils::SetStr(ops[4]->GetOpDesc(), "group", tempStr);
    ge::AttrUtils::SetInt(ops[4]->GetOpDesc(), "fusion_id", 3);
    ge::AttrUtils::SetInt(ops[4]->GetOpDesc(), "fusion", 2);
    ge::AttrUtils::SetInt(ops[4]->GetOpDesc(), "root_rank", 0);
    ret = fusionHcomBroadcastOp.GetFusionOpInfo(ops[4], fusionOps);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ge::AttrUtils::SetStr(ops[0]->GetOpDesc(), "group", "hccl_world_group");
    ge::AttrUtils::SetInt(ops[1]->GetOpDesc(), "fusion_id", -1);
}

TEST_F(HcomGraphOptimizerTest, ut_FuseOps_Bcast)
{
    HcclResult ret;
    ge::ComputeGraph graph("test_graph");
    HcomBroadcastFusion fusionHcomBroadcastOp;
    std::map<std::string, std::vector<ge::NodePtr>> fusionOps;

    std::vector<ge::NodePtr> ops(3);
    std::vector<ge::NodePtr> nodeVec_0;
    nodeVec_0.push_back(ops[0]);
    nodeVec_0.push_back(ops[1]);
    nodeVec_0.push_back(ops[2]);
    fusionOps["hccl_world_group"] = nodeVec_0;

    MOCKER_CPP(&HcomBroadcastFusion::RunFusionOps).stubs().will(returnValue(HCCL_SUCCESS));
    ret = fusionHcomBroadcastOp.FuseOps(graph, nodeVec_0);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
    ge::AttrUtils::SetStr(ops[0]->GetOpDesc(), "group", "hccl_world_group");
}

TEST_F(HcomGraphOptimizerTest, ut_RunFusionOps_reduce)
{
    HcclResult ret;
    ge::ComputeGraph graph("test_graph");
    HcomReduceFusion fusionHcomReduceOp;
    std::vector<ge::NodePtr> fusionOps(3);
    std::vector<ge::NodePtr> InOps(3);
    std::vector<ge::NodePtr> OutOps(3);

    fusionOps[0] = std::make_shared<NodeTest>();
    fusionOps[0]->Init();
    fusionOps[1] = std::make_shared<NodeTest>();
    fusionOps[1]->Init();
    fusionOps[2] = std::make_shared<NodeTest>();
    fusionOps[2]->Init();

    InOps[0] = std::make_shared<NodeTest>();
    InOps[0]->Init();
    InOps[1] = std::make_shared<NodeTest>();
    InOps[1]->Init();
    InOps[2] = std::make_shared<NodeTest>();
    InOps[2]->Init();

    OutOps[0] = std::make_shared<NodeTest>();
    OutOps[0]->Init();
    OutOps[1] = std::make_shared<NodeTest>();
    OutOps[1]->Init();
    OutOps[2] = std::make_shared<NodeTest>();
    OutOps[2]->Init();

    InOps[0]->GetOutControlAnchor()->LinkTo(fusionOps[0]->GetInControlAnchor());
    InOps[0]->GetOutDataAnchor(0)->LinkTo(fusionOps[0]->GetInDataAnchor(0));
    // InOps[0]->GetOutDataAnchor(0)->LinkTo(fusionOps[0]->GetInControlAnchor());
    // InOps[1]->GetOutControlAnchor()->LinkTo(fusionOps[1]->GetInControlAnchor());
    // InOps[1]->GetOutDataAnchor(0)->LinkTo(fusionOps[1]->GetInDataAnchor(0));
    InOps[1]->GetOutDataAnchor(0)->LinkTo(fusionOps[1]->GetInControlAnchor());
    // InOps[2]->GetOutControlAnchor()->LinkTo(fusionOps[2]->GetInControlAnchor());
    InOps[2]->GetOutDataAnchor(0)->LinkTo(fusionOps[2]->GetInDataAnchor(0));
    // InOps[2]->GetOutDataAnchor(0)->LinkTo(fusionOps[2]->GetInControlAnchor());

    fusionOps[0]->GetOutControlAnchor()->LinkTo(OutOps[0]->GetInControlAnchor());
    fusionOps[0]->GetOutDataAnchor(0)->LinkTo(OutOps[0]->GetInDataAnchor(0));
    // fusionOps[0]->GetOutDataAnchor(0)->LinkTo(OutOps[0]->GetInControlAnchor());
    // fusionOps[1]->GetOutControlAnchor()->LinkTo(OutOps[1]->GetInControlAnchor());
    fusionOps[1]->GetOutDataAnchor(0)->LinkTo(OutOps[1]->GetInDataAnchor(0));
    fusionOps[1]->GetOutDataAnchor(0)->LinkTo(OutOps[1]->GetInControlAnchor());
    // fusionOps[2]->GetOutControlAnchor()->LinkTo(OutOps[2]->GetInControlAnchor());
    fusionOps[2]->GetOutDataAnchor(0)->LinkTo(OutOps[2]->GetInDataAnchor(0));
    // fusionOps[2]->GetOutDataAnchor(0)->LinkTo(OutOps[2]->GetInControlAnchor());

    ret = fusionHcomReduceOp.RunFusionOpsReduce(graph, fusionOps);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(HcomGraphOptimizerTest, ut_GetFusionOpInfo_Reduce)
{
    HcclResult ret;
    ge::ComputeGraph graph("test_graph");
    HcomReduceFusion fusionHcomReduceOp;
    FusionInfos fusionOps;
    std::vector<ge::NodePtr> ops(5);

    ge::AttrUtils::HasAttr(ops[0]->GetOpDesc(), "DUMMY_SET_FALSE_GROUP");
    ret = fusionHcomReduceOp.GetFusionOpInfo(ops[0], fusionOps);
    EXPECT_EQ(ret, HCCL_E_PARA);

    std::string tempStrReduction = "sum";
    ge::AttrUtils::SetStr(ops[1]->GetOpDesc(), "reduction", tempStrReduction);

    ge::AttrUtils::HasAttr(ops[1]->GetOpDesc(), "DUMMY_SET_TRUE_GROUP");
    std::string tempStr = HCCL_WORLD_GROUP;
    ret = fusionHcomReduceOp.GetFusionOpInfo(ops[1], fusionOps);

    ge::AttrUtils::SetInt(ops[1]->GetOpDesc(), "fusion", 0);
    ret = fusionHcomReduceOp.GetFusionOpInfo(ops[1], fusionOps);

    ge::AttrUtils::SetInt(ops[1]->GetOpDesc(), "fusion", 1);
    ret = fusionHcomReduceOp.GetFusionOpInfo(ops[1], fusionOps);

    ge::AttrUtils::SetStr(ops[1]->GetOpDesc(), "group", tempStr);
    ge::AttrUtils::SetInt(ops[1]->GetOpDesc(), "fusion", 1);
    ret = fusionHcomReduceOp.GetFusionOpInfo(ops[1], fusionOps);

    ge::AttrUtils::SetStr(ops[1]->GetOpDesc(), "group", tempStr);
    ge::AttrUtils::SetInt(ops[1]->GetOpDesc(), "fusion_id", 2);
    ge::AttrUtils::SetInt(ops[1]->GetOpDesc(), "fusion", 2);
    ge::AttrUtils::SetInt(ops[1]->GetOpDesc(), "root_rank", 0);
    ret = fusionHcomReduceOp.GetFusionOpInfo(ops[1], fusionOps);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(fusionOps.size(), 1);
    EXPECT_EQ(fusionOps["hccl_world_group"].size(), 0);
    // EXPECT_EQ(fusionOps["hccl_world_group"][2].size(), 1);

    ge::AttrUtils::SetStr(ops[2]->GetOpDesc(), "group", tempStr);
    ge::AttrUtils::SetInt(ops[2]->GetOpDesc(), "fusion_id", 3);
    ge::AttrUtils::SetInt(ops[2]->GetOpDesc(), "fusion", 1);
    ge::AttrUtils::SetInt(ops[2]->GetOpDesc(), "root_rank", 0);
    ret = fusionHcomReduceOp.GetFusionOpInfo(ops[2], fusionOps);
    EXPECT_EQ(ret, HCCL_E_PARA);

    ge::AttrUtils::SetStr(ops[3]->GetOpDesc(), "group", tempStr);
    ge::AttrUtils::SetInt(ops[3]->GetOpDesc(), "fusion_id", -1);
    ge::AttrUtils::SetInt(ops[3]->GetOpDesc(), "fusion", 2);
    ge::AttrUtils::SetInt(ops[3]->GetOpDesc(), "root_rank", 0);
    ret = fusionHcomReduceOp.GetFusionOpInfo(ops[3], fusionOps);
    EXPECT_EQ(ret, HCCL_E_PARA);

    ge::AttrUtils::SetStr(ops[4]->GetOpDesc(), "group", tempStr);
    ge::AttrUtils::SetInt(ops[4]->GetOpDesc(), "fusion_id", 2);
    ge::AttrUtils::SetInt(ops[4]->GetOpDesc(), "fusion", 2);
    ge::AttrUtils::SetInt(ops[4]->GetOpDesc(), "root_rank", 0);
    ret = fusionHcomReduceOp.GetFusionOpInfo(ops[4], fusionOps);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ge::AttrUtils::SetStr(ops[0]->GetOpDesc(), "group", "hccl_world_group");
    ge::AttrUtils::SetInt(ops[1]->GetOpDesc(), "fusion_id", -1);

    MOCKER(HcomGetCCLBufferAvailableSize).stubs().with(mockcpp::any()).will(invoke(stub_HcomGetCCLBufferAvailableSize));

    MOCKER(&HcomOpUtils::GetAllInputsTensorMemSize).stubs().will(invoke(stub_GetAllInputsTensorMemSize));

    FusionInfos fusionOpsTemp;
    ret = fusionHcomReduceOp.GetFusionOpsSlices(fusionOps, fusionOpsTemp);
}

TEST_F(HcomGraphOptimizerTest, ut_FuseOps_Reduce)
{
    HcclResult ret;
    ge::ComputeGraph graph("test_graph");
    HcomReduceFusion fusionHcomReduceOp;
    std::map<std::string, std::vector<ge::NodePtr>> fusionOps;

    std::vector<ge::NodePtr> ops(3);
    std::vector<ge::NodePtr> nodeVec_0;
    nodeVec_0.push_back(ops[0]);
    nodeVec_0.push_back(ops[1]);
    nodeVec_0.push_back(ops[2]);
    fusionOps["hccl_world_group"] = nodeVec_0;

    MOCKER_CPP(&HcomReduceFusion::RunFusionOpsReduce).stubs().will(returnValue(HCCL_SUCCESS));
    ret = fusionHcomReduceOp.FuseOps(graph, nodeVec_0);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
    ge::AttrUtils::SetStr(ops[0]->GetOpDesc(), "group", "hccl_world_group");
}

TEST_F(HcomGraphOptimizerTest, ut_GetFusionSegments_1)
{
    std::vector<ge::NodePtr> ops(7);
    std::vector<ge::NodePtr> nodes;
    nodes.push_back(ops[0]);
    nodes.push_back(ops[1]);
    nodes.push_back(ops[2]);
    nodes.push_back(ops[3]);
    nodes.push_back(ops[4]);
    nodes.push_back(ops[5]);
    nodes.push_back(ops[6]);

    uint64_t inputTensorSize = 200 * 1024 * 1024;
    MOCKER(&HcomOpUtils::GetAllInputsTensorOriginSize)
        .stubs()
        .with(mockcpp::any(), outBound(inputTensorSize))
        .will(returnValue(HCCL_SUCCESS));

    std::vector<uint32_t> segmentIndex;
    HcomBroadcastFusion fusionHcomBroadcastOp;
    fusionHcomBroadcastOp.fusionTensorSizeLimit_ = 500 * 1024 * 1024;
    HcclResult ret = fusionHcomBroadcastOp.GetFusionSegments(nodes, segmentIndex);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(segmentIndex.size(), 4);
    EXPECT_EQ(segmentIndex[0], 1);
    EXPECT_EQ(segmentIndex[1], 3);
    EXPECT_EQ(segmentIndex[2], 5);
    EXPECT_EQ(segmentIndex[3], 6);
    GlobalMockObject::verify();
}

TEST_F(HcomGraphOptimizerTest, ut_GetFusionSegments_2)
{
    std::vector<ge::NodePtr> ops(3);
    std::vector<ge::NodePtr> nodes;
    nodes.push_back(ops[0]);
    nodes.push_back(ops[1]);
    nodes.push_back(ops[2]);

    uint64_t inputTensorSize = 500 * 1024 * 1024;
    MOCKER(&HcomOpUtils::GetAllInputsTensorOriginSize)
        .stubs()
        .with(mockcpp::any(), outBound(inputTensorSize))
        .will(returnValue(HCCL_SUCCESS));

    std::vector<uint32_t> segmentIndex;
    HcomBroadcastFusion fusionHcomBroadcastOp;
    fusionHcomBroadcastOp.fusionTensorSizeLimit_ = 500 * 1024 * 1024;
    HcclResult ret = fusionHcomBroadcastOp.GetFusionSegments(nodes, segmentIndex);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(segmentIndex.size(), 3);
    EXPECT_EQ(segmentIndex[0], 0);
    EXPECT_EQ(segmentIndex[1], 1);
    EXPECT_EQ(segmentIndex[2], 2);
    GlobalMockObject::verify();
}

HcclResult GetOffDeviceTypeWithoutDevMock(DevType& devType)
{
    devType = DevType::DEV_TYPE_310P3;
    HCCL_DEBUG("[offline] Get devtype[%u]....", devType);
    return HCCL_SUCCESS;
}

TEST_F(HcomGraphOptimizerTest, ut_OptimizeFusedGraph_allreduce)
{
    ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test_graph");
    ge::OpDesc op;
    const string type = HCCL_KERNEL_OP_TYPE_ALLREDUCE;
    op.SetType(type);

    auto descPtr0 = std::make_shared<ge::OpDesc>("Allreduce0", HCCL_KERNEL_OP_TYPE_ALLREDUCE);
    EXPECT_EQ(descPtr0->GetType(), HCCL_KERNEL_OP_TYPE_ALLREDUCE);
    auto addedNodePtr0 = graph->AddNode(descPtr0);
    EXPECT_NE(addedNodePtr0, nullptr);

    auto descPtr1 = std::make_shared<ge::OpDesc>("Allreduce1", HCCL_KERNEL_OP_TYPE_ALLREDUCE);
    EXPECT_EQ(descPtr1->GetType(), HCCL_KERNEL_OP_TYPE_ALLREDUCE);
    auto addedNodePtr1 = graph->AddNode(descPtr1);
    EXPECT_NE(addedNodePtr1, nullptr);

    u64 streamNumber = 4;
    char* group = "127.0.0.1%eth0_60000_0_1698475280390992";
    MOCKER(HcomGetWorkspaceSubStreamNum)
        .stubs()
        .with(mockcpp::any(), outBound(streamNumber))
        .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&HcomGraphOptimizer::GetTaskNumAndCheckForceUnknown).stubs().will(returnValue(HCCL_SUCCESS));

    u32 rankSize = 8;
    MOCKER(HcomGetRankSize).stubs().with(mockcpp::any(), outBound(&rankSize)).will(returnValue(HCCL_SUCCESS));

    HcomGraphOptimizer graphOptimizer;
    ge::Status ge_ret = graphOptimizer.OptimizeFusedGraph(*graph);
    EXPECT_EQ(ge_ret, ge::SUCCESS);

    MOCKER(IsOfflineCompilation).stubs().will(returnValue(true));

    MOCKER(GetOffDeviceTypeWithoutDev).stubs().will(invoke(GetOffDeviceTypeWithoutDevMock));
    ge_ret = graphOptimizer.OptimizeFusedGraph(*graph);

    GlobalMockObject::verify();
}

TEST_F(HcomGraphOptimizerTest, ut_OptimizeFusedGraph_broadcast)
{
    ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test_graph");
    ge::OpDesc op;
    const string type = HCCL_KERNEL_OP_TYPE_BROADCAST;
    op.SetType(type);

    auto descPtr0 = std::make_shared<ge::OpDesc>("Broadcast0", HCCL_KERNEL_OP_TYPE_BROADCAST);
    EXPECT_EQ(descPtr0->GetType(), HCCL_KERNEL_OP_TYPE_BROADCAST);
    auto addedNodePtr0 = graph->AddNode(descPtr0);
    EXPECT_NE(addedNodePtr0, nullptr);

    auto descPtr1 = std::make_shared<ge::OpDesc>("Allreduce1", HCCL_KERNEL_OP_TYPE_BROADCAST);
    EXPECT_EQ(descPtr1->GetType(), HCCL_KERNEL_OP_TYPE_BROADCAST);
    auto addedNodePtr1 = graph->AddNode(descPtr1);
    EXPECT_NE(addedNodePtr1, nullptr);

    u64 streamNumber = 4;
    MOCKER(HcomGetWorkspaceSubStreamNum)
        .stubs()
        .with(mockcpp::any(), outBound(streamNumber))
        .will(returnValue(HCCL_SUCCESS));

    u32 rankSize = 8;
    MOCKER(HcomGetRankSize).stubs().with(mockcpp::any(), outBound(&rankSize)).will(returnValue(HCCL_SUCCESS));

    HcomGraphOptimizer graphOptimizer;
    ge::Status ge_ret = graphOptimizer.OptimizeFusedGraph(*graph);
    EXPECT_EQ(ge_ret, ge::SUCCESS);

    GlobalMockObject::verify();
}

TEST_F(HcomGraphOptimizerTest, ut_OptimizeFusedGraph_broadcast_unknown)
{
    ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test_graph");
    ge::OpDesc op;
    const string type = HCCL_KERNEL_OP_TYPE_BROADCAST;
    op.SetType(type);

    auto descPtr0 = std::make_shared<ge::OpDesc>("Broadcast0", HCCL_KERNEL_OP_TYPE_BROADCAST);
    EXPECT_EQ(descPtr0->GetType(), HCCL_KERNEL_OP_TYPE_BROADCAST);
    auto addedNodePtr0 = graph->AddNode(descPtr0);
    EXPECT_NE(addedNodePtr0, nullptr);

    auto descPtr1 = std::make_shared<ge::OpDesc>("Allreduce1", HCCL_KERNEL_OP_TYPE_BROADCAST);
    EXPECT_EQ(descPtr1->GetType(), HCCL_KERNEL_OP_TYPE_BROADCAST);
    auto addedNodePtr1 = graph->AddNode(descPtr1);
    EXPECT_NE(addedNodePtr1, nullptr);

    u64 streamNumber = 4;
    MOCKER(HcomGetWorkspaceSubStreamNum)
        .stubs()
        .with(mockcpp::any(), outBound(streamNumber))
        .will(returnValue(HCCL_SUCCESS));

    u32 rankSize = 8;
    MOCKER(HcomGetRankSize).stubs().with(mockcpp::any(), outBound(&rankSize)).will(returnValue(HCCL_SUCCESS));

    MOCKER(&ge::NodeUtils::GetNodeUnknownShapeStatus)
        .stubs()
        .with(mockcpp::any(), outBound(true))
        .will(returnValue(ge::GRAPH_SUCCESS));

    HcomGraphOptimizer graphOptimizer;
    ge::Status ge_ret = graphOptimizer.OptimizeFusedGraph(*graph);
    EXPECT_EQ(ge_ret, ge::SUCCESS);

    GlobalMockObject::verify();
}

TEST_F(HcomGraphOptimizerTest, ut_GetCommFromOpDesc_by_group_pytorch)
{
    HcclResult ret;
    ge::ComputeGraph graph("test_graph");
    HcomGraphOptimizer hcomGraphOptimizer;
    std::vector<ge::NodePtr> option(2);
    int64_t hcomComm = 0;
    std::string sGroup;
    std::string tempStr = HCCL_WORLD_GROUP;

    ge::AttrUtils::HasAttr(option[0]->GetOpDesc(), "DUMMY_SET_TRUE_GROUP");
    ge::AttrUtils::SetStr(option[0]->GetOpDesc(), "group", tempStr);
    ge::OpDescPtr op1 = option[0]->GetOpDesc();
    ret = hcomGraphOptimizer.GetCommFromOpDesc(op1, hcomComm, sGroup);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(HcomGraphOptimizerTest, ut_GetFusionOpInfo_Reduce_by_comm_pytorch)
{
    HcclResult ret;
    ge::ComputeGraph graph("test_graph");
    HcomReduceFusion fusionHcomReduceOp;
    FusionInfos fusionOps;
    std::vector<ge::NodePtr> ops(1);

    std::string tempStrReduction = "sum";

    ge::AttrUtils::HasAttr(ops[0]->GetOpDesc(), "DUMMY_SET_TRUE_COMM");
    ge::AttrUtils::HasAttr(ops[0]->GetOpDesc(), "comm");

    ge::AttrUtils::SetStr(ops[0]->GetOpDesc(), "reduction", tempStrReduction);
    ge::AttrUtils::SetInt(ops[0]->GetOpDesc(), "comm", 645678156);
    ge::AttrUtils::SetInt(ops[0]->GetOpDesc(), "fusion_id", 2);
    ge::AttrUtils::SetInt(ops[0]->GetOpDesc(), "fusion", 2);
    ge::AttrUtils::SetInt(ops[0]->GetOpDesc(), "root_rank", 0);
    MOCKER(HcclCommGraphGetIdentifier).stubs().will(returnValue(HCCL_SUCCESS));
    ret = fusionHcomReduceOp.GetFusionOpInfo(ops[0], fusionOps);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ge::AttrUtils::HasAttr(ops[0]->GetOpDesc(), "DUMMY_SET_FALSE_COMM");

    MOCKER(HcomGetCCLBufferAvailableSize).stubs().with(mockcpp::any()).will(invoke(stub_HcomGetCCLBufferAvailableSize));

    MOCKER(&HcomOpUtils::GetAllInputsTensorMemSize).stubs().will(invoke(stub_GetAllInputsTensorMemSize));

    FusionInfos fusionOpsTemp;
    ret = fusionHcomReduceOp.GetFusionOpsSlices(fusionOps, fusionOpsTemp);
}

TEST_F(HcomGraphOptimizerTest, ut_GetFusionOpInfo_AllReduce_by_comm_pytorch)
{
    HcclResult ret;
    ge::ComputeGraph graph("test_graph");
    HcomAllReduceFusion fusionHcomAllReduceOp;
    FusionInfos fusionOps;
    std::vector<ge::NodePtr> ops(1);

    std::string tempStrReduction = "sum";

    ge::AttrUtils::HasAttr(ops[0]->GetOpDesc(), "DUMMY_SET_TRUE_COMM");
    ge::AttrUtils::HasAttr(ops[0]->GetOpDesc(), "comm");

    ge::AttrUtils::SetStr(ops[0]->GetOpDesc(), "reduction", tempStrReduction);
    ge::AttrUtils::SetInt(ops[0]->GetOpDesc(), "comm", 645678156);
    ge::AttrUtils::SetInt(ops[0]->GetOpDesc(), "fusion_id", 2);
    ge::AttrUtils::SetInt(ops[0]->GetOpDesc(), "fusion", 2);
    ge::AttrUtils::SetInt(ops[0]->GetOpDesc(), "root_rank", 0);
    MOCKER(HcclCommGraphGetIdentifier).stubs().will(returnValue(HCCL_SUCCESS));
    ret = fusionHcomAllReduceOp.GetFusionOpInfo(ops[0], fusionOps);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ge::AttrUtils::HasAttr(ops[0]->GetOpDesc(), "DUMMY_SET_FALSE_COMM");

    MOCKER(HcomGetCCLBufferAvailableSize).stubs().with(mockcpp::any()).will(invoke(stub_HcomGetCCLBufferAvailableSize));

    MOCKER(&HcomOpUtils::GetAllInputsTensorMemSize).stubs().will(invoke(stub_GetAllInputsTensorMemSize));

    FusionInfos fusionOpsTemp;
    ret = fusionHcomAllReduceOp.GetFusionOpsSlices(fusionOps, fusionOpsTemp);
}

TEST_F(HcomGraphOptimizerTest, ut_FuseOps_AllReduce_by_comm_pytorch)
{
    HcclResult ret;
    bool bRet;
    ge::ComputeGraph graph("test_graph");
    ge::OpDescPtr nodeComm = 0;
    ge::OpDescPtr nodeFusionId = 0;
    HcomAllReduceFusion fusionHcomAllReduceOp;
    std::map<std::string, std::vector<ge::NodePtr>> fusionOps;

    std::vector<ge::NodePtr> ops(3);
    std::vector<ge::NodePtr> nodeVec_0;
    int64_t comm = 645678545;
    int64_t fusionid = HCOM_ATTR_FUSION_ID_DEFAULT;
    ge::AttrUtils::HasAttr(ops[0]->GetOpDesc(), "DUMMY_SET_TRUE_COMM");
    ge::AttrUtils::HasAttr(ops[0]->GetOpDesc(), "comm");
    bRet = ge::AttrUtils::SetInt(nodeComm, "comm", comm);
    EXPECT_EQ(bRet, true);
    bRet = ge::AttrUtils::SetInt(nodeFusionId, "fusion_id", fusionid);
    EXPECT_EQ(bRet, true);
    nodeVec_0.push_back(ops[0]);
    nodeVec_0.push_back(ops[1]);
    nodeVec_0.push_back(ops[2]);
    fusionOps["hccl_world_group"] = nodeVec_0;

    MOCKER_CPP(&HcomAllReduceFusion::RunFusionOpsReduce).stubs().will(returnValue(HCCL_SUCCESS));
    ret = fusionHcomAllReduceOp.FuseOps(graph, fusionOps["hccl_world_group"]);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ge::AttrUtils::HasAttr(ops[0]->GetOpDesc(), "DUMMY_SET_FALSE_COMM");
    GlobalMockObject::verify();
}

TEST_F(HcomGraphOptimizerTest, ut_RunFusionOpss_allreduce_by_comm_pytorch)
{
    HcclResult ret;
    ge::ComputeGraph graph("test_graph");
    HcomAllReduceFusion fusionAllHcomReduceOp;
    std::vector<ge::NodePtr> fusionOps(3);
    std::vector<ge::NodePtr> InOps(3);
    std::vector<ge::NodePtr> OutOps(3);

    fusionOps[0] = std::make_shared<NodeTest>();
    fusionOps[0]->Init();
    fusionOps[1] = std::make_shared<NodeTest>();
    fusionOps[1]->Init();
    fusionOps[2] = std::make_shared<NodeTest>();
    fusionOps[2]->Init();

    InOps[0] = std::make_shared<NodeTest>();
    InOps[0]->Init();
    InOps[1] = std::make_shared<NodeTest>();
    InOps[1]->Init();
    InOps[2] = std::make_shared<NodeTest>();
    InOps[2]->Init();

    OutOps[0] = std::make_shared<NodeTest>();
    OutOps[0]->Init();
    OutOps[1] = std::make_shared<NodeTest>();
    OutOps[1]->Init();
    OutOps[2] = std::make_shared<NodeTest>();
    OutOps[2]->Init();

    InOps[0]->GetOutControlAnchor()->LinkTo(fusionOps[0]->GetInControlAnchor());
    InOps[0]->GetOutDataAnchor(0)->LinkTo(fusionOps[0]->GetInDataAnchor(0));

    InOps[1]->GetOutDataAnchor(0)->LinkTo(fusionOps[1]->GetInControlAnchor());

    InOps[2]->GetOutDataAnchor(0)->LinkTo(fusionOps[2]->GetInDataAnchor(0));

    fusionOps[0]->GetOutControlAnchor()->LinkTo(OutOps[0]->GetInControlAnchor());
    fusionOps[0]->GetOutDataAnchor(0)->LinkTo(OutOps[0]->GetInDataAnchor(0));

    fusionOps[1]->GetOutDataAnchor(0)->LinkTo(OutOps[1]->GetInDataAnchor(0));
    fusionOps[1]->GetOutDataAnchor(0)->LinkTo(OutOps[1]->GetInControlAnchor());

    fusionOps[2]->GetOutDataAnchor(0)->LinkTo(OutOps[2]->GetInDataAnchor(0));

    ret = fusionAllHcomReduceOp.RunFusionOpsReduce(graph, fusionOps);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(HcomGraphOptimizerTest, ut_GetFusionOpInfo_Bcast_by_comm_pytorch1)
{
    HcclResult ret;
    ge::ComputeGraph graph("test_graph");
    HcomBroadcastFusion fusionHcomBroadcastOp;
    FusionInfos fusionOps;
    std::vector<ge::NodePtr> ops(1);

    ge::AttrUtils::HasAttr(ops[0]->GetOpDesc(), "DUMMY_SET_TRUE_COMM");
    ge::AttrUtils::HasAttr(ops[0]->GetOpDesc(), "comm");

    ge::AttrUtils::SetInt(ops[0]->GetOpDesc(), "comm", 645678156);
    ge::AttrUtils::SetInt(ops[0]->GetOpDesc(), "fusion_id", 2);
    ge::AttrUtils::SetInt(ops[0]->GetOpDesc(), "fusion", 2);
    ge::AttrUtils::SetInt(ops[0]->GetOpDesc(), "root_rank", 0);
    MOCKER(HcclCommGraphGetIdentifier).stubs().will(returnValue(HCCL_SUCCESS));
    ret = fusionHcomBroadcastOp.GetFusionOpInfo(ops[0], fusionOps);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ge::AttrUtils::HasAttr(ops[0]->GetOpDesc(), "DUMMY_SET_FALSE_COMM");
}

TEST_F(HcomGraphOptimizerTest, ut_GetFusionOpInfo_Bcast_by_comm_pytorch2)
{
    HcclResult ret;
    ge::ComputeGraph graph("test_graph");
    HcomBroadcastFusion fusionHcomBroadcastOp;
    FusionInfos fusionOps;
    std::vector<ge::NodePtr> ops(1);

    ge::AttrUtils::HasAttr(ops[0]->GetOpDesc(), "DUMMY_SET_TRUE_COMM");
    ge::AttrUtils::HasAttr(ops[0]->GetOpDesc(), "DUMMY_SET_TRUE_GROUP");
    ge::AttrUtils::HasAttr(ops[0]->GetOpDesc(), "comm");

    ge::AttrUtils::SetInt(ops[0]->GetOpDesc(), "comm", 0);
    ge::AttrUtils::SetStr(ops[0]->GetOpDesc(), "group", "hccl_world_group");
    ge::AttrUtils::SetInt(ops[0]->GetOpDesc(), "fusion_id", 2);
    ge::AttrUtils::SetInt(ops[0]->GetOpDesc(), "fusion", 2);
    ge::AttrUtils::SetInt(ops[0]->GetOpDesc(), "root_rank", 0);
    MOCKER(HcclCommGraphGetIdentifier).stubs().will(returnValue(HCCL_SUCCESS));
    ret = fusionHcomBroadcastOp.GetFusionOpInfo(ops[0], fusionOps);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ge::AttrUtils::HasAttr(ops[0]->GetOpDesc(), "DUMMY_SET_FALSE_COMM");
    ge::AttrUtils::HasAttr(ops[0]->GetOpDesc(), "DUMMY_SET_FALSE_GROUP");
}

TEST_F(HcomGraphOptimizerTest, ut_GetCommFromOpDesc_by_comm_pytorch1)
{
    HcclResult ret;
    ge::ComputeGraph graph("test_graph");
    HcomGraphOptimizer hcomGraphOptimizer;
    std::vector<ge::NodePtr> ops(1);
    int64_t hcomComm = 0;
    std::string sGroup;

    ge::AttrUtils::HasAttr(ops[0]->GetOpDesc(), "DUMMY_SET_TRUE_COMM");
    ge::AttrUtils::HasAttr(ops[0]->GetOpDesc(), "comm");
    ge::AttrUtils::SetInt(ops[0]->GetOpDesc(), "comm", 645678156);
    ge::OpDescPtr op0 = ops[0]->GetOpDesc();
    ret = hcomGraphOptimizer.GetCommFromOpDesc(op0, hcomComm, sGroup);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ge::AttrUtils::HasAttr(ops[0]->GetOpDesc(), "DUMMY_SET_FALSE_COMM");
}

TEST_F(HcomGraphOptimizerTest, ut_GetCommFromOpDesc_by_comm_pytorch2)
{
    HcclResult ret;
    ge::ComputeGraph graph("test_graph");
    HcomGraphOptimizer hcomGraphOptimizer;
    std::vector<ge::NodePtr> ops(1);
    int64_t hcomComm = 0;
    std::string sGroup;

    ge::AttrUtils::HasAttr(ops[0]->GetOpDesc(), "DUMMY_SET_TRUE_COMM");
    ge::AttrUtils::HasAttr(ops[0]->GetOpDesc(), "comm");
    ge::AttrUtils::SetInt(ops[0]->GetOpDesc(), "comm", 0);
    ge::OpDescPtr op0 = ops[0]->GetOpDesc();
    ret = hcomGraphOptimizer.GetCommFromOpDesc(op0, hcomComm, sGroup);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ge::AttrUtils::HasAttr(ops[0]->GetOpDesc(), "DUMMY_SET_FALSE_COMM");
}

TEST_F(HcomGraphOptimizerTest, ut_HcomCalcOpRunningParam_by_comm_pytorch)
{
    HcclResult ret;
    ge::ComputeGraph graph("test_graph");
    HcomGraphOptimizer hcomGraphOptimizer;
    std::vector<ge::Node> ops(1);
    int64_t hcomComm = 0;
    u64 streamNum = 8;
    std::string sGroup;
    const string type = HCCL_KERNEL_OP_TYPE_ALLREDUCE;
    ops[0].GetOpDesc()->SetType(type);

    auto descPtr0 = std::make_shared<ge::OpDesc>("Allreduce0", HCCL_KERNEL_OP_TYPE_ALLREDUCE);
    EXPECT_EQ(descPtr0->GetType(), HCCL_KERNEL_OP_TYPE_ALLREDUCE);

    ge::AttrUtils::HasAttr(ops[0].GetOpDesc(), "DUMMY_SET_TRUE_COMM");
    ge::AttrUtils::HasAttr(ops[0].GetOpDesc(), "comm");
    ge::AttrUtils::SetInt(ops[0].GetOpDesc(), "comm", 645678156);
    MOCKER(HcclCommGraphGetWorkspaceSubStreamNum).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(HcclCommGraphGetAllReduceScratchSize).stubs().will(returnValue(HCCL_SUCCESS));

    ge::AttrUtils::SetInt(ops[0].GetOpDesc(), "used_stream_num", streamNum);
    MOCKER_CPP(&HcomGraphOptimizer::GetOriginalGraphShapeTypeFromDesc).stubs().will(returnValue(HCCL_SUCCESS));

    u32 shapeType = ORIGINAL_GRAPH_UNKNOWNSHAPE_TYPE;

    MOCKER_CPP_VIRTUAL(hcomGraphOptimizer, &HcomGraphOptimizer::SetOpOutputMemSize)
        .stubs()
        .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&HcomGraphOptimizer::GetTaskNumAndCheckForceUnknown).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&HcomGraphOptimizer::SetOpAtomicInputIndex).stubs().will(returnValue(HCCL_SUCCESS));

    ge::AttrUtils::HasAttr(ops[0].GetOpDesc(), "DUMMY_SET_FALSE_COMM");

    MOCKER_CPP(&HcomGraphOptimizer::GetLookupUpdateWorkspace).stubs().will(returnValue(HCCL_SUCCESS));
    const string coll_type = HCCL_KERNEL_OP_TYPE_ALLREDUCE;
    ops[0].GetOpDesc()->SetType(coll_type);
    ret = hcomGraphOptimizer.HcomCalcOpRunningParam(ops[0], false);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
}

TEST_F(HcomGraphOptimizerTest, utFusionVersion)
{
    HcclResult ret;
    std::string socVersion;
    HcomOpUtils ops;
    ret = ops.CreateFusionConfigVersion(socVersion);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}
#if 1
TEST_F(HcomGraphOptimizerTest, utFusionPath_Tune)
{
    HcclResult ret;
    HcomOpUtils ops;
    std::string fusionPath;
    std::string fusionFile;
    char* env = nullptr;

    ret = ops.GetPathFromEnv(env, fusionPath);
    EXPECT_EQ(ret, HCCL_E_AGAIN);

    env = "/tmp";
    ret = ops.GetPathFromEnv(env, fusionPath);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = ops.GetFileNameFromPath(fusionPath, fusionFile);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    env = "./test";
    ret = ops.GetPathFromEnv(env, fusionPath);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = ops.GetFileNameFromPath(fusionPath, fusionFile);
    EXPECT_EQ(ret, HCCL_E_AGAIN);
}
#endif

TEST_F(HcomGraphOptimizerTest, utGetDeviceAndServerNum)
{
    HcclResult ret;
    HcomOpUtils ops;

    ge::Node node;
    ge::AttrUtils::HasAttr(node.GetOpDesc(), "DUMMY_SET_TRUE_MAXNUM");
    ge::AttrUtils::HasAttr(node.GetOpDesc(), "DUMMY_SET_TRUE_EMBEDDINGDIM");
    ge::AttrUtils::HasAttr(node.GetOpDesc(), "DUMMY_SET_TRUE_FLAGS");

    ge::AttrUtils::SetInt(node.GetOpDesc(), "flags", 1);

    std::string curGroup = "aa";
    ge::AttrUtils::SetStr(node.GetOpDesc(), "group", curGroup);
    s32 deviceNumPerServer = 2;
    s32 serverNum = 2;
    bool multiModuleDiffDeviceNumMode = false;

    MOCKER(GetClusterInfoAndDeviceNum).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER(IsOfflineCompilation).stubs().will(returnValue(true));

    ret = ops.GetDeviceAndServerNum(node, deviceNumPerServer, serverNum, multiModuleDiffDeviceNumMode);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
}

TEST_F(HcomGraphOptimizerTest, utCalculateSegmentIndex)
{
    nlohmann::json rank_table
        = {{"status", "completed"},
           {"version", "1.0"},
           {"server_count", "1"},
           {"server_list",
            {{
                {"server_id", "10.0.0.10"},
                {"host_nic_ip", "192.168.0.12:0,192.168.1.12:199"},
                {"device",
                 {
                     {{"rank_id", "0"}, {"device_id", "0"}, {"device_ip", "192.168.0.12,192.168.1.12"}

                     },
                 }},
            }}}};

    char file_name_t[] = "./ut_hcom_get_new_rank_info_muti_ip.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(1) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    } else {
        HCCL_ERROR("open %s failed", file_name_t);
    }

    outfile.close();
    int ret = HCCL_SUCCESS;

    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    u32 localrankid = 0;
    u32 localranksize = 0;
    char* rank_table_file = "./ut_hcom_get_new_rank_info_muti_ip.json";
    char* rank_ID = "0";

    ret = HcomInitByFile(rank_table_file, rank_ID);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    u64 tensorFusionLimit = std::numeric_limits<u64>::max();
    std::string fusionHash = "001";
    std::vector<u32> segmentIndex;
    std::string fusionPath = "./Ascend910A_gradient_fusion.json";
    nlohmann::json root;
    root[0]["modelhash"] = "001";
    root[0]["modelgraphid"] = 1;
    root[0]["modelvalue"]["gradientSize"] = {16384000, 4000, 67108864, 16384, 150994944, 16384, 2359296, 1024,
                                             3538944,  1024, 2654208,  1536,  1843200,   768,   139392,  384};
    root[0]["modelvalue"]["gradientTime"] = {9833418590329, 15080,  100303, 86064,   408791, 246415,  634287, 686785,
                                             103,           564542, 100,    1686982, 117497, 3490117, 202901, 117711};
    fstream jFile;
    jFile.open(fusionPath, std::ios::out | std::ios::trunc);
    jFile.close();

    HcomAllReduceFusion ops;
    ret = ops.GetInformationFromLibrary(fusionPath, fusionHash, tensorFusionLimit, segmentIndex);
    EXPECT_EQ(ret, HCCL_E_AGAIN);

    jFile.open(fusionPath, std::ios::out);
    jFile << root;
    jFile.close();

    ret = HcomDestroy();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    remove(file_name_t);
    remove(fusionPath.c_str());
    GlobalMockObject::verify();
}

extern vector<float> CalculateSizeRatio(const vector<float>& sliceSize, float totalSize);
TEST_F(HcomGraphOptimizerTest, utCalculateSizeRatio)
{
    float totalSize = 100.00;
    vector<float> sliceSize{41.00, 40.00, 9.00, 10.00};
    vector<float> ratio;
    ratio = CalculateSizeRatio(sliceSize, totalSize);
}

TEST_F(HcomGraphOptimizerTest, utCalculateSegmentIndexFromHomeExport)
{
    HcclResult ret;
    float tensorFusionLimit = 3.40282e+38;
    std::string fusionHash = "4569785153";
    std::string fusionSocVersion = "Ascend910A";
    std::string fusionStartPath = "/usr/local/ascend/compiler/lib64/";
    std::vector<u32> segmentIndex;
    std::vector<u32> index{0};
    HcomAllReduceFusion ops;

    MOCKER(HcomOpUtils::CreateFusionConfigVersion)
        .stubs()
        .with(outBound(fusionSocVersion))
        .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&HcomAllReduceFusion::GetPathFromDefault).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&HcomAllReduceFusion::GetInformationFromLibrary)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    ret = ops.CalculateSegmentIndex(fusionHash, tensorFusionLimit, segmentIndex);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
}

ge::graphStatus
FakeGetOption1(ge::GEThreadLocalContext* that, const std::string& optionExec, std::string& dumpDebugValue)
{
    nlohmann::json group_list = {{{"group_name", "aa"}, {"group_rank_list", {0, 1}}}};
    if (optionExec == ge::OPTION_EXEC_HCOM_GROUPLIST) {
        dumpDebugValue = group_list.dump();
    }
    return ge::GRAPH_SUCCESS;
}

TEST_F(HcomGraphOptimizerTest, ut_offlinebuild_calcSubStreamNum)
{
    ge::Status ret;
    HcomGraphOptimizer graphOptimizer;
    ge::NodePtr nodeptr(new NodeTest);
    std::string type = HCCL_KERNEL_OP_TYPE_ALLTOALLV;
    nodeptr->GetOpDesc()->SetType(type);

    std::string curGroup = "aa";
    ge::AttrUtils::SetStr(nodeptr->GetOpDesc(), "group", curGroup);

    MOCKER_CPP(&ge::GEThreadLocalContext::GetOption).stubs().will(invoke(FakeGetOption1));

    MOCKER(&ge::AttrUtils::SetInt).stubs().will(returnValue(false));

    ret = graphOptimizer.HcomCalcOpRunningParam(*nodeptr, false);

    MOCKER_CPP(&HcomGraphOptimizer::CalAndSetOpWorkerSpaceForKnowShape)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    type = HCCL_KERNEL_OP_TYPE_BROADCAST;
    std::string nodeName = "ALL_GATHER_NO_CALCULATION";
    nodeptr->GetOpDesc()->SetType(type);
    nodeptr->GetOpDesc()->SetName(nodeName);
    ret = graphOptimizer.HcomCalcOpRunningParam(*nodeptr, false);

    GlobalMockObject::verify();
}

struct OpInfo {
    ge::NodePtr nodePtr;
    std::pair<std::string, std::string> opName;
    std::vector<std::pair<std::string, int64_t>> attrInt;
    std::vector<std::pair<std::string, std::string>> attrStr;
};

HcclResult CreateNodePtr(OpInfo& opInfo, ge::ComputeGraph& graph)
{
    ge::OpDescPtr opDescPtr = std::make_shared<ge::OpDesc>(opInfo.opName.first.c_str(), opInfo.opName.second.c_str());
    EXPECT_NE(opDescPtr, nullptr);
    opDescPtr->SetName(opInfo.opName.first.c_str());

    for (auto& it : opInfo.attrInt) {
        bool bErr = ge::AttrUtils::SetInt(opDescPtr, it.first.c_str(), it.second);
        CHK_PRT_RET(
            !bErr, HCCL_ERROR("node[%s] set attr: %s failed", opInfo.opName.first.c_str(), it.first.c_str()),
            HCCL_E_INTERNAL);
    }
    for (auto& it : opInfo.attrStr) {
        bool bErr = ge::AttrUtils::SetStr(opDescPtr, it.first.c_str(), it.second.c_str());
        CHK_PRT_RET(
            !bErr, HCCL_ERROR("node[%s] set attr: %s failed", opInfo.opName.first.c_str(), it.first.c_str()),
            HCCL_E_INTERNAL);
    }

    opInfo.nodePtr = graph.AddNode(opDescPtr);
    CHK_PRT_RET(
        (opInfo.nodePtr == nullptr), HCCL_ERROR("[Create]node[%s] failed", opInfo.opName.first.c_str()),
        HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

TEST_F(HcomGraphOptimizerTest, ut_FuseHcomAlltoAllVCNode)
{
    MOCKER_CPP(&HcomAlltoAllVCFusion::RunFusionOpsAlltoAllVC).stubs().will(returnValue(HCCL_SUCCESS));

    HcclResult ret;
    ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test_graph");

    u32 alltoallvcNum = 2;
    for (u32 i = 0; i < alltoallvcNum; i++) {
        // alltoallvc
        OpInfo alltoallvcOpInfo;
        alltoallvcOpInfo.opName = std::make_pair("AlltoAllVC_" + std::to_string(i), HCCL_KERNEL_OP_TYPE_ALLTOALLVC);
        alltoallvcOpInfo.attrInt.push_back(std::make_pair("rank", 0));
        alltoallvcOpInfo.attrInt.push_back(std::make_pair("fusion", 2));
        alltoallvcOpInfo.attrInt.push_back(std::make_pair("fusion_id", 1));
        alltoallvcOpInfo.attrStr.push_back(std::make_pair("group", HCCL_WORLD_GROUP));
        ret = CreateNodePtr(alltoallvcOpInfo, *graph);
        EXPECT_EQ(ret, HCCL_SUCCESS);

        ge::AttrUtils::HasAttr(alltoallvcOpInfo.nodePtr->GetOpDesc(), "DUMMY_SET_TRUE_GROUP");
        alltoallvcOpInfo.nodePtr->GetOpDesc()->SetType(HCCL_KERNEL_OP_TYPE_ALLTOALLVC);
    }
    HcomFusionOptimizer graphOptimizer;
    ge::Status ge_ret = graphOptimizer.FuseHcomAlltoAllVCNode(*graph);
    EXPECT_EQ(ge_ret, ge::SUCCESS);
    GlobalMockObject::verify();
}

TEST_F(HcomGraphOptimizerTest, ut_RunFusionOpsAlltoAllVC)
{
    MOCKER_CPP(&HcomAlltoAllVCFusion::AddOpsEdge).stubs().will(returnValue(HCCL_SUCCESS));

    HcclResult ret;
    ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test_graph");

    HcomAlltoAllVCFusion fusionHcomAlltoAllVCOp;
    constexpr u32 fusionOpNum = 2;
    std::vector<ge::NodePtr> fusionOps(fusionOpNum);
    std::vector<ge::NodePtr> InOps(fusionOpNum * 2);
    std::vector<ge::NodePtr> OutOps(fusionOpNum);

    for (u32 i = 0; i < fusionOps.size(); ++i) {
        fusionOps[i] = std::make_shared<NodeTest>();
        fusionOps[i]->Init();
        fusionOps[i]->Init(); // 多调用一次Init, 让IndataAnchor的size + 1
    }
    for (u32 i = 0; i < InOps.size(); ++i) {
        InOps[i] = std::make_shared<NodeTest>();
        InOps[i]->Init();
    }
    for (u32 i = 0; i < OutOps.size(); ++i) {
        OutOps[i] = std::make_shared<NodeTest>();
        OutOps[i]->Init();
    }

    // link
    for (u32 i = 0; i < fusionOps.size(); ++i) {
        InOps[i * 2]->GetOutDataAnchor(0)->LinkTo(fusionOps[i]->GetInDataAnchor(0));
        InOps[i * 2 + 1]->GetOutDataAnchor(0)->LinkTo(fusionOps[i]->GetInDataAnchor(1));
        fusionOps[i]->GetOutDataAnchor(0)->LinkTo(OutOps[i]->GetInDataAnchor(0));
    }

    ret = fusionHcomAlltoAllVCOp.RunFusionOpsAlltoAllVC(*graph, fusionOps);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
}

TEST_F(HcomGraphOptimizerTest, ut_FuseHcomAllGatherNode)
{
    MOCKER_CPP(&HcomAllGatherFusion::RunFusionOpsAllGather).stubs().will(returnValue(HCCL_SUCCESS));

    HcclResult ret;
    ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test_graph");

    u32 allgatherNum = 2;
    for (u32 i = 0; i < allgatherNum; i++) {
        // allgather
        OpInfo allgatherOpInfo;
        allgatherOpInfo.opName = std::make_pair("AllGather_" + std::to_string(i), HCCL_KERNEL_OP_TYPE_ALLGATHER);
        allgatherOpInfo.attrInt.push_back(std::make_pair("rank_size", 3));
        allgatherOpInfo.attrInt.push_back(std::make_pair("fusion", 2));
        allgatherOpInfo.attrInt.push_back(std::make_pair("fusion_id", 1));
        allgatherOpInfo.attrStr.push_back(std::make_pair("group", HCCL_WORLD_GROUP));
        ret = CreateNodePtr(allgatherOpInfo, *graph);
        EXPECT_EQ(ret, HCCL_SUCCESS);

        ge::AttrUtils::HasAttr(allgatherOpInfo.nodePtr->GetOpDesc(), "DUMMY_SET_TRUE_GROUP");
        allgatherOpInfo.nodePtr->GetOpDesc()->SetType(HCCL_KERNEL_OP_TYPE_ALLGATHER);
    }
    HcomFusionOptimizer graphOptimizer;
    ret = graphOptimizer.HcomOptimizeOriginalGraph(*graph);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ge::Status ge_ret = graphOptimizer.FuseHcomAllgatherNode(*graph);
    EXPECT_EQ(ge_ret, ge::SUCCESS);
    ge_ret = graphOptimizer.HcomOptimizeSetAttr(*graph);
    EXPECT_EQ(ge_ret, ge::SUCCESS);
    GlobalMockObject::verify();
}

TEST_F(HcomGraphOptimizerTest, ut_RunFusionOpsAllGather)
{
    MOCKER_CPP(&HcomAllGatherFusion::AddOpsEdge).stubs().will(returnValue(HCCL_SUCCESS));

    HcclResult ret;
    ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test_graph");

    HcomAllGatherFusion fusionHcomAllGatherOp;
    constexpr u32 fusionOpNum = 2;
    std::vector<ge::NodePtr> fusionOps(fusionOpNum);
    std::vector<ge::NodePtr> InOps(fusionOpNum);
    std::vector<ge::NodePtr> OutOps(fusionOpNum);

    for (u32 i = 0; i < fusionOps.size(); ++i) {
        fusionOps[i] = std::make_shared<NodeTest>();
        fusionOps[i]->Init();
    }
    for (u32 i = 0; i < InOps.size(); ++i) {
        InOps[i] = std::make_shared<NodeTest>();
        InOps[i]->Init();
    }
    for (u32 i = 0; i < OutOps.size(); ++i) {
        OutOps[i] = std::make_shared<NodeTest>();
        OutOps[i]->Init();
    }

    // link
    for (u32 i = 0; i < fusionOps.size(); ++i) {
        InOps[i]->GetOutDataAnchor(0)->LinkTo(fusionOps[i]->GetInDataAnchor(0));
        InOps[i]->GetOutControlAnchor()->LinkTo(fusionOps[i]->GetInControlAnchor());
        fusionOps[i]->GetOutDataAnchor(0)->LinkTo(OutOps[i]->GetInDataAnchor(0));
        fusionOps[i]->GetOutControlAnchor()->LinkTo(OutOps[i]->GetInControlAnchor());
    }

    ret = fusionHcomAllGatherOp.RunFusionOpsAllGather(*graph, fusionOps);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
}

TEST_F(HcomGraphOptimizerTest, ut_FuseHcomReduceScatterNode)
{
    MOCKER_CPP(&HcomReduceScatterFusion::RunFusionOpsReduceScatter).stubs().will(returnValue(HCCL_SUCCESS));

    HcclResult ret;
    ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test_graph");

    u32 reducescatterNum = 2;
    for (u32 i = 0; i < reducescatterNum; i++) {
        // reducescatter
        OpInfo reducescatterOpInfo;
        reducescatterOpInfo.opName
            = std::make_pair("ReduceScatter_" + std::to_string(i), HCCL_KERNEL_OP_TYPE_REDUCESCATTER);
        reducescatterOpInfo.attrStr.push_back(std::make_pair("reduction", "sum"));
        reducescatterOpInfo.attrInt.push_back(std::make_pair("rank_size", 2));
        reducescatterOpInfo.attrInt.push_back(std::make_pair("fusion", 2));
        reducescatterOpInfo.attrInt.push_back(std::make_pair("fusion_id", 1));
        reducescatterOpInfo.attrStr.push_back(std::make_pair("group", HCCL_WORLD_GROUP));
        ret = CreateNodePtr(reducescatterOpInfo, *graph);
        EXPECT_EQ(ret, HCCL_SUCCESS);

        ge::AttrUtils::HasAttr(reducescatterOpInfo.nodePtr->GetOpDesc(), "DUMMY_SET_TRUE_GROUP");
        reducescatterOpInfo.nodePtr->GetOpDesc()->SetType(HCCL_KERNEL_OP_TYPE_REDUCESCATTER);
    }
    HcomFusionOptimizer graphOptimizer;
    ge::Status ge_ret = graphOptimizer.FuseHcomReduceScatterNode(*graph);
    EXPECT_EQ(ge_ret, ge::SUCCESS);
    ret = graphOptimizer.HcomOptimizeSetAttr(*graph);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
}

TEST_F(HcomGraphOptimizerTest, ut_FuseHcomReduceScatterNode1)
{
    MOCKER_CPP(&HcomReduceScatterFusion::RunFusionOpsReduceScatter).stubs().will(returnValue(HCCL_SUCCESS));

    HcclResult ret;
    ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test_graph");

    u32 reducescatterNum = 2;
    for (u32 i = 0; i < reducescatterNum; i++) {
        // reducescatter
        OpInfo reducescatterOpInfo;
        reducescatterOpInfo.opName
            = std::make_pair("ReduceScatter_" + std::to_string(i), HCCL_KERNEL_OP_TYPE_REDUCESCATTER);
        reducescatterOpInfo.attrStr.push_back(std::make_pair("reduction", "sum"));
        reducescatterOpInfo.attrInt.push_back(std::make_pair("rank_size", 2));
        reducescatterOpInfo.attrInt.push_back(std::make_pair("fusion", 2));
        reducescatterOpInfo.attrInt.push_back(std::make_pair("fusion_id", 1));
        reducescatterOpInfo.attrStr.push_back(std::make_pair("group", HCCL_WORLD_GROUP));
        ret = CreateNodePtr(reducescatterOpInfo, *graph);
        EXPECT_EQ(ret, HCCL_SUCCESS);

        ge::AttrUtils::HasAttr(reducescatterOpInfo.nodePtr->GetOpDesc(), "DUMMY_SET_TRUE_GROUP");
        reducescatterOpInfo.nodePtr->GetOpDesc()->SetType(HCCL_KERNEL_OP_TYPE_REDUCESCATTER);
    }

    MOCKER(HcomGetCCLBufferAvailableSize).stubs().with(mockcpp::any()).will(invoke(stub_HcomGetCCLBufferAvailableSize));

    MOCKER(&HcomOpUtils::GetAllInputsTensorMemSize).stubs().will(invoke(stub_GetAllInputsTensorMemSize));

    HcomFusionOptimizer graphOptimizer;
    ge::Status ge_ret = graphOptimizer.FuseHcomReduceScatterNode(*graph);
    EXPECT_EQ(ge_ret, ge::SUCCESS);
    GlobalMockObject::verify();
}

TEST_F(HcomGraphOptimizerTest, ut_FuseHcomReduceNode1)
{
    MOCKER_CPP(&HcomReduceScatterFusion::RunFusionOpsReduce).stubs().will(returnValue(HCCL_SUCCESS));

    HcclResult ret;
    ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test_graph");

    u32 reducescatterNum = 2;
    for (u32 i = 0; i < reducescatterNum; i++) {
        OpInfo reducescatterOpInfo;
        reducescatterOpInfo.opName = std::make_pair("Reduce_" + std::to_string(i), HCCL_KERNEL_OP_TYPE_REDUCE);
        reducescatterOpInfo.attrStr.push_back(std::make_pair("reduction", "sum"));
        reducescatterOpInfo.attrInt.push_back(std::make_pair("rank_size", 2));
        reducescatterOpInfo.attrInt.push_back(std::make_pair("fusion", 2));
        reducescatterOpInfo.attrInt.push_back(std::make_pair("fusion_id", 1));
        reducescatterOpInfo.attrStr.push_back(std::make_pair("group", HCCL_WORLD_GROUP));
        ret = CreateNodePtr(reducescatterOpInfo, *graph);
        EXPECT_EQ(ret, HCCL_SUCCESS);

        ge::AttrUtils::HasAttr(reducescatterOpInfo.nodePtr->GetOpDesc(), "DUMMY_SET_TRUE_GROUP");
        reducescatterOpInfo.nodePtr->GetOpDesc()->SetType(HCCL_KERNEL_OP_TYPE_REDUCE);
    }

    HcomGraphOptimizer graphOptimizer;
    ge::Status ge_ret = graphOptimizer.FuseHcomReduceNode(*graph);
    EXPECT_EQ(ge_ret, ge::SUCCESS);
    GlobalMockObject::verify();
}

TEST_F(HcomGraphOptimizerTest, ut_FuseHcomReduceNode2)
{
    MOCKER_CPP(&HcomReduceScatterFusion::RunFusionOpsReduce).stubs().will(returnValue(HCCL_SUCCESS));

    HcclResult ret;
    ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test_graph");

    u32 reducescatterNum = 2;
    for (u32 i = 0; i < reducescatterNum; i++) {
        OpInfo reducescatterOpInfo;
        reducescatterOpInfo.opName = std::make_pair("AllReduce_" + std::to_string(i), HCCL_KERNEL_OP_TYPE_ALLREDUCE);
        reducescatterOpInfo.attrStr.push_back(std::make_pair("reduction", "sum"));
        reducescatterOpInfo.attrInt.push_back(std::make_pair("rank_size", 2));
        reducescatterOpInfo.attrInt.push_back(std::make_pair("fusion", 2));
        reducescatterOpInfo.attrInt.push_back(std::make_pair("fusion_id", 1));
        reducescatterOpInfo.attrStr.push_back(std::make_pair("group", HCCL_WORLD_GROUP));
        ret = CreateNodePtr(reducescatterOpInfo, *graph);
        EXPECT_EQ(ret, HCCL_SUCCESS);

        ge::AttrUtils::HasAttr(reducescatterOpInfo.nodePtr->GetOpDesc(), "DUMMY_SET_TRUE_GROUP");
        reducescatterOpInfo.nodePtr->GetOpDesc()->SetType(HCCL_KERNEL_OP_TYPE_ALLREDUCE);
    }
    HcomGraphOptimizer graphOptimizer;
    ge::Status ge_ret = graphOptimizer.FuseHcomAllReduceNode(*graph);
    EXPECT_EQ(ge_ret, ge::SUCCESS);
    GlobalMockObject::verify();
}

TEST_F(HcomGraphOptimizerTest, ut_RunFusionReduceScatter)
{
    MOCKER_CPP(&HcomReduceScatterFusion::AddOpsEdge).stubs().will(returnValue(HCCL_SUCCESS));

    HcclResult ret;
    ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test_graph");

    HcomReduceScatterFusion fusionHcomReduceScatterOp;
    constexpr u32 fusionOpNum = 2;
    std::vector<ge::NodePtr> fusionOps(fusionOpNum);
    std::vector<ge::NodePtr> InOps(fusionOpNum);
    std::vector<ge::NodePtr> OutOps(fusionOpNum);

    for (u32 i = 0; i < fusionOps.size(); ++i) {
        fusionOps[i] = std::make_shared<NodeTest>();
        fusionOps[i]->Init();
    }
    for (u32 i = 0; i < InOps.size(); ++i) {
        InOps[i] = std::make_shared<NodeTest>();
        InOps[i]->Init();
    }
    for (u32 i = 0; i < OutOps.size(); ++i) {
        OutOps[i] = std::make_shared<NodeTest>();
        OutOps[i]->Init();
    }

    // link
    for (u32 i = 0; i < fusionOps.size(); ++i) {
        InOps[i]->GetOutDataAnchor(0)->LinkTo(fusionOps[i]->GetInDataAnchor(0));
        InOps[i]->GetOutControlAnchor()->LinkTo(fusionOps[i]->GetInControlAnchor());
        fusionOps[i]->GetOutDataAnchor(0)->LinkTo(OutOps[i]->GetInDataAnchor(0));
        fusionOps[i]->GetOutControlAnchor()->LinkTo(OutOps[i]->GetInControlAnchor());
    }

    ret = fusionHcomReduceScatterOp.RunFusionOpsReduceScatter(*graph, fusionOps);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
}

HcclResult GetReduceScatterVCountsStub(const ge::Node& node, std::vector<int64_t>& sendCounts)
{
    sendCounts.push_back(1);
    sendCounts.push_back(2);
    sendCounts.push_back(3);
    sendCounts.push_back(4);
    return HCCL_SUCCESS;
}

HcclResult GetDeviceTypeA2Stub(const char* group, DevType& deviceType)
{
    deviceType = DevType::DEV_TYPE_910B;
    return HCCL_SUCCESS;
}

HcclResult
stub_GetOpWorkspaceMemSize(HcomGraphOptimizer* that, ge::Node& node, const std::string& sCollectiveType, u64& opMemSize)
{
    opMemSize = 2048 * 1024 * 1024;
    return HCCL_SUCCESS;
}

ge::graphStatus GetOption2(ge::GEContext* that, const std::string& optionExec, std::string& dumpDebugValue)
{
    dumpDebugValue = "1";
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GetOption3(ge::GEContext* that, const std::string& optionExec, std::string& dumpDebugValue)
{
    dumpDebugValue = "2";
    return ge::GRAPH_SUCCESS;
}

TEST_F(HcomGraphOptimizerTest, ut_SetknownShapAttr)
{
    HcclResult ret;
    int64_t memSize = GetExternalInputCCLBuffSize() + 1;
    HcomGraphOptimizer graphOptimizer;
    graphOptimizer.optionFeatureBaseRefreshable_ = 1;
    ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test_graph");
    const string type = "HcomAllToAllVC";
    ge::OpDescPtr opPtr_ = std::make_shared<ge::OpDesc>();
    opPtr_->SetType(type);
    graph->AddNode(opPtr_);

    MOCKER(&ge::NodeUtils::GetNodeUnknownShapeStatus)
        .stubs()
        .with(mockcpp::any(), outBound(false))
        .will(returnValue(ge::GRAPH_SUCCESS));

    MOCKER(&HcomOpUtils::GetAllInputsTensorMemSize).stubs().will(invoke(stub_GetAllInputsTensorMemSize));

    MOCKER(
        &HcomGraphOptimizer::GetOpWorkspaceMemSize,
        HcclResult(HcomGraphOptimizer::*)(ge::Node & node, const std::string& sCollectiveType, u64& opMemSize))
        .stubs()
        .will(invoke(stub_GetOpWorkspaceMemSize));

    MOCKER_CPP(&ge::GEContext::GetOption).stubs().will(invoke(GetOption2));

    ret = graphOptimizer.SetUnknownShapeAttr(*graph, true);

    MOCKER_CPP(&ge::GEContext::GetOption).stubs().will(invoke(GetOption3));
    ret = graphOptimizer.SetUnknownShapeAttr(*graph, true);
    GlobalMockObject::verify();
}

HcclResult stub_GetVectorFromTensorGraphOptimizer(const ge::GeTensor* tensor, std::vector<int64_t>& vector)
{
    vector.resize(4 * 4);
    return HCCL_SUCCESS;
}
const std::vector<HcclAlgoType> GetExternalInputHcclAlgoConfig_stub()
{
    static std::vector<HcclAlgoType> hcclAlgoConfig(4, HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT);
    hcclAlgoConfig[0] = HcclAlgoType::HCCL_ALGO_TYPE_FULLMESH;
    hcclAlgoConfig[1] = HcclAlgoType::HCCL_ALGO_TYPE_FULLMESH;
    hcclAlgoConfig[2] = HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT;
    hcclAlgoConfig[3] = HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT;
    return hcclAlgoConfig;
}

TEST_F(HcomGraphOptimizerTest, ut_getAlltoAllvcStagedScratchMemSize)
{
    ge::NodePtr nodeptr(new NodeTest);
    HcomOpUtils hcomKernelInfo;
    HcomGraphOptimizer KernelInfo;

    u64 opMemSize = 0;
    u32 rankSize = 4;

    MOCKER(GetExternalInputHcclAlgoConfig)
        .stubs()
        .with(mockcpp::any())
        .will(invoke(GetExternalInputHcclAlgoConfig_stub));

    MOCKER(&HcomOpUtils::GetVectorFromTensor).stubs().will(invoke(stub_GetVectorFromTensorGraphOptimizer));

    std::vector<int64_t> sendCountMatrix(16, 1);
    ge::AttrUtils::SetListInt(nodeptr->GetOpDesc(), "send_count_matrix", sendCountMatrix);
    u32 rankId = 0;

    MOCKER(HcomGetRankId).stubs().with(mockcpp::any(), outBound(&rankId)).will(returnValue(HCCL_SUCCESS));

    MOCKER(HcomGetRankSize).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER(&HcomOpUtils::CheckAlltoAllvcRank).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER(HcomGetAlltoAllvcStagedWorkSpaceMemSize).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    HcclResult ret = KernelInfo.GetOpWorkspaceMemSize(*nodeptr, HCCL_KERNEL_OP_TYPE_ALLTOALLVC, opMemSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
}

TEST_F(HcomGraphOptimizerTest, ut_getAlltoAllCountsDispl_sendCountMatrix)
{
    ge::NodePtr nodeptr(new NodeTest);
    HcomOpUtils graphOptimizer;

    std::vector<int64_t> sendCountMatrix;

    HcclResult ret = graphOptimizer.GetAlltoAllCountMatrix(*(nodeptr.get()), sendCountMatrix);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

std::string optionExecTest = "Ascend910";
ge::graphStatus
TaskNumGetOption_1(ge::GEThreadLocalContext* that, const std::string& optionExec, std::string& dumpDebugValue)
{
    nlohmann::json group_list
        = {{{"group_name", "aa"}, {"group_rank_list", {0, 1}}},
           {{"group_name", "off_group_rank_list"}, {"group_rank_list", {0, 1, 2, 3, 4, 5, 6, 7}}}};
    if (optionExec == ge::OPTION_EXEC_HCOM_GROUPLIST) {
        dumpDebugValue = group_list.dump();
    } else if (optionExec == ge::OPTION_EXEC_HCOM_RANK_MAPPING) {
        dumpDebugValue = R"({"status": "completed","version": "1.1","node_list":[{"node_id": "0","rank_list":[
        {"rank_id": "0","item_id": "0","rank_ip":"192.168.2.10"},
        {"rank_id": "1","item_id": "-1","rank_ip":"192.168.2.11"}]}]})";
    } else if (optionExec == ge::OPTION_EXEC_RANK_TABLE) {
        dumpDebugValue = R"({"status": "completed","version": "1.1","node_list":[{"node_id": "0","rank_list":[
        {"rank_id": "0","item_id": "0","rank_ip":"192.168.2.10"},
        {"rank_id": "1","item_id": "0","rank_ip":"192.168.2.11"},
        {"rank_id": "2","item_id": "0","rank_ip":"192.168.2.11"},
        {"rank_id": "3","item_id": "0","rank_ip":"192.168.2.11"},
        {"rank_id": "4","item_id": "0","rank_ip":"192.168.2.11"},
        {"rank_id": "5","item_id": "0","rank_ip":"192.168.2.11"},
        {"rank_id": "6","item_id": "0","rank_ip":"192.168.2.11"},
        {"rank_id": "7","item_id": "0","rank_ip":"192.168.2.11"}]}]})";
    } else if (optionExec == "ge.socVersion") {
        dumpDebugValue = optionExecTest;
    } else if (optionExec == ge::OPTION_EXEC_RANK_TABLE_FILE) {
        dumpDebugValue = "./llt/ace/comop/hccl/stub/workspace/ut_task_num_one_server_hcom_test.json";
    } else if (optionExec == "ge.offline_hccl_compile") {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

TEST_F(HcomGraphOptimizerTest, ut_CalcOpTaskNum_1server_1)
{
    HcclResult ret;
    nlohmann::json rank_table = rank_table_1server_8rank;
    char file_name_t[] = "./llt/ace/comop/hccl/stub/workspace/ut_task_num_one_server_hcom_test.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(1) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    } else {
        HCCL_ERROR("open %s failed", file_name_t);
    }

    outfile.close();

    ge::OpDesc op;
    ge ::Status ge_ret = ge::INTERNAL_ERROR;
    HcomGraphOptimizer hcomKernelInfo;

    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    DevType deviceType = DevType::DEV_TYPE_910;

    MOCKER(hrtGetDeviceType).stubs().with(outBound(deviceType)).will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&ge::GEThreadLocalContext::GetOption).stubs().will(invoke(TaskNumGetOption_1));

    char* rank_table_file = "./llt/ace/comop/hccl/stub/workspace/ut_task_num_one_server_hcom_test.json";
    char* rank_ID = "0";

    HCCL_INFO("HcomInitByFile START.");
    ret = HcomInitByFile(rank_table_file, rank_ID);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    HCCL_INFO("HcomInitByFile OK. pid[%d]", SalGetPid());

    ge::NodePtr nodeptr(new NodeTest);
    int64_t RANK_SIZE = 4;
    ge::AttrUtils::SetInt(nodeptr->GetOpDesc(), "rank_size", RANK_SIZE);
    std::string tempStr = HCCL_WORLD_GROUP;
    ge::AttrUtils::SetStr(nodeptr->GetOpDesc(), "group", tempStr);

    std::string type;
    type = HCCL_KERNEL_OP_TYPE_ALLREDUCE;
    nodeptr->GetOpDesc()->SetType(type);
    std::string name = HCCL_KERNEL_OP_TYPE_ALLREDUCE + "1server";
    nodeptr->GetOpDesc()->SetName(name);
    ge_ret = hcomKernelInfo.CalcOpRunningParam(*nodeptr, true);
    EXPECT_EQ(ge_ret, ge::SUCCESS);

    type = HCCL_KERNEL_OP_TYPE_ALLGATHER;
    nodeptr->GetOpDesc()->SetType(type);
    ge_ret = hcomKernelInfo.CalcOpRunningParam(*nodeptr, true);
    EXPECT_EQ(ge_ret, ge::SUCCESS);

    type = HCCL_KERNEL_OP_TYPE_REDUCESCATTER;
    nodeptr->GetOpDesc()->SetType(type);
    ge_ret = hcomKernelInfo.CalcOpRunningParam(*nodeptr, true);
    EXPECT_EQ(ge_ret, ge::SUCCESS);

    type = HCCL_KERNEL_OP_TYPE_ALLREDUCE;
    nodeptr->GetOpDesc()->SetType(type);
    ge_ret = hcomKernelInfo.CalcOpRunningParam(*nodeptr, true);
    EXPECT_EQ(ge_ret, ge::SUCCESS);

    type = HCCL_KERNEL_OP_TYPE_ALLGATHER;
    nodeptr->GetOpDesc()->SetType(type);
    ge_ret = hcomKernelInfo.CalcOpRunningParam(*nodeptr, true);
    EXPECT_EQ(ge_ret, ge::SUCCESS);

    type = HCCL_KERNEL_OP_TYPE_REDUCESCATTER;
    nodeptr->GetOpDesc()->SetType(type);
    ge_ret = hcomKernelInfo.CalcOpRunningParam(*nodeptr, true);
    EXPECT_EQ(ge_ret, ge::SUCCESS);

    type = HCCL_KERNEL_OP_TYPE_SEND;
    nodeptr->GetOpDesc()->SetType(type);
    ge_ret = hcomKernelInfo.CalcOpRunningParam(*nodeptr, true);
    EXPECT_EQ(ge_ret, ge::SUCCESS);

    type = HCCL_KERNEL_OP_TYPE_BROADCAST;
    nodeptr->GetOpDesc()->SetType(type);
    ge_ret = hcomKernelInfo.CalcOpRunningParam(*nodeptr, true);
    EXPECT_EQ(ge_ret, ge::SUCCESS);

    optionExecTest = "Ascend910_9391";
    ge_ret = hcomKernelInfo.CalcOpRunningParam(*nodeptr, true);
    EXPECT_EQ(ge_ret, ge::SUCCESS);

    ret = HcomDestroy();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    remove(file_name_t);

    GlobalMockObject::verify();
}

TEST_F(HcomGraphOptimizerTest, ut_HcomOptimizeOriginalGraph)
{
    MOCKER_CPP(&HcomAllGatherFusion::RunFusionOpsAllGather).stubs().will(returnValue(HCCL_SUCCESS));

    HcclResult ret;
    ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test_graph");

    u32 allgatherNum = 2;
    for (u32 i = 0; i < allgatherNum; i++) {
        // allgather
        OpInfo allgatherOpInfo;
        allgatherOpInfo.opName = std::make_pair("AllGather_" + std::to_string(i), HCCL_KERNEL_OP_TYPE_ALLGATHER);
        allgatherOpInfo.attrInt.push_back(std::make_pair("rank_size", 3));
        allgatherOpInfo.attrInt.push_back(std::make_pair("fusion", 2));
        allgatherOpInfo.attrInt.push_back(std::make_pair("fusion_id", 1));
        allgatherOpInfo.attrStr.push_back(std::make_pair("group", HCCL_WORLD_GROUP));
        ret = CreateNodePtr(allgatherOpInfo, *graph);
        EXPECT_EQ(ret, HCCL_SUCCESS);

        ge::AttrUtils::HasAttr(allgatherOpInfo.nodePtr->GetOpDesc(), "DUMMY_SET_TRUE_GROUP");
        allgatherOpInfo.nodePtr->GetOpDesc()->SetType(HCCL_KERNEL_OP_TYPE_ALLGATHER);
    }
    HcomFusionOptimizer graphOptimizer;
    ge::Status ge_ret = graphOptimizer.HcomOptimizeOriginalGraph(*graph);
    EXPECT_EQ(ge_ret, ge::SUCCESS);
    ge_ret = graphOptimizer.OptimizeFusedGraph(*graph);
    EXPECT_EQ(ge_ret, ge::SUCCESS);
    ge_ret = graphOptimizer.OptimizeGraphPrepare(*graph);
    EXPECT_EQ(ge_ret, ge::SUCCESS);
    ge_ret = graphOptimizer.OptimizeOriginalGraph(*graph);
    EXPECT_EQ(ge_ret, ge::SUCCESS);
    ge_ret = graphOptimizer.OptimizeWholeGraph(*graph);
    EXPECT_EQ(ge_ret, ge::SUCCESS);
    ge::GraphOptimizerAttribute attrs;
    ge_ret = graphOptimizer.GetAttributes(attrs);
    EXPECT_EQ(ge_ret, ge::SUCCESS);
    ge_ret = graphOptimizer.Finalize();
    EXPECT_EQ(ge_ret, ge::SUCCESS);
    GlobalMockObject::verify();
}

TEST_F(HcomGraphOptimizerTest, ut_CalAndSetOpWorkerSpaceForKnowShape)
{
    u32 shapeType = ORIGINAL_GRAPH_KNOWNSHAPE_TYPE;
    MOCKER_CPP(&HcomGraphOptimizer::GetOriginalGraphShapeTypeFromDesc)
        .stubs()
        .with(mockcpp::any(), outBound(shapeType))
        .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&HcomGraphOptimizer::GetOpWorkspaceMemSize).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    ge::NodePtr nodeptr(new NodeTest);
    HcomOpUtils hcomKernelInfo;
    HcomGraphOptimizer KernelInfo;
    std::string sCollectiveType = "sCollectiveType";
    u64 opMemSize = 48000;

    KernelInfo.CalAndSetOpWorkerSpaceForKnowShape(*nodeptr, sCollectiveType, opMemSize);
    GlobalMockObject::verify();
}

TEST_F(HcomGraphOptimizerTest, ut_OptimizeOriginalGraphDynamicGraphNoSuperkernel)
{
    ge::Status ge_ret;
    MOCKER(&ge::NodeUtils::GetNodeUnknownShapeStatus)
        .stubs()
        .with(mockcpp::any(), outBound(true))
        .will(returnValue(ge::GRAPH_SUCCESS));

    HcomGraphOptimizer graphOptimizer;
    ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test_graph");
    const string type = "HcomAllReduce";
    ge::OpDescPtr opPtr_ = std::make_shared<ge::OpDesc>();
    opPtr_->SetType(type);
    graph->AddNode(opPtr_);
    SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB);
    ge_ret = graphOptimizer.OptimizeOriginalGraph(*graph);
    EXPECT_EQ(ge_ret, ge::SUCCESS);
    GlobalMockObject::verify();
}

TEST_F(HcomGraphOptimizerTest, ut_GetStreamNumOfflineComp_ErrorTest)
{
    HcclResult ret;
    DevType deviceType = DevType::DEV_TYPE_910_93;
    MOCKER(GetOffDeviceTypeWithoutDev).stubs().with(outBound(deviceType)).will(returnValue(HCCL_SUCCESS));

    std::string sCollectiveType = "TEST";
    std::string curGroup = "HCCL_WORLD_GROUP";
    u64 streamNum = 1;
    ret = GetStreamNumOfflineComp(sCollectiveType, curGroup, streamNum);
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);

    GlobalMockObject::verify();
}
