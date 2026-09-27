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
#include "hvd_graph_optimizer.h"
#include "hvd_ops_kernel_builder.h"
#include "graph/node.h"
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

#include "plugin_manager.h"
#include "external/ge/ge_api_types.h"  // ge对内options
#include "framework/common/ge_types.h" // ge对外options
#include "hccl/hcom.h"
#include "hccl/hcom_executor.h"
#include "ranktable/v80_rank_table.h"
#include <iostream>
#include <fstream>
#include "graph/utils/node_utils.h"
#include "hvd_all_reduce_fusion.h"
#include "hcom_ops_kernel_info_store.h"
#include "graph/debug/ge_attr_define.h"
#include "comm.h"

using namespace std;
using namespace hccl;

static nlohmann::json allreduce_topo_switch_connect
    = {{"topology type", "switch connection"},
       {"topology desc",
        {{{"node type", "TOR"},
          {"node name", "tor0"},
          {"link info",
           {{{"link id", "0"},
             {"local port name", "port0"},
             {"local ip address", "100.100.83.1"},
             {"opposite type", "SERVER"},
             {"opposite name", "server0"},
             {"opposite port name", "eth8"},
             {"opposite ip address", "100.100.83.178"}}}}}}}};

class HvdGraphOptimizerTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        nlohmann::json rank_table
            = {{"status", "completed"},
               {"deploy_mode", "lab"},
               {"device_num", "4"},
               {"server_num", "2"},
               {"boardType", "0"},
               {"para_plane_location", "device"},
               {"para_plane_nic_num", "2"},
               {"para_plane_nic_name", {"eth0", "eth1"}},
               {"instance_count", "4"},
               {"device_count", "4"},
               {"instance_list",
                {{{"pod_name", ""},
                  {"rank_id", "0"},
                  {"server_id", "10.0.0.10"},
                  {"devices", {{{"device_id", "1"}, {"device_ip", "192.168.0.12"}, {"ref_ip", "192.168.10.13"}}}}},
                 {{"pod_name", ""},
                  {"rank_id", "1"},
                  {"server_id", "10.0.0.10"},
                  {"devices", {{{"device_id", "0"}, {"device_ip", "192.168.1.12"}, {"ref_ip", "192.168.11.13"}}}}},
                 {{"pod_name", ""},
                  {"rank_id", "2"},
                  {"server_id", "10.0.0.11"},
                  {"devices", {{{"device_id", "0"}, {"device_ip", "192.168.0.14"}, {"ref_ip", "192.168.10.15"}}}}},
                 {{"pod_name", ""},
                  {"rank_id", "3"},
                  {"server_id", "10.0.0.11"},
                  {"devices", {{{"device_id", "1"}, {"device_ip", "192.168.1.14"}, {"ref_ip", "192.168.11.15"}}}}}}},
               {"server_list",
                {
                    {{"server_id", "192.168.10.2"},
                     {"para_plane_info",
                      {{{"eth1", "192.168.210.2"}, {"ref_ip", "192.168.210.1"}},
                       {{"eth0", "192.168.200.2"}, {"ref_ip", "192.168.200.1"}}}}

                    },
                    {{"server_id", "192.168.10.3"},
                     {"para_plane_info",
                      {{{"eth0", "192.168.200.3"}, {"ref_ip", "192.168.200.1"}},
                       {{"eth1", "192.168.210.3"}, {"ref_ip", "192.168.210.1"}}}}

                    },

                }}};
        char file_name[] = "./ut_HvdGraphOptimizerTest.json";

        std::ofstream outfile(file_name, std::ios::out | std::ios::trunc | std::ios::binary);

        if (outfile.is_open()) {
            HCCL_INFO("open %s success", file_name);
        } else {
            HCCL_ERROR("open %s failed", file_name);
        }

        outfile << std::setw(4) << rank_table << std::endl;
        outfile.close();

        std::cout << "\033[36m--HvdGraphOptimizerTest SetUP--\033[0m" << std::endl;
    }
    static void TearDownTestCase()
    {
        char file_name[] = "./ut_HvdGraphOptimizerTest.json";
        remove(file_name);
        std::cout << "\033[36m--HvdGraphOptimizerTest TearDown--\033[0m" << std::endl;
    }
    // Some expensive resource shared by all tests.
    virtual void SetUp()
    {
        s32 portNum = 7;
        MOCKER(hrtGetHccsPortNum).stubs().with(mockcpp::any(), outBound(portNum)).will(returnValue(HCCL_SUCCESS));
        std::cout << "A Test SetUP" << std::endl;
    }
    virtual void TearDown() { std::cout << "A Test TearDown" << std::endl; }
};

class NodeTest : public ge::Node {
public:
    NodeTest() { ; };
    ~NodeTest() { ; };
};

TEST_F(HvdGraphOptimizerTest, ut_Finalize)
{
    HvdGraphOptimizer optimizer;
    HcclResult ret;
    ge ::Status ge_ret = ge::INTERNAL_ERROR;
    ge_ret = optimizer.Finalize();
    EXPECT_EQ(ge_ret, ge::SUCCESS);
}

TEST_F(HvdGraphOptimizerTest, ut_OptimizeOriginalGraph)
{
    HvdGraphOptimizer optimizer;
    HcclResult ret;
    ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test_graph");
    const string type = "HorovodBroadcast";
    ge::OpDescPtr opPtr_ = std::make_shared<ge::OpDesc>();
    opPtr_->SetType(type);
    graph->AddNode(opPtr_);

    ge ::Status ge_ret = ge::INTERNAL_ERROR;
    ge_ret = optimizer.OptimizeOriginalGraph(*graph);
    EXPECT_EQ(ge_ret, ge::SUCCESS);
}

TEST_F(HvdGraphOptimizerTest, ut_OptimizeGraphPrepare)
{
    HvdGraphOptimizer optimizer;
    HcclResult ret;
    ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test_graph");
    const string type = "HorovodBroadcast";
    ge::OpDescPtr opPtr_ = std::make_shared<ge::OpDesc>();
    opPtr_->SetType(type);
    graph->AddNode(opPtr_);

    ge ::Status ge_ret = ge::INTERNAL_ERROR;
    ge_ret = optimizer.OptimizeGraphPrepare(*graph);
    EXPECT_EQ(ge_ret, ge::SUCCESS);
}

TEST_F(HvdGraphOptimizerTest, ut_AssignNewStream)
{
    u32 opCount = 0;
    ge::NodePtr nodeptr(new NodeTest);
    std::string waitStream = "";
    std::string sendStream = "";

    HvdGraphOptimizer optimizer;
    HcclResult ret;
    ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test_graph");
    const string type = "HorovodBroadcast";
    ge::OpDescPtr opPtr_ = std::make_shared<ge::OpDesc>();
    opPtr_->SetType(type);
    graph->AddNode(opPtr_);

    ge ::Status ge_ret = ge::INTERNAL_ERROR;
    ge_ret = optimizer.AssignNewStream(opCount, nodeptr, waitStream, sendStream);
    EXPECT_EQ(ge_ret, ge::SUCCESS);
}

TEST_F(HvdGraphOptimizerTest, ut_AddControlAnchor)
{
    ge::NodePtr nodeptr(new NodeTest);
    ge::NodePtr addedNodePtr(new NodeTest);

    HvdGraphOptimizer optimizer;
    HcclResult ret;

    ge ::Status ge_ret = ge::INTERNAL_ERROR;
    ge_ret = optimizer.AddControlAnchor(nodeptr, addedNodePtr);
    EXPECT_EQ(ge_ret, ge::SUCCESS);
}

TEST_F(HvdGraphOptimizerTest, ut_AddDependence)
{
    HvdGraphOptimizer optimizer;
    ge::NodePtr nodeptr(new NodeTest);
    ge::NodePtr addedNodePtr(new NodeTest);
    std::vector<ge::NodePtr> preDependNodes;
    preDependNodes.push_back(nodeptr);
    preDependNodes.push_back(addedNodePtr);

    ge ::Status ge_ret = ge::INTERNAL_ERROR;
    ge_ret = optimizer.AddDependence(preDependNodes);
    EXPECT_EQ(ge_ret, 2);
}

TEST_F(HvdGraphOptimizerTest, ut_OptimizeWholeGraph)
{
    HvdGraphOptimizer optimizer;
    ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test_graph");
    const string type = "HorovodBroadcast";
    ge::OpDescPtr opPtr_ = std::make_shared<ge::OpDesc>();
    opPtr_->SetType(type);
    graph->AddNode(opPtr_);
    ge ::Status ge_ret = ge::INTERNAL_ERROR;
    ge_ret = optimizer.OptimizeWholeGraph(*graph);
    EXPECT_EQ(ge_ret, ge::SUCCESS);
}
TEST_F(HvdGraphOptimizerTest, ut_GetAttributes)
{
    HvdGraphOptimizer optimizer;
    ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test_graph");
    const string type = "HorovodBroadcast";
    ge::OpDescPtr opPtr_ = std::make_shared<ge::OpDesc>();
    opPtr_->SetType(type);
    graph->AddNode(opPtr_);
    ge ::Status ge_ret = ge::INTERNAL_ERROR;
    ge::GraphOptimizerAttribute attrs;
    ge_ret = optimizer.GetAttributes(attrs);
    EXPECT_EQ(ge_ret, ge::SUCCESS);
}
TEST_F(HvdGraphOptimizerTest, ut_OptimizeFusedGraph)
{
    HvdGraphOptimizer optimizer;
    HcclResult ret;
    ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test_graph");
    const string type = "HcomAllToAllV";
    ge::OpDescPtr opPtr_ = std::make_shared<ge::OpDesc>();
    opPtr_->SetType(type);
    graph->AddNode(opPtr_);

    ge ::Status ge_ret = ge::INTERNAL_ERROR;
    ge_ret = optimizer.OptimizeFusedGraph(*graph);
    EXPECT_EQ(ge_ret, ge::SUCCESS);
}

TEST_F(HvdGraphOptimizerTest, ut_CalcOpRunningParam)
{
    HvdGraphOptimizer optimizer;
    ge::Node node;
    ge::AttrUtils::HasAttr(node.GetOpDesc(), "DUMMY_SET_TRUE_MAXNUM");
    ge::AttrUtils::HasAttr(node.GetOpDesc(), "DUMMY_SET_TRUE_EMBEDDINGDIM");
    ge::AttrUtils::HasAttr(node.GetOpDesc(), "DUMMY_SET_TRUE_FLAGS");

    ge::AttrUtils::SetInt(node.GetOpDesc(), "flags", 1);
    ge ::Status ge_ret = ge::INTERNAL_ERROR;
    ge_ret = optimizer.CalcOpRunningParam(node);
}

TEST_F(HvdGraphOptimizerTest, ut_SetOpOutputMemSize)
{
    HvdGraphOptimizer optimizer;
    std::string sCollectiveType = "";
    ge::Node node;
    ge::AttrUtils::HasAttr(node.GetOpDesc(), "DUMMY_SET_TRUE_MAXNUM");
    ge::AttrUtils::HasAttr(node.GetOpDesc(), "DUMMY_SET_TRUE_EMBEDDINGDIM");
    ge::AttrUtils::HasAttr(node.GetOpDesc(), "DUMMY_SET_TRUE_FLAGS");

    ge::AttrUtils::SetInt(node.GetOpDesc(), "flags", 1);
    ge ::Status ge_ret = ge::INTERNAL_ERROR;
    ge_ret = optimizer.SetOpOutputMemSize(node, sCollectiveType);
    EXPECT_EQ(ge_ret, ge::SUCCESS);
}

TEST_F(HvdGraphOptimizerTest, ut_CalcHCCLOutputMemSize)
{
    HvdGraphOptimizer optimizer;
    std::string sCollectiveType = "";
    int64_t memSize = 2;
    ge ::Status ge_ret = ge::INTERNAL_ERROR;
    ge_ret = optimizer.CalcHCCLOutputMemSize(sCollectiveType, memSize);
    EXPECT_EQ(ge_ret, ge::SUCCESS);
}

TEST_F(HvdGraphOptimizerTest, ut_SetOpMemAttr)
{
    HvdGraphOptimizer optimizer;
    std::string sCollectiveType = "";
    ge::Node node;
    ge::AttrUtils::HasAttr(node.GetOpDesc(), "DUMMY_SET_TRUE_MAXNUM");
    ge::AttrUtils::HasAttr(node.GetOpDesc(), "DUMMY_SET_TRUE_EMBEDDINGDIM");
    ge::AttrUtils::HasAttr(node.GetOpDesc(), "DUMMY_SET_TRUE_FLAGS");

    ge::AttrUtils::SetInt(node.GetOpDesc(), "flags", 1);
    ge ::Status ge_ret = ge::INTERNAL_ERROR;
    ge_ret = optimizer.SetOpMemAttr(node, sCollectiveType);
    EXPECT_EQ(ge_ret, ge::SUCCESS);
}

TEST_F(HvdGraphOptimizerTest, ut_SetOpAtomicInputIndex)
{
    HvdGraphOptimizer optimizer;
    std::string sCollectiveType = "HorovodAllreduce";
    ge::Node node;
    ge::AttrUtils::HasAttr(node.GetOpDesc(), "DUMMY_SET_TRUE_MAXNUM");
    ge::AttrUtils::HasAttr(node.GetOpDesc(), "DUMMY_SET_TRUE_EMBEDDINGDIM");
    ge::AttrUtils::HasAttr(node.GetOpDesc(), "DUMMY_SET_TRUE_FLAGS");

    ge::AttrUtils::SetInt(node.GetOpDesc(), "flags", 1);
    ge ::Status ge_ret = ge::INTERNAL_ERROR;
    ge_ret = optimizer.SetOpAtomicInputIndex(node, sCollectiveType);
    EXPECT_EQ(ge_ret, ge::SUCCESS);
}
