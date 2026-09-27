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
#include "hcom_ops_kernel_info_store.h"
#include "hcom_ops_kernel_builder.h"
#include "hcom_graph_optimizer.h"
#include "hccl_comm_pub.h"
#undef protected
#undef private
#include "hccl/base.h"
#include <hccl/hccl_types.h>

#include "stream_pub.h"
#include "mem_host_pub.h"
#include "mem_device_pub.h"
#include "sal.h"
#include "hccl_impl.h"
#include "llt_hccl_stub_pub.h"
#include "externalinput.h"
#include "config.h"
#include "topoinfo_ranktableParser_pub.h"

#include "plugin_manager.h"
#include "external/ge/ge_api_types.h"  // ge对内options
#include "framework/common/ge_types.h" // ge对外options
#include "hcom_pub.h"
#include "hccl/hcom.h"
#include "hccl/hcom_executor.h"
#include "ranktable/v80_rank_table.h"
#include <iostream>
#include <fstream>
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "hcom_op_utils.h"
#include "llt_hccl_stub_ge.h"
#include "offline_build_config_parse.h"
#include "param_check_pub.h"
#include "ut_rank_table.h"

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

class HcomKernelBuilderTest : public testing::Test {
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
        char file_name[] = "./ut_HcomKernelBuilderTest.json";

        WriteRankTableFile(file_name, rank_table);

        std::cout << "\033[36m--HcomKernelInfoTest SetUP--\033[0m" << std::endl;
    }
    static void TearDownTestCase()
    {
        char file_name[] = "./ut_HcomKernelBuilderTest.json";
        remove(file_name);
        std::cout << "\033[36m--HcomKernelInfoTest TearDown--\033[0m" << std::endl;
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

ge::graphStatus
OfflineRankMappingOption(ge::GEThreadLocalContext* that, const std::string& optionExec, std::string& dumpDebugValue)
{
    nlohmann::json group_list
        = {{{"group_name", "aa"}, {"group_rank_list", {0, 1}}},
           {{"group_name", "off_group_rank_list"}, {"group_rank_list", {0, 1, 2, 3, 4, 5, 6, 7}}}};
    if (optionExec == ge::OPTION_EXEC_HCOM_GROUPLIST) {
        dumpDebugValue = group_list.dump();
    } else if (optionExec == ge::OPTION_EXEC_RANK_TABLE) {
        dumpDebugValue = R"({"status": "completed","version": "1.1","node_list":[{"node_id": "0","rank_list":[
        {"rank_id": "0","item_id": "0","rank_ip":"192.168.2.10"},
        {"rank_id": "1","item_id": "0","rank_ip":"192.168.2.10"},
        {"rank_id": "2","item_id": "0","rank_ip":"192.168.2.10"},
        {"rank_id": "3","item_id": "0","rank_ip":"192.168.2.10"},
        {"rank_id": "4","item_id": "0","rank_ip":"192.168.2.10"},
        {"rank_id": "5","item_id": "0","rank_ip":"192.168.2.10"},
        {"rank_id": "6","item_id": "0","rank_ip":"192.168.2.10"},
        {"rank_id": "7","item_id": "0","rank_ip":"192.168.2.10"},
        {"rank_id": "8","item_id": "-1","rank_ip":"192.168.2.11"}]}]})";
    } else if (optionExec == "ge.socVersion") {
        dumpDebugValue = "Ascend910";
    }
    HCCL_INFO("dumpDebugValue:[%s]", dumpDebugValue.c_str());
    return ge::GRAPH_SUCCESS;
}

TEST_F(HcomKernelBuilderTest, ut_CalcOpRunningParam_common)
{
    struct model_feature feature;
    u32 segment_num = 10;
    std::vector<u32> segment_index;
    HcclResult ret;

    nlohmann::json rank_table
        = {{"status", "completed"},
           {"deploy_mode", "lab"},
           {"group_count", "1"},
           {"chip_info", "910"},
           {"board_id", "0x0000"},
           {"para_plane_nic_location", "device"},
           {"para_plane_nic_num", "1"},
           {"para_plane_nic_name", {"eth0"}},
           {"group_list",
            {{
                {"group_name", ""},
                {"device_num", "1"},
                {"server_num", "1"},
                {"instance_count", "1"},
                {"instance_list",
                 {{{"rank_id", "0"},
                   {"server_id", "172.17.1.120"},
                   {"devices", {{{"device_id", "0"}, {"device_ip", "192.168.1.120"}}}}}}},
            }}}};

    char file_name_t[] = "./ut_CalcOpRunningParam_common.json";
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
    // HcomOpsKernelInfoStore hcomKernelInfo;
    HcomOpsKernelBuilder hcomKernelInfo;

    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    char* rank_table_file = "./ut_CalcOpRunningParam_common.json";
    char* rank_ID = "0";

    HCCL_INFO("HcomInitByFile START.");
    ret = HcomInitByFile(rank_table_file, rank_ID);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    HCCL_INFO("HcomInitByFile OK.");

    MOCKER_CPP(&HcomOpsKernelBuilder::GetAndSetTaskNum).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&HcomOpsKernelBuilder::GetOriginalGraphShapeTypeFromDesc).stubs().will(returnValue(HCCL_SUCCESS));

    ge::NodePtr nodeptr(new NodeTest);
    nodeptr->GetOpDesc()->SetType("");
    ge_ret = hcomKernelInfo.CalcOpRunningParam(*nodeptr);
    EXPECT_EQ(ge_ret, ge::INTERNAL_ERROR);

    std::string type;
    type = HCCL_KERNEL_OP_TYPE_BROADCAST;
    nodeptr->GetOpDesc()->SetType(type);
    ge_ret = hcomKernelInfo.CalcOpRunningParam(*nodeptr);
    EXPECT_EQ(ge_ret, ge::INTERNAL_ERROR);
    ge::AttrUtils::SetStr(nodeptr->GetOpDesc(), "_super_kernel_scope", "super_kernel_scope");
    ge_ret = hcomKernelInfo.CalcOpRunningParam(*nodeptr);
    // s64 streamNum;
    // ge::AttrUtils::GetInt(nodeptr->GetOpDesc(), "used_stream_num", streamNum);
    // EXPECT_EQ(streamNum, HCCL_STREAM_NUM_1);
    std::vector<int64_t> workSpaceBytes = nodeptr->GetOpDesc()->GetWorkspaceBytes();

    type = HCCL_KERNEL_OP_TYPE_REDUCESCATTER;
    nodeptr->GetOpDesc()->SetType(type);
    ge::AttrUtils::HasAttr(nodeptr->GetOpDesc(), "DUMMY_SET_FALSE_RANK_SIZE");
    int64_t RANK_SIZE = 1;
    ge::AttrUtils::SetInt(nodeptr->GetOpDesc(), "rank_size", RANK_SIZE);
    ge_ret = hcomKernelInfo.CalcOpRunningParam(*nodeptr);
    EXPECT_EQ(ge_ret, ge::SUCCESS);
    workSpaceBytes.clear();
    workSpaceBytes = nodeptr->GetOpDesc()->GetWorkspaceBytes();

    type = HCCL_KERNEL_OP_TYPE_ALLGATHER;
    nodeptr->GetOpDesc()->SetType(type);
    ge_ret = hcomKernelInfo.CalcOpRunningParam(*nodeptr);
    EXPECT_EQ(ge_ret, ge::SUCCESS);
    workSpaceBytes.clear();
    workSpaceBytes = nodeptr->GetOpDesc()->GetWorkspaceBytes();

    type = HCCL_KERNEL_OP_TYPE_ALLREDUCE;
    nodeptr->GetOpDesc()->SetType(type);
    ge_ret = hcomKernelInfo.CalcOpRunningParam(*nodeptr);
    EXPECT_EQ(ge_ret, ge::SUCCESS);
    MOCKER_CPP(&ge::GEThreadLocalContext::GetOption).stubs().will(invoke(OfflineRankMappingOption));
    uint32_t graphId = 1;
    MOCKER_CPP(&HcomOpsKernelBuilder::GetRootGraphID)
        .stubs()
        .with(mockcpp::any(), outBound(graphId))
        .will(returnValue(HCCL_SUCCESS));
    std::string curGroup = "off_group_rank_list";
    ge::AttrUtils::HasAttr(nodeptr->GetOpDesc(), "DUMMY_SET_TRUE_GROUP");
    ge::AttrUtils::SetStr(nodeptr->GetOpDesc(), "group", curGroup);
    ge_ret = hcomKernelInfo.CalcOpRunningParam(*nodeptr);
    EXPECT_EQ(ge_ret, ge::SUCCESS);

    type = HCCL_KERNEL_OP_TYPE_RECEIVE;
    nodeptr->GetOpDesc()->SetType(type);
    ge::AttrUtils::HasAttr(nodeptr->GetOpDesc(), "DUMMY_SET_TRUE_DTYPE");
    ge::AttrUtils::HasAttr(nodeptr->GetOpDesc(), "DUMMY_SET_TRUE_SHAPE");
    ge_ret = hcomKernelInfo.CalcOpRunningParam(*nodeptr);
    EXPECT_EQ(ge_ret, ge::SUCCESS);
    MOCKER_CPP(&ge::GEThreadLocalContext::GetOption).stubs().will(invoke(OfflineRankMappingOption));
    ge_ret = hcomKernelInfo.CalcOpRunningParam(*nodeptr);
    EXPECT_EQ(ge_ret, ge::SUCCESS);
    ge::AttrUtils::HasAttr(nodeptr->GetOpDesc(), "DUMMY_SET_FALSE_DTYPE");
    ge::AttrUtils::HasAttr(nodeptr->GetOpDesc(), "DUMMY_SET_FALSE_SHAPE");

    ret = HcomDestroy();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    remove(file_name_t);
    GlobalMockObject::verify();
}

TEST_F(HcomKernelBuilderTest, ut_CalcOpRunningParam_common_51)
{
    struct model_feature feature;
    u32 segment_num = 10;
    std::vector<u32> segment_index;
    HcclResult ret;

    nlohmann::json rank_table
        = {{"status", "completed"},
           {"deploy_mode", "lab"},
           {"group_count", "1"},
           {"chip_info", "910"},
           {"board_id", "0x0000"},
           {"para_plane_nic_location", "device"},
           {"para_plane_nic_num", "1"},
           {"para_plane_nic_name", {"eth0"}},
           {"group_list",
            {{
                {"group_name", ""},
                {"device_num", "1"},
                {"server_num", "1"},
                {"instance_count", "1"},
                {"instance_list",
                 {{{"rank_id", "0"},
                   {"server_id", "172.17.1.120"},
                   {"devices", {{{"device_id", "0"}, {"device_ip", "192.168.1.120"}}}}}}},
            }}}};

    char file_name_t[] = "./ut_CalcOpRunningParam_common_51.json";
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
    // HcomOpsKernelInfoStore hcomKernelInfo;
    HcomOpsKernelBuilder hcomKernelInfo;

    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    char* rank_table_file = "./ut_CalcOpRunningParam_common_51.json";
    char* rank_ID = "0";

    HCCL_INFO("HcomInitByFile START.");
    ret = HcomInitByFile(rank_table_file, rank_ID);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    HCCL_INFO("HcomInitByFile OK.");

    MOCKER_CPP(&HcomOpsKernelBuilder::GetAndSetTaskNum).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&HcomOpsKernelBuilder::GetOriginalGraphShapeTypeFromDesc).stubs().will(returnValue(HCCL_SUCCESS));

    DevType type610 = DevType::DEV_TYPE_310P1;
    MOCKER(GetOffDeviceTypeWithoutDev).stubs().with(outBound(type610)).will(returnValue(HCCL_SUCCESS));

    u32 numHccsLink = 0;
    MOCKER(HcomGetHccsLinkNum).stubs().with(mockcpp::any(), outBound(numHccsLink)).will(returnValue(HCCL_SUCCESS));
    u32 rankSize = 2;
    MOCKER(HcomGetRankSize).stubs().with(mockcpp::any(), outBoundP(&rankSize)).will(returnValue(HCCL_SUCCESS));

    ge::NodePtr nodeptr(new NodeTest);
    nodeptr->GetOpDesc()->SetType("");
    ge_ret = hcomKernelInfo.CalcOpRunningParam(*nodeptr);
    EXPECT_EQ(ge_ret, ge::INTERNAL_ERROR);

    std::string type;
    type = HCCL_KERNEL_OP_TYPE_BROADCAST;
    nodeptr->GetOpDesc()->SetType(type);
    ge::AttrUtils::SetStr(nodeptr->GetOpDesc(), "group", HCCL_WORLD_GROUP);
    ge_ret = hcomKernelInfo.CalcOpRunningParam(*nodeptr);
    EXPECT_EQ(ge_ret, ge::SUCCESS);
    // s64 streamNum;
    // ge::AttrUtils::GetInt(nodeptr->GetOpDesc(), "used_stream_num", streamNum);
    // EXPECT_EQ(streamNum, HCCL_STREAM_NUM_1);
    std::vector<int64_t> workSpaceBytes = nodeptr->GetOpDesc()->GetWorkspaceBytes();

    type = HCCL_KERNEL_OP_TYPE_REDUCESCATTER;
    nodeptr->GetOpDesc()->SetType(type);
    ge::AttrUtils::HasAttr(nodeptr->GetOpDesc(), "DUMMY_SET_FALSE_RANK_SIZE");
    int64_t RANK_SIZE = 1;
    ge::AttrUtils::SetInt(nodeptr->GetOpDesc(), "rank_size", RANK_SIZE);
    ge_ret = hcomKernelInfo.CalcOpRunningParam(*nodeptr);
    EXPECT_EQ(ge_ret, ge::SUCCESS);
    workSpaceBytes.clear();
    workSpaceBytes = nodeptr->GetOpDesc()->GetWorkspaceBytes();

    type = HCCL_KERNEL_OP_TYPE_ALLGATHER;
    nodeptr->GetOpDesc()->SetType(type);
    ge_ret = hcomKernelInfo.CalcOpRunningParam(*nodeptr);
    EXPECT_EQ(ge_ret, ge::SUCCESS);
    workSpaceBytes.clear();
    workSpaceBytes = nodeptr->GetOpDesc()->GetWorkspaceBytes();

    type = HCCL_KERNEL_OP_TYPE_ALLREDUCE;
    nodeptr->GetOpDesc()->SetType(type);
    ge_ret = hcomKernelInfo.CalcOpRunningParam(*nodeptr);
    EXPECT_EQ(ge_ret, ge::SUCCESS);
    MOCKER_CPP(&ge::GEThreadLocalContext::GetOption).stubs().will(invoke(OfflineRankMappingOption));
    std::string curGroup = "off_group_rank_list";
    ge::AttrUtils::HasAttr(nodeptr->GetOpDesc(), "DUMMY_SET_TRUE_GROUP");
    ge::AttrUtils::SetStr(nodeptr->GetOpDesc(), "group", curGroup);
    ge_ret = hcomKernelInfo.CalcOpRunningParam(*nodeptr);
    EXPECT_EQ(ge_ret, ge::SUCCESS);

    type = HCCL_KERNEL_OP_TYPE_RECEIVE;
    nodeptr->GetOpDesc()->SetType(type);
    ge::AttrUtils::HasAttr(nodeptr->GetOpDesc(), "DUMMY_SET_TRUE_DTYPE");
    ge::AttrUtils::HasAttr(nodeptr->GetOpDesc(), "DUMMY_SET_TRUE_SHAPE");
    ge_ret = hcomKernelInfo.CalcOpRunningParam(*nodeptr);
    EXPECT_EQ(ge_ret, ge::SUCCESS);
    MOCKER_CPP(&ge::GEThreadLocalContext::GetOption).stubs().will(invoke(OfflineRankMappingOption));
    ge_ret = hcomKernelInfo.CalcOpRunningParam(*nodeptr);
    EXPECT_EQ(ge_ret, ge::SUCCESS);
    ge::AttrUtils::HasAttr(nodeptr->GetOpDesc(), "DUMMY_SET_FALSE_DTYPE");
    ge::AttrUtils::HasAttr(nodeptr->GetOpDesc(), "DUMMY_SET_FALSE_SHAPE");

    ret = HcomDestroy();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    remove(file_name_t);
    GlobalMockObject::verify();
}

TEST_F(HcomKernelBuilderTest, ut_generateTask)
{
    ge::NodePtr nodeptr(new NodeTest);
    ge::Buffer tempBuffer;
    ge::RunContext runContext_dummy;
    // hccl::HcomOpsKernelInfoStore hcomKernelInfo;
    HcomOpsKernelBuilder hcomKernelInfo;

    std::vector<domi::TaskDef> taskDefList;

    rtError_t rt_ret = RT_ERROR_NONE;
    rtStream_t stream;
    s64 streamId = 10000;
    nodeptr->GetOpDesc()->SetStreamId((s64)streamId);

    std::string name = "HcomTag";
    nodeptr->GetOpDesc()->SetName(name);

    // -------------------HcomBroadcast test----------------
    std::string type = HCCL_KERNEL_OP_TYPE_BROADCAST;
    nodeptr->GetOpDesc()->SetType(type);

    ge::AttrUtils::HasAttr(nodeptr->GetOpDesc(), "DUMMY_SET_TRUE_GROUP");
    std::string tempStr = HCCL_WORLD_GROUP;
    ge::AttrUtils::SetStr(nodeptr->GetOpDesc(), "group", tempStr);
    ge::AttrUtils::HasAttr(nodeptr->GetOpDesc(), "DUMMY_SET_TRUE_SRCRANK");
    ge::AttrUtils::HasAttr(nodeptr->GetOpDesc(), "DUMMY_SET_TRUE_DESTRANK");
    ge::AttrUtils::HasAttr(nodeptr->GetOpDesc(), "DUMMY_SET_TRUE_SRTAG");
    ge::AttrUtils::HasAttr(nodeptr->GetOpDesc(), "DUMMY_SET_FALSE_TAG");
    ge::AttrUtils::HasAttr(nodeptr->GetOpDesc(), "DUMMY_SET_TRUE_FISSION");
    ge::AttrUtils::HasAttr(nodeptr->GetOpDesc(), "DUMMY_SET_TRUE_DUMPSIZE");
    ge::AttrUtils::HasAttr(nodeptr->GetOpDesc(), "DUMMY_SET_TRUE_DUMPTYPE");
    s64 tempInt = 5;
    ge::AttrUtils::SetInt(nodeptr->GetOpDesc(), "dest_rank", tempInt);
    ge::AttrUtils::SetInt(nodeptr->GetOpDesc(), "src_rank", tempInt);
    ge::AttrUtils::SetInt(nodeptr->GetOpDesc(), "sr_tag", tempInt);
    ge::AttrUtils::SetInt(nodeptr->GetOpDesc(), "_fission_factor", 1);
    ge::AttrUtils::SetInt(nodeptr->GetOpDesc(), "global_workspace_size", 1);
    ge::AttrUtils::SetInt(nodeptr->GetOpDesc(), "global_workspace_type", 0);
    HCCL_INFO("node[%p] run context[%p]", nodeptr.get(), &runContext_dummy);
    HCCL_INFO("----------%s", nodeptr->GetOpDesc()->GetType().c_str());
    ge::AttrUtils::HasAttr(nodeptr->GetOpDesc(), "DUMMY_SET_TRUE_NEEDMAPRANK");
    ge::AttrUtils::SetBool(nodeptr->GetOpDesc(), "_need_map_rank_id", true);

    s32 ret = hcomKernelInfo.GenerateTask(*nodeptr, runContext_dummy, taskDefList);
    EXPECT_EQ(ret, ge::SUCCESS);

    u32 result_type = taskDefList[0].type();
    u32 result_stream_id = taskDefList[0].stream_id();
    std::string result_hccl_hccl_type = taskDefList[0].mutable_kernel_hccl()->hccl_type();
    std::string result_private_def = taskDefList[0].private_def();
    char private_def_buf[sizeof(HCCL_KERNEL_INFO_PRIVATE_DEF)];
    sal_memcpy(&private_def_buf[0], sizeof(private_def_buf), result_private_def.c_str(), sizeof(private_def_buf));
    HCCL_KERNEL_INFO_PRIVATE_DEF* privateDefBuf = (HCCL_KERNEL_INFO_PRIVATE_DEF*)&private_def_buf[0];
    std::string result_group = reinterpret_cast<const char*>(privateDefBuf->group);
    // std::string result_tag = reinterpret_cast<const char*>(privateDefBuf->tag);
    u32 result_srcRank = (privateDefBuf->srcRank);
    u32 result_destRank = (privateDefBuf->destRank);
    u32 result_srTag = (privateDefBuf->srTag);
    EXPECT_EQ(result_type, RT_MODEL_TASK_HCCL);
    EXPECT_EQ(result_stream_id, streamId);
    EXPECT_EQ(result_hccl_hccl_type, type);
    EXPECT_EQ(result_group, tempStr);
    // EXPECT_EQ(result_tag, name);
    EXPECT_EQ(result_srcRank, 0);
    EXPECT_EQ(result_destRank, 0);
    EXPECT_EQ(result_srTag, 0);

    // -------------------HcomSend test----------------
    MOCKER(HcomGetRankId).expects(atMost(8)).will(returnValue(HCCL_SUCCESS));
    type = HCCL_KERNEL_OP_TYPE_SEND;
    nodeptr->GetOpDesc()->SetType(type);
    ge::AttrUtils::HasAttr(nodeptr->GetOpDesc(), "DUMMY_SET_FALSE_TAG");
    ret = hcomKernelInfo.GenerateTask(*nodeptr, runContext_dummy, taskDefList);
    EXPECT_EQ(ret, ge::SUCCESS);
    result_type = taskDefList[1].type();
    result_stream_id = taskDefList[1].stream_id();
    result_hccl_hccl_type = taskDefList[1].mutable_kernel_hccl()->hccl_type();
    result_private_def = taskDefList[1].private_def();
    sal_memcpy(&private_def_buf[0], sizeof(private_def_buf), result_private_def.c_str(), sizeof(private_def_buf));
    privateDefBuf = (HCCL_KERNEL_INFO_PRIVATE_DEF*)&private_def_buf[0];
    result_group = reinterpret_cast<const char*>(privateDefBuf->group);
    // result_tag = reinterpret_cast<const char*>(privateDefBuf->tag);
    result_srcRank = (privateDefBuf->srcRank);
    result_destRank = (privateDefBuf->destRank);
    result_srTag = (privateDefBuf->srTag);
    EXPECT_EQ(result_type, RT_MODEL_TASK_HCCL);
    EXPECT_EQ(result_stream_id, streamId);
    EXPECT_EQ(result_hccl_hccl_type, type);
    EXPECT_EQ(result_group, tempStr);
    std::string tmpTag = result_group + "5" + "0" + "5";
    // EXPECT_EQ(result_tag, tmpTag);
    EXPECT_EQ(result_srcRank, 0);
    EXPECT_EQ(result_destRank, tempInt);
    EXPECT_EQ(result_srTag, tempInt);

    // Send: 未设定 destRank 时，报错
    ge::AttrUtils::HasAttr(nodeptr->GetOpDesc(), "DUMMY_SET_FALSE_DESTRANK");
    ret = hcomKernelInfo.GenerateTask(*nodeptr, runContext_dummy, taskDefList);
    EXPECT_EQ(ret, ge::INTERNAL_ERROR);
    ge::AttrUtils::HasAttr(nodeptr->GetOpDesc(), "DUMMY_SET_TRUE_DESTRANK");

    // Send: 未设定 srTag 时，报错
    ge::AttrUtils::HasAttr(nodeptr->GetOpDesc(), "DUMMY_SET_FALSE_SRTAG");
    ret = hcomKernelInfo.GenerateTask(*nodeptr, runContext_dummy, taskDefList);
    EXPECT_EQ(ret, ge::INTERNAL_ERROR);
    ge::AttrUtils::HasAttr(nodeptr->GetOpDesc(), "DUMMY_SET_TRUE_SRTAG");

    // -------------------HcomReceive test----------------
    type = HCCL_KERNEL_OP_TYPE_RECEIVE;
    nodeptr->GetOpDesc()->SetType(type);
    ge::AttrUtils::HasAttr(nodeptr->GetOpDesc(), "DUMMY_SET_FALSE_TAG");
    ge::AttrUtils::HasAttr(nodeptr->GetOpDesc(), "DUMMY_SET_FALSE_GROUP");
    ret = hcomKernelInfo.GenerateTask(*nodeptr, runContext_dummy, taskDefList);
    EXPECT_EQ(ret, ge::SUCCESS);
    result_type = taskDefList[2].type();
    result_stream_id = taskDefList[2].stream_id();
    result_hccl_hccl_type = taskDefList[2].mutable_kernel_hccl()->hccl_type();
    result_private_def = taskDefList[2].private_def();
    sal_memcpy(&private_def_buf[0], sizeof(private_def_buf), result_private_def.c_str(), sizeof(private_def_buf));
    privateDefBuf = (HCCL_KERNEL_INFO_PRIVATE_DEF*)&private_def_buf[0];
    result_group = reinterpret_cast<const char*>(privateDefBuf->group);
    // result_tag = reinterpret_cast<const char*>(privateDefBuf->tag);
    result_srcRank = (privateDefBuf->srcRank);
    result_destRank = (privateDefBuf->destRank);
    result_srTag = (privateDefBuf->srTag);
    EXPECT_EQ(result_type, RT_MODEL_TASK_HCCL);
    EXPECT_EQ(result_stream_id, streamId);
    EXPECT_EQ(result_hccl_hccl_type, type);
    EXPECT_EQ(result_group, tempStr);
    tmpTag = result_group + "5" + "5" + "0";
    // EXPECT_EQ(result_tag, tmpTag);
    EXPECT_EQ(result_srcRank, tempInt);
    EXPECT_EQ(result_destRank, 0);
    EXPECT_EQ(result_srTag, tempInt);
    ge::AttrUtils::HasAttr(nodeptr->GetOpDesc(), "DUMMY_SET_TRUE_GROUP");

    // Receive: srcRank未设定时，报错
    ge::AttrUtils::HasAttr(nodeptr->GetOpDesc(), "DUMMY_SET_FALSE_SRCRANK");
    ret = hcomKernelInfo.GenerateTask(*nodeptr, runContext_dummy, taskDefList);
    EXPECT_EQ(ret, ge::INTERNAL_ERROR);
    ge::AttrUtils::HasAttr(nodeptr->GetOpDesc(), "DUMMY_SET_TRUE_SRCRANK");

    // Receive: srTag未设定时，报错
    ge::AttrUtils::HasAttr(nodeptr->GetOpDesc(), "DUMMY_SET_FALSE_SRTAG");
    ret = hcomKernelInfo.GenerateTask(*nodeptr, runContext_dummy, taskDefList);
    EXPECT_EQ(ret, ge::INTERNAL_ERROR);
    ge::AttrUtils::HasAttr(nodeptr->GetOpDesc(), "DUMMY_SET_TRUE_SRTAG");

    // -------------------HcomAllReduce test----------------
    type = HCCL_KERNEL_OP_TYPE_ALLREDUCE;
    nodeptr->GetOpDesc()->SetType(type);
    ge::AttrUtils::HasAttr(nodeptr->GetOpDesc(), "DUMMY_SET_FALSE_TAG");
    ret = hcomKernelInfo.GenerateTask(*nodeptr, runContext_dummy, taskDefList);
    EXPECT_EQ(ret, ge::SUCCESS);
    ge::AttrUtils::HasAttr(nodeptr->GetOpDesc(), "DUMMY_SET_FALSE_TAG");

    // -------------------hcom_remote_read test----------------
    type = HCCL_KERNEL_OP_TYPE_REMOTE_READ;
    nodeptr->GetOpDesc()->SetType(type);
    ge::AttrUtils::HasAttr(nodeptr->GetOpDesc(), "DUMMY_SET_FALSE_TAG");
    ret = hcomKernelInfo.GenerateTask(*nodeptr, runContext_dummy, taskDefList);
    EXPECT_EQ(ret, ge::SUCCESS);
    MOCKER(IsOfflineCompilation).stubs().with(mockcpp::any()).will(returnValue(true));
    ret = hcomKernelInfo.GenerateTask(*nodeptr, runContext_dummy, taskDefList);
    EXPECT_EQ(ret, ge::SUCCESS);
    ge::AttrUtils::HasAttr(nodeptr->GetOpDesc(), "DUMMY_SET_FALSE_TAG");

    // -------------------incalid type test----------------
    type = " ";
    nodeptr->GetOpDesc()->SetType(type);
    ge::AttrUtils::HasAttr(nodeptr->GetOpDesc(), "DUMMY_SET_FALSE_TAG");
    ret = hcomKernelInfo.GenerateTask(*nodeptr, runContext_dummy, taskDefList);
    EXPECT_EQ(ret, ge::INTERNAL_ERROR);

    MOCKER_CPP(&HcomOpsKernelBuilder::GetOpIntAttr).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
}

TEST_F(HcomKernelBuilderTest, ut_GenerateTask_unknown)
{
    ge::Status ret;
    // HcomOpsKernelInfoStore  hcomOpsKernelInfoStore_;
    map<string, string> options;
    HcomOpsKernelBuilder hcomOpsKernelInfoStore_;
    ret = hcomOpsKernelInfoStore_.Initialize(options);
    EXPECT_EQ(ret, ge::SUCCESS);
    bool is_unknown = true;
    ge::NodePtr nodeptr(new NodeTest);
    ge::RunContext runContext;
    std::vector<domi::TaskDef> taskDefList;

    MOCKER(&ge::NodeUtils::GetNodeUnknownShapeStatus)
        .stubs()
        .with(mockcpp::any(), outBound(is_unknown))
        .will(returnValue(ge::GRAPH_SUCCESS));
    ret = hcomOpsKernelInfoStore_.GenerateTask(*nodeptr, runContext, taskDefList);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
    ret = hcomOpsKernelInfoStore_.Finalize();
    EXPECT_EQ(ret, ge::SUCCESS);
}

HcclResult MockGetOffDeviceTypeWithoutDev(DevType& devType)
{
    devType = DevType::DEV_TYPE_310P3;
    HCCL_DEBUG("[offline] Get devtype[%u]....", devType);
    return HCCL_SUCCESS;
}

TEST_F(HcomKernelBuilderTest, ut_CalcOpRunningParam_unknown)
{
    ge::Status ret;
    // HcomOpsKernelInfoStore  hcomOpsKernelInfoStore_;
    HcomOpsKernelBuilder hcomOpsKernelInfoStore_;
    bool is_unknown = true;
    ge::NodePtr nodeptr(new NodeTest);
    ge::RunContext runContext;
    std::vector<domi::TaskDef> taskDefList;

    MOCKER(&ge::NodeUtils::GetNodeUnknownShapeStatus)
        .stubs()
        .with(mockcpp::any(), outBound(is_unknown))
        .will(returnValue(ge::GRAPH_SUCCESS));
    MOCKER_CPP(&HcomOpsKernelBuilder::GetAndSetTaskNum).stubs().will(returnValue(HCCL_SUCCESS));
    ret = hcomOpsKernelInfoStore_.CalcOpRunningParam(*nodeptr);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    MOCKER(IsOfflineCompilation).stubs().will(returnValue(true));

    MOCKER(GetOffDeviceTypeWithoutDev).stubs().will(invoke(MockGetOffDeviceTypeWithoutDev));
    hcomOpsKernelInfoStore_.CalcOpRunningParam(*nodeptr);
    GlobalMockObject::verify();
}

TEST_F(HcomKernelBuilderTest, ut_CalcOpRunningParam_V51)
{
    struct model_feature feature;
    u32 segment_num = 10;
    std::vector<u32> segment_index;
    HcclResult ret;

    nlohmann::json rank_table
        = {{"status", "completed"},
           {"deploy_mode", "lab"},
           {"group_count", "1"},
           {"chip_info", "310P3"},
           {"board_id", "0x2000"},
           {"para_plane_nic_location", "device"},
           {"para_plane_nic_num", "1"},
           {"para_plane_nic_name", {"eth0"}},
           {"group_list",
            {{
                {"group_name", ""},
                {"device_num", "1"},
                {"server_num", "1"},
                {"instance_count", "1"},
                {"instance_list",
                 {{{"rank_id", "0"},
                   {"server_id", "172.17.1.120"},
                   {"devices", {{{"device_id", "0"}, {"device_ip", "192.168.1.120"}}}}}}},
            }}}};

    char file_name_t[] = "./ut_CalcOpRunningParam_V51.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(1) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    } else {
        HCCL_ERROR("open %s failed", file_name_t);
    }

    outfile.close();

    set_board_id(0x2000);

    ge::OpDesc op;
    ge ::Status ge_ret = ge::INTERNAL_ERROR;
    HcomOpsKernelBuilder hcomKernelInfo;

    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    char* rank_table_file = "./ut_CalcOpRunningParam_V51.json";
    char* rank_ID = "0";
    HCCL_INFO("HcomInitByFile START.");
    ret = HcomInitByFile(rank_table_file, rank_ID);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    HCCL_INFO("HcomInitByFile OK.");

    MOCKER_CPP(&HcomOpsKernelBuilder::GetAndSetTaskNum).stubs().will(returnValue(HCCL_SUCCESS));
    ge::NodePtr nodeptr(new NodeTest);
    ge_ret = hcomKernelInfo.CalcOpRunningParam(*nodeptr);
    EXPECT_EQ(ge_ret, ge::INTERNAL_ERROR);

    std::string type;
    type = HCCL_KERNEL_OP_TYPE_BROADCAST;
    nodeptr->GetOpDesc()->SetType(type);
    ge_ret = hcomKernelInfo.CalcOpRunningParam(*nodeptr);
    EXPECT_EQ(ge_ret, ge::SUCCESS);
    // s64 streamNum;
    // ge::AttrUtils::GetInt(nodeptr->GetOpDesc(), "used_stream_num", streamNum);
    // EXPECT_EQ(streamNum, HCCL_STREAM_NUM_1);
    std::vector<int64_t> workSpaceBytes = nodeptr->GetOpDesc()->GetWorkspaceBytes();

    type = HCCL_KERNEL_OP_TYPE_REDUCESCATTER;
    nodeptr->GetOpDesc()->SetType(type);
    ge::AttrUtils::HasAttr(nodeptr->GetOpDesc(), "DUMMY_SET_FALSE_RANK_SIZE");
    int64_t RANK_SIZE = 1;
    ge::AttrUtils::SetInt(nodeptr->GetOpDesc(), "rank_size", RANK_SIZE);
    ge_ret = hcomKernelInfo.CalcOpRunningParam(*nodeptr);
    EXPECT_EQ(ge_ret, ge::SUCCESS);
    workSpaceBytes.clear();
    workSpaceBytes = nodeptr->GetOpDesc()->GetWorkspaceBytes();

    type = HCCL_KERNEL_OP_TYPE_ALLGATHER;
    nodeptr->GetOpDesc()->SetType(type);
    ge_ret = hcomKernelInfo.CalcOpRunningParam(*nodeptr);
    EXPECT_EQ(ge_ret, ge::SUCCESS);
    workSpaceBytes.clear();
    workSpaceBytes = nodeptr->GetOpDesc()->GetWorkspaceBytes();

    MOCKER(HcomOpUtils::GetAllReduceScratchMemSize).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    type = HCCL_KERNEL_OP_TYPE_ALLREDUCE;
    nodeptr->GetOpDesc()->SetType(type);
    ge_ret = hcomKernelInfo.CalcOpRunningParam(*nodeptr);
    EXPECT_EQ(ge_ret, ge::SUCCESS);

    type = HCCL_KERNEL_OP_TYPE_RECEIVE;
    nodeptr->GetOpDesc()->SetType(type);
    ge::AttrUtils::HasAttr(nodeptr->GetOpDesc(), "DUMMY_SET_TRUE_DTYPE");
    ge::AttrUtils::HasAttr(nodeptr->GetOpDesc(), "DUMMY_SET_TRUE_SHAPE");
    ge_ret = hcomKernelInfo.CalcOpRunningParam(*nodeptr);
    EXPECT_EQ(ge_ret, ge::SUCCESS);
    ge::AttrUtils::HasAttr(nodeptr->GetOpDesc(), "DUMMY_SET_FALSE_DTYPE");
    ge::AttrUtils::HasAttr(nodeptr->GetOpDesc(), "DUMMY_SET_FALSE_SHAPE");

    ret = HcomDestroy();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    set_board_id(0);
    remove(file_name_t);
    GlobalMockObject::verify();
}

TEST_F(HcomKernelBuilderTest, ut_GenerateTask_AlltoAllV)
{
    ge::NodePtr nodeptr(new NodeTest);
    HcomOpsKernelBuilder hcomKernelInfo;

    s64 streamId = 10000;
    nodeptr->GetOpDesc()->SetStreamId((s64)streamId);

    std::string name = "HcomTag";
    nodeptr->GetOpDesc()->SetName(name);

    nodeptr->GetOpDesc()->SetType(HCCL_KERNEL_OP_TYPE_ALLTOALLV);
    ge::AttrUtils::SetStr(nodeptr->GetOpDesc(), "group", HCCL_WORLD_GROUP);
    ge::AttrUtils::SetInt(nodeptr->GetOpDesc(), "sr_tag", 5);
    std::vector<int64_t> sendCounts;
    ge::AttrUtils::SetListInt(nodeptr->GetOpDesc(), "send_counts", sendCounts);

    ge::RunContext runContext;
    std::vector<domi::TaskDef> taskDefList;

    s32 ret = hcomKernelInfo.GenerateTask(*nodeptr, runContext, taskDefList);
    EXPECT_EQ(ret, ge::SUCCESS);
}

TEST_F(HcomKernelBuilderTest, ut_GenerateTask_AlltoAllVC)
{
    ge::NodePtr nodeptr(new NodeTest);
    HcomOpsKernelBuilder hcomKernelInfo;

    s64 streamId = 10000;
    nodeptr->GetOpDesc()->SetStreamId((s64)streamId);

    std::string name = "HcomTag";
    nodeptr->GetOpDesc()->SetName(name);

    nodeptr->GetOpDesc()->SetType(HCCL_KERNEL_OP_TYPE_ALLTOALLVC);
    ge::AttrUtils::SetStr(nodeptr->GetOpDesc(), "group", HCCL_WORLD_GROUP);
    ge::AttrUtils::SetInt(nodeptr->GetOpDesc(), "sr_tag", 5);
    std::vector<int64_t> sendCounts;
    ge::AttrUtils::SetListInt(nodeptr->GetOpDesc(), "send_counts", sendCounts);

    ge::RunContext runContext;
    std::vector<domi::TaskDef> taskDefList;

    s32 ret = hcomKernelInfo.GenerateTask(*nodeptr, runContext, taskDefList);
    EXPECT_EQ(ret, ge::SUCCESS);
}

TEST_F(HcomKernelBuilderTest, ut_generateTask_by_comm_pytorch)
{
    ge::NodePtr nodeptr(new NodeTest);
    ge::Buffer tempBuffer;
    ge::RunContext runContext_dummy;
    HcomOpsKernelBuilder hcomKernelBuilder;

    std::vector<domi::TaskDef> taskDefList;

    rtError_t rt_ret = RT_ERROR_NONE;
    rtStream_t stream;
    s64 streamId = 10000;
    nodeptr->GetOpDesc()->SetStreamId((s64)streamId);

    std::string name = "HcomTag";
    nodeptr->GetOpDesc()->SetName(name);

    // -------------------HcomBroadcast test----------------
    std::string type = HCCL_KERNEL_OP_TYPE_BROADCAST;
    nodeptr->GetOpDesc()->SetType(type);

    ge::AttrUtils::HasAttr(nodeptr->GetOpDesc(), "DUMMY_SET_TRUE_COMM");
    ge::AttrUtils::HasAttr(nodeptr->GetOpDesc(), "comm");
    int64_t hcomComm = 645678156;
    ge::AttrUtils::SetInt(nodeptr->GetOpDesc(), "comm", hcomComm);
    ge::AttrUtils::HasAttr(nodeptr->GetOpDesc(), "DUMMY_SET_TRUE_SRCRANK");
    ge::AttrUtils::HasAttr(nodeptr->GetOpDesc(), "DUMMY_SET_TRUE_DESTRANK");
    ge::AttrUtils::HasAttr(nodeptr->GetOpDesc(), "DUMMY_SET_TRUE_SRTAG");
    ge::AttrUtils::HasAttr(nodeptr->GetOpDesc(), "DUMMY_SET_FALSE_TAG");
    ge::AttrUtils::HasAttr(nodeptr->GetOpDesc(), "DUMMY_SET_TRUE_FISSION");
    ge::AttrUtils::HasAttr(nodeptr->GetOpDesc(), "DUMMY_SET_TRUE_DUMPSIZE");
    ge::AttrUtils::HasAttr(nodeptr->GetOpDesc(), "DUMMY_SET_TRUE_DUMPTYPE");
    s64 tempInt = 5;
    ge::AttrUtils::SetInt(nodeptr->GetOpDesc(), "dest_rank", tempInt);
    ge::AttrUtils::SetInt(nodeptr->GetOpDesc(), "src_rank", tempInt);
    ge::AttrUtils::SetInt(nodeptr->GetOpDesc(), "sr_tag", tempInt);
    ge::AttrUtils::SetInt(nodeptr->GetOpDesc(), "_fission_factor", 1);
    ge::AttrUtils::SetInt(nodeptr->GetOpDesc(), "global_workspace_size", 1);
    ge::AttrUtils::SetInt(nodeptr->GetOpDesc(), "global_workspace_type", 0);
    HCCL_INFO("node[%p] run context[%p]", nodeptr.get(), &runContext_dummy);
    HCCL_INFO("----------%s", nodeptr->GetOpDesc()->GetType().c_str());
    s32 ret = hcomKernelBuilder.GenerateTask(*nodeptr, runContext_dummy, taskDefList);
    EXPECT_EQ(ret, ge::SUCCESS);

    u32 result_type = taskDefList[0].type();
    u32 result_stream_id = taskDefList[0].stream_id();
    std::string result_hccl_hccl_type = taskDefList[0].mutable_kernel_hccl()->hccl_type();
    std::string result_private_def = taskDefList[0].private_def();
    char private_def_buf[sizeof(HCCL_KERNEL_INFO_PRIVATE_DEF)];
    sal_memcpy(&private_def_buf[0], sizeof(private_def_buf), result_private_def.c_str(), sizeof(private_def_buf));
    HCCL_KERNEL_INFO_PRIVATE_DEF* privateDefBuf = (HCCL_KERNEL_INFO_PRIVATE_DEF*)&private_def_buf[0];
    int64_t result_comm = (privateDefBuf->comm);
    u32 result_srcRank = (privateDefBuf->srcRank);
    u32 result_destRank = (privateDefBuf->destRank);
    u32 result_srTag = (privateDefBuf->srTag);
    EXPECT_EQ(result_type, RT_MODEL_TASK_HCCL);
    EXPECT_EQ(result_stream_id, streamId);
    EXPECT_EQ(result_hccl_hccl_type, type);
    EXPECT_EQ(result_comm, hcomComm);
    EXPECT_EQ(result_srcRank, 0);
    EXPECT_EQ(result_destRank, 0);
    EXPECT_EQ(result_srTag, 0);

    // -------------------HcomSend test----------------
    MOCKER(HcclCommGraphGetRankId).expects(atMost(8)).will(returnValue(HCCL_SUCCESS));
    type = HCCL_KERNEL_OP_TYPE_SEND;
    nodeptr->GetOpDesc()->SetType(type);
    ge::AttrUtils::HasAttr(nodeptr->GetOpDesc(), "DUMMY_SET_FALSE_TAG");
    ret = hcomKernelBuilder.GenerateTask(*nodeptr, runContext_dummy, taskDefList);
    EXPECT_EQ(ret, ge::SUCCESS);
    result_type = taskDefList[1].type();
    result_stream_id = taskDefList[1].stream_id();
    result_hccl_hccl_type = taskDefList[1].mutable_kernel_hccl()->hccl_type();
    result_private_def = taskDefList[1].private_def();
    sal_memcpy(&private_def_buf[0], sizeof(private_def_buf), result_private_def.c_str(), sizeof(private_def_buf));
    privateDefBuf = (HCCL_KERNEL_INFO_PRIVATE_DEF*)&private_def_buf[0];
    result_comm = (privateDefBuf->comm);
    result_srcRank = (privateDefBuf->srcRank);
    result_destRank = (privateDefBuf->destRank);
    result_srTag = (privateDefBuf->srTag);
    EXPECT_EQ(result_type, RT_MODEL_TASK_HCCL);
    EXPECT_EQ(result_stream_id, streamId);
    EXPECT_EQ(result_hccl_hccl_type, type);
    EXPECT_EQ(result_comm, hcomComm);
    EXPECT_EQ(result_srcRank, 0);
    EXPECT_EQ(result_destRank, tempInt);
    EXPECT_EQ(result_srTag, tempInt);

    // Send: 未设定 destRank 时，报错
    ge::AttrUtils::HasAttr(nodeptr->GetOpDesc(), "DUMMY_SET_FALSE_DESTRANK");
    ret = hcomKernelBuilder.GenerateTask(*nodeptr, runContext_dummy, taskDefList);
    EXPECT_EQ(ret, ge::INTERNAL_ERROR);
    ge::AttrUtils::HasAttr(nodeptr->GetOpDesc(), "DUMMY_SET_TRUE_DESTRANK");

    // Send: 未设定 srTag 时，报错
    ge::AttrUtils::HasAttr(nodeptr->GetOpDesc(), "DUMMY_SET_FALSE_SRTAG");
    ret = hcomKernelBuilder.GenerateTask(*nodeptr, runContext_dummy, taskDefList);
    EXPECT_EQ(ret, ge::INTERNAL_ERROR);
    ge::AttrUtils::HasAttr(nodeptr->GetOpDesc(), "DUMMY_SET_TRUE_SRTAG");

    // -------------------HcomReceive test----------------
    type = HCCL_KERNEL_OP_TYPE_RECEIVE;
    nodeptr->GetOpDesc()->SetType(type);
    ge::AttrUtils::HasAttr(nodeptr->GetOpDesc(), "DUMMY_SET_FALSE_TAG");
    ge::AttrUtils::HasAttr(nodeptr->GetOpDesc(), "DUMMY_SET_FALSE_GROUP");
    ret = hcomKernelBuilder.GenerateTask(*nodeptr, runContext_dummy, taskDefList);
    EXPECT_EQ(ret, ge::SUCCESS);
    result_type = taskDefList[2].type();
    result_stream_id = taskDefList[2].stream_id();
    result_hccl_hccl_type = taskDefList[2].mutable_kernel_hccl()->hccl_type();
    result_private_def = taskDefList[2].private_def();
    sal_memcpy(&private_def_buf[0], sizeof(private_def_buf), result_private_def.c_str(), sizeof(private_def_buf));
    privateDefBuf = (HCCL_KERNEL_INFO_PRIVATE_DEF*)&private_def_buf[0];
    result_comm = (privateDefBuf->comm);
    result_srcRank = (privateDefBuf->srcRank);
    result_destRank = (privateDefBuf->destRank);
    result_srTag = (privateDefBuf->srTag);
    EXPECT_EQ(result_type, RT_MODEL_TASK_HCCL);
    EXPECT_EQ(result_stream_id, streamId);
    EXPECT_EQ(result_hccl_hccl_type, type);
    EXPECT_EQ(result_comm, hcomComm);
    EXPECT_EQ(result_srcRank, tempInt);
    EXPECT_EQ(result_destRank, 0);
    EXPECT_EQ(result_srTag, tempInt);
    ge::AttrUtils::HasAttr(nodeptr->GetOpDesc(), "DUMMY_SET_TRUE_GROUP");

    // Receive: srcRank未设定时，报错
    ge::AttrUtils::HasAttr(nodeptr->GetOpDesc(), "DUMMY_SET_FALSE_SRCRANK");
    ret = hcomKernelBuilder.GenerateTask(*nodeptr, runContext_dummy, taskDefList);
    EXPECT_EQ(ret, ge::INTERNAL_ERROR);
    ge::AttrUtils::HasAttr(nodeptr->GetOpDesc(), "DUMMY_SET_TRUE_SRCRANK");

    // Receive: srTag未设定时，报错
    ge::AttrUtils::HasAttr(nodeptr->GetOpDesc(), "DUMMY_SET_FALSE_SRTAG");
    ret = hcomKernelBuilder.GenerateTask(*nodeptr, runContext_dummy, taskDefList);
    EXPECT_EQ(ret, ge::INTERNAL_ERROR);
    ge::AttrUtils::HasAttr(nodeptr->GetOpDesc(), "DUMMY_SET_TRUE_SRTAG");

    // -------------------HcomAllReduce test----------------
    type = HCCL_KERNEL_OP_TYPE_ALLREDUCE;
    nodeptr->GetOpDesc()->SetType(type);
    ge::AttrUtils::HasAttr(nodeptr->GetOpDesc(), "DUMMY_SET_FALSE_TAG");
    ret = hcomKernelBuilder.GenerateTask(*nodeptr, runContext_dummy, taskDefList);
    EXPECT_EQ(ret, ge::SUCCESS);
    ge::AttrUtils::HasAttr(nodeptr->GetOpDesc(), "DUMMY_SET_FALSE_TAG");

    // -------------------hcom_remote_read test----------------
    type = HCCL_KERNEL_OP_TYPE_REMOTE_READ;
    nodeptr->GetOpDesc()->SetType(type);
    ge::AttrUtils::HasAttr(nodeptr->GetOpDesc(), "DUMMY_SET_FALSE_TAG");
    ret = hcomKernelBuilder.GenerateTask(*nodeptr, runContext_dummy, taskDefList);
    EXPECT_EQ(ret, ge::SUCCESS);
    ge::AttrUtils::HasAttr(nodeptr->GetOpDesc(), "DUMMY_SET_FALSE_TAG");

    // -------------------incalid type test----------------
    type = " ";
    nodeptr->GetOpDesc()->SetType(type);
    ge::AttrUtils::HasAttr(nodeptr->GetOpDesc(), "DUMMY_SET_FALSE_TAG");
    ret = hcomKernelBuilder.GenerateTask(*nodeptr, runContext_dummy, taskDefList);
    EXPECT_EQ(ret, ge::INTERNAL_ERROR);
    ge::AttrUtils::HasAttr(nodeptr->GetOpDesc(), "DUMMY_SET_FALSE_COMM");
}

TEST_F(HcomKernelBuilderTest, ut_generateTask_by_comm_pytorch2)
{
    ge::NodePtr nodeptr(new NodeTest);
    ge::Buffer tempBuffer;
    ge::RunContext runContext_dummy;
    HcomOpsKernelBuilder hcomKernelBuilder;

    std::vector<domi::TaskDef> taskDefList;

    rtError_t rt_ret = RT_ERROR_NONE;
    rtStream_t stream;
    s64 streamId = 10000;
    nodeptr->GetOpDesc()->SetStreamId((s64)streamId);

    std::string name = "HcomTag";
    nodeptr->GetOpDesc()->SetName(name);

    // -------------------HcomBroadcast test----------------
    std::string type = HCCL_KERNEL_OP_TYPE_BROADCAST;
    nodeptr->GetOpDesc()->SetType(type);

    ge::AttrUtils::HasAttr(nodeptr->GetOpDesc(), "DUMMY_SET_TRUE_COMM");
    ge::AttrUtils::HasAttr(nodeptr->GetOpDesc(), "comm");
    int64_t hcomComm = 0;
    ge::AttrUtils::SetInt(nodeptr->GetOpDesc(), "comm", hcomComm);
    ge::AttrUtils::HasAttr(nodeptr->GetOpDesc(), "DUMMY_SET_TRUE_SRCRANK");
    ge::AttrUtils::HasAttr(nodeptr->GetOpDesc(), "DUMMY_SET_TRUE_DESTRANK");
    ge::AttrUtils::HasAttr(nodeptr->GetOpDesc(), "DUMMY_SET_TRUE_SRTAG");
    ge::AttrUtils::HasAttr(nodeptr->GetOpDesc(), "DUMMY_SET_FALSE_TAG");
    ge::AttrUtils::HasAttr(nodeptr->GetOpDesc(), "DUMMY_SET_TRUE_FISSION");
    ge::AttrUtils::HasAttr(nodeptr->GetOpDesc(), "DUMMY_SET_TRUE_DUMPSIZE");
    ge::AttrUtils::HasAttr(nodeptr->GetOpDesc(), "DUMMY_SET_TRUE_DUMPTYPE");
    s64 tempInt = 5;
    ge::AttrUtils::SetInt(nodeptr->GetOpDesc(), "dest_rank", tempInt);
    ge::AttrUtils::SetInt(nodeptr->GetOpDesc(), "src_rank", tempInt);
    ge::AttrUtils::SetInt(nodeptr->GetOpDesc(), "sr_tag", tempInt);
    ge::AttrUtils::SetInt(nodeptr->GetOpDesc(), "_fission_factor", 1);
    ge::AttrUtils::SetInt(nodeptr->GetOpDesc(), "global_workspace_size", 1);
    ge::AttrUtils::SetInt(nodeptr->GetOpDesc(), "global_workspace_type", 0);
    HCCL_INFO("node[%p] run context[%p]", nodeptr.get(), &runContext_dummy);
    HCCL_INFO("----------%s", nodeptr->GetOpDesc()->GetType().c_str());
    s32 ret = hcomKernelBuilder.GenerateTask(*nodeptr, runContext_dummy, taskDefList);
    EXPECT_EQ(ret, ge::SUCCESS);

    u32 result_type = taskDefList[0].type();
    u32 result_stream_id = taskDefList[0].stream_id();
    std::string result_hccl_hccl_type = taskDefList[0].mutable_kernel_hccl()->hccl_type();
    std::string result_private_def = taskDefList[0].private_def();
    char private_def_buf[sizeof(HCCL_KERNEL_INFO_PRIVATE_DEF)];
    sal_memcpy(&private_def_buf[0], sizeof(private_def_buf), result_private_def.c_str(), sizeof(private_def_buf));
    HCCL_KERNEL_INFO_PRIVATE_DEF* privateDefBuf = (HCCL_KERNEL_INFO_PRIVATE_DEF*)&private_def_buf[0];
    int64_t result_comm = (privateDefBuf->comm);
    u32 result_srcRank = (privateDefBuf->srcRank);
    u32 result_destRank = (privateDefBuf->destRank);
    u32 result_srTag = (privateDefBuf->srTag);
    EXPECT_EQ(result_type, RT_MODEL_TASK_HCCL);
    EXPECT_EQ(result_stream_id, streamId);
    EXPECT_EQ(result_hccl_hccl_type, type);
    EXPECT_EQ(result_comm, hcomComm);
    EXPECT_EQ(result_srcRank, 0);
    EXPECT_EQ(result_destRank, 0);
    EXPECT_EQ(result_srTag, 0);

    // -------------------HcomSend test----------------
    MOCKER(HcclCommGraphGetRankId).expects(atMost(8)).will(returnValue(HCCL_SUCCESS));
    type = HCCL_KERNEL_OP_TYPE_SEND;
    nodeptr->GetOpDesc()->SetType(type);
    ge::AttrUtils::HasAttr(nodeptr->GetOpDesc(), "DUMMY_SET_FALSE_TAG");
    ret = hcomKernelBuilder.GenerateTask(*nodeptr, runContext_dummy, taskDefList);
    EXPECT_EQ(ret, ge::SUCCESS);
    result_type = taskDefList[1].type();
    result_stream_id = taskDefList[1].stream_id();
    result_hccl_hccl_type = taskDefList[1].mutable_kernel_hccl()->hccl_type();
    result_private_def = taskDefList[1].private_def();
    sal_memcpy(&private_def_buf[0], sizeof(private_def_buf), result_private_def.c_str(), sizeof(private_def_buf));
    privateDefBuf = (HCCL_KERNEL_INFO_PRIVATE_DEF*)&private_def_buf[0];
    result_comm = (privateDefBuf->comm);
    result_srcRank = (privateDefBuf->srcRank);
    result_destRank = (privateDefBuf->destRank);
    result_srTag = (privateDefBuf->srTag);
    EXPECT_EQ(result_type, RT_MODEL_TASK_HCCL);
    EXPECT_EQ(result_stream_id, streamId);
    EXPECT_EQ(result_hccl_hccl_type, type);
    EXPECT_EQ(result_comm, hcomComm);
    EXPECT_EQ(result_srcRank, 0);
    EXPECT_EQ(result_destRank, tempInt);
    EXPECT_EQ(result_srTag, tempInt);

    // Send: 未设定 destRank 时，报错
    ge::AttrUtils::HasAttr(nodeptr->GetOpDesc(), "DUMMY_SET_FALSE_DESTRANK");
    ret = hcomKernelBuilder.GenerateTask(*nodeptr, runContext_dummy, taskDefList);
    EXPECT_EQ(ret, ge::INTERNAL_ERROR);
    ge::AttrUtils::HasAttr(nodeptr->GetOpDesc(), "DUMMY_SET_TRUE_DESTRANK");

    // Send: 未设定 srTag 时，报错
    ge::AttrUtils::HasAttr(nodeptr->GetOpDesc(), "DUMMY_SET_FALSE_SRTAG");
    ret = hcomKernelBuilder.GenerateTask(*nodeptr, runContext_dummy, taskDefList);
    EXPECT_EQ(ret, ge::INTERNAL_ERROR);
    ge::AttrUtils::HasAttr(nodeptr->GetOpDesc(), "DUMMY_SET_TRUE_SRTAG");

    // -------------------HcomReceive test----------------
    type = HCCL_KERNEL_OP_TYPE_RECEIVE;
    nodeptr->GetOpDesc()->SetType(type);
    ge::AttrUtils::HasAttr(nodeptr->GetOpDesc(), "DUMMY_SET_FALSE_TAG");
    ge::AttrUtils::HasAttr(nodeptr->GetOpDesc(), "DUMMY_SET_FALSE_GROUP");
    ret = hcomKernelBuilder.GenerateTask(*nodeptr, runContext_dummy, taskDefList);
    EXPECT_EQ(ret, ge::SUCCESS);
    result_type = taskDefList[2].type();
    result_stream_id = taskDefList[2].stream_id();
    result_hccl_hccl_type = taskDefList[2].mutable_kernel_hccl()->hccl_type();
    result_private_def = taskDefList[2].private_def();
    sal_memcpy(&private_def_buf[0], sizeof(private_def_buf), result_private_def.c_str(), sizeof(private_def_buf));
    privateDefBuf = (HCCL_KERNEL_INFO_PRIVATE_DEF*)&private_def_buf[0];
    result_comm = (privateDefBuf->comm);
    result_srcRank = (privateDefBuf->srcRank);
    result_destRank = (privateDefBuf->destRank);
    result_srTag = (privateDefBuf->srTag);
    EXPECT_EQ(result_type, RT_MODEL_TASK_HCCL);
    EXPECT_EQ(result_stream_id, streamId);
    EXPECT_EQ(result_hccl_hccl_type, type);
    EXPECT_EQ(result_comm, hcomComm);
    EXPECT_EQ(result_srcRank, tempInt);
    EXPECT_EQ(result_destRank, 0);
    EXPECT_EQ(result_srTag, tempInt);
    ge::AttrUtils::HasAttr(nodeptr->GetOpDesc(), "DUMMY_SET_TRUE_GROUP");

    // Receive: srcRank未设定时，报错
    ge::AttrUtils::HasAttr(nodeptr->GetOpDesc(), "DUMMY_SET_FALSE_SRCRANK");
    ret = hcomKernelBuilder.GenerateTask(*nodeptr, runContext_dummy, taskDefList);
    EXPECT_EQ(ret, ge::INTERNAL_ERROR);
    ge::AttrUtils::HasAttr(nodeptr->GetOpDesc(), "DUMMY_SET_TRUE_SRCRANK");

    // Receive: srTag未设定时，报错
    ge::AttrUtils::HasAttr(nodeptr->GetOpDesc(), "DUMMY_SET_FALSE_SRTAG");
    ret = hcomKernelBuilder.GenerateTask(*nodeptr, runContext_dummy, taskDefList);
    EXPECT_EQ(ret, ge::INTERNAL_ERROR);
    ge::AttrUtils::HasAttr(nodeptr->GetOpDesc(), "DUMMY_SET_TRUE_SRTAG");

    // -------------------HcomAllReduce test----------------
    type = HCCL_KERNEL_OP_TYPE_ALLREDUCE;
    nodeptr->GetOpDesc()->SetType(type);
    ge::AttrUtils::HasAttr(nodeptr->GetOpDesc(), "DUMMY_SET_FALSE_TAG");
    ret = hcomKernelBuilder.GenerateTask(*nodeptr, runContext_dummy, taskDefList);
    EXPECT_EQ(ret, ge::SUCCESS);
    ge::AttrUtils::HasAttr(nodeptr->GetOpDesc(), "DUMMY_SET_FALSE_TAG");

    // -------------------hcom_remote_read test----------------
    type = HCCL_KERNEL_OP_TYPE_REMOTE_READ;
    nodeptr->GetOpDesc()->SetType(type);
    ge::AttrUtils::HasAttr(nodeptr->GetOpDesc(), "DUMMY_SET_FALSE_TAG");
    ret = hcomKernelBuilder.GenerateTask(*nodeptr, runContext_dummy, taskDefList);
    EXPECT_EQ(ret, ge::SUCCESS);
    ge::AttrUtils::HasAttr(nodeptr->GetOpDesc(), "DUMMY_SET_FALSE_TAG");

    // -------------------incalid type test----------------
    type = " ";
    nodeptr->GetOpDesc()->SetType(type);
    ge::AttrUtils::HasAttr(nodeptr->GetOpDesc(), "DUMMY_SET_FALSE_TAG");
    ret = hcomKernelBuilder.GenerateTask(*nodeptr, runContext_dummy, taskDefList);
    EXPECT_EQ(ret, ge::INTERNAL_ERROR);
    ge::AttrUtils::HasAttr(nodeptr->GetOpDesc(), "DUMMY_SET_FALSE_COMM");
}

HcclResult FakeGetOffDeviceTypeWithoutDev(DevType& devType)
{
    devType = DevType::DEV_TYPE_910B;
    return HCCL_SUCCESS;
}

TEST_F(HcomKernelBuilderTest, ut_GetDevAndSerNumFromRankTable)
{
    HcclResult ret;
    nlohmann::json rank_table = rank_table_1server_8rank;
    char file_name_t[] = "./ut_task_num_one_server_hcom_test.json";
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
    HcomOpsKernelBuilder hcomKernelInfo;

    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    MOCKER(GetOffDeviceTypeWithoutDev).stubs().will(invoke(FakeGetOffDeviceTypeWithoutDev));

    char* rank_table_file = "./ut_task_num_one_server_hcom_test.json";
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

    s32 serverNum = 2;
    s32 deviceNumPerServer = 15;
    bool multiModuleDiffDeviceNumMode = false;
    ret = HcomOpUtils::GetDevAndSerNumFromRankTable(deviceNumPerServer, serverNum, multiModuleDiffDeviceNumMode);

    ret = HcomDestroy();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    remove(file_name_t);

    GlobalMockObject::verify();
}

TEST_F(HcomKernelBuilderTest, ut_GetCombineComTaskNum)
{
    s32 serverNum = 2;
    s32 deviceNumPerServer = 15;
    u32 intraTaskNum = 0;
    u32 interTaskNum = 0;

    HcclResult ret = HcomOpUtils::GetCombineComTaskNum(
        HCCL_KERNEL_OP_TYPE_ALLREDUCE, serverNum, deviceNumPerServer, intraTaskNum, interTaskNum);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(intraTaskNum, 0);
    EXPECT_EQ(interTaskNum, 551);
    ret = HcomOpUtils::GetCombineComTaskNum(
        HCCL_KERNEL_OP_TYPE_ALLGATHER, serverNum, deviceNumPerServer, intraTaskNum, interTaskNum);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(intraTaskNum, 0);
    EXPECT_EQ(interTaskNum, 261);
    ret = HcomOpUtils::GetCombineComTaskNum(
        HCCL_KERNEL_OP_TYPE_REDUCESCATTER, serverNum, deviceNumPerServer, intraTaskNum, interTaskNum);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(intraTaskNum, 0);
    EXPECT_EQ(interTaskNum, 319);
    ret = HcomOpUtils::GetCombineComTaskNum(
        HCCL_KERNEL_OP_TYPE_ALLTOALL, serverNum, deviceNumPerServer, intraTaskNum, interTaskNum);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(intraTaskNum, 0);
    EXPECT_EQ(interTaskNum, 406);
    ret = HcomOpUtils::GetCombineComTaskNum(
        HCCL_KERNEL_OP_TYPE_REMOTE_READ, serverNum, deviceNumPerServer, intraTaskNum, interTaskNum);
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);
    GlobalMockObject::verify();
}

TEST_F(HcomKernelBuilderTest, ut_getAlltoAllvStagedScratchMemSize)
{
    ge::NodePtr nodeptr(new NodeTest);
    HcomOpsKernelBuilder hcomKernelInfo;

    MOCKER_CPP(
        &HcomOpsKernelBuilder::GetAlltoAllCountsDispl,
        HcclResult(HcomOpsKernelBuilder::*)(
            ge::Node & node, std::vector<int64_t> & sendCounts, std::vector<int64_t> & sendDispls,
            std::vector<int64_t> & recvCounts, std::vector<int64_t> & recvDispls))
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    MOCKER(HcomGetAlltoAllStagedWorkSpaceMemSize).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    const string sGroup = "test_group";
    u64 getMemSize = 0;
    const int64_t hcomComm = 0;
    HcclResult ret = hcomKernelInfo.GetAlltoAllvStagedScratchMemSize(*(nodeptr.get()), hcomComm, sGroup, 4, getMemSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();

    std::vector<int64_t> sendCounts;
    ge::AttrUtils::SetListInt(nodeptr->GetOpDesc(), "send_counts", sendCounts);

    MOCKER_CPP(
        &HcomOpsKernelBuilder::GetAlltoAllCountsDispl,
        HcclResult(HcomOpsKernelBuilder::*)(
            const ge::OpDescPtr& op, std::vector<int64_t>& sendCounts, std::vector<int64_t>& sendDispls,
            std::vector<int64_t>& recvCounts, std::vector<int64_t>& recvDispls))
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    MOCKER(HcomGetAlltoAllStagedWorkSpaceMemSize).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    ret = hcomKernelInfo.GetAlltoAllvStagedScratchMemSize(*(nodeptr.get()), hcomComm, sGroup, 4, getMemSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
}

TEST_F(HcomKernelBuilderTest, ut_getReduceScatterVCountsDispl)
{
    ge::NodePtr nodeptr(new NodeTest);
    HcomOpsKernelBuilder hcomKernelInfo;

    std::vector<int64_t> sendCounts;
    std::vector<int64_t> sendDispls;
    std::vector<int64_t> recvCount;

    HcclResult ret = hcomKernelInfo.GetReduceScatterVCountsDispl(*(nodeptr.get()), sendCounts, sendDispls, recvCount);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

HcclResult stub_GetVectorFromTensor(const ge::GeTensor* tensor, std::vector<int64_t>& vector)
{
    vector.resize(4 * 4);
    return HCCL_SUCCESS;
}

TEST_F(HcomKernelBuilderTest, ut_getAlltoAllvcStagedScratchMemSize)
{
    ge::NodePtr nodeptr(new NodeTest);
    HcomOpUtils hcomKernelInfo;
    u32 rankSize = 4;

    MOCKER(&HcomOpUtils::GetVectorFromTensor).stubs().will(invoke(stub_GetVectorFromTensor));

    std::vector<int64_t> sendCountMatrix(16, 1);
    ge::AttrUtils::SetListInt(nodeptr->GetOpDesc(), "send_count_matrix", sendCountMatrix);
    u32 rankId = 0;

    MOCKER(HcomGetRankId).stubs().with(mockcpp::any(), outBound(&rankId)).will(returnValue(HCCL_SUCCESS));

    MOCKER(HcomGetAlltoAllvcStagedWorkSpaceMemSize).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    const string sGroup = "test_group";
    u64 getMemSize = 0;
    const int64_t hcomComm = 0;
    HcclResult ret
        = hcomKernelInfo.GetAlltoAllvcStagedScratchMemSize(*(nodeptr.get()), hcomComm, sGroup, 4, getMemSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
}

TEST_F(HcomKernelBuilderTest, ut_CheckAlltoAllvcRank)
{
    ge::NodePtr nodeptr(new NodeTest);
    HcomOpUtils hcomKernelInfo;
    u32 alltoallvcRank = 0;
    const string sGroup = "test_group";
    const int64_t hcomComm = 0;

    MOCKER(&HcomOpUtils::GetRankId).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    ge::AttrUtils::SetInt((*(nodeptr.get())).GetOpDesc(), "rank", alltoallvcRank);
    HcclResult ret = hcomKernelInfo.CheckAlltoAllvcRank(*(nodeptr.get()), hcomComm, sGroup);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    GlobalMockObject::verify();
}

#if 1
TEST_F(HcomKernelBuilderTest, ut_getAlltoAllCountsDispl)
{
    ge::NodePtr nodeptr(new NodeTest);
    HcomOpsKernelBuilder hcomKernelInfo;

    std::vector<int64_t> sendCounts;
    std::vector<int64_t> sendDispls;
    std::vector<int64_t> recvCounts;
    std::vector<int64_t> recvDispls;

    HcclResult ret
        = hcomKernelInfo.GetAlltoAllCountsDispl(*(nodeptr.get()), sendCounts, sendDispls, recvCounts, recvDispls);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}
#endif

TEST_F(HcomKernelBuilderTest, ut_getAllGatherVCountsDispl)
{
    ge::NodePtr nodeptr(new NodeTest);
    HcomOpsKernelBuilder hcomKernelInfo;

    std::vector<int64_t> sendCount;
    std::vector<int64_t> recvCounts;
    std::vector<int64_t> recvDispls;

    HcclResult ret = hcomKernelInfo.GetAllGatherVCountsDispl(*(nodeptr.get()), sendCount, recvCounts, recvDispls);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

#if 1
TEST_F(HcomKernelBuilderTest, ut_getAlltoAllCountsDispl_across_graph)
{
    ge::NodePtr nodeptr(new NodeTest);
    HcomOpsKernelBuilder hcomKernelInfo;
    MOCKER(&ge::OpDescUtils::GetInputConstData)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue((ge::ConstGeTensorBarePtr) nullptr));

    std::vector<int64_t> sendCounts;
    std::vector<int64_t> sendDispls;
    std::vector<int64_t> recvCounts;
    std::vector<int64_t> recvDispls;

    HcclResult ret
        = hcomKernelInfo.GetAlltoAllCountsDispl(*(nodeptr.get()), sendCounts, sendDispls, recvCounts, recvDispls);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

const std::vector<HcclAlgoType> GetExternalInputHcclAlgoConfig_stub1()
{
    static std::vector<HcclAlgoType> hcclAlgoConfig(4, HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT);
    hcclAlgoConfig[0] = HcclAlgoType::HCCL_ALGO_TYPE_FULLMESH;
    hcclAlgoConfig[1] = HcclAlgoType::HCCL_ALGO_TYPE_FULLMESH;
    hcclAlgoConfig[2] = HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT;
    hcclAlgoConfig[3] = HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT;
    return hcclAlgoConfig;
}

HcclResult GetDeviceTypeA2Stub(const char* group, DevType& deviceType)
{
    deviceType = DevType::DEV_TYPE_910B;
    return HCCL_SUCCESS;
}

TEST_F(HcomKernelBuilderTest, ut_getOpWorkspaceMemSize)
{
    ge::NodePtr nodeptr(new NodeTest);
    HcomOpsKernelBuilder hcomKernelInfo;

    MOCKER(GetExternalInputHcclAlgoConfig)
        .stubs()
        .with(mockcpp::any())
        .will(invoke(GetExternalInputHcclAlgoConfig_stub1));

    u64 opMemSize = 0;
    u32 rankSize = 3;
    MOCKER(HcomGetRankSize)
        .stubs()
        .with(mockcpp::any(), outBoundP(&rankSize, sizeof(rankSize)))
        .will(returnValue(HCCL_SUCCESS));

    MOCKER(HcomGetDevId).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER(hrtGetDeviceIndexByPhyId).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER(hrtSetDevice).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER(hrtResetDevice).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER(hrtCtxSetCurrent).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtCtxGetCurrent).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&HcomOpsKernelBuilder::GetAlltoAllvStagedScratchMemSize)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    HcclResult ret = hcomKernelInfo.GetOpWorkspaceMemSize(*nodeptr, HCCL_KERNEL_OP_TYPE_ALLTOALLV, opMemSize);

    MOCKER(GetDeviceType, HcclResult(const char*, DevType&)).stubs().will(invoke(GetDeviceTypeA2Stub));
    ret = hcomKernelInfo.GetOpWorkspaceMemSize(*nodeptr, HCCL_KERNEL_OP_TYPE_REDUCESCATTERV, opMemSize);
    EXPECT_EQ(HCCL_SUCCESS, ret);
    GlobalMockObject::verify();
}

ge::graphStatus
FakeGetOption2(ge::GEThreadLocalContext* that, const std::string& optionExec, std::string& dumpDebugValue)
{
    nlohmann::json group_list
        = {{{"group_name", "aa"}, {"group_rank_list", {0, 1}}},
           {{"group_name", "off_group_rank_list"}, {"group_rank_list", {0, 1, 2, 3, 4, 5, 6, 7}}}};
    if (optionExec == ge::OPTION_EXEC_HCOM_GROUPLIST) {
        dumpDebugValue = group_list.dump();
    } else if (optionExec == ge::OPTION_EXEC_HCOM_RANK_MAPPING) {
        dumpDebugValue = R"([{"rank_id": "0","device_index": [0,0,0]},{"rank_id": "1","device_index": [0,1,1]}])";
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
        dumpDebugValue = "Ascend910";
    }
    return ge::GRAPH_SUCCESS;
}

TEST_F(HcomKernelBuilderTest, ut_offlinebuild_calcSubStreamNum)
{
    ge::Status ret;
    HcomOpsKernelBuilder hcomOpsKernelInfoStore_;
    ge::NodePtr nodeptr(new NodeTest);
    std::string type = HCCL_KERNEL_OP_TYPE_ALLTOALLV;
    nodeptr->GetOpDesc()->SetType(type);

    std::string curGroup = "aa";
    ge::AttrUtils::SetStr(nodeptr->GetOpDesc(), "group", curGroup);

    MOCKER_CPP(&ge::GEThreadLocalContext::GetOption).stubs().will(invoke(FakeGetOption2));

    MOCKER(&ge::AttrUtils::SetInt).stubs().will(returnValue(false));

    ret = hcomOpsKernelInfoStore_.HcomCalcOpRunningParam(*nodeptr);

    MOCKER_CPP(&HcomOpsKernelBuilder::CalAndSetOpWorkerSpaceForKnowShape)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    type = HCCL_KERNEL_OP_TYPE_BROADCAST;
    std::string nodeName = "ALL_GATHER_NO_CALCULATION";
    nodeptr->GetOpDesc()->SetType(type);
    nodeptr->GetOpDesc()->SetName(nodeName);
    ret = hcomOpsKernelInfoStore_.HcomCalcOpRunningParam(*nodeptr);
    GlobalMockObject::verify();
}
#endif

TEST_F(HcomKernelBuilderTest, ut_offlinebuild_calcSubStreamNumAllToAllVC)
{
    HcclResult ret;
    HcomOpsKernelBuilder hcomOpsKernelInfoStore;
    ge::NodePtr nodeptr(new NodeTest);
    std::string type = HCCL_KERNEL_OP_TYPE_ALLTOALLVC;
    nodeptr->GetOpDesc()->SetType(type);

    std::string curGroup = "off_group_rank_list";
    ge::AttrUtils::HasAttr(nodeptr->GetOpDesc(), "DUMMY_SET_TRUE_GROUP");
    ge::AttrUtils::SetStr(nodeptr->GetOpDesc(), "group", curGroup);

    MOCKER_CPP(&ge::GEThreadLocalContext::GetOption).stubs().will(invoke(FakeGetOption2));

    u32 rankId = 0;
    MOCKER(HcomGetRankId).stubs().with(mockcpp::any(), outBound(&rankId)).will(returnValue(HCCL_SUCCESS));

    MOCKER(HcomGetRankSize).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER(&HcomOpUtils::CheckAlltoAllvcRank).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&HcomGraphOptimizer::GetOriginalGraphShapeTypeFromDesc).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER(HcomGetAlltoAllvcStagedWorkSpaceMemSize).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    ge::AttrUtils::HasAttr(nodeptr->GetOpDesc(), "DUMMY_SET_SEND_COUNT_MATRIX");
    ge::AttrUtils::SetInt(nodeptr->GetOpDesc(), "rank", 7);
    ret = hcomOpsKernelInfoStore.HcomCalcOpRunningParam(*nodeptr);
    EXPECT_EQ(ret, ge::SUCCESS);
    GlobalMockObject::verify();
}
#if 1
TEST_F(HcomKernelBuilderTest, ut_GenerateTaskDef)
{
    ge::NodePtr nodeptr(new NodeTest);
    HcomOpsKernelBuilder hcomKernelInfo;

    s64 streamId = 10000;
    s64 Id = 1;
    nodeptr->GetOpDesc()->SetStreamId((s64)streamId);
    std::string name = "HcomTag";
    nodeptr->GetOpDesc()->SetName(name);
    std::string type = HCCL_KERNEL_OP_TYPE_GATHER;
    nodeptr->GetOpDesc()->SetType(type);
    nodeptr->GetOpDesc()->SetId((s64)Id);
    HCCL_KERNEL_INFO_PRIVATE_DEF privateDefBuf;
    domi::TaskDef taskDef;

    s32 ret = hcomKernelInfo.GenerateTaskDef(*nodeptr, privateDefBuf, taskDef);
    EXPECT_EQ(ret, ge::SUCCESS);
}
#endif

ge::graphStatus
TaskNumGetOption(ge::GEThreadLocalContext* that, const std::string& optionExec, std::string& dumpDebugValue)
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
        dumpDebugValue = "Ascend910";
    } else if (optionExec == ge::OPTION_EXEC_RANK_TABLE_FILE) {
        dumpDebugValue = "./ut_task_num_one_server_hcom_test.json";
    } else if (optionExec == "ge.offline_hccl_compile") {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

TEST_F(HcomKernelBuilderTest, ut_CalcOpTaskNum)
{
    HcclResult ret;
    nlohmann::json rank_table = rank_table_910_2server_8rank;
    char file_name_t[] = "./ut_task_num_one_server_hcom_test.json";
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
    // HcomOpsKernelInfoStore hcomKernelInfo;
    HcomOpsKernelBuilder hcomKernelInfo;

    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    DevType deviceType = DevType::DEV_TYPE_910;
    MOCKER(hrtGetDeviceType).stubs().with(outBound(deviceType)).will(returnValue(HCCL_SUCCESS));

    char* rank_table_file = "./ut_task_num_one_server_hcom_test.json";
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
    ge_ret = hcomKernelInfo.CalcOpRunningParam(*nodeptr);
    EXPECT_EQ(ge_ret, ge::SUCCESS);

    type = HCCL_KERNEL_OP_TYPE_ALLGATHER;
    nodeptr->GetOpDesc()->SetType(type);
    ge_ret = hcomKernelInfo.CalcOpRunningParam(*nodeptr);
    EXPECT_EQ(ge_ret, ge::SUCCESS);

    type = HCCL_KERNEL_OP_TYPE_REDUCESCATTER;
    nodeptr->GetOpDesc()->SetType(type);
    ge_ret = hcomKernelInfo.CalcOpRunningParam(*nodeptr);
    EXPECT_EQ(ge_ret, ge::SUCCESS);

    MOCKER_CPP(&ge::GEThreadLocalContext::GetOption).stubs().will(invoke(TaskNumGetOption));

    type = HCCL_KERNEL_OP_TYPE_ALLREDUCE;
    nodeptr->GetOpDesc()->SetType(type);
    ge_ret = hcomKernelInfo.CalcOpRunningParam(*nodeptr);
    EXPECT_EQ(ge_ret, ge::SUCCESS);

    type = HCCL_KERNEL_OP_TYPE_ALLGATHER;
    nodeptr->GetOpDesc()->SetType(type);
    ge_ret = hcomKernelInfo.CalcOpRunningParam(*nodeptr);
    EXPECT_EQ(ge_ret, ge::SUCCESS);

    type = HCCL_KERNEL_OP_TYPE_REDUCESCATTER;
    nodeptr->GetOpDesc()->SetType(type);
    ge_ret = hcomKernelInfo.CalcOpRunningParam(*nodeptr);
    EXPECT_EQ(ge_ret, ge::SUCCESS);

    ret = HcomDestroy();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    remove(file_name_t);

    GlobalMockObject::verify();
}

ge::graphStatus
OfflineRankMappingOption1(ge::GEThreadLocalContext* that, const std::string& optionExec, std::string& dumpDebugValue)
{
    nlohmann::json group_list
        = {{{"group_name", "aa"}, {"group_rank_list", {0, 1}}},
           {{"group_name", "off_group_rank_list"}, {"group_rank_list", {0, 1, 2, 3, 4, 5, 6, 7}}}};
    if (optionExec == ge::OPTION_EXEC_HCOM_GROUPLIST) {
        dumpDebugValue = group_list.dump();
        return ge::GRAPH_SUCCESS;
    } else if (
        optionExec == "ge.exec.rankTable" || optionExec == "ge.offline_hccl_compile"
        || optionExec == "ge.exec.hcomRankMapping") {
        return ge::GRAPH_FAILED;
    } else if (optionExec == "ge.exec.rankMap") {
        dumpDebugValue
            = R"({"rank_map":[{"logic_rank_id":1,"model_rank_id":0},{"logic_rank_id":2,"model_rank_id":1}]})";
        return ge::GRAPH_SUCCESS;
    } else if (optionExec == "ge.socVersion") {
        dumpDebugValue = "Ascend910";
        return ge::GRAPH_SUCCESS;
    } else if (optionExec == "ge.exec.rankTableFile") {
        dumpDebugValue = "./ut_task_num_one_server_stream_test.json";
        return ge::GRAPH_SUCCESS;
    }
    dumpDebugValue.push_back('1');
    return ge::GRAPH_SUCCESS;
}

TEST_F(HcomKernelBuilderTest, ut_CalcOpTaskNum_1server_stream)
{
    HcclResult ret;
    nlohmann::json rank_table = rank_table_1server_8rank;
    char file_name_t[] = "./ut_task_num_one_server_stream_test.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(1) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    } else {
        HCCL_ERROR("open %s failed", file_name_t);
    }

    outfile.close();
    std::shared_ptr<hccl::hcclComm> comm;
    comm.reset(new (std::nothrow) hccl::hcclComm());

    ge::OpDesc op;
    ge ::Status ge_ret = ge::INTERNAL_ERROR;
    HcomOpsKernelBuilder hcomKernelInfo;

    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    DevType deviceType = DevType::DEV_TYPE_910;
    MOCKER(hrtGetDeviceType).stubs().with(outBound(deviceType)).will(returnValue(HCCL_SUCCESS));

    MOCKER(HcomGetCommByGroup).stubs().with(mockcpp::any(), outBound(comm)).will(returnValue(HCCL_SUCCESS));

    comm->deviceType_ = deviceType;
    ret = comm->GetDevType(deviceType);

    char* rank_table_file = "./ut_task_num_one_server_stream_test.json";
    char* rank_ID = "0";

    HcomInfo hcom_info;
    std::string rankTableM;
    std::string realFilePath;

    ret = HcomLoadRanktableFile(rank_table_file, rankTableM, realFilePath);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = CfgGetClusterInfo(rankTableM, rank_ID, hcom_info.params, hcom_info.rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = comm->init(hcom_info.params, hcom_info.rankTable);

    HCCL_INFO("HcomInitByFile START.");
    ret = HcomInitByFile(rank_table_file, rank_ID);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    HCCL_INFO("HcomInitByFile OK. pid[%d]", SalGetPid());

    ge::NodePtr nodeptr(new NodeTest);
    int64_t RANK_SIZE = 4;
    ge::AttrUtils::SetInt(nodeptr->GetOpDesc(), "rank_size", RANK_SIZE);
    std::string tempStr = "aa";
    ge::AttrUtils::HasAttr(nodeptr->GetOpDesc(), "DUMMY_SET_TRUE_GROUP");
    ge::AttrUtils::SetStr(nodeptr->GetOpDesc(), "group", tempStr);

    MOCKER_CPP(&ge::GEThreadLocalContext::GetOption).stubs().will(invoke(OfflineRankMappingOption1));
    std::string type;
    type = HCCL_KERNEL_OP_TYPE_ALLREDUCE;
    nodeptr->GetOpDesc()->SetType(type);
    std::string name = HCCL_KERNEL_OP_TYPE_ALLREDUCE + "1server";
    nodeptr->GetOpDesc()->SetName(name);
    ge_ret = hcomKernelInfo.CalcOpRunningParam(*nodeptr);
    EXPECT_EQ(ge_ret, ge::SUCCESS);

    s32 deviceNumPerServer = 8;
    s32 serverNum = 9;

    MOCKER(HcomOpUtils::GetDeviceAndServerNum)
        .stubs()
        //.with(mockcpp::any())
        .with(mockcpp::any(), outBound(deviceNumPerServer), outBound(serverNum), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    ge_ret = hcomKernelInfo.CalcOpRunningParam(*nodeptr);
    EXPECT_EQ(ge_ret, ge::SUCCESS);

    ret = HcomDestroy();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    remove(file_name_t);

    GlobalMockObject::verify();
}
TEST_F(HcomKernelBuilderTest, ut_CalcOpTaskNum_1server)
{
    HcclResult ret;
    nlohmann::json rank_table = rank_table_1server_8rank;
    char file_name_t[] = "./ut_task_num_one_server_hcom_test.json";
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
    HcomOpsKernelBuilder hcomKernelInfo;

    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    DevType deviceType = DevType::DEV_TYPE_910;
    MOCKER(hrtGetDeviceType).stubs().with(outBound(deviceType)).will(returnValue(HCCL_SUCCESS));

    char* rank_table_file = "./ut_task_num_one_server_hcom_test.json";
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
    ge_ret = hcomKernelInfo.CalcOpRunningParam(*nodeptr);
    EXPECT_EQ(ge_ret, ge::SUCCESS);

    type = HCCL_KERNEL_OP_TYPE_ALLGATHER;
    nodeptr->GetOpDesc()->SetType(type);
    ge_ret = hcomKernelInfo.CalcOpRunningParam(*nodeptr);
    EXPECT_EQ(ge_ret, ge::SUCCESS);

    type = HCCL_KERNEL_OP_TYPE_REDUCESCATTER;
    nodeptr->GetOpDesc()->SetType(type);
    ge_ret = hcomKernelInfo.CalcOpRunningParam(*nodeptr);
    EXPECT_EQ(ge_ret, ge::SUCCESS);

    MOCKER_CPP(&ge::GEThreadLocalContext::GetOption).stubs().will(invoke(TaskNumGetOption));

    type = HCCL_KERNEL_OP_TYPE_ALLREDUCE;
    nodeptr->GetOpDesc()->SetType(type);
    ge_ret = hcomKernelInfo.CalcOpRunningParam(*nodeptr);
    EXPECT_EQ(ge_ret, ge::SUCCESS);

    type = HCCL_KERNEL_OP_TYPE_ALLGATHER;
    nodeptr->GetOpDesc()->SetType(type);
    ge_ret = hcomKernelInfo.CalcOpRunningParam(*nodeptr);
    EXPECT_EQ(ge_ret, ge::SUCCESS);

    type = HCCL_KERNEL_OP_TYPE_REDUCESCATTER;
    nodeptr->GetOpDesc()->SetType(type);
    ge_ret = hcomKernelInfo.CalcOpRunningParam(*nodeptr);
    EXPECT_EQ(ge_ret, ge::SUCCESS);

    ret = HcomDestroy();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    remove(file_name_t);

    GlobalMockObject::verify();
}

TEST_F(HcomKernelBuilderTest, ut_GetAlgoLevel1)
{
    int ret = HCCL_SUCCESS;
    AlgTypeLevel1 algType1;
    std::string opType = "allreduce";

    MOCKER(LoadCannVersionInfoFile).stubs().will(returnValue(HCCL_SUCCESS));

    setenv("HCCL_ALGO", "level0:NA;level1:ring", 1);
    ret = HcomOpUtils::GetAlgoLevel1(8, opType, algType1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    unsetenv("HCCL_ALGO");

    setenv("HCCL_ALGO", "level0:NA;level1:null", 1);
    ret = HcomOpUtils::GetAlgoLevel1(8, opType, algType1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    unsetenv("HCCL_ALGO");

    setenv("HCCL_ALGO", "level1:ring", 1);
    ret = HcomOpUtils::GetAlgoLevel1(8, opType, algType1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    unsetenv("HCCL_ALGO");

    setenv("HCCL_ALGO", "level0:NA;level1:asd", 1);
    HcomOpUtils::GetAlgoLevel1(8, opType, algType1);
    unsetenv("HCCL_ALGO");

    GlobalMockObject::verify();
}

TEST_F(HcomKernelBuilderTest, ut_Hcom_SplitHcclOpType)
{
    int ret = HCCL_SUCCESS;
    std::string splitedConfig;
    std::string opType = "allreduce";

    std::string config1 = "allreduce=level0:NA;level1:ring/allgather=level0:NA;level1:NHR/"
                          "broadcast=level0:NA;level1:NHR/reducescatter=level0:NA;level1:NHR";
    std::string config2 = "allreduce=level0:NA;level1:ring";
    std::string config3 = "/allreduce=level0:NA;level1:ring";
    std::string config4 = "allreduce=level0:NA;level1:ring/";
    std::string config5 = "allreduce/level0:NA;level1:ring";

    ret = HcomOpUtils::SplitHcclOpType(config1, opType, splitedConfig);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcomOpUtils::SplitHcclOpType(config2, opType, splitedConfig);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcomOpUtils::SplitHcclOpType(config3, opType, splitedConfig);
    EXPECT_EQ(ret, HCCL_E_PARA);

    ret = HcomOpUtils::SplitHcclOpType(config4, opType, splitedConfig);
    EXPECT_EQ(ret, HCCL_E_PARA);

    ret = HcomOpUtils::SplitHcclOpType(config5, opType, splitedConfig);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(HcomKernelBuilderTest, ut_CalAndSetOpWorkerSpaceForKnowShape)
{
    u32 shapeType = ORIGINAL_GRAPH_KNOWNSHAPE_TYPE;
    MOCKER_CPP(&HcomOpsKernelBuilder::GetOriginalGraphShapeTypeFromDesc)
        .stubs()
        .with(mockcpp::any(), outBound(shapeType))
        .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&HcomOpsKernelBuilder::GetOpWorkspaceMemSize)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    ge::NodePtr nodeptr(new NodeTest);
    HcomOpUtils hcomKernelInfo;
    HcomOpsKernelBuilder KernelInfo;
    std::string sCollectiveType = "sCollectiveType";
    u64 opMemSize = 48000;

    KernelInfo.CalAndSetOpWorkerSpaceForKnowShape(*nodeptr, sCollectiveType, opMemSize);
    GlobalMockObject::verify();
}

TEST_F(HcomKernelBuilderTest, ut_TestAttachStream)
{
    ge::NodePtr nodeptr(new NodeTest);
    HcomOpUtils hcomKernelInfo;
    HcomOpsKernelBuilder KernelInfo;

    MOCKER(&GetExternalInputHcclAicpuUnfold).stubs().will(returnValue(true));
    EXPECT_EQ(KernelInfo.SetAttachedStreamInfoList(*nodeptr, "test_group"), HCCL_SUCCESS);

    GlobalMockObject::verify();
}

TEST_F(HcomKernelBuilderTest, ut_GenerateTaskAivCoreLimit)
{
    ge::NodePtr nodeptr(new NodeTest);
    ge::Buffer tempBuffer;
    ge::RunContext runContext_dummy;
    HcomOpsKernelBuilder hcomKernelInfo;

    std::vector<domi::TaskDef> taskDefList;

    rtError_t rt_ret = RT_ERROR_NONE;
    rtStream_t stream;
    s64 streamId = 10000;
    nodeptr->GetOpDesc()->SetStreamId((s64)streamId);

    std::string name = "HcomTag";
    nodeptr->GetOpDesc()->SetName(name);

    std::string type = HCCL_KERNEL_OP_TYPE_ALLGATHER;
    nodeptr->GetOpDesc()->SetType(type);

    ge::AttrUtils::HasAttr(nodeptr->GetOpDesc(), "DUMMY_SET_TRUE_GROUP");
    std::string tempStr = HCCL_WORLD_GROUP;
    ge::AttrUtils::SetStr(nodeptr->GetOpDesc(), "group", tempStr);
    ge::AttrUtils::HasAttr(nodeptr->GetOpDesc(), "DUMMY_SET_TRUE_SRCRANK");
    ge::AttrUtils::HasAttr(nodeptr->GetOpDesc(), "DUMMY_SET_TRUE_DESTRANK");
    ge::AttrUtils::HasAttr(nodeptr->GetOpDesc(), "DUMMY_SET_TRUE_SRTAG");
    ge::AttrUtils::HasAttr(nodeptr->GetOpDesc(), "DUMMY_SET_FALSE_TAG");
    ge::AttrUtils::HasAttr(nodeptr->GetOpDesc(), "DUMMY_SET_TRUE_FISSION");
    ge::AttrUtils::HasAttr(nodeptr->GetOpDesc(), "DUMMY_SET_TRUE_DUMPSIZE");
    ge::AttrUtils::HasAttr(nodeptr->GetOpDesc(), "DUMMY_SET_TRUE_DUMPTYPE");
    s64 tempInt = 5;
    ge::AttrUtils::SetInt(nodeptr->GetOpDesc(), "dest_rank", tempInt);
    ge::AttrUtils::SetInt(nodeptr->GetOpDesc(), "src_rank", tempInt);
    ge::AttrUtils::SetInt(nodeptr->GetOpDesc(), "sr_tag", tempInt);
    ge::AttrUtils::SetInt(nodeptr->GetOpDesc(), "_fission_factor", 1);
    ge::AttrUtils::SetInt(nodeptr->GetOpDesc(), "global_workspace_size", 1);
    ge::AttrUtils::SetInt(nodeptr->GetOpDesc(), "global_workspace_type", 0);
    ge::AttrUtils::HasAttr(nodeptr->GetOpDesc(), "DUMMY_SET_TRUE_NEEDMAPRANK");
    ge::AttrUtils::SetBool(nodeptr->GetOpDesc(), "_need_map_rank_id", true);

    // -------------------aiv core limit----------------
    // AivCoreLimit use default value 48
    HCCL_KERNEL_INFO_PRIVATE_DEF privateDef;
    MOCKER_CPP(&HcomOpsKernelBuilder::GenerateTaskPrivateDef)
        .stubs()
        .with(mockcpp::any(), spy(privateDef), mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    log_level_set_stub(DLOG_DEBUG);
    ge::AttrUtils::HasAttr(nodeptr->GetOpDesc(), "ATTR_OP_VECTORCORE_NUM_CLEAR");
    s32 ret = hcomKernelInfo.GenerateTask(*nodeptr, runContext_dummy, taskDefList);
    EXPECT_EQ(ret, ge::SUCCESS);
    EXPECT_EQ(privateDef.aivCoreLimit, 48U);

    // AivCoreLimit from ge.hardwareInfo
    std::string hardwareInfo = "ge.hardwareInfo";
    std::string hardwareInfoStr = "vector_core_cnt:5";
    MOCKER_CPP(&ge::GEContext::GetOption)
        .stubs()
        .with(eq(hardwareInfo), outBound(hardwareInfoStr))
        .will(returnValue(ge::GRAPH_SUCCESS));
    ret = hcomKernelInfo.GenerateTask(*nodeptr, runContext_dummy, taskDefList);
    EXPECT_EQ(ret, ge::SUCCESS);
    EXPECT_EQ(privateDef.aivCoreLimit, 5U);

    // AivCoreLimit from _op_vectorcore_num
    ge::AttrUtils::HasAttr(nodeptr->GetOpDesc(), "ATTR_OP_VECTORCORE_NUM");
    std::string aviCoreNum("4");
    ge::AttrUtils::SetStr(nodeptr->GetOpDesc(), "_op_vectorcore_num", aviCoreNum);
    ret = hcomKernelInfo.GenerateTask(*nodeptr, runContext_dummy, taskDefList);
    EXPECT_EQ(ret, ge::SUCCESS);
    EXPECT_EQ(privateDef.aivCoreLimit, 4U);

    GlobalMockObject::verify();
}
