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
#include "hcom_ops_kernel_info_store.h"
#include "hvd_ops_kernel_builder.h"
#undef private
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

class HvdKernelBuilderTest : public testing::Test {
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
        char file_name[] = "./ut_HvdKernelBuilderTest.json";

        std::ofstream outfile(file_name, std::ios::out | std::ios::trunc | std::ios::binary);

        if (outfile.is_open()) {
            HCCL_INFO("open %s success", file_name);
        } else {
            HCCL_ERROR("open %s failed", file_name);
        }

        outfile << std::setw(4) << rank_table << std::endl;
        outfile.close();

        std::cout << "\033[36m--HvdKernelBuilderTest SetUP--\033[0m" << std::endl;
    }
    static void TearDownTestCase()
    {
        char file_name[] = "./ut_HvdKernelBuilderTest.json";
        remove(file_name);
        std::cout << "\033[36m--HvdKernelBuilderTest TearDown--\033[0m" << std::endl;
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

#if 1
TEST_F(HvdKernelBuilderTest, ut_CalcOpRunningParam)
{
    HvdOpsKernelBuilder hvdKernelBuilder;
    HcclResult ret;
    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ge::NodePtr nodeptr(new NodeTest);
    ge ::Status ge_ret = ge::INTERNAL_ERROR;
    nodeptr->GetOpDesc()->SetType("INTERNAL_ERROR");
    ge_ret = hvdKernelBuilder.CalcOpRunningParam(*nodeptr);
    EXPECT_EQ(ge_ret, ge::INTERNAL_ERROR);

    std::string type = HVD_KERNEL_OP_TYPE_BROADCAST;
    nodeptr->GetOpDesc()->SetType(type);

    ge_ret = hvdKernelBuilder.CalcOpRunningParam(*nodeptr);
    EXPECT_EQ(ge_ret, ge::SUCCESS);

    type = HVD_KERNEL_OP_TYPE_ALLREDUCE;
    nodeptr->GetOpDesc()->SetType(type);
    ge_ret = hvdKernelBuilder.CalcOpRunningParam(*nodeptr);
    EXPECT_EQ(ge_ret, ge::SUCCESS);
}
#endif
TEST_F(HvdKernelBuilderTest, ut_generateTask)
{
    ge::NodePtr nodeptr(new NodeTest);
    ge::Buffer tempBuffer;
    ge::RunContext runContext_dummy;
    HvdOpsKernelBuilder hvdKernelBuilder;

    std::vector<domi::TaskDef> taskDefList;
    rtError_t rt_ret = RT_ERROR_NONE;
    rtStream_t stream;
    s64 streamId = 10000;

    nodeptr->GetOpDesc()->SetStreamId((s64)streamId);
    std::string type = HVD_KERNEL_OP_TYPE_ALLREDUCE;
    nodeptr->GetOpDesc()->SetType(type);
    ge::AttrUtils::HasAttr(nodeptr->GetOpDesc(), "DUMMY_SET_TRUE_EVENTID");
    std::string eventID = "event7";
    ge::AttrUtils::SetStr(nodeptr->GetOpDesc(), "event_id", eventID);

    ge ::Status ge_ret = ge::INTERNAL_ERROR;
    ge_ret = hvdKernelBuilder.GenerateTask(*nodeptr, runContext_dummy, taskDefList);
    EXPECT_EQ(ge_ret, ge::SUCCESS);
}
