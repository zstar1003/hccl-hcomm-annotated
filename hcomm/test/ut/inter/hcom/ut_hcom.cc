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
#include "hccl_communicator.h"
#include "hccl_comm_pub.h"
#include "topoinfo_ranktableParser_pub.h"
#include "topoinfo_ranktableStandard.h"
#include "topoinfo_roletableParser.h"
#undef protected
#undef private

#include "hcom_pub.h"
#include "comm.h"
#include "hccl/base.h"
#include <hccl/hccl_types.h>
#include "hccl/hcom.h"
#include "hccl/hcom_executor.h"

#include "stream_pub.h"
#include "mem_host_pub.h"
#include "mem_device_pub.h"

#include "gradient_segment.h"
#include "sal.h"
#include "llt_hccl_stub_pub.h"
#include "externalinput.h"
#include "hcom_private.h"
#include "config.h"
#include "sal.h"
#include "stream_pub.h"
#include "opexecounter_pub.h"

#include "externalinput_pub.h"
#include "../misc/ut_rank_table.h"
#include "ranktable/v80_rank_table.h"
#include "param_check_pub.h"
#include "dltrace_function.h"
#include <iostream>
#include <fstream>
#include "device_capacity.h"
#include "hcom_private.h"
#include "profiler_base_pub.h"
#include "param_check_pub_v2.h"
#include "op_base.h"

using namespace std;
using namespace hccl;

extern HcclResult HcomSetGradFusionByIndex(const char* group, u32 segmentNum, const u32* IdxList);
extern HcclResult HcomSetGradFusionBySize(const char* group, u32 segmentNum, const float* sizeList);
extern HcclResult HcomDestroyBackloggedGroup(const std::string& group);
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

class HcomTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        nlohmann::json rank_table = rank_table_910_2server_8rank;
        char file_name[] = "./ut_hcom.json";

        std::ofstream outfile(file_name, std::ios::out | std::ios::trunc | std::ios::binary);

        if (outfile.is_open()) {
            HCCL_INFO("open %s success", file_name);
        } else {
            HCCL_ERROR("open %s failed", file_name);
        }

        outfile << std::setw(4) << rank_table << std::endl;
        outfile.close();

        char file_name_v610[] = "./ut_hcom_v610.json";

        std::ofstream outfile_v610(file_name_v610, std::ios::out | std::ios::trunc | std::ios::binary);

        if (outfile.is_open()) {
            HCCL_INFO("open %s success", file_name_v610);
        } else {
            HCCL_ERROR("open %s failed", file_name_v610);
        }

        outfile_v610 << std::setw(4) << g_rank_table_610_2rank_1server << std::endl;
        outfile_v610.close();
        std::cout << "\033[36m--HcomTest SetUP--\033[0m" << std::endl;
    }
    static void TearDownTestCase()
    {
        char file_name[] = "./ut_hcom.json";
        remove(file_name);
        char file_name_v610[] = "./ut_hcom_v610.json";
        remove(file_name_v610);
        std::cout << "\033[36m--HcomTest TearDown--\033[0m" << std::endl;
    }
    // Some expensive resource shared by all tests.
    virtual void SetUp()
    {
        s32 portNum = 7;
        MOCKER(hrtGetHccsPortNum).stubs().with(mockcpp::any(), outBound(portNum)).will(returnValue(HCCL_SUCCESS));
        setenv("HCCL_OP_RETRY_ENABLE", "L0:0, L1:0, L2:0", 1);
        std::cout << "A Test SetUP" << std::endl;
    }
    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test TearDown" << std::endl;
    }
};

TEST_F(HcomTest, ut_hcom_broadcast)
{
    DlTraceFunction::GetInstance().DlTraceFunctionInit();
    HcclCommunicator impl;
    MOCKER_CPP_VIRTUAL(
        impl, &HcclCommunicator::Init,
        HcclResult(HcclCommunicator::*)(HcclCommParams & params, const RankTable_t& rankTable))
        .expects(atMost(1))
        .will(returnValue(0));

    char* rank_table_file = "./ut_hcom.json";
    char* rank_ID = "0";
    HcclResult ret = hrtSetDevice(0);
    ret = HcomInitByFile(rank_table_file, rank_ID);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();

    s8* sendbuf = (s8*)sal_malloc(10 * sizeof(s8));
    sal_memset(sendbuf, 10 * sizeof(s8), 0, 10 * sizeof(s8));

    rtStream_t stream;

    rtError_t rt_ret = aclrtCreateStream(&stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    MOCKER_CPP(&hcclComm::GetRankTableCrc).stubs().will(returnValue(0));

    MOCKER_CPP(&hcclComm::GetAlgType).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&hcclComm::GetNumBlocks).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&hcclComm::Broadcast).expects(atMost(1)).will(returnValue(0));
    ret = HcomBroadcast("tag", sendbuf, 10, HCCL_DATA_TYPE_INT8, 0, NULL, stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();

    MOCKER_CPP(&hcclComm::GetAlgType).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&hcclComm::Broadcast).expects(atMost(1)).will(returnValue(0));
    ret = HcomBroadcast("tag", sendbuf, 10, HCCL_DATA_TYPE_INT8, 1, NULL, stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();

    aclrtSynchronizeStream(stream);

    rt_ret = aclrtDestroyStream(stream);

    ret = HcomDestroy();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    sal_free(sendbuf);
    // remove(file_name);
}

TEST_F(HcomTest, ut_hcom_get_data_size)
{
    s32 ret = 0;

    ret = get_data_size(HCCL_DATA_TYPE_INT8);
    EXPECT_EQ(ret, 1);

    ret = get_data_size(HCCL_DATA_TYPE_INT32);
    EXPECT_EQ(ret, 4);

    ret = get_data_size(HCCL_DATA_TYPE_INT16);
    EXPECT_EQ(ret, 2);

    ret = get_data_size(HCCL_DATA_TYPE_FP32);
    EXPECT_EQ(ret, 4);

    ret = get_data_size(HCCL_DATA_TYPE_RESERVED);
    EXPECT_EQ(ret, 0);
}

TEST_F(HcomTest, ut_hcom_check_rank_id_reterr)
{
    HcclResult ret = HCCL_SUCCESS;
    ret = CheckRankId("ERR");
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(HcomTest, ut_hcom_cfg_check_file_path_test)
{
    s32 ret = 0;

    std::string file_path = "./testjson.json";
    std::string file_type = ".json";

    ret = CheckFilePath(file_path, file_type);
    EXPECT_EQ(ret, true);
}

TEST_F(HcomTest, ut_hcom_cfg_get_file_name_test)
{
    s32 ret = HCCL_SUCCESS;

    std::string file_path = "./testjson.json";
    std::string file_type = ".json";
    std::string file_name = "";

    ret = GetFileName(file_path, file_type, file_name);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ((file_name == "testjson"), true);
}

TEST_F(HcomTest, ut_hcom_CfgGetCcInfo_severnum0_ERR)
{
    nlohmann::json rank_table
        = {{"status", "completed"},
           {"deploy_mode", "lab"},
           {"group_count", "1"},
           {"chip_info", "910"},
           {"board_id", "0x0000"},
           {"para_plane_nic_location", "device"},
           {"para_plane_nic_num", "2"},
           {"para_plane_nic_name", {"eth0", "eth1"}},
           {"group_list",
            {{{"group_name", ""},
              {"device_num", "4"},
              {"server_num", "0"},
              {"instance_count", "4"},
              {"instance_list",
               {{{"rank_id", "0"},
                 {"server_id", "10.0.0.10"},
                 {"devices", {{{"device_id", "1"}, {"device_ip", "192.168.0.12"}}}}},
                {{"rank_id", "1"},
                 {"server_id", "10.0.0.10"},
                 {"devices", {{{"device_id", "0"}, {"device_ip", "192.168.1.12"}}}}},
                {{"rank_id", "2"},
                 {"server_id", "10.0.0.11"},
                 {"devices", {{{"device_id", "0"}, {"device_ip", "192.168.0.14"}}}}},
                {{"rank_id", "3"},
                 {"server_id", "10.0.0.11"},
                 {"devices", {{{"device_id", "1"}, {"device_ip", "192.168.1.14"}}}}}}},
              {"server_list",
               {
                   {{"server_id", "192.168.10.2"},
                    {"para_plane_info",
                     {{
                          {"eth1", "192.168.210.2"},
                      },
                      {
                          {"eth0", "192.168.200.2"},
                      }}}

                   },
                   {{"server_id", "192.168.10.3"},
                    {"para_plane_info",
                     {{
                          {"eth0", "192.168.200.3"},
                      },
                      {
                          {"eth1", "192.168.210.3"},
                      }}}

                   },

               }}}}}};

    char file_name[] = "./ut_hcom_CfgGetCcInfo_severnum0_ERR.json";
    std::ofstream outfile(file_name, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(4) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name);
    } else {
        HCCL_ERROR("open %s failed", file_name);
    }

    outfile.close();

    s32 ret = HCCL_SUCCESS;
    HcomInfo hcom;
    std::string identify = "0";
    std::string ranktable_file(file_name);
    std::string rankTableM;
    std::string realFilePath;
    ret = HcomLoadRanktableFile(ranktable_file.c_str(), rankTableM, realFilePath);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = CfgGetClusterInfo(rankTableM, identify, hcom.params, hcom.rankTable);
    EXPECT_EQ(ret, HCCL_E_PARA);
    remove(file_name);
}

TEST_F(HcomTest, ut_hcom_CfgGetCcInfo_group_count0_ERR)
{
    nlohmann::json rank_table
        = {{"status", "completed"},
           {"deploy_mode", "lab"},
           {"group_count", "0"},
           {"chip_info", "910"},
           {"board_id", "0x0000"},
           {"para_plane_nic_location", "device"},
           {"para_plane_nic_num", "2"},
           {"para_plane_nic_name", {"eth0", "eth1"}},
           {"group_list",
            {{{"group_name", ""},
              {"device_num", "4"},
              {"server_num", "4"},
              {"instance_count", "4"},
              {"instance_list",
               {{{"rank_id", "0"},
                 {"server_id", "10.0.0.10"},
                 {"devices", {{{"device_id", "1"}, {"device_ip", "192.168.0.12"}}}}},
                {{"rank_id", "1"},
                 {"server_id", "10.0.0.10"},
                 {"devices", {{{"device_id", "0"}, {"device_ip", "192.168.1.12"}}}}},
                {{"rank_id", "2"},
                 {"server_id", "10.0.0.11"},
                 {"devices", {{{"device_id", "0"}, {"device_ip", "192.168.0.14"}}}}},
                {{"rank_id", "3"},
                 {"server_id", "10.0.0.11"},
                 {"devices", {{{"device_id", "1"}, {"device_ip", "192.168.1.14"}}}}}}},
              {"server_list",
               {
                   {{"server_id", "192.168.10.2"},
                    {"para_plane_info",
                     {{
                          {"eth1", "192.168.210.2"},
                      },
                      {
                          {"eth0", "192.168.200.2"},
                      }}}

                   },
                   {{"server_id", "192.168.10.3"},
                    {"para_plane_info",
                     {{
                          {"eth0", "192.168.200.3"},
                      },
                      {
                          {"eth1", "192.168.210.3"},
                      }}}

                   },

               }}}}}};

    char file_name[] = "./ut_hcom_CfgGetCcInfo_group_count0_ERR.json";
    std::ofstream outfile(file_name, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(4) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name);
    } else {
        HCCL_ERROR("open %s failed", file_name);
    }

    outfile.close();

    s32 ret = HCCL_SUCCESS;
    HcomInfo hcom;
    std::string identify = "0";
    std::string ranktable_file(file_name);
    std::string rankTableM;
    std::string realFilePath;
    ret = HcomLoadRanktableFile(ranktable_file.c_str(), rankTableM, realFilePath);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = CfgGetClusterInfo(rankTableM, identify, hcom.params, hcom.rankTable);
    EXPECT_EQ(ret, HCCL_E_PARA);
    remove(file_name);
}

TEST_F(HcomTest, ut_hcom_CfgGetCcInfo_para_plane_nic_num0_ERR)
{
    nlohmann::json rank_table
        = {{"status", "completed"},
           {"deploy_mode", "lab"},
           {"group_count", "1"},
           {"chip_info", "910"},
           {"board_id", "0x0000"},
           {"para_plane_nic_location", "device"},
           {"para_plane_nic_num", "0"},
           {"para_plane_nic_name", {}},
           {"group_list",
            {{{"group_name", ""},
              {"device_num", "4"},
              {"server_num", "4"},
              {"instance_count", "4"},
              {"instance_list",
               {{{"rank_id", "0"},
                 {"server_id", "10.0.0.10"},
                 {"devices", {{{"device_id", "1"}, {"device_ip", "192.168.0.12"}}}}},
                {{"rank_id", "1"},
                 {"server_id", "10.0.0.10"},
                 {"devices", {{{"device_id", "0"}, {"device_ip", "192.168.1.12"}}}}},
                {{"rank_id", "2"},
                 {"server_id", "10.0.0.11"},
                 {"devices", {{{"device_id", "0"}, {"device_ip", "192.168.0.14"}}}}},
                {{"rank_id", "3"},
                 {"server_id", "10.0.0.11"},
                 {"devices", {{{"device_id", "1"}, {"device_ip", "192.168.1.14"}}}}}}},
              {"server_list",
               {
                   {{"server_id", "192.168.10.2"},
                    {"para_plane_info",
                     {{
                          {"eth1", "192.168.210.2"},
                      },
                      {
                          {"eth0", "192.168.200.2"},
                      }}}

                   },
                   {{"server_id", "192.168.10.3"},
                    {"para_plane_info",
                     {{
                          {"eth0", "192.168.200.3"},
                      },
                      {
                          {"eth1", "192.168.210.3"},
                      }}}

                   },

               }}}}}};

    char file_name[] = "./ut_hcom_CfgGetCcInfo_para_plane_nic_num0_ERR.json";
    std::ofstream outfile(file_name, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(4) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name);
    } else {
        HCCL_ERROR("open %s failed", file_name);
    }

    outfile.close();

    s32 ret = HCCL_SUCCESS;
    HcomInfo hcom;
    std::string identify = "0";
    std::string ranktable_file(file_name);
    std::string rankTableM;
    std::string realFilePath;
    ret = HcomLoadRanktableFile(ranktable_file.c_str(), rankTableM, realFilePath);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = CfgGetClusterInfo(rankTableM, identify, hcom.params, hcom.rankTable);
    EXPECT_EQ(ret, HCCL_E_UNAVAIL);
    remove(file_name);
}

TEST_F(HcomTest, ut_hcom_CfgGetCcInfo_device_num0_ERR)
{
    nlohmann::json rank_table
        = {{"status", "completed"},
           {"deploy_mode", "lab"},
           {"group_count", "1"},
           {"chip_info", "910"},
           {"board_id", "0x0000"},
           {"para_plane_nic_location", "device"},
           {"para_plane_nic_num", "1"},
           {"para_plane_nic_name", {}},
           {"group_list",
            {{{"group_name", ""},
              {"device_num", "0"},
              {"server_num", "4"},
              {"instance_count", "4"},
              {"instance_list",
               {{{"rank_id", "0"},
                 {"server_id", "10.0.0.10"},
                 {"devices", {{{"device_id", "1"}, {"device_ip", "192.168.0.12"}}}}},
                {{"rank_id", "1"},
                 {"server_id", "10.0.0.10"},
                 {"devices", {{{"device_id", "0"}, {"device_ip", "192.168.1.12"}}}}},
                {{"rank_id", "2"},
                 {"server_id", "10.0.0.11"},
                 {"devices", {{{"device_id", "0"}, {"device_ip", "192.168.0.14"}}}}},
                {{"rank_id", "3"},
                 {"server_id", "10.0.0.11"},
                 {"devices", {{{"device_id", "1"}, {"device_ip", "192.168.1.14"}}}}}}},
              {"server_list",
               {
                   {{"server_id", "192.168.10.2"},
                    {"para_plane_info",
                     {{
                          {"eth1", "192.168.210.2"},
                      },
                      {
                          {"eth0", "192.168.200.2"},
                      }}}

                   },
                   {{"server_id", "192.168.10.3"},
                    {"para_plane_info",
                     {{
                          {"eth0", "192.168.200.3"},
                      },
                      {
                          {"eth1", "192.168.210.3"},
                      }}}

                   },

               }}}}}};

    char file_name[] = "./ut_hcom_CfgGetCcInfo_device_num0_ERR.json";
    std::ofstream outfile(file_name, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(4) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name);
    } else {
        HCCL_ERROR("open %s failed", file_name);
    }

    outfile.close();

    s32 ret = HCCL_SUCCESS;
    HcomInfo hcom;
    std::string identify = "0";
    std::string ranktable_file(file_name);
    std::string rankTableM;
    std::string realFilePath;
    ret = HcomLoadRanktableFile(ranktable_file.c_str(), rankTableM, realFilePath);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = CfgGetClusterInfo(rankTableM, identify, hcom.params, hcom.rankTable);
    EXPECT_EQ(ret, HCCL_E_PARA);
    remove(file_name);
}

TEST_F(HcomTest, ut_hcom_CfgGetCcInfo_instance_count0_ERR)
{
    nlohmann::json rank_table
        = {{"status", "completed"},
           {"deploy_mode", "lab"},
           {"group_count", "1"},
           {"chip_info", "910"},
           {"board_id", "0x0000"},
           {"para_plane_nic_location", "device"},
           {"para_plane_nic_num", "1"},
           {"para_plane_nic_name", {}},
           {"group_list",
            {{{"group_name", ""},
              {"device_num", "4"},
              {"server_num", "4"},
              {"instance_count", "0"},
              {"instance_list",
               {{{"rank_id", "0"},
                 {"server_id", "10.0.0.10"},
                 {"devices", {{{"device_id", "1"}, {"device_ip", "192.168.0.12"}}}}},
                {{"rank_id", "1"},
                 {"server_id", "10.0.0.10"},
                 {"devices", {{{"device_id", "0"}, {"device_ip", "192.168.1.12"}}}}},
                {{"rank_id", "2"},
                 {"server_id", "10.0.0.11"},
                 {"devices", {{{"device_id", "0"}, {"device_ip", "192.168.0.14"}}}}},
                {{"rank_id", "3"},
                 {"server_id", "10.0.0.11"},
                 {"devices", {{{"device_id", "1"}, {"device_ip", "192.168.1.14"}}}}}}},
              {"server_list",
               {
                   {{"server_id", "192.168.10.2"},
                    {"para_plane_info",
                     {{
                          {"eth1", "192.168.210.2"},
                      },
                      {
                          {"eth0", "192.168.200.2"},
                      }}}

                   },
                   {{"server_id", "192.168.10.3"},
                    {"para_plane_info",
                     {{
                          {"eth0", "192.168.200.3"},
                      },
                      {
                          {"eth1", "192.168.210.3"},
                      }}}

                   },

               }}}}}};

    char file_name[] = "./ut_hcom_CfgGetCcInfo_instance_count0_ERR.json";
    std::ofstream outfile(file_name, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(4) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name);
    } else {
        HCCL_ERROR("open %s failed", file_name);
    }

    outfile.close();

    s32 ret = HCCL_SUCCESS;
    HcomInfo hcom;
    std::string identify = "0";
    std::string ranktable_file(file_name);
    std::string rankTableM;
    std::string realFilePath;
    ret = HcomLoadRanktableFile(ranktable_file.c_str(), rankTableM, realFilePath);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = CfgGetClusterInfo(rankTableM, identify, hcom.params, hcom.rankTable);
    EXPECT_EQ(ret, HCCL_E_PARA);
    remove(file_name);
}

#if 1
TEST_F(HcomTest, ut_hcom_get_invalid_jsonPropertyinfo)
{
    nlohmann::json rank_table
        = {{"status", "completed"},
           {"version", "1.0"},
           {"server_count", "1"},
           {"server",
            {{
                {"server_id", "10.0.0.10"},
                {"host_nic_ip", "192.168.0.12:0,192.168.1.12:199"},
                {"device",
                 {
                     {{"rank_id", "0"}, {"device_id", "0"}, {"device_ip", "192.168.0.12,192.168.0.13"}

                     },
                     {{"rank_id", "1"}, {"device_id", "1"}, {"device_ip", "192.168.1.12,192.168.1.13"}

                     },
                     {{"rank_id", "2"}, {"device_id", "2"}, {"device_ip", "192.168.2.12,192.168.2.13"}

                     },
                 }},
            }}}};

    char file_name_t[] = "./ut_hcom_get_invalid_jsonPropertyinfo.json";
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

    string rank_table_file("./ut_hcom_get_invalid_jsonPropertyinfo.json");

    HcomInfo hcom;
    std::string identify = "0";
    std::string rankTableM;
    std::string realFilePath;
    ret = HcomLoadRanktableFile(rank_table_file.c_str(), rankTableM, realFilePath);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = CfgGetClusterInfo(rankTableM, identify, hcom.params, hcom.rankTable);
    EXPECT_NE(ret, HCCL_SUCCESS);
    remove(file_name_t);
}
#endif

TEST_F(HcomTest, ut_hcom_load_rank_table_from_file_to_json_fail)
{
    s32 ret = HCCL_SUCCESS;
    char file_name[] = "./jobstart_hccl_invalid_json_file.json";
    std::ofstream outfile(file_name, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(4) << "invalid json format" << std::endl;
        HCCL_INFO("open %s success", file_name);
    } else {
        HCCL_ERROR("open %s failed", file_name);
    }

    outfile.close();
    HcomInfo hcom_info;
    std::string identify = "0";
    std::string ranktable_file(file_name);
    std::string rankTableM;
    std::string realFilePath;
    ret = HcomLoadRanktableFile(ranktable_file.c_str(), rankTableM, realFilePath);
    EXPECT_NE(ret, HCCL_SUCCESS);

    remove(file_name);
}

TEST_F(HcomTest, ut_hcom_JsonFile_LoadFile)
{
    s32 ret = HCCL_SUCCESS;

    HcomInfo hcom_info;
    std::string rankTableM;
    std::string realFilePath;
    std::string ranktable_file("");
    ret = HcomLoadRanktableFile(ranktable_file.c_str(), rankTableM, realFilePath);
    EXPECT_NE(ret, HCCL_SUCCESS);

    ranktable_file = "./testjson.txt";
    ret = HcomLoadRanktableFile(ranktable_file.c_str(), rankTableM, realFilePath);
    EXPECT_NE(ret, HCCL_SUCCESS);

    ranktable_file = "./testjson.json";
    ret = HcomLoadRanktableFile(ranktable_file.c_str(), rankTableM, realFilePath);
    EXPECT_NE(ret, HCCL_SUCCESS);

    ranktable_file = "./%s%%%ld.json";
    ret = HcomLoadRanktableFile(ranktable_file.c_str(), rankTableM, realFilePath);
    EXPECT_EQ(ret, HCCL_E_PARA);

    ranktable_file = "./*%%ld.json";
    ret = HcomLoadRanktableFile(ranktable_file.c_str(), rankTableM, realFilePath);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(HcomTest, ut_hcom_get_hcom_info_boardType2_alg0)
{
    nlohmann::json rank_table = rank_table_910_2server_8rank;

    setenv("HCCL_ALG_TYPE", "0", 1);
    char file_name[] = "./ut_hcom_get_hcom_info_boardType400.json";
    std::ofstream outfile(file_name, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(4) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name);
    } else {
        HCCL_ERROR("open %s failed", file_name);
    }

    outfile.close();

    std::string identify = "0";
    s32 ret = HCCL_SUCCESS;
    HcomInfo hcom_info;
    std::string ranktable_file(file_name);
    std::string rankTableM;
    std::string realFilePath;

    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcomLoadRanktableFile(ranktable_file.c_str(), rankTableM, realFilePath);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = CfgGetClusterInfo(rankTableM, identify, hcom_info.params, hcom_info.rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = hrtResetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    unsetenv("HCCL_ALG_TYPE");
    remove(file_name);
}

#if 1
TEST_F(HcomTest, ut_hcom_get_hcom_info_4rank2server_ERR)
{
    nlohmann::json rank_table = {
        {"status", "completed"},
        {"group_count", "1"},
        {"group_list",
         {{
             {"group_name", ""},
             {"instance_count", "2"},
             {"device_count", "8"},
             {"instance_list",
              {
                  {{"pod_name", "tf-0"},
                   {"server_id", "10.0.0.10"},
                   {"devices",
                    {{{"device_id", "0"}, {"device_ip", "192.168.0.10"}},
                     {{"device_id", "1"}, {"device_ip", "192.168.0.11"}},
                     {{"device_id", "2"}, {"device_ip", "192.168.0.12"}},
                     {{"device_id", "3"}, {"device_ip", "192.168.0.13"}}}}},
                  {{"pod_name", "tf-1"},
                   {"server_id", "10.0.0.11"},
                   {"devices",
                    {{{"device_id", "0"}, {"device_ip", "192.168.0.21"}},
                     {{"device_id", "1"}, {"device_ip", "192.168.0.22"}},
                     {{"device_id", "2"}, {"device_ip", "192.168.0.23"}},
                     {{"device_id", "3"}, {"device_ip", "192.168.0.24"}}}}},
              }},
         }}},
    };

    char file_name[] = "./ut_hcom_get_hcom_info_4rank2server_ERR.json";
    std::ofstream outfile(file_name, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(4) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name);
    } else {
        HCCL_ERROR("open %s failed", file_name);
    }

    outfile.close();

    ResetInitState();

    char* rank_ID = "tf-0";
    HcclResult ret = HcomInitByFile(file_name, rank_ID);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    HcomDestroy();
    ResetInitState();
    ret = HcomDestroy();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    remove(file_name);
}
#endif

HcclResult fake_CheckRanklistValid(std::vector<RankInfo_t>& rankList) { return HCCL_SUCCESS; }

TEST_F(HcomTest, ut_hcom_get_cloud_hcom_info2)
{
    nlohmann::json rank_table = {
        {"status", "completed"},
        {"group_count", "1"},
        {"group_list",
         {{
             {"group_name", ""},
             {"instance_count", "2"},
             {"device_count", "16"},
             {"instance_list",
              {
                  {{"pod_name", "tf-0"},
                   {"server_id", "10.0.0.10"},
                   {"devices",
                    {{{"device_id", "0"}, {"device_ip", "192.168.0.10"}},
                     {{"device_id", "1"}, {"device_ip", "192.168.0.11"}},
                     {{"device_id", "2"}, {"device_ip", "192.168.0.12"}},
                     {{"device_id", "3"}, {"device_ip", "192.168.0.13"}},
                     {{"device_id", "4"}, {"device_ip", "192.168.0.14"}},
                     {{"device_id", "5"}, {"device_ip", "192.168.0.15"}},
                     {{"device_id", "6"}, {"device_ip", "192.168.0.16"}},
                     {{"device_id", "7"}, {"device_ip", "192.168.0.17"}}}}},
                  {{"pod_name", "tf-1"},
                   {"server_id", "10.0.0.11"},
                   {"devices",
                    {{{"device_id", "0"}, {"device_ip", "192.168.0.21"}},
                     {{"device_id", "1"}, {"device_ip", "192.168.0.22"}},
                     {{"device_id", "2"}, {"device_ip", "192.168.0.23"}},
                     {{"device_id", "3"}, {"device_ip", "192.168.0.24"}},
                     {{"device_id", "4"}, {"device_ip", "192.168.0.20"}},
                     {{"device_id", "5"}, {"device_ip", "192.168.0.25"}},
                     {{"device_id", "6"}, {"device_ip", "192.168.0.26"}},
                     {{"device_id", "7"}, {"device_ip", "192.168.0.27"}}}}},
              }},
         }}},
    };

    char file_name[] = "./ut_hcom_get_cloud_hcom_info_boardid_ERR.json";
    std::ofstream outfile(file_name, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(4) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name);
    } else {
        HCCL_ERROR("open %s failed", file_name);
    }
    outfile.close();

    std::string identify = "tf-0";
    s32 ret = HCCL_SUCCESS;
    HcomInfo hcom_info;
    std::string ranktable_file(file_name);
    std::string rankTableM;
    std::string realFilePath;

    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcomLoadRanktableFile(ranktable_file.c_str(), rankTableM, realFilePath);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = CfgGetClusterInfo(rankTableM, identify, hcom_info.params, hcom_info.rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = hrtResetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    remove(file_name);
}

TEST_F(HcomTest, ut_hcom_get_cloud_hcom_info1)
{
    nlohmann::json rank_table = {
        {"status", "completed"},
        {"group_count", "1"},
        {"group_list",
         {{
             {"group_name", ""},
             {"instance_count", "2"},
             {"device_count", "16"},
             {"instance_list",
              {
                  {{"pod_name", "tf-0"},
                   {"server_id", "10.0.0.10"},
                   {"devices",
                    {{{"device_id", "0"}, {"device_ip", "192.168.0.10"}},
                     {{"device_id", "1"}, {"device_ip", "192.168.0.11"}},
                     {{"device_id", "2"}, {"device_ip", "192.168.0.12"}},
                     {{"device_id", "3"}, {"device_ip", "192.168.0.13"}},
                     {{"device_id", "4"}, {"device_ip", "192.168.0.14"}},
                     {{"device_id", "5"}, {"device_ip", "192.168.0.15"}},
                     {{"device_id", "6"}, {"device_ip", "192.168.0.16"}},
                     {{"device_id", "7"}, {"device_ip", "192.168.0.17"}}}}},
                  {{"pod_name", "tf-1"},
                   {"server_id", "10.0.0.11"},
                   {"devices",
                    {{{"device_id", "0"}, {"device_ip", "192.168.0.21"}},
                     {{"device_id", "1"}, {"device_ip", "192.168.0.22"}},
                     {{"device_id", "2"}, {"device_ip", "192.168.0.23"}},
                     {{"device_id", "3"}, {"device_ip", "192.168.0.24"}},
                     {{"device_id", "4"}, {"device_ip", "192.168.0.20"}},
                     {{"device_id", "5"}, {"device_ip", "192.168.0.25"}},
                     {{"device_id", "6"}, {"device_ip", "192.168.0.26"}},
                     {{"device_id", "7"}, {"device_ip", "192.168.0.27"}}}}},
              }},
         }}},
    };

    char file_name[] = "./ut_hcom_get_cloud_hcom_info_boardid_ERR.json";
    std::ofstream outfile(file_name, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(4) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name);
    } else {
        HCCL_ERROR("open %s failed", file_name);
    }
    outfile.close();

    std::string identify = "tf-0";
    s32 ret = HCCL_SUCCESS;
    HcomInfo hcom_info;
    std::string ranktable_file(file_name);
    std::string rankTableM;
    std::string realFilePath;

    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcomLoadRanktableFile(ranktable_file.c_str(), rankTableM, realFilePath);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = CfgGetClusterInfo(rankTableM, identify, hcom_info.params, hcom_info.rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = hrtResetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    remove(file_name);
}

TEST_F(HcomTest, ut_hcom_get_cloud_hcom_info_boardType2_)
{
    nlohmann::json rank_table = {
        {"status", "completed"},
        {"group_count", "1"},
        {"group_list",
         {{
             {"group_name", ""},
             {"instance_count", "2"},
             {"device_count", "16"},
             {"instance_list",
              {
                  {{"pod_name", "tf-0"},
                   {"server_id", "10.0.0.10"},
                   {"devices",
                    {{{"device_id", "0"}, {"device_ip", "192.168.0.10"}},
                     {{"device_id", "1"}, {"device_ip", "192.168.0.11"}},
                     {{"device_id", "2"}, {"device_ip", "192.168.0.12"}},
                     {{"device_id", "3"}, {"device_ip", "192.168.0.13"}},
                     {{"device_id", "4"}, {"device_ip", "192.168.0.14"}},
                     {{"device_id", "5"}, {"device_ip", "192.168.0.15"}},
                     {{"device_id", "6"}, {"device_ip", "192.168.0.16"}},
                     {{"device_id", "7"}, {"device_ip", "192.168.0.17"}}}}},
                  {{"pod_name", "tf-1"},
                   {"server_id", "10.0.0.11"},
                   {"devices",
                    {{{"device_id", "0"}, {"device_ip", "192.168.0.21"}},
                     {{"device_id", "1"}, {"device_ip", "192.168.0.22"}},
                     {{"device_id", "2"}, {"device_ip", "192.168.0.23"}},
                     {{"device_id", "3"}, {"device_ip", "192.168.0.24"}},
                     {{"device_id", "4"}, {"device_ip", "192.168.0.20"}},
                     {{"device_id", "5"}, {"device_ip", "192.168.0.25"}},
                     {{"device_id", "6"}, {"device_ip", "192.168.0.26"}},
                     {{"device_id", "7"}, {"device_ip", "192.168.0.27"}}}}},
              }},
         }}},
    };

    char file_name[] = "./ut_hcom_get_cloud_hcom_info_boardType2_.json";
    std::ofstream outfile(file_name, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(4) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name);
    } else {
        HCCL_ERROR("open %s failed", file_name);
    }
    outfile.close();
    std::string identify = "tf-0";
    s32 ret = HCCL_SUCCESS;
    HcomInfo hcom_info;
    std::string ranktable_file(file_name);
    std::string rankTableM;
    std::string realFilePath;

    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcomLoadRanktableFile(ranktable_file.c_str(), rankTableM, realFilePath);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = CfgGetClusterInfo(rankTableM, identify, hcom_info.params, hcom_info.rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = hrtResetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    remove(file_name);
}

TEST_F(HcomTest, ut_hcom_put_cloud_ranktable_info_other)
{
    nlohmann::json rank_table = {
        {"status", "completed"},
        {"group_count", "1"},
        {"group_list",
         {{
             {"group_name", ""},
             {"instance_count", "2"},
             {"device_count", "16"},
             {"instance_list",
              {
                  {{"pod_name", "tf-0"},
                   {"server_id", "10.0.0.10"},
                   {"devices",
                    {{{"device_id", "0"}, {"device_ip", "192.168.0.10"}},
                     {{"device_id", "1"}, {"device_ip", "192.168.0.11"}},
                     {{"device_id", "2"}, {"device_ip", "192.168.0.12"}},
                     {{"device_id", "3"}, {"device_ip", "192.168.0.13"}},
                     {{"device_id", "4"}, {"device_ip", "192.168.0.14"}},
                     {{"device_id", "5"}, {"device_ip", "192.168.0.15"}},
                     {{"device_id", "6"}, {"device_ip", "192.168.0.16"}},
                     {{"device_id", "7"}, {"device_ip", "192.168.0.17"}}}}},
                  {{"pod_name", "tf-1"},
                   {"server_id", "10.0.0.11"},
                   {"devices",
                    {{{"device_id", "0"}, {"device_ip", "192.168.0.21"}},
                     {{"device_id", "1"}, {"device_ip", "192.168.0.22"}},
                     {{"device_id", "2"}, {"device_ip", "192.168.0.23"}},
                     {{"device_id", "3"}, {"device_ip", "192.168.0.24"}},
                     {{"device_id", "4"}, {"device_ip", "192.168.0.25"}},
                     {{"device_id", "5"}, {"device_ip", "192.168.0.26"}},
                     {{"device_id", "6"}, {"device_ip", "192.168.0.27"}},
                     {{"device_id", "7"}, {"device_ip", "192.168.0.28"}}

                    }}},
              }},
         }}},
    };

    char file_name[] = "./ut_hcom_put_cloud_ranktable_info_other.json";
    std::ofstream outfile(file_name, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(4) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name);
    } else {
        HCCL_ERROR("open %s failed", file_name);
    }

    outfile.close();

    std::string identify = "tf-0";
    s32 ret = HCCL_SUCCESS;
    HcomInfo hcom_info;
    std::string ranktable_file(file_name);
    std::string rankTableM;
    std::string realFilePath;

    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcomLoadRanktableFile(ranktable_file.c_str(), rankTableM, realFilePath);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = CfgGetClusterInfo(rankTableM, identify, hcom_info.params, hcom_info.rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = DisplayCloudRankTableInfo(hcom_info.params, hcom_info.rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = hrtResetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    remove(file_name);
}

TEST_F(HcomTest, ut_hcom_get_hcom_info_boardType0)
{
    nlohmann::json rank_table = rank_table_910_2server_8rank;

    char file_name[] = "./ut_hcom_get_hcom_info_boardType0.json";
    std::ofstream outfile(file_name, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(4) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name);
    } else {
        HCCL_ERROR("open %s failed", file_name);
    }

    outfile.close();
    set_board_id(0x0000);
    outfile.close();

    std::string identify = "0";
    s32 ret = HCCL_SUCCESS;
    HcomInfo hcom_info;
    std::string ranktable_file(file_name);
    std::string rankTableM;
    std::string realFilePath;

    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcomLoadRanktableFile(ranktable_file.c_str(), rankTableM, realFilePath);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = CfgGetClusterInfo(rankTableM, identify, hcom_info.params, hcom_info.rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = hrtResetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    set_board_id(0);
    remove(file_name);
}

TEST_F(HcomTest, ut_hcom_get_hcom_info_nicLocationErr)
{
    nlohmann::json rank_table
        = {{"status", "completed"},
           {"deploy_mode", "lab"},
           {"group_count", "1"},
           {"chip_info", "910"},
           {"board_id", "0x0000"},
           {"para_plane_nic_location", "device"},
           {"para_plane_nic_num", "2"},
           {"para_plane_nic_name", {"eth0", "eth1"}},
           {"group_list",
            {{{"group_name", ""},
              {"device_num", "2"},
              {"server_num", "2"},
              {"instance_count", "2"},
              {"instance_list",
               {{{"rank_id", "0"},
                 {"server_id", "10.0.0.10"},
                 {"devices", {{{"device_id", "0"}, {"device_ip", "192.168.0.12"}}}}},
                {{"rank_id", "1"},
                 {"server_id", "10.0.0.11"},
                 {"devices", {{{"device_id", "0"}, {"device_ip", "192.168.0.12"}}}}}}},
              {"server_list",
               {
                   {{"server_id", "192.168.10.10"},
                    {"para_plane_info",
                     {{
                          {"eth1", "192.168.210.2"},
                      },
                      {
                          {"eth0", "192.168.200.2"},
                      }}}

                   },
                   {{"server_id", "192.168.10.11"},
                    {"para_plane_info",
                     {{
                          {"eth0", "192.168.200.3"},
                      },
                      {
                          {"eth1", "192.168.210.3"},
                      }}}

                   },

               }}}}}};

    char file_name[] = "./ut_hcom_get_hcom_info_nicLocationErr.json";
    std::ofstream outfile(file_name, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(4) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name);
    } else {
        HCCL_ERROR("open %s failed", file_name);
    }

    outfile.close();

    set_board_id(0x0000);
    std::string identify = "0";
    s32 ret = HCCL_SUCCESS;
    HcomInfo hcom_info;
    std::string ranktable_file(file_name);
    std::string rankTableM;
    std::string realFilePath;

    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcomLoadRanktableFile(ranktable_file.c_str(), rankTableM, realFilePath);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = CfgGetClusterInfo(rankTableM, identify, hcom_info.params, hcom_info.rankTable);
    EXPECT_EQ(ret, HCCL_E_PARA);

    ret = hrtResetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    set_board_id(0);
    remove(file_name);
}

#if 1
TEST_F(HcomTest, ut_hcom_get_hcom_info_serverNoExit)
{
    nlohmann::json rank_table
        = {{"status", "completed"},
           {"deploy_mode", "lab"},
           {"group_count", "1"},
           {"chip_info", "910"},
           {"board_id", "0x0000"},
           {"para_plane_nic_location", "device"},
           {"para_plane_nic_num", "8"},
           {"para_plane_nic_name", {"eth0", "eth1", "eth2", "eth3", "eth4", "eth5", "eth6", "eth7"}},
           {"group_list",
            {{{"group_name", ""},
              {"device_num", "16"},
              {"server_num", "2"},
              {"instance_count", "16"},
              {"instance_list",
               {
                   {{"rank_id", "0"},
                    {"server_id", "10.0.200.10"},
                    {"devices", {{{"device_id", "0"}, {"device_ip", "192.168.0.10"}}}}},

                   {{"rank_id", "1"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "1"}, {"device_ip", "192.168.0.11"}}}}},
                   {{"rank_id", "2"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "2"}, {"device_ip", "192.168.0.13"}}}}},

                   {{"rank_id", "3"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "3"}, {"device_ip", "192.168.0.14"}}}}},
                   {{"rank_id", "4"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "4"}, {"device_ip", "192.168.0.15"}}}}},

                   {{"rank_id", "5"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "5"}, {"device_ip", "192.168.0.16"}}}}},
                   {{"rank_id", "6"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "6"}, {"device_ip", "192.168.0.17"}}}}},

                   {{"rank_id", "7"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "7"}, {"device_ip", "192.168.0.18"}}}}},
                   {{"rank_id", "8"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "0"}, {"device_ip", "192.168.0.20"}}}}},

                   {{"rank_id", "9"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "1"}, {"device_ip", "192.168.0.21"}}}}},
                   {{"rank_id", "10"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "2"}, {"device_ip", "192.168.0.22"}}}}},

                   {{"rank_id", "11"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "3"}, {"device_ip", "192.168.0.23"}}}}},
                   {{"rank_id", "12"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "4"}, {"device_ip", "192.168.0.24"}}}}},

                   {{"rank_id", "13"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "5"}, {"device_ip", "192.168.0.25"}}}}},
                   {{"rank_id", "14"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "6"}, {"device_ip", "192.168.0.26"}}}}},

                   {{"rank_id", "15"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "7"}, {"device_ip", "192.168.0.27"}}}}},
               }},
              {"server_list",
               {
                   {{"server_id", "10.0.0.10"},
                    {"para_plane_info",
                     {
                         {
                             {"eth0", "192.168.200.2"},
                         },
                         {
                             {"eth1", "192.168.201.2"},
                         },
                         {
                             {"eth2", "192.168.202.2"},
                         },
                         {
                             {"eth3", "192.168.203.2"},
                         },
                         {
                             {"eth4", "192.168.204.2"},
                         },
                         {
                             {"eth5", "192.168.205.2"},
                         },
                         {
                             {"eth6", "192.168.206.2"},
                         },
                         {
                             {"eth7", "192.168.207.2"},
                         },
                     }}

                   },
                   {{"server_id", "10.0.0.11"},
                    {"para_plane_info",
                     {
                         {
                             {"eth0", "192.168.210.3"},
                         },
                         {
                             {"eth1", "192.168.211.3"},
                         },
                         {
                             {"eth2", "192.168.212.3"},
                         },
                         {
                             {"eth3", "192.168.213.3"},
                         },
                         {
                             {"eth4", "192.168.214.3"},
                         },
                         {
                             {"eth5", "192.168.215.3"},
                         },
                         {
                             {"eth6", "192.168.216.3"},
                         },
                         {
                             {"eth7", "192.168.217.3"},
                         },
                     }}

                   },

               }}}}}};

    char file_name[] = "./ut_hcom_get_hcom_info_serverNoExit.json";
    std::ofstream outfile(file_name, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(4) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name);
    } else {
        HCCL_ERROR("open %s failed", file_name);
    }

    outfile.close();

    set_board_id(0);
    std::string identify = "0";
    s32 ret = HCCL_SUCCESS;
    HcomInfo hcom_info;
    std::string ranktable_file(file_name);
    std::string rankTableM;
    std::string realFilePath;

    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcomLoadRanktableFile(ranktable_file.c_str(), rankTableM, realFilePath);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = CfgGetClusterInfo(rankTableM, identify, hcom_info.params, hcom_info.rankTable);
    EXPECT_EQ(ret, HCCL_E_PARA);

    ret = hrtResetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    set_board_id(0);
    remove(file_name);
}
#endif

TEST_F(HcomTest, ut_hcom_get_hcom_info_deviceIpErr)
{
    nlohmann::json rank_table
        = {{"status", "completed"},
           {"deploy_mode", "lab"},
           {"group_count", "1"},
           {"chip_info", "910"},
           {"board_id", "0x0000"},
           {"para_plane_nic_location", "device"},
           {"para_plane_nic_num", "8"},
           {"para_plane_nic_name", {"eth0", "eth1", "eth2", "eth3", "eth4", "eth5", "eth6", "eth7"}},
           {"group_list",
            {{{"group_name", ""},
              {"device_num", "16"},
              {"server_num", "2"},
              {"instance_count", "16"},
              {"instance_list",
               {
                   {{"rank_id", "0"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "0"}, {"device_ip", "192.168.0.11"}}}}},

                   {{"rank_id", "1"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "1"}, {"device_ip", "192.168.0.11"}}}}},
                   {{"rank_id", "2"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "2"}, {"device_ip", "192.168.0.13"}}}}},

                   {{"rank_id", "3"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "3"}, {"device_ip", "192.168.0.14"}}}}},
                   {{"rank_id", "4"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "4"}, {"device_ip", "192.168.0.15"}}}}},

                   {{"rank_id", "5"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "5"}, {"device_ip", "192.168.0.16"}}}}},
                   {{"rank_id", "6"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "6"}, {"device_ip", "192.168.0.17"}}}}},

                   {{"rank_id", "7"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "7"}, {"device_ip", "192.168.0.18"}}}}},
                   {{"rank_id", "8"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "0"}, {"device_ip", "192.168.0.20"}}}}},

                   {{"rank_id", "9"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "1"}, {"device_ip", "192.168.0.21"}}}}},
                   {{"rank_id", "10"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "2"}, {"device_ip", "192.168.0.22"}}}}},

                   {{"rank_id", "11"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "3"}, {"device_ip", "192.168.0.23"}}}}},
                   {{"rank_id", "12"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "4"}, {"device_ip", "192.168.0.24"}}}}},

                   {{"rank_id", "13"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "5"}, {"device_ip", "192.168.0.25"}}}}},
                   {{"rank_id", "14"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "6"}, {"device_ip", "192.168.0.26"}}}}},

                   {{"rank_id", "15"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "7"}, {"device_ip", "192.168.0.27"}}}}},
               }},
              {"server_list",
               {
                   {{"server_id", "10.0.0.10"},
                    {"para_plane_info",
                     {
                         {
                             {"eth0", "192.168.200.2"},
                         },
                         {
                             {"eth1", "192.168.201.2"},
                         },
                         {
                             {"eth2", "192.168.202.2"},
                         },
                         {
                             {"eth3", "192.168.203.2"},
                         },
                         {
                             {"eth4", "192.168.204.2"},
                         },
                         {
                             {"eth5", "192.168.205.2"},
                         },
                         {
                             {"eth6", "192.168.206.2"},
                         },
                         {
                             {"eth7", "192.168.207.2"},
                         },
                     }}

                   },
                   {{"server_id", "10.0.0.11"},
                    {"para_plane_info",
                     {
                         {
                             {"eth0", "192.168.210.3"},
                         },
                         {
                             {"eth1", "192.168.211.3"},
                         },
                         {
                             {"eth2", "192.168.212.3"},
                         },
                         {
                             {"eth3", "192.168.213.3"},
                         },
                         {
                             {"eth4", "192.168.214.3"},
                         },
                         {
                             {"eth5", "192.168.215.3"},
                         },
                         {
                             {"eth6", "192.168.216.3"},
                         },
                         {
                             {"eth7", "192.168.217.3"},
                         },
                     }}

                   },

               }}}}}};

    char file_name[] = "./ut_hcom_get_hcom_info_deviceIpErr.json";
    std::ofstream outfile(file_name, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(4) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name);
    } else {
        HCCL_ERROR("open %s failed", file_name);
    }

    outfile.close();

    set_board_id(0);
    std::string identify = "0";
    s32 ret = HCCL_SUCCESS;
    HcomInfo hcom_info;
    std::string ranktable_file(file_name);
    std::string rankTableM;
    std::string realFilePath;

    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcomLoadRanktableFile(ranktable_file.c_str(), rankTableM, realFilePath);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = CfgGetClusterInfo(rankTableM, identify, hcom_info.params, hcom_info.rankTable);
    EXPECT_EQ(ret, HCCL_E_PARA);

    ret = hrtResetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    set_board_id(0);
    remove(file_name);
}

TEST_F(HcomTest, ut_hcom_get_hcom_info_serverIdIpv4Err)
{
    nlohmann::json rank_table
        = {{"status", "completed"},
           {"deploy_mode", "lab"},
           {"group_count", "1"},
           {"chip_info", "910"},
           {"board_id", "0x0000"},
           {"para_plane_nic_location", "device"},
           {"para_plane_nic_num", "8"},
           {"para_plane_nic_name", {"eth0", "eth1", "eth2", "eth3", "eth4", "eth5", "eth6", "eth7"}},
           {"group_list",
            {{{"group_name", ""},
              {"device_num", "16"},
              {"server_num", "2"},
              {"instance_count", "16"},
              {"instance_list",
               {
                   {{"rank_id", "0"},
                    {"server_id", "10.277.0.10"},
                    {"devices", {{{"device_id", "0"}, {"device_ip", "192.168.0.11"}}}}},

                   {{"rank_id", "1"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "1"}, {"device_ip", "192.168.0.12"}}}}},
                   {{"rank_id", "2"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "2"}, {"device_ip", "192.168.0.13"}}}}},

                   {{"rank_id", "3"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "3"}, {"device_ip", "192.168.0.14"}}}}},
                   {{"rank_id", "4"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "4"}, {"device_ip", "192.168.0.15"}}}}},

                   {{"rank_id", "5"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "5"}, {"device_ip", "192.168.0.16"}}}}},
                   {{"rank_id", "6"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "6"}, {"device_ip", "192.168.0.17"}}}}},

                   {{"rank_id", "7"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "7"}, {"device_ip", "192.168.0.18"}}}}},
                   {{"rank_id", "8"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "0"}, {"device_ip", "192.168.0.20"}}}}},

                   {{"rank_id", "9"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "1"}, {"device_ip", "192.168.0.21"}}}}},
                   {{"rank_id", "10"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "2"}, {"device_ip", "192.168.0.22"}}}}},

                   {{"rank_id", "11"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "3"}, {"device_ip", "192.168.0.23"}}}}},
                   {{"rank_id", "12"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "4"}, {"device_ip", "192.168.0.24"}}}}},

                   {{"rank_id", "13"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "5"}, {"device_ip", "192.168.0.25"}}}}},
                   {{"rank_id", "14"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "6"}, {"device_ip", "192.168.0.26"}}}}},

                   {{"rank_id", "15"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "7"}, {"device_ip", "192.168.0.27"}}}}},
               }},
              {"server_list",
               {
                   {{"server_id", "10.0.0.10"},
                    {"para_plane_info",
                     {
                         {
                             {"eth0", "192.168.200.2"},
                         },
                         {
                             {"eth1", "192.168.201.2"},
                         },
                         {
                             {"eth2", "192.168.202.2"},
                         },
                         {
                             {"eth3", "192.168.203.2"},
                         },
                         {
                             {"eth4", "192.168.204.2"},
                         },
                         {
                             {"eth5", "192.168.205.2"},
                         },
                         {
                             {"eth6", "192.168.206.2"},
                         },
                         {
                             {"eth7", "192.168.207.2"},
                         },
                     }}

                   },
                   {{"server_id", "10.0.0.11"},
                    {"para_plane_info",
                     {
                         {
                             {"eth0", "192.168.210.3"},
                         },
                         {
                             {"eth1", "192.168.211.3"},
                         },
                         {
                             {"eth2", "192.168.212.3"},
                         },
                         {
                             {"eth3", "192.168.213.3"},
                         },
                         {
                             {"eth4", "192.168.214.3"},
                         },
                         {
                             {"eth5", "192.168.215.3"},
                         },
                         {
                             {"eth6", "192.168.216.3"},
                         },
                         {
                             {"eth7", "192.168.217.3"},
                         },
                     }}

                   },

               }}}}}};

    char file_name[] = "./ut_hcom_get_hcom_info_serverIdIpv4Err.json";
    std::ofstream outfile(file_name, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(4) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name);
    } else {
        HCCL_ERROR("open %s failed", file_name);
    }

    outfile.close();

    set_board_id(0x0000);
    std::string identify = "0";
    s32 ret = HCCL_SUCCESS;
    HcomInfo hcom_info;
    std::string ranktable_file(file_name);
    std::string rankTableM;
    std::string realFilePath;

    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcomLoadRanktableFile(ranktable_file.c_str(), rankTableM, realFilePath);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = CfgGetClusterInfo(rankTableM, identify, hcom_info.params, hcom_info.rankTable);
    EXPECT_EQ(ret, HCCL_E_PARA);

    ret = hrtResetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    set_board_id(0);
    remove(file_name);
}

TEST_F(HcomTest, ut_hcom_get_hcom_info_deviceIdErr)
{
    nlohmann::json rank_table
        = {{"status", "completed"},
           {"deploy_mode", "lab"},
           {"group_count", "1"},
           {"chip_info", "910"},
           {"board_id", "0x0000"},
           {"para_plane_nic_location", "device"},
           {"para_plane_nic_num", "8"},
           {"para_plane_nic_name", {"eth0", "eth1", "eth2", "eth3", "eth4", "eth5", "eth6", "eth7"}},
           {"group_list",
            {{{"group_name", ""},
              {"device_num", "16"},
              {"server_num", "2"},
              {"instance_count", "16"},
              {"instance_list",
               {
                   {{"rank_id", "0"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "0"}, {"device_ip", "192.168.0.11"}}}}},

                   {{"rank_id", "1"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "0"}, {"device_ip", "192.168.0.12"}}}}},
                   {{"rank_id", "2"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "2"}, {"device_ip", "192.168.0.13"}}}}},

                   {{"rank_id", "3"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "3"}, {"device_ip", "192.168.0.14"}}}}},
                   {{"rank_id", "4"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "4"}, {"device_ip", "192.168.0.15"}}}}},

                   {{"rank_id", "5"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "5"}, {"device_ip", "192.168.0.16"}}}}},
                   {{"rank_id", "6"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "6"}, {"device_ip", "192.168.0.17"}}}}},

                   {{"rank_id", "7"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "7"}, {"device_ip", "192.168.0.18"}}}}},
                   {{"rank_id", "8"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "0"}, {"device_ip", "192.168.0.20"}}}}},

                   {{"rank_id", "9"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "1"}, {"device_ip", "192.168.0.21"}}}}},
                   {{"rank_id", "10"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "2"}, {"device_ip", "192.168.0.22"}}}}},

                   {{"rank_id", "11"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "3"}, {"device_ip", "192.168.0.23"}}}}},
                   {{"rank_id", "12"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "4"}, {"device_ip", "192.168.0.24"}}}}},

                   {{"rank_id", "13"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "5"}, {"device_ip", "192.168.0.25"}}}}},
                   {{"rank_id", "14"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "6"}, {"device_ip", "192.168.0.26"}}}}},

                   {{"rank_id", "15"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "7"}, {"device_ip", "192.168.0.27"}}}}},
               }},
              {"server_list",
               {
                   {{"server_id", "10.0.0.10"},
                    {"para_plane_info",
                     {
                         {
                             {"eth0", "192.168.200.2"},
                         },
                         {
                             {"eth1", "192.168.201.2"},
                         },
                         {
                             {"eth2", "192.168.202.2"},
                         },
                         {
                             {"eth3", "192.168.203.2"},
                         },
                         {
                             {"eth4", "192.168.204.2"},
                         },
                         {
                             {"eth5", "192.168.205.2"},
                         },
                         {
                             {"eth6", "192.168.206.2"},
                         },
                         {
                             {"eth7", "192.168.207.2"},
                         },
                     }}

                   },
                   {{"server_id", "10.0.0.11"},
                    {"para_plane_info",
                     {
                         {
                             {"eth0", "192.168.210.3"},
                         },
                         {
                             {"eth1", "192.168.211.3"},
                         },
                         {
                             {"eth2", "192.168.212.3"},
                         },
                         {
                             {"eth3", "192.168.213.3"},
                         },
                         {
                             {"eth4", "192.168.214.3"},
                         },
                         {
                             {"eth5", "192.168.215.3"},
                         },
                         {
                             {"eth6", "192.168.216.3"},
                         },
                         {
                             {"eth7", "192.168.217.3"},
                         },
                     }}

                   },

               }}}}}};

    char file_name[] = "./ut_hcom_get_hcom_info_deviceIdErr.json";
    std::ofstream outfile(file_name, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(4) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name);
    } else {
        HCCL_ERROR("open %s failed", file_name);
    }

    outfile.close();

    set_board_id(0);
    std::string identify = "0";
    s32 ret = HCCL_SUCCESS;
    HcomInfo hcom_info;
    std::string ranktable_file(file_name);
    std::string rankTableM;
    std::string realFilePath;

    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcomInitByFile(file_name, identify.c_str());
    EXPECT_EQ(ret, HCCL_E_PARA);

    ret = HcomDestroy();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = hrtResetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    set_board_id(0);
    remove(file_name);
}

TEST_F(HcomTest, ut_hcom_get_hcom_info_eth0Ipv4Err)
{
    nlohmann::json rank_table
        = {{"status", "completed"},
           {"deploy_mode", "lab"},
           {"group_count", "1"},
           {"chip_info", "910"},
           {"board_id", "0x0000"},
           {"para_plane_nic_location", "device"},
           {"para_plane_nic_num", "8"},
           {"para_plane_nic_name", {"eth0", "eth1", "eth2", "eth3", "eth4", "eth5", "eth6", "eth7"}},
           {"group_list",
            {{{"group_name", ""},
              {"device_num", "16"},
              {"server_num", "2"},
              {"instance_count", "16"},
              {"instance_list",
               {
                   {{"rank_id", "0"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "0"}, {"device_ip", "192.168.0.xx.11"}}}}},

                   {{"rank_id", "1"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "1"}, {"device_ip", "192.168.0.12"}}}}},
                   {{"rank_id", "2"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "2"}, {"device_ip", "192.168.0.13"}}}}},

                   {{"rank_id", "3"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "3"}, {"device_ip", "192.168.0.14"}}}}},
                   {{"rank_id", "4"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "4"}, {"device_ip", "192.168.0.15"}}}}},

                   {{"rank_id", "5"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "5"}, {"device_ip", "192.168.0.16"}}}}},
                   {{"rank_id", "6"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "6"}, {"device_ip", "192.168.0.17"}}}}},

                   {{"rank_id", "7"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "7"}, {"device_ip", "192.168.0.18"}}}}},
                   {{"rank_id", "8"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "0"}, {"device_ip", "192.168.0.20"}}}}},

                   {{"rank_id", "9"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "1"}, {"device_ip", "192.168.0.21"}}}}},
                   {{"rank_id", "10"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "2"}, {"device_ip", "192.168.0.22"}}}}},

                   {{"rank_id", "11"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "3"}, {"device_ip", "192.168.0.23"}}}}},
                   {{"rank_id", "12"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "4"}, {"device_ip", "192.168.0.24"}}}}},

                   {{"rank_id", "13"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "5"}, {"device_ip", "192.168.0.25"}}}}},
                   {{"rank_id", "14"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "6"}, {"device_ip", "192.168.0.26"}}}}},

                   {{"rank_id", "15"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "7"}, {"device_ip", "192.168.0.27"}}}}},
               }},
              {"server_list",
               {
                   {{"server_id", "10.0.0.10"},
                    {"para_plane_info",
                     {
                         {
                             {"eth0", "192.168.2xx.2"},
                         },
                         {
                             {"eth1", "192.168.201.2"},
                         },
                         {
                             {"eth2", "192.168.202.2"},
                         },
                         {
                             {"eth3", "192.168.203.2"},
                         },
                         {
                             {"eth4", "192.168.204.2"},
                         },
                         {
                             {"eth5", "192.168.205.2"},
                         },
                         {
                             {"eth6", "192.168.206.2"},
                         },
                         {
                             {"eth7", "192.168.207.2"},
                         },
                     }}

                   },
                   {{"server_id", "10.0.0.11"},
                    {"para_plane_info",
                     {
                         {
                             {"eth0", "192.168.210.3"},
                         },
                         {
                             {"eth1", "192.168.211.3"},
                         },
                         {
                             {"eth2", "192.168.212.3"},
                         },
                         {
                             {"eth3", "192.168.213.3"},
                         },
                         {
                             {"eth4", "192.168.214.3"},
                         },
                         {
                             {"eth5", "192.168.215.3"},
                         },
                         {
                             {"eth6", "192.168.216.3"},
                         },
                         {
                             {"eth7", "192.168.217.3"},
                         },
                     }}

                   },

               }}}}}};

    char file_name[] = "./ut_hcom_get_hcom_info_eth0Ipv4Err.json";
    std::ofstream outfile(file_name, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(4) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name);
    } else {
        HCCL_ERROR("open %s failed", file_name);
    }

    outfile.close();

    set_board_id(0x0000);

    std::string identify = "0";
    s32 ret = HCCL_SUCCESS;
    HcomInfo hcom_info;
    std::string ranktable_file(file_name);
    std::string rankTableM;
    std::string realFilePath;

    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcomLoadRanktableFile(ranktable_file.c_str(), rankTableM, realFilePath);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = CfgGetClusterInfo(rankTableM, identify, hcom_info.params, hcom_info.rankTable);
    EXPECT_EQ(ret, HCCL_E_PARA);

    ret = hrtResetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    set_board_id(0);
    remove(file_name);
}

#if 1
TEST_F(HcomTest, ut_hcom_get_hcom_info_groupnameErr)
{
    nlohmann::json rank_table = {
        {"status", "completed"},
        {"group_count", "2"},
        {"group_list",
         {{
              {"group_name", "1"},
              {"instance_count", "1"},
              {"device_count", "8"},
              {"instance_list",
               {{{"pod_name", "tf-0"},
                 {"server_id", "10.0.0.10"},
                 {"devices",
                  {{{"device_id", "0"}, {"device_ip", "192.168.0.10"}},
                   {{"device_id", "1"}, {"device_ip", "192.168.0.11"}},
                   {{"device_id", "2"}, {"device_ip", "192.168.0.12"}},
                   {{"device_id", "3"}, {"device_ip", "192.168.0.13"}},
                   {{"device_id", "4"}, {"device_ip", "192.168.0.14"}},
                   {{"device_id", "5"}, {"device_ip", "192.168.0.15"}},
                   {{"device_id", "6"}, {"device_ip", "192.168.0.16"}},
                   {{"device_id", "7"}, {"device_ip", "192.168.0.17"}}}}}}},
          },
          {
              {"group_name", "1"},
              {"instance_count", "1"},
              {"device_count", "8"},
              {"instance_list",
               {
                   {{"pod_name", "tf-1"},
                    {"server_id", "10.0.0.11"},
                    {"devices",
                     {{{"device_id", "0"}, {"device_ip", "192.168.0.21"}},
                      {{"device_id", "1"}, {"device_ip", "192.168.0.22"}},
                      {{"device_id", "2"}, {"device_ip", "192.168.0.23"}},
                      {{"device_id", "3"}, {"device_ip", "192.168.0.24"}},
                      {{"device_id", "4"}, {"device_ip", "192.168.0.20"}},
                      {{"device_id", "5"}, {"device_ip", "192.168.0.25"}},
                      {{"device_id", "6"}, {"device_ip", "192.168.0.26"}},
                      {{"device_id", "7"}, {"device_ip", "192.168.0.27"}}}}},
               }},
          }}},
    };

    char file_name[] = "./ut_hcom_get_hcom_info_groupnameErr.json";
    std::ofstream outfile(file_name, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(4) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name);
    } else {
        HCCL_ERROR("open %s failed", file_name);
    }

    outfile.close();

    std::string identify = "0";
    s32 ret = HCCL_SUCCESS;
    HcomInfo hcom_info;
    std::string ranktable_file(file_name);
    std::string rankTableM;
    std::string realFilePath;

    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcomLoadRanktableFile(ranktable_file.c_str(), rankTableM, realFilePath);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = CfgGetClusterInfo(rankTableM, identify, hcom_info.params, hcom_info.rankTable);
    EXPECT_EQ(ret, HCCL_E_PARA);

    ret = hrtResetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    remove(file_name);
}
#endif

#if 1
TEST_F(HcomTest, ut_hcom_get_hcom_info_podnameErr)
{
    nlohmann::json rank_table = {
        {"status", "completed"},
        {"group_count", "1"},
        {"group_list",
         {{
             {"group_name", ""},
             {"instance_count", "2"},
             {"device_count", "16"},
             {"instance_list",
              {
                  {{"pod_name", "tf-0"},
                   {"server_id", "10.0.0.10"},
                   {"devices",
                    {{{"device_id", "0"}, {"device_ip", "192.168.0.10"}},
                     {{"device_id", "1"}, {"device_ip", "192.168.0.11"}},
                     {{"device_id", "2"}, {"device_ip", "192.168.0.12"}},
                     {{"device_id", "3"}, {"device_ip", "192.168.0.13"}},
                     {{"device_id", "4"}, {"device_ip", "192.168.0.14"}},
                     {{"device_id", "5"}, {"device_ip", "192.168.0.15"}},
                     {{"device_id", "6"}, {"device_ip", "192.168.0.16"}},
                     {{"device_id", "7"}, {"device_ip", "192.168.0.17"}}}}},
                  {{"pod_name", "tf-0"},
                   {"server_id", "10.0.0.11"},
                   {"devices",
                    {{{"device_id", "0"}, {"device_ip", "192.168.0.21"}},
                     {{"device_id", "1"}, {"device_ip", "192.168.0.22"}},
                     {{"device_id", "2"}, {"device_ip", "192.168.0.23"}},
                     {{"device_id", "3"}, {"device_ip", "192.168.0.24"}},
                     {{"device_id", "4"}, {"device_ip", "192.168.0.20"}},
                     {{"device_id", "5"}, {"device_ip", "192.168.0.25"}},
                     {{"device_id", "6"}, {"device_ip", "192.168.0.26"}},
                     {{"device_id", "7"}, {"device_ip", "192.168.0.27"}}}}},
              }},
         }}},
    };

    char file_name[] = "./ut_hcom_get_hcom_info_podnameErr.json";
    std::ofstream outfile(file_name, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(4) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name);
    } else {
        HCCL_ERROR("open %s failed", file_name);
    }

    outfile.close();

    std::string identify = "0";
    s32 ret = HCCL_SUCCESS;
    HcomInfo hcom_info;
    std::string ranktable_file(file_name);
    std::string rankTableM;
    std::string realFilePath;

    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcomLoadRanktableFile(ranktable_file.c_str(), rankTableM, realFilePath);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = CfgGetClusterInfo(rankTableM, identify, hcom_info.params, hcom_info.rankTable);
    EXPECT_EQ(ret, HCCL_E_PARA);
    ret = hrtResetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    remove(file_name);
}

#endif

TEST_F(HcomTest, ut_hcom_get_hcom_info_boardType400)
{
    nlohmann::json rank_table = rank_table_910_2server_8rank;

    char file_name[] = "./ut_hcom_get_hcom_info_boardType400.json";
    std::ofstream outfile(file_name, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(4) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name);
    } else {
        HCCL_ERROR("open %s failed", file_name);
    }

    outfile.close();

    std::string identify = "0";
    s32 ret = HCCL_SUCCESS;
    HcomInfo hcom_info;
    std::string ranktable_file(file_name);
    std::string rankTableM;
    std::string realFilePath;

    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcomLoadRanktableFile(ranktable_file.c_str(), rankTableM, realFilePath);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = CfgGetClusterInfo(rankTableM, identify, hcom_info.params, hcom_info.rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = hrtResetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    remove(file_name);
}

TEST_F(HcomTest, ut_hcom_get_hcom_info_boardType500)
{
    nlohmann::json rank_table = rank_table_910_2server_8rank;

    char file_name[] = "./ut_hcom_get_hcom_info_boardType500.json";
    std::ofstream outfile(file_name, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(4) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name);
    } else {
        HCCL_ERROR("open %s failed", file_name);
    }

    outfile.close();

    std::string identify = "0";
    s32 ret = HCCL_SUCCESS;
    HcomInfo hcom_info;
    std::string ranktable_file(file_name);
    std::string rankTableM;
    std::string realFilePath;

    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcomLoadRanktableFile(ranktable_file.c_str(), rankTableM, realFilePath);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = CfgGetClusterInfo(rankTableM, identify, hcom_info.params, hcom_info.rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = hrtResetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    remove(file_name);
}

TEST_F(HcomTest, ut_hcom_get_hcom_info_boardType500_put)
{
    nlohmann::json rank_table = rank_table_910_2server_8rank;

    char file_name[] = "./ut_hcom_get_hcom_info_boardType500_put.json";
    std::ofstream outfile(file_name, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(4) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name);
    } else {
        HCCL_ERROR("open %s failed", file_name);
    }

    outfile.close();

    HcomInfo hcom;

    std::string identify = "0";
    s32 ret = HCCL_SUCCESS;
    std::string ranktable_file(file_name);
    std::string rankTableM;
    std::string realFilePath;

    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcomLoadRanktableFile(ranktable_file.c_str(), rankTableM, realFilePath);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = CfgGetClusterInfo(rankTableM, identify, hcom.params, hcom.rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = hrtResetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    hcom.params.id.internal[0] = '\0';
    hcom.rankTable.nicNames.push_back("et0");
    ret = DisplayRanktableInfo(hcom.params, hcom.rankTable);
    ret = DisplayCloudRankTableInfo(hcom.params, hcom.rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    hcom.params.id.internal[0] = 'T';
    hcom.params.id.internal[1] = '\0';
    hcom.rankTable.nicNames.push_back("et0");
    ret = DisplayRanktableInfo(hcom.params, hcom.rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = DisplayCloudRankTableInfo(hcom.params, hcom.rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = DisplayRanktableInfo(hcom.params, hcom.rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = DisplayCloudRankTableInfo(hcom.params, hcom.rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = DisplayRanktableInfo(hcom.params, hcom.rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    hcom.params.id.internal[0] = 'T';
    hcom.params.id.internal[1] = '\0';
    hcom.rankTable.nicNames.push_back("et0");
    ret = DisplayRanktableInfo(hcom.params, hcom.rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = DisplayRanktableInfo(hcom.params, hcom.rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = DisplayRanktableInfo(hcom.params, hcom.rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = DisplayRanktableInfo(hcom.params, hcom.rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = DisplayRanktableInfo(hcom.params, hcom.rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    remove(file_name);
}

TEST_F(HcomTest, ut_hcom_get_hcom_info_boardType_arm880)
{
    nlohmann::json rank_table
        = {{"status", "completed"},
           {"deploy_mode", "lab"},
           {"group_count", "1"},
           {"chip_info", "910"},
           {"board_id", "0x002F"},
           {"para_plane_nic_location", "device"},
           {"para_plane_nic_num", "8"},
           {"para_plane_nic_name", {"eth0", "eth1", "eth2", "eth3", "eth4", "eth5", "eth6", "eth7"}},
           {"group_list",
            {{{"group_name", ""},
              {"device_num", "16"},
              {"server_num", "2"},
              {"instance_count", "16"},
              {"instance_list",
               {
                   {{"rank_id", "0"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "0"}, {"device_ip", "192.168.0.11"}}}}},

                   {{"rank_id", "1"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "1"}, {"device_ip", "192.168.0.12"}}}}},
                   {{"rank_id", "2"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "2"}, {"device_ip", "192.168.0.13"}}}}},

                   {{"rank_id", "3"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "3"}, {"device_ip", "192.168.0.14"}}}}},
                   {{"rank_id", "4"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "4"}, {"device_ip", "192.168.0.15"}}}}},

                   {{"rank_id", "5"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "5"}, {"device_ip", "192.168.0.16"}}}}},
                   {{"rank_id", "6"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "6"}, {"device_ip", "192.168.0.17"}}}}},

                   {{"rank_id", "7"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "7"}, {"device_ip", "192.168.0.18"}}}}},
                   {{"rank_id", "8"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "0"}, {"device_ip", "192.168.0.20"}}}}},

                   {{"rank_id", "9"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "1"}, {"device_ip", "192.168.0.21"}}}}},
                   {{"rank_id", "10"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "2"}, {"device_ip", "192.168.0.22"}}}}},

                   {{"rank_id", "11"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "3"}, {"device_ip", "192.168.0.23"}}}}},
                   {{"rank_id", "12"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "4"}, {"device_ip", "192.168.0.24"}}}}},

                   {{"rank_id", "13"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "5"}, {"device_ip", "192.168.0.25"}}}}},
                   {{"rank_id", "14"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "6"}, {"device_ip", "192.168.0.26"}}}}},

                   {{"rank_id", "15"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "7"}, {"device_ip", "192.168.0.27"}}}}},
               }},
              {"server_list",
               {
                   {{"server_id", "10.0.0.10"},
                    {"para_plane_info",
                     {
                         {
                             {"eth0", "192.168.200.2"},
                         },
                         {
                             {"eth1", "192.168.201.2"},
                         },
                         {
                             {"eth2", "192.168.202.2"},
                         },
                         {
                             {"eth3", "192.168.203.2"},
                         },
                         {
                             {"eth4", "192.168.204.2"},
                         },
                         {
                             {"eth5", "192.168.205.2"},
                         },
                         {
                             {"eth6", "192.168.206.2"},
                         },
                         {
                             {"eth7", "192.168.207.2"},
                         },
                     }}

                   },
                   {{"server_id", "10.0.0.11"},
                    {"para_plane_info",
                     {
                         {
                             {"eth0", "192.168.210.3"},
                         },
                         {
                             {"eth1", "192.168.211.3"},
                         },
                         {
                             {"eth2", "192.168.212.3"},
                         },
                         {
                             {"eth3", "192.168.213.3"},
                         },
                         {
                             {"eth4", "192.168.214.3"},
                         },
                         {
                             {"eth5", "192.168.215.3"},
                         },
                         {
                             {"eth6", "192.168.216.3"},
                         },
                         {
                             {"eth7", "192.168.217.3"},
                         },
                     }}

                   },

               }}}}}};

    char file_name[] = "./ut_hcom_get_hcom_info_boardType1000.json";
    std::ofstream outfile(file_name, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(4) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name);
    } else {
        HCCL_ERROR("open %s failed", file_name);
    }

    outfile.close();

    set_board_id(0x002F);
    std::string identify = "0";
    s32 ret = HCCL_SUCCESS;
    HcomInfo hcom_info;
    std::string ranktable_file(file_name);
    std::string rankTableM;
    std::string realFilePath;

    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcomLoadRanktableFile(ranktable_file.c_str(), rankTableM, realFilePath);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = CfgGetClusterInfo(rankTableM, identify, hcom_info.params, hcom_info.rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = hrtResetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    set_board_id(0);

    remove(file_name);
}

TEST_F(HcomTest, ut_hcom_get_hcom_info_boardType_arm880_1)
{
    nlohmann::json rank_table
        = {{"status", "completed"},
           {"deploy_mode", "lab"},
           {"group_count", "1"},
           {"chip_info", "910"},
           {"board_id", "0x002f"},
           {"para_plane_nic_location", "device"},
           {"para_plane_nic_num", "8"},
           {"para_plane_nic_name", {"eth0", "eth1", "eth2", "eth3", "eth4", "eth5", "eth6", "eth7"}},
           {"group_list",
            {{{"group_name", ""},
              {"device_num", "16"},
              {"server_num", "2"},
              {"instance_count", "16"},
              {"instance_list",
               {
                   {{"rank_id", "0"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "0"}, {"device_ip", "192.168.0.11"}}}}},

                   {{"rank_id", "1"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "1"}, {"device_ip", "192.168.0.12"}}}}},
                   {{"rank_id", "2"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "2"}, {"device_ip", "192.168.0.13"}}}}},

                   {{"rank_id", "3"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "3"}, {"device_ip", "192.168.0.14"}}}}},
                   {{"rank_id", "4"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "4"}, {"device_ip", "192.168.0.15"}}}}},

                   {{"rank_id", "5"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "5"}, {"device_ip", "192.168.0.16"}}}}},
                   {{"rank_id", "6"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "6"}, {"device_ip", "192.168.0.17"}}}}},

                   {{"rank_id", "7"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "7"}, {"device_ip", "192.168.0.18"}}}}},
                   {{"rank_id", "8"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "0"}, {"device_ip", "192.168.0.20"}}}}},

                   {{"rank_id", "9"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "1"}, {"device_ip", "192.168.0.21"}}}}},
                   {{"rank_id", "10"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "2"}, {"device_ip", "192.168.0.22"}}}}},

                   {{"rank_id", "11"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "3"}, {"device_ip", "192.168.0.23"}}}}},
                   {{"rank_id", "12"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "4"}, {"device_ip", "192.168.0.24"}}}}},

                   {{"rank_id", "13"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "5"}, {"device_ip", "192.168.0.25"}}}}},
                   {{"rank_id", "14"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "6"}, {"device_ip", "192.168.0.26"}}}}},

                   {{"rank_id", "15"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "7"}, {"device_ip", "192.168.0.27"}}}}},
               }},
              {"server_list",
               {
                   {{"server_id", "10.0.0.10"},
                    {"para_plane_info",
                     {
                         {
                             {"eth0", "192.168.200.2"},
                         },
                         {
                             {"eth1", "192.168.201.2"},
                         },
                         {
                             {"eth2", "192.168.202.2"},
                         },
                         {
                             {"eth3", "192.168.203.2"},
                         },
                         {
                             {"eth4", "192.168.204.2"},
                         },
                         {
                             {"eth5", "192.168.205.2"},
                         },
                         {
                             {"eth6", "192.168.206.2"},
                         },
                         {
                             {"eth7", "192.168.207.2"},
                         },
                     }}

                   },
                   {{"server_id", "10.0.0.11"},
                    {"para_plane_info",
                     {
                         {
                             {"eth0", "192.168.210.3"},
                         },
                         {
                             {"eth1", "192.168.211.3"},
                         },
                         {
                             {"eth2", "192.168.212.3"},
                         },
                         {
                             {"eth3", "192.168.213.3"},
                         },
                         {
                             {"eth4", "192.168.214.3"},
                         },
                         {
                             {"eth5", "192.168.215.3"},
                         },
                         {
                             {"eth6", "192.168.216.3"},
                         },
                         {
                             {"eth7", "192.168.217.3"},
                         },
                     }}

                   },

               }}}}}};

    char file_name[] = "./ut_hcom_get_hcom_info_boardType1000.json";
    std::ofstream outfile(file_name, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(4) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name);
    } else {
        HCCL_ERROR("open %s failed", file_name);
    }

    outfile.close();
    set_board_id(0x002F);
    std::string identify = "0";
    s32 ret = HCCL_SUCCESS;
    HcomInfo hcom_info;
    std::string ranktable_file(file_name);
    std::string rankTableM;
    std::string realFilePath;

    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcomLoadRanktableFile(ranktable_file.c_str(), rankTableM, realFilePath);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = CfgGetClusterInfo(rankTableM, identify, hcom_info.params, hcom_info.rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = hrtResetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    set_board_id(0);

    remove(file_name);
}

TEST_F(HcomTest, ut_hcom_get_hcom_info_device_per_server_err)
{
    nlohmann::json rank_table
        = {{"status", "completed"},
           {"deploy_mode", "lab"},
           {"group_count", "1"},
           {"chip_info", "910"},
           {"board_id", "0x0000"},
           {"para_plane_nic_location", "device"},
           {"para_plane_nic_num", "2"},
           {"para_plane_nic_name", {"eth0", "eth1"}},
           {"group_list",
            {{{"group_name", ""},
              {"device_num", "3"},
              {"server_num", "2"},
              {"instance_count", "4"},
              {"instance_list",
               {{{"rank_id", "0"},
                 {"server_id", "10.0.0.10"},
                 {"devices", {{{"device_id", "1"}, {"device_ip", "192.168.0.12"}}}}},
                {{"rank_id", "1"},
                 {"server_id", "10.0.0.11"},
                 {"devices", {{{"device_id", "0"}, {"device_ip", "192.168.0.11"}}}}},
                {{"rank_id", "2"},
                 {"server_id", "10.0.0.11"},
                 {"devices", {{{"device_id", "0"}, {"device_ip", "192.168.0.11"}}}}},
                {{"rank_id", "3"},
                 {"server_id", "10.0.0.11"},
                 {"devices", {{{"device_id", "0"}, {"device_ip", "192.168.0.11"}}}}}}},
              {"server_list",
               {
                   {{"server_id", "192.168.10.2"},
                    {"para_plane_info",
                     {{
                          {"eth1", "192.168.210.2"},
                      },
                      {
                          {"eth0", "192.168.200.2"},
                      }}}

                   },
                   {{"server_id", "192.168.10.3"},
                    {"para_plane_info",
                     {{
                          {"eth0", "192.168.200.3"},
                      },
                      {
                          {"eth1", "192.168.210.3"},
                      }}}

                   },

               }}}}}};

    char file_name[] = "./ut_hcom_get_hcom_info_device_per_server_err.json";
    std::ofstream outfile(file_name, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(4) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name);
    } else {
        HCCL_ERROR("open %s failed", file_name);
    }

    outfile.close();

    std::string identify = "0";
    s32 ret = HCCL_SUCCESS;
    HcomInfo hcom_info;
    std::string ranktable_file(file_name);
    std::string rankTableM;
    std::string realFilePath;

    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcomLoadRanktableFile(ranktable_file.c_str(), rankTableM, realFilePath);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = CfgGetClusterInfo(rankTableM, identify, hcom_info.params, hcom_info.rankTable);
    EXPECT_EQ(ret, HCCL_E_PARA);

    ret = hrtResetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    remove(file_name);
}

TEST_F(HcomTest, ut_hcom_get_hcom_info_deviceNum_check)
{
    nlohmann::json rank_table
        = {{"status", "completed"},
           {"deploy_mode", "lab"},
           {"group_count", "1"},
           {"chip_info", "910"},
           {"board_id", "0x0000"},
           {"para_plane_nic_location", "device"},
           {"para_plane_nic_num", "2"},
           {"para_plane_nic_name", {"eth0", "eth1"}},
           {"group_list",
            {{{"group_name", ""},
              {"device_num", "4"},
              {"server_num", "2"},
              {"instance_count", "4"},
              {"instance_list",
               {{{"rank_id", "0"},
                 {"server_id", "10.0.0.10"},
                 {"devices", {{{"device_id", "0"}, {"device_ip", "192.168.0.12"}}}}},
                {{"rank_id", "1"},
                 {"server_id", "10.0.0.10"},
                 {"devices", {{{"device_id", "1"}, {"device_ip", "192.168.1.12"}}}}},
                {{"rank_id", "2"},
                 {"server_id", "10.0.0.10"},
                 {"devices", {{{"device_id", "2"}, {"device_ip", "192.168.0.14"}}}}},
                {{"rank_id", "3"},
                 {"server_id", "10.0.0.11"},
                 {"devices", {{{"device_id", "0"}, {"device_ip", "192.168.1.14"}}}}}}},
              {"server_list",
               {
                   {{"server_id", "192.168.10.2"},
                    {"para_plane_info",
                     {{
                          {"eth1", "192.168.210.2"},
                      },
                      {
                          {"eth0", "192.168.200.2"},
                      }}}

                   },
                   {{"server_id", "192.168.10.3"},
                    {"para_plane_info",
                     {{
                          {"eth0", "192.168.200.3"},
                      },
                      {
                          {"eth1", "192.168.210.3"},
                      }}}

                   },

               }}}}}};

    char file_name[] = "./ut_hcom_get_hcom_info_err2.json";
    std::ofstream outfile(file_name, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(4) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name);
    } else {
        HCCL_ERROR("open %s failed", file_name);
    }

    outfile.close();

    std::string identify = "0";
    s32 ret = HCCL_SUCCESS;
    HcomInfo hcom_info;
    std::string ranktable_file(file_name);
    std::string rankTableM;
    std::string realFilePath;

    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcomLoadRanktableFile(ranktable_file.c_str(), rankTableM, realFilePath);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = CfgGetClusterInfo(rankTableM, identify, hcom_info.params, hcom_info.rankTable);
    EXPECT_NE(ret, HCCL_SUCCESS);

    ret = hrtResetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    remove(file_name);
}

TEST_F(HcomTest, ut_hcom_get_hcom_info_deviceid_check)
{
    nlohmann::json rank_table = rank_table_910_2server_8rank;

    char file_name[] = "./ut_hcom_get_hcom_info_err2.json";
    std::ofstream outfile(file_name, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(4) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name);
    } else {
        HCCL_ERROR("open %s failed", file_name);
    }

    outfile.close();

    std::string identify = "0";
    s32 ret = HCCL_SUCCESS;
    HcomInfo hcom_info;
    std::string ranktable_file(file_name);
    std::string rankTableM;
    std::string realFilePath;

    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcomLoadRanktableFile(ranktable_file.c_str(), rankTableM, realFilePath);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = CfgGetClusterInfo(rankTableM, identify, hcom_info.params, hcom_info.rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = hrtResetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    remove(file_name);
}

TEST_F(HcomTest, ut_hcom_get_severId_checkErr)
{
    nlohmann::json rank_table
        = {{"status", "completed"},
           {"deploy_mode", "lab"},
           {"group_count", "1"},
           {"chip_info", "910"},
           {"board_id", "0x0000"},
           {"para_plane_nic_location", "device"},
           {"para_plane_nic_num", "2"},
           {"para_plane_nic_name", {"eth0", "eth1"}},
           {"group_list",
            {{{"group_name", ""},
              {"device_num", "4"},
              {"server_num", "2"},
              {"instance_count", "4"},
              {"instance_list",
               {{{"rank_id", "0"},
                 {"server_id_Err", "10.0.0.10"},
                 {"devices", {{{"device_id", "1"}, {"device_ip", "192.168.0.12"}}}}},
                {{"rank_id", "1"},
                 {"server_id_Err", "10.0.0.10"},
                 {"devices", {{{"device_id", "0"}, {"device_ip", "192.168.1.12"}}}}},
                {{"rank_id", "2"},
                 {"server_id_Err", "10.0.0.11"},
                 {"devices", {{{"device_id", "0"}, {"device_ip", "192.168.0.14"}}}}},
                {{"rank_id", "3"},
                 {"server_id_Err", "10.0.0.11"},
                 {"devices", {{{"device_id", "1"}, {"device_ip", "192.168.1.14"}}}}}}},
              {"server_list",
               {
                   {{"server_id", "192.168.10.2"},
                    {"para_plane_info",
                     {{
                          {"eth1", "192.168.210.2"},
                      },
                      {
                          {"eth0", "192.168.200.2"},
                      }}}

                   },
                   {{"server_id", "192.168.10.3"},
                    {"para_plane_info",
                     {{
                          {"eth0", "192.168.200.3"},
                      },
                      {
                          {"eth1", "192.168.210.3"},
                      }}}

                   },

               }}}}}};

    char file_name[] = "./ut_hcom_get_severId_checkErr.json";
    std::ofstream outfile(file_name, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(4) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name);
    } else {
        HCCL_ERROR("open %s failed", file_name);
    }

    outfile.close();

    std::string identify = "0";
    s32 ret = HCCL_SUCCESS;
    HcomInfo hcom_info;
    std::string ranktable_file(file_name);
    std::string rankTableM;
    std::string realFilePath;

    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcomLoadRanktableFile(ranktable_file.c_str(), rankTableM, realFilePath);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = CfgGetClusterInfo(rankTableM, identify, hcom_info.params, hcom_info.rankTable);
    EXPECT_NE(ret, HCCL_SUCCESS);

    ret = hrtResetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    remove(file_name);
}

TEST_F(HcomTest, ut_hcom_get_json_property_checkErr)
{
    nlohmann::json rank_table
        = {{"status", "completed"},
           {"deploy_mode", "lab"},
           {"group_count", "1"},
           {"chip_info", "910"},
           {"board_id", "0x0000"},
           {"para_plane_nic_location", "device"},
           {"para_plane_nic_num", "2"},
           {"para_plane_nic_name", {"eth0", "eth1"}},
           {"group_list",
            {{{"group_name", ""},
              {"device_num", "4"},
              {"server_num", "2"},
              {"instance_count", "4"},
              {"instance_list_Err",
               {{{"rank_id", "0"},
                 {"server_id", "10.0.0.10"},
                 {"devices", {{{"device_id", "1"}, {"device_ip", "192.168.0.12"}}}}},
                {{"rank_id", "1"},
                 {"server_id", "10.0.0.10"},
                 {"devices", {{{"device_id", "0"}, {"device_ip", "192.168.1.12"}}}}},
                {{"rank_id", "2"},
                 {"server_id", "10.0.0.11"},
                 {"devices", {{{"device_id", "0"}, {"device_ip", "192.168.0.14"}}}}},
                {{"rank_id", "3"},
                 {"server_id", "10.0.0.11"},
                 {"devices", {{{"device_id", "1"}, {"device_ip", "192.168.1.14"}}}}}}},
              {"server_list",
               {
                   {{"server_id", "192.168.10.2"},
                    {"para_plane_info",
                     {{
                          {"eth1", "192.168.210.2"},
                      },
                      {
                          {"eth0", "192.168.200.2"},
                      }}}

                   },
                   {{"server_id", "192.168.10.3"},
                    {"para_plane_info",
                     {{
                          {"eth0", "192.168.200.3"},
                      },
                      {
                          {"eth1", "192.168.210.3"},
                      }}}

                   },

               }}}}}};

    char file_name[] = "./ut_hcom_get_json_property_checkErr.json";
    std::ofstream outfile(file_name, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(4) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name);
    } else {
        HCCL_ERROR("open %s failed", file_name);
    }

    outfile.close();

    std::string identify = "0";
    s32 ret = HCCL_SUCCESS;
    HcomInfo hcom_info;
    std::string ranktable_file(file_name);
    std::string rankTableM;
    std::string realFilePath;

    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcomLoadRanktableFile(ranktable_file.c_str(), rankTableM, realFilePath);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = CfgGetClusterInfo(rankTableM, identify, hcom_info.params, hcom_info.rankTable);
    EXPECT_NE(ret, HCCL_SUCCESS);

    ret = hrtResetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    remove(file_name);
}

TEST_F(HcomTest, ut_hcom_get_json_chip_info_checkErr)
{
    nlohmann::json rank_table
        = {{"status", "completed"},
           {"deploy_mode", "lab"},
           {"group_count", "1"},
           {"chip_info", "210"},
           {"board_id", "0x0000"},
           {"para_plane_nic_location", "device"},
           {"para_plane_nic_num", "2"},
           {"para_plane_nic_name", {"eth0", "eth1"}},
           {"group_list",
            {{{"group_name", ""},
              {"device_num", "4"},
              {"server_num", "2"},
              {"instance_count", "4"},
              {"instance_list",
               {{{"rank_id", "0"},
                 {"server_id", "10.0.0.10"},
                 {"devices", {{{"device_id", "1"}, {"device_ip", "192.168.0.12"}}}}},
                {{"rank_id", "1"},
                 {"server_id", "10.0.0.10"},
                 {"devices", {{{"device_id", "0"}, {"device_ip", "192.168.1.12"}}}}},
                {{"rank_id", "2"},
                 {"server_id", "10.0.0.11"},
                 {"devices", {{{"device_id", "0"}, {"device_ip", "192.168.0.14"}}}}},
                {{"rank_id", "3"},
                 {"server_id", "10.0.0.11"},
                 {"devices", {{{"device_id", "1"}, {"device_ip", "192.168.1.14"}}}}}}},
              {"server_list",
               {
                   {{"server_id", "192.168.10.2"},
                    {"para_plane_info",
                     {{
                          {"eth1", "192.168.210.2"},
                      },
                      {
                          {"eth0", "192.168.200.2"},
                      }}}

                   },
                   {{"server_id", "192.168.10.3"},
                    {"para_plane_info",
                     {{
                          {"eth0", "192.168.200.3"},
                      },
                      {
                          {"eth1", "192.168.210.3"},
                      }}}

                   },

               }}}}}};

    char file_name[] = "./ut_hcom_get_json_property_checkErr.json";
    std::ofstream outfile(file_name, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(4) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name);
    } else {
        HCCL_ERROR("open %s failed", file_name);
    }

    outfile.close();

    std::string identify = "0";
    s32 ret = HCCL_SUCCESS;
    HcomInfo hcom_info;
    std::string ranktable_file(file_name);
    std::string rankTableM;
    std::string realFilePath;

    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcomLoadRanktableFile(ranktable_file.c_str(), rankTableM, realFilePath);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = CfgGetClusterInfo(rankTableM, identify, hcom_info.params, hcom_info.rankTable);
    EXPECT_EQ(ret, HCCL_E_UNAVAIL);

    ret = hrtResetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    remove(file_name);
}

#if 1

TEST_F(HcomTest, ut_hcom_allreduce)
{
    rtModel_t model = (void*)1;

    HcclCommunicator impl;
    MOCKER_CPP_VIRTUAL(
        impl, &HcclCommunicator::Init,
        HcclResult(HcclCommunicator::*)(HcclCommParams & params, const RankTable_t& rankTable))
        .expects(atMost(1))
        .will(returnValue(0));
    char* rank_table_file = "./ut_hcom.json";
    char* rank_ID = "0";
    HcclResult ret = hrtSetDevice(0);
    ret = HcomInitByFile(rank_table_file, rank_ID);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();

    rtStream_t stream;
    rtError_t rt_ret = aclrtCreateStream(&stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    u32 rank_size = 0;
    u32* rank_size_t = &rank_size;
    u32 rank_id = 0;
    u32* rank_id_t = &rank_id;
    ret = HcomGetRankSize(HCCL_WORLD_GROUP, rank_size_t);
    // printf("rank_size is %d \n",rank_size);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = HcomGetRankId(HCCL_WORLD_GROUP, rank_id_t);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    s8* sendbuf = (s8*)sal_malloc(10 * sizeof(s8));
    sal_memset(sendbuf, 10 * sizeof(s8), 0, 10 * sizeof(s8));
    s8* recv = (s8*)sal_malloc(10 * sizeof(s8));
    sal_memset(recv, 10 * sizeof(s8), 0, 10 * sizeof(s8));

    MOCKER_CPP_VIRTUAL(impl, &HcclCommunicator::AllReduce).expects(atMost(1)).will(returnValue(0));

    MOCKER_CPP(&hcclComm::GetRankTableCrc).stubs().will(returnValue(0));

    MOCKER_CPP(&hcclComm::GetAlgType).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&hcclComm::GetRankSize).expects(atMost(1)).will(returnValue(0));

    MOCKER_CPP(&hcclComm::GetGroupRank).expects(atMost(1)).will(returnValue(0));

    ret = HcomAllReduce("tag", sendbuf, recv, 10, HCCL_DATA_TYPE_INT8, HCCL_REDUCE_SUM, NULL, stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();

    aclrtSynchronizeStream(stream);
    rt_ret = aclrtDestroyStream(stream);
    ret = HcomDestroy();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    sal_free(sendbuf);
    sal_free(recv);
    // remove(file_name);
}

#endif

static void DoReduceScatterVTest(HcclDataType dataType, HcclResult expectedRet)
{
    nlohmann::json rank_table = rank_table_910_2server_8rank;
    char file_name[] = "./st_hcom.json";
    WriteRankTableFile(file_name, rank_table);

    HcclCommunicator impl;
    MOCKER_CPP_VIRTUAL(
        impl, &HcclCommunicator::Init,
        HcclResult(HcclCommunicator::*)(HcclCommParams & params, const RankTable_t& rankTable))
        .expects(atMost(1))
        .will(returnValue(0));
    char* rank_table_file = "./st_hcom.json";
    char* rank_ID = "0";
    hrtSetDevice(0);
    HcclResult ret = HcomInitByFile(rank_table_file, rank_ID);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();

    s8* sendbuf = (s8*)sal_malloc(16 * 10 * sizeof(s8));
    sal_memset(sendbuf, 16 * 10 * sizeof(s8), 0, 16 * 10 * sizeof(s8));
    s8* recvbuf = (s8*)sal_malloc(10 * sizeof(s8));
    sal_memset(recvbuf, 10 * sizeof(s8), 0, 10 * sizeof(s8));

    rtStream_t stream;

    rtError_t rt_ret = aclrtCreateStream(&stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    MOCKER_CPP_VIRTUAL(impl, &HcclCommunicator::ReduceScatterV).expects(atMost(1)).will(returnValue(0));

    MOCKER_CPP(&hcclComm::GetRankTableCrc).stubs().will(returnValue(0));

    MOCKER_CPP(&hcclComm::GetAlgType).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&hcclComm::GetRankSize).expects(atMost(1)).will(returnValue(0));

    MOCKER_CPP(&hcclComm::GetGroupRank).expects(atMost(1)).will(returnValue(0));

    // 构造入参
    int32_t rankSize = 2;
    vector<u64> sendCounts(rankSize, 10);
    vector<u64> sdispls(rankSize, 0);
    for (int i = 0; i < rankSize; i++) {
        sdispls[i] = 10 * i;
    }

    ret = HcomReduceScatterV(
        "tag", sendbuf, sendCounts.data(), sdispls.data(), recvbuf, 10, dataType, HCCL_REDUCE_SUM, HCCL_WORLD_GROUP,
        stream);
    EXPECT_EQ(ret, expectedRet);
    GlobalMockObject::verify();

    aclrtSynchronizeStream(stream);
    rt_ret = aclrtDestroyStream(stream);

    ret = HcomDestroy();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    sal_free(sendbuf);
    sal_free(recvbuf);

    remove(file_name);
}

TEST_F(HcomTest, ut_hcom_reducescatterv) { DoReduceScatterVTest(HCCL_DATA_TYPE_INT8, HCCL_SUCCESS); }

TEST_F(HcomTest, ut_hcom_reducescatterv_check_int64) { DoReduceScatterVTest(HCCL_DATA_TYPE_INT64, HCCL_E_NOT_SUPPORT); }

TEST_F(HcomTest, ut_hcom_send_receive_same_server)
{
    rtModel_t model = (void*)1;

    HcclCommunicator impl;
    MOCKER_CPP_VIRTUAL(
        impl, &HcclCommunicator::Init,
        HcclResult(HcclCommunicator::*)(HcclCommParams & params, const RankTable_t& rankTable))
        .expects(atMost(1))
        .will(returnValue(0));
    char* rank_table_file = "./ut_hcom.json";
    char* rank_ID = "0";
    HcclResult ret = hrtSetDevice(0);
    ret = HcomInitByFile(rank_table_file, rank_ID);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();

    rtStream_t stream;
    rtError_t rt_ret = aclrtCreateStream(&stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    u32 rank_size = 0;
    u32* rank_size_t = &rank_size;
    u32 rank_id = 0;
    u32* rank_id_t = &rank_id;
    ret = HcomGetRankSize(HCCL_WORLD_GROUP, rank_size_t);
    // printf("rank_size is %d \n",rank_size);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = HcomGetRankId(HCCL_WORLD_GROUP, rank_id_t);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    s8* sendbuf = (s8*)sal_malloc(10 * sizeof(s8));
    sal_memset(sendbuf, 10 * sizeof(s8), 0, 10 * sizeof(s8));
    s8* recv = (s8*)sal_malloc(10 * sizeof(s8));
    sal_memset(recv, 10 * sizeof(s8), 0, 10 * sizeof(s8));

    MOCKER_CPP(&hcclComm::GetRankTableCrc).stubs().will(returnValue(0));

    MOCKER_CPP(&hcclComm::GetAlgType).stubs().will(returnValue(HCCL_SUCCESS));

    aclrtSetDevice(0);
    MOCKER_CPP_VIRTUAL(impl, &HcclCommunicator::Send).expects(atMost(1)).will(returnValue(0));
    ret = HcomSend("tag", sendbuf, 10, HCCL_DATA_TYPE_INT8, 1, 0, NULL, stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();

    MOCKER_CPP(&hcclComm::GetAlgType).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP_VIRTUAL(impl, &HcclCommunicator::Receive).expects(atMost(1)).will(returnValue(0));
    aclrtSetDevice(0);
    ret = HcomReceive("tag", recv, 10, HCCL_DATA_TYPE_INT8, 1, 0, NULL, stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();

    aclrtSynchronizeStream(stream);

    rt_ret = aclrtDestroyStream(stream);

    ret = HcomDestroy();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    sal_free(sendbuf);
    sal_free(recv);
    // remove(file_name);
}

TEST_F(HcomTest, ut_hcom_send_receive)
{
    rtModel_t model = (void*)1;
    rtModel_t model2 = (void*)2;
    HcclCommunicator impl;
    MOCKER_CPP_VIRTUAL(
        impl, &HcclCommunicator::Init,
        HcclResult(HcclCommunicator::*)(HcclCommParams & params, const RankTable_t& rankTable))
        .expects(atMost(1))
        .will(returnValue(0));
    char* rank_table_file = "./ut_hcom.json";
    char* rank_ID = "0";
    HcclResult ret = hrtSetDevice(0);
    ret = HcomInitByFile(rank_table_file, rank_ID);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();

    rtStream_t stream;
    rtError_t rt_ret = aclrtCreateStream(&stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    rtStream_t stream2;
    rt_ret = aclrtCreateStream(&stream2, 5);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    u32 rank_size = 0;
    u32* rank_size_t = &rank_size;
    u32 rank_id = 0;
    u32* rank_id_t = &rank_id;
    ret = HcomGetRankSize(HCCL_WORLD_GROUP, rank_size_t);
    // printf("rank_size is %d \n",rank_size);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = HcomGetRankId(HCCL_WORLD_GROUP, rank_id_t);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    s8* sendbuf = (s8*)sal_malloc(10 * sizeof(s8));
    sal_memset(sendbuf, 10 * sizeof(s8), 0, 10 * sizeof(s8));
    s8* recv = (s8*)sal_malloc(10 * sizeof(s8));
    sal_memset(recv, 10 * sizeof(s8), 0, 10 * sizeof(s8));

    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    MOCKER_CPP(&hcclComm::GetRankTableCrc).stubs().will(returnValue(0));

    MOCKER_CPP(&hcclComm::GetAlgType).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&hcclComm::send).stubs().will(returnValue(0));
    ret = HcomSend("tag", sendbuf, 10, HCCL_DATA_TYPE_INT8, 8, 0, NULL, stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();

    MOCKER_CPP(&hcclComm::GetAlgType).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&hcclComm::receive).stubs().will(returnValue(0));
    ret = HcomReceive("tag", recv, 10, HCCL_DATA_TYPE_INT8, 8, 0, NULL, stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();

    aclrtSynchronizeStream(stream);

    rt_ret = aclrtDestroyStream(stream);
    rt_ret = aclrtDestroyStream(stream2);

    ret = HcomDestroy();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    sal_free(sendbuf);
    sal_free(recv);

    // remove(file_name);
}
TEST_F(HcomTest, ut_610_hcom_send_receive)
{
    rtModel_t model = (void*)1;
    rtModel_t model2 = (void*)2;
    HcclCommunicator impl;
    MOCKER_CPP_VIRTUAL(
        impl, &HcclCommunicator::Init,
        HcclResult(HcclCommunicator::*)(HcclCommParams & params, const RankTable_t& rankTable))
        .expects(atMost(1))
        .will(returnValue(0));
    char* rank_table_file = "./ut_hcom_v610.json";
    char* rank_ID = "0";
    HcclResult ret = hrtSetDevice(0);
    set_board_id(0x2000);
    ret = HcomInitByFile(rank_table_file, rank_ID);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();

    rtStream_t stream;
    rtError_t rt_ret = aclrtCreateStream(&stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    rtStream_t stream2;
    rt_ret = aclrtCreateStream(&stream2, 5);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    u32 rank_size = 0;
    u32* rank_size_t = &rank_size;
    u32 rank_id = 0;
    u32* rank_id_t = &rank_id;
    ret = HcomGetRankSize(HCCL_WORLD_GROUP, rank_size_t);
    // printf("rank_size is %d \n",rank_size);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = HcomGetRankId(HCCL_WORLD_GROUP, rank_id_t);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    s8* sendbuf = (s8*)sal_malloc(10 * sizeof(s8));
    sal_memset(sendbuf, 10 * sizeof(s8), 0, 10 * sizeof(s8));
    s8* recv = (s8*)sal_malloc(10 * sizeof(s8));
    sal_memset(recv, 10 * sizeof(s8), 0, 10 * sizeof(s8));

    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    MOCKER_CPP(&hcclComm::GetRankTableCrc).stubs().will(returnValue(0));

    MOCKER_CPP(&hcclComm::GetAlgType).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&hcclComm::send).stubs().will(returnValue(0));
    ret = HcomSend("tag", sendbuf, 10, HCCL_DATA_TYPE_INT8, 1, 0, NULL, stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();

    MOCKER_CPP(&hcclComm::GetAlgType).stubs().will(returnValue(HCCL_SUCCESS));

    ret = hrtSetDevice(0);
    MOCKER_CPP(&hcclComm::receive).stubs().will(returnValue(0));
    ret = HcomReceive("tag", recv, 10, HCCL_DATA_TYPE_INT8, 1, 0, NULL, stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();

    aclrtSynchronizeStream(stream);

    rt_ret = aclrtDestroyStream(stream);
    rt_ret = aclrtDestroyStream(stream2);

    ret = HcomDestroy();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    set_board_id(0);
    sal_free(sendbuf);
    sal_free(recv);

    // remove(file_name);
}

TEST_F(HcomTest, ut_hcom_rankid_valid_check)
{
    rtModel_t model = (void*)1;

    hrtSetDevice(18);

    MOCKER(HcomCheckrtMemcpyAddrAsync).stubs().will(returnValue(false));

    char* rank_table_file = "./ut_hcom.json";
    char* rank_ID = "0";
    HcclResult ret = HcomInitByFile(rank_table_file, rank_ID);
    EXPECT_EQ(ret, HCCL_E_UNAVAIL);

    ret = HcomDestroy();
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(HcomTest, ut_hcom_allgather)
{
    HcclCommunicator impl;
    MOCKER_CPP_VIRTUAL(
        impl, &HcclCommunicator::Init,
        HcclResult(HcclCommunicator::*)(HcclCommParams & params, const RankTable_t& rankTable))
        .expects(atMost(1))
        .will(returnValue(0));
    char* rank_table_file = "./ut_hcom.json";
    char* rank_ID = "0";
    hrtSetDevice(0);
    HcclResult ret = HcomInitByFile(rank_table_file, rank_ID);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();

    s8* sendbuf = (s8*)sal_malloc(10 * sizeof(s8));
    sal_memset(sendbuf, 10 * sizeof(s8), 0, 10 * sizeof(s8));
    s8* recv = (s8*)sal_malloc(16 * 10 * sizeof(s8));
    sal_memset(recv, 16 * 10 * sizeof(s8), 0, 16 * 10 * sizeof(s8));

    rtStream_t stream;

    rtError_t rt_ret = aclrtCreateStream(&stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    MOCKER_CPP_VIRTUAL(impl, &HcclCommunicator::AllGather).expects(atMost(1)).will(returnValue(0));

    MOCKER_CPP(&hcclComm::GetRankTableCrc).stubs().will(returnValue(0));

    MOCKER_CPP(&hcclComm::GetAlgType).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&hcclComm::GetRankSize).expects(atMost(1)).will(returnValue(0));

    MOCKER_CPP(&hcclComm::GetGroupRank).expects(atMost(1)).will(returnValue(0));

    ret = HcomAllGather("tag", sendbuf, recv, 10, HCCL_DATA_TYPE_INT8, NULL, stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();

    aclrtSynchronizeStream(stream);
    rt_ret = aclrtDestroyStream(stream);

    ret = HcomDestroy();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    sal_free(sendbuf);
    sal_free(recv);
    // remove(file_name);
}

TEST_F(HcomTest, ut_hcom_allgatherv)
{
    HcclCommunicator impl;
    MOCKER_CPP_VIRTUAL(
        impl, &HcclCommunicator::Init,
        HcclResult(HcclCommunicator::*)(HcclCommParams & params, const RankTable_t& rankTable))
        .expects(atMost(1))
        .will(returnValue(0));
    char* rank_table_file = "./ut_hcom.json";
    char* rank_ID = "0";
    hrtSetDevice(0);
    HcclResult ret = HcomInitByFile(rank_table_file, rank_ID);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();

    s8* sendbuf = (s8*)sal_malloc(10 * sizeof(s8));
    sal_memset(sendbuf, 10 * sizeof(s8), 0, 10 * sizeof(s8));
    s8* recv = (s8*)sal_malloc(16 * 10 * sizeof(s8));
    sal_memset(recv, 16 * 10 * sizeof(s8), 0, 16 * 10 * sizeof(s8));

    rtStream_t stream;

    rtError_t rt_ret = aclrtCreateStream(&stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    MOCKER_CPP_VIRTUAL(impl, &HcclCommunicator::AllGatherV).expects(atMost(1)).will(returnValue(0));

    MOCKER_CPP(&hcclComm::GetRankTableCrc).stubs().will(returnValue(0));

    MOCKER_CPP(&hcclComm::GetAlgType).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&hcclComm::GetRankSize).expects(atMost(1)).will(returnValue(0));

    MOCKER_CPP(&hcclComm::GetGroupRank).expects(atMost(1)).will(returnValue(0));

    // 构造入参
    int32_t rankSize = 2;
    vector<u64> recvCounts(rankSize, 10);
    vector<u64> rdispls(rankSize, 0);
    for (int i = 0; i < rankSize; i++) {
        rdispls[i] = 10 * i;
    }

    ret = HcomAllGatherV(
        "tag", sendbuf, 10, recv, recvCounts.data(), rdispls.data(), HCCL_DATA_TYPE_INT8, NULL, stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();

    aclrtSynchronizeStream(stream);
    rt_ret = aclrtDestroyStream(stream);

    ret = HcomDestroy();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    sal_free(sendbuf);
    sal_free(recv);
    // remove(file_name);
}

TEST_F(HcomTest, ut_hcom_reduce)
{
    HcclCommunicator impl;
    MOCKER_CPP_VIRTUAL(
        impl, &HcclCommunicator::Init,
        HcclResult(HcclCommunicator::*)(HcclCommParams & params, const RankTable_t& rankTable))
        .expects(atMost(1))
        .will(returnValue(0));
    char* rank_table_file = "./ut_hcom.json";
    char* rank_ID = "0";
    HcclResult ret = hrtSetDevice(0);
    ret = HcomInitByFile(rank_table_file, rank_ID);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();

    s8* sendbuf = (s8*)sal_malloc(10 * sizeof(s8));
    sal_memset(sendbuf, 10 * sizeof(s8), 0, 10 * sizeof(s8));
    s8* recv = (s8*)sal_malloc(10 * sizeof(s8));
    sal_memset(recv, 10 * sizeof(s8), 0, 10 * sizeof(s8));

    rtStream_t stream;

    rtError_t rt_ret = aclrtCreateStream(&stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    MOCKER_CPP(&hcclComm::Reduce).expects(atMost(1)).will(returnValue(0));

    MOCKER_CPP(&hcclComm::GetRankTableCrc).stubs().will(returnValue(0));

    MOCKER_CPP(&hcclComm::GetAlgType).stubs().will(returnValue(HCCL_SUCCESS));

    ret = HcomReduce("tag", sendbuf, recv, 10, HCCL_DATA_TYPE_INT8, HCCL_REDUCE_SUM, 0, NULL, stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();

    aclrtSynchronizeStream(stream);

    rt_ret = aclrtDestroyStream(stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    ret = HcomDestroy();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    sal_free(sendbuf);
    sal_free(recv);
    // remove(file_name);
}

TEST_F(HcomTest, ut_hcom_reducescatter)
{
    HcclCommunicator impl;
    MOCKER_CPP_VIRTUAL(
        impl, &HcclCommunicator::Init,
        HcclResult(HcclCommunicator::*)(HcclCommParams & params, const RankTable_t& rankTable))
        .expects(atMost(1))
        .will(returnValue(0));
    char* rank_table_file = "./ut_hcom.json";
    char* rank_ID = "0";
    HcclResult ret = hrtSetDevice(0);
    ret = HcomInitByFile(rank_table_file, rank_ID);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();

    s8* sendbuf = (s8*)sal_malloc(16 * 10 * sizeof(s8));
    sal_memset(sendbuf, 16 * 10 * sizeof(s8), 0, 16 * 10 * sizeof(s8));
    s8* recv = (s8*)sal_malloc(10 * sizeof(s8));
    sal_memset(recv, 10 * sizeof(s8), 0, 10 * sizeof(s8));

    rtStream_t stream;

    rtError_t rt_ret = aclrtCreateStream(&stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    MOCKER_CPP(&hcclComm::ReduceScatter).expects(atMost(1)).will(returnValue(0));

    MOCKER_CPP(&hcclComm::GetRankTableCrc).stubs().will(returnValue(0));

    MOCKER_CPP(&hcclComm::GetAlgType).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&hcclComm::GetRankSize).expects(atMost(1)).will(returnValue(0));

    MOCKER_CPP(&hcclComm::GetGroupRank).expects(atMost(1)).will(returnValue(0));

    ret = HcomReduceScatter("tag", sendbuf, recv, 10, HCCL_DATA_TYPE_INT8, HCCL_REDUCE_SUM, NULL, stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();

    aclrtSynchronizeStream(stream);

    rt_ret = aclrtDestroyStream(stream);

    ret = HcomDestroy();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    sal_free(sendbuf);
    sal_free(recv);
    // remove(file_name);
}

TEST_F(HcomTest, ut_hcom_get_hcom_info_boardType2)
{
    nlohmann::json rank_table = rank_table_910_2server_8rank;

    char file_name[] = "./ut_hcom_get_hcom_info_boardType2.json";
    std::ofstream outfile(file_name, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(4) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name);
    } else {
        HCCL_ERROR("open %s failed", file_name);
    }

    outfile.close();

    s32 ret = HCCL_SUCCESS;

    HcomInfo hcom_info;
    std::string ranktable_file(file_name);
    std::string rankTableM;
    std::string realFilePath;

    char* rank_ID = "0";
    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcomLoadRanktableFile(ranktable_file.c_str(), rankTableM, realFilePath);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = CfgGetClusterInfo(rankTableM, rank_ID, hcom_info.params, hcom_info.rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = hrtResetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    remove(file_name);
}

TEST_F(HcomTest, ut_hcom_get_rank_info)
{
    nlohmann::json rank_table = rank_table_910_1server_1rank;

    char file_name_t[] = "./ut_hcom_get_rank_info.json";
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
    char* rank_table_file = "./ut_hcom_get_rank_info.json";
    char* rank_ID = "0";

    ret = HcomInitByFile(rank_table_file, rank_ID);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcomGetLocalRankSize(NULL, &localranksize);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcomGetLocalRankId(NULL, &localrankid);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcomDestroy();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    remove(file_name_t);

    EXPECT_EQ(localranksize, 1);
    EXPECT_EQ(localrankid, 0);
}

TEST_F(HcomTest, ut_HcclCommGraphAllGather)
{
    hrtSetDevice(0);
    s8* sendbuf = (s8*)sal_malloc(10 * sizeof(s8));
    sal_memset(sendbuf, 10 * sizeof(s8), 0, 10 * sizeof(s8));
    s8* recv = (s8*)sal_malloc(16 * 10 * sizeof(s8));
    sal_memset(recv, 16 * 10 * sizeof(s8), 0, 16 * 10 * sizeof(s8));

    rtStream_t stream;

    rtError_t rt_ret = aclrtCreateStream(&stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    hccl::hcclComm* comm = new hccl::hcclComm(1, 1, "123");
    s64 opBaseHcom = (s64)comm;

    MOCKER_CPP(&hcclComm::GetRankTableCrc).stubs().will(returnValue(0));

    MOCKER_CPP(&hcclComm::GetRankSize).expects(atMost(1)).will(returnValue(0));

    MOCKER_CPP(&hcclComm::GetGroupRank).expects(atMost(1)).will(returnValue(0));

    MOCKER_CPP(&hcclComm::AllGather).expects(atMost(1)).will(returnValue(0));

    MOCKER_CPP(&hcclComm::GetAlgType).stubs().will(returnValue(HCCL_SUCCESS));

    int ret = HCCL_SUCCESS;
    ret = HcclCommGraphAllGather("tag", sendbuf, recv, 10, HCCL_DATA_TYPE_INT8, 0, stream);
    EXPECT_EQ(ret, HCCL_E_PTR);

    ret = HcclCommGraphAllGather("tag", sendbuf, recv, 10, HCCL_DATA_TYPE_INT8, opBaseHcom, stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();

    aclrtSynchronizeStream(stream);
    rt_ret = aclrtDestroyStream(stream);

    sal_free(sendbuf);
    sal_free(recv);
    delete comm;
}

TEST_F(HcomTest, ut_HcclCommGraphAllReduce)
{
    HcclResult ret = hrtSetDevice(0);

    rtStream_t stream;
    rtError_t rt_ret = aclrtCreateStream(&stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    s8* sendbuf = (s8*)sal_malloc(10 * sizeof(s8));
    sal_memset(sendbuf, 10 * sizeof(s8), 0, 10 * sizeof(s8));
    s8* recv = (s8*)sal_malloc(10 * sizeof(s8));
    sal_memset(recv, 10 * sizeof(s8), 0, 10 * sizeof(s8));

    hccl::hcclComm* comm = new hccl::hcclComm(1, 1, "123");
    s64 opBaseHcom = (s64)comm;

    MOCKER_CPP(&hcclComm::GetRankTableCrc).stubs().will(returnValue(0));

    MOCKER_CPP(&hcclComm::GetRankSize).expects(atMost(1)).will(returnValue(0));

    MOCKER_CPP(&hcclComm::GetGroupRank).expects(atMost(1)).will(returnValue(0));

    MOCKER_CPP(&hcclComm::AllReduce).expects(atMost(1)).will(returnValue(0));

    MOCKER_CPP(&hcclComm::GetAlgType).stubs().will(returnValue(HCCL_SUCCESS));

    ret = HcclCommGraphAllReduce("tag", sendbuf, recv, 10, HCCL_DATA_TYPE_INT8, HCCL_REDUCE_SUM, 0, stream);
    EXPECT_EQ(ret, HCCL_E_PTR);

    ret = HcclCommGraphAllReduce("tag", sendbuf, recv, 10, HCCL_DATA_TYPE_INT8, HCCL_REDUCE_SUM, opBaseHcom, stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();

    aclrtSynchronizeStream(stream);
    rt_ret = aclrtDestroyStream(stream);

    sal_free(sendbuf);
    sal_free(recv);
    delete comm;
}

TEST_F(HcomTest, ut_HcclCommGraphReduce)
{
    HcclResult ret = hrtSetDevice(0);

    s8* sendbuf = (s8*)sal_malloc(10 * sizeof(s8));
    sal_memset(sendbuf, 10 * sizeof(s8), 0, 10 * sizeof(s8));
    s8* recv = (s8*)sal_malloc(10 * sizeof(s8));
    sal_memset(recv, 10 * sizeof(s8), 0, 10 * sizeof(s8));

    rtStream_t stream;

    rtError_t rt_ret = aclrtCreateStream(&stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    hccl::hcclComm* comm = new hccl::hcclComm(1, 1, "123");
    s64 opBaseHcom = (s64)comm;

    MOCKER_CPP(&hcclComm::Reduce).expects(atMost(1)).will(returnValue(0));

    MOCKER_CPP(&hcclComm::GetRankTableCrc).stubs().will(returnValue(0));

    MOCKER_CPP(&hcclComm::GetRankSize).expects(atMost(1)).will(returnValue(0));

    MOCKER_CPP(&hcclComm::GetGroupRank).expects(atMost(1)).will(returnValue(0));

    MOCKER(HcomCheckUserRank).expects(atMost(1)).will(returnValue(0));

    MOCKER_CPP(&hcclComm::GetAlgType).stubs().will(returnValue(HCCL_SUCCESS));

    ret = HcclCommGraphReduce("tag", sendbuf, recv, 10, HCCL_DATA_TYPE_INT8, HCCL_REDUCE_SUM, 0, 0, stream);
    EXPECT_EQ(ret, HCCL_E_PTR);

    ret = HcclCommGraphReduce("tag", sendbuf, recv, 10, HCCL_DATA_TYPE_INT8, HCCL_REDUCE_SUM, 0, opBaseHcom, stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();

    aclrtSynchronizeStream(stream);

    rt_ret = aclrtDestroyStream(stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    sal_free(sendbuf);
    sal_free(recv);
    delete comm;
}

TEST_F(HcomTest, ut_HcclCommGraphBroadcast)
{
    HcclResult ret = hrtSetDevice(0);

    s8* sendbuf = (s8*)sal_malloc(10 * sizeof(s8));
    sal_memset(sendbuf, 10 * sizeof(s8), 0, 10 * sizeof(s8));

    rtStream_t stream;

    rtError_t rt_ret = aclrtCreateStream(&stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    hccl::hcclComm* comm = new hccl::hcclComm(1, 1, "123");
    s64 opBaseHcom = (s64)comm;

    MOCKER_CPP(&hcclComm::Broadcast).expects(atMost(1)).will(returnValue(0));

    MOCKER_CPP(&hcclComm::GetRankTableCrc).stubs().will(returnValue(0));

    MOCKER_CPP(&hcclComm::GetRankSize).expects(atMost(1)).will(returnValue(0));

    MOCKER_CPP(&hcclComm::GetGroupRank).expects(atMost(1)).will(returnValue(0));

    MOCKER(HcomCheckUserRank).expects(atMost(1)).will(returnValue(0));

    MOCKER_CPP(&hcclComm::GetAlgType).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&hcclComm::GetNumBlocks).stubs().will(returnValue(HCCL_SUCCESS));

    ret = HcclCommGraphBroadcast("tag", sendbuf, 10, HCCL_DATA_TYPE_INT8, 0, 0, stream);
    EXPECT_EQ(ret, HCCL_E_PTR);

    ret = HcclCommGraphBroadcast("tag", sendbuf, 10, HCCL_DATA_TYPE_INT8, 0, opBaseHcom, stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();

    aclrtSynchronizeStream(stream);

    rt_ret = aclrtDestroyStream(stream);

    sal_free(sendbuf);
    delete comm;
}

TEST_F(HcomTest, ut_HcclCommGraphReduceScatter)
{
    HcclResult ret = hrtSetDevice(0);

    s8* sendbuf = (s8*)sal_malloc(16 * 10 * sizeof(s8));
    sal_memset(sendbuf, 16 * 10 * sizeof(s8), 0, 16 * 10 * sizeof(s8));
    s8* recv = (s8*)sal_malloc(10 * sizeof(s8));
    sal_memset(recv, 10 * sizeof(s8), 0, 10 * sizeof(s8));

    rtStream_t stream;

    rtError_t rt_ret = aclrtCreateStream(&stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    hccl::hcclComm* comm = new hccl::hcclComm(1, 1, "123");
    s64 opBaseHcom = (s64)comm;

    MOCKER_CPP(&hcclComm::ReduceScatter).expects(atMost(1)).will(returnValue(0));

    MOCKER(HcomCheckUserRank).expects(atMost(1)).will(returnValue(0));

    MOCKER_CPP(&hcclComm::GetRankTableCrc).stubs().will(returnValue(0));

    MOCKER_CPP(&hcclComm::GetAlgType).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&hcclComm::GetRankSize).expects(atMost(1)).will(returnValue(0));

    MOCKER_CPP(&hcclComm::GetGroupRank).expects(atMost(1)).will(returnValue(0));

    ret = HcclCommGraphReduceScatter("tag", sendbuf, recv, 10, HCCL_DATA_TYPE_INT8, HCCL_REDUCE_SUM, 0, stream);
    EXPECT_EQ(ret, HCCL_E_PTR);

    ret = HcclCommGraphReduceScatter(
        "tag", sendbuf, recv, 10, HCCL_DATA_TYPE_INT8, HCCL_REDUCE_SUM, opBaseHcom, stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();

    aclrtSynchronizeStream(stream);

    rt_ret = aclrtDestroyStream(stream);

    sal_free(sendbuf);
    sal_free(recv);
    delete comm;
}

TEST_F(HcomTest, ut_HcclCommGraphSendRecv)
{
    HcclResult ret = hrtSetDevice(0);

    rtStream_t stream;
    rtError_t rt_ret = aclrtCreateStream(&stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    rtStream_t stream2;
    rt_ret = aclrtCreateStream(&stream2);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    s8* sendbuf = (s8*)sal_malloc(10 * sizeof(s8));
    sal_memset(sendbuf, 10 * sizeof(s8), 0, 10 * sizeof(s8));
    s8* recv = (s8*)sal_malloc(10 * sizeof(s8));
    sal_memset(recv, 10 * sizeof(s8), 0, 10 * sizeof(s8));

    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    hccl::hcclComm* comm = new hccl::hcclComm(1, 1, "123");
    s64 opBaseHcom = (s64)comm;

    MOCKER_CPP(&hcclComm::GetRankTableCrc).stubs().will(returnValue(0));

    MOCKER_CPP(&hcclComm::GetAlgType).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&hcclComm::send).stubs().will(returnValue(0));

    MOCKER_CPP(&hcclComm::GetRankSize).expects(atMost(1)).will(returnValue(0));

    MOCKER_CPP(&hcclComm::GetUserRank).expects(atMost(1)).will(returnValue(0));

    MOCKER(HcomCheckUserRank).expects(atMost(1)).will(returnValue(0));

    MOCKER(HcclCommGraphGetRankId).expects(atMost(1)).will(returnValue(0));

    ret = HcclCommGraphSend("tag", sendbuf, 10, HCCL_DATA_TYPE_INT8, 8, 0, 0, stream);
    EXPECT_EQ(ret, HCCL_E_PTR);

    ret = HcclCommGraphSend("tag", sendbuf, 10, HCCL_DATA_TYPE_INT8, 8, 0, opBaseHcom, stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();

    MOCKER_CPP(&hcclComm::receive).stubs().will(returnValue(0));

    MOCKER_CPP(&hcclComm::GetRankTableCrc).stubs().will(returnValue(0));

    MOCKER_CPP(&hcclComm::GetRankSize).expects(atMost(1)).will(returnValue(0));

    MOCKER_CPP(&hcclComm::GetUserRank).expects(atMost(1)).will(returnValue(0));

    MOCKER(HcomCheckUserRank).expects(atMost(1)).will(returnValue(0));

    MOCKER(HcclCommGraphGetRankId).expects(atMost(1)).will(returnValue(0));

    MOCKER_CPP(&hcclComm::GetAlgType).stubs().will(returnValue(HCCL_SUCCESS));

    ret = HcclCommGraphReceive("tag", recv, 10, HCCL_DATA_TYPE_INT8, 8, 0, 0, stream);
    EXPECT_EQ(ret, HCCL_E_PTR);

    ret = HcclCommGraphReceive("tag", recv, 10, HCCL_DATA_TYPE_INT8, 8, 0, opBaseHcom, stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();

    aclrtSynchronizeStream(stream);

    rt_ret = aclrtDestroyStream(stream);
    rt_ret = aclrtDestroyStream(stream2);

    sal_free(sendbuf);
    sal_free(recv);
    delete comm;
}

TEST_F(HcomTest, ut_HcclCommGraphGetRankId)
{
    HcclResult ret = hrtSetDevice(0);

    rtStream_t stream;
    rtError_t rt_ret = aclrtCreateStream(&stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    hccl::hcclComm* comm = new hccl::hcclComm(1, 1, "123");
    s64 opBaseHcom = (s64)comm;

    ret = HcclCommGraphGetRankId(opBaseHcom, nullptr);
    EXPECT_EQ(ret, HCCL_E_PTR);

    aclrtSynchronizeStream(stream);
    rt_ret = aclrtDestroyStream(stream);
    delete comm;
}

TEST_F(HcomTest, ut_HcclCommGraphAlltoAllV)
{
    s32 deviceId = 0;
    char* identify = "0";
    s32 rankSize = 1;
    s32 rank = atoi(identify);
    u64 count = 2;
    HCCL_INFO("alltoall_int32 : identify[%d], count[%llu]", rank, count);
    u32 devLogicId = 0xFFFF;
    HcclResult ret = hrtGetDeviceIndexByPhyId(deviceId, devLogicId);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = hrtSetDevice(devLogicId);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    const int COUNT_PER_RANK = count;
    u64 memSize = COUNT_PER_RANK * rankSize * sizeof(s32);
    HostMem hostSendMem = HostMem::alloc(memSize);
    memset_s(hostSendMem.ptr(), memSize, 0, COUNT_PER_RANK * rankSize);
    for (u32 i = 0; i < COUNT_PER_RANK * rankSize; i++) {
        *((s32*)hostSendMem.ptr() + i) = rank + 1;
    }

    // 构造入参
    vector<u64> sendCounts(rankSize, COUNT_PER_RANK);
    vector<u64> recvCounts(rankSize, COUNT_PER_RANK);
    vector<u64> sdispls(rankSize, 0);
    vector<u64> rdispls(rankSize, 0);
    for (int i = 0; i < rankSize; i++) {
        sdispls[i] = COUNT_PER_RANK * i;
        rdispls[i] = COUNT_PER_RANK * i;
        HCCL_INFO("num[%d] displs[%d]", i, COUNT_PER_RANK * i);
    }

    DeviceMem sendMem = DeviceMem::alloc(memSize);
    ret = hrtMemSyncCopy(
        sendMem.ptr(), memSize, hostSendMem.ptr(), memSize, HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    DeviceMem recvMem = DeviceMem::alloc(memSize);

    hccl::Stream stream(StreamType::STREAM_TYPE_OFFLINE);

    hccl::hcclComm* comm = new hccl::hcclComm(1, 1, "123");
    s64 opBaseHcom = (s64)comm;

    MOCKER_CPP(&hcclComm::AlltoAllV).expects(atMost(1)).will(returnValue(0));

    MOCKER_CPP(&hcclComm::GetRankTableCrc).stubs().will(returnValue(0));

    MOCKER_CPP(&hcclComm::GetRankSize).expects(atMost(1)).will(returnValue(0));

    MOCKER_CPP(&hcclComm::GetUserRank).expects(atMost(1)).will(returnValue(0));

    MOCKER_CPP(&hcclComm::GetAlgType).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&hcclComm::GetNumBlocks).stubs().will(returnValue(HCCL_SUCCESS));

    ret = HcclCommGraphAlltoAllV(
        sendMem.ptr(), sendCounts.data(), sdispls.data(), HCCL_DATA_TYPE_INT32, recvMem.ptr(), recvCounts.data(),
        rdispls.data(), HCCL_DATA_TYPE_INT32, opBaseHcom, stream.ptr(), "hcom_alltoallv");
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = hcclStreamSynchronize(stream.ptr());

    EXPECT_EQ(ret, HCCL_SUCCESS);
    (void)aclrtResetDevice(0);
    delete comm;
}

#define HCCL_COM_DATA_SIZE 1024
#if 1
TEST_F(HcomTest, ut_hcom_allreduce_cloud)
{
    nlohmann::json rank_table = {
        {"status", "completed"},
        {"chip_info", "910"},
        {"group_count", "1"},
        {"group_list",
         {{
             {"group_name", ""},
             {"instance_count", "1"},
             {"device_count", "1"},
             {"instance_list",
              {
                  {{"pod_name", "tf-bae43"},
                   {"server_id", "10.0.0.10"},
                   {"devices", {{{"device_id", "0"}, {"device_ip", "192.168.0.12"}}}}},
              }},
         }}},
    };

    char file_name_t[] = "./ut_hcom_allreduce_cloud.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(1) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    } else {
        HCCL_ERROR("open %s failed", file_name_t);
    }

    outfile.close();

    int ret = HCCL_SUCCESS;
    rtError_t rt_ret = RT_ERROR_NONE;
    rtStream_t stream;
    s8* sendbuf;
    s8* recvbuf;
    s32 rank = 0;
    s32 errors = 0;
    s32 count = HCCL_COM_DATA_SIZE;
    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    char* rank_table_file = "./ut_hcom_allreduce_cloud.json";
    char* pod_name = "tf-bae43";

    ret = HcomInitByFile(rank_table_file, pod_name);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    rt_ret = aclrtCreateStream(&stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    sendbuf = (s8*)sal_malloc(count);
    sal_memset(sendbuf, count, 0, count);
    recvbuf = (s8*)sal_malloc(count);
    sal_memset(recvbuf, count, 0, count);

    for (int j = 0; j < count; j++) {
        sendbuf[j] = 2;
    }

    ret = HcomAllReduce("testreduce", sendbuf, recvbuf, count, HCCL_DATA_TYPE_INT8, HCCL_REDUCE_SUM, NULL, stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    rt_ret = aclrtSynchronizeStream(stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    for (int j = 0; j < count; j++) {
        if (recvbuf[j] != 2) {
            errors++;
        }
    }

    rt_ret = aclrtDestroyStream(stream);

    ret = HcomDestroy();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    sal_free(sendbuf);
    sal_free(recvbuf);
    remove(file_name_t);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    EXPECT_EQ(errors, 0);
}
#endif

#if 1
TEST_F(HcomTest, ut_hcom_creatgroup)
{
    nlohmann::json rank_table_group
        = {{"status", "completed"},
           {"deploy_mode", "lab"},
           {"group_count", "1"},
           {"chip_info", "910"},
           {"board_id", "0x3011"},
           {"para_plane_nic_location", "device"},
           {"para_plane_nic_num", "8"},
           {"para_plane_nic_name", {"eth0", "eth1", "eth2", "eth3", "eth4", "eth5", "eth6", "eth7"}},
           {"group_list",
            {{{"group_name", ""},
              {"device_num", "16"},
              {"server_num", "2"},
              {"instance_count", "16"},
              {"instance_list",
               {
                   {{"rank_id", "0"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "0"}, {"device_ip", "192.168.0.31"}}}}},

                   {{"rank_id", "1"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "1"}, {"device_ip", "192.168.0.32"}}}}},
                   {{"rank_id", "2"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "2"}, {"device_ip", "192.168.0.33"}}}}},

                   {{"rank_id", "3"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "3"}, {"device_ip", "192.168.0.34"}}}}},
                   {{"rank_id", "4"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "4"}, {"device_ip", "192.168.0.35"}}}}},

                   {{"rank_id", "5"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "5"}, {"device_ip", "192.168.0.36"}}}}},
                   {{"rank_id", "6"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "6"}, {"device_ip", "192.168.0.37"}}}}},

                   {{"rank_id", "7"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "7"}, {"device_ip", "192.168.0.38"}}}}},
                   {{"rank_id", "8"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "0"}, {"device_ip", "192.168.0.40"}}}}},

                   {{"rank_id", "9"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "1"}, {"device_ip", "192.168.0.41"}}}}},
                   {{"rank_id", "10"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "2"}, {"device_ip", "192.168.0.42"}}}}},

                   {{"rank_id", "11"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "3"}, {"device_ip", "192.168.0.43"}}}}},
                   {{"rank_id", "12"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "4"}, {"device_ip", "192.168.0.44"}}}}},

                   {{"rank_id", "13"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "5"}, {"device_ip", "192.168.0.45"}}}}},
                   {{"rank_id", "14"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "6"}, {"device_ip", "192.168.0.46"}}}}},

                   {{"rank_id", "15"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "7"}, {"device_ip", "192.168.0.47"}}}}},
               }},
              {"server_list",
               {
                   {{"server_id", "10.0.0.10"},
                    {"para_plane_info",
                     {
                         {
                             {"eth0", "192.168.200.2"},
                         },
                         {
                             {"eth1", "192.168.201.2"},
                         },
                         {
                             {"eth2", "192.168.202.2"},
                         },
                         {
                             {"eth3", "192.168.203.2"},
                         },
                         {
                             {"eth4", "192.168.204.2"},
                         },
                         {
                             {"eth5", "192.168.205.2"},
                         },
                         {
                             {"eth6", "192.168.206.2"},
                         },
                         {
                             {"eth7", "192.168.207.2"},
                         },
                     }}

                   },
                   {{"server_id", "10.0.0.11"},
                    {"para_plane_info",
                     {
                         {
                             {"eth0", "192.168.210.3"},
                         },
                         {
                             {"eth1", "192.168.211.3"},
                         },
                         {
                             {"eth2", "192.168.212.3"},
                         },
                         {
                             {"eth3", "192.168.213.3"},
                         },
                         {
                             {"eth4", "192.168.214.3"},
                         },
                         {
                             {"eth5", "192.168.215.3"},
                         },
                         {
                             {"eth6", "192.168.216.3"},
                         },
                         {
                             {"eth7", "192.168.217.3"},
                         },
                     }}

                   },

               }}}}}};

    char file_name_t[] = "./st_hcom_creatgroup.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(1) << rank_table_group << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    } else {
        HCCL_ERROR("open %s failed", file_name_t);
    }

    outfile.close();

    const u32 groupRanksNum = 1;
    char* strGroup1 = "group1";
    char* strGroup2 = "group2";
    char* strGroupErr = "group_err";
    u32 rankNum = 1;
    u32 worldRank;
    u32 groupRank;
    // std::vector<u32> groupRanks;
    u32 groupRanks[1] = {0};
    int ret = HCCL_SUCCESS;
    rtError_t rt_ret = RT_ERROR_NONE;
    rtStream_t stream;
    rtStream_t stream1;
    s8* sendbuf;
    s8* recvbuf;
    s32 rank = 0;
    s32 errors = 0;
    s32 count = HCCL_COM_DATA_SIZE;
    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    char* rank_table_file = "./st_hcom_creatgroup.json";
    char* pod_name = "0";
    set_board_id(0x3011);

    HcclRootInfo rootInfo;
    ret = hccl::hcclComm::GetUniqueId(&rootInfo);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = HcomInitByFile(rank_table_file, pod_name);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    rt_ret = aclrtCreateStream(&stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    // groupRanks.push_back(0);
    // groupRanks.push_back(4);

    HCCL_INFO("this is hcom_group");
    ret = HcomCreateGroup(strGroup1, groupRanksNum, (u32*)groupRanks);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcomCreateGroup(strGroup2, groupRanksNum, (u32*)groupRanks);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = HcomGetRankSize(strGroup1, &rankNum);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = HcomGetRankId(strGroup1, &worldRank);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    groupRank = 0;
    ret = HcomGetWorldRankFromGroupRank(strGroup1, groupRank, &worldRank);
    HCCL_INFO("groupRank:%d worldRank:%d", groupRank, worldRank);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    worldRank = 0;
    ret = HcomGetGroupRankFromWorldRank(worldRank, strGroup1, &groupRank);
    HCCL_INFO("worldRank:%d groupRank:%d", worldRank, groupRank);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = HcomGetRankSize(HCCL_WORLD_GROUP, &rankNum);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = HcomGetRankId(HCCL_WORLD_GROUP, &worldRank);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    groupRank = 0;
    ret = HcomGetWorldRankFromGroupRank(HCCL_WORLD_GROUP, groupRank, &worldRank);
    HCCL_INFO("groupRank:%d worldRank:%d", groupRank, worldRank);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    worldRank = 0;
    ret = HcomGetGroupRankFromWorldRank(worldRank, HCCL_WORLD_GROUP, &groupRank);
    HCCL_INFO("worldRank:%d groupRank:%d", worldRank, groupRank);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcomCreateGroup(strGroup1, groupRanksNum, (u32*)groupRanks);
    EXPECT_EQ(ret, HCCL_E_PARA);

    ret = HcomDestroyGroup(strGroupErr);
    EXPECT_NE(ret, HCCL_SUCCESS);

    ret = HcomGetRankSize(strGroupErr, &rankNum);
    EXPECT_EQ(ret, HCCL_E_NOT_FOUND);

    sendbuf = (s8*)sal_malloc(count);
    sal_memset(sendbuf, count, 0, count);
    recvbuf = (s8*)sal_malloc(count);
    sal_memset(recvbuf, count, 0, count);

    for (int j = 0; j < count; j++) {
        sendbuf[j] = 2;
    }

    HCCL_INFO("HcomReduce start");
    ret = HcomReduce("testreduce", sendbuf, recvbuf, count, HCCL_DATA_TYPE_INT8, HCCL_REDUCE_SUM, 0, strGroup1, stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    HCCL_INFO("HcomReduce end");

    rt_ret = aclrtSynchronizeStream(stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    HCCL_INFO("hcom_reduce5");

    for (int j = 0; j < count; j++) {
        if (recvbuf[j] != 2) {
            errors++;
        }
    }

    ret = HcomReduce(
        "testreduce1", sendbuf, recvbuf, count, HCCL_DATA_TYPE_INT8, HCCL_REDUCE_SUM, 0, strGroupErr, stream);
    EXPECT_EQ(ret, HCCL_E_NOT_FOUND);
    HCCL_INFO("hcom_reduce0");

    aclrtSynchronizeStream(stream);

    HCCL_INFO("hcom_reduce1");
    ret = HcomDestroyGroup(strGroup1);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    HCCL_INFO("this is hcom_group end");

    rt_ret = aclrtDestroyStream(stream);

    ret = HcomDestroy();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    set_board_id(0);
    remove(file_name_t);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    EXPECT_EQ(errors, 0);

    sal_free(sendbuf);
    sal_free(recvbuf);
}
#endif

TEST_F(HcomTest, ut_hcom_backlog_group)
{
    const u32 groupRanksNum = 4;
    char* strGroup = "group1";
    u32 groupRanks[4] = {0, 1, 2, 3};
    int ret = HCCL_SUCCESS;
    ret = HcomCreateGroup(strGroup, groupRanksNum, (u32*)groupRanks);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = HcomDestroyGroup(strGroup);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcomCreateGroup(strGroup, groupRanksNum, (u32*)groupRanks);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // GROUP 已存在
    ret = HcomCreateGroup(strGroup, groupRanksNum, (u32*)groupRanks);
    EXPECT_EQ(ret, HCCL_E_PARA);

    nlohmann::json rank_table = rank_table_1server_8rank;
    char file_name_t[] = "./ut_hcom_test_rank_table_1server_8rank.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(1) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    } else {
        HCCL_ERROR("open %s failed", file_name_t);
    }

    outfile.close();
    s32 deviceId = 0;
    char* identify = "0";
    s32 rankSize = 1;
    s32 rank = atoi(identify);
    u32 devLogicId = 0;
    ret = hrtSetDevice(devLogicId);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    char* rankTableFile = "./ut_hcom_test_rank_table_1server_8rank.json";
    ret = HcomInitByFile(rankTableFile, identify);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // GROUP 已存在
    ret = HcomCreateGroup(strGroup, groupRanksNum, (u32*)groupRanks);
    EXPECT_EQ(ret, HCCL_E_PARA);

    ret = HcomDestroyGroup(strGroup);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcomDestroy();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    remove(file_name_t);
}

TEST_F(HcomTest, ut_hcom_gradient_segment)
{
    struct model_feature feature;
    u32 segment_num;
    std::vector<u32> segment_index;

    HcclCommunicator impl;
    MOCKER_CPP_VIRTUAL(
        impl, &HcclCommunicator::Init,
        HcclResult(HcclCommunicator::*)(HcclCommParams & params, const RankTable_t& rankTable))
        .expects(atMost(1))
        .will(returnValue(0));
    char* rank_table_file = "./ut_hcom.json";
    char* rank_ID = "0";
    HcclResult ret = hrtSetDevice(0);
    ret = HcomInitByFile(rank_table_file, rank_ID);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();

    char group[] = "1";
    char model_name[] = "";
    feature.gradient_num = 2;
    feature.gradient_size = (float*)sal_malloc(2 * sizeof(float));
    sal_memset(feature.gradient_size, 2 * sizeof(float), 0, 2 * sizeof(float));
    feature.gradient_time = (float*)sal_malloc(2 * sizeof(float));
    sal_memset(feature.gradient_time, 2 * sizeof(float), 0, 2 * sizeof(float));
    feature.model_name = model_name;
    MOCKER(GetGradientSegment).expects(atMost(1)).will(returnValue(0));

    bool isConfig = true;
    u32 len = segment_index.size();
    ret = HcomGetSplitStrategy(group, &feature, segment_index.data(), &len, isConfig);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    sal_free(feature.gradient_size);
    sal_free(feature.gradient_time);

    GlobalMockObject::verify();

    ret = HcomDestroy();
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(HcomTest, ut_hcom_gradient_segment_global_set_and_get_notInOrder)
{
    HcclResult ret;
    u32 segList[3] = {1, 4, 3};
    char setGroup[] = "1";
    ret = HcomSetGradFusionByIndex(setGroup, 3, segList);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(HcomTest, ut_hcom_gradient_segment_global_set_and_get_lenZero)
{
    HcclResult ret;
    u32 segList[3] = {1, 2, 3};
    char setGroup[] = "1";
    ret = HcomSetGradFusionByIndex(setGroup, 0, segList);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(HcomTest, ut_hcom_gradient_segment_global_size_set_and_get_sumNotHundred)
{
    HcclResult ret;
    float segList[3] = {20, 20, 20};
    char setGroup[] = "1";
    ret = HcomSetGradFusionBySize(setGroup, 3, segList);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(HcomTest, ut_hcom_gradient_segment_global_size_set_and_get_lenZero)
{
    HcclResult ret;
    float segList[3] = {40, 40, 20};
    char setGroup[] = "1";
    ret = HcomSetGradFusionBySize(setGroup, 0, segList);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(HcomTest, ut_hcom_gradient_segment_global_set_and_get)
{
    struct model_feature feature;
    std::vector<u32> segment_index;
    HcclResult ret;
    u32 segList[2] = {1, 4};

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

    char file_name_t[] = "./ut_hcom_gradient_segment_global_set_and_get.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(1) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    } else {
        HCCL_ERROR("open %s failed", file_name_t);
    }

    outfile.close();

    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    char* rank_table_file = "./ut_hcom_gradient_segment_global_set_and_get.json";
    char* rank_ID = "0";

    char setGroup[] = "1";
    char getGroup[] = "1";
    char model_name[] = "resnet50";
    feature.gradient_num = 5;
    feature.gradient_size = (float*)sal_malloc(5 * sizeof(float));
    sal_memset(feature.gradient_size, 5 * sizeof(float), 0, 5 * sizeof(float));
    feature.gradient_time = (float*)sal_malloc(5 * sizeof(float));
    sal_memset(feature.gradient_time, 5 * sizeof(float), 0, 5 * sizeof(float));
    feature.model_name = model_name;

    ret = HcomSetGradFusionByIndex(setGroup, 2, segList);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcomInitByFile(rank_table_file, rank_ID);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    bool isConfig = true;
    u32 len = segment_index.size();
    ret = HcomGetSplitStrategy(group, &feature, segment_index.data(), &len, isConfig);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(segment_index[0], 1);
    EXPECT_EQ(segment_index[1], 4);

    sal_free(feature.gradient_size);
    sal_free(feature.gradient_time);

    ret = HcomDestroy();
    remove(file_name_t);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(HcomTest, ut_hcom_gradient_segment_global_size_set_and_get)
{
    struct model_feature feature;
    std::vector<u32> segment_index;
    HcclResult ret;
    float segList[3] = {20, 40, 40};

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

    char file_name_t[] = "./ut_hcom_gradient_segment_global_size_set_and_get.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(1) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    } else {
        HCCL_ERROR("open %s failed", file_name_t);
    }

    outfile.close();

    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    char* rank_table_file = "./ut_hcom_gradient_segment_global_size_set_and_get.json";
    char* rank_ID = "0";

    char setGroup[] = "1";
    char getGroup[] = "1";
    char model_name[] = "resnet50";
    float gradient_array[30] = {4096,    8257536, 8704,    8704,    4194304, 2560, 2560,    9437184, 2560, 2560,
                                4194304, 8704,    8704,    4194304, 2560,    2560, 9437184, 2560,    2560, 4194304,
                                8704,    8704,    8388608, 4194304, 2560,    2560, 9437184, 2560,    2560, 2097152};
    feature.gradient_num = 30;
    feature.gradient_size = gradient_array;
    feature.gradient_time = (float*)sal_malloc(30 * sizeof(float));
    sal_memset(feature.gradient_time, 30 * sizeof(float), 0, 30 * sizeof(float));
    feature.model_name = model_name;
    g_segmentSizeMap.clear();
    ret = HcomSetGradFusionBySize(setGroup, 3, segList);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcomInitByFile(rank_table_file, rank_ID);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    bool isConfig = true;
    u32 len = segment_index.size();
    ret = HcomGetSplitStrategy(group, &feature, segment_index.data(), &len, isConfig);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(segment_index[0], 6);
    EXPECT_EQ(segment_index[1], 18);
    EXPECT_EQ(segment_index[2], 29);
    sal_free(feature.gradient_time);

    ret = HcomDestroy();
    remove(file_name_t);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(HcomTest, ut_hcom_gradient_segment_global_size_set_close_and_get)
{
    struct model_feature feature;
    std::vector<u32> segment_index;
    HcclResult ret;
    float segList[3] = {96, 2, 2};

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

    char file_name_t[] = "./ut_hcom_gradient_segment_global_size_set_close_and_get.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(1) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    } else {
        HCCL_ERROR("open %s failed", file_name_t);
    }

    outfile.close();

    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    char* rank_table_file = "./ut_hcom_gradient_segment_global_size_set_close_and_get.json";
    char* rank_ID = "0";

    char setGroup[] = "1";
    char getGroup[] = "1";
    char model_name[] = "resnet50";
    float gradient_array[30] = {4096,    8257536, 8704,    8704,    4194304, 2560, 2560,    9437184, 2560, 2560,
                                4194304, 8704,    8704,    4194304, 2560,    2560, 9437184, 2560,    2560, 4194304,
                                8704,    8704,    8388608, 4194304, 2560,    2560, 9437184, 2560,    2560, 2097152};
    feature.gradient_num = 30;
    feature.gradient_size = gradient_array;
    feature.gradient_time = (float*)sal_malloc(30 * sizeof(float));
    sal_memset(feature.gradient_time, 30 * sizeof(float), 0, 30 * sizeof(float));
    feature.model_name = model_name;
    g_segmentSizeMap.clear();
    ret = HcomSetGradFusionBySize(setGroup, 3, segList);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcomInitByFile(rank_table_file, rank_ID);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    bool isConfig = true;
    u32 len = segment_index.size();
    ret = HcomGetSplitStrategy(group, &feature, segment_index.data(), &len, isConfig);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(segment_index[0], 25);
    EXPECT_EQ(segment_index[1], 28);
    EXPECT_EQ(segment_index[2], 29);
    sal_free(feature.gradient_time);

    ret = HcomDestroy();
    remove(file_name_t);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(HcomTest, ut_hcom_gradient_segment_global_size_set_and_get_gradient_1)
{
    struct model_feature feature;
    std::vector<u32> segment_index;
    HcclResult ret;
    float segList[2] = {50, 50};

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

    char file_name_t[] = "./ut_hcom_gradient_segment_global_size_set_and_get_gradient_1.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(1) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    } else {
        HCCL_ERROR("open %s failed", file_name_t);
    }

    outfile.close();

    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    char* rank_table_file = "./ut_hcom_gradient_segment_global_size_set_and_get_gradient_1.json";
    char* rank_ID = "0";

    char setGroup[] = "1";
    char getGroup[] = "1";
    char model_name[] = "resnet50";
    float gradient_array[1] = {4096};
    feature.gradient_num = 1;
    feature.gradient_size = gradient_array;
    feature.gradient_time = (float*)sal_malloc(1 * sizeof(float));
    sal_memset(feature.gradient_time, 1 * sizeof(float), 0, 1 * sizeof(float));
    feature.model_name = model_name;
    g_segmentSizeMap.clear();
    ret = HcomSetGradFusionBySize(setGroup, 2, segList);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcomInitByFile(rank_table_file, rank_ID);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    bool isConfig = true;
    u32 len = segment_index.size();
    ret = HcomGetSplitStrategy(group, &feature, segment_index.data(), &len, isConfig);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    sal_free(feature.gradient_time);

    ret = HcomDestroy();
    remove(file_name_t);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

#if 1
TEST_F(HcomTest, ut_hcom_get_hcom_info_podnameEmpty)
{
    nlohmann::json rank_table = {
        {"status", "completed"},
        {"group_count", "2"},
        {"group_list",
         {{
              {"group_name", "group1"},
              {"instance_count", "1"},
              {"device_count", "8"},
              {"instance_list",
               {{{"pod_name", ""},
                 {"server_id", "10.0.0.10"},
                 {"devices",
                  {{{"device_id", "0"}, {"device_ip", "192.168.0.10"}},
                   {{"device_id", "1"}, {"device_ip", "192.168.0.11"}},
                   {{"device_id", "2"}, {"device_ip", "192.168.0.12"}},
                   {{"device_id", "3"}, {"device_ip", "192.168.0.13"}},
                   {{"device_id", "4"}, {"device_ip", "192.168.0.14"}},
                   {{"device_id", "5"}, {"device_ip", "192.168.0.15"}},
                   {{"device_id", "6"}, {"device_ip", "192.168.0.16"}},
                   {{"device_id", "7"}, {"device_ip", "192.168.0.17"}}}}}}},
          },
          {
              {"group_name", "1"},
              {"instance_count", "1"},
              {"device_count", "8"},
              {"instance_list",
               {
                   {{"pod_name", "tf-1"},
                    {"server_id", "10.0.0.11"},
                    {"devices",
                     {{{"device_id", "0"}, {"device_ip", "192.168.0.21"}},
                      {{"device_id", "1"}, {"device_ip", "192.168.0.22"}},
                      {{"device_id", "2"}, {"device_ip", "192.168.0.23"}},
                      {{"device_id", "3"}, {"device_ip", "192.168.0.24"}},
                      {{"device_id", "4"}, {"device_ip", "192.168.0.20"}},
                      {{"device_id", "5"}, {"device_ip", "192.168.0.25"}},
                      {{"device_id", "6"}, {"device_ip", "192.168.0.26"}},
                      {{"device_id", "7"}, {"device_ip", "192.168.0.27"}}}}},
               }},
          }}},
    };

    char file_name[] = "./ut_hcom_get_hcom_info_podnameEmpty.json";
    std::ofstream outfile(file_name, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(4) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name);
    } else {
        HCCL_ERROR("open %s failed", file_name);
    }

    outfile.close();

    std::string identify = "0";
    s32 ret = HCCL_SUCCESS;
    HcomInfo hcom_info;
    std::string ranktable_file(file_name);
    std::string rankTableM;
    std::string realFilePath;

    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcomLoadRanktableFile(ranktable_file.c_str(), rankTableM, realFilePath);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = CfgGetClusterInfo(rankTableM, identify, hcom_info.params, hcom_info.rankTable);
    EXPECT_EQ(ret, HCCL_E_PARA);

    ret = hrtResetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    remove(file_name);
}
#endif

#if 1
TEST_F(HcomTest, ut_hcom_get_hcom_info_instance_countIsZero)
{
    nlohmann::json rank_table = {
        {"status", "completed"},
        {"group_count", "2"},
        {"group_list",
         {{
              {"group_name", "group1"},
              {"instance_count", "0"},
              {"device_count", "8"},
              {"instance_list",
               {{{"pod_name", "tf-0"},
                 {"server_id", "10.0.0.10"},
                 {"devices",
                  {{{"device_id", "0"}, {"device_ip", "192.168.0.10"}},
                   {{"device_id", "1"}, {"device_ip", "192.168.0.11"}},
                   {{"device_id", "2"}, {"device_ip", "192.168.0.12"}},
                   {{"device_id", "3"}, {"device_ip", "192.168.0.13"}},
                   {{"device_id", "4"}, {"device_ip", "192.168.0.14"}},
                   {{"device_id", "5"}, {"device_ip", "192.168.0.15"}},
                   {{"device_id", "6"}, {"device_ip", "192.168.0.16"}},
                   {{"device_id", "7"}, {"device_ip", "192.168.0.17"}}}}}}},
          },
          {
              {"group_name", "1"},
              {"instance_count", "1"},
              {"device_count", "8"},
              {"instance_list",
               {
                   {{"pod_name", "tf-1"},
                    {"server_id", "10.0.0.11"},
                    {"devices",
                     {{{"device_id", "0"}, {"device_ip", "192.168.0.21"}},
                      {{"device_id", "1"}, {"device_ip", "192.168.0.22"}},
                      {{"device_id", "2"}, {"device_ip", "192.168.0.23"}},
                      {{"device_id", "3"}, {"device_ip", "192.168.0.24"}},
                      {{"device_id", "4"}, {"device_ip", "192.168.0.20"}},
                      {{"device_id", "5"}, {"device_ip", "192.168.0.25"}},
                      {{"device_id", "6"}, {"device_ip", "192.168.0.26"}},
                      {{"device_id", "7"}, {"device_ip", "192.168.0.27"}}}}},
               }},
          }}},
    };

    char file_name[] = "./ut_hcom_get_hcom_info_instance_countIsZero.json";
    std::ofstream outfile(file_name, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(4) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name);
    } else {
        HCCL_ERROR("open %s failed", file_name);
    }

    outfile.close();

    std::string identify = "0";
    s32 ret = HCCL_SUCCESS;
    HcomInfo hcom_info;
    std::string ranktable_file(file_name);
    std::string rankTableM;
    std::string realFilePath;

    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcomLoadRanktableFile(ranktable_file.c_str(), rankTableM, realFilePath);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = CfgGetClusterInfo(rankTableM, identify, hcom_info.params, hcom_info.rankTable);
    EXPECT_EQ(ret, HCCL_E_PARA);

    ret = hrtResetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    remove(file_name);
}
#endif

#if 1
TEST_F(HcomTest, ut_hcom_get_hcom_info_group_count_zero)
{
    nlohmann::json rank_table = {
        {"status", "completed"},
        {"group_count", "0"},
        {"group_list",
         {{
              {"group_name", "group1"},
              {"instance_count", "1"},
              {"device_count", "8"},
              {"instance_list",
               {{{"pod_name", "tf-0"},
                 {"server_id", "10.0.0.10"},
                 {"devices",
                  {{{"device_id", "0"}, {"device_ip", "192.168.0.10"}},
                   {{"device_id", "1"}, {"device_ip", "192.168.0.11"}},
                   {{"device_id", "2"}, {"device_ip", "192.168.0.12"}},
                   {{"device_id", "3"}, {"device_ip", "192.168.0.13"}},
                   {{"device_id", "4"}, {"device_ip", "192.168.0.14"}},
                   {{"device_id", "5"}, {"device_ip", "192.168.0.15"}},
                   {{"device_id", "6"}, {"device_ip", "192.168.0.16"}},
                   {{"device_id", "7"}, {"device_ip", "192.168.0.17"}}}}}}},
          },
          {
              {"group_name", "1"},
              {"instance_count", "1"},
              {"device_count", "8"},
              {"instance_list",
               {
                   {{"pod_name", "tf-1"},
                    {"server_id", "10.0.0.11"},
                    {"devices",
                     {{{"device_id", "0"}, {"device_ip", "192.168.0.21"}},
                      {{"device_id", "1"}, {"device_ip", "192.168.0.22"}},
                      {{"device_id", "2"}, {"device_ip", "192.168.0.23"}},
                      {{"device_id", "3"}, {"device_ip", "192.168.0.24"}},
                      {{"device_id", "4"}, {"device_ip", "192.168.0.20"}},
                      {{"device_id", "5"}, {"device_ip", "192.168.0.25"}},
                      {{"device_id", "6"}, {"device_ip", "192.168.0.26"}},
                      {{"device_id", "7"}, {"device_ip", "192.168.0.27"}}}}},
               }},
          }}},
    };

    char file_name[] = "./ut_hcom_get_hcom_info_group_count_zero.json";
    std::ofstream outfile(file_name, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(4) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name);
    } else {
        HCCL_ERROR("open %s failed", file_name);
    }

    outfile.close();

    std::string identify = "0";
    s32 ret = HCCL_SUCCESS;
    HcomInfo hcom_info;
    std::string ranktable_file(file_name);
    std::string rankTableM;
    std::string realFilePath;

    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcomLoadRanktableFile(ranktable_file.c_str(), rankTableM, realFilePath);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = CfgGetClusterInfo(rankTableM, identify, hcom_info.params, hcom_info.rankTable);
    EXPECT_EQ(ret, HCCL_E_PARA);

    ret = hrtResetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    remove(file_name);
}
#endif

#if 1
TEST_F(HcomTest, ut_hcom_get_hcom_info_deviceidErrBig)
{
    nlohmann::json rank_table = {
        {"status", "completed"},
        {"group_count", "2"},
        {"group_list",
         {{
              {"group_name", "group1"},
              {"device_count", "8"},
              {"instance_count", "1"},
              {"instance_list",
               {{{"pod_name", "tf-0"},
                 {"server_id", "10.0.0.10"},
                 {"devices",
                  {{{"device_id", "0"}, {"device_ip", "192.168.0.10"}},
                   {{"device_id", "1"}, {"device_ip", "192.168.0.11"}},
                   {{"device_id", "2"}, {"device_ip", "192.168.0.12"}},
                   {{"device_id", "3"}, {"device_ip", "192.168.0.13"}},
                   {{"device_id", "10"}, {"device_ip", "192.168.0.14"}},
                   {{"device_id", "5"}, {"device_ip", "192.168.0.15"}},
                   {{"device_id", "6"}, {"device_ip", "192.168.0.16"}},
                   {{"device_id", "7"}, {"device_ip", "192.168.0.17"}}}}}}},
          },
          {
              {"group_name", "1"},
              {"instance_count", "1"},
              {"device_count", "8"},
              {"instance_list",
               {
                   {{"pod_name", "tf-1"},
                    {"server_id", "10.0.0.11"},
                    {"devices",
                     {{{"device_id", "0"}, {"device_ip", "192.168.0.21"}},
                      {{"device_id", "1"}, {"device_ip", "192.168.0.22"}},
                      {{"device_id", "2"}, {"device_ip", "192.168.0.23"}},
                      {{"device_id", "3"}, {"device_ip", "192.168.0.24"}},
                      {{"device_id", "4"}, {"device_ip", "192.168.0.20"}},
                      {{"device_id", "5"}, {"device_ip", "192.168.0.25"}},
                      {{"device_id", "6"}, {"device_ip", "192.168.0.26"}},
                      {{"device_id", "7"}, {"device_ip", "192.168.0.27"}}}}},
               }},
          }}},
    };

    char file_name[] = "./ut_hcom_get_hcom_info_deviceidErrBig.json";
    std::ofstream outfile(file_name, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(4) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name);
    } else {
        HCCL_ERROR("open %s failed", file_name);
    }

    outfile.close();

    std::string identify = "0";
    s32 ret = HCCL_SUCCESS;
    HcomInfo hcom_info;
    std::string ranktable_file(file_name);
    std::string rankTableM;
    std::string realFilePath;

    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcomLoadRanktableFile(ranktable_file.c_str(), rankTableM, realFilePath);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = CfgGetClusterInfo(rankTableM, identify, hcom_info.params, hcom_info.rankTable);
    EXPECT_EQ(ret, HCCL_E_PARA);

    ret = hrtResetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    remove(file_name);
}
#endif

#if 1
TEST_F(HcomTest, ut_hcom_get_hcom_info_propety_err)
{
    nlohmann::json rank_table = {
        {"status", "completed"},
        {"group_count", "2"},
        {"group_list",
         {{
              {"group_name", "group1"},
              {"instance_count", "1"},
              {"device_count", "8"},
              {"instance_list",
               {{{"pod_name", "tf-0"},
                 {"server_id", "10.0.0.10"},
                 {"devices",
                  {{{"device_id", "0"}, {"device_ip", "192.168.0.10"}},
                   {{"device_id", "1"}, {"device_ip", "192.168.0.11"}},
                   {{"device_id", 2}, {"device_ip", "192.168.0.12"}},
                   {{"device_id", "3"}, {"device_ip", "192.168.0.13"}},
                   {{"device_id", "4"}, {"device_ip", "192.168.0.14"}},
                   {{"device_id", "5"}, {"device_ip", "192.168.0.15"}},
                   {{"device_id", "6"}, {"device_ip", "192.168.0.16"}},
                   {{"device_id", "7"}, {"device_ip", "192.168.0.17"}}}}}}},
          },
          {
              {"group_name", "1"},
              {"instance_count", "1"},
              {"device_count", "8"},
              {"instance_list",
               {
                   {{"pod_name", "tf-1"},
                    {"server_id", "10.0.0.11"},
                    {"devices",
                     {{{"device_id", "0"}, {"device_ip", "192.168.0.21"}},
                      {{"device_id", "1"}, {"device_ip", "192.168.0.22"}},
                      {{"device_id", "2"}, {"device_ip", "192.168.0.23"}},
                      {{"device_id", "3"}, {"device_ip", "192.168.0.24"}},
                      {{"device_id", "4"}, {"device_ip", "192.168.0.20"}},
                      {{"device_id", "5"}, {"device_ip", "192.168.0.25"}},
                      {{"device_id", "6"}, {"device_ip", "192.168.0.26"}},
                      {{"device_id", "7"}, {"device_ip", "192.168.0.27"}}}}},
               }},
          }}},
    };

    char file_name[] = "./ut_hcom_get_hcom_info_propety_err.json";
    std::ofstream outfile(file_name, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(4) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name);
    } else {
        HCCL_ERROR("open %s failed", file_name);
    }

    outfile.close();

    std::string identify = "0";
    s32 ret = HCCL_SUCCESS;
    HcomInfo hcom_info;
    std::string ranktable_file(file_name);
    std::string rankTableM;
    std::string realFilePath;

    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcomLoadRanktableFile(ranktable_file.c_str(), rankTableM, realFilePath);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = CfgGetClusterInfo(rankTableM, identify, hcom_info.params, hcom_info.rankTable);
    EXPECT_EQ(ret, HCCL_E_PARA);

    ret = hrtResetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    remove(file_name);
}
#endif

#if 1
TEST_F(HcomTest, ut_hcom_get_hcom_info_propety_err2)
{
    nlohmann::json rank_table = {
        {"status", "completed"},
        {"group_count", 2},
        {"group_list",
         {{
              {"group_name", "group1"},
              {"instance_count", "1"},
              {"device_count", "8"},
              {"instance_list",
               {{{"pod_name", "tf-0"},
                 {"server_id", "10.0.0.10"},
                 {"devices",
                  {{{"device_id", "0"}, {"device_ip", "192.168.0.10"}},
                   {{"device_id", "1"}, {"device_ip", "192.168.0.11"}},
                   {{"device_id", "2"}, {"device_ip", "192.168.0.12"}},
                   {{"device_id", "3"}, {"device_ip", "192.168.0.13"}},
                   {{"device_id", "4"}, {"device_ip", "192.168.0.14"}},
                   {{"device_id", "5"}, {"device_ip", "192.168.0.15"}},
                   {{"device_id", "6"}, {"device_ip", "192.168.0.16"}},
                   {{"device_id", "7"}, {"device_ip", "192.168.0.17"}}}}}}},
          },
          {
              {"group_name", "1"},
              {"instance_count", "1"},
              {"device_count", "8"},
              {"instance_list",
               {
                   {{"pod_name", "tf-1"},
                    {"server_id", "10.0.0.11"},
                    {"devices",
                     {{{"device_id", "0"}, {"device_ip", "192.168.0.21"}},
                      {{"device_id", "1"}, {"device_ip", "192.168.0.22"}},
                      {{"device_id", "2"}, {"device_ip", "192.168.0.23"}},
                      {{"device_id", "3"}, {"device_ip", "192.168.0.24"}},
                      {{"device_id", "4"}, {"device_ip", "192.168.0.20"}},
                      {{"device_id", "5"}, {"device_ip", "192.168.0.25"}},
                      {{"device_id", "6"}, {"device_ip", "192.168.0.26"}},
                      {{"device_id", "7"}, {"device_ip", "192.168.0.27"}}}}},
               }},
          }}},
    };

    char file_name[] = "./ut_hcom_get_hcom_info_propety_err.json";
    std::ofstream outfile(file_name, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(4) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name);
    } else {
        HCCL_ERROR("open %s failed", file_name);
    }

    outfile.close();

    std::string identify = "0";
    s32 ret = HCCL_SUCCESS;
    HcomInfo hcom_info;
    std::string ranktable_file(file_name);
    std::string rankTableM;
    std::string realFilePath;

    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcomLoadRanktableFile(ranktable_file.c_str(), rankTableM, realFilePath);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = CfgGetClusterInfo(rankTableM, identify, hcom_info.params, hcom_info.rankTable);
    EXPECT_EQ(ret, HCCL_E_PARA);

    ret = hrtResetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    remove(file_name);
}
#endif

#if 1

TEST_F(HcomTest, ut_hcom_get_hcom_info_ech_server_devNum_err)
{
    nlohmann::json rank_table
        = {{"status", "completed"},
           {"deploy_mode", "lab"},
           {"group_count", "1"},
           {"chip_info", "910"},
           {"board_id", "0x0000"},
           {"para_plane_nic_location", "device"},
           {"para_plane_nic_num", "8"},
           {"para_plane_nic_name", {"eth0", "eth1", "eth2", "eth3", "eth4", "eth5", "eth6", "eth7"}},
           {"group_list",
            {{{"group_name", ""},
              {"device_num", "16"},
              {"server_num", "2"},
              {"instance_count", "16"},
              {"instance_list",
               {
                   {{"rank_id", "0"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "0"}, {"device_ip", "192.168.0.11"}}}}},

                   {{"rank_id", "1"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "1"}, {"device_ip", "192.168.0.12"}}}}},
                   {{"rank_id", "2"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "2"}, {"device_ip", "192.168.0.13"}}}}},

                   {{"rank_id", "3"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "3"}, {"device_ip", "192.168.0.14"}}}}},
                   {{"rank_id", "4"},
                    {"server_id", "10.0.0.15"},
                    {"devices", {{{"device_id", "4"}, {"device_ip", "192.168.0.15"}}}}},

                   {{"rank_id", "5"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "5"}, {"device_ip", "192.168.0.16"}}}}},
                   {{"rank_id", "6"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "6"}, {"device_ip", "192.168.0.17"}}}}},

                   {{"rank_id", "7"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "7"}, {"device_ip", "192.168.0.18"}}}}},
                   {{"rank_id", "8"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "0"}, {"device_ip", "192.168.0.20"}}}}},

                   {{"rank_id", "9"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "1"}, {"device_ip", "192.168.0.21"}}}}},
                   {{"rank_id", "10"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "2"}, {"device_ip", "192.168.0.22"}}}}},

                   {{"rank_id", "11"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "3"}, {"device_ip", "192.168.0.23"}}}}},
                   {{"rank_id", "12"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "4"}, {"device_ip", "192.168.0.24"}}}}},

                   {{"rank_id", "13"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "5"}, {"device_ip", "192.168.0.25"}}}}},
                   {{"rank_id", "14"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "6"}, {"device_ip", "192.168.0.26"}}}}},

                   {{"rank_id", "15"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "7"}, {"device_ip", "192.168.0.27"}}}}},
               }},
              {"server_list",
               {
                   {{"server_id", "10.0.0.10"},
                    {"para_plane_info",
                     {
                         {
                             {"eth0", "192.168.200.2"},
                         },
                         {
                             {"eth1", "192.168.201.2"},
                         },
                         {
                             {"eth2", "192.168.202.2"},
                         },
                         {
                             {"eth3", "192.168.203.2"},
                         },
                         {
                             {"eth4", "192.168.204.2"},
                         },
                         {
                             {"eth5", "192.168.205.2"},
                         },
                         {
                             {"eth6", "192.168.206.2"},
                         },
                         {
                             {"eth7", "192.168.207.2"},
                         },
                     }}

                   },
                   {{"server_id", "10.0.0.11"},
                    {"para_plane_info",
                     {
                         {
                             {"eth0", "192.168.210.3"},
                         },
                         {
                             {"eth1", "192.168.211.3"},
                         },
                         {
                             {"eth2", "192.168.212.3"},
                         },
                         {
                             {"eth3", "192.168.213.3"},
                         },
                         {
                             {"eth4", "192.168.214.3"},
                         },
                         {
                             {"eth5", "192.168.215.3"},
                         },
                         {
                             {"eth6", "192.168.216.3"},
                         },
                         {
                             {"eth7", "192.168.217.3"},
                         },
                     }}

                   },

               }}}}}};

    char file_name[] = "./ut_hcom_get_hcom_info_ech_server_devNum_err.json";
    std::ofstream outfile(file_name, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(4) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name);
    } else {
        HCCL_ERROR("open %s failed", file_name);
    }

    outfile.close();

    std::string identify = "0";
    s32 ret = HCCL_SUCCESS;
    HcomInfo hcom_info;
    std::string ranktable_file(file_name);
    std::string rankTableM;
    std::string realFilePath;

    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcomLoadRanktableFile(ranktable_file.c_str(), rankTableM, realFilePath);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = CfgGetClusterInfo(rankTableM, identify, hcom_info.params, hcom_info.rankTable);
    EXPECT_EQ(ret, HCCL_E_PARA);
    ret = hrtResetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    remove(file_name);
}
#endif
TEST_F(HcomTest, ut_hcom_get_hcom_info_groupsizeErr)
{
    nlohmann::json rank_table
        = {{"status", "completed"},
           {"deploy_mode", "lab"},
           {"group_count", "1"},
           {"chip_info", "910"},
           {"board_id", "0x0000"},
           {"para_plane_nic_location", "device"},
           {"para_plane_nic_num", "8"},
           {"para_plane_nic_name", {"eth0", "eth1", "eth2", "eth3", "eth4", "eth5", "eth6", "eth7"}},
           {"group_list",
            {

            }}};

    char file_name[] = "./ut_hcom_get_hcom_info_groupsizeErr.json";
    std::ofstream outfile(file_name, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(4) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name);
    } else {
        HCCL_ERROR("open %s failed", file_name);
    }

    outfile.close();

    set_board_id(0x0000);
    std::string identify = "0";
    s32 ret = HCCL_SUCCESS;
    HcomInfo hcom_info;
    std::string ranktable_file(file_name);
    std::string rankTableM;
    std::string realFilePath;

    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcomLoadRanktableFile(ranktable_file.c_str(), rankTableM, realFilePath);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = CfgGetClusterInfo(rankTableM, identify, hcom_info.params, hcom_info.rankTable);
    EXPECT_EQ(ret, HCCL_E_PARA);

    ret = hrtResetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    set_board_id(0x0000);

    remove(file_name);
}

TEST_F(HcomTest, ut_hcom_get_hcom_info_910boardidErr)
{
    nlohmann::json rank_table
        = {{"status", "completed"},
           {"deploy_mode", "lab"},
           {"group_count", "1"},
           {"chip_info", "910"},
           {"board_id", "0x0000"},
           {"para_plane_nic_location", "device"},
           {"para_plane_nic_num", "8"},
           {"para_plane_nic_name", {"eth0", "eth1", "eth2", "eth3", "eth4", "eth5", "eth6", "eth7"}},
           {"group_list",
            {{{"group_name", ""},
              {"device_num", "16"},
              {"server_num", "2"},
              {"instance_count", "16"},
              {"instance_list",
               {
                   {{"rank_id", "0"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "0"}, {"device_ip", "192.168.0.11"}}}}},

                   {{"rank_id", "1"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "1"}, {"device_ip", "192.168.0.12"}}}}},
                   {{"rank_id", "2"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "2"}, {"device_ip", "192.168.0.13"}}}}},

                   {{"rank_id", "3"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "3"}, {"device_ip", "192.168.0.14"}}}}},
                   {{"rank_id", "4"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "4"}, {"device_ip", "192.168.0.15"}}}}},

                   {{"rank_id", "5"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "5"}, {"device_ip", "192.168.0.16"}}}}},
                   {{"rank_id", "6"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "6"}, {"device_ip", "192.168.0.17"}}}}},

                   {{"rank_id", "7"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "7"}, {"device_ip", "192.168.0.18"}}}}},
                   {{"rank_id", "8"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "0"}, {"device_ip", "192.168.0.20"}}}}},

                   {{"rank_id", "9"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "1"}, {"device_ip", "192.168.0.21"}}}}},
                   {{"rank_id", "10"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "2"}, {"device_ip", "192.168.0.22"}}}}},

                   {{"rank_id", "11"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "3"}, {"device_ip", "192.168.0.23"}}}}},
                   {{"rank_id", "12"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "4"}, {"device_ip", "192.168.0.24"}}}}},

                   {{"rank_id", "13"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "5"}, {"device_ip", "192.168.0.25"}}}}},
                   {{"rank_id", "14"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "6"}, {"device_ip", "192.168.0.26"}}}}},

                   {{"rank_id", "15"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "7"}, {"device_ip", "192.168.0.27"}}}}},
               }},
              {"server_list",
               {
                   {{"server_id", "10.0.0.10"},
                    {"para_plane_info",
                     {
                         {
                             {"eth0", "192.168.200.2"},
                         },
                         {
                             {"eth1", "192.168.201.2"},
                         },
                         {
                             {"eth2", "192.168.202.2"},
                         },
                         {
                             {"eth3", "192.168.203.2"},
                         },
                         {
                             {"eth4", "192.168.204.2"},
                         },
                         {
                             {"eth5", "192.168.205.2"},
                         },
                         {
                             {"eth6", "192.168.206.2"},
                         },
                         {
                             {"eth7", "192.168.207.2"},
                         },
                     }}

                   },
                   {{"server_id", "10.0.0.11"},
                    {"para_plane_info",
                     {
                         {
                             {"eth0", "192.168.210.3"},
                         },
                         {
                             {"eth1", "192.168.211.3"},
                         },
                         {
                             {"eth2", "192.168.212.3"},
                         },
                         {
                             {"eth3", "192.168.213.3"},
                         },
                         {
                             {"eth4", "192.168.214.3"},
                         },
                         {
                             {"eth5", "192.168.215.3"},
                         },
                         {
                             {"eth6", "192.168.216.3"},
                         },
                         {
                             {"eth7", "192.168.217.3"},
                         },
                     }}

                   },

               }}}}}};

    char file_name[] = "./ut_hcom_get_hcom_info_910boardidErr.json";
    std::ofstream outfile(file_name, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(4) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name);
    } else {
        HCCL_ERROR("open %s failed", file_name);
    }

    outfile.close();

    set_board_id(0x5000);
    std::string identify = "0";
    s32 ret = HCCL_SUCCESS;
    HcomInfo hcom_info;
    std::string ranktable_file(file_name);
    std::string rankTableM;
    std::string realFilePath;

    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcomLoadRanktableFile(ranktable_file.c_str(), rankTableM, realFilePath);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = CfgGetClusterInfo(rankTableM, identify, hcom_info.params, hcom_info.rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = hrtResetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    set_board_id(0x0000);
    remove(file_name);
}

TEST_F(HcomTest, ut_hcom_get_hcom_info_910boardidErr2)
{
    nlohmann::json rank_table
        = {{"status", "completed"},
           {"deploy_mode", "lab"},
           {"group_count", "1"},
           {"chip_info", "910"},
           {"board_id", "0x00005"},
           {"para_plane_nic_location", "device"},
           {"para_plane_nic_num", "8"},
           {"para_plane_nic_name", {"eth0", "eth1", "eth2", "eth3", "eth4", "eth5", "eth6", "eth7"}},
           {"group_list",
            {{{"group_name", ""},
              {"device_num", "16"},
              {"server_num", "2"},
              {"instance_count", "16"},
              {"instance_list",
               {
                   {{"rank_id", "0"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "0"}, {"device_ip", "192.168.0.11"}}}}},

                   {{"rank_id", "1"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "1"}, {"device_ip", "192.168.0.12"}}}}},
                   {{"rank_id", "2"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "2"}, {"device_ip", "192.168.0.13"}}}}},

                   {{"rank_id", "3"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "3"}, {"device_ip", "192.168.0.14"}}}}},
                   {{"rank_id", "4"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "4"}, {"device_ip", "192.168.0.15"}}}}},

                   {{"rank_id", "5"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "5"}, {"device_ip", "192.168.0.16"}}}}},
                   {{"rank_id", "6"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "6"}, {"device_ip", "192.168.0.17"}}}}},

                   {{"rank_id", "7"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "7"}, {"device_ip", "192.168.0.18"}}}}},
                   {{"rank_id", "8"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "0"}, {"device_ip", "192.168.0.20"}}}}},

                   {{"rank_id", "9"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "1"}, {"device_ip", "192.168.0.21"}}}}},
                   {{"rank_id", "10"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "2"}, {"device_ip", "192.168.0.22"}}}}},

                   {{"rank_id", "11"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "3"}, {"device_ip", "192.168.0.23"}}}}},
                   {{"rank_id", "12"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "4"}, {"device_ip", "192.168.0.24"}}}}},

                   {{"rank_id", "13"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "5"}, {"device_ip", "192.168.0.25"}}}}},
                   {{"rank_id", "14"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "6"}, {"device_ip", "192.168.0.26"}}}}},

                   {{"rank_id", "15"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "7"}, {"device_ip", "192.168.0.27"}}}}},
               }},
              {"server_list",
               {
                   {{"server_id", "10.0.0.10"},
                    {"para_plane_info",
                     {
                         {
                             {"eth0", "192.168.200.2"},
                         },
                         {
                             {"eth1", "192.168.201.2"},
                         },
                         {
                             {"eth2", "192.168.202.2"},
                         },
                         {
                             {"eth3", "192.168.203.2"},
                         },
                         {
                             {"eth4", "192.168.204.2"},
                         },
                         {
                             {"eth5", "192.168.205.2"},
                         },
                         {
                             {"eth6", "192.168.206.2"},
                         },
                         {
                             {"eth7", "192.168.207.2"},
                         },
                     }}

                   },
                   {{"server_id", "10.0.0.11"},
                    {"para_plane_info",
                     {
                         {
                             {"eth0", "192.168.210.3"},
                         },
                         {
                             {"eth1", "192.168.211.3"},
                         },
                         {
                             {"eth2", "192.168.212.3"},
                         },
                         {
                             {"eth3", "192.168.213.3"},
                         },
                         {
                             {"eth4", "192.168.214.3"},
                         },
                         {
                             {"eth5", "192.168.215.3"},
                         },
                         {
                             {"eth6", "192.168.216.3"},
                         },
                         {
                             {"eth7", "192.168.217.3"},
                         },
                     }}

                   },

               }}}}}};

    char file_name[] = "./ut_hcom_get_hcom_info_910boardidErr.json";
    std::ofstream outfile(file_name, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(4) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name);
    } else {
        HCCL_ERROR("open %s failed", file_name);
    }

    outfile.close();

    set_board_id(0x0000);
    std::string identify = "0";
    s32 ret = HCCL_SUCCESS;
    HcomInfo hcom_info;
    std::string ranktable_file(file_name);
    std::string rankTableM;
    std::string realFilePath;

    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcomLoadRanktableFile(ranktable_file.c_str(), rankTableM, realFilePath);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = CfgGetClusterInfo(rankTableM, identify, hcom_info.params, hcom_info.rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = hrtResetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    set_board_id(0x0000);
    remove(file_name);
}

TEST_F(HcomTest, ut_hcom_get_hcom_info_910boardidErr3)
{
    nlohmann::json rank_table
        = {{"status", "completed"},
           {"deploy_mode", "lab"},
           {"group_count", "1"},
           {"chip_info", "910"},
           {"board_id", "0x0000"},
           {"para_plane_nic_location", "device"},
           {"para_plane_nic_num", "8"},
           {"para_plane_nic_name", {"eth0", "eth1", "eth2", "eth3", "eth4", "eth5", "eth6", "eth7"}},
           {"group_list",
            {{{"group_name", ""},
              {"device_num", "16"},
              {"server_num", "2"},
              {"instance_count", "16"},
              {"instance_list",
               {
                   {{"rank_id", "0"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "0"}, {"device_ip", "192.168.0.11"}}}}},

                   {{"rank_id", "1"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "1"}, {"device_ip", "192.168.0.12"}}}}},
                   {{"rank_id", "2"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "2"}, {"device_ip", "192.168.0.13"}}}}},

                   {{"rank_id", "3"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "3"}, {"device_ip", "192.168.0.14"}}}}},
                   {{"rank_id", "4"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "4"}, {"device_ip", "192.168.0.15"}}}}},

                   {{"rank_id", "5"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "5"}, {"device_ip", "192.168.0.16"}}}}},
                   {{"rank_id", "6"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "6"}, {"device_ip", "192.168.0.17"}}}}},

                   {{"rank_id", "7"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "7"}, {"device_ip", "192.168.0.18"}}}}},
                   {{"rank_id", "8"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "0"}, {"device_ip", "192.168.0.20"}}}}},

                   {{"rank_id", "9"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "1"}, {"device_ip", "192.168.0.21"}}}}},
                   {{"rank_id", "10"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "2"}, {"device_ip", "192.168.0.22"}}}}},

                   {{"rank_id", "11"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "3"}, {"device_ip", "192.168.0.23"}}}}},
                   {{"rank_id", "12"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "4"}, {"device_ip", "192.168.0.24"}}}}},

                   {{"rank_id", "13"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "5"}, {"device_ip", "192.168.0.25"}}}}},
                   {{"rank_id", "14"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "6"}, {"device_ip", "192.168.0.26"}}}}},

                   {{"rank_id", "15"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "7"}, {"device_ip", "192.168.0.27"}}}}},
               }},
              {"server_list",
               {
                   {{"server_id", "10.0.0.10"},
                    {"para_plane_info",
                     {
                         {
                             {"eth0", "192.168.200.2"},
                         },
                         {
                             {"eth1", "192.168.201.2"},
                         },
                         {
                             {"eth2", "192.168.202.2"},
                         },
                         {
                             {"eth3", "192.168.203.2"},
                         },
                         {
                             {"eth4", "192.168.204.2"},
                         },
                         {
                             {"eth5", "192.168.205.2"},
                         },
                         {
                             {"eth6", "192.168.206.2"},
                         },
                         {
                             {"eth7", "192.168.207.2"},
                         },
                     }}

                   },
                   {{"server_id", "10.0.0.11"},
                    {"para_plane_info",
                     {
                         {
                             {"eth0", "192.168.210.3"},
                         },
                         {
                             {"eth1", "192.168.211.3"},
                         },
                         {
                             {"eth2", "192.168.212.3"},
                         },
                         {
                             {"eth3", "192.168.213.3"},
                         },
                         {
                             {"eth4", "192.168.214.3"},
                         },
                         {
                             {"eth5", "192.168.215.3"},
                         },
                         {
                             {"eth6", "192.168.216.3"},
                         },
                         {
                             {"eth7", "192.168.217.3"},
                         },
                     }}

                   },

               }}}}}};

    char file_name[] = "./ut_hcom_get_hcom_info_boardidErr.json";
    std::ofstream outfile(file_name, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(4) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name);
    } else {
        HCCL_ERROR("open %s failed", file_name);
    }

    outfile.close();

    set_board_id(0x0000);
    std::string identify = "0";
    s32 ret = HCCL_SUCCESS;
    HcomInfo hcom_info;
    std::string ranktable_file(file_name);
    std::string rankTableM;
    std::string realFilePath;

    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcomLoadRanktableFile(ranktable_file.c_str(), rankTableM, realFilePath);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = CfgGetClusterInfo(rankTableM, identify, hcom_info.params, hcom_info.rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = hrtResetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    set_board_id(0x0000);
    remove(file_name);
}

TEST_F(HcomTest, ut_hcom_get_hcom_info_910boardidErr4)
{
    nlohmann::json rank_table
        = {{"status", "completed"},
           {"deploy_mode", "lab"},
           {"group_count", "1"},
           {"chip_info", "910"},
           {"board_id", "0x00"},
           {"para_plane_nic_location", "device"},
           {"para_plane_nic_num", "8"},
           {"para_plane_nic_name", {"eth0", "eth1", "eth2", "eth3", "eth4", "eth5", "eth6", "eth7"}},
           {"group_list",
            {{{"group_name", ""},
              {"device_num", "16"},
              {"server_num", "2"},
              {"instance_count", "16"},
              {"instance_list",
               {
                   {{"rank_id", "0"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "0"}, {"device_ip", "192.168.0.11"}}}}},

                   {{"rank_id", "1"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "1"}, {"device_ip", "192.168.0.12"}}}}},
                   {{"rank_id", "2"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "2"}, {"device_ip", "192.168.0.13"}}}}},

                   {{"rank_id", "3"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "3"}, {"device_ip", "192.168.0.14"}}}}},
                   {{"rank_id", "4"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "4"}, {"device_ip", "192.168.0.15"}}}}},

                   {{"rank_id", "5"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "5"}, {"device_ip", "192.168.0.16"}}}}},
                   {{"rank_id", "6"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "6"}, {"device_ip", "192.168.0.17"}}}}},

                   {{"rank_id", "7"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "7"}, {"device_ip", "192.168.0.18"}}}}},
                   {{"rank_id", "8"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "0"}, {"device_ip", "192.168.0.20"}}}}},

                   {{"rank_id", "9"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "1"}, {"device_ip", "192.168.0.21"}}}}},
                   {{"rank_id", "10"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "2"}, {"device_ip", "192.168.0.22"}}}}},

                   {{"rank_id", "11"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "3"}, {"device_ip", "192.168.0.23"}}}}},
                   {{"rank_id", "12"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "4"}, {"device_ip", "192.168.0.24"}}}}},

                   {{"rank_id", "13"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "5"}, {"device_ip", "192.168.0.25"}}}}},
                   {{"rank_id", "14"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "6"}, {"device_ip", "192.168.0.26"}}}}},

                   {{"rank_id", "15"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "7"}, {"device_ip", "192.168.0.27"}}}}},
               }},
              {"server_list",
               {
                   {{"server_id", "10.0.0.10"},
                    {"para_plane_info",
                     {
                         {
                             {"eth0", "192.168.200.2"},
                         },
                         {
                             {"eth1", "192.168.201.2"},
                         },
                         {
                             {"eth2", "192.168.202.2"},
                         },
                         {
                             {"eth3", "192.168.203.2"},
                         },
                         {
                             {"eth4", "192.168.204.2"},
                         },
                         {
                             {"eth5", "192.168.205.2"},
                         },
                         {
                             {"eth6", "192.168.206.2"},
                         },
                         {
                             {"eth7", "192.168.207.2"},
                         },
                     }}

                   },
                   {{"server_id", "10.0.0.11"},
                    {"para_plane_info",
                     {
                         {
                             {"eth0", "192.168.210.3"},
                         },
                         {
                             {"eth1", "192.168.211.3"},
                         },
                         {
                             {"eth2", "192.168.212.3"},
                         },
                         {
                             {"eth3", "192.168.213.3"},
                         },
                         {
                             {"eth4", "192.168.214.3"},
                         },
                         {
                             {"eth5", "192.168.215.3"},
                         },
                         {
                             {"eth6", "192.168.216.3"},
                         },
                         {
                             {"eth7", "192.168.217.3"},
                         },
                     }}

                   },

               }}}}}};

    char file_name[] = "./ut_hcom_get_hcom_info_910boardidErr4.json";
    std::ofstream outfile(file_name, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(4) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name);
    } else {
        HCCL_ERROR("open %s failed", file_name);
    }

    outfile.close();

    set_board_id(0x0020);
    std::string identify = "0";
    s32 ret = HCCL_SUCCESS;
    HcomInfo hcom_info;
    std::string ranktable_file(file_name);
    std::string rankTableM;
    std::string realFilePath;

    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcomLoadRanktableFile(ranktable_file.c_str(), rankTableM, realFilePath);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = CfgGetClusterInfo(rankTableM, identify, hcom_info.params, hcom_info.rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = hrtResetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    set_board_id(0x0000);
    remove(file_name);
}

TEST_F(HcomTest, ut_hcom_get_hcom_info_chip_info_arm)
{
    nlohmann::json rank_table
        = {{"status", "completed"},
           {"deploy_mode", "lab"},
           {"group_count", "1"},
           {"chip_info", "910"},
           {"board_id", "0x002A"},
           {"para_plane_nic_location", "device"},
           {"para_plane_nic_num", "8"},
           {"para_plane_nic_name", {"eth0", "eth1", "eth2", "eth3", "eth4", "eth5", "eth6", "eth7"}},
           {"group_list",
            {{{"group_name", ""},
              {"device_num", "16"},
              {"server_num", "2"},
              {"instance_count", "16"},
              {"instance_list",
               {
                   {{"rank_id", "0"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "0"}, {"device_ip", "192.168.0.11"}}}}},

                   {{"rank_id", "1"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "1"}, {"device_ip", "192.168.0.12"}}}}},
                   {{"rank_id", "2"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "2"}, {"device_ip", "192.168.0.13"}}}}},

                   {{"rank_id", "3"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "3"}, {"device_ip", "192.168.0.14"}}}}},
                   {{"rank_id", "4"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "4"}, {"device_ip", "192.168.0.15"}}}}},

                   {{"rank_id", "5"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "5"}, {"device_ip", "192.168.0.16"}}}}},
                   {{"rank_id", "6"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "6"}, {"device_ip", "192.168.0.17"}}}}},

                   {{"rank_id", "7"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "7"}, {"device_ip", "192.168.0.18"}}}}},
                   {{"rank_id", "8"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "0"}, {"device_ip", "192.168.0.20"}}}}},

                   {{"rank_id", "9"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "1"}, {"device_ip", "192.168.0.21"}}}}},
                   {{"rank_id", "10"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "2"}, {"device_ip", "192.168.0.22"}}}}},

                   {{"rank_id", "11"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "3"}, {"device_ip", "192.168.0.23"}}}}},
                   {{"rank_id", "12"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "4"}, {"device_ip", "192.168.0.24"}}}}},

                   {{"rank_id", "13"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "5"}, {"device_ip", "192.168.0.25"}}}}},
                   {{"rank_id", "14"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "6"}, {"device_ip", "192.168.0.26"}}}}},

                   {{"rank_id", "15"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "7"}, {"device_ip", "192.168.0.27"}}}}},
               }},
              {"server_list",
               {
                   {{"server_id", "10.0.0.10"},
                    {"para_plane_info",
                     {
                         {
                             {"eth0", "192.168.200.2"},
                         },
                         {
                             {"eth1", "192.168.201.2"},
                         },
                         {
                             {"eth2", "192.168.202.2"},
                         },
                         {
                             {"eth3", "192.168.203.2"},
                         },
                         {
                             {"eth4", "192.168.204.2"},
                         },
                         {
                             {"eth5", "192.168.205.2"},
                         },
                         {
                             {"eth6", "192.168.206.2"},
                         },
                         {
                             {"eth7", "192.168.207.2"},
                         },
                     }}

                   },
                   {{"server_id", "10.0.0.11"},
                    {"para_plane_info",
                     {
                         {
                             {"eth0", "192.168.210.3"},
                         },
                         {
                             {"eth1", "192.168.211.3"},
                         },
                         {
                             {"eth2", "192.168.212.3"},
                         },
                         {
                             {"eth3", "192.168.213.3"},
                         },
                         {
                             {"eth4", "192.168.214.3"},
                         },
                         {
                             {"eth5", "192.168.215.3"},
                         },
                         {
                             {"eth6", "192.168.216.3"},
                         },
                         {
                             {"eth7", "192.168.217.3"},
                         },
                     }}

                   },

               }}}}}};

    char file_name[] = "./ut_hcom_get_hcom_info_chip_infoErr.json";
    std::ofstream outfile(file_name, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(4) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name);
    } else {
        HCCL_ERROR("open %s failed", file_name);
    }

    outfile.close();

    set_board_id(0x002B);
    std::string identify = "0";
    s32 ret = HCCL_SUCCESS;
    HcomInfo hcom_info;
    std::string ranktable_file(file_name);
    std::string rankTableM;
    std::string realFilePath;

    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcomLoadRanktableFile(ranktable_file.c_str(), rankTableM, realFilePath);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = CfgGetClusterInfo(rankTableM, identify, hcom_info.params, hcom_info.rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = hrtResetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    set_board_id(0x0000);
    remove(file_name);
}

TEST_F(HcomTest, ut_hcom_get_hcom_info_chip_infoErr3)
{
    nlohmann::json rank_table
        = {{"status", "completed"},
           {"deploy_mode", "lab"},
           {"group_count", "1"},
           {"chip_info", "910"},
           {"board_id", "0x003A"},
           {"para_plane_nic_location", "device"},
           {"para_plane_nic_num", "8"},
           {"para_plane_nic_name", {"eth0", "eth1", "eth2", "eth3", "eth4", "eth5", "eth6", "eth7"}},
           {"group_list",
            {{{"group_name", ""},
              {"device_num", "16"},
              {"server_num", "2"},
              {"instance_count", "16"},
              {"instance_list",
               {
                   {{"rank_id", "0"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "0"}, {"device_ip", "192.168.0.11"}}}}},

                   {{"rank_id", "1"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "1"}, {"device_ip", "192.168.0.12"}}}}},
                   {{"rank_id", "2"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "2"}, {"device_ip", "192.168.0.13"}}}}},

                   {{"rank_id", "3"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "3"}, {"device_ip", "192.168.0.14"}}}}},
                   {{"rank_id", "4"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "4"}, {"device_ip", "192.168.0.15"}}}}},

                   {{"rank_id", "5"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "5"}, {"device_ip", "192.168.0.16"}}}}},
                   {{"rank_id", "6"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "6"}, {"device_ip", "192.168.0.17"}}}}},

                   {{"rank_id", "7"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "7"}, {"device_ip", "192.168.0.18"}}}}},
                   {{"rank_id", "8"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "0"}, {"device_ip", "192.168.0.20"}}}}},

                   {{"rank_id", "9"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "1"}, {"device_ip", "192.168.0.21"}}}}},
                   {{"rank_id", "10"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "2"}, {"device_ip", "192.168.0.22"}}}}},

                   {{"rank_id", "11"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "3"}, {"device_ip", "192.168.0.23"}}}}},
                   {{"rank_id", "12"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "4"}, {"device_ip", "192.168.0.24"}}}}},

                   {{"rank_id", "13"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "5"}, {"device_ip", "192.168.0.25"}}}}},
                   {{"rank_id", "14"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "6"}, {"device_ip", "192.168.0.26"}}}}},

                   {{"rank_id", "15"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "7"}, {"device_ip", "192.168.0.27"}}}}},
               }},
              {"server_list",
               {
                   {{"server_id", "10.0.0.10"},
                    {"para_plane_info",
                     {
                         {
                             {"eth0", "192.168.200.2"},
                         },
                         {
                             {"eth1", "192.168.201.2"},
                         },
                         {
                             {"eth2", "192.168.202.2"},
                         },
                         {
                             {"eth3", "192.168.203.2"},
                         },
                         {
                             {"eth4", "192.168.204.2"},
                         },
                         {
                             {"eth5", "192.168.205.2"},
                         },
                         {
                             {"eth6", "192.168.206.2"},
                         },
                         {
                             {"eth7", "192.168.207.2"},
                         },
                     }}

                   },
                   {{"server_id", "10.0.0.11"},
                    {"para_plane_info",
                     {
                         {
                             {"eth0", "192.168.210.3"},
                         },
                         {
                             {"eth1", "192.168.211.3"},
                         },
                         {
                             {"eth2", "192.168.212.3"},
                         },
                         {
                             {"eth3", "192.168.213.3"},
                         },
                         {
                             {"eth4", "192.168.214.3"},
                         },
                         {
                             {"eth5", "192.168.215.3"},
                         },
                         {
                             {"eth6", "192.168.216.3"},
                         },
                         {
                             {"eth7", "192.168.217.3"},
                         },
                     }}

                   },

               }}}}}};

    char file_name[] = "./ut_hcom_get_hcom_info_chip_infoErr.json";
    std::ofstream outfile(file_name, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(4) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name);
    } else {
        HCCL_ERROR("open %s failed", file_name);
    }

    outfile.close();

    set_board_id(0x002C);
    std::string identify = "0";
    s32 ret = HCCL_SUCCESS;
    HcomInfo hcom_info;
    std::string ranktable_file(file_name);
    std::string rankTableM;
    std::string realFilePath;

    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcomLoadRanktableFile(ranktable_file.c_str(), rankTableM, realFilePath);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = CfgGetClusterInfo(rankTableM, identify, hcom_info.params, hcom_info.rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = hrtResetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    set_board_id(0x0000);
    remove(file_name);
}

TEST_F(HcomTest, ut_hcom_get_new_rank_info)
{
    nlohmann::json rank_table
        = {{"status", "completed"},
           {"version", "1.0"},
           {"server_count", "1"},
           {"server_list",
            {{
                {"server_id", "10.0.0.10"},
                {"host_nic_ip", "192.168.0.12:0"},
                {"device",
                 {
                     {{"rank_id", "0"}, {"device_id", "0"}, {"device_ip", "192.168.0.12"}

                     },
                 }},
            }}}};

    char file_name_t[] = "./ut_hcom_get_new_rank_info.json";
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
    u32 numHccsLink = 0;
    char* rank_table_file = "./ut_hcom_get_new_rank_info.json";
    char* rank_ID = "0";

    ret = HcomInitByFile(rank_table_file, rank_ID);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcomGetLocalRankSize(NULL, &localranksize);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcomGetLocalRankId(NULL, &localrankid);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcomGetHccsLinkNum(NULL, &numHccsLink);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcomDestroy();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    remove(file_name_t);

    EXPECT_EQ(localranksize, 1);
    EXPECT_EQ(localrankid, 0);
    ret = hrtResetDevice(0);
}

TEST_F(HcomTest, ut_hcom_get_new_rank_info_ERR)
{
    nlohmann::json rank_table
        = {{"status", "completed"},
           {"version", "2.0"},
           {"server_count", "1"},
           {"server_list",
            {{
                {"server_id", "10.0.0.10"},
                {"host_nic_ip", "192.168.0.12:0"},
                {"device",
                 {
                     {{"rank_id", "0"}, {"device_id", "0"}, {"device_ip", "192.168.0.12"}

                     },
                 }},
            }}}};

    char file_name_t[] = "./ut_hcom_get_new_rank_info_ERR.json";
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
    char* rank_table_file = "./ut_hcom_get_new_rank_info_ERR.json";
    char* rank_ID = "0";
    std::string rankTableM;
    std::string realFilePath;

    HcomInfo hcom_info;
    ret = HcomLoadRanktableFile(rank_table_file, rankTableM, realFilePath);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = CfgGetClusterInfo(rankTableM, rank_ID, hcom_info.params, hcom_info.rankTable);
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);

    remove(file_name_t);
}

TEST_F(HcomTest, ut_hcom_get_new_rank_info_serverCountERR)
{
    nlohmann::json rank_table
        = {{"status", "completed"},
           {"version", "1.0"},
           {"server_count", "2"},
           {"server_list",
            {{
                {"server_id", "10.0.0.10"},
                {"host_nic_ip", "192.168.0.12:0"},
                {"device",
                 {
                     {{"rank_id", "0"}, {"device_id", "0"}, {"device_ip", "192.168.0.12"}

                     },
                 }},
            }}}};

    char file_name_t[] = "./ut_hcom_get_new_rank_info_serverCountERR.json";
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
    char* rank_table_file = "./ut_hcom_get_new_rank_info_serverCountERR.json";
    char* rank_ID = "0";
    std::string rankTableM;
    std::string realFilePath;

    HcomInfo hcom_info;
    ret = HcomLoadRanktableFile(rank_table_file, rankTableM, realFilePath);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = CfgGetClusterInfo(rankTableM, rank_ID, hcom_info.params, hcom_info.rankTable);
    EXPECT_EQ(ret, HCCL_E_PARA);

    remove(file_name_t);
}

TEST_F(HcomTest, ut_hcom_get_new_rank_info_muti_ip)
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

    ret = HcomDestroy();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    remove(file_name_t);
}

TEST_F(HcomTest, ut_hcom_get_new_rank_info_devId_err)
{
    nlohmann::json rank_table
        = {{"status", "completed"},
           {"version", "1.0"},
           {"server_count", "1"},
           {"server_list",
            {{
                {"server_id", "10.0.0.10"},
                {"host_nic_ip", "192.168.0.12:198,192.168.1.12:199"},
                {"device",
                 {
                     {{"rank_id", "0"}, {"device_id", "10"}, {"device_ip", "192.168.0.12,192.168.1.12"}

                     },
                 }},
            }}}};

    char file_name_t[] = "./ut_hcom_get_new_rank_info_devId_err.json";
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
    char* rank_table_file = "./ut_hcom_get_new_rank_info_devId_err.json";
    char* rank_ID = "0";
    std::string rankTableM;
    std::string realFilePath;

    HcomInfo hcom_info;
    ret = HcomLoadRanktableFile(rank_table_file, rankTableM, realFilePath);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = CfgGetClusterInfo(rankTableM, rank_ID, hcom_info.params, hcom_info.rankTable);
    EXPECT_EQ(ret, HCCL_E_PARA);

    remove(file_name_t);
}

TEST_F(HcomTest, ut_hcom_get_new_rank_info_rankId_err)
{
    nlohmann::json rank_table
        = {{"status", "completed"},
           {"version", "1.0"},
           {"server_count", "1"},
           {"server_list",
            {{
                {"server_id", "10.0.0.10"},
                {"host_nic_ip", "192.168.0.12:198,192.168.1.12:199"},
                {"device",
                 {
                     {{"rank_id", "2"}, {"device_id", "0"}, {"device_ip", "192.168.0.12,192.168.1.12"}

                     },
                 }},
            }}}};

    char file_name_t[] = "./ut_hcom_get_new_rank_info_rankId_err.json";
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
    char* rank_table_file = "./ut_hcom_get_new_rank_info_rankId_err.json";
    char* rank_ID = "2";
    std::string rankTableM;
    std::string realFilePath;

    HcomInfo hcom_info;
    ret = HcomLoadRanktableFile(rank_table_file, rankTableM, realFilePath);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = CfgGetClusterInfo(rankTableM, rank_ID, hcom_info.params, hcom_info.rankTable);
    EXPECT_EQ(ret, HCCL_E_PARA);

    remove(file_name_t);
}

TEST_F(HcomTest, ut_hcom_get_new_rank_info_mutiserver_devID)
{
    nlohmann::json rank_table
        = {{"status", "completed"},
           {"version", "1.0"},
           {"server_count", "2"},
           {"server_list",
            {{
                 {"server_id", "10.0.0.10"},
                 {"host_nic_ip", "192.168.0.12:0,192.168.1.12:199"},
                 {"device",
                  {{{"rank_id", "0"}, {"device_id", "0"}, {"device_ip", "192.168.0.12,192.168.0.13"}

                   },
                   {{"rank_id", "1"}, {"device_id", "1"}, {"device_ip", "192.168.1.12,192.168.1.13"}

                   }}},
             },
             {
                 {"server_id", "10.0.0.11"},
                 {"host_nic_ip", "192.168.2.12:0,192.168.3.12:199"},
                 {"device",
                  {{{"rank_id", "2"}, {"device_id", "2"}, {"device_ip", "192.168.2.12,192.168.2.13"}

                   },
                   {{"rank_id", "3"}, {"device_id", "3"}, {"device_ip", "192.168.3.12,192.168.3.13"}

                   }}},
             }}}};

    char file_name_t[] = "./ut_hcom_get_new_rank_info_mutiserver_devID.json";
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
    char* rank_table_file = "./ut_hcom_get_new_rank_info_mutiserver_devID.json";
    char* rank_ID = "0";
    std::string rankTableM;
    std::string realFilePath;

    HcomInfo hcom_info;
    ret = HcomLoadRanktableFile(rank_table_file, rankTableM, realFilePath);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = CfgGetClusterInfo(rankTableM, rank_ID, hcom_info.params, hcom_info.rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    remove(file_name_t);
}

TEST_F(HcomTest, ut_hcom_get_new_rank_info_sameRankid)
{
    nlohmann::json rank_table
        = {{"status", "completed"},
           {"version", "1.0"},
           {"server_count", "2"},
           {"server_list",
            {{
                 {"server_id", "10.0.0.10"},
                 {"host_nic_ip", "192.168.0.12:0,192.168.1.12:199"},
                 {"device",
                  {{{"rank_id", "0"}, {"device_id", "0"}, {"device_ip", "192.168.0.12,192.168.0.13"}

                   },
                   {{"rank_id", "1"}, {"device_id", "1"}, {"device_ip", "192.168.1.12,192.168.1.13"}

                   }}},
             },
             {
                 {"server_id", "10.0.0.11"},
                 {"host_nic_ip", "192.168.2.12:0,192.168.3.12:199"},
                 {"device",
                  {{{"rank_id", "0"}, {"device_id", "0"}, {"device_ip", "192.168.2.12,192.168.2.13"}

                   },
                   {{"rank_id", "1"}, {"device_id", "1"}, {"device_ip", "192.168.3.12,192.168.3.13"}

                   }}},
             }}}};

    char file_name_t[] = "./ut_hcom_get_new_rank_info_sameRankid.json";
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
    char* rank_table_file = "./ut_hcom_get_new_rank_info_sameRankid.json";
    char* rank_ID = "0";
    std::string rankTableM;
    std::string realFilePath;

    HcomInfo hcom_info;
    ret = HcomLoadRanktableFile(rank_table_file, rankTableM, realFilePath);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = CfgGetClusterInfo(rankTableM, rank_ID, hcom_info.params, hcom_info.rankTable);
    EXPECT_EQ(ret, HCCL_E_PARA);

    remove(file_name_t);
}

TEST_F(HcomTest, ut_hcom_get_new_rank_info_muti)
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
                     {{"rank_id", "0"}, {"device_id", "0"}, {"device_ip", "192.168.0.12,192.168.0.13"}

                     },
                     {{"rank_id", "1"}, {"device_id", "1"}, {"device_ip", "192.168.1.12,192.168.1.13"}

                     },
                     {{"rank_id", "2"}, {"device_id", "2"}, {"device_ip", "192.168.2.12,192.168.2.13"}

                     },
                 }},
            }}}};

    char file_name_t[] = "./ut_hcom_get_new_rank_info_muti.json";
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
    char* rank_table_file = "./ut_hcom_get_new_rank_info_muti.json";
    char* rank_ID = "0";
    std::string rankTableM;
    std::string realFilePath;

    HcomInfo hcom_info;
    ret = HcomLoadRanktableFile(rank_table_file, rankTableM, realFilePath);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = CfgGetClusterInfo(rankTableM, rank_ID, hcom_info.params, hcom_info.rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    remove(file_name_t);
}

#if 1
TEST_F(HcomTest, ut_hcom_get_new_ranktable_info)
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
                     {{"rank_id", "0"}, {"device_id", "0"}, {"device_ip", "192.168.0.12,192.168.0.13"}

                     },
                     {{"rank_id", "1"}, {"device_id", "1"}, {"device_ip", "192.168.1.12,192.168.1.13"}

                     },
                     {{"rank_id", "2"}, {"device_id", "2"}, {"device_ip", "192.168.2.12,192.168.2.13"}

                     },
                 }},
            }}}};

    char file_name_t[] = "./ut_hcom_get_new_ranktable_info.json";
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

    string rank_table_file("./ut_hcom_get_new_ranktable_info.json");
    string rank_ID("0");

    HcomInfo hcom_info;
    std::string rankTableM;
    std::string realFilePath;

    ret = HcomLoadRanktableFile(rank_table_file.c_str(), rankTableM, realFilePath);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    TopoInfoRanktableParser myTopoRanktable(rankTableM, rank_ID);
    ret = myTopoRanktable.Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = myTopoRanktable.GetClusterInfo(hcom_info.params, hcom_info.rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = myTopoRanktable.GetClusterInfo(hcom_info.rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    remove(file_name_t);
}
#endif

#if 1
TEST_F(HcomTest, ut_hcom_get_new_ranktable_info_rankID)
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
                     {{"rank_id", "2"}, {"device_id", "0"}, {"device_ip", "192.168.0.12,192.168.0.13"}

                     },
                     {{"rank_id", "1"}, {"device_id", "1"}, {"device_ip", "192.168.1.12,192.168.1.13"}

                     },
                     {{"rank_id", "0"}, {"device_id", "2"}, {"device_ip", "192.168.2.12,192.168.2.13"}

                     },
                 }},
            }}}};

    char file_name_t[] = "./ut_hcom_get_new_ranktable_info_rankID.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(1) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    } else {
        HCCL_ERROR("open %s failed", file_name_t);
    }

    outfile.close();
    int ret = HCCL_SUCCESS;

    ret = hrtSetDevice(2);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    string rank_table_file("./ut_hcom_get_new_ranktable_info_rankID.json");
    string rank_ID("0");
    std::string rankTableM;
    std::string realFilePath;

    HcomInfo hcom_info;
    ret = HcomLoadRanktableFile(rank_table_file.c_str(), rankTableM, realFilePath);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = CfgGetClusterInfo(rankTableM, rank_ID, hcom_info.params, hcom_info.rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = hrtSetDevice(0);
    remove(file_name_t);
}
#endif

TEST_F(HcomTest, ut_hcom_test_config)
{
    HcomInfo hcom_info;
    RankInfo_t myRank;
    hcom_info.rankTable.rankList.push_back(myRank);
    HcclResult ret = CheckRankTableConfigInfo(hcom_info.rankTable.rankList, 2, 1);
    EXPECT_EQ(ret, HCCL_E_PARA);
    ret = CheckDeviceId(hcom_info.rankTable.rankList, 1, 0);
    EXPECT_EQ(ret, HCCL_E_PARA);
    ret = CheckRankListInfo(hcom_info.rankTable.rankList, 0, 1);
    EXPECT_EQ(ret, HCCL_E_PARA);
    ret = CheckRankListInfo(hcom_info.rankTable.rankList, 1, 0);
    EXPECT_EQ(ret, HCCL_E_PARA);
    ret = CheckRankListInfo(hcom_info.rankTable.rankList, 4097, 1);
    EXPECT_EQ(ret, HCCL_E_PARA);
    ret = CheckRankListInfo(hcom_info.rankTable.rankList, 1, 4097);
    EXPECT_EQ(ret, HCCL_E_PARA);
    ret = CheckDeviceNumValid(hcom_info.rankTable.rankList, 2, 0);
    EXPECT_EQ(ret, HCCL_E_PARA);
}
#if 1
TEST_F(HcomTest, ut_hcom_get_new_ranktable_info_noIP)
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
                     {{"rank_id", "2"}, {"device_id", "0"}

                     },
                     {{"rank_id", "1"}, {"device_id", "1"}, {"device_ip", ""}

                     },
                     {{"rank_id", "0"}, {"device_id", "2"}, {"device_ip", "10.0.0.10"}

                     },
                 }},
            }}}};

    char file_name_t[] = "./ut_hcom_get_new_ranktable_info_noIP.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(1) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    } else {
        HCCL_ERROR("open %s failed", file_name_t);
    }

    outfile.close();
    int ret = HCCL_SUCCESS;

    ret = hrtSetDevice(2);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    string rank_table_file("./ut_hcom_get_new_ranktable_info_noIP.json");
    string rank_ID("0");
    std::string rankTableM;
    std::string realFilePath;

    HcomInfo hcom_info;
    ret = HcomLoadRanktableFile(rank_table_file.c_str(), rankTableM, realFilePath);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = CfgGetClusterInfo(rankTableM, rank_ID, hcom_info.params, hcom_info.rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = hrtSetDevice(0);
    remove(file_name_t);
}
#endif

TEST_F(HcomTest, ut_hcom_alltoallv)
{
    nlohmann::json rank_table = rank_table_910_1server_1rank;
    char file_name_t[] = "./ut_hcom_alltoallv.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(1) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    } else {
        HCCL_ERROR("open %s failed", file_name_t);
    }

    outfile.close();
    s32 deviceId = 0;
    char* identify = "0";
    s32 rankSize = 1;
    s32 rank = atoi(identify);
    u64 count = 2;
    HCCL_INFO("alltoall_int32 : identify[%d], count[%llu]", rank, count);
    u32 devLogicId = 0xFFFF;
    HcclResult ret = hrtGetDeviceIndexByPhyId(deviceId, devLogicId);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = hrtSetDevice(devLogicId);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    char* rankTableFile = "./ut_hcom_alltoallv.json";
    ret = HcomInitByFile(rankTableFile, identify);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ResetInitState();

    const int COUNT_PER_RANK = count;
    u64 memSize = COUNT_PER_RANK * rankSize * sizeof(s32);
    HostMem hostSendMem = HostMem::alloc(memSize);
    memset_s(hostSendMem.ptr(), memSize, 0, COUNT_PER_RANK * rankSize);
    for (u32 i = 0; i < COUNT_PER_RANK * rankSize; i++) {
        *((s32*)hostSendMem.ptr() + i) = rank + 1;
    }

    // 构造入参
    vector<u64> sendCounts(rankSize, COUNT_PER_RANK);
    vector<u64> recvCounts(rankSize, COUNT_PER_RANK);
    vector<u64> sdispls(rankSize, 0);
    vector<u64> rdispls(rankSize, 0);
    for (int i = 0; i < rankSize; i++) {
        sdispls[i] = COUNT_PER_RANK * i;
        rdispls[i] = COUNT_PER_RANK * i;
        HCCL_INFO("num[%d] displs[%d]", i, COUNT_PER_RANK * i);
    }

    DeviceMem sendMem = DeviceMem::alloc(memSize);
    ret = hrtMemSyncCopy(
        sendMem.ptr(), memSize, hostSendMem.ptr(), memSize, HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    DeviceMem recvMem = DeviceMem::alloc(memSize);

    hccl::Stream stream(StreamType::STREAM_TYPE_OFFLINE);
    MOCKER_CPP(&hcclComm::GetNumBlocks).stubs().will(returnValue(HCCL_SUCCESS));
    ret = HcomAlltoAllV(
        sendMem.ptr(), sendCounts.data(), sdispls.data(), HCCL_DATA_TYPE_INT32, recvMem.ptr(), recvCounts.data(),
        rdispls.data(), HCCL_DATA_TYPE_INT32, nullptr, stream.ptr(), "hcom_alltoallv");
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = hcclStreamSynchronize(stream.ptr());

    HCCL_INFO("HcomDestory start");

    ret = HcomDestroy();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    remove(file_name_t);
    (void)aclrtResetDevice(0);
}

TEST_F(HcomTest, ut_hcom_alltoallv_null_input)
{
    nlohmann::json rank_table = rank_table_910_1server_1rank;
    char file_name_t[] = "./ut_hcom_alltoallv_null_input.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(1) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    } else {
        HCCL_ERROR("open %s failed", file_name_t);
    }

    outfile.close();
    s32 deviceId = 0;
    char* identify = "0";
    s32 rankSize = 1;
    s32 rank = atoi(identify);
    u64 count = 2;
    HCCL_INFO("alltoall_int32 : identify[%d], count[%llu]", rank, count);
    u32 devLogicId = 0xFFFF;
    HcclResult ret = hrtGetDeviceIndexByPhyId(deviceId, devLogicId);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = hrtSetDevice(devLogicId);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    char* rankTableFile = "./ut_hcom_alltoallv_null_input.json";
    ret = HcomInitByFile(rankTableFile, identify);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ResetInitState();

    const int COUNT_PER_RANK = count;
    u64 memSize = COUNT_PER_RANK * rankSize * sizeof(s32);

    // 构造入参
    vector<u64> sendCounts(rankSize, 0);
    vector<u64> recvCounts(rankSize, 0);
    vector<u64> sdispls(rankSize, 0);
    vector<u64> rdispls(rankSize, 0);

    hccl::Stream stream(StreamType::STREAM_TYPE_OFFLINE);
    MOCKER_CPP(&hcclComm::GetNumBlocks).stubs().will(returnValue(HCCL_SUCCESS));
    ret = HcomAlltoAllV(
        nullptr, sendCounts.data(), sdispls.data(), HCCL_DATA_TYPE_INT32, nullptr, recvCounts.data(), rdispls.data(),
        HCCL_DATA_TYPE_INT32, nullptr, stream.ptr(), "hcom_alltoallv");
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = hcclStreamSynchronize(stream.ptr());

    HCCL_INFO("HcomDestory start");

    ret = HcomDestroy();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    remove(file_name_t);
    (void)aclrtResetDevice(0);
}

TEST_F(HcomTest, ut_hcom_init_by_string)
{
    nlohmann::json rank_table
        = {{"collective_id", "192.168.3.3-9527-0001"},
           {"master_ip", "192.168.0.100"},
           {"master_port", "18000"},
           {"status", "completed"},
           {"version", "1.1"},
           {"node_list",
            {{{"node_addr", "192.168.0.101"}, {"ranks", {{{"rank_id", "0"}, {"device_id", "0"}}}}},
             {{"node_addr", "192.168.1.101"}, {"ranks", {{{"rank_id", "1"}, {"device_id", "0"}}}}},
             {{"node_addr", "192.168.2.101"}, {"ranks", {{{"rank_id", "2"}, {"device_id", "0"}}}}},
             {{"node_addr", "192.168.3.101"}, {"ranks", {{{"rank_id", "3"}, {"device_id", "0"}}}}}}}};
    MOCKER(Is310PDevice).stubs().will(returnValue(true));

    std::string rank_table_string = rank_table.dump();
    HcclResult ret;

    ret = HcomInitByString(rank_table_string.c_str(), "0");
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = HcomDestroy();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcomInitByString(rank_table_string.c_str(), "1");
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = HcomDestroy();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::string rank_table_string_invalid(40 * 1024 * 1024 + 1, 'a');
    ret = HcomInitByString(rank_table_string_invalid.c_str(), "2");
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(HcomTest, ut_hcom_get_dev_phy_id)
{
    nlohmann::json rank_table = rank_table_910_1server_1rank;
    char file_name_t[] = "./ut_hcom_get_dev_phy_id.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(1) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    } else {
        HCCL_ERROR("open %s failed", file_name_t);
    }

    outfile.close();
    s32 deviceId = 0;
    char* identify = "0";
    s32 rankSize = 1;
    s32 rank = atoi(identify);
    u64 count = 2;
    HCCL_INFO("alltoall_int32 : identify[%d], count[%llu]", rank, count);
    u32 devLogicId = 0xFFFF;
    HcclResult ret = hrtGetDeviceIndexByPhyId(deviceId, devLogicId);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = hrtSetDevice(devLogicId);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    char* rankTableFile = "./ut_hcom_get_dev_phy_id.json";
    ret = HcomInitByFile(rankTableFile, identify);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    const char* group = HCCL_WORLD_GROUP;
    s32 devId = 0;
    ret = HcomGetDevId(group, &devId);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    HCCL_INFO("HcomDestory start");

    ret = HcomDestroy();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    remove(file_name_t);
    (void)aclrtResetDevice(0);
}

TEST_F(HcomTest, ut_hcom_HcclCommGraphGetDevId)
{
    hccl::hcclComm* comm = new hccl::hcclComm(1, 1, "123");
    comm->communicator_.reset(new (std::nothrow) HcclCommunicator());
    comm->communicator_->deviceLogicId_ = 1;
    s64 opBaseHcom = (s64)comm;
    s32 devId = 0;
    HcclResult ret = HcclCommGraphGetDevId(opBaseHcom, &devId);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    delete comm;
}

TEST_F(HcomTest, ut_hcom_get_dev_phy_id_group)
{
    const char* group = "test_group";
    s32 devId = 0;

    MOCKER(HcomGetRankId).expects(atMost(1)).will(returnValue(HCCL_SUCCESS));
    MOCKER(HcomGetWorldRankFromGroupRank).expects(atMost(1)).will(returnValue(HCCL_SUCCESS));

    HcclResult ret = HcomGetDevId(group, &devId);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
}

TEST_F(HcomTest, ut_HcclCommGraphUnloadTask)
{
    hccl::hcclComm* comm = new hccl::hcclComm(1, 1, "123");
    s64 opBaseHcom = (s64)comm;
    std::string tag = "test_tag";
    MOCKER_CPP(&hcclComm::ClearOpResource).expects(atMost(1)).will(returnValue(HCCL_SUCCESS));
    HcclResult ret = HcclCommGraphUnloadTask(opBaseHcom, tag.c_str());
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
    delete comm;
}

TEST_F(HcomTest, ut_HcclCommGlobalWorkSpace)
{
    hccl::hcclComm* comm = new hccl::hcclComm(1, 1, "123");
    s64 opBaseHcom = (s64)comm;
    std::vector<void*> globalWorkSpaceAddr;
    MOCKER_CPP(&hcclComm::SetGlobalWorkSpace).expects(atMost(1)).will(returnValue(HCCL_SUCCESS));
    HcclResult ret = HcclCommSetGlobalWorkSpace(opBaseHcom, globalWorkSpaceAddr);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
    delete comm;
}

#if 1
TEST_F(HcomTest, ut_hcom_get_new_ranktable_info_serverId)
{
    nlohmann::json rank_table
        = {{"status", "completed"},
           {"version", "1.0"},
           {"server_count", "1"},
           {"server_list",
            {{
                {"server_id", "167772170"},
                {"host_nic_ip", "192.168.0.12:0,192.168.1.12:199"},
                {"device",
                 {
                     {{"rank_id", "2"}, {"device_id", "0"}, {"device_ip", "192.168.0.12,192.168.0.13"}

                     },
                     {{"rank_id", "1"}, {"device_id", "1"}, {"device_ip", "192.168.1.12,192.168.1.13"}

                     },
                     {{"rank_id", "0"}, {"device_id", "2"}, {"device_ip", "192.168.2.12,192.168.2.13"}

                     },
                 }},
            }}}};

    char file_name_t[] = "./ut_hcom_get_new_ranktable_info_serverId.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(1) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    } else {
        HCCL_ERROR("open %s failed", file_name_t);
    }

    outfile.close();
    int ret = HCCL_SUCCESS;

    ret = hrtSetDevice(2);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    string rank_table_file("./ut_hcom_get_new_ranktable_info_serverId.json");
    string rank_ID("0");
    std::string rankTableM;
    std::string realFilePath;

    HcomInfo hcom_info;
    ret = HcomLoadRanktableFile(rank_table_file.c_str(), rankTableM, realFilePath);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = CfgGetClusterInfo(rankTableM, rank_ID, hcom_info.params, hcom_info.rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = hrtSetDevice(0);
    remove(file_name_t);
}
#endif

#if 1
TEST_F(HcomTest, ut_hcom_get_new_ranktable_info_empty_serverId)
{
    nlohmann::json rank_table
        = {{"status", "completed"},
           {"version", "1.0"},
           {"server_count", "1"},
           {"server_list",
            {{
                {"server_id", ""},
                {"host_nic_ip", "192.168.0.12:0,192.168.1.12:199"},
                {"device",
                 {
                     {{"rank_id", "2"}, {"device_id", "0"}, {"device_ip", "192.168.0.12,192.168.0.13"}

                     },
                     {{"rank_id", "1"}, {"device_id", "1"}, {"device_ip", "192.168.1.12,192.168.1.13"}

                     },
                     {{"rank_id", "0"}, {"device_id", "2"}, {"device_ip", "192.168.2.12,192.168.2.13"}

                     },
                 }},
            }}}};

    char file_name_t[] = "./ut_hcom_get_new_ranktable_info_empty_serverId.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(1) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    } else {
        HCCL_ERROR("open %s failed", file_name_t);
    }

    outfile.close();
    int ret = HCCL_SUCCESS;

    ret = hrtSetDevice(2);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    string rank_table_file("./ut_hcom_get_new_ranktable_info_empty_serverId.json");
    string rank_ID("0");
    std::string rankTableM;
    std::string realFilePath;

    HcomInfo hcom_info;
    ret = HcomLoadRanktableFile(rank_table_file.c_str(), rankTableM, realFilePath);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = CfgGetClusterInfo(rankTableM, rank_ID, hcom_info.params, hcom_info.rankTable);
    EXPECT_EQ(ret, HCCL_E_PARA);
    ret = hrtSetDevice(0);
    remove(file_name_t);
}
#endif

#if 1
TEST_F(HcomTest, ut_hcom_get_new_ranktable_info_exception_serverId)
{
    nlohmann::json rank_table
        = {{"status", "completed"},
           {"version", "1.0"},
           {"server_count", "1"},
           {"server_list",
            {{
                {"server_id", "4294967296"},
                {"host_nic_ip", "192.168.0.12:0,192.168.1.12:199"},
                {"device",
                 {
                     {{"rank_id", "2"}, {"device_id", "0"}, {"device_ip", "192.168.0.12,192.168.0.13"}

                     },
                     {{"rank_id", "1"}, {"device_id", "1"}, {"device_ip", "192.168.1.12,192.168.1.13"}

                     },
                     {{"rank_id", "0"}, {"device_id", "2"}, {"device_ip", "192.168.2.12,192.168.2.13"}

                     },
                 }},
            }}}};

    char file_name_t[] = "./ut_hcom_get_new_ranktable_info_exception_serverId.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(1) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    } else {
        HCCL_ERROR("open %s failed", file_name_t);
    }

    outfile.close();
    int ret = HCCL_SUCCESS;

    ret = hrtSetDevice(2);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    string rank_table_file("./ut_hcom_get_new_ranktable_info_exception_serverId.json");
    string rank_ID("0");
    std::string rankTableM;
    std::string realFilePath;

    HcomInfo hcom_info;
    ret = HcomLoadRanktableFile(rank_table_file.c_str(), rankTableM, realFilePath);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = CfgGetClusterInfo(rankTableM, rank_ID, hcom_info.params, hcom_info.rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = hrtSetDevice(0);
    remove(file_name_t);
}
#endif

#if 1
TEST_F(HcomTest, ut_rank_select_err)
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
                     {{"rank_id", "0"}, {"device_id", "0"}, {"device_ip", "192.168.0.1"}

                     },
                     {{"rank_id", "1"}, {"device_id", "1"}, {"device_ip", "192.168.0.2"}

                     },
                     {{"rank_id", "2"}, {"device_id", "6"}, {"device_ip", "192.168.0.3"}

                     },
                     {{"rank_id", "3"}, {"device_id", "7"}, {"device_ip", "192.168.0.4"}

                     },
                 }},
            }}}};
    char file_name_t[] = "./ut_mpi_broadcast_4ranks_2server_ring_float_root10_4096_not_equa.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(4) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    } else {
        HCCL_ERROR("open %s failed", file_name_t);
    }

    outfile.close();

    int ret = HCCL_SUCCESS;
    string rank_ID("0");
    ret = HcomInitByFile(file_name_t, rank_ID.c_str());
    EXPECT_EQ(ret, HCCL_E_PARA);
    set_board_id(0);
    ret = HcomDestroy();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    remove(file_name_t);
}
#endif

#if 1
TEST_F(HcomTest, ut_rank_inner_server_4p_select_full)
{
    int ret = 0;
    TopoInfoParse test;
    test.deviceNum_ = 4;
    test.deviceNumPerServer_ = 4;
    test.serverNum_ = 1;
    test.serverId_ = "1";
    test.deviceType_ = DevType::DEV_TYPE_910;
    for (int i = 0; i < 8; ++i)
        for (int j = i + 1; j < 8; ++j)
            for (int k = j + 1; k < 8; ++k)
                for (int m = k + 1; m < 8; ++m) {
                    RankInfo tmp;
                    tmp.serverIdx = 0;
                    tmp.serverId = "1";
                    tmp.devicePhyId = i;
                    test.rankList_.push_back(tmp);
                    tmp.devicePhyId = j;
                    test.rankList_.push_back(tmp);
                    tmp.devicePhyId = k;
                    test.rankList_.push_back(tmp);
                    tmp.devicePhyId = m;
                    test.rankList_.push_back(tmp);
                    if (test.CheckServerInnerRankInfo() == 0)
                        ret++;
                    test.rankList_.clear();
                }
    EXPECT_EQ(ret, 8);
}
#endif

TEST_F(HcomTest, ut_hcom_reducescatter_cloud)
{
    nlohmann::json rank_table = {
        {"status", "completed"},
        {"group_count", "1"},
        {"group_list",
         {{
             {"group_name", ""},
             {"instance_count", "1"},
             {"device_count", "1"},
             {"instance_list",
              {
                  {{"pod_name", "tf-bae43"},
                   {"server_id", "10.0.0.10"},
                   {"devices", {{{"device_id", "0"}, {"device_ip", "192.168.0.12"}}}}},
              }},
         }}},
    };

    char file_name_t[] = "./ut_hcom_reducescatter_cloud.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(1) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    } else {
        HCCL_ERROR("open %s failed", file_name_t);
    }

    outfile.close();

    int ret = HCCL_SUCCESS;
    rtError_t rt_ret = RT_ERROR_NONE;
    rtStream_t stream;
    s8* sendbuf;
    s8* recvbuf;
    s32 rank = 0;
    s32 errors = 0;
    s32 count = HCCL_COM_DATA_SIZE;
    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    char* rank_table_file = "./ut_hcom_reducescatter_cloud.json";
    char* pod_name = "tf-bae43";

    ret = HcomInitByFile(rank_table_file, pod_name);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    rt_ret = aclrtCreateStream(&stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    sendbuf = (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(sendbuf, count * sizeof(s8), 0, count * sizeof(s8));
    recvbuf = (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(recvbuf, count * sizeof(s8), 0, count * sizeof(s8));

    for (int j = 0; j < count; j++) {
        sendbuf[j] = 2;
    }
    //-----------------Set Workspace Resource Start------------------//:
    rtModel_t model = (rtModel_t)NULL;
    rt_ret = rtModelCreate(&model, 0);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    u64 stream_list_size = 0;

    const u32 groupRanksNum = 1;
    char* strGroup1 = "group1";
    u32 groupRanks[1] = {0};
    ret = HcomGetWorkspaceSubStreamNum(strGroup1, stream_list_size);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(stream_list_size, 0);
    ret = HcomCreateGroup(strGroup1, groupRanksNum, (u32*)groupRanks);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = HcomGetWorkspaceSubStreamNum(strGroup1, stream_list_size);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcomGetWorkspaceSubStreamNum(HCCL_WORLD_GROUP, stream_list_size);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    HCCL_INFO("get stream_list_size[%d] success", stream_list_size);
    vector<HcclRtStream> streamList(stream_list_size);
    // 生成从stream
    for (s32 i = 0; i < stream_list_size; i++) {
        rt_ret = aclrtCreateStreamWithConfig(&streamList[i], 0, ACL_STREAM_PERSISTENT);
        EXPECT_EQ(rt_ret, RT_ERROR_NONE);
        // 从流bind到model
        rt_ret = rtModelBindStream(model, streamList[i], RT_MODEL_WAIT_ACTIVE_STREAM);
        EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    }

    u32 rankSize = 0;
    ret = HcomGetRankSize(HCCL_WORLD_GROUP, &rankSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    u64 memSize = 0;
    ret = HcomGetWorkspaceMemSize("HcomReduceScatter", count, HCCL_DATA_TYPE_INT8, HCCL_WORLD_GROUP, memSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    void* memptr = nullptr;
    ret = hrtMalloc(&memptr, memSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = HcomSetWorkspaceResource(
        "testreducescatter", HCCL_WORLD_GROUP, streamList.data(), streamList.size(), memptr, memSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = HcomRpcSetWorkspaceResource(
        "testreducescatter", HCCL_WORLD_GROUP, streamList.data(), streamList.size(), memptr, memSize);
    //-----------------Set Workspace Resource End------------------//
    ret = HcomReduceScatter(
        "testreducescatter", sendbuf, recvbuf, count, HCCL_DATA_TYPE_INT8, HCCL_REDUCE_SUM, HCCL_WORLD_GROUP, stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    rt_ret = aclrtSynchronizeStream(stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    for (int j = 0; j < count; j++) {
        if (recvbuf[j] != 2) {
            HCCL_ERROR("ERR recvbuf[%d] = [%d] ", j, recvbuf[j]);
            errors++;
            break;
        }
    }

    sal_free(sendbuf);
    sal_free(recvbuf);
    rt_ret = aclrtDestroyStream(stream);

    ret = HcomDestroy();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    for (s32 i = 0; i < stream_list_size; i++) {
        rt_ret = rtModelUnbindStream(model, streamList[i]);
        EXPECT_EQ(rt_ret, RT_ERROR_NONE);

        rt_ret = aclrtDestroyStream(streamList[i]);
        EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    }
    remove(file_name_t);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    EXPECT_EQ(errors, 0);
}

HcclResult Stub_GetAlgType_DEFAULT(hcclComm* comm, AlgType& algType)
{
    HCCL_INFO("==TMP== point1");
    algType = AlgType();
    return HCCL_SUCCESS;
}

HcclResult Stub_GetAlgType_Reserved(hcclComm* comm, AlgType& algType)
{
    algType = AlgType::Reserved();
    return HCCL_SUCCESS;
}

HcclResult Stub_GetAlgType_mesh_plus_ring(hcclComm* comm, AlgType& algType)
{
    algType.algoLevel0 = AlgTypeLevel0::ALG_LEVEL0_4P_MESH;
    algType.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_RING;
    return HCCL_SUCCESS;
}

HcclResult Stub_GetAlgType_Reserved_plus_NHR_V1(hcclComm* comm, AlgType& algType)
{
    algType.algoLevel0 = AlgTypeLevel0::ALG_LEVEL0_RESERVED;
    algType.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_NHR_V1;
    return HCCL_SUCCESS;
}

HcclResult Stub_GetAlgType_pipeline(HcclCommunicator* comm, AlgType& algType, HcclCMDType opType)
{
    algType.algoLevel0 = AlgTypeLevel0::ALG_LEVEL0_NP_MESH;
    algType.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_PIPELINE;
    return HCCL_SUCCESS;
}

HcclResult
Stub_GetAlgType_ALG_ALLGATHER_REDUCESCATTER_GRAPH_PIPELINE(HcclCommunicator* comm, AlgType& algType, HcclCMDType opType)
{
    algType.algoLevel0 = AlgTypeLevel0::ALG_LEVEL0_NP_MESH;

    return HCCL_SUCCESS;
}

TEST_F(HcomTest, ut_HcomGetAlgorithm)
{
    nlohmann::json rank_table = {
        {"status", "completed"},
        {"chip_info", "910"},
        {"group_count", "1"},
        {"group_list",
         {{
             {"group_name", ""},
             {"instance_count", "1"},
             {"device_count", "1"},
             {"instance_list",
              {
                  {{"pod_name", "tf-bae43"},
                   {"server_id", "10.0.0.10"},
                   {"devices", {{{"device_id", "0"}, {"device_ip", "192.168.0.12"}}}}},
              }},
         }}},
    };

    char file_name_t[] = "./ut_HcomGetAlgorithm.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(1) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    } else {
        HCCL_ERROR("open %s failed", file_name_t);
    }

    outfile.close();

    int ret = HCCL_SUCCESS;
    rtError_t rt_ret = RT_ERROR_NONE;

    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    char* rank_table_file = "./ut_HcomGetAlgorithm.json";
    char* pod_name = "tf-bae43";

    ret = HcomInitByFile(rank_table_file, pod_name);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    u32 level = 1;
    std::string algo;
    MOCKER_CPP(&hcclComm::GetAlgType).stubs().will(invoke(Stub_GetAlgType_DEFAULT));
    ret = HcomGetAlgorithm(level, algo);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();

    level = 0;
    MOCKER_CPP(&hcclComm::GetAlgType).stubs().will(invoke(Stub_GetAlgType_mesh_plus_ring));
    ret = HcomGetAlgorithm(level, algo);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();

    level = 1;
    MOCKER_CPP(&hcclComm::GetAlgType).stubs().will(invoke(Stub_GetAlgType_mesh_plus_ring));
    ret = HcomGetAlgorithm(level, algo);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();

    level = 0;
    MOCKER_CPP(&hcclComm::GetAlgType).stubs().will(invoke(Stub_GetAlgType_Reserved_plus_NHR_V1));
    ret = HcomGetAlgorithm(level, algo);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();

    level = 0;
    MOCKER_CPP(&hcclComm::GetAlgType).stubs().will(invoke(Stub_GetAlgType_Reserved));
    ret = HcomGetAlgorithm(level, algo);
    EXPECT_EQ(ret, HCCL_E_NOT_FOUND);
    GlobalMockObject::verify();

    level = 1;
    MOCKER_CPP(&hcclComm::GetAlgType).stubs().will(invoke(Stub_GetAlgType_Reserved));
    ret = HcomGetAlgorithm(level, algo);
    EXPECT_EQ(ret, HCCL_E_NOT_FOUND);
    GlobalMockObject::verify();

    ret = HcomDestroy();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    remove(file_name_t);
}

#if 1
TEST_F(HcomTest, ut_check_and_assign_nic_info)
{
    int ret = 0;
    TopoInfoParse test;
    std::vector<u32> nicIdx;

    test.serverNum_ = 2;
    test.deviceNum_ = 16;
    test.deviceNumPerServer_ = 8;
    nicIdx.push_back(0);
    nicIdx.push_back(1);
    for (u32 i = 0; i < 16; i++) {
        RankInfo tmp;
        tmp.devicePhyId = i % 8;
        test.rankList_.push_back(tmp);
    }
    for (u32 i = 0; i < 8; i++) {
        test.rankList_[i].serverIdx = 0;
        test.rankList_[i].serverId = "1";
        test.rankList_[i].nicIp.push_back(HcclIpAddress(0));
    }
    for (u32 i = 8; i < 16; i++) {
        test.rankList_[i].serverIdx = 1;
        test.rankList_[i].serverId = "2";
        test.rankList_[i].nicIp.push_back(HcclIpAddress(0));
    }

    test.rankList_[0].nicIp[0] = HcclIpAddress(3232261989);
    ret = test.CheckAndAssignNicInfo(nicIdx);
    EXPECT_EQ(ret, HCCL_E_PARA);

    test.rankList_[1].nicIp[0] = HcclIpAddress(3232261989);
    test.rankList_[8].nicIp[0] = HcclIpAddress(3232261989);
    test.rankList_[9].nicIp[0] = HcclIpAddress(3232261989);

    ret = test.CheckAndAssignNicInfo(nicIdx);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    test.rankList_.clear();
    nicIdx.clear();
}
#endif

void* hcom_get_cur_hcom_ctx(void* parg)
{
    const char* group = "test_group";
    s32 devId = 0;
    HcclResult ret = HCCL_SUCCESS;

    ret = hrtSetDevice(MAX_MODULE_DEVICE_NUM);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    MOCKER(HcomGetRankId).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(HcomGetWorldRankFromGroupRank).stubs().will(returnValue(HCCL_SUCCESS));

    ret = HcomGetDevId(group, &devId);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    return (nullptr);
}

void* hcom_get_cur_hcom_ctx_second(void* parg)
{
    const char* group = "test_group";
    s32 devId = 0;
    HcclResult ret = HCCL_SUCCESS;

    ret = hrtSetDevice(15);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    MOCKER(HcomGetRankId).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(HcomGetWorldRankFromGroupRank).stubs().will(returnValue(HCCL_SUCCESS));

    ret = HcomGetDevId(group, &devId);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    return (nullptr);
}

TEST_F(HcomTest, ut_HcomGetCurHcomCtx)
{
    sal_thread_t tid;

    tid = sal_thread_create("thread", hcom_get_cur_hcom_ctx, (void*)nullptr);
    EXPECT_NE(tid, (sal_thread_t) nullptr);

    while (sal_thread_is_running(tid)) {
        SaluSleep(SAL_MILLISECOND_USEC * 10);
    }

    tid = sal_thread_create("thread", hcom_get_cur_hcom_ctx_second, (void*)nullptr);
    EXPECT_NE(tid, (sal_thread_t) nullptr);

    while (sal_thread_is_running(tid)) {
        SaluSleep(SAL_MILLISECOND_USEC * 10);
    }

    GlobalMockObject::verify();
}

TEST_F(HcomTest, ut_HcomGetWorkspaceMemSize_exception)
{
    u64 memSize = 0;
    HcclResult ret = HcomGetWorkspaceMemSize("HcomReduceScatter", 1, HCCL_DATA_TYPE_INT8, "test_group", memSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(HcomTest, ut_hcom_HcclCommGraphAlltoAllVC)
{
    s32 deviceId = 0;
    char* identify = "0";
    s32 rankSize = 1;
    s32 rank = atoi(identify);
    u64 count = 2;
    HCCL_INFO("alltoall_int32 : identify[%d], count[%llu]", rank, count);
    u32 devLogicId = 0xFFFF;
    HcclResult ret = hrtGetDeviceIndexByPhyId(deviceId, devLogicId);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = hrtSetDevice(devLogicId);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    const int COUNT_PER_RANK = count;
    u64 memSize = COUNT_PER_RANK * rankSize * sizeof(s32);
    HostMem hostSendMem = HostMem::alloc(memSize);
    memset_s(hostSendMem.ptr(), memSize, 0, COUNT_PER_RANK * rankSize);
    for (u32 i = 0; i < COUNT_PER_RANK * rankSize; i++) {
        *((s32*)hostSendMem.ptr() + i) = rank + 1;
    }

    // �������
    vector<u64> sendCounts(rankSize, COUNT_PER_RANK);
    vector<u64> recvCounts(rankSize, COUNT_PER_RANK);
    vector<u64> sdispls(rankSize, 0);
    vector<u64> rdispls(rankSize, 0);
    for (int i = 0; i < rankSize; i++) {
        sdispls[i] = COUNT_PER_RANK * i;
        rdispls[i] = COUNT_PER_RANK * i;
        HCCL_INFO("num[%d] displs[%d]", i, COUNT_PER_RANK * i);
    }

    DeviceMem sendMem = DeviceMem::alloc(memSize);
    ret = hrtMemSyncCopy(
        sendMem.ptr(), memSize, hostSendMem.ptr(), memSize, HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    DeviceMem recvMem = DeviceMem::alloc(memSize);

    hccl::Stream stream(StreamType::STREAM_TYPE_OFFLINE);

    hccl::hcclComm* comm = new hccl::hcclComm(1, 1, "123");
    s64 opBaseHcom = (s64)comm;

    MOCKER_CPP(&hcclComm::AlltoAllVC).expects(atMost(1)).will(returnValue(0));

    MOCKER_CPP(&hcclComm::GetRankTableCrc).stubs().will(returnValue(0));

    MOCKER_CPP(&hcclComm::GetRankSize).expects(atMost(1)).will(returnValue(0));

    MOCKER_CPP(&hcclComm::GetGroupRank).expects(atMost(1)).will(returnValue(0));

    MOCKER_CPP(&hcclComm::GetAlgType).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&hcclComm::GetNumBlocks).stubs().will(returnValue(HCCL_SUCCESS));

    ret = HcclCommGraphAlltoAllVC(
        sendMem.ptr(), sendMem.ptr(), HCCL_DATA_TYPE_INT32, recvMem.ptr(), HCCL_DATA_TYPE_INT32, opBaseHcom,
        stream.ptr(), "hcom_alltoallvc");
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = hcclStreamSynchronize(stream.ptr());

    EXPECT_EQ(ret, HCCL_SUCCESS);
    (void)aclrtResetDevice(0);
    delete comm;
}

#if 1
TEST_F(HcomTest, ut_hcom_get_hcom_info_eth0Err)
{
    nlohmann::json rank_table
        = {{"status", "completed"},
           {"deploy_mode", "lab"},
           {"group_count", "1"},
           {"chip_info", "910"},
           {"board_id", "0x0000"},
           {"para_plane_nic_location", "device"},
           {"para_plane_nic_num", "8"},
           {"para_plane_nic_name", {"eth0", "eth1", "eth2", "eth3", "eth4", "eth5", "eth6", "eth7"}},
           {"group_list",
            {{{"group_name", ""},
              {"device_num", "16"},
              {"server_num", "2"},
              {"instance_count", "16"},
              {"instance_list",
               {
                   {{"rank_id", "0"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "0"}, {"device_ip", "192.168.0.11"}}}}},

                   {{"rank_id", "1"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "1"}, {"device_ip", "192.168.0.12"}}}}},
                   {{"rank_id", "2"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "2"}, {"device_ip", "192.168.0.13"}}}}},

                   {{"rank_id", "3"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "3"}, {"device_ip", "192.168.0.14"}}}}},
                   {{"rank_id", "4"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "4"}, {"device_ip", "192.168.0.15"}}}}},

                   {{"rank_id", "5"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "5"}, {"device_ip", "192.168.0.16"}}}}},
                   {{"rank_id", "6"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "6"}, {"device_ip", "192.168.0.17"}}}}},

                   {{"rank_id", "7"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "7"}, {"device_ip", "192.168.0.18"}}}}},
                   {{"rank_id", "8"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "0"}, {"device_ip", "192.168.0.20"}}}}},

                   {{"rank_id", "9"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "1"}, {"device_ip", "192.168.0.21"}}}}},
                   {{"rank_id", "10"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "2"}, {"device_ip", "192.168.0.22"}}}}},

                   {{"rank_id", "11"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "3"}, {"device_ip", "192.168.0.23"}}}}},
                   {{"rank_id", "12"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "4"}, {"device_ip", "192.168.0.24"}}}}},

                   {{"rank_id", "13"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "5"}, {"device_ip", "192.168.0.25"}}}}},
                   {{"rank_id", "14"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "6"}, {"device_ip", "192.168.0.26"}}}}},

                   {{"rank_id", "15"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "7"}, {"device_ip", "192.168.0.27"}}}}},
               }},
              {"server_list",
               {
                   {{"server_id", "10.0.0.10"},
                    {"para_plane_info",
                     {
                         {
                             {"eth0", "192.168.200.2"},
                         },
                         {
                             {"eth0", "192.168.201.2"},
                         },
                         {
                             {"eth2", "192.168.202.2"},
                         },
                         {
                             {"eth3", "192.168.203.2"},
                         },
                         {
                             {"eth4", "192.168.204.2"},
                         },
                         {
                             {"eth5", "192.168.205.2"},
                         },
                         {
                             {"eth6", "192.168.206.2"},
                         },
                         {
                             {"eth7", "192.168.207.2"},
                         },
                     }}

                   },
                   {{"server_id", "10.0.0.11"},
                    {"para_plane_info",
                     {
                         {
                             {"eth0", "192.168.210.3"},
                         },
                         {
                             {"eth1", "192.168.211.3"},
                         },
                         {
                             {"eth2", "192.168.212.3"},
                         },
                         {
                             {"eth3", "192.168.213.3"},
                         },
                         {
                             {"eth4", "192.168.214.3"},
                         },
                         {
                             {"eth5", "192.168.215.3"},
                         },
                         {
                             {"eth6", "192.168.216.3"},
                         },
                         {
                             {"eth7", "192.168.217.3"},
                         },
                     }}

                   },

               }}}}}};

    char file_name[] = "./ut_hcom_get_hcom_info_eth0Err.json";
    std::ofstream outfile(file_name, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(4) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name);
    } else {
        HCCL_ERROR("open %s failed", file_name);
    }

    outfile.close();
    s32 ret = HCCL_SUCCESS;
    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    string rank_table_file("./ut_hcom_get_hcom_info_eth0Err.json");
    string rank_ID("0");

    HcomInfo hcom_info;
    std::string rankTableM;
    std::string realFilePath;

    ret = HcomLoadRanktableFile(rank_table_file.c_str(), rankTableM, realFilePath);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    TopoInfoRanktableParser myTopoRanktable(rankTableM, rank_ID);
    TopoinfoRanktableStandard myTopoinfoRanktableStandard(rankTableM, rank_ID);
    set_board_id(0x0000);
    s32 rankId = -1;
    HcomInfo hcom;
    std::string identify = "0";
    ret = myTopoRanktable.LoadFile(file_name);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = myTopoRanktable.RefreshStatus();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    bool status_completed = myTopoRanktable.IsReady();
    printf("status ready:%d \n", status_completed);

    ret = myTopoinfoRanktableStandard.GetDeployMode(hcom.cloudFlag);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = myTopoinfoRanktableStandard.GetHcomInfo(hcom.params, hcom.rankTable);

    EXPECT_EQ(ret, HCCL_E_NOT_FOUND);
    set_board_id(0);
    remove(file_name);
}

TEST_F(HcomTest, ut_hcom_get_hcom_info_eth0Err1)
{
    nlohmann::json rank_table
        = {{"status", "completed"},
           {"deploy_mode", "lab"},
           {"group_count", "1"},
           {"chip_info", "910"},
           {"board_id", "0x0000"},
           {"para_plane_nic_location", "device"},
           {"para_plane_nic_num", "8"},
           {"para_plane_nic_name", {"eth0", "eth0", "eth2", "eth3", "eth4", "eth5", "eth6", "eth7"}},
           {"group_list",
            {{{"group_name", ""},
              {"device_num", "16"},
              {"server_num", "2"},
              {"instance_count", "16"},
              {"instance_list",
               {
                   {{"rank_id", "0"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "0"}, {"device_ip", "192.168.0.11"}}}}},

                   {{"rank_id", "1"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "1"}, {"device_ip", "192.168.0.12"}}}}},
                   {{"rank_id", "2"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "2"}, {"device_ip", "192.168.0.13"}}}}},

                   {{"rank_id", "3"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "3"}, {"device_ip", "192.168.0.14"}}}}},
                   {{"rank_id", "4"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "4"}, {"device_ip", "192.168.0.15"}}}}},

                   {{"rank_id", "5"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "5"}, {"device_ip", "192.168.0.16"}}}}},
                   {{"rank_id", "6"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "6"}, {"device_ip", "192.168.0.17"}}}}},

                   {{"rank_id", "7"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "7"}, {"device_ip", "192.168.0.18"}}}}},
                   {{"rank_id", "8"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "0"}, {"device_ip", "192.168.0.20"}}}}},

                   {{"rank_id", "9"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "1"}, {"device_ip", "192.168.0.21"}}}}},
                   {{"rank_id", "10"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "2"}, {"device_ip", "192.168.0.22"}}}}},

                   {{"rank_id", "11"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "3"}, {"device_ip", "192.168.0.23"}}}}},
                   {{"rank_id", "12"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "4"}, {"device_ip", "192.168.0.24"}}}}},

                   {{"rank_id", "13"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "5"}, {"device_ip", "192.168.0.25"}}}}},
                   {{"rank_id", "14"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "6"}, {"device_ip", "192.168.0.26"}}}}},

                   {{"rank_id", "15"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "7"}, {"device_ip", "192.168.0.27"}}}}},
               }},
              {"server_list",
               {
                   {{"server_id", "10.0.0.10"},
                    {"para_plane_info",
                     {
                         {
                             {"eth0", "192.168.200.2"},
                         },
                         {
                             {"eth1", "192.168.201.2"},
                         },
                         {
                             {"eth2", "192.168.202.2"},
                         },
                         {
                             {"eth3", "192.168.203.2"},
                         },
                         {
                             {"eth4", "192.168.204.2"},
                         },
                         {
                             {"eth5", "192.168.205.2"},
                         },
                         {
                             {"eth6", "192.168.206.2"},
                         },
                         {
                             {"eth7", "192.168.207.2"},
                         },
                     }}

                   },
                   {{"server_id", "10.0.0.11"},
                    {"para_plane_info",
                     {
                         {
                             {"eth0", "192.168.210.3"},
                         },
                         {
                             {"eth1", "192.168.211.3"},
                         },
                         {
                             {"eth2", "192.168.212.3"},
                         },
                         {
                             {"eth3", "192.168.213.3"},
                         },
                         {
                             {"eth4", "192.168.214.3"},
                         },
                         {
                             {"eth5", "192.168.215.3"},
                         },
                         {
                             {"eth6", "192.168.216.3"},
                         },
                         {
                             {"eth7", "192.168.217.3"},
                         },
                     }}

                   },

               }}}}}};

    char file_name[] = "./ut_hcom_get_hcom_info_eth0Err1.json";
    std::ofstream outfile(file_name, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(4) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name);
    } else {
        HCCL_ERROR("open %s failed", file_name);
    }

    outfile.close();
    s32 ret = HCCL_SUCCESS;
    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    string rank_table_file("./ut_hcom_get_hcom_info_eth0Err1.json");
    string rank_ID("0");

    HcomInfo hcom_info;
    std::string rankTableM;
    std::string realFilePath;

    ret = HcomLoadRanktableFile(rank_table_file.c_str(), rankTableM, realFilePath);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    s32 rankId = -1;

    set_board_id(0x0000);
    TopoInfoRanktableParser myTopoRanktable(rankTableM, rank_ID);
    TopoinfoRanktableStandard myTopoinfoRanktableStandard(rankTableM, rank_ID);
    HcomInfo hcom;
    std::string identify = "0";
    ret = myTopoRanktable.LoadFile(file_name);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = myTopoRanktable.RefreshStatus();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    bool status_completed = myTopoRanktable.IsReady();
    printf("status ready:%d \n", status_completed);

    ret = myTopoinfoRanktableStandard.GetDeployMode(hcom.cloudFlag);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = myTopoinfoRanktableStandard.GetHcomInfo(hcom.params, hcom.rankTable);

    set_board_id(0);
    EXPECT_EQ(ret, HCCL_E_NOT_FOUND);
    remove(file_name);
}

TEST_F(HcomTest, ut_hcom_get_hcom_info_eth0IPErr)
{
    nlohmann::json rank_table
        = {{"status", "completed"},
           {"deploy_mode", "lab"},
           {"group_count", "1"},
           {"chip_info", "910"},
           {"board_id", "0x0000"},
           {"para_plane_nic_location", "device"},
           {"para_plane_nic_num", "8"},
           {"para_plane_nic_name", {"eth0", "eth1", "eth2", "eth3", "eth4", "eth5", "eth6", "eth7"}},
           {"group_list",
            {{{"group_name", ""},
              {"device_num", "16"},
              {"server_num", "2"},
              {"instance_count", "16"},
              {"instance_list",
               {
                   {{"rank_id", "0"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "0"}, {"device_ip", "192.168.0.11"}}}}},

                   {{"rank_id", "1"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "1"}, {"device_ip", "192.168.0.12"}}}}},
                   {{"rank_id", "2"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "2"}, {"device_ip", "192.168.0.13"}}}}},

                   {{"rank_id", "3"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "3"}, {"device_ip", "192.168.0.14"}}}}},
                   {{"rank_id", "4"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "4"}, {"device_ip", "192.168.0.15"}}}}},

                   {{"rank_id", "5"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "5"}, {"device_ip", "192.168.0.16"}}}}},
                   {{"rank_id", "6"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "6"}, {"device_ip", "192.168.0.17"}}}}},

                   {{"rank_id", "7"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "7"}, {"device_ip", "192.168.0.18"}}}}},
                   {{"rank_id", "8"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "0"}, {"device_ip", "192.168.0.20"}}}}},

                   {{"rank_id", "9"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "1"}, {"device_ip", "192.168.0.21"}}}}},
                   {{"rank_id", "10"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "2"}, {"device_ip", "192.168.0.22"}}}}},

                   {{"rank_id", "11"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "3"}, {"device_ip", "192.168.0.23"}}}}},
                   {{"rank_id", "12"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "4"}, {"device_ip", "192.168.0.24"}}}}},

                   {{"rank_id", "13"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "5"}, {"device_ip", "192.168.0.25"}}}}},
                   {{"rank_id", "14"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "6"}, {"device_ip", "192.168.0.26"}}}}},

                   {{"rank_id", "15"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "7"}, {"device_ip", "192.168.0.27"}}}}},
               }},
              {"server_list",
               {
                   {{"server_id", "10.0.0.10"},
                    {"para_plane_info",
                     {
                         {
                             {"eth0", "192.168.200.2"},
                         },
                         {
                             {"eth1", "192.168.200.2"},
                         },
                         {
                             {"eth2", "192.168.202.2"},
                         },
                         {
                             {"eth3", "192.168.203.2"},
                         },
                         {
                             {"eth4", "192.168.204.2"},
                         },
                         {
                             {"eth5", "192.168.205.2"},
                         },
                         {
                             {"eth6", "192.168.206.2"},
                         },
                         {
                             {"eth7", "192.168.207.2"},
                         },
                     }}

                   },
                   {{"server_id", "10.0.0.11"},
                    {"para_plane_info",
                     {
                         {
                             {"eth0", "192.168.210.3"},
                         },
                         {
                             {"eth1", "192.168.211.3"},
                         },
                         {
                             {"eth2", "192.168.212.3"},
                         },
                         {
                             {"eth3", "192.168.213.3"},
                         },
                         {
                             {"eth4", "192.168.214.3"},
                         },
                         {
                             {"eth5", "192.168.215.3"},
                         },
                         {
                             {"eth6", "192.168.216.3"},
                         },
                         {
                             {"eth7", "192.168.217.3"},
                         },
                     }}

                   },

               }}}}}};

    char file_name[] = "./ut_hcom_get_hcom_info_eth0IPErr.json";
    std::ofstream outfile(file_name, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(4) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name);
    } else {
        HCCL_ERROR("open %s failed", file_name);
    }

    outfile.close();
    s32 ret = HCCL_SUCCESS;
    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    string rank_table_file("./ut_hcom_get_hcom_info_eth0IPErr.json");
    string rank_ID("0");

    HcomInfo hcom_info;
    std::string rankTableM;
    std::string realFilePath;

    ret = HcomLoadRanktableFile(rank_table_file.c_str(), rankTableM, realFilePath);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    TopoInfoRanktableParser myTopoRanktable(rankTableM, rank_ID);
    TopoinfoRanktableStandard myTopoinfoRanktableStandard(rankTableM, rank_ID);
    s32 rankId = -1;
    HcomInfo hcom;
    set_board_id(0x0000);
    std::string identify = "0";
    ret = myTopoRanktable.LoadFile(file_name);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = myTopoRanktable.RefreshStatus();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    bool status_completed = myTopoRanktable.IsReady();
    printf("status ready:%d \n", status_completed);

    ret = myTopoinfoRanktableStandard.GetDeployMode(hcom.cloudFlag);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = myTopoinfoRanktableStandard.GetHcomInfo(hcom.params, hcom.rankTable);

    EXPECT_EQ(ret, HCCL_E_NOT_FOUND);
    set_board_id(0);
    remove(file_name);
}
#endif

#if 1
TEST_F(HcomTest, ut_hcom_CheckPortValid)
{
    HcclResult ret = HCCL_SUCCESS;
    u32 port = 18000;
    HcomInfo hcom_info;
    ret = CheckPortValid(port);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = CheckPortValid(65536);
    EXPECT_EQ(ret, HCCL_E_PARA);
    HcclIpAddress ipAddr1(1694542016);
    std::vector<RoleTableNodeInfo> clientsInfoCtx;
    RoleTableNodeInfo NodeInfo;
    NodeInfo.id = 2;
    NodeInfo.serverId = "10.78.132.102";
    NodeInfo.ipAddr = ipAddr1;
    NodeInfo.port = 18000;
    NodeInfo.rankId = 1;
    NodeInfo.hostIp = ipAddr1;
    NodeInfo.hostPort = 18000;
    NodeInfo.devicePhyId = 10;
    clientsInfoCtx.push_back(NodeInfo);
    clientsInfoCtx.push_back(NodeInfo);
    RoleTableInfo roleTableInfo;
    roleTableInfo.servers = clientsInfoCtx;
    roleTableInfo.clients = clientsInfoCtx;
    ret = CheckRoleAndRankConsistent(roleTableInfo, hcom_info.rankTable);
    EXPECT_EQ(ret, HCCL_E_PARA);
    hcom_info.rankTable.rankNum = 4;
    ret = CheckRoleAndRankConsistent(roleTableInfo, hcom_info.rankTable);
    EXPECT_EQ(ret, HCCL_E_PARA);
    std::string rankTableM;
    ret = CfgGetRoleTableInfo(rankTableM, roleTableInfo);
    EXPECT_EQ(ret, HCCL_E_PARA);
}
#endif

TEST_F(HcomTest, ut_hcom_TopoInfoRanktableParser)
{
    nlohmann::json rank_table
        = {{"status", "completed"},
           {"deploy_mode", "lab"},
           {"group_count", "1"},
           {"chip_info", "910"},
           {"board_id", "0x0000"},
           {"para_plane_nic_location", "device"},
           {"para_plane_nic_num", "8"},
           {"para_plane_nic_name", {"eth0", "eth1", "eth2", "eth3", "eth4", "eth5", "eth6", "eth7"}},
           {"group_list",
            {{{"group_name", ""},
              {"device_num", "16"},
              {"server_num", "2"},
              {"instance_count", "16"},
              {"instance_list",
               {
                   {{"rank_id", "0"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "0"}, {"device_ip", "192.168.0.11"}}}}},

                   {{"rank_id", "1"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "1"}, {"device_ip", "192.168.0.12"}}}}},
                   {{"rank_id", "2"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "2"}, {"device_ip", "192.168.0.13"}}}}},

                   {{"rank_id", "3"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "3"}, {"device_ip", "192.168.0.14"}}}}},
                   {{"rank_id", "4"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "4"}, {"device_ip", "192.168.0.15"}}}}},

                   {{"rank_id", "5"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "5"}, {"device_ip", "192.168.0.16"}}}}},
                   {{"rank_id", "6"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "6"}, {"device_ip", "192.168.0.17"}}}}},

                   {{"rank_id", "7"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "7"}, {"device_ip", "192.168.0.18"}}}}},
                   {{"rank_id", "8"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "0"}, {"device_ip", "192.168.0.20"}}}}},

                   {{"rank_id", "9"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "1"}, {"device_ip", "192.168.0.21"}}}}},
                   {{"rank_id", "10"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "2"}, {"device_ip", "192.168.0.22"}}}}},

                   {{"rank_id", "11"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "3"}, {"device_ip", "192.168.0.23"}}}}},
                   {{"rank_id", "12"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "4"}, {"device_ip", "192.168.0.24"}}}}},

                   {{"rank_id", "13"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "5"}, {"device_ip", "192.168.0.25"}}}}},
                   {{"rank_id", "14"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "6"}, {"device_ip", "192.168.0.26"}}}}},

                   {{"rank_id", "15"},
                    {"server_id", "10.0.0.11"},
                    {"devices", {{{"device_id", "7"}, {"device_ip", "192.168.0.27"}}}}},
               }},
              {"server_list",
               {
                   {{"server_id", "10.0.0.10"},
                    {"para_plane_info",
                     {
                         {
                             {"eth0", "192.168.200.2"},
                         },
                         {
                             {"eth1", "192.168.200.2"},
                         },
                         {
                             {"eth2", "192.168.202.2"},
                         },
                         {
                             {"eth3", "192.168.203.2"},
                         },
                         {
                             {"eth4", "192.168.204.2"},
                         },
                         {
                             {"eth5", "192.168.205.2"},
                         },
                         {
                             {"eth6", "192.168.206.2"},
                         },
                         {
                             {"eth7", "192.168.207.2"},
                         },
                     }}

                   },
                   {{"server_id", "10.0.0.11"},
                    {"para_plane_info",
                     {
                         {
                             {"eth0", "192.168.210.3"},
                         },
                         {
                             {"eth1", "192.168.211.3"},
                         },
                         {
                             {"eth2", "192.168.212.3"},
                         },
                         {
                             {"eth3", "192.168.213.3"},
                         },
                         {
                             {"eth4", "192.168.214.3"},
                         },
                         {
                             {"eth5", "192.168.215.3"},
                         },
                         {
                             {"eth6", "192.168.216.3"},
                         },
                         {
                             {"eth7", "192.168.217.3"},
                         },
                     }}

                   },

               }}}}}};

    char file_name[] = "./ut_hcom_TopoInfoRanktableParser.json";
    std::ofstream outfile(file_name, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(4) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name);
    } else {
        HCCL_ERROR("open %s failed", file_name);
    }

    outfile.close();
    s32 ret = HCCL_SUCCESS;
    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    string rank_table_file("./ut_hcom_TopoInfoRanktableParser.json");
    string rank_ID("0");

    HcomInfo hcom_info;
    std::string rankTableM;
    std::string realFilePath;

    ret = HcomLoadRanktableFile(rank_table_file.c_str(), rankTableM, realFilePath);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    TopoInfoRanktableParser myTopoRanktable(rankTableM, rank_ID);
    TopoinfoRanktableStandard myTopoinfoRanktableStandard(rankTableM, rank_ID);
    s32 rankId = -1;
    HcomInfo hcom;
    set_board_id(0x0000);
    std::string identify = "0";
    ret = myTopoRanktable.LoadFile(file_name);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = myTopoRanktable.RefreshStatus();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    u32 index = 1;
    ret = myTopoRanktable.GetJsonArrayMemberProperty(rank_table, index, rank_table_file.c_str(), index);
    EXPECT_EQ(ret, HCCL_E_PARA);

    index = 0;
    ret = myTopoRanktable.GetJsonArrayMemberProperty(rank_table, index, rank_table_file.c_str(), index);
    EXPECT_EQ(ret, HCCL_E_PARA);
    ret = myTopoRanktable.LoadConfigString("10.78.232.102");
    EXPECT_EQ(ret, HCCL_E_PARA);
    bool status_completed = myTopoRanktable.IsReady();
    printf("status ready:%d \n", status_completed);

    ret = myTopoinfoRanktableStandard.GetDeployMode(hcom.cloudFlag);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    index = 1;
    ret = myTopoinfoRanktableStandard.GetServerList(rank_table, index, hcom.rankTable, index);
    EXPECT_EQ(ret, HCCL_E_PARA);
    ret = myTopoinfoRanktableStandard.GetSingleServer(rank_table, index, hcom.rankTable);
    EXPECT_EQ(ret, HCCL_E_PARA);
    ret = myTopoinfoRanktableStandard.GetHcomInfo(hcom.params, hcom.rankTable);
    EXPECT_EQ(ret, HCCL_E_NOT_FOUND);

    TopoinfoRoletable myTopoinfoRoletable(rankTableM);
    std::vector<RoleTableNodeInfo> clientsInfoCtx;
    RoleTableNodeInfo NodeInfo;

    HcclIpAddress ipAddr1(1694542016);
    NodeInfo.id = 2;
    NodeInfo.serverId = "10.78.132.102";
    NodeInfo.ipAddr = ipAddr1;
    NodeInfo.port = 10;
    NodeInfo.rankId = 1;
    NodeInfo.hostIp = ipAddr1;
    NodeInfo.hostPort = 10;
    NodeInfo.devicePhyId = 10;
    clientsInfoCtx.push_back(NodeInfo);
    clientsInfoCtx.push_back(NodeInfo);
    ret = myTopoinfoRoletable.GetSingleNode(rank_table, index, clientsInfoCtx);
    EXPECT_EQ(ret, HCCL_E_PARA);
    ret = myTopoinfoRoletable.GetServersInfo(clientsInfoCtx);
    EXPECT_EQ(ret, HCCL_E_NOT_FOUND);
    ret = myTopoinfoRoletable.GetClientsInfo(clientsInfoCtx);
    EXPECT_EQ(ret, HCCL_E_NOT_FOUND);
    RoleTableInfo roleTableInfo;
    roleTableInfo.servers = clientsInfoCtx;
    roleTableInfo.clients = clientsInfoCtx;

    MOCKER_CPP(&TopoInfoRanktableParser::LoadConfigString).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    ret = myTopoinfoRoletable.ParserRoleTable(roleTableInfo);
    EXPECT_EQ(ret, HCCL_E_NOT_FOUND);
    set_board_id(0);
    remove(file_name);
    GlobalMockObject::verify();
}

#if 1
nlohmann::json ranktable_invalid_superPodId
    = {{"status", "completed"},
       {"version", "1.2"},
       {"server_list",
        {{
            {"server_id", "10.155.111.140"},
            {"device",
             {
                 {{"rank_id", "0"}, {"device_id", "0"}, {"device_ip", "192.1.27.6"}},
             }},
        }}},
       {"super_pod_list",
        {{
            {"server_list",
             {
                 {{"server_id", "10.155.111.140"}, {"server_index", "1"}},
             }},
        }}}};
TEST_F(HcomTest, ut_hcom_91093_InitByFile_invalid_superPodId)
{
    char file_name_t[] = "./ranktable_invalid_superPodId.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(1) << ranktable_invalid_superPodId << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    } else {
        HCCL_ERROR("open %s failed", file_name_t);
    }

    outfile.close();
    int ret = HCCL_SUCCESS;

    DevType type91093 = DevType::DEV_TYPE_910_93;
    MOCKER(hrtGetDeviceType).stubs().with(outBound(type91093)).will(returnValue(HCCL_SUCCESS));

    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    u32 localrankid = 0;
    u32 localranksize = 0;
    char* rank_table_file = "./ranktable_invalid_superPodId.json";
    char* rank_ID = "0";

    MOCKER_CPP(
        &HcclCommunicatorAttrs::CheckSuperDeviceId, HcclResult(HcclCommunicatorAttrs::*)(const RankTable_t& rankTable))
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    ret = HcomInitByFile(rank_table_file, rank_ID);
    EXPECT_NE(ret, HCCL_SUCCESS);

    ret = HcomDestroy();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    remove(file_name_t);

    ret = hrtResetDevice(0);
    GlobalMockObject::verify();
}

nlohmann::json ranktable_invalid_serverId
    = {{"status", "completed"},
       {"version", "1.2"},
       {"server_list",
        {{
            {"server_id", "10.155.111.140"},
            {"device",
             {
                 {{"rank_id", "0"}, {"device_id", "0"}, {"super_device_id", "0"}, {"device_ip", "192.1.27.6"}},
             }},
        }}},
       {"super_pod_list",
        {{
            {"super_pod_id", "0"},
            {"server_list",
             {
                 {{"server_id", "10.155.111.666"}},
             }},
        }}}};
TEST_F(HcomTest, ut_hcom_91093_InitByFile_invalid_serverId)
{
    char file_name_t[] = "./ranktable_invalid_serverId.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(1) << ranktable_invalid_serverId << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    } else {
        HCCL_ERROR("open %s failed", file_name_t);
    }

    outfile.close();
    int ret = HCCL_SUCCESS;

    DevType type91093 = DevType::DEV_TYPE_910_93;
    MOCKER(hrtGetDeviceType).stubs().with(outBound(type91093)).will(returnValue(HCCL_SUCCESS));

    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    u32 localrankid = 0;
    u32 localranksize = 0;
    char* rank_table_file = "./ranktable_invalid_serverId.json";
    char* rank_ID = "0";

    ret = HcomInitByFile(rank_table_file, rank_ID);
    EXPECT_NE(ret, HCCL_SUCCESS);

    ret = HcomDestroy();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    remove(file_name_t);

    ret = hrtResetDevice(0);
    GlobalMockObject::verify();
}
#endif

#if 1
nlohmann::json super_pod_ranktable_1_2_4
    = {{"status", "completed"},
       {"version", "1.2"},
       {"server_list",
        {{
             {"server_id", "10.155.111.140"},
             {"device",
              {
                  {{"rank_id", "0"}, {"device_id", "0"}, {"super_device_id", "0"}, {"device_ip", "192.1.27.6"}},
                  {{"rank_id", "1"}, {"device_id", "1"}, {"super_device_id", "1"}, {"device_ip", "192.2.27.6"}},
                  {{"rank_id", "2"}, {"device_id", "2"}, {"super_device_id", "2"}, {"device_ip", "192.3.27.6"}},
                  {{"rank_id", "3"}, {"device_id", "3"}, {"super_device_id", "3"}, {"device_ip", "192.4.27.6"}},
              }},
         },
         {
             {"server_id", "10.155.111.141"},
             {"device",
              {
                  {{"rank_id", "4"}, {"device_id", "0"}, {"super_device_id", "4"}, {"device_ip", "192.1.27.7"}},
                  {{"rank_id", "5"}, {"device_id", "1"}, {"super_device_id", "5"}, {"device_ip", "192.2.27.7"}},
                  {{"rank_id", "6"}, {"device_id", "2"}, {"super_device_id", "6"}, {"device_ip", "192.3.27.7"}},
                  {{"rank_id", "7"}, {"device_id", "3"}, {"super_device_id", "7"}, {"device_ip", "192.4.27.7"}},
              }},
         }}},
       {"super_pod_list",
        {{
            {"super_pod_id", "0"},
            {"server_list",
             {
                 {{"server_id", "10.155.111.140"}},
                 {{"server_id", "10.155.111.141"}},
             }},
        }}}};

TEST_F(HcomTest, ut_hcom_91093_InitByFile)
{
    setenv("HCCL_INTER_HCCS_DISABLE", "false", 1);

    char file_name_t[] = "./super_pod_ranktable_1_2_4.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(1) << super_pod_ranktable_1_2_4 << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    } else {
        HCCL_ERROR("open %s failed", file_name_t);
    }

    outfile.close();
    int ret = HCCL_SUCCESS;

    DevType type91093 = DevType::DEV_TYPE_910_93;
    MOCKER(hrtGetDeviceType).stubs().with(outBound(type91093)).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtRaGetSingleSocketVnicIpInfo)
        .stubs()
        .with(mockcpp::any())
        .will(invoke(stub_hrtRaGetSingleSocketVnicIpInfo));
    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    u32 localrankid = 0;
    u32 localranksize = 0;
    char* rank_table_file = "./super_pod_ranktable_1_2_4.json";
    char* rank_ID = "0";

    MOCKER_CPP(
        &HcclCommunicatorAttrs::CheckSuperDeviceId, HcclResult(HcclCommunicatorAttrs::*)(const RankTable_t& rankTable))
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    ret = HcomInitByFile(rank_table_file, rank_ID);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcomGetLocalRankSize(NULL, &localranksize);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcomGetLocalRankId(NULL, &localrankid);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcomDestroy();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    remove(file_name_t);

    EXPECT_EQ(localranksize, 4);
    EXPECT_EQ(localrankid, 0);
    ret = hrtResetDevice(0);
    unsetenv("HCCL_INTER_HCCS_DISABLE");
    GlobalMockObject::verify();
}
#endif

TEST_F(HcomTest, ut_remote_acess_error)
{
    shared_ptr<RemoteAccess> RemoteAccess;
    RemoteAccess.reset(new (std::nothrow) hccl::RemoteAccess());
    vector<MemRegisterAddr> addrInfos;
    RmaRankTable rankTable;
    rankTable.serverNum = 0;
    HcclResult ret = RemoteAccess->Init(0, addrInfos, rankTable);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(HcomTest, ut_Destroy_backlogged_group)
{
    std::vector<u32> ranklist;
    HcomInfo& hcomInfo = HcomGetCtxHomInfo();
    hcomInfo.isHcomInit = true;

    HcclResult ret;
    std::string group = "testgroup";
    ret = HcomDestroyBackloggedGroup(group);
    EXPECT_EQ(ret, HCCL_E_PARA);
    hcomInfo.isHcomInit = false;
    ret = HcomDestroyBackloggedGroup(group);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
}

TEST_F(HcomTest, should_return_substream_num_when_910b_graph_allreduce_pipeline)
{
    HcclCommunicator comm;
    comm.deviceType_ = DevType::DEV_TYPE_910B;
    comm.userRankSize_ = 8;
    comm.moduleNum_ = 2;

    u64 streamNum = 0;
    u64 dataSize = 0;
    HcclCMDType opType = HcclCMDType::HCCL_CMD_ALLREDUCE;

    MOCKER_CPP_VIRTUAL(comm, &HcclCommunicator::GetAlgType).stubs().will(invoke(Stub_GetAlgType_pipeline));

    HcclResult result = comm.GetWorkspaceSubStreamNum(streamNum, dataSize, opType);

    EXPECT_EQ(result, HCCL_SUCCESS);
    EXPECT_EQ(streamNum, comm.userRankSize_ / comm.moduleNum_ - 1);
}

TEST_F(HcomTest, should_return_substream_num_when_910b_allgather_graph_pipeline)
{
    HcclCommunicator comm;
    comm.deviceType_ = DevType::DEV_TYPE_910B;
    comm.moduleNum_ = 2;
    comm.deviceNumPerAggregation_ = 8;
    comm.userRankSize_ = 16;

    u64 dataSize = HCCL_SMALL_COUNT_1_MB + 1;
    u64 streamNum = 0;

    HcclCMDType opType = HcclCMDType::HCCL_CMD_ALLGATHER;

    MOCKER_CPP_VIRTUAL(comm, &HcclCommunicator::GetAlgType)
        .stubs()
        .will(invoke(Stub_GetAlgType_ALG_ALLGATHER_REDUCESCATTER_GRAPH_PIPELINE));

    HcclResult result = comm.GetWorkspaceSubStreamNum(streamNum, dataSize, opType);

    EXPECT_EQ(result, HCCL_SUCCESS);
    EXPECT_EQ(streamNum, comm.userRankSize_ / comm.moduleNum_);
}

TEST_F(HcomTest, should_return_substream_num_when_910_93_allgather_graph_pipeline)
{
    HcclCommunicator comm;
    comm.deviceType_ = DevType::DEV_TYPE_910_93;
    comm.userRankSize_ = 8;
    comm.moduleNum_ = 2;
    comm.serverNum_ = 1;

    u64 streamNum = 0;
    constexpr u64 streamForSmallCount = 3;
    u64 dataSize = HCCL_SMALL_COUNT_512_KB - 1;
    HcclCMDType opType = HcclCMDType::HCCL_CMD_ALLGATHER;

    MOCKER_CPP_VIRTUAL(comm, &HcclCommunicator::GetAlgType)
        .stubs()
        .will(invoke(Stub_GetAlgType_ALG_ALLGATHER_REDUCESCATTER_GRAPH_PIPELINE));

    HcclResult result = comm.GetWorkspaceSubStreamNum(streamNum, dataSize, opType);

    EXPECT_EQ(result, HCCL_SUCCESS);
    EXPECT_EQ(streamNum, streamForSmallCount);
}

TEST_F(HcomTest, should_return_substream_num_when_910b_reducescatter_graph_pipeline)
{
    HcclCommunicator comm;
    comm.deviceType_ = DevType::DEV_TYPE_910B;
    comm.moduleNum_ = 2;
    comm.deviceNumPerAggregation_ = 4;
    comm.userRankSize_ = 8;

    u64 dataSize = HCCL_SMALL_COUNT_1_MB - 1;
    u64 streamNum = 0;

    HcclCMDType opType = HcclCMDType::HCCL_CMD_REDUCE_SCATTER;

    MOCKER_CPP_VIRTUAL(comm, &HcclCommunicator::GetAlgType)
        .stubs()
        .will(invoke(Stub_GetAlgType_ALG_ALLGATHER_REDUCESCATTER_GRAPH_PIPELINE));

    HcclResult result = comm.GetWorkspaceSubStreamNum(streamNum, dataSize, opType);

    EXPECT_EQ(result, HCCL_SUCCESS);
    EXPECT_EQ(streamNum, comm.userRankSize_ / comm.moduleNum_);
}

TEST_F(HcomTest, should_return_substream_num_when_910b_allgather_graph_pipeline_level1algo)
{
    HcclCommunicator comm;
    comm.deviceType_ = DevType::DEV_TYPE_910B;
    comm.moduleNum_ = 8;
    comm.deviceNumPerAggregation_ = 8;
    comm.userRankSize_ = 64;

    u64 dataSize = HCCL_SMALL_COUNT_1_MB - 1;
    u64 streamNum = 0;

    HcclCMDType opType = HcclCMDType::HCCL_CMD_ALLGATHER;

    MOCKER_CPP_VIRTUAL(comm, &HcclCommunicator::GetAlgType).stubs().will(invoke(Stub_GetAlgType_pipeline));

    HcclResult result = comm.GetWorkspaceSubStreamNum(streamNum, dataSize, opType);

    EXPECT_EQ(result, HCCL_SUCCESS);
    EXPECT_EQ(streamNum, comm.userRankSize_ / comm.moduleNum_);
}

TEST_F(HcomTest, should_return_substream_num_when_910b_reducescatter_graph_pipeline_level1algo)
{
    HcclCommunicator comm;
    comm.deviceType_ = DevType::DEV_TYPE_910B;
    comm.moduleNum_ = 8;
    comm.deviceNumPerAggregation_ = 4;
    comm.userRankSize_ = 32;

    u64 dataSize = HCCL_SMALL_COUNT_1_MB - 1;
    u64 streamNum = 0;

    HcclCMDType opType = HcclCMDType::HCCL_CMD_REDUCE_SCATTER;

    MOCKER_CPP_VIRTUAL(comm, &HcclCommunicator::GetAlgType).stubs().will(invoke(Stub_GetAlgType_pipeline));

    HcclResult result = comm.GetWorkspaceSubStreamNum(streamNum, dataSize, opType);

    EXPECT_EQ(result, HCCL_SUCCESS);
    EXPECT_EQ(streamNum, comm.userRankSize_ / comm.moduleNum_);
}

TEST_F(HcomTest, should_return_substream_num_when_910b_reduce_order_preservation)
{
    HcclCommunicator comm;
    comm.deviceType_ = DevType::DEV_TYPE_910B;
    comm.userRankSize_ = 5;
    comm.deviceNumPerAggregation_ = 5;
    comm.moduleNum_ = 1;

    u64 streamNum = 0;
    u64 dataSize = 0;
    HcclCMDType opType = HcclCMDType::HCCL_CMD_ALLREDUCE;

    MOCKER_CPP_VIRTUAL(comm, &HcclCommunicator::GetAlgType).stubs().will(invoke(Stub_GetAlgType_pipeline));
    MOCKER(GetExternalInputHcclDeterministicV2).stubs().will(returnValue(2));

    HcclResult result = comm.GetWorkspaceSubStreamNum(streamNum, dataSize, opType);
    EXPECT_EQ(result, HCCL_SUCCESS);
    EXPECT_EQ(streamNum, 7);
    GlobalMockObject::verify();
}

TEST_F(HcomTest, ut_group_fail_test)
{
    hcclComm comm(0, 0, "tag");
    std::string group = "";
    u32 groupRank = 0;
    u32 userRank = 0;
    std::vector<u32> groupRanks;
    std::shared_ptr<hcclComm> groupComm = nullptr;
    HcclResult ret = comm.CreateGroup(group, groupRank, userRank, groupRanks, groupComm);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(HcomTest, ut_HcomGetInitStatus_When_ParamIsNullptr_Expect_ReturnIsHCCL_E_PTR)
{
    HcclResult ret = HcomGetInitStatus(nullptr);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(HcomTest, ut_HclCommGraphSetWorkspaceResource_When_ParamIsNullptr_Expect_ReturnIsHCCL_E_PTR)
{
    std::string tag = "tag";
    std::vector<rtStream_t> stream;
    HcclResult ret = HcclCommGraphSetWorkspaceResource(tag, 0, stream, nullptr, 0);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(HcomTest, ut_HcomGetSplitStrategy_When_ParamIsNullptr_Expect_ReturnIsHCCL_E_PTR)
{
    std::string group = "group_name_0";
    struct model_feature feature;
    feature.model_name = reinterpret_cast<const char*>(0x1234);
    feature.gradient_size = reinterpret_cast<float*>(0x1234);
    feature.gradient_time = reinterpret_cast<float*>(0x1234);
    HcclResult ret = HcomGetSplitStrategy(
        group.c_str(), feature, nullptr, nullptr, nullptr, GradSplitForceMode::FORCE_NONE,
        OriginalGraphShapeType::KNOWN_SHAPE);
    EXPECT_EQ(ret, HCCL_E_PTR);

    u32 ptr = 0;
    ret = HcomGetSplitStrategy(
        group.c_str(), feature, &ptr, nullptr, nullptr, GradSplitForceMode::FORCE_NONE,
        OriginalGraphShapeType::KNOWN_SHAPE);
    EXPECT_EQ(ret, HCCL_E_PTR);

    ret = HcomGetSplitStrategy(
        group.c_str(), feature, &ptr, &ptr, nullptr, GradSplitForceMode::FORCE_NONE,
        OriginalGraphShapeType::KNOWN_SHAPE);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

// ============================================================================================
// GetAllReduceScratchSize 相关 UT（u32->u64 count 参数拓宽验证）
// 对应业务代码：src/legacy/ascend910/framework/hcom/hcom.cc
//   - HcclCommGraphGetAllReduceScratchSize(s64, const u64, const HcclDataType, u64 &)
//   - HcomGetAllReduceScratchSize(const char *, const u64, const HcclDataType, u64 &)
// ============================================================================================
namespace {
// 超过 UINT32_MAX 的 count 值：验证 u64 拓宽生效（u32 参数会截断为 0）
constexpr u64 U64_COUNT_EXCEED_U32 = 0x100000000ULL;
// 哨兵值：大 count 分支不写 outScratchSize 时保留此值，可检测 u32 截断为 0 的情况
constexpr u64 SENTINEL_SCRATCH = 7777ULL;
// mock 委托函数回写的任意值，用于验证出参是否正确传递
constexpr u64 EXPECTED_DELEGATE_OUT = 12345ULL;
// FP32 每个元素占 4 字节（SIZE_TABLE）；小数据 scratch = unitSize * count * (DEVICE_EIGHT - 1) = 4 * count * 7
constexpr u64 FP32_UNIT_SIZE = 4ULL;
constexpr u64 DEVICE_EIGHT_MINUS_ONE = 7ULL;

// 获取全局 HcomInfo 上下文引用
HcomInfo& GetScratchHcomInfo() { return HcomGetCtxHomInfo(); }

// 清空 HcomInfo 中的 pComm 和 group 映射，保证用例间无状态污染
void ClearScratchHcomInfo()
{
    HcomInfo& info = GetScratchHcomInfo();
    info.pComm.reset();
    info.hcomGroupMap.clear();
}

// 设置全局 pComm 为默认构造的 hcclComm 对象
void SetScratchPComm() { GetScratchHcomInfo().pComm.reset(new (std::nothrow) hccl::hcclComm()); }

// 在 hcomGroupMap 中插入一个子 group，withComm 控制是否带有效的 pSubComm
void SetScratchSubGroup(const std::string& group, bool withComm)
{
    HcclGroupParams params{};
    if (withComm) {
        params.pSubComm.reset(new (std::nothrow) hccl::hcclComm());
    }
    GetScratchHcomInfo().hcomGroupMap[group] = params;
}

// mock HcomCheckGroupName 返回成功
void MockCheckGroupNameOk() { MOCKER(HcomCheckGroupName).stubs().will(returnValue(HCCL_SUCCESS)); }

// mock HcomCheckGroupName 返回参数错误
void MockCheckGroupNameFail() { MOCKER(HcomCheckGroupName).stubs().will(returnValue(HCCL_E_PARA)); }

// mock HcclGetCommHandle 返回失败
void MockGetCommHandleFail() { MOCKER(HcclGetCommHandle).stubs().will(returnValue(HCCL_E_PARA)); }

// mock hcclComm::GetAllReduceScratchSize（const 成员函数），使委托路径返回受控结果并通过出参回写 outVal
void MockHcclCommScratchSize(HcclResult ret, u64 outVal)
{
    MOCKER_CPP(
        &hccl::hcclComm::GetAllReduceScratchSize,
        HcclResult(hccl::hcclComm::*)(const u64, const HcclDataType, u64&) const)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), outBound(outVal))
        .will(returnValue(ret));
}
} // namespace

// --------------------------------------------------------------------------------------------
// HcclCommGraphGetAllReduceScratchSize：opBaseHcom 为 0 时 CHK_PTR_NULL 返回 HCCL_E_PTR；
// 否则委托 hcclComm->GetAllReduceScratchSize(count, dataType, outScratchSize)
// --------------------------------------------------------------------------------------------
class HcclCommGraphGetAllReduceScratchSizeTest : public testing::Test {
protected:
    void SetUp() override
    {
        ClearScratchHcomInfo();
        // 在 SetUp 中构造 hcclComm，生命周期受 fixture 控制，避免 static 局部对象永不析构的问题
        comm_ = std::make_unique<hccl::hcclComm>();
    }

    void TearDown() override
    {
        // 在 TearDown 中释放 hcclComm，确保资源及时回收
        comm_.reset();
        ClearScratchHcomInfo();
        GlobalMockObject::verify();
    }

    // 返回非空 opBaseHcom 指针，底层由成员 comm_ 支撑。被测函数的委托方法已被 mock，对象不会真正执行设备操作
    s64 MakeOpHandle() const { return reinterpret_cast<s64>(comm_.get()); }

private:
    // fixture 成员，构造与析构时机由 SetUp/TearDown 显式控制
    std::unique_ptr<hccl::hcclComm> comm_;
};

// 参数验证：opBaseHcom 为 0（空指针）时，必须在委托前被 CHK_PTR_NULL 拒绝
TEST_F(
    HcclCommGraphGetAllReduceScratchSizeTest,
    Ut_HcclCommGraphGetAllReduceScratchSize_When_OpBaseHcomZero_Expect_ReturnPtrError)
{
    const u64 count = 100;
    u64 outScratchSize = 0;
    HcclResult ret = HcclCommGraphGetAllReduceScratchSize(0, count, HCCL_DATA_TYPE_FP32, outScratchSize);
    EXPECT_EQ(ret, HCCL_E_PTR) << "空 opBaseHcom 必须通过 CHK_PTR_NULL 返回 HCCL_E_PTR";
}

// 正常路径：有效句柄 + 正常 count -> 委托成功且出参正确传递
TEST_F(
    HcclCommGraphGetAllReduceScratchSizeTest,
    Ut_HcclCommGraphGetAllReduceScratchSize_When_ValidHandleNormalCount_Expect_DelegateSuccess)
{
    MockHcclCommScratchSize(HCCL_SUCCESS, EXPECTED_DELEGATE_OUT);
    u64 outScratchSize = 0;
    HcclResult ret = HcclCommGraphGetAllReduceScratchSize(MakeOpHandle(), 100, HCCL_DATA_TYPE_FP32, outScratchSize);
    EXPECT_EQ(ret, HCCL_SUCCESS) << "有效句柄必须透传委托的成功返回值";
    EXPECT_EQ(outScratchSize, EXPECTED_DELEGATE_OUT) << "outScratchSize 必须从委托函数正确传递";
}

// 边界条件：count 超过 UINT32_MAX 必须作为完整 u64 被接受（本次 u32->u64 修改的核心验证点）
TEST_F(
    HcclCommGraphGetAllReduceScratchSizeTest,
    Ut_HcclCommGraphGetAllReduceScratchSize_When_CountExceedsU32Max_Expect_DelegateAcceptsFullU64)
{
    MockHcclCommScratchSize(HCCL_SUCCESS, EXPECTED_DELEGATE_OUT);
    u64 outScratchSize = 0;
    HcclResult ret = HcclCommGraphGetAllReduceScratchSize(
        MakeOpHandle(), U64_COUNT_EXCEED_U32, HCCL_DATA_TYPE_FP32, outScratchSize);
    EXPECT_EQ(ret, HCCL_SUCCESS) << "超过 UINT32_MAX 的 u64 count 不应破坏委托调用";
    EXPECT_EQ(outScratchSize, EXPECTED_DELEGATE_OUT) << "大 u64 count 下 outScratchSize 必须正确传递";
}

// 边界条件：count == 0 时仍应成功委托
TEST_F(
    HcclCommGraphGetAllReduceScratchSizeTest,
    Ut_HcclCommGraphGetAllReduceScratchSize_When_CountIsZero_Expect_DelegateSuccess)
{
    MockHcclCommScratchSize(HCCL_SUCCESS, EXPECTED_DELEGATE_OUT);
    u64 outScratchSize = 0;
    HcclResult ret = HcclCommGraphGetAllReduceScratchSize(MakeOpHandle(), 0, HCCL_DATA_TYPE_FP32, outScratchSize);
    EXPECT_EQ(ret, HCCL_SUCCESS) << "count 为 0 时必须透传委托的成功返回值";
}

// 异常路径：委托函数返回错误时，错误码必须原样透传
TEST_F(
    HcclCommGraphGetAllReduceScratchSizeTest,
    Ut_HcclCommGraphGetAllReduceScratchSize_When_DelegateReturnsError_Expect_ErrorPropagated)
{
    MockHcclCommScratchSize(HCCL_E_PARA, 0);
    u64 outScratchSize = 0;
    HcclResult ret = HcclCommGraphGetAllReduceScratchSize(MakeOpHandle(), 100, HCCL_DATA_TYPE_FP32, outScratchSize);
    EXPECT_EQ(ret, HCCL_E_PARA) << "委托函数的错误必须原样透传给调用方";
}

// --------------------------------------------------------------------------------------------
// HcomGetAllReduceScratchSize 分支说明：
//   A  pComm != nullptr
//      A1  strGroup == HCCL_WORLD_GROUP          -> 委托
//      A2  子 group
//          A2a 在映射表中
//              A2a-i  pSubComm == nullptr        -> HCCL_E_PTR
//              A2a-ii pSubComm != nullptr        -> 委托
//          A2b 不在映射表                          -> 小数据计算 scratch = memSize*(8-1); 返回 SUCCESS
//   B  else if group==nullptr || HcclGetCommHandle 失败 -> outScratchSize=0; 返回 SUCCESS
// --------------------------------------------------------------------------------------------
class HcomGetAllReduceScratchSizeTest : public testing::Test {
protected:
    void SetUp() override { ClearScratchHcomInfo(); }

    void TearDown() override
    {
        ClearScratchHcomInfo();
        GlobalMockObject::verify();
    }
};

// 分支 B（参数验证）：pComm 为空且 group 为 nullptr -> 返回成功，outScratchSize 置 0，不查找通信域
TEST_F(HcomGetAllReduceScratchSizeTest, Ut_HcomGetAllReduceScratchSize_When_PCommNullGroupNull_Expect_SuccessZeroOut)
{
    u64 outScratchSize = SENTINEL_SCRATCH;
    HcclResult ret = HcomGetAllReduceScratchSize(nullptr, 100, HCCL_DATA_TYPE_FP32, outScratchSize);
    EXPECT_EQ(ret, HCCL_SUCCESS) << "group 为空且无 pComm 时必须返回成功";
    EXPECT_EQ(outScratchSize, 0ULL) << "未初始化分支必须将 outScratchSize 重置为 0";
}

// 分支 B（异常）：pComm 为空、group 非空但 HcclGetCommHandle 失败 -> 返回成功，outScratchSize 置 0
TEST_F(HcomGetAllReduceScratchSizeTest, Ut_HcomGetAllReduceScratchSize_When_PCommNullHandleFails_Expect_SuccessZeroOut)
{
    MockGetCommHandleFail();
    u64 outScratchSize = SENTINEL_SCRATCH;
    HcclResult ret = HcomGetAllReduceScratchSize("any_group", 100, HCCL_DATA_TYPE_FP32, outScratchSize);
    EXPECT_EQ(ret, HCCL_SUCCESS) << "通信域句柄查找失败时必须回退为成功并置零 scratch";
    EXPECT_EQ(outScratchSize, 0ULL) << "未调用初始化时 outScratchSize 必须重置为 0";
}

// 分支 A1（正常）：pComm 已设置、world group（group=nullptr）-> 委托成功且出参正确传递
TEST_F(HcomGetAllReduceScratchSizeTest, Ut_HcomGetAllReduceScratchSize_When_PCommSetWorldGroup_Expect_DelegateSuccess)
{
    SetScratchPComm();
    MockCheckGroupNameOk();
    MockHcclCommScratchSize(HCCL_SUCCESS, EXPECTED_DELEGATE_OUT);
    u64 outScratchSize = 0;
    HcclResult ret = HcomGetAllReduceScratchSize(nullptr, 100, HCCL_DATA_TYPE_FP32, outScratchSize);
    EXPECT_EQ(ret, HCCL_SUCCESS) << "world group 必须委托到 hcclComm->GetAllReduceScratchSize";
    EXPECT_EQ(outScratchSize, EXPECTED_DELEGATE_OUT) << "outScratchSize 必须从委托函数正确传递";
}

// 分支 A2a-ii（正常）：pComm 已设置、子 group 在映射表中且 pSubComm 有效 -> 委托成功
TEST_F(
    HcomGetAllReduceScratchSizeTest, Ut_HcomGetAllReduceScratchSize_When_PCommSetSubGroupInMap_Expect_DelegateSuccess)
{
    SetScratchPComm();
    SetScratchSubGroup("sub_group", true);
    MockCheckGroupNameOk();
    MockHcclCommScratchSize(HCCL_SUCCESS, EXPECTED_DELEGATE_OUT);
    u64 outScratchSize = 0;
    HcclResult ret = HcomGetAllReduceScratchSize("sub_group", 100, HCCL_DATA_TYPE_FP32, outScratchSize);
    EXPECT_EQ(ret, HCCL_SUCCESS) << "映射表中的子 group 必须通过 pSubComm 委托";
    EXPECT_EQ(outScratchSize, EXPECTED_DELEGATE_OUT) << "outScratchSize 必须从委托函数正确传递";
}

// 分支 A2a-i（异常）：pComm 已设置、子 group 在映射表中但 pSubComm 为空 -> 返回 HCCL_E_PTR
TEST_F(
    HcomGetAllReduceScratchSizeTest,
    Ut_HcomGetAllReduceScratchSize_When_PCommSetSubGroupInMapNullSubComm_Expect_PtrError)
{
    SetScratchPComm();
    SetScratchSubGroup("sub_group", false);
    MockCheckGroupNameOk();
    u64 outScratchSize = 0;
    HcclResult ret = HcomGetAllReduceScratchSize("sub_group", 100, HCCL_DATA_TYPE_FP32, outScratchSize);
    EXPECT_EQ(ret, HCCL_E_PTR) << "映射表中 pSubComm 为空时必须返回 HCCL_E_PTR";
}

// 分支 A2b（正常）：pComm 已设置、子 group 不在映射表中、小数据 -> 计算 scratch = unitSize * count * 7
TEST_F(
    HcomGetAllReduceScratchSizeTest,
    Ut_HcomGetAllReduceScratchSize_When_PCommSetSubGroupNotInMapSmallData_Expect_ComputedScratch)
{
    SetScratchPComm();
    MockCheckGroupNameOk();
    const u64 count = 100;
    u64 outScratchSize = 0;
    HcclResult ret = HcomGetAllReduceScratchSize("missing_group", count, HCCL_DATA_TYPE_FP32, outScratchSize);
    EXPECT_EQ(ret, HCCL_SUCCESS) << "不在映射表中的子 group 必须计算 scratch 并返回成功";
    const u64 expected = FP32_UNIT_SIZE * count * DEVICE_EIGHT_MINUS_ONE;
    EXPECT_EQ(outScratchSize, expected) << "小数据 scratch 必须为 memSize * (DEVICE_EIGHT - 1)";
}

// 分支 A2b（边界，u32->u64 修复验证）：count > UINT32_MAX 时 memSize 保持为大值，跳过小数据分支，outScratchSize
// 不被改写。 若参数仍为 u32，count 会被截断为 0，memSize 算成 0，错误进入小数据分支并将 outScratchSize 覆盖为 0
TEST_F(
    HcomGetAllReduceScratchSizeTest,
    Ut_HcomGetAllReduceScratchSize_When_PCommSetSubGroupNotInMapLargeU64Count_Expect_OutUnchanged)
{
    SetScratchPComm();
    MockCheckGroupNameOk();
    u64 outScratchSize = SENTINEL_SCRATCH;
    HcclResult ret
        = HcomGetAllReduceScratchSize("missing_group", U64_COUNT_EXCEED_U32, HCCL_DATA_TYPE_FP32, outScratchSize);
    EXPECT_EQ(ret, HCCL_SUCCESS) << "大 u64 count 不应破坏不在映射表中的分支";
    EXPECT_EQ(outScratchSize, SENTINEL_SCRATCH) << "大 u64 count 必须跳过小数据分支（无截断）";
}

// 分支 A2b（边界）：count == 0 -> memSize 为 0，小数据分支 -> scratch 为 0
TEST_F(
    HcomGetAllReduceScratchSizeTest,
    Ut_HcomGetAllReduceScratchSize_When_PCommSetSubGroupNotInMapZeroCount_Expect_ZeroScratch)
{
    SetScratchPComm();
    MockCheckGroupNameOk();
    u64 outScratchSize = SENTINEL_SCRATCH;
    HcclResult ret = HcomGetAllReduceScratchSize("missing_group", 0, HCCL_DATA_TYPE_FP32, outScratchSize);
    EXPECT_EQ(ret, HCCL_SUCCESS) << "count 为 0 时必须计算 scratch 并返回成功";
    EXPECT_EQ(outScratchSize, 0ULL) << "count 为 0 时小数据分支必须产生 0 scratch";
}

// 参数验证 / 异常：HcomCheckGroupName 失败时，错误码必须通过 CHK_RET 透传
TEST_F(HcomGetAllReduceScratchSizeTest, Ut_HcomGetAllReduceScratchSize_When_CheckGroupNameFails_Expect_ErrorPropagated)
{
    SetScratchPComm();
    MockCheckGroupNameFail();
    u64 outScratchSize = 0;
    HcclResult ret = HcomGetAllReduceScratchSize("any_group", 100, HCCL_DATA_TYPE_FP32, outScratchSize);
    EXPECT_EQ(ret, HCCL_E_PARA) << "HcomCheckGroupName 失败时错误码必须透传";
}
