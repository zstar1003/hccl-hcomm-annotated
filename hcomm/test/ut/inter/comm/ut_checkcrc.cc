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
#include <slog.h>
#include "mem_host_pub.h"
#include <string>
#include <stdlib.h>

using namespace std;
using namespace hccl;

class CheckCrcTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "\033[36m--CheckCrcTest SetUP--\033[0m" << std::endl; }
    static void TearDownTestCase() { std::cout << "\033[36m--CheckCrcTest TearDown--\033[0m" << std::endl; }
    virtual void SetUp() { std::cout << "A Test SetUP" << std::endl; }
    virtual void TearDown() { std::cout << "A Test TearDown" << std::endl; }
};

TEST_F(CheckCrcTest, ut_CheckCrc_CalcFileCrc)
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
    std::string rank_table_file("./ut_hcom_get_hcom_info_eth0IPErr.json");
    std::string rankTable;
    std::string realFilePath;

    HcclResult ret = HcomLoadRanktableFile(rank_table_file.c_str(), rankTable, realFilePath);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    u32 srcCrc1 = 10;
    ret = CalcCrc::HcclCalcCrc(rankTable, srcCrc1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    remove(file_name);
}
