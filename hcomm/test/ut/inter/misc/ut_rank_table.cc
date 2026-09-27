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

#include "llt_hccl_stub_pub.h"
#define private public
#define protected public
#include "rank_table_pub.h"
#undef private
#undef protected
#include <iostream>

using namespace std;
using namespace cltm;

class RanktableTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "RanktableTest SetUP" << std::endl; }
    static void TearDownTestCase() { std::cout << "RanktableTest TearDown" << std::endl; }
    virtual void SetUp() { std::cout << "A Test SetUP" << std::endl; }
    virtual void TearDown() { std::cout << "A Test TearDown" << std::endl; }
};

#define MAX_BUF_SIZE (4 * 1024 * 1024)

TEST_F(RanktableTest, ut_CheckUniqueInfo)
{
    std::string ipAddr = "10.78.232.102";
    RankTable rankTable(NULL, MAX_BUF_SIZE);

    cltmResult_t ret = rankTable.CheckUniqueInfo(UNIQUE_TYPE_HOST_IP, ipAddr);
    EXPECT_EQ(ret, CLTM_SUCCESS);

    ipAddr = "10.78.232.1021";
    ret = rankTable.CheckUniqueInfo(UNIQUE_TYPE_HOST_IP, ipAddr);
    EXPECT_EQ(ret, CLTM_E_PARA);
}

TEST_F(RanktableTest, ut_init)
{
    nlohmann::json rank_table
        = {{"group_count", "1"},
           {"allocated_group_resource",
            {{{"group_name", "world_group"},
              {"dev_num", "1"},
              {"server_num", "1"},
              {"allocated_resource",
               {{{"server_id", "100.12.4.1"},
                 {"chip_info", "910"},
                 {"board_id", "0"},
                 {"avail_dev_count", "1"},
                 {"para_plane_eth_count", "2"},
                 {"para_plane_eth",
                  {{{"eth_name", "eth0"}, {"host_ip_addr", "172.17.0.114"}},
                   {{"eth_name", "eth0"}, {"host_ip_addr", "172.17.0.115"}}}},
                 {"avail_dev", {{{"device_index", "0"}, {"device_ip_addr", "172.17.1.116"}}}}}}}}}}};
    std::string allocatedResouce = rank_table.dump();

    char* inputRes = (char*)sal_malloc(MAX_BUF_SIZE);
    sal_strncpy(inputRes, MAX_BUF_SIZE, allocatedResouce.c_str(), allocatedResouce.length());
    char* rankTableBuf = (char*)sal_malloc(MAX_BUF_SIZE);
    unsigned int maxBufSize = MAX_BUF_SIZE;
    unsigned int usedBufSize = 0;
    RankTable rankTable(NULL, MAX_BUF_SIZE);
    rankTable.allocRes_ = inputRes;

    cltmResult_t ret = rankTable.init();
    EXPECT_EQ(ret, CLTM_SUCCESS);

    sal_free(inputRes);
    sal_free(rankTableBuf);
}

TEST_F(RanktableTest, ut_AddServer)
{
    nlohmann::json rank_table
        = {{"group_count", "1"},
           {"allocated_group_resource",
            {{{"group_name", "world_group"},
              {"dev_num", "1"},
              {"server_num", "1"},
              {"allocated_resource",
               {{{"server_id", "100.12.4.1"},
                 {"chip_info", "910"},
                 {"board_id", "0"},
                 {"avail_dev_count", "1"},
                 {"para_plane_eth_count", "2"},
                 {"para_plane_eth",
                  {{{"eth_name", "eth0"}, {"host_ip_addr", "172.17.0.114"}},
                   {{"eth_name", "eth0"}, {"host_ip_addr", "172.17.0.115"}}}},
                 {"avail_dev", {{{"device_index", "0"}, {"device_ip_addr", "172.17.1.116"}}}}}}}}}}};
    std::string allocatedResouce = rank_table.dump();
    RankTable rankTable(NULL, MAX_BUF_SIZE);
    cltmResult_t ret = rankTable.AddServer(rank_table);
    EXPECT_EQ(ret, CLTM_E_PARA);
}

TEST_F(RanktableTest, ut_AddGroup)
{
    nlohmann::json rank_table
        = {{"group_count", "1"},
           {"allocated_group_resource",
            {{{"group_name", "world_group"},
              {"dev_num", "1"},
              {"server_num", "1"},
              {"allocated_resource",
               {{{"server_id", "100.12.4.1"},
                 {"chip_info", "910"},
                 {"board_id", "0"},
                 {"avail_dev_count", "1"},
                 {"para_plane_eth_count", "2"},
                 {"para_plane_eth",
                  {{{"eth_name", "eth0"}, {"host_ip_addr", "172.17.0.114"}},
                   {{"eth_name", "eth0"}, {"host_ip_addr", "172.17.0.115"}}}},
                 {"avail_dev", {{{"device_index", "0"}, {"device_ip_addr", "172.17.1.116"}}}}}}}}}}};
    std::string allocatedResouce = rank_table.dump();
    RankTable rankTable(NULL, MAX_BUF_SIZE);
    cltmResult_t ret = rankTable.AddGroup(rank_table);
    EXPECT_EQ(ret, CLTM_E_PARA);
}

TEST_F(RanktableTest, ut_GetGroupsInfo)
{
    nlohmann::json rank_table
        = {{"group_count", "1"},
           {"allocated_group_resource",
            {{{"group_name", "world_group"},
              {"dev_num", "1"},
              {"server_num", "1"},
              {"allocated_resource",
               {{{"server_id", "100.12.4.1"},
                 {"chip_info", "910"},
                 {"board_id", "0"},
                 {"avail_dev_count", "1"},
                 {"para_plane_eth_count", "2"},
                 {"para_plane_eth",
                  {{{"eth_name", "eth0"}, {"host_ip_addr", "172.17.0.114"}},
                   {{"eth_name", "eth0"}, {"host_ip_addr", "172.17.0.115"}}}},
                 {"avail_dev", {{{"device_index", "0"}, {"device_ip_addr", "172.17.1.116"}}}}}}}}}}};
    std::string allocatedResouce = rank_table.dump();
    RankTable rankTable(NULL, MAX_BUF_SIZE);
    cltmResult_t ret = rankTable.GetGroupsInfo(rank_table);
    EXPECT_EQ(ret, CLTM_E_PARA);
}

TEST_F(RanktableTest, ut_GenerateRankTable)
{
    nlohmann::json rank_table
        = {{"group_count", "1"},
           {"allocated_group_resource",
            {{{"group_name", "world_group"},
              {"dev_num", "1"},
              {"server_num", "1"},
              {"allocated_resource",
               {{{"server_id", "100.12.4.1"},
                 {"chip_info", "910"},
                 {"board_id", "0"},
                 {"avail_dev_count", "1"},
                 {"para_plane_eth_count", "2"},
                 {"para_plane_eth",
                  {{{"eth_name", "eth0"}, {"host_ip_addr", "172.17.0.114"}},
                   {{"eth_name", "eth0"}, {"host_ip_addr", "172.17.0.115"}}}},
                 {"avail_dev", {{{"device_index", "0"}, {"device_ip_addr", "172.17.1.116"}}}}}}}}}}};
    std::string allocatedResouce = rank_table.dump();

    char* inputRes = (char*)sal_malloc(MAX_BUF_SIZE);
    sal_strncpy(inputRes, MAX_BUF_SIZE, allocatedResouce.c_str(), allocatedResouce.length());
    char* rankTableBuf = (char*)sal_malloc(MAX_BUF_SIZE);
    unsigned int maxBufSize = MAX_BUF_SIZE;
    unsigned int usedBufSize = 0;

    RankTable rankTable(NULL, MAX_BUF_SIZE);
    cltmResult_t ret = rankTable.GenerateRankTable(rankTableBuf, &usedBufSize);
    EXPECT_EQ(ret, CLTM_E_PARA);

    sal_free(inputRes);
    sal_free(rankTableBuf);
}
