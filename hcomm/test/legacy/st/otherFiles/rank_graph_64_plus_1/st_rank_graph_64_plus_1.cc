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
#include <mockcpp/mokc.h>
#include <mockcpp/mockcpp.hpp>
#include <iostream>

#define private public
#define protected public

#include "rank_graph_builder.h"
#include "rank_gph.h"
#include "phy_topo_builder.h"
#include "phy_topo.h"
#include "detour_service.h"
#include "diff_rank_updater.h"
#include "graph.h"

#undef private
#undef protected

using namespace Hccl;
using namespace std;

class RankGraph64Plus1Test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "RankGraph64Plus1Test SetUP" << endl; }

    static void TearDownTestCase() { cout << "RankGraph64Plus1Test TearDown" << endl; }

    virtual void SetUp()
    {
        PhyTopo::GetInstance()->Clear();                       // PhyTopo是单例，每个用例开始前需要重置
        MOCKER_CPP(&DetourService::InsertDetourLinks).stubs(); // 64+1场景暂时不涉及绕路，将绕路接口打桩成空函数
        cout << "A Test case in RankGraph64Plus1Test SetUP" << endl;
    }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        cout << "A Test case in RankGraph64Plus1Test TearDown" << endl;
    }
};

const std::string RANK_TABLE_2X2 = R"(
{
    "version": "2.0",
	"rank_count" : 4,
	"rank_list": [
		{
			"rank_id": 0,
			"device_id": 0,
			"local_id": 0,
			"level_list":  [
				{
					"net_layer": 0,
					"net_instance_id" : "az0-rack0",
					"net_type": "TOPO_FILE_DESC",
					"net_attr": "",
					"rank_addr_list": [
					{
					"addr_type": "IPV4",
                    "addr": "192.168.100.100",
                    "ports": [ "0/0" ]
					},
					{
					"addr_type": "IPV4",
                    "addr": "192.168.100.101",
                    "ports": [ "0/1" ]
					},
					{
					"addr_type": "IPV4",
                    "addr": "192.168.100.102",
                    "ports": [ "0/2" ]
					},
					{
					"addr_type": "IPV4",
                    "addr": "192.168.100.110",
                    "ports": [ "1/0" ]
					},
					{
					"addr_type": "IPV4",
                    "addr": "192.168.100.111",
                    "ports": [ "1/1" ]
					},
					{
					"addr_type": "IPV4",
                    "addr": "192.168.100.112",
                    "ports": [ "1/2" ]
					}
					]
				},
                {   
                    "net_layer": 1,
                    "net_instance_id" : "az0-layer1",
					"net_type":"CLOS",
					"net_attr": "",
					"rank_addr_list": [
					{
					"addr_type": "IPV4",
                    "addr": "192.168.100.108",
                    "ports": [ "0/8" ]
					}
					]
                }
			]
		},
		{
			"rank_id": 1,
			"device_id": 1,
			"local_id": 1,
			"level_list":  [
				{
					"net_layer": 0,
					"net_instance_id" : "az0-rack0",
					"net_type": "TOPO_FILE_DESC",
					"net_attr": "",
					"rank_addr_list": [
					{
					"addr_type": "IPV4",
                    "addr": "192.168.101.100",
                    "ports": [ "0/0" ]
					},
					{
					"addr_type": "IPV4",
                    "addr": "192.168.101.101",
                    "ports": [ "0/1" ]
					},
					{
					"addr_type": "IPV4",
                    "addr": "192.168.101.102",
                    "ports": [ "0/2" ]
					},
					{
					"addr_type": "IPV4",
                    "addr": "192.168.101.110",
                    "ports": [ "1/0" ]
					},
					{
					"addr_type": "IPV4",
                    "addr": "192.168.101.111",
                    "ports": [ "1/1" ]
					},
					{
					"addr_type": "IPV4",
                    "addr": "192.168.101.112",
                    "ports": [ "1/2" ]
					}
					]
				},
                {   
                    "net_layer": 1,
                    "net_instance_id" : "az0-layer1",
					"net_type":"CLOS",
					"net_attr": "",
					"rank_addr_list": [
					{
					"addr_type": "IPV4",
                    "addr": "192.168.101.108",
                    "ports": [ "0/8" ]
					}
					]
                }
			]
		},
		{
			"rank_id": 2,
			"device_id": 8,
			"local_id": 8,
			"level_list":  [
				{
					"net_layer": 0,
					"net_instance_id" : "az0-rack0",
					"net_type": "TOPO_FILE_DESC",
					"net_attr": "",
					"rank_addr_list": [
					{
					"addr_type": "IPV4",
                    "addr": "192.168.108.100",
                    "ports": [ "0/0" ]
					},
					{
					"addr_type": "IPV4",
                    "addr": "192.168.108.101",
                    "ports": [ "0/1" ]
					},
					{
					"addr_type": "IPV4",
                    "addr": "192.168.108.102",
                    "ports": [ "0/2" ]
					},
					{
					"addr_type": "IPV4",
                    "addr": "192.168.108.110",
                    "ports": [ "1/0" ]
					},
					{
					"addr_type": "IPV4",
                    "addr": "192.168.108.111",
                    "ports": [ "1/1" ]
					},
					{
					"addr_type": "IPV4",
                    "addr": "192.168.108.112",
                    "ports": [ "1/2" ]
					}
					]
				},
                {   
                    "net_layer": 1,
                    "net_instance_id" : "az0-layer1",
					"net_type":"CLOS",
					"net_attr": "",
					"rank_addr_list": [
					{
					"addr_type": "IPV4",
                    "addr": "192.168.108.108",
                    "ports": [ "0/8" ]
					}
					]
                }
			]
		},
		{
			"rank_id": 3,
			"device_id": 9,
			"local_id": 9,
			"level_list":  [
				{
					"net_layer": 0,
					"net_instance_id" : "az0-rack0",
					"net_type": "TOPO_FILE_DESC",
					"net_attr": "",
					"rank_addr_list": [
					{
					"addr_type": "IPV4",
                    "addr": "192.168.109.100",
                    "ports": [ "0/0" ]
					},
					{
					"addr_type": "IPV4",
                    "addr": "192.168.109.101",
                    "ports": [ "0/1" ]
					},
					{
					"addr_type": "IPV4",
                    "addr": "192.168.109.102",
                    "ports": [ "0/2" ]
					},
					{
					"addr_type": "IPV4",
                    "addr": "192.168.109.110",
                    "ports": [ "1/0" ]
					},
					{
					"addr_type": "IPV4",
                    "addr": "192.168.109.111",
                    "ports": [ "1/1" ]
					},
					{
					"addr_type": "IPV4",
                    "addr": "192.168.109.112",
                    "ports": [ "1/2" ]
					}
					]
				},
                {   
                    "net_layer": 1,
                    "net_instance_id" : "az0-layer1",
					"net_type":"CLOS",
					"net_attr": "",
					"rank_addr_list": [
					{
					"addr_type": "IPV4",
                    "addr": "192.168.109.108",
                    "ports": [ "0/8" ]
					}
					]
                }
			]
		}
	]
}
)";

TEST_F(RankGraph64Plus1Test, test_checkpoint_normal_to_backup)
{
    // 快照恢复，正常场景切换到备份场景
    JsonParser parser;
    RankTableInfo rankTableInfo;
    parser.ParseString(RANK_TABLE_2X2, rankTableInfo);
    TopoInfo topoInfo;
    string topoFilePath = "llt/ace/comop/hccl/orion/st/otherFiles/rank_graph_64_plus_1/topo_2x2plus1.json";
    parser.ParseFile(topoFilePath, topoInfo);

    // 更新 rankTableInfo
    const char* changeInfoFilePath
        = "llt/ace/comop/hccl/orion/st/otherFiles/rank_graph_64_plus_1/changeInfo_2x2_normal_to_backup.json";
    EXPECT_EQ(DiffRankUpdater(changeInfoFilePath, rankTableInfo), HcclResult::HCCL_SUCCESS);

    RankGraphBuilder rankGraphBuilder;
    unique_ptr<RankGraph> rankGraph = rankGraphBuilder.RecoverBuild(rankTableInfo, topoInfo, 0);

    EXPECT_NE(rankGraph, nullptr);
}
