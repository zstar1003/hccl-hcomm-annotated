/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_RANKTABLE_STUB_64_1_H
#define HCCLV2_RANKTABLE_STUB_64_1_H

#include <string>

const std::string RankTable2pClos = R"(
    {
  "version": "2.0",
  "status": "completed",
  "detour": "true",
  "rank_count": 2,
  "rank_list": [
    {
      "rank_id": 0,
      "device_id": 0,
      "local_id": 0,
      "replaced_local_id": 10,
      "device_port": 6666,
      "level_list": [
        {
          "net_layer": 0,
          "net_instance_id": "az0-rack0",
          "net_type": "TOPO_FILE_DESC",
          "net_attr": "",
          "rank_addr_list": [
            {
              "addr_type": "IPV4",
              "addr": "192.168.100.10",
              "ports": [ "0/0"]
            },
			 {
              "addr_type": "IPV4",
              "addr": "192.168.100.11",
              "ports": [ "0/1" ]
            },
			{
              "addr_type": "IPV4",
              "addr": "192.168.100.12",
              "ports": [ "0/2" ]
            },
            {
              "addr_type": "IPV4",
              "addr": "192.168.100.13",
              "ports": [ "0/3" ]
            },
			{
              "addr_type": "IPV4",
              "addr": "192.168.100.14",
              "ports": [ "0/4" ]
            }
          ],
          "control_plane": {
            "addr_type": "IPV4",
            "addr": "192.168.100.100",
            "listen_port": 8000
          }
        }
      ]
    },
    {
      "rank_id": 1,
      "device_id": 1,
      "local_id": 1,
      "replaced_local_id": 10,
      "device_port": 6666,
      "level_list": [
        {
          "net_layer": 0,
          "net_instance_id": "az0-rack0",
          "net_type": "TOPO_FILE_DESC",
          "net_attr": "",
          "rank_addr_list": [
            {
              "addr_type": "IPV4",
              "addr": "192.168.100.20",
              "ports": [ "0/0"]
            },
			 {
              "addr_type": "IPV4",
              "addr": "192.168.100.21",
              "ports": [ "1/1" ]
            },
			{
              "addr_type": "IPV4",
              "addr": "192.168.100.22",
              "ports": [ "1/2" ]
            },
            {
              "addr_type": "IPV4",
              "addr": "192.168.100.23",
              "ports": [ "1/3" ]
            },
			{
              "addr_type": "IPV4",
              "addr": "192.168.100.24",
              "ports": [ "1/4" ]
            }
          ],
          "control_plane": {
            "addr_type": "IPV4",
            "addr": "192.168.100.102",
            "listen_port": 8001
          }
        }
      ]
    }
  ]
}
)";

const std::string RANK_TABLE_4P_REPLACE_RANK1 = R"(
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
                    "addr": "192.168.164.100",
                    "ports": [ "0/0" ]
					},
					{
					"addr_type": "IPV4",
                    "addr": "192.168.164.101",
                    "ports": [ "0/1" ]
					},
					{
					"addr_type": "IPV4",
                    "addr": "192.168.164.102",
                    "ports": [ "0/2" ]
					},
					{
					"addr_type": "IPV4",
                    "addr": "192.168.164.103",
                    "ports": [ "0/3" ]
					},
					{
					"addr_type": "IPV4",
                    "addr": "192.168.164.104",
                    "ports": [ "0/4" ]
					},
					{
					"addr_type": "IPV4",
                    "addr": "192.168.164.105",
                    "ports": [ "0/5" ]
					},
					{
					"addr_type": "IPV4",
                    "addr": "192.168.164.106",
                    "ports": [ "0/6" ]
					},
					{
					"addr_type": "IPV4",
                    "addr": "192.168.164.107",
                    "ports": [ "0/7" ]
					},
					{
					"addr_type": "IPV4",
                    "addr": "192.168.164.108",
                    "ports": [ "0/8" ]
					},
					{
					"addr_type": "IPV4",
                    "addr": "192.168.164.110",
                    "ports": [ "1/0" ]
					},
					{
					"addr_type": "IPV4",
                    "addr": "192.168.164.111",
                    "ports": [ "1/1" ]
					},
					{
					"addr_type": "IPV4",
                    "addr": "192.168.164.112",
                    "ports": [ "1/2" ]
					},
					{
					"addr_type": "IPV4",
                    "addr": "192.168.164.113",
                    "ports": [ "1/3" ]
					},
					{
					"addr_type": "IPV4",
                    "addr": "192.168.164.114",
                    "ports": [ "1/4" ]
					},
					{
					"addr_type": "IPV4",
                    "addr": "192.168.164.115",
                    "ports": [ "1/5" ]
					},
					{
					"addr_type": "IPV4",
                    "addr": "192.168.164.116",
                    "ports": [ "1/6" ]
					},
					{
					"addr_type": "IPV4",
                    "addr": "192.168.164.117",
                    "ports": [ "1/7" ]
					},
					{
					"addr_type": "IPV4",
                    "addr": "192.168.164.118",
                    "ports": [ "1/8" ]
					}
					]
				}
			]
		},
		{
			"rank_id": 1,
			"device_id": 64,
			"local_id": 64,
            "replaced_local_id": 1,
			"level_list":  [
				{
					"net_layer": 0,
					"net_instance_id" : "az0-rack0",
					"net_type": "TOPO_FILE_DESC",
					"net_attr": "",
					"rank_addr_list": [
					{
					"addr_type": "IPV4",
                    "addr": "192.168.164.100",
                    "ports": [ "0/0" ]
					},
					{
					"addr_type": "IPV4",
                    "addr": "192.168.164.101",
                    "ports": [ "0/1" ]
					},
					{
					"addr_type": "IPV4",
                    "addr": "192.168.164.102",
                    "ports": [ "0/2" ]
					},
					{
					"addr_type": "IPV4",
                    "addr": "192.168.164.103",
                    "ports": [ "0/3" ]
					},
					{
					"addr_type": "IPV4",
                    "addr": "192.168.164.104",
                    "ports": [ "0/4" ]
					},
					{
					"addr_type": "IPV4",
                    "addr": "192.168.164.105",
                    "ports": [ "0/5" ]
					},
					{
					"addr_type": "IPV4",
                    "addr": "192.168.164.106",
                    "ports": [ "0/6" ]
					},
					{
					"addr_type": "IPV4",
                    "addr": "192.168.164.107",
                    "ports": [ "0/7" ]
					},
					{
					"addr_type": "IPV4",
                    "addr": "192.168.164.108",
                    "ports": [ "0/8" ]
					},
					{
					"addr_type": "IPV4",
                    "addr": "192.168.164.110",
                    "ports": [ "1/0" ]
					},
					{
					"addr_type": "IPV4",
                    "addr": "192.168.164.111",
                    "ports": [ "1/1" ]
					},
					{
					"addr_type": "IPV4",
                    "addr": "192.168.164.112",
                    "ports": [ "1/2" ]
					},
					{
					"addr_type": "IPV4",
                    "addr": "192.168.164.113",
                    "ports": [ "1/3" ]
					},
					{
					"addr_type": "IPV4",
                    "addr": "192.168.164.114",
                    "ports": [ "1/4" ]
					},
					{
					"addr_type": "IPV4",
                    "addr": "192.168.164.115",
                    "ports": [ "1/5" ]
					},
					{
					"addr_type": "IPV4",
                    "addr": "192.168.164.116",
                    "ports": [ "1/6" ]
					},
					{
					"addr_type": "IPV4",
                    "addr": "192.168.164.117",
                    "ports": [ "1/7" ]
					},
					{
					"addr_type": "IPV4",
                    "addr": "192.168.164.118",
                    "ports": [ "1/8" ]
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
                    "addr": "192.168.164.100",
                    "ports": [ "0/0" ]
					},
					{
					"addr_type": "IPV4",
                    "addr": "192.168.164.101",
                    "ports": [ "0/1" ]
					},
					{
					"addr_type": "IPV4",
                    "addr": "192.168.164.102",
                    "ports": [ "0/2" ]
					},
					{
					"addr_type": "IPV4",
                    "addr": "192.168.164.103",
                    "ports": [ "0/3" ]
					},
					{
					"addr_type": "IPV4",
                    "addr": "192.168.164.104",
                    "ports": [ "0/4" ]
					},
					{
					"addr_type": "IPV4",
                    "addr": "192.168.164.105",
                    "ports": [ "0/5" ]
					},
					{
					"addr_type": "IPV4",
                    "addr": "192.168.164.106",
                    "ports": [ "0/6" ]
					},
					{
					"addr_type": "IPV4",
                    "addr": "192.168.164.107",
                    "ports": [ "0/7" ]
					},
					{
					"addr_type": "IPV4",
                    "addr": "192.168.164.108",
                    "ports": [ "0/8" ]
					},
					{
					"addr_type": "IPV4",
                    "addr": "192.168.164.110",
                    "ports": [ "1/0" ]
					},
					{
					"addr_type": "IPV4",
                    "addr": "192.168.164.111",
                    "ports": [ "1/1" ]
					},
					{
					"addr_type": "IPV4",
                    "addr": "192.168.164.112",
                    "ports": [ "1/2" ]
					},
					{
					"addr_type": "IPV4",
                    "addr": "192.168.164.113",
                    "ports": [ "1/3" ]
					},
					{
					"addr_type": "IPV4",
                    "addr": "192.168.164.114",
                    "ports": [ "1/4" ]
					},
					{
					"addr_type": "IPV4",
                    "addr": "192.168.164.115",
                    "ports": [ "1/5" ]
					},
					{
					"addr_type": "IPV4",
                    "addr": "192.168.164.116",
                    "ports": [ "1/6" ]
					},
					{
					"addr_type": "IPV4",
                    "addr": "192.168.164.117",
                    "ports": [ "1/7" ]
					},
					{
					"addr_type": "IPV4",
                    "addr": "192.168.164.118",
                    "ports": [ "1/8" ]
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
                    "addr": "192.168.164.100",
                    "ports": [ "0/0" ]
					},
					{
					"addr_type": "IPV4",
                    "addr": "192.168.164.101",
                    "ports": [ "0/1" ]
					},
					{
					"addr_type": "IPV4",
                    "addr": "192.168.164.102",
                    "ports": [ "0/2" ]
					},
					{
					"addr_type": "IPV4",
                    "addr": "192.168.164.103",
                    "ports": [ "0/3" ]
					},
					{
					"addr_type": "IPV4",
                    "addr": "192.168.164.104",
                    "ports": [ "0/4" ]
					},
					{
					"addr_type": "IPV4",
                    "addr": "192.168.164.105",
                    "ports": [ "0/5" ]
					},
					{
					"addr_type": "IPV4",
                    "addr": "192.168.164.106",
                    "ports": [ "0/6" ]
					},
					{
					"addr_type": "IPV4",
                    "addr": "192.168.164.107",
                    "ports": [ "0/7" ]
					},
					{
					"addr_type": "IPV4",
                    "addr": "192.168.164.108",
                    "ports": [ "0/8" ]
					},
					{
					"addr_type": "IPV4",
                    "addr": "192.168.164.110",
                    "ports": [ "1/0" ]
					},
					{
					"addr_type": "IPV4",
                    "addr": "192.168.164.111",
                    "ports": [ "1/1" ]
					},
					{
					"addr_type": "IPV4",
                    "addr": "192.168.164.112",
                    "ports": [ "1/2" ]
					},
					{
					"addr_type": "IPV4",
                    "addr": "192.168.164.113",
                    "ports": [ "1/3" ]
					},
					{
					"addr_type": "IPV4",
                    "addr": "192.168.164.114",
                    "ports": [ "1/4" ]
					},
					{
					"addr_type": "IPV4",
                    "addr": "192.168.164.115",
                    "ports": [ "1/5" ]
					},
					{
					"addr_type": "IPV4",
                    "addr": "192.168.164.116",
                    "ports": [ "1/6" ]
					},
					{
					"addr_type": "IPV4",
                    "addr": "192.168.164.117",
                    "ports": [ "1/7" ]
					},
					{
					"addr_type": "IPV4",
                    "addr": "192.168.164.118",
                    "ports": [ "1/8" ]
					}
					]
				}
			]
		}
	]
}
)";

#endif
