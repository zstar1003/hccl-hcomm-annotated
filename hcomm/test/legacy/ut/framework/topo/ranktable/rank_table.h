/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_RANK_TABLE_STUB_H
#define HCCLV2_RANK_TABLE_STUB_H

#include <string>

const char ranktablePath[] = "ranktable.json";
const char ranktable4pPath[] = "ranktable.json";
const char topoPath[] = "topo.json";

const std::string RankTable1Ser8Dev = R"(
    {
    "server_count":"1",
    "server_list":
    [
        {
            "device":[
                        {
                        "device_id":"0",
                        "rank_id":"0"
                        },
                        {
                        "device_id":"1",
                        "rank_id":"1"
                        },
                        {
                        "device_id":"2",
                        "rank_id":"2"
                        },
                        {
                        "device_id":"3",
                        "rank_id":"3"
                        },
                        {
                        "device_id":"4",
                        "rank_id":"4"
                        },
                        {
                        "device_id":"5",
                        "rank_id":"5"
                        },
                        {
                        "device_id":"6",
                        "rank_id":"6"
                        },
                        {
                        "device_id":"7",
                        "rank_id":"7"
                        }
                    ],
            "server_id":"1"
        }
    ],
    "status":"completed",
    "version":"1.0"
    }
    )";

const std::string RankTable4p = R"(
{
  "version": "2.0",
  "status": "completed",
  "detour": "true",
  "rank_count": 4,
  "rank_list": [
    {
      "rank_id": 0,
      "device_id": 0,
      "local_id": 0,
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
              "addr": "192.168.100.1",
              "ports": [ "0/0"]
            },{
              "addr_type": "IPV4",
              "addr": "192.168.100.2",
              "ports": [ "0/1"]
            },{
              "addr_type": "IPV4",
              "addr": "192.168.100.3",
              "ports": [ "0/2"]
            }
          ]
        },
        {
          "net_layer": 1,
          "net_instance_id": "az0-rack0",
          "net_type": "CLOS",
          "net_attr": "",
          "rank_addr_list": [
            {
              "addr_type": "IPV4",
              "addr": "192.168.101.1",
              "ports": [ "0/7", "0/8"]
            }
          ]
        },{
          "net_layer": 2,
          "net_instance_id": "az0-rack0",
          "net_type": "CLOS",
          "net_attr": "",
          "rank_addr_list": [
            {
              "addr_type": "IPV4",
              "addr": "192.168.102.1",
              "ports": [ "1/7", "1/8"]
            }
          ]
        }
      ]
    },
    {
      "rank_id": 1,
      "device_id": 1,
      "local_id": 1,
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
              "addr": "192.168.100.11",
              "ports": [ "0/0"]
            },{
              "addr_type": "IPV4",
              "addr": "192.168.100.12",
              "ports": [ "0/1"]
            },{
              "addr_type": "IPV4",
              "addr": "192.168.100.13",
              "ports": [ "0/2"]
            }
          ]
        },
        {
          "net_layer": 1,
          "net_instance_id": "az0-rack0",
          "net_type": "CLOS",
          "net_attr": "",
          "rank_addr_list": [
            {
              "addr_type": "IPV4",
              "addr": "192.168.101.11",
              "ports": [ "0/7", "0/8"]
            }
          ]
        },{
          "net_layer": 2,
          "net_instance_id": "az0-rack0",
          "net_type": "CLOS",
          "net_attr": "",
          "rank_addr_list": [
            {
              "addr_type": "IPV4",
              "addr": "192.168.102.11",
              "ports": [ "1/7", "1/8"]
            }
          ]
        }
      ]
    },
    {
      "rank_id": 2,
      "device_id": 2,
      "local_id": 2,
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
              "addr": "192.168.100.21",
              "ports": [ "0/0"]
            },{
              "addr_type": "IPV4",
              "addr": "192.168.100.22",
              "ports": [ "0/1"]
            },{
              "addr_type": "IPV4",
              "addr": "192.168.100.23",
              "ports": [ "0/2"]
            }
          ]
        },
        {
          "net_layer": 1,
          "net_instance_id": "az0-rack0",
          "net_type": "CLOS",
          "net_attr": "",
          "rank_addr_list": [
            {
              "addr_type": "IPV4",
              "addr": "192.168.101.21",
              "ports": [ "0/7", "0/8"]
            }
          ]
        },{
          "net_layer": 2,
          "net_instance_id": "az0-rack0",
          "net_type": "CLOS",
          "net_attr": "",
          "rank_addr_list": [
            {
              "addr_type": "IPV4",
              "addr": "192.168.102.21",
              "ports": [ "1/7", "1/8"]
            }
          ]
        }
      ]
    },
    {
      "rank_id": 3,
      "device_id": 3,
      "local_id": 3,
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
              "addr": "192.168.100.31",
              "ports": [ "0/0"]
            },{
              "addr_type": "IPV4",
              "addr": "192.168.100.32",
              "ports": [ "0/1"]
            },{
              "addr_type": "IPV4",
              "addr": "192.168.100.33",
              "ports": [ "0/2"]
            }
          ]
        },
        {
          "net_layer": 1,
          "net_instance_id": "az0-rack0",
          "net_type": "CLOS",
          "net_attr": "",
          "rank_addr_list": [
            {
              "addr_type": "IPV4",
              "addr": "192.168.101.31",
              "ports": [ "0/7", "0/8"]
            }
          ]
        },{
          "net_layer": 2,
          "net_instance_id": "az0-rack0",
          "net_type": "CLOS",
          "net_attr": "",
          "rank_addr_list": [
            {
              "addr_type": "IPV4",
              "addr": "192.168.102.31",
              "ports": [ "1/7", "1/8"]
            }
          ]
        }
      ]
    }
  ]
}
)";

const std::string Topo1Ser8Dev = R"(
    {
  "version": "2.0",
  "peer_count": 4,
  "peer_list": [
    {
      "local_id": 0
    },
    {
      "local_id": 1
    },
    {
      "local_id": 2
    },
    {
      "local_id": 3
    }
  ],
  "edge_count": 14,
  "edge_list": [
    {
      "net_layer": 0,
      "link_type": "PEER2PEER",
      "protocols": [
        "UB_CTP"
      ],
      "local_a": 0,
      "local_a_ports": [
        "0/0"
      ],
      "local_b": 1,
      "local_b_ports": [
        "0/0"
      ],
      "position": "DEVICE"
    },
    {
      "net_layer": 0,
      "link_type": "PEER2PEER",
      "protocols": [
        "UB_CTP"
      ],
      "local_a": 0,
      "local_a_ports": [
        "0/1"
      ],
      "local_b": 2,
      "local_b_ports": [
        "0/0"
      ],
      "position": "DEVICE"
    },
    {
      "net_layer": 0,
      "link_type": "PEER2PEER",
      "protocols": [
        "UB_CTP"
      ],
      "local_a": 0,
      "local_a_ports": [
        "0/2"
      ],
      "local_b": 3,
      "local_b_ports": [
        "0/0"
      ],
      "position": "DEVICE"
    },
    {
      "net_layer": 0,
      "link_type": "PEER2PEER",
      "protocols": [
        "UB_CTP"
      ],
      "local_a": 1,
      "local_a_ports": [
        "0/1"
      ],
      "local_b": 2,
      "local_b_ports": [
        "0/1"
      ],
      "position": "DEVICE"
    },
    {
      "net_layer": 0,
      "link_type": "PEER2PEER",
      "protocols": [
        "UB_CTP"
      ],
      "local_a": 1,
      "local_a_ports": [
        "0/2"
      ],
      "local_b": 3,
      "local_b_ports": [
        "0/1"
      ],
      "position": "DEVICE"
    },
    {
      "net_layer": 0,
      "link_type": "PEER2PEER",
      "protocols": [
        "UB_CTP"
      ],
      "local_a": 2,
      "local_a_ports": [
        "0/2"
      ],
      "local_b": 3,
      "local_b_ports": [
        "0/2"
      ],
      "position": "DEVICE"
    },
    {
      "net_layer": 1,
      "link_type": "PEER2NET",
      "protocols": [
        "UB_CTP"
      ],
      "local_a": 0,
      "local_a_ports": [
        "0/7",
        "0/8"
      ],
      "position": "DEVICE"
    },
    {
      "net_layer": 1,
      "link_type": "PEER2NET",
      "protocols": [
        "UB_CTP"
      ],
      "local_a": 1,
      "local_a_ports": [
        "0/7",
        "0/8"
      ],
      "position": "DEVICE"
    },
    {
      "net_layer": 1,
      "link_type": "PEER2NET",
      "protocols": [
        "UB_CTP"
      ],
      "local_a": 2,
      "local_a_ports": [
        "0/7",
        "0/8"
      ],
      "position": "DEVICE"
    },
    {
      "net_layer": 1,
      "link_type": "PEER2NET",
      "protocols": [
        "UB_CTP"
      ],
      "local_a": 3,
      "local_a_ports": [
        "0/7",
        "0/8"
      ],
      "position": "DEVICE"
    },
    {
      "net_layer": 2,
      "link_type": "PEER2NET",
      "protocols": [
        "UB_CTP"
      ],
      "local_a": 0,
      "local_a_ports": [
        "1/7",
        "1/8"
      ],
      "position": "HOST"
    },
    {
      "net_layer": 2,
      "link_type": "PEER2NET",
      "protocols": [
        "UB_CTP"
      ],
      "local_a": 1,
      "local_a_ports": [
        "1/7",
        "1/8"
      ],
      "position": "HOST"
    },
    {
      "net_layer": 2,
      "link_type": "PEER2NET",
      "protocols": [
        "UB_CTP"
      ],
      "local_a": 2,
      "local_a_ports": [
        "1/7",
        "1/8"
      ],
      "position": "HOST"
    },
    {
      "net_layer": 2,
      "link_type": "PEER2NET",
      "protocols": [
        "UB_CTP"
      ],
      "local_a": 3,
      "local_a_ports": [
        "1/7",
        "1/8"
      ],
      "position": "HOST"
    }
  ]
}
    )";

void GenRankTableFile1Ser8Dev();

void DelRankTableFile();

void GenRankTableFile4p();

void DelRankTableFile4p();

void GenTopoFile();

void DelTopoFile();

#endif
