/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_RANKTABLE_BUILDER_ST_H
#define HCCLV2_RANKTABLE_BUILDER_ST_H

#include <string>

const std::string RankTable3p = R"(
    {
  "version": "2.0",
  "status": "completed",
  "detour": "true",
  "rank_count": 3,
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
          "net_type": "CLOS",
          "net_attr": "",
          "rank_addr_list": [
            {
              "addr_type": "IPV4",
              "addr": "192.168.100.11",
              "ports": [ "0/1"],
              "plane_id": "planeA"
            },
            {
              "addr_type": "IPV4",
              "addr": "192.168.100.10",
              "ports": [ "0/3"],
              "plane_id": "planeB"
            }
          ],
          "control_plane": {
            "addr_type": "IPV4",
            "addr": "192.168.100.100",
            "listen_port": 8000
          }
        },
        {
          "net_layer": 1,
          "net_instance_id": "az0",
          "net_type": "CLOS",
          "net_attr": "",
          "rank_addr_list": [
            {
              "addr_type": "IPV4",
              "addr": "192.168.100.20",
              "ports": [ "0/5"],
              "plane_id": "planeA"
            },
            {
              "addr_type": "IPV4",
              "addr": "192.168.100.21",
              "ports": [ "0/7"],
              "plane_id": "planeB"
            }
          ],
          "control_plane": {
            "addr_type": "IPV4",
            "addr": "192.168.100.102",
            "listen_port": 8001
          }
        },
        {
          "net_layer": 2,
          "net_instance_id": "all",
          "net_type": "CLOS",
          "net_attr": "",
          "rank_addr_list": [
            {
              "addr_type": "IPV4",
              "addr": "192.168.100.30",
              "ports": [ "0/9"],
              "plane_id": "planeA"
            },
            {
              "addr_type": "IPV4",
              "addr": "192.168.100.31",
              "ports": [ "0/11"],
              "plane_id": "planeB"
            }
          ],
          "control_plane": {
            "addr_type": "IPV4",
            "addr": "192.168.100.103",
            "listen_port": 8003
          }
        }
      ]
    },
    {
      "rank_id": 1,
      "device_id": 0,
      "local_id": 1,
      "replaced_local_id": 10,
      "device_port": 6666,
      "level_list": [
        {
          "net_layer": 0,
          "net_instance_id": "az0-rack0",
          "net_type": "CLOS",
          "net_attr": "",
          "rank_addr_list": [
            {
              "addr_type": "IPV4",
              "addr": "192.168.100.11",
              "ports": [ "1/0"],
              "plane_id": "planeA"
            },
            {
              "addr_type": "IPV4",
              "addr": "192.168.100.10",
              "ports": [ "1/3"],
              "plane_id": "planeB"
            }
          ],
          "control_plane": {
            "addr_type": "IPV4",
            "addr": "192.168.100.100",
            "listen_port": 8000
          }
        },
        {
          "net_layer": 1,
          "net_instance_id": "az0",
          "net_type": "CLOS",
          "net_attr": "",
          "rank_addr_list": [
            {
              "addr_type": "IPV4",
              "addr": "192.168.100.20",
              "ports": [ "1/5"],
              "plane_id": "planeA"
            },
            {
              "addr_type": "IPV4",
              "addr": "192.168.100.21",
              "ports": [ "1/7"],
              "plane_id": "planeB"
            }
          ],
          "control_plane": {
            "addr_type": "IPV4",
            "addr": "192.168.100.102",
            "listen_port": 8001
          }
        },
        {
          "net_layer": 2,
          "net_instance_id": "all",
          "net_type": "CLOS",
          "net_attr": "",
          "rank_addr_list": [
            {
              "addr_type": "IPV4",
              "addr": "192.168.100.30",
              "ports": [ "1/9"],
              "plane_id": "planeA"
            },
            {
              "addr_type": "IPV4",
              "addr": "192.168.100.31",
              "ports": [ "1/11"],
              "plane_id": "planeB"
            }
          ],
          "control_plane": {
            "addr_type": "IPV4",
            "addr": "192.168.100.103",
            "listen_port": 8003
          }
        }
      ]
    },
    {
      "rank_id": 2,
      "device_id": 0,
      "local_id": 2,
      "replaced_local_id": 10,
      "device_port": 6666,
      "level_list": [
        {
          "net_layer": 0,
          "net_instance_id": "az0-rack0",
          "net_type": "CLOS",
          "net_attr": "",
          "rank_addr_list": [
            {
              "addr_type": "IPV4",
              "addr": "192.168.100.11",
              "ports": [ "2/0"],
              "plane_id": "planeA"
            },
            {
              "addr_type": "IPV4",
              "addr": "192.168.100.10",
              "ports": [ "2/3"],
              "plane_id": "planeB"
            }
          ],
          "control_plane": {
            "addr_type": "IPV4",
            "addr": "192.168.100.100",
            "listen_port": 8000
          }
        },
        {
          "net_layer": 1,
          "net_instance_id": "az0",
          "net_type": "CLOS",
          "net_attr": "",
          "rank_addr_list": [
            {
              "addr_type": "IPV4",
              "addr": "192.168.100.20",
              "ports": [ "2/5"],
              "plane_id": "planeA"
            },
            {
              "addr_type": "IPV4",
              "addr": "192.168.100.21",
              "ports": [ "2/7"],
              "plane_id": "planeB"
            }
          ],
          "control_plane": {
            "addr_type": "IPV4",
            "addr": "192.168.100.102",
            "listen_port": 8001
          }
        },
        {
          "net_layer": 2,
          "net_instance_id": "all",
          "net_type": "CLOS",
          "net_attr": "",
          "rank_addr_list": [
            {
              "addr_type": "IPV4",
              "addr": "192.168.100.30",
              "ports": [ "2/9"],
              "plane_id": "planeA"
            },
            {
              "addr_type": "IPV4",
              "addr": "192.168.100.31",
              "ports": [ "2/11"],
              "plane_id": "planeB"
            }
          ],
          "control_plane": {
            "addr_type": "IPV4",
            "addr": "192.168.100.103",
            "listen_port": 8003
          }
        }
      ]
    }
  ]
}
)";

const std::string RankTable2p = R"(
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
          "net_type": "CLOS",
          "net_attr": "",
          "rank_addr_list": [
            {
              "addr_type": "IPV4",
              "addr": "192.168.100.11",
              "ports": [ "0/1", "0/2" ],
              "plane_id": "planeA"
            },
            {
              "addr_type": "IPV4",
              "addr": "192.168.100.10",
              "ports": [ "1/1", "1/2" ],
              "plane_id": "planeB"
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
          "net_type": "CLOS",
          "net_attr": "",
          "rank_addr_list": [
            {
              "addr_type": "IPV4",
              "addr": "192.168.100.20",
              "ports": [ "0/3", "0/4" ],
              "plane_id": "planeA"
            },
            {
              "addr_type": "IPV4",
              "addr": "192.168.100.21",
              "ports": [ "1/3", "1/4" ],
              "plane_id": "planeB"
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

const std::string RankTable_4p = R"(
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
          "net_instance_id": "az0",
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
          "net_instance_id": "all",
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
          "net_instance_id": "az0",
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
          "net_instance_id": "all",
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
      "device_port": 3333,
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
          "net_instance_id": "az0",
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
          "net_instance_id": "all",
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
      "device_port": 4222,
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
          "net_instance_id": "az0",
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
          "net_instance_id": "all",
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

const std::string RankTable2p_err = R"(
    {
        "version": "2.0",
        "rank_count" : "2",
        "rank_list": [
            {
                "rank_id": 0,
                "local_id": 0,
                "level_list":  [
                    {
                        "level": 0,
                        "id" : "az0-rack0", 
                        "fabric_type": "INNER", 
                        "rank_addr_type": "",
                        "rank_addrs": []
                    }
                ]
            },
            {
                "rank_id": 1,
                "local_id": 9,
                "level_list":  [
                    {
                        "level": 0,
                        "id" : "az0-rack0", 
                        "fabric_type": "INNER", 
                        "rank_addr_type": "",
                        "rank_addrs": [] 
                    }
                ]
            }
        ],

        "replace_count" : 1,
        "replace_list" : [ 
            {"level": 0,  "group_id" : "az0-rack0", "backup_local_id": 64, "target_local_id": 1}
        ]
    }
)";

const std::string RankTable4p_2X2 = R"(
    {
        "version": "2.0",
        "rank_count" : "4",
        "rank_list": [
            {
                "rank_id": 0,
                "local_id": 0,
                "level_list":  [
                    {
                        "level": 0,
                        "id" : "az0-rack0", 
                        "fabric_type": "INNER", 
                        "rank_addr_type": "",
                        "rank_addrs": []
                    }
                ]
            },
            {
                "rank_id": 1,
                "local_id": 1,
                "level_list":  [
                    {
                        "level": 0,
                        "id" : "az0-rack0", 
                        "fabric_type": "INNER", 
                        "rank_addr_type": "",
                        "rank_addrs": []
                    }
                ]
            },
            {
                "rank_id": 2,
                "local_id": 8,
                "level_list":  [
                    {
                        "level": 0,
                        "id" : "az0-rack0", 
                        "fabric_type": "INNER", 
                        "rank_addr_type": "",
                        "rank_addrs": []
                    }
                ]
            },
            {
                "rank_id": 3,
                "local_id": 9,
                "level_list":  [
                    {
                        "level": 0,
                        "id" : "az0-rack0", 
                        "fabric_type": "INNER", 
                        "rank_addr_type": "",
                        "rank_addrs": []
                    }
                ]
            }
        ],

        "replace_count" : 1,
        "replace_list" : [ 
            {"level": 0,  "group_id" : "az0-rack0", "backup_local_id": 64, "target_local_id": 1}
        ]
    }
)";
#endif
