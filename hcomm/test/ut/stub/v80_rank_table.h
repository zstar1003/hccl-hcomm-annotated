/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once
#include <nlohmann/json.hpp>
// ranktable 910 1p
static nlohmann::json rank_table_910_1server_1rank
    = {{"status", "completed"},
       {"deploy_mode", "lab"},
       {"group_count", "1"},
       {"chip_info", "910"},
       {"board_id", "0x0000"},
       {"para_plane_nic_location", "device"},
       {"para_plane_nic_num", "1"},
       {"para_plane_nic_name", {"eth0"}},
       {"group_list",
        {{{"group_name", ""},
          {"device_num", "1"},
          {"server_num", "1"},
          {"instance_count", "1"},
          {"instance_list",
           {
               {{"rank_id", "0"},
                {"server_id", "10.0.0.10"},
                {"devices", {{{"device_id", "0"}, {"device_ip", "192.168.0.12"}}}}},
           }},
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
           }}}}}};

// ranktable 910 2p
static nlohmann::json rank_table_910_1server_2rank
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
          {"server_num", "1"},
          {"instance_count", "2"},
          {"instance_list",
           {
               {{"rank_id", "0"},
                {"server_id", "10.0.0.10"},
                {"devices", {{{"device_id", "0"}, {"device_ip", "192.168.0.12"}}}}},

               {{"rank_id", "1"},
                {"server_id", "10.0.0.10"},
                {"devices", {{{"device_id", "1"}, {"device_ip", "192.168.0.14"}}}}},
           }},
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
           }}}}}};

// ranktable 910 4p
static nlohmann::json rank_table_910_1server_4rank
    = {{"status", "completed"},
       {"deploy_mode", "lab"},
       {"group_count", "1"},
       {"chip_info", "910"},
       {"board_id", "0x0000"},
       {"para_plane_nic_location", "device"},
       {"para_plane_nic_num", "4"},
       {"para_plane_nic_name", {"eth0", "eth1", "eth2", "eth3"}},
       {"group_list",
        {{{"group_name", ""},
          {"device_num", "4"},
          {"server_num", "1"},
          {"instance_count", "4"},
          {"instance_list",
           {
               {{"rank_id", "0"},
                {"server_id", "10.0.0.10"},
                {"devices", {{{"device_id", "0"}, {"device_ip", "192.168.0.12"}, {"device_port", "16666"}}}}},

               {{"rank_id", "1"},
                {"server_id", "10.0.0.10"},
                {"devices", {{{"device_id", "1"}, {"device_ip", "192.168.0.14"}, {"device_port", "16666"}}}}},
               {{"rank_id", "2"},
                {"server_id", "10.0.0.10"},
                {"devices", {{{"device_id", "2"}, {"device_ip", "192.168.0.16"}, {"device_port", "16666"}}}}},

               {{"rank_id", "3"},
                {"server_id", "10.0.0.10"},
                {"devices", {{{"device_id", "3"}, {"device_ip", "192.168.0.18"}, {"device_port", "16666"}}}}},

           }},
          {"server_list",
           {
               {{"server_id", "10.0.0.10"},
                {"para_plane_info",
                 {{
                      {"eth0", "192.168.210.2"},
                  },
                  {
                      {"eth1", "192.168.200.2"},
                  },
                  {
                      {"eth2", "192.168.210.2"},
                  },
                  {
                      {"eth3", "192.168.200.2"},
                  }}}

               },
           }}}}}};

static nlohmann::json rank_table_910_1server_4rank_new
    = {{"status", "completed"},
       {"deploy_mode", "lab"},
       {"group_count", "1"},
       {"chip_info", "910"},
       {"board_id", "0x0000"},
       {"para_plane_nic_location", "device"},
       {"para_plane_nic_num", "4"},
       {"para_plane_nic_name", {"eth0", "eth1", "eth2", "eth3"}},
       {"group_list",
        {{{"group_name", ""},
          {"device_num", "4"},
          {"server_num", "1"},
          {"instance_count", "4"},
          {"instance_list",
           {
               {{"rank_id", "0"},
                {"server_id", "10.0.0.10"},
                {"devices", {{{"device_id", "0"}, {"device_ip", "192.168.0.2"}, {"device_port", "16666"}}}}},

               {{"rank_id", "1"},
                {"server_id", "10.0.0.10"},
                {"devices", {{{"device_id", "1"}, {"device_ip", "192.168.0.4"}, {"device_port", "16666"}}}}},
               {{"rank_id", "2"},
                {"server_id", "10.0.0.10"},
                {"devices", {{{"device_id", "2"}, {"device_ip", "192.168.0.6"}, {"device_port", "16666"}}}}},

               {{"rank_id", "3"},
                {"server_id", "10.0.0.10"},
                {"devices", {{{"device_id", "3"}, {"device_ip", "192.168.0.8"}, {"device_port", "16666"}}}}},

           }},
          {"server_list",
           {
               {{"server_id", "10.0.0.10"},
                {"para_plane_info",
                 {{
                      {"eth0", "192.168.210.2"},
                  },
                  {
                      {"eth1", "192.168.200.2"},
                  },
                  {
                      {"eth2", "192.168.210.2"},
                  },
                  {
                      {"eth3", "192.168.200.2"},
                  }}}

               },
           }}}}}};

// ranktable 910B 2*4p
static nlohmann::json rank_table_910_2server_4rank
    = {{"status", "completed"},
       {"deploy_mode", "lab"},
       {"group_count", "1"},
       {"chip_info", "910"},
       {"board_id", "0x0000"},
       {"para_plane_nic_location", "device"},
       {"para_plane_nic_num", "8"},
       {"para_plane_nic_name", {"eth0", "eth1", "eth2", "eth3"}},
       {"group_list",
        {{{"group_name", ""},
          {"device_num", "8"},
          {"server_num", "2"},
          {"instance_count", "8"},
          {"instance_list",
           {
               {{"rank_id", "0"},
                {"server_id", "10.0.0.10"},
                {"devices", {{{"device_id", "0"}, {"device_ip", "192.168.0.12"}, {"device_port", "16666"}}}}},
               {{"rank_id", "1"},
                {"server_id", "10.0.0.10"},
                {"devices", {{{"device_id", "1"}, {"device_ip", "192.168.0.14"}, {"device_port", "16666"}}}}},
               {{"rank_id", "2"},
                {"server_id", "10.0.0.10"},
                {"devices", {{{"device_id", "2"}, {"device_ip", "192.168.0.16"}, {"device_port", "16666"}}}}},
               {{"rank_id", "3"},
                {"server_id", "10.0.0.10"},
                {"devices", {{{"device_id", "3"}, {"device_ip", "192.168.0.18"}, {"device_port", "16666"}}}}},
               {{"rank_id", "4"},
                {"server_id", "10.0.0.11"},
                {"devices", {{{"device_id", "0"}, {"device_ip", "192.168.0.22"}, {"device_port", "16666"}}}}},
               {{"rank_id", "5"},
                {"server_id", "10.0.0.11"},
                {"devices", {{{"device_id", "1"}, {"device_ip", "192.168.0.24"}, {"device_port", "16666"}}}}},
               {{"rank_id", "6"},
                {"server_id", "10.0.0.11"},
                {"devices", {{{"device_id", "2"}, {"device_ip", "192.168.0.26"}, {"device_port", "16666"}}}}},
               {{"rank_id", "7"},
                {"server_id", "10.0.0.11"},
                {"devices", {{{"device_id", "3"}, {"device_ip", "192.168.0.28"}, {"device_port", "16666"}}}}},
           }},
          {"server_list",
           {
               {{"server_id", "10.0.0.10"},
                {"para_plane_info",
                 {{
                      {"eth0", "192.168.210.2"},
                  },
                  {
                      {"eth1", "192.168.200.2"},
                  },
                  {
                      {"eth2", "192.168.210.2"},
                  },
                  {
                      {"eth3", "192.168.200.2"},
                  }}}

               },
               {{"server_id", "10.0.0.11"},
                {"para_plane_info",
                 {{
                      {"eth0", "192.168.210.3"},
                  },
                  {
                      {"eth1", "192.168.200.3"},
                  },
                  {
                      {"eth2", "192.168.210.3"},
                  },
                  {
                      {"eth3", "192.168.200.3"},
                  }}}

               },
           }}}}}};

static nlohmann::json rank_table_910_2server_4rank_new
    = {{"status", "completed"},
       {"deploy_mode", "lab"},
       {"group_count", "1"},
       {"chip_info", "910"},
       {"board_id", "0x0000"},
       {"para_plane_nic_location", "device"},
       {"para_plane_nic_num", "8"},
       {"para_plane_nic_name", {"eth0", "eth1", "eth2", "eth3"}},
       {"group_list",
        {{{"group_name", ""},
          {"device_num", "8"},
          {"server_num", "2"},
          {"instance_count", "8"},
          {"instance_list",
           {
               {{"rank_id", "0"},
                {"server_id", "10.0.0.10"},
                {"devices", {{{"device_id", "0"}, {"device_ip", "192.168.0.32"}, {"device_port", "16666"}}}}},
               {{"rank_id", "1"},
                {"server_id", "10.0.0.10"},
                {"devices", {{{"device_id", "1"}, {"device_ip", "192.168.0.34"}, {"device_port", "16666"}}}}},
               {{"rank_id", "2"},
                {"server_id", "10.0.0.10"},
                {"devices", {{{"device_id", "2"}, {"device_ip", "192.168.0.36"}, {"device_port", "16666"}}}}},
               {{"rank_id", "3"},
                {"server_id", "10.0.0.10"},
                {"devices", {{{"device_id", "3"}, {"device_ip", "192.168.0.38"}, {"device_port", "16666"}}}}},
               {{"rank_id", "4"},
                {"server_id", "10.0.0.11"},
                {"devices", {{{"device_id", "0"}, {"device_ip", "192.168.0.42"}, {"device_port", "16666"}}}}},
               {{"rank_id", "5"},
                {"server_id", "10.0.0.11"},
                {"devices", {{{"device_id", "1"}, {"device_ip", "192.168.0.44"}, {"device_port", "16666"}}}}},
               {{"rank_id", "6"},
                {"server_id", "10.0.0.11"},
                {"devices", {{{"device_id", "2"}, {"device_ip", "192.168.0.46"}, {"device_port", "16666"}}}}},
               {{"rank_id", "7"},
                {"server_id", "10.0.0.11"},
                {"devices", {{{"device_id", "3"}, {"device_ip", "192.168.0.48"}, {"device_port", "16666"}}}}},
           }},
          {"server_list",
           {
               {{"server_id", "10.0.0.10"},
                {"para_plane_info",
                 {{
                      {"eth0", "192.168.210.2"},
                  },
                  {
                      {"eth1", "192.168.200.2"},
                  },
                  {
                      {"eth2", "192.168.210.2"},
                  },
                  {
                      {"eth3", "192.168.200.2"},
                  }}}

               },
               {{"server_id", "10.0.0.11"},
                {"para_plane_info",
                 {{
                      {"eth0", "192.168.210.3"},
                  },
                  {
                      {"eth1", "192.168.200.3"},
                  },
                  {
                      {"eth2", "192.168.210.3"},
                  },
                  {
                      {"eth3", "192.168.200.3"},
                  }}}

               },
           }}}}}};

// ranktable 910 8p
static nlohmann::json rank_table_950_1server_8rank = nlohmann::json::object(
    {{"version", "2.0"},
     {"rank_count", "4"},
     {"rank_list", nlohmann::json::array(
                       {nlohmann::json::object(
                            {{"rank_id", 0},
                             {"local_id", 0},
                             {"level_list", nlohmann::json::array({nlohmann::json::object(
                                                {{"level", 0},
                                                 {"id", "az0-rack0"},
                                                 {"fabric_type", "INNER"},
                                                 {"rank_addr_type", ""},
                                                 {"rank_addrs", nlohmann::json::array()}})})}}),
                        nlohmann::json::object(
                            {{"rank_id", 1},
                             {"local_id", 1},
                             {"level_list", nlohmann::json::array({nlohmann::json::object(
                                                {{"level", 0},
                                                 {"id", "az0-rack0"},
                                                 {"fabric_type", "INNER"},
                                                 {"rank_addr_type", ""},
                                                 {"rank_addrs", nlohmann::json::array()}})})}}),
                        nlohmann::json::object(
                            {{"rank_id", 2},
                             {"local_id", 2},
                             {"level_list", nlohmann::json::array({nlohmann::json::object(
                                                {{"level", 0},
                                                 {"id", "az0-rack0"},
                                                 {"fabric_type", "INNER"},
                                                 {"rank_addr_type", ""},
                                                 {"rank_addrs", nlohmann::json::array()}})})}}),
                        nlohmann::json::object(
                            {{"rank_id", 3},
                             {"local_id", 3},
                             {"level_list", nlohmann::json::array({nlohmann::json::object(
                                                {{"level", 0},
                                                 {"id", "az0-rack0"},
                                                 {"fabric_type", "INNER"},
                                                 {"rank_addr_type", ""},
                                                 {"rank_addrs", nlohmann::json::array()}})})}})})}});

static nlohmann::json rank_table_1server_8rank
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
          {"device_num", "8"},
          {"server_num", "1"},
          {"instance_count", "8"},
          {"instance_list",
           {{{"rank_id", "0"},
             {"server_id", "10.0.0.10"},
             {"devices", {{{"device_id", "0"}, {"device_ip", "192.168.0.11"}, {"device_port", "16666"}}}}},

            {{"rank_id", "1"},
             {"server_id", "10.0.0.10"},
             {"devices", {{{"device_id", "1"}, {"device_ip", "192.168.0.12"}, {"device_port", "16666"}}}}},
            {{"rank_id", "2"},
             {"server_id", "10.0.0.10"},
             {"devices", {{{"device_id", "2"}, {"device_ip", "192.168.0.13"}, {"device_port", "16666"}}}}},

            {{"rank_id", "3"},
             {"server_id", "10.0.0.10"},
             {"devices", {{{"device_id", "3"}, {"device_ip", "192.168.0.14"}, {"device_port", "16666"}}}}},
            {{"rank_id", "4"},
             {"server_id", "10.0.0.10"},
             {"devices", {{{"device_id", "4"}, {"device_ip", "192.168.0.15"}, {"device_port", "16666"}}}}},

            {{"rank_id", "5"},
             {"server_id", "10.0.0.10"},
             {"devices", {{{"device_id", "5"}, {"device_ip", "192.168.0.16"}, {"device_port", "16666"}}}}},
            {{"rank_id", "6"},
             {"server_id", "10.0.0.10"},
             {"devices", {{{"device_id", "6"}, {"device_ip", "192.168.0.17"}, {"device_port", "16666"}}}}},

            {{"rank_id", "7"},
             {"server_id", "10.0.0.10"},
             {"devices", {{{"device_id", "7"}, {"device_ip", "192.168.0.18"}, {"device_port", "16666"}}}}}}},
          {"server_list",
           {{{"server_id", "10.0.0.10"},
             {"para_plane_info",
              {
                  {
                      {"eth0", "192.168.200.2"},
                  },
                  {
                      {"eth1", "192.168.200.3"},
                  },
                  {
                      {"eth2", "192.168.200.4"},
                  },
                  {
                      {"eth3", "192.168.200.5"},
                  },
                  {
                      {"eth4", "192.168.200.6"},
                  },
                  {
                      {"eth5", "192.168.200.7"},
                  },
                  {
                      {"eth6", "192.168.200.8"},
                  },
                  {
                      {"eth7", "192.168.200.9"},
                  },
              }}

           }}}}}}};

// ranktable 910 2* 8p
static nlohmann::json rank_table_910_2server_8rank
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
                         {"eth2", "192.168.200.2"},
                     },
                     {
                         {"eth3", "192.168.200.2"},
                     },
                     {
                         {"eth4", "192.168.200.2"},
                     },
                     {
                         {"eth5", "192.168.200.2"},
                     },
                     {
                         {"eth6", "192.168.200.2"},
                     },
                     {
                         {"eth7", "192.168.200.2"},
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
                         {"eth1", "192.168.210.3"},
                     },
                     {
                         {"eth2", "192.168.200.3"},
                     },
                     {
                         {"eth3", "192.168.210.3"},
                     },
                     {
                         {"eth4", "192.168.200.2"},
                     },
                     {
                         {"eth5", "192.168.200.2"},
                     },
                     {
                         {"eth6", "192.168.200.2"},
                     },
                     {
                         {"eth7", "192.168.200.2"},
                     },
                 }}

               },

           }}}}}};
static nlohmann::json rank_table_910_2server_8rank_HostNic_new
    = {{"status", "completed"},
       {"version", "1.0"},
       {"server_count", "2"},
       {"server_list",
        {{
             {"server_id", "10.0.0.10"},
             {"host_ip", "192.168.0.12"},
             {"device",
              {
                  {{"rank_id", "0"}, {"device_id", "0"}, {"device_ip", "192.168.0.12"}

                  },
                  {{"rank_id", "1"}, {"device_id", "1"}, {"device_ip", "192.168.1.12"}

                  },
                  {{"rank_id", "2"}, {"device_id", "2"}, {"device_ip", "192.168.2.12"}

                  },
                  {{"rank_id", "3"}, {"device_id", "3"}, {"device_ip", "192.168.3.12"}

                  },
                  {{"rank_id", "4"}, {"device_id", "4"}, {"device_ip", "192.168.4.12"}

                  },
                  {{"rank_id", "5"}, {"device_id", "5"}, {"device_ip", "192.168.5.12"}

                  },
                  {{"rank_id", "6"}, {"device_id", "6"}, {"device_ip", "192.168.6.12"}

                  },
                  {{"rank_id", "7"}, {"device_id", "7"}, {"device_ip", "192.168.7.12"}

                  },
              }},
         },
         {
             {"server_id", "10.0.0.11"},
             {"host_ip", "192.168.0.13"},
             {"device",
              {
                  {{"rank_id", "8"}, {"device_id", "0"}, {"device_ip", "192.168.8.12"}

                  },
                  {{"rank_id", "9"}, {"device_id", "1"}, {"device_ip", "192.168.9.12"}

                  },
                  {{"rank_id", "10"}, {"device_id", "2"}, {"device_ip", "192.168.10.12"}

                  },
                  {{"rank_id", "11"}, {"device_id", "3"}, {"device_ip", "192.168.11.12"}

                  },
                  {{"rank_id", "12"}, {"device_id", "4"}, {"device_ip", "192.168.12.12"}

                  },
                  {{"rank_id", "13"}, {"device_id", "5"}, {"device_ip", "192.168.13.12"}

                  },
                  {{"rank_id", "14"}, {"device_id", "6"}, {"device_ip", "192.168.14.12"}

                  },
                  {{"rank_id", "15"}, {"device_id", "7"}, {"device_ip", "192.168.15.12"}

                  },
              }},
         }}}};
static nlohmann::json rank_table_910_2server_1rank_HostNic_new
    = {{"status", "completed"},
       {"version", "1.0"},
       {"server_count", "2"},
       {"server_list",
        {{
             {"server_id", "10.0.0.10"},
             {"host_ip", "192.168.0.12"},
             {"device",
              {
                  {
                      {"rank_id", "0"},
                      {"device_id", "0"},

                  },
              }},
         },
         {
             {"server_id", "10.0.0.11"},
             {"host_ip", "192.168.0.13"},
             {"device",
              {
                  {
                      {"rank_id", "1"},
                      {"device_id", "0"},

                  },
              }},
         }}}};
static nlohmann::json rank_table_910_1server_1rank_for_multiDie
    = {{"status", "completed"},
       {"version", "1.0"},
       {"server_count", "1"},
       {"server_list",
        {{
            {"server_id", "10.0.0.10"},
            {"device",
             {
                 {
                     {"rank_id", "0"},
                     {"device_id", "0"},

                 },
             }},
        }}}};

static nlohmann::json rank_table_board3000_2server_8rank
    = {{"status", "completed"},
       {"deploy_mode", "lab"},
       {"group_count", "1"},
       {"chip_info", "910"},
       {"board_id", "0x3000"},
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

static nlohmann::json rank_table_910_2server_8rank_4nic
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
               {{"rank_id", "4"}, {"server_id", "10.0.0.10"}, {"devices", {{{"device_id", "4"}, {"device_ip", ""}}}}},

               {{"rank_id", "5"}, {"server_id", "10.0.0.10"}, {"devices", {{{"device_id", "5"}, {"device_ip", ""}}}}},
               {{"rank_id", "6"}, {"server_id", "10.0.0.10"}, {"devices", {{{"device_id", "6"}, {"device_ip", ""}}}}},

               {{"rank_id", "7"}, {"server_id", "10.0.0.10"}, {"devices", {{{"device_id", "7"}, {"device_ip", ""}}}}},
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
               {{"rank_id", "12"}, {"server_id", "10.0.0.11"}, {"devices", {{{"device_id", "4"}, {"device_ip", ""}}}}},

               {{"rank_id", "13"}, {"server_id", "10.0.0.11"}, {"devices", {{{"device_id", "5"}, {"device_ip", ""}}}}},
               {{"rank_id", "14"}, {"server_id", "10.0.0.11"}, {"devices", {{{"device_id", "6"}, {"device_ip", ""}}}}},

               {{"rank_id", "15"}, {"server_id", "10.0.0.11"}, {"devices", {{{"device_id", "7"}, {"device_ip", ""}}}}},
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
                         {"eth2", "192.168.200.2"},
                     },
                     {
                         {"eth3", "192.168.200.2"},
                     },
                     {
                         {"eth4", "192.168.200.2"},
                     },
                     {
                         {"eth5", "192.168.200.2"},
                     },
                     {
                         {"eth6", "192.168.200.2"},
                     },
                     {
                         {"eth7", "192.168.200.2"},
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
                         {"eth1", "192.168.210.3"},
                     },
                     {
                         {"eth2", "192.168.200.3"},
                     },
                     {
                         {"eth3", "192.168.210.3"},
                     },
                     {
                         {"eth4", "192.168.200.2"},
                     },
                     {
                         {"eth5", "192.168.200.2"},
                     },
                     {
                         {"eth6", "192.168.200.2"},
                     },
                     {
                         {"eth7", "192.168.200.2"},
                     },
                 }}

               },

           }}}}}};

static nlohmann::json rank_table_910_2server_8rank_2nic
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
               {{"rank_id", "2"}, {"server_id", "10.0.0.10"}, {"devices", {{{"device_id", "2"}, {"device_ip", ""}}}}},

               {{"rank_id", "3"}, {"server_id", "10.0.0.10"}, {"devices", {{{"device_id", "3"}, {"device_ip", ""}}}}},
               {{"rank_id", "4"}, {"server_id", "10.0.0.10"}, {"devices", {{{"device_id", "4"}, {"device_ip", ""}}}}},

               {{"rank_id", "5"}, {"server_id", "10.0.0.10"}, {"devices", {{{"device_id", "5"}, {"device_ip", ""}}}}},
               {{"rank_id", "6"}, {"server_id", "10.0.0.10"}, {"devices", {{{"device_id", "6"}, {"device_ip", ""}}}}},

               {{"rank_id", "7"}, {"server_id", "10.0.0.10"}, {"devices", {{{"device_id", "7"}, {"device_ip", ""}}}}},
               {{"rank_id", "8"},
                {"server_id", "10.0.0.11"},
                {"devices", {{{"device_id", "0"}, {"device_ip", "192.168.0.20"}}}}},

               {{"rank_id", "9"},
                {"server_id", "10.0.0.11"},
                {"devices", {{{"device_id", "1"}, {"device_ip", "192.168.0.21"}}}}},
               {{"rank_id", "10"}, {"server_id", "10.0.0.11"}, {"devices", {{{"device_id", "2"}, {"device_ip", ""}}}}},

               {{"rank_id", "11"}, {"server_id", "10.0.0.11"}, {"devices", {{{"device_id", "3"}, {"device_ip", ""}}}}},
               {{"rank_id", "12"}, {"server_id", "10.0.0.11"}, {"devices", {{{"device_id", "4"}, {"device_ip", ""}}}}},

               {{"rank_id", "13"}, {"server_id", "10.0.0.11"}, {"devices", {{{"device_id", "5"}, {"device_ip", ""}}}}},
               {{"rank_id", "14"}, {"server_id", "10.0.0.11"}, {"devices", {{{"device_id", "6"}, {"device_ip", ""}}}}},

               {{"rank_id", "15"}, {"server_id", "10.0.0.11"}, {"devices", {{{"device_id", "7"}, {"device_ip", ""}}}}},
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
                         {"eth2", "192.168.200.2"},
                     },
                     {
                         {"eth3", "192.168.200.2"},
                     },
                     {
                         {"eth4", "192.168.200.2"},
                     },
                     {
                         {"eth5", "192.168.200.2"},
                     },
                     {
                         {"eth6", "192.168.200.2"},
                     },
                     {
                         {"eth7", "192.168.200.2"},
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
                         {"eth1", "192.168.210.3"},
                     },
                     {
                         {"eth2", "192.168.200.3"},
                     },
                     {
                         {"eth3", "192.168.210.3"},
                     },
                     {
                         {"eth4", "192.168.200.2"},
                     },
                     {
                         {"eth5", "192.168.200.2"},
                     },
                     {
                         {"eth6", "192.168.200.2"},
                     },
                     {
                         {"eth7", "192.168.200.2"},
                     },
                 }}

               },

           }}}}}};
static nlohmann::json rank_table_910_2server_8rank_1nic
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

               {{"rank_id", "1"}, {"server_id", "10.0.0.10"}, {"devices", {{{"device_id", "1"}, {"device_ip", ""}}}}},
               {{"rank_id", "2"}, {"server_id", "10.0.0.10"}, {"devices", {{{"device_id", "2"}, {"device_ip", ""}}}}},

               {{"rank_id", "3"}, {"server_id", "10.0.0.10"}, {"devices", {{{"device_id", "3"}, {"device_ip", ""}}}}},
               {{"rank_id", "4"}, {"server_id", "10.0.0.10"}, {"devices", {{{"device_id", "4"}, {"device_ip", ""}}}}},

               {{"rank_id", "5"}, {"server_id", "10.0.0.10"}, {"devices", {{{"device_id", "5"}, {"device_ip", ""}}}}},
               {{"rank_id", "6"}, {"server_id", "10.0.0.10"}, {"devices", {{{"device_id", "6"}, {"device_ip", ""}}}}},

               {{"rank_id", "7"}, {"server_id", "10.0.0.10"}, {"devices", {{{"device_id", "7"}, {"device_ip", ""}}}}},
               {{"rank_id", "8"},
                {"server_id", "10.0.0.11"},
                {"devices", {{{"device_id", "0"}, {"device_ip", "192.168.0.20"}}}}},

               {{"rank_id", "9"}, {"server_id", "10.0.0.11"}, {"devices", {{{"device_id", "1"}, {"device_ip", ""}}}}},
               {{"rank_id", "10"}, {"server_id", "10.0.0.11"}, {"devices", {{{"device_id", "2"}, {"device_ip", ""}}}}},

               {{"rank_id", "11"}, {"server_id", "10.0.0.11"}, {"devices", {{{"device_id", "3"}, {"device_ip", ""}}}}},
               {{"rank_id", "12"}, {"server_id", "10.0.0.11"}, {"devices", {{{"device_id", "4"}, {"device_ip", ""}}}}},

               {{"rank_id", "13"}, {"server_id", "10.0.0.11"}, {"devices", {{{"device_id", "5"}, {"device_ip", ""}}}}},
               {{"rank_id", "14"}, {"server_id", "10.0.0.11"}, {"devices", {{{"device_id", "6"}, {"device_ip", ""}}}}},

               {{"rank_id", "15"}, {"server_id", "10.0.0.11"}, {"devices", {{{"device_id", "7"}, {"device_ip", ""}}}}},
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
                         {"eth2", "192.168.200.2"},
                     },
                     {
                         {"eth3", "192.168.200.2"},
                     },
                     {
                         {"eth4", "192.168.200.2"},
                     },
                     {
                         {"eth5", "192.168.200.2"},
                     },
                     {
                         {"eth6", "192.168.200.2"},
                     },
                     {
                         {"eth7", "192.168.200.2"},
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
                         {"eth1", "192.168.210.3"},
                     },
                     {
                         {"eth2", "192.168.200.3"},
                     },
                     {
                         {"eth3", "192.168.210.3"},
                     },
                     {
                         {"eth4", "192.168.200.2"},
                     },
                     {
                         {"eth5", "192.168.200.2"},
                     },
                     {
                         {"eth6", "192.168.200.2"},
                     },
                     {
                         {"eth7", "192.168.200.2"},
                     },
                 }}

               },

           }}}}}};

static nlohmann::json rank_table_910_2server_8rank_1nic_serverIdx_descend
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
                {"server_id", "10.0.0.11"},
                {"devices", {{{"device_id", "0"}, {"device_ip", "192.168.0.11"}}}}},

               {{"rank_id", "1"}, {"server_id", "10.0.0.11"}, {"devices", {{{"device_id", "1"}, {"device_ip", ""}}}}},
               {{"rank_id", "2"}, {"server_id", "10.0.0.11"}, {"devices", {{{"device_id", "2"}, {"device_ip", ""}}}}},

               {{"rank_id", "3"}, {"server_id", "10.0.0.11"}, {"devices", {{{"device_id", "3"}, {"device_ip", ""}}}}},
               {{"rank_id", "4"}, {"server_id", "10.0.0.11"}, {"devices", {{{"device_id", "4"}, {"device_ip", ""}}}}},

               {{"rank_id", "5"}, {"server_id", "10.0.0.11"}, {"devices", {{{"device_id", "5"}, {"device_ip", ""}}}}},
               {{"rank_id", "6"}, {"server_id", "10.0.0.11"}, {"devices", {{{"device_id", "6"}, {"device_ip", ""}}}}},

               {{"rank_id", "7"}, {"server_id", "10.0.0.11"}, {"devices", {{{"device_id", "7"}, {"device_ip", ""}}}}},
               {{"rank_id", "8"},
                {"server_id", "10.0.0.10"},
                {"devices", {{{"device_id", "0"}, {"device_ip", "192.168.0.20"}}}}},

               {{"rank_id", "9"}, {"server_id", "10.0.0.10"}, {"devices", {{{"device_id", "1"}, {"device_ip", ""}}}}},
               {{"rank_id", "10"}, {"server_id", "10.0.0.10"}, {"devices", {{{"device_id", "2"}, {"device_ip", ""}}}}},

               {{"rank_id", "11"}, {"server_id", "10.0.0.10"}, {"devices", {{{"device_id", "3"}, {"device_ip", ""}}}}},
               {{"rank_id", "12"}, {"server_id", "10.0.0.10"}, {"devices", {{{"device_id", "4"}, {"device_ip", ""}}}}},

               {{"rank_id", "13"}, {"server_id", "10.0.0.10"}, {"devices", {{{"device_id", "5"}, {"device_ip", ""}}}}},
               {{"rank_id", "14"}, {"server_id", "10.0.0.10"}, {"devices", {{{"device_id", "6"}, {"device_ip", ""}}}}},

               {{"rank_id", "15"}, {"server_id", "10.0.0.10"}, {"devices", {{{"device_id", "7"}, {"device_ip", ""}}}}},
           }},
          {"server_list",
           {
               {{"server_id", "10.0.0.11"},
                {"para_plane_info",
                 {
                     {
                         {"eth0", "192.168.200.2"},
                     },
                     {
                         {"eth1", "192.168.200.2"},
                     },
                     {
                         {"eth2", "192.168.200.2"},
                     },
                     {
                         {"eth3", "192.168.200.2"},
                     },
                     {
                         {"eth4", "192.168.200.2"},
                     },
                     {
                         {"eth5", "192.168.200.2"},
                     },
                     {
                         {"eth6", "192.168.200.2"},
                     },
                     {
                         {"eth7", "192.168.200.2"},
                     },
                 }}

               },
               {{"server_id", "10.0.0.10"},
                {"para_plane_info",
                 {
                     {
                         {"eth0", "192.168.210.3"},
                     },
                     {
                         {"eth1", "192.168.210.3"},
                     },
                     {
                         {"eth2", "192.168.200.3"},
                     },
                     {
                         {"eth3", "192.168.210.3"},
                     },
                     {
                         {"eth4", "192.168.200.2"},
                     },
                     {
                         {"eth5", "192.168.200.2"},
                     },
                     {
                         {"eth6", "192.168.200.2"},
                     },
                     {
                         {"eth7", "192.168.200.2"},
                     },
                 }}

               },

           }}}}}};

static const std::string rank_table_2pg_2server
    = "{\"collective_id\":\"192.168.3.3-9527-0001\",\"master_ip\":\"192.168.200.1\",\"master_port\":\"18000\","
      "\"node_list\":[{\"node_addr\":\"192.168.200.1\",\"ranks\":[{\"rank_id\":\"0\",\"device_id\":\"0\"},{\"rank_id\":"
      "\"1\",\"device_id\":\"1\"}]},"
      "{\"node_addr\":\"192.168.200.2\",\"ranks\":[{\"rank_id\":\"2\",\"device_id\":\"0\"},{\"rank_id\":\"3\",\"device_"
      "id\":\"1\"}]}],\"status\":\"completed\","
      "\"version\":\"1.1\"}";

static const std::string rank_table_2pg_1server
    = "{\"collective_id\":\"192.168.3.3-9527-0001\",\"master_ip\":\"192.168.201.1\",\"master_port\":\"18000\","
      "\"node_list\":[{\"node_addr\":\"192.168.201.1\",\"ranks\":[{\"rank_id\":\"0\",\"device_id\":\"0\"},"
      "{\"rank_id\":\"1\",\"device_id\":\"1\"}]}],\"status\":\"completed\",\"version\":\"1.1\"}";

static const std::string rank_table_2rank_1server_heterog
    = "{\"collective_id\":\"192.168.3.3-9527-0001\","
      "\"node_list\":[{\"node_addr\":\"192.168.202.1\",\"ranks\":[{\"rank_id\":\"0\",\"device_id\":\"0\"},"
      "{\"rank_id\":\"1\",\"device_id\":\"-1\"}]}],\"status\":\"completed\",\"version\":\"1.1\"}";

static const std::string rank_table_1ps_1worker
    = "{\"collective_id\":\"192.168.3.3-9527-0001\",\"node_list\":[{\"node_addr\":\"10.78.130.93\",\"ranks\":"
      "[{\"device_id\":\"0\",\"port\":\"60008\",\"rank_id\":\"0\",\"rank_ip\":\"192.101.210.100\"},"
      "{\"device_id\":\"-1\",\"port\":\"5555\",\"rank_id\":\"1\"}]}],\"status\":\"completed\",\"version\":\"1.1\"}";

static const std::string rank_table_1ps_1worker_1
    = "{\"collective_id\":\"192.168.3.3-9527-0001\",\"node_list\":[{\"node_addr\":\"10.78.130.94\",\"ranks\":"
      "[{\"device_id\":\"0\",\"port\":\"60009\",\"rank_id\":\"0\",\"rank_ip\":\"192.101.210.101\"},"
      "{\"device_id\":\"-1\",\"port\":\"5556\",\"rank_id\":\"1\"}]}],\"status\":\"completed\",\"version\":\"1.1\"}";
