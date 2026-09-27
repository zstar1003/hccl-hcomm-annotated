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

#include "topoinfo_ranktableParser_pub.h"

static nlohmann::json g_rank_table_610_8rank_1server
    = {{"status", "completed"},
       {"deploy_mode", "lab"},
       {"group_count", "1"},
       {"chip_info", "310P3"},
       {"board_id", "0x2000"},
       {"para_plane_nic_location", "device"},
       {"para_plane_nic_num", "1"},
       {"para_plane_nic_name", {"eth0"}},
       {"group_list",
        {{{"group_name", ""},
          {"device_num", "8"},
          {"server_num", "1"},
          {"instance_count", "8"},
          {"instance_list",
           {{{"rank_id", "0"},
             {"server_id", "10.0.0.10"},
             {"devices", {{{"device_id", "0"}, {"device_ip", "192.168.0.11"}}}}},

            {{"rank_id", "1"},
             {"server_id", "10.0.0.10"},
             {"devices", {{{"device_id", "2"}, {"device_ip", "192.168.0.12"}}}}},
            {{"rank_id", "2"},
             {"server_id", "10.0.0.10"},
             {"devices", {{{"device_id", "4"}, {"device_ip", "192.168.0.13"}}}}},

            {{"rank_id", "3"},
             {"server_id", "10.0.0.10"},
             {"devices", {{{"device_id", "6"}, {"device_ip", "192.168.0.14"}}}}},
            {{"rank_id", "4"},
             {"server_id", "10.0.0.10"},
             {"devices", {{{"device_id", "8"}, {"device_ip", "192.168.0.15"}}}}},

            {{"rank_id", "5"},
             {"server_id", "10.0.0.10"},
             {"devices", {{{"device_id", "10"}, {"device_ip", "192.168.0.16"}}}}},
            {{"rank_id", "6"},
             {"server_id", "10.0.0.10"},
             {"devices", {{{"device_id", "12"}, {"device_ip", "192.168.0.17"}}}}},

            {{"rank_id", "7"},
             {"server_id", "10.0.0.10"},
             {"devices", {{{"device_id", "14"}, {"device_ip", "192.168.0.18"}}}}}}},
          {"server_list",
           {{{"server_id", "10.0.0.10"},
             {"para_plane_info",
              {{
                  {"eth0", "192.168.200.2"},
              }}}}}}}}}};

static nlohmann::json g_rank_table_610_5rank_1server
    = {{"status", "completed"},
       {"deploy_mode", "lab"},
       {"group_count", "1"},
       {"chip_info", "310P3"},
       {"board_id", "0x2000"},
       {"para_plane_nic_location", "device"},
       {"para_plane_nic_num", "1"},
       {"para_plane_nic_name", {"eth0"}},
       {"group_list",
        {{{"group_name", ""},
          {"device_num", "5"},
          {"server_num", "1"},
          {"instance_count", "5"},
          {"instance_list",
           {{{"rank_id", "0"},
             {"server_id", "10.0.0.10"},
             {"devices", {{{"device_id", "0"}, {"device_ip", "192.168.0.11"}}}}},

            {{"rank_id", "1"},
             {"server_id", "10.0.0.10"},
             {"devices", {{{"device_id", "2"}, {"device_ip", "192.168.0.12"}}}}},
            {{"rank_id", "2"},
             {"server_id", "10.0.0.10"},
             {"devices", {{{"device_id", "4"}, {"device_ip", "192.168.0.13"}}}}},

            {{"rank_id", "3"},
             {"server_id", "10.0.0.10"},
             {"devices", {{{"device_id", "6"}, {"device_ip", "192.168.0.14"}}}}},
            {{"rank_id", "4"},
             {"server_id", "10.0.0.10"},
             {"devices", {{{"device_id", "8"}, {"device_ip", "192.168.0.15"}}}}}}},
          {"server_list",
           {{{"server_id", "10.0.0.10"},
             {"para_plane_info",
              {{
                  {"eth0", "192.168.200.2"},
              }}}}}}}}}};

static nlohmann::json g_rank_table_610_4rank_1server
    = {{"status", "completed"},
       {"deploy_mode", "lab"},
       {"group_count", "1"},
       {"chip_info", "310P3"},
       {"board_id", "0x2000"},
       {"para_plane_nic_location", "device"},
       {"para_plane_nic_num", "1"},
       {"para_plane_nic_name", {"eth0"}},
       {"group_list",
        {{{"group_name", ""},
          {"device_num", "4"},
          {"server_num", "1"},
          {"instance_count", "4"},
          {"instance_list",
           {{{"rank_id", "0"},
             {"server_id", "10.0.0.10"},
             {"devices", {{{"device_id", "0"}, {"device_ip", "192.168.0.11"}}}}},

            {{"rank_id", "1"},
             {"server_id", "10.0.0.10"},
             {"devices", {{{"device_id", "2"}, {"device_ip", "192.168.0.12"}}}}},
            {{"rank_id", "2"},
             {"server_id", "10.0.0.10"},
             {"devices", {{{"device_id", "4"}, {"device_ip", "192.168.0.13"}}}}},

            {{"rank_id", "3"},
             {"server_id", "10.0.0.10"},
             {"devices", {{{"device_id", "6"}, {"device_ip", "192.168.0.14"}}}}}}},
          {"server_list",
           {{{"server_id", "10.0.0.10"},
             {"para_plane_info",
              {{
                  {"eth0", "192.168.200.2"},
              }}}}}}}}}};

static nlohmann::json g_rank_table_610_3rank_1server
    = {{"status", "completed"},
       {"deploy_mode", "lab"},
       {"group_count", "1"},
       {"chip_info", "310P3"},
       {"board_id", "0x2000"},
       {"para_plane_nic_location", "device"},
       {"para_plane_nic_num", "1"},
       {"para_plane_nic_name", {"eth0"}},
       {"group_list",
        {{{"group_name", ""},
          {"device_num", "3"},
          {"server_num", "1"},
          {"instance_count", "3"},
          {"instance_list",
           {{{"rank_id", "0"},
             {"server_id", "10.0.0.10"},
             {"devices", {{{"device_id", "0"}, {"device_ip", "192.168.0.11"}}}}},

            {{"rank_id", "1"},
             {"server_id", "10.0.0.10"},
             {"devices", {{{"device_id", "2"}, {"device_ip", "192.168.0.12"}}}}},
            {{"rank_id", "2"},
             {"server_id", "10.0.0.10"},
             {"devices", {{{"device_id", "4"}, {"device_ip", "192.168.0.13"}}}}}}},
          {"server_list",
           {{{"server_id", "10.0.0.10"},
             {"para_plane_info",
              {{
                  {"eth0", "192.168.200.2"},
              }}}}}}}}}};

static nlohmann::json g_rank_table_610_2rank_1server
    = {{"status", "completed"},
       {"deploy_mode", "lab"},
       {"group_count", "1"},
       {"chip_info", "310P3"},
       {"board_id", "0x2000"},
       {"para_plane_nic_location", "device"},
       {"para_plane_nic_num", "1"},
       {"para_plane_nic_name", {"eth0"}},
       {"group_list",
        {{{"group_name", ""},
          {"device_num", "2"},
          {"server_num", "1"},
          {"instance_count", "2"},
          {"instance_list",
           {{{"rank_id", "0"},
             {"server_id", "10.0.0.10"},
             {"devices", {{{"device_id", "0"}, {"device_ip", "192.168.0.11"}}}}},

            {{"rank_id", "1"},
             {"server_id", "10.0.0.10"},
             {"devices", {{{"device_id", "2"}, {"device_ip", "192.168.0.12"}}}}}}},
          {"server_list",
           {{{"server_id", "10.0.0.10"},
             {"para_plane_info",
              {{
                  {"eth0", "192.168.200.2"},
              }}}}}}}}}};
