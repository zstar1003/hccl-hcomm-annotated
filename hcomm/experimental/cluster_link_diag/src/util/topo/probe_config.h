/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <map>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

using json = nlohmann::json;

struct ProbeTopoTracertConfig {
    std::map<std::string, std::vector<std::string>> scope;
    int sport_begin = 49152;
    int sport_count = 1;
    int tree_probe_sport_count = 1;
    bool topology_optimized = true;
    bool l2_path_aware = true;
    std::string allpath_output = "allpath.json";
    std::string l2_path_output = "l2_fullmesh_path.json";
    std::string output_subdir;
};

struct ProbeControllerPingpongConfig {
    std::map<std::string, std::vector<std::string>> scope;
    int times = 50;
    int turns = 1000;
    int payload_len = 12;
    int interval_ms = 1;
};

struct ProbeRuntimeConfig {
    bool enabled = false;
    ProbeTopoTracertConfig probe_topo_tracert;
    ProbeControllerPingpongConfig probe_controller_pingpong;
};

ProbeRuntimeConfig load_probe_runtime_config(const std::string& json_path);
json make_control_topo_json(const std::map<std::string, std::vector<std::string>>& scope);
std::string format_probe_scope(const std::map<std::string, std::vector<std::string>>& scope);
std::string make_probe_topo_json_path(const std::string& project_path, const ProbeRuntimeConfig& config);
std::string make_probe_topo_lldp_json_path(const std::string& project_path, const ProbeRuntimeConfig& config);
std::string make_allpath_json_path(const std::string& project_path, const ProbeRuntimeConfig& config);
std::string make_l2_path_json_path(const std::string& project_path, const ProbeRuntimeConfig& config);
