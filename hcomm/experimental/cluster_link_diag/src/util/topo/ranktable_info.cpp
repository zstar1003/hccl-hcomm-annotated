/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ranktable_info.h"

void RanktableInfo::set_server_ip(const json& j)
{
    try {
        auto& control_topo = j["control_topo"];
        int n = std::min(server_list.size(), control_topo.size());
        for (int i = 0; i < n; i++) {
            server_list[i].server_ip = control_topo[i].front();
        }
    } catch (const std::exception& e) {
        server_list.clear();
        throw std::runtime_error("[RanktableInfo]json invalid");
    }
}

void RanktableInfo::set_server_ip(const std::string& json_path)
{
    std::ifstream ifs(json_path);
    if (!ifs.is_open()) {
        throw std::runtime_error("[RanktableInfo]json unavailable");
    }
    json j = json::parse(ifs);
    set_server_ip(j);
}

void RanktableInfo::set_server_ip(const std::vector<std::string>& server_ips)
{
    if (server_ips.size() != server_list.size()) {
        throw std::runtime_error("[RanktableInfo]set_server_ip error, server_ips size != server_list size");
    }
    int n = server_list.size();
    for (int i = 0; i < n; i++) {
        server_list[i].server_ip = server_ips[i];
    }
}
