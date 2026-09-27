/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "control_topo.h"

#include <iostream>
#include <string>
#include <regex>
#include <vector>
#include <sstream>
#include <cctype>

std::map<std::string, int> ControlTopo::name_repeat;

std::string clean_ip(const std::string& ip)
{
    if (ip == "localhost") {
        return ip;
    }
    // 正则表达式匹配 IPv4 地址（四个 0-255 的数字，用点分隔）
    std::regex ipv4_pattern(
        R"((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)(?!\d))");

    std::smatch match;
    if (std::regex_search(ip, match, ipv4_pattern)) {
        return match.str(0); // 返回第一个匹配的合法 IP
    }
    std::cout << "[Warning]Invalid IP: " << ip << std::endl;
    return ip; // 没有找到合法 IP 返回空字符串
}

void ControlTopo::recursive_cache(const ControlTopo::control_tree& tree, const std::vector<std::string> path) const
{
    if (tree.kids.empty()) {
        cache[tree.ip] = path.empty() ? std::vector<std::string>{clean_ip(tree.ip)} : path;
        return;
    }

    auto route = std::vector<std::string>{clean_ip(tree.ip)};
    route.insert(route.end(), path.begin(), path.end());
    cache[tree.ip] = route;
    for (auto& kid : tree.kids) {
        recursive_cache(kid, route);
    }
}

std::vector<std::string> ControlTopo::route_to_device(const std::string& device_id_or_ip) const
{
    if (cache.empty()) {
        for (auto& kid : control_tree_root.kids) {
            recursive_cache(kid);
        }
    }
    if (cache.find(device_id_or_ip) != cache.end()) {
        return cache[device_id_or_ip];
    } else {
        throw std::runtime_error("[ControlTopo]device_id or ip not found");
    }
    return {};
}
