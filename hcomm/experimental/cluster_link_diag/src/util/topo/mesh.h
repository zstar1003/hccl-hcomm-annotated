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
#include <fstream>
#include <vector>
#include <map>
#include <string>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

struct Mesh {
    int level;
    std::string name;
    std::map<std::string, std::vector<std::tuple<std::string, std::string, int>>> kid_to_link;
    std::map<std::string, std::map<std::string, std::vector<std::tuple<std::string, std::string, int>>>>
        kid_to_kid_link;

    Mesh(
        int level, const std::string& name,
        const std::map<std::string, std::vector<std::tuple<std::string, std::string, int>>>& kid_to_link = {},
        const std::map<std::string, std::map<std::string, std::vector<std::tuple<std::string, std::string, int>>>>&
            kid_to_kid_link
        = {});

    Mesh(const json& j);
    Mesh(const std::string& json_path);
    Mesh() = default;

    static Mesh from_json(const json& j);
    static Mesh from_json(const std::string& json_path);
    json to_json() const;
    void to_json(const std::string& json_path);

    void add_kid(const std::string& kid);
    int add_kid_link(const std::string& kid, const std::string& ip, bool near_ip, int& global_index);
    // near_ip is false, ip should belong to kid2
    int add_kid_to_kid_link(const std::string& kid1, const std::string& kid2, const std::string& ip, int& global_index);

    const std::vector<std::string>& get_kids() const;

    static int get_unfinished_link();

    std::vector<std::string> get_ring();

protected:
    int add_kid_to_kid_link(
        const std::string& kid1, const std::string& kid2, const std::string& ip, bool near_ip, const int global_index);
    static const std::vector<u_int32_t> SUBNET_MASK_LEN;
    u_int32_t ip_to_int(const std::string& ip) const
    {
        u_int32_t res = 0;
        u_int32_t num = 0;
        for (auto c : ip) {
            if (c == '.') {
                res = (res << 8) + num;
                num = 0;
            } else {
                num = num * 10 + (c - '0');
            }
        }
        res = (res << 8) + num;
        return res;
    }
    bool same_subnet(const std::string& ip1, const std::string& ip2) const
    {
        u_int32_t SUBNET_MASK = (0xFFFFFFFF << (32 - SUBNET_MASK_LEN[level])) & 0xFFFFFFFF;
        u_int32_t ip1_int = ip_to_int(ip1);
        u_int32_t ip2_int = ip_to_int(ip2);
        bool res = (ip1_int & SUBNET_MASK) == (ip2_int & SUBNET_MASK);
        return res;
    }

    bool
    recursive_search_ring(const std::string kid, std::vector<std::string>& ring, std::map<std::string, bool>& visited);

private:
    mutable std::vector<std::string> _kids;
};
