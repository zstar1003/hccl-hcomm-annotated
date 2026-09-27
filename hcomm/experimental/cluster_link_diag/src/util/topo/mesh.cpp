/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "mesh.h"

const std::vector<u_int32_t> Mesh::SUBNET_MASK_LEN = {24, 31};
// Mesh 构造函数
Mesh::Mesh(
    int level, const std::string& name,
    const std::map<std::string, std::vector<std::tuple<std::string, std::string, int>>>& kid_to_link,
    const std::map<std::string, std::map<std::string, std::vector<std::tuple<std::string, std::string, int>>>>&
        kid_to_kid_link)
    : level(level),
      name(name),
      kid_to_link(kid_to_link),
      kid_to_kid_link(kid_to_kid_link)
{}

Mesh::Mesh(const json& j) { *this = from_json(j); }

Mesh::Mesh(const std::string& json_path) { *this = from_json(json_path); }

// JSON 转换方法
Mesh Mesh::from_json(const json& j)
{
    std::map<std::string, std::map<std::string, std::vector<std::tuple<std::string, std::string, int>>>>
        json_kid_to_kid_link;
    if (j.contains("kid_to_kid_link")) {
        json_kid_to_kid_link
            = j.at("kid_to_kid_link")
                  .get<std::map<
                      std::string, std::map<std::string, std::vector<std::tuple<std::string, std::string, int>>>>>();
    }
    return Mesh{
        j.at("level").get<int>(), j.at("name").get<std::string>(),
        j.at("kid_to_link").get<std::map<std::string, std::vector<std::tuple<std::string, std::string, int>>>>(),
        json_kid_to_kid_link};
}

Mesh Mesh::from_json(const std::string& json_path)
{
    std::ifstream ifs(json_path);
    if (!ifs.is_open()) {
        throw std::runtime_error("[ControlTopo]json unavailable");
    }
    json j = json::parse(ifs);
    return from_json(j);
}

json Mesh::to_json() const
{
    auto res = json{{"level", level}, {"name", name}, {"kid_to_link", kid_to_link}};
    if (!kid_to_kid_link.empty()) {
        res["kid_to_kid_link"] = kid_to_kid_link;
    }
    return res;
}

void Mesh::to_json(const std::string& json_path)
{
    std::ofstream ofs(json_path);
    if (!ofs.is_open()) {
        throw std::runtime_error("[Mesh]json unavailable");
    }
    ofs << to_json().dump(2);
}

void Mesh::add_kid(const std::string& kid)
{
    kid_to_link.emplace(kid, std::vector<std::tuple<std::string, std::string, int>>());
}

static int unfinished_link = 0;

bool try_match_existing_link(
    std::vector<std::tuple<std::string, std::string, int>>& links, const std::string& ip, bool near_ip, int& link_index)
{
    for (auto& link : links) {
        auto& first_ip = std::get<0>(link);
        auto& second_ip = std::get<1>(link);
        link_index = std::get<2>(link);
        if (ip == first_ip || ip == second_ip) {
            return true;
        }
        if (!near_ip && same_subnet(ip, first_ip) && (second_ip == "")) {
            second_ip = ip;
            unfinished_link--;
            return true;
        }
        if (near_ip && same_subnet(ip, second_ip) && (first_ip == "")) {
            first_ip = ip;
            unfinished_link--;
            return true;
        }
    }
    return false;
}

int Mesh::add_kid_link(const std::string& kid, const std::string& ip, bool near_ip, int& global_index)
{
    if (ip.empty()) {
        return 0;
    }
    if (kid_to_link.find(kid) == kid_to_link.end()) {
        throw std::runtime_error("[Mesh]kid not exist");
    }
    int matched_link_index = 0;
    if (try_match_existing_link(kid_to_link[kid], ip, near_ip, matched_link_index)) {
        return matched_link_index;
    }
    auto link = std::tuple<std::string, std::string, int>("", "", global_index++);
    auto& first_ip = std::get<0>(link);
    auto& second_ip = std::get<1>(link);
    auto link_index = std::get<2>(link);
    (near_ip ? first_ip : second_ip) = ip;
    kid_to_link[kid].emplace_back(link);
    unfinished_link++;
    return link_index;
}

int Mesh::add_kid_to_kid_link(
    const std::string& kid1, const std::string& kid2, const std::string& ip, bool near_ip, const int global_index)
{
    if (ip.empty()) {
        return 0;
    }
    if (kid_to_kid_link.find(kid1) == kid_to_kid_link.end()) {
        kid_to_kid_link[kid1] = std::map<std::string, std::vector<std::tuple<std::string, std::string, int>>>();
    }
    if (kid_to_kid_link[kid1].find(kid2) == kid_to_kid_link[kid1].end()) {
        kid_to_kid_link[kid1][kid2] = std::vector<std::tuple<std::string, std::string, int>>();
    }
    int matched_link_index = 0;
    if (try_match_existing_link(kid_to_kid_link[kid1][kid2], ip, near_ip, matched_link_index)) {
        return matched_link_index;
    }
    auto link = std::tuple<std::string, std::string, int>("", "", global_index);
    auto& first_ip = std::get<0>(link);
    auto& second_ip = std::get<1>(link);
    auto link_index = std::get<2>(link);
    (near_ip ? first_ip : second_ip) = ip;
    kid_to_kid_link[kid1][kid2].emplace_back(link);
    unfinished_link++;
    return link_index;
}

int Mesh::add_kid_to_kid_link(
    const std::string& kid1, const std::string& kid2, const std::string& ip, int& global_index)
{
    int res1 = add_kid_to_kid_link(kid1, kid2, ip, false, global_index);
    int res2 = add_kid_to_kid_link(kid2, kid1, ip, true, global_index);
    if (res1 != res2) {
        throw std::runtime_error("[Mesh]add_kid_to_kid_link error, link pair index not equal");
    }
    if (res1 == global_index) {
        global_index++;
    }
    return global_index;
}

const std::vector<std::string>& Mesh::get_kids() const
{
    if (_kids.empty()) {
        for (auto& kid_link : kid_to_link) {
            _kids.emplace_back(kid_link.first);
        }
    }
    return _kids;
}

int Mesh::get_unfinished_link() { return unfinished_link; }

bool Mesh::recursive_search_ring(
    const std::string kid, std::vector<std::string>& ring, std::map<std::string, bool>& visited)
{
    auto all_kids = get_kids();
    int n = all_kids.size();

    ring.emplace_back(kid);
    visited[kid] = true;

    std::vector<std::string> next_jump_choice;

    for (auto& next_jump : all_kids) {
        if (ring.size() == n && next_jump == ring[0]) {
            ring.emplace_back(next_jump);
            return true;
        }
        if (visited[next_jump]) {
            continue;
        }
        if (kid_to_kid_link.find(kid) == kid_to_kid_link.end()
            || kid_to_kid_link[kid].find(next_jump) == kid_to_kid_link[kid].end()) {
            next_jump_choice.emplace_back(next_jump);
        }
    }
    for (auto& next_jump : next_jump_choice) {
        if (recursive_search_ring(next_jump, ring, visited)) {
            return true;
        }
    }

    ring.pop_back();
    visited[kid] = false;
    return false;
}

std::vector<std::string> Mesh::get_ring()
{
    auto all_kids = get_kids();
    std::vector<std::string> ring;
    std::map<std::string, bool> visited;
    for (auto& kid : all_kids) {
        visited[kid] = false;
    }
    bool res = recursive_search_ring(all_kids[0], ring, visited);
    if (!res) {
        throw std::runtime_error("[Mesh]get_ring error, no ring found");
    }
    return ring;
}
