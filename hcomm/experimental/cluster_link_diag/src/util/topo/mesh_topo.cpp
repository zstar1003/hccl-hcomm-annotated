/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>

#include "mesh_topo.h"
#include <limits>
#include <regex>

MeshTopo::MeshTopo(const ControlTopo& control_topo) {}

MeshTopo::MeshTopo(const std::map<std::string, Mesh, MeshComparator>& mesh_map, const std::string& root_mesh)
    : mesh_map(mesh_map),
      root_mesh(root_mesh)
{}

MeshTopo::MeshTopo(const json& j) { *this = from_json(j); }

MeshTopo::MeshTopo(const std::string& json_path) { *this = from_json(json_path); }

// JSON 转换方法
MeshTopo MeshTopo::from_json(const json& j)
{
    MeshTopo topo;
    topo.mesh_map = j.at("mesh_map").get<std::map<std::string, Mesh, MeshComparator>>();
    topo.root_mesh = j.at("root_mesh").get<std::string>();
    if (j.contains("tracert_res_id")) {
        topo.tracert_res_id
            = j.at("tracert_res_id").get<std::map<std::string, std::map<std::string, std::vector<std::vector<int>>>>>();
    }
    return topo;
}

MeshTopo MeshTopo::from_json(const std::string& json_path)
{
    std::ifstream ifs(json_path);
    if (!ifs.is_open()) {
        throw std::runtime_error("[ControlTopo] json unavailable");
    }
    json j = json::parse(ifs);
    return from_json(j);
}

json MeshTopo::to_json() const
{
    return json{{"mesh_map", mesh_map}, {"root_mesh", root_mesh}, {"tracert_res_id", tracert_res_id}};
}

void MeshTopo::to_json(const std::string& json_path)
{
    std::ofstream ofs(json_path);
    if (!ofs.is_open()) {
        throw std::runtime_error("[MeshTopo] json path unavailable");
    }
    ofs << to_json().dump(2);
}

void MeshTopo::add_mesh(const Mesh& mesh)
{
    if (mesh_map.find(mesh.name) != mesh_map.end()) {
        throw std::runtime_error("[MeshTopo]mesh name already exists");
    }
    mesh_map[mesh.name] = mesh;
}

Mesh& MeshTopo::get_mesh(const std::string& mesh_name)
{
    if (mesh_map.find(mesh_name) == mesh_map.end()) {
        throw std::runtime_error("[MeshTopo]mesh not found");
    }
    return mesh_map[mesh_name];
}

std::string MeshTopo::get_mesh_name(int level, int index)
{
    std::ostringstream mesh_name;
    mesh_name << "TOPO[" << level << "," << index << "]";
    return mesh_name.str();
}

int MeshTopo::parse_mesh_name(const std::string& mesh_name)
{
    static const std::regex mesh_name_pattern(R"(TOPO\[(\d+),(\d+)\])");
    std::smatch match;
    if (std::regex_match(mesh_name, match, mesh_name_pattern)) {
        const int level = std::stoi(match[1].str());
        const int index = std::stoi(match[2].str());
        return level << MESH_INDEX_LEN | index;
    }
    return std::numeric_limits<int>::max();
}

std::vector<std::string> MeshTopo::get_leaf_kids(const Mesh& mesh) const
{
    auto kids = mesh.get_kids();
    std::vector<std::string> leaf_kids;
    for (auto kid : kids) {
        while (mesh_map.find(kid) != mesh_map.end()) {
            auto it = mesh_map.at(kid).kid_to_link.begin();
            auto new_kid = it->first;
            if (it++ != mesh_map.at(kid).kid_to_link.end()) {
                new_kid = it->first;
            }
            kid = new_kid;
        }
        leaf_kids.emplace_back(kid);
    }
    return leaf_kids;
}

std::vector<std::string> MeshTopo::get_leaf_kids(const std::string& mesh_name) const
{
    if (mesh_map.find(mesh_name) == mesh_map.end()) {
        return std::vector<std::string>{mesh_name};
    }
    return get_leaf_kids(mesh_map.at(mesh_name));
}

void MeshTopo::add_kid(const std::string& mesh_name, const std::string& kid_name)
{
    get_mesh(mesh_name).add_kid(kid_name);
    mesh_father[kid_name] = mesh_name;
}

std::string MeshTopo::get_mesh_father(const std::string& mesh_name) const { return mesh_father.at(mesh_name); }
