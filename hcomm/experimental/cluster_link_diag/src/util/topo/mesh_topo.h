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
#include <vector>
#include <map>
#include <string>

#include "control_topo.h"
#include "mesh.h"

namespace nlohmann {
template <>
struct adl_serializer<Mesh> {
    static void from_json(const json& j, Mesh& p) { p = Mesh::from_json(j); }
    static void to_json(json& j, const Mesh& p) { j = p.to_json(); }
};
} // namespace nlohmann

class MeshTopo {
protected:
    static const u_int32_t MESH_INDEX_LEN = 8;

    struct MeshComparator {
        bool operator()(const std::string& left_mesh_name, const std::string& right_mesh_name) const
        {
            return parse_mesh_name(left_mesh_name) < parse_mesh_name(right_mesh_name);
        }
    };

public:
    std::map<std::string, Mesh, MeshComparator> mesh_map;
    std::map<std::string, std::string> mesh_father;
    std::string root_mesh;

    std::map<std::string, std::map<std::string, std::vector<std::vector<int>>>> tracert_res_id;

    MeshTopo(const ControlTopo& control_topo);
    MeshTopo(const std::map<std::string, Mesh, MeshComparator>& mesh_map, const std::string& root_mesh);
    MeshTopo(const json& j);
    MeshTopo(const std::string& json_path);
    MeshTopo() = default;

    static MeshTopo from_json(const json& j);
    static MeshTopo from_json(const std::string& json_path);
    json to_json() const;
    void to_json(const std::string& json_path);

    void add_mesh(const Mesh& mesh);
    Mesh& get_mesh(const std::string& mesh_name);
    static std::string get_mesh_name(int level, int index);
    static int parse_mesh_name(const std::string& mesh_name);

    std::vector<std::string> get_leaf_kids(const Mesh& mesh) const;
    std::vector<std::string> get_leaf_kids(const std::string& mesh_name) const;
    void add_kid(const std::string& mesh_name, const std::string& kid_name);

    std::string get_mesh_father(const std::string& mesh_name) const;
};
