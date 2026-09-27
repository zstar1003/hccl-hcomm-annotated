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
#include <type_traits>
#include <tuple>
#include <algorithm>
#include <stdexcept>
#include <utility>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

class ControlTopo {
private:
    static std::map<std::string, int> name_repeat;

public:
    struct control_tree {
        std::string ip;
        std::vector<control_tree> kids;

        // 使用委托构造函数简化
        explicit control_tree(std::string ip, std::vector<control_tree> kids = {})
            : ip(std::move(ip)),
              kids(std::move(kids))
        {}

        control_tree() = default;

        // 使用完美转发添加子节点
        template <typename T>
        void add_kid(T&& kid)
        {
            kids.emplace_back(std::forward<T>(kid));
        }
    };

private:
    control_tree control_tree_root{"root"};
    std::vector<std::string> leaf_list;
    std::vector<control_tree> server_tree_list;

    // 优雅递归实现
    int recursive_build_tree(control_tree& tree, const json& j)
    {
        if (j.is_array()) {
            int max_kid_rank = -1;
            for (auto& element : j) {
                int kid_rank = recursive_build_tree(tree, element);
                max_kid_rank = std::max(max_kid_rank, kid_rank);
            }
            int current_rank = max_kid_rank + 1;
            if (current_rank == 1) {
                server_tree_list.emplace_back(tree);
            }
            return current_rank;
        } else if (j.is_object()) {
            int max_kid_rank = -1;
            for (auto& [key, value] : j.items()) {
                auto name = key;
                name_repeat[name]++;
                if (name_repeat[name] > 1) {
                    name += '_' + std::to_string(name_repeat[name]);
                }
                control_tree child{name};
                int kid_rank = recursive_build_tree(child, value);
                tree.add_kid(std::move(child));
                max_kid_rank = std::max(max_kid_rank, kid_rank);
            }
            int current_rank = max_kid_rank + 1;
            if (current_rank == 1) {
                server_tree_list.emplace_back(tree);
            }
            return current_rank;
        } else if (j.is_string()) {
            auto name = j.get<std::string>();
            name_repeat[name]++;
            if (name_repeat[name] > 1) {
                name += '_' + std::to_string(name_repeat[name]);
            }
            tree.add_kid(control_tree{name});
            leaf_list.emplace_back(name);
            return 0;
        }

        throw std::runtime_error("[ControlTopo] Unsupported JSON type in control topology");
    }

public:
    // 第一个构造函数：接受json对象
    explicit ControlTopo(const json& j)
    {
        try {
            auto topo = j.at("control_topo"); // 使用at()确保键存在
            recursive_build_tree(control_tree_root, topo);
        } catch (const std::exception& e) {
            throw std::runtime_error("[ControlTopo] Invalid JSON structure: " + std::string(e.what()));
        }
    }

    // 第二个构造函数：接受文件路径
    explicit ControlTopo(const std::string& json_path)
    {
        std::ifstream ifs(json_path);
        if (!ifs.is_open()) {
            throw std::runtime_error("[ControlTopo] JSON file unavailable: " + json_path);
        }

        try {
            json j = json::parse(ifs);
            *this = ControlTopo(j); // 委托给第一个构造函数
        } catch (const std::exception& e) {
            throw std::runtime_error("[ControlTopo] JSON parse error: " + std::string(e.what()));
        }
    }

    void set_controller_ip(std::string controller_ip) { control_tree_root.ip = std::move(controller_ip); }

    const control_tree& ControlTreeRoot() const { return control_tree_root; }

    const std::vector<std::string>& LeafList() const { return leaf_list; }

    const std::vector<control_tree>& ServerTreeList() const { return server_tree_list; }

    std::vector<std::string> route_to_device(const std::string& device_id_or_ip) const;

private:
    mutable std::map<std::string, std::vector<std::string>> cache;
    void recursive_cache(const control_tree& tree, const std::vector<std::string> path = {}) const;
};

std::string clean_ip(const std::string& ip);
