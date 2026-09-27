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
#include <nlohmann/json.hpp>

using json = nlohmann::json;

class RanktableInfo {
public:
    struct Device {
        std::string device_id;
        std::string device_ip;
        std::string rank_id;

        Device(std::string device_id, std::string device_ip, std::string rank_id)
            : device_id(device_id),
              device_ip(device_ip),
              rank_id(rank_id)
        {}

        Device() = default;
    };

    struct Server {
        std::vector<Device> device;
        std::string server_id;
        std::string server_ip;

        Server(std::string server_id, std::string server_ip) : server_id(server_id), server_ip(server_ip) {}

        Server() = default;
    };

    std::vector<Server> server_list;
    std::string controller_ip;

    // 第一个构造函数：接受json对象
    explicit RanktableInfo(const json& j)
    {
        try {
            for (auto& server : j["server_list"]) {
                Server s;
                s.server_id = server["server_id"];
                for (auto& device : server["device"]) {
                    Device d(device["device_id"], device["device_ip"], device["rank_id"]);
                    s.device.emplace_back(d);
                }
                server_list.emplace_back(s);
            }
        } catch (const std::exception& e) {
            server_list.clear();
            throw std::runtime_error("[RanktableInfo]json invalid");
        }
    }

    // 第二个构造函数：接受文件路径，委托给第一个构造函数
    explicit RanktableInfo(const std::string& json_path)
        : RanktableInfo([&]() {
              std::ifstream ifs(json_path);
              if (!ifs.is_open()) {
                  throw std::runtime_error("[RanktableInfo]json unavailable");
              }
              return json::parse(ifs);
          }())
    {}

    void set_server_ip(const json& j);

    void set_server_ip(const std::string& json_path);

    void set_server_ip(const std::vector<std::string>& server_ips);

    void set_controller_ip(const std::string& controller_ip) { this->controller_ip = controller_ip; }
};
