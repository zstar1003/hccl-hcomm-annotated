/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "test_common.h"

#include <filesystem>
#include <fstream>
#include <string>

#include "topo/probe_config.h"

namespace {
class TempJson {
public:
    explicit TempJson(const json& value)
    {
        static int sequence = 0;
        path_ = std::filesystem::temp_directory_path() / ("disp_probe_t1_" + std::to_string(++sequence) + ".json");
        std::ofstream(path_) << value.dump(2);
    }

    ~TempJson()
    {
        std::error_code ec;
        std::filesystem::remove(path_, ec);
    }

    std::string path() const { return path_.string(); }

private:
    std::filesystem::path path_;
};

json valid_config()
{
    return {
        {"probe_scope", {{"10.0.0.1", json::array({0, "1"})}}},
        {"probe_topo", {{"tracert", json::object()}}},
        {"probe_controller", {{"pingpong", json::object()}}}};
}

json valid_v2_config()
{
    return {
        {"schema_version", 2},
        {"controller", "node-01"},
        {"deploy", {{"to_path", "/root/disp_probe"}}},
        {"hosts", json::array({
                      {{"id", "node-01"}, {"ip", "10.0.0.1"}, {"user", "root"}, {"ssh_key", "/root/.ssh/id_ed25519"}},
                      {{"id", "node-02"}, {"ip", "10.0.0.2"}, {"user", "root"}, {"password_env", "HOST_02_PASS"}},
                  })},
        {"probe",
         {{"scope",
           {{"node-01", {{"device_range", json::array({0, 2})}}}, {"node-02", {{"devices", json::array({0, 1, 3})}}}}},
          {"topology",
           {{"sport_begin", 49160},
            {"sport_count", 4},
            {"tree_probe_sport_count", 2},
            {"topology_optimized", false},
            {"l2_path_aware", false},
            {"output_subdir", "case-a"}}},
          {"pingpong", {{"times", 10}, {"turns", 20}, {"payload_len", 64}, {"interval_ms", 5}}}}}};
}
} // namespace

int main()
{
    {
        TempJson file(valid_config());
        const auto config = load_probe_runtime_config(file.path());
        EXPECT_TRUE(config.enabled);
        EXPECT_EQ(config.probe_topo_tracert.sport_begin, 49152);
        EXPECT_EQ(config.probe_topo_tracert.sport_count, 1);
        EXPECT_EQ(config.probe_controller_pingpong.times, 50);
        EXPECT_EQ(config.probe_controller_pingpong.turns, 1000);
        EXPECT_EQ(config.probe_controller_pingpong.payload_len, 12);
        EXPECT_EQ(config.probe_controller_pingpong.interval_ms, 1);
        EXPECT_EQ(config.probe_topo_tracert.scope.at("10.0.0.1").at(0), "0");
        EXPECT_EQ(config.probe_topo_tracert.scope.at("10.0.0.1").at(1), "1");
    }
    {
        json legacy = {{"control_topo", json::array({"0", "1"})}};
        TempJson file(legacy);
        EXPECT_FALSE(load_probe_runtime_config(file.path()).enabled);
    }
    {
        auto value = valid_config();
        value["probe_scope"]["10.0.0.1"] = json::array({"0", "0"});
        TempJson file(value);
        EXPECT_THROW(load_probe_runtime_config(file.path()));
    }
    {
        auto value = valid_config();
        value["probe_scope"]["10.0.0.1"] = json::array({"0-7"});
        TempJson file(value);
        EXPECT_THROW(load_probe_runtime_config(file.path()));
    }
    {
        auto value = valid_config();
        value["probe_topo"]["tracert"]["output_subdir"] = "../escape";
        TempJson file(value);
        EXPECT_THROW(load_probe_runtime_config(file.path()));
    }
    {
        auto value = valid_config();
        value.erase("probe_controller");
        TempJson file(value);
        EXPECT_THROW(load_probe_runtime_config(file.path()));
    }
    {
        auto value = valid_config();
        value["probe_controller"]["pingpong"]["times"] = 0;
        TempJson file(value);
        EXPECT_THROW(load_probe_runtime_config(file.path()));
    }
    {
        auto value = valid_config();
        value["probe_controller"]["pingpong"]["payload_len"] = 1024;
        TempJson file(value);
        EXPECT_EQ(load_probe_runtime_config(file.path()).probe_controller_pingpong.payload_len, 1024);
    }
    {
        auto value = valid_config();
        value["probe_controller"]["pingpong"]["payload_len"] = 0;
        TempJson file(value);
        EXPECT_THROW(load_probe_runtime_config(file.path()));
    }
    {
        auto value = valid_config();
        value["probe_controller"]["pingpong"]["payload_len"] = 1501;
        TempJson file(value);
        EXPECT_THROW(load_probe_runtime_config(file.path()));
    }
    {
        auto value = valid_config();
        value["probe_controller"]["pingpong"]["interval_ms"] = 20;
        TempJson file(value);
        EXPECT_EQ(load_probe_runtime_config(file.path()).probe_controller_pingpong.interval_ms, 20);
    }
    {
        auto value = valid_config();
        value["probe_controller"]["pingpong"]["interval_ms"] = 0;
        TempJson file(value);
        EXPECT_THROW(load_probe_runtime_config(file.path()));
    }
    {
        TempJson file(valid_v2_config());
        const auto config = load_probe_runtime_config(file.path());
        EXPECT_TRUE(config.enabled);
        EXPECT_EQ(config.probe_topo_tracert.scope.at("10.0.0.1").size(), 3);
        EXPECT_EQ(config.probe_topo_tracert.scope.at("10.0.0.1").at(0), "0");
        EXPECT_EQ(config.probe_topo_tracert.scope.at("10.0.0.1").at(2), "2");
        EXPECT_EQ(config.probe_topo_tracert.scope.at("10.0.0.2").at(2), "3");
        EXPECT_EQ(config.probe_controller_pingpong.scope.at("10.0.0.2").at(2), "3");
        EXPECT_EQ(config.probe_topo_tracert.sport_begin, 49160);
        EXPECT_EQ(config.probe_topo_tracert.sport_count, 4);
        EXPECT_EQ(config.probe_topo_tracert.tree_probe_sport_count, 2);
        EXPECT_FALSE(config.probe_topo_tracert.topology_optimized);
        EXPECT_FALSE(config.probe_topo_tracert.l2_path_aware);
        EXPECT_EQ(config.probe_topo_tracert.output_subdir, "case-a");
        EXPECT_EQ(config.probe_controller_pingpong.times, 10);
        EXPECT_EQ(config.probe_controller_pingpong.turns, 20);
        EXPECT_EQ(config.probe_controller_pingpong.payload_len, 64);
        EXPECT_EQ(config.probe_controller_pingpong.interval_ms, 5);
    }
    {
        auto value = valid_v2_config();
        value["probe"]["scope"]["node-03"] = {{"device_range", json::array({0, 1})}};
        TempJson file(value);
        EXPECT_THROW(load_probe_runtime_config(file.path()));
    }
    {
        auto value = valid_v2_config();
        value["hosts"][1]["ip"] = "10.0.0.1";
        TempJson file(value);
        EXPECT_THROW(load_probe_runtime_config(file.path()));
    }
    {
        auto value = valid_v2_config();
        value["probe"]["scope"]["node-01"] = {{"device_range", json::array({2, 1})}};
        TempJson file(value);
        EXPECT_THROW(load_probe_runtime_config(file.path()));
    }

    return test::finish("T1 configuration parsing");
}
