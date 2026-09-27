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

#include <map>
#include <set>

#include "helper/probe_helper.h"
#include "helper/probe_plan.h"

namespace {
Mesh make_domain(int index, const std::vector<std::string>& devices)
{
    Mesh mesh(0, MeshTopo::get_mesh_name(0, index));
    for (const auto& device : devices) {
        mesh.add_kid(device);
    }
    return mesh;
}
} // namespace

int main()
{
    MeshTopo topology;
    topology.add_mesh(make_domain(0, {"10.0.0.1", "10.0.0.2"}));
    topology.add_mesh(make_domain(1, {"10.0.1.1", "10.0.1.2"}));
    topology.add_mesh(make_domain(2, {"10.0.2.1", "10.0.2.2"}));

    const auto l1 = probe_plan::build_l0_pinglist_from_lldp_topo(topology);
    size_t l1_tasks = 0;
    for (const auto& [mesh, tasks] : l1) {
        l1_tasks += tasks.size();
    }
    EXPECT_EQ(l1.size(), 3U);
    EXPECT_EQ(l1_tasks, 3U);

    const std::map<std::string, std::string> labels = {
        {"10.0.0.1", "dev0"}, {"10.0.0.2", "dev1"}, {"10.0.1.1", "dev2"},
        {"10.0.1.2", "dev3"}, {"10.0.2.1", "dev4"}, {"10.0.2.2", "dev5"},
    };
    const auto l2 = probe_plan::build_l2_fullmesh_ping_tasks(topology, labels, 49152);
    EXPECT_EQ(l2.size(), 12U); // 3 adjacent domain pairs * 2 * 2

    std::set<std::tuple<std::string, std::string, int>> unique;
    for (const auto& task : l2) {
        unique.emplace(task.from_ip, task.to_ip, task.src_port);
        EXPECT_EQ(task.src_port, 49152);
        EXPECT_TRUE(task.tag.rfind("l2-fullmesh:", 0) == 0);
    }
    EXPECT_EQ(unique.size(), l2.size());

    MeshTopo one_domain;
    one_domain.add_mesh(make_domain(0, {"10.1.0.1", "10.1.0.2", "10.1.0.3"}));
    const auto ring = probe_plan::build_l0_pinglist_from_lldp_topo(one_domain);
    EXPECT_EQ(ring.at("TOPO[0,0]").size(), 3U);
    EXPECT_TRUE(probe_plan::build_l2_fullmesh_ping_tasks(one_domain, {}, 49152).empty());

    MeshTopo even_domain;
    even_domain.add_mesh(make_domain(0, {"10.2.0.1", "10.2.0.2", "10.2.0.3", "10.2.0.4"}));
    const auto odd_core = probe_plan::build_l0_pinglist_from_lldp_topo(even_domain);
    const auto& odd_core_tasks = odd_core.at("TOPO[0,0]");
    EXPECT_EQ(odd_core_tasks.size(), 4U);
    EXPECT_EQ(std::get<0>(odd_core_tasks[0]), "10.2.0.1");
    EXPECT_EQ(std::get<1>(odd_core_tasks[0]), "10.2.0.2");
    EXPECT_EQ(std::get<0>(odd_core_tasks[1]), "10.2.0.2");
    EXPECT_EQ(std::get<1>(odd_core_tasks[1]), "10.2.0.3");
    EXPECT_EQ(std::get<0>(odd_core_tasks[2]), "10.2.0.3");
    EXPECT_EQ(std::get<1>(odd_core_tasks[2]), "10.2.0.1");
    EXPECT_EQ(std::get<0>(odd_core_tasks[3]), "10.2.0.4");

    {
        std::map<std::string, std::vector<std::tuple<std::string, std::string, int>>> pinglist = {
            {"mesh-a",
             {
                 {"10.0.0.1", "10.0.0.2", 10},
                 {"10.0.0.1", "10.0.0.3", 10},
                 {"10.0.0.4", "10.0.0.5", 20},
             }},
            {"mesh-b",
             {
                 {"10.0.0.1", "10.0.0.6", 1},
             }},
        };
        const auto limited = ProbeHelper::limit_pinglist_targets_per_source(pinglist, 16);
        EXPECT_EQ(limited.at("mesh-a").size(), 3U);
        EXPECT_EQ(std::get<2>(limited.at("mesh-a").at(0)), 10);
        EXPECT_EQ(std::get<2>(limited.at("mesh-a").at(1)), 6);
        EXPECT_EQ(std::get<2>(limited.at("mesh-a").at(2)), 16);
        EXPECT_TRUE(limited.find("mesh-b") == limited.end());
    }

    return test::finish("T3 probe plan sizing");
}
