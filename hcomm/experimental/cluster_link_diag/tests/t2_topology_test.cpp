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
#include "probe_helper_fixture.h"

#include "topo/mesh_topo.h"

int main()
{
    ControlTopo control(test::make_control_json({
        {"10.0.0.1", {"0", "1"}},
        {"10.0.0.2", {"2"}},
    }));
    EXPECT_EQ(control.LeafList().size(), 3U);
    EXPECT_EQ(control.ServerTreeList().size(), 2U);
    EXPECT_EQ(control.route_to_device("1").front(), "10.0.0.1");

    auto helper = test::make_probe_helper(
        control, {
                     {"192.168.0.1", "192.168.0.2"},
                     {"192.168.0.3"},
                 });
    auto lldp_topo = helper.probe_lldp_topo([](const std::vector<std::string>&) {
        return std::vector<std::string>{"172.16.0.10", "172.16.0.10", ""};
    });

    EXPECT_EQ(helper.lldp_device_groups.size(), 2U);
    EXPECT_EQ(lldp_topo.mesh_map.size(), 2U);
    EXPECT_EQ(lldp_topo.get_leaf_kids("TOPO[0,0]").size(), 2U);
    EXPECT_EQ(lldp_topo.get_leaf_kids("TOPO[0,1]").size(), 1U);

    const auto encoded = lldp_topo.to_json();
    const MeshTopo decoded(encoded);
    EXPECT_EQ(decoded.mesh_map.size(), lldp_topo.mesh_map.size());
    EXPECT_EQ(decoded.root_mesh, lldp_topo.root_mesh);

    EXPECT_THROW(lldp_topo.add_mesh(lldp_topo.get_mesh("TOPO[0,0]")));
    EXPECT_THROW(lldp_topo.get_mesh("missing"));

    auto incomplete_helper = test::make_probe_helper(
        control, {
                     {"192.168.0.1"},
                     {"192.168.0.3"},
                 });
    EXPECT_FALSE(incomplete_helper.device_count_complete());
    EXPECT_EQ(incomplete_helper.configured_device_count(), 3U);
    EXPECT_EQ(incomplete_helper.discovered_device_count(), 2U);

    return test::finish("T2 topology modeling");
}
