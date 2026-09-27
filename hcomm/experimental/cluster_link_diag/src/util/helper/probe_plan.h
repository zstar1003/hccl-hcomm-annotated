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

#include <map>
#include <string>
#include <tuple>
#include <vector>

#include "probe_helper.h"
#include "topo/mesh_topo.h"

namespace probe_plan {
using PingList = std::map<std::string, std::vector<std::tuple<std::string, std::string, int>>>;

PingList build_l0_pinglist_from_lldp_topo(MeshTopo mesh_topo);

std::vector<ProbeHelper::ExplicitPingTask> build_l2_fullmesh_ping_tasks(
    const MeshTopo& mesh_topo, const std::map<std::string, std::string>& device_label_by_ip, int src_port);
} // namespace probe_plan
