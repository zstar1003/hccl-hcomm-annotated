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
#include <vector>

#include "helper/probe_helper.h"

namespace test {
inline json make_control_json(const std::map<std::string, std::vector<std::string>>& hosts)
{
    json root;
    root["control_topo"] = hosts;
    return root;
}

inline ProbeHelper make_probe_helper(ControlTopo& topo, std::vector<std::vector<std::string>> device_ip_groups)
{
    auto ip_callback
        = [groups = std::move(device_ip_groups)](const std::vector<std::string>&, const std::vector<int>&) mutable {
              return groups;
          };
    auto trace_callback = [](const std::vector<std::string>&, const std::vector<std::string>&,
                             const std::vector<std::vector<std::string>>&, const std::vector<std::vector<int>>&) {
        return std::vector<std::vector<std::vector<std::vector<std::string>>>>{};
    };
    auto set_pinglist_callback = [](const std::vector<std::string>&, const std::vector<std::string>&,
                                    const std::vector<std::vector<std::string>>&, const std::vector<std::vector<int>>&,
                                    const std::vector<std::vector<int>>&, const std::vector<int>&) {};
    auto ping_callback = [](const std::vector<std::string>&, const std::vector<std::string>&, const std::vector<int>&) {
        return ProbeHelper::PingpongRawResult{};
    };
    return ProbeHelper(&topo, ip_callback, trace_callback, set_pinglist_callback, ping_callback);
}
} // namespace test
