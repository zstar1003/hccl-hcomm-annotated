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

#include <cmath>

namespace {
using PingList = std::map<std::string, std::vector<std::tuple<std::string, std::string, int>>>;

ProbeHelper make_solver(ControlTopo& control)
{
    auto helper = test::make_probe_helper(control, {{"A", "B", "C"}});
    helper.global_id = 2;
    helper.meshTopo.tracert_res_id["A"]["B"] = {{0}};
    helper.meshTopo.tracert_res_id["A"]["C"] = {{0, 1}};
    helper.meshTopo.tracert_res_id["B"]["C"] = {{1}};
    return helper;
}
} // namespace

int main()
{
    ControlTopo control(test::make_control_json({{"host", {"0", "1", "2"}}}));
    auto helper = make_solver(control);

    const PingList exact_plan = {{"mesh", {{"A", "B", 1}, {"A", "C", 1}}}};
    const ProbeHelper::PingpongMetricMatrix exact_values = {{3.0f, 8.0f}, {}, {}};
    const auto exact = helper.solve_pingpong_res(exact_values, exact_plan);
    EXPECT_NEAR(exact[0], 3.0, 1e-4);
    EXPECT_NEAR(exact[1], 5.0, 1e-4);

    const PingList over_plan = {{"mesh", {{"A", "B", 1}, {"A", "C", 1}, {"B", "C", 1}}}};
    const ProbeHelper::PingpongMetricMatrix over_values = {{3.0f, 8.0f}, {5.0f}, {}};
    const auto over = helper.solve_pingpong_res(over_values, over_plan);
    EXPECT_NEAR(over[0], 3.0, 1e-4);
    EXPECT_NEAR(over[1], 5.0, 1e-4);

    auto deficient_helper = make_solver(control);
    deficient_helper.meshTopo.tracert_res_id["A"]["B"] = {{0, 1}};
    deficient_helper.meshTopo.tracert_res_id["A"]["C"] = {{0, 1}};
    const auto deficient = deficient_helper.solve_pingpong_res(exact_values, exact_plan);
    EXPECT_TRUE(std::isnan(deficient[0]));
    EXPECT_TRUE(std::isnan(deficient[1]));

    const ProbeHelper::PingpongMetricMatrix invalid_values = {{NAN, 8.0f}, {}, {}};
    const auto invalid = helper.solve_pingpong_res(invalid_values, exact_plan);
    EXPECT_TRUE(std::isnan(invalid[0]));
    EXPECT_TRUE(std::isnan(invalid[1]));

    return test::finish("T5 link equation solver");
}
