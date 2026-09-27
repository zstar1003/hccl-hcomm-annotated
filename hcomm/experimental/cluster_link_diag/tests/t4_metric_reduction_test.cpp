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

int main()
{
    ControlTopo control(test::make_control_json({{"10.0.0.1", {"0"}}}));
    const auto helper = test::make_probe_helper(control, {{"192.168.0.1"}});
    const ProbeHelper::PingpongRawResult raw = {{
        {10, 20, 15, 8},
        {11, 22, 16, 0},
        {12, 24, 17, 11},
        {13},
    }};

    const auto p90 = helper.reduce_pingpong_res(raw, ProbeHelper::PingpongMetric::P90Lat, 10);
    const auto p99 = helper.reduce_pingpong_res(raw, ProbeHelper::PingpongMetric::P99Lat, 10);
    const auto mean = helper.reduce_pingpong_res(raw, ProbeHelper::PingpongMetric::MeanLat, 10);
    const auto pass = helper.reduce_pingpong_res(raw, ProbeHelper::PingpongMetric::LogPassRate, 10);

    EXPECT_NEAR(p90[0][0], 10.0, 1e-6);
    EXPECT_NEAR(p99[0][0], 20.0, 1e-6);
    EXPECT_NEAR(mean[0][0], 15.0, 1e-6);
    EXPECT_NEAR(pass[0][0], std::log2(0.8), 1e-6);
    EXPECT_NEAR(pass[0][1], -1e10, 1.0);
    EXPECT_TRUE(std::isnan(pass[0][2]));
    EXPECT_TRUE(std::isnan(pass[0][3]));
    EXPECT_TRUE(std::isnan(p99[0][3]));
    EXPECT_THROW(helper.reduce_pingpong_res(raw, ProbeHelper::PingpongMetric::LogPassRate, 0));

    return test::finish("T4 metric reduction");
}
