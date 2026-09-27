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

#include "tool/metrics_collector/metrics_collector.h"

int main()
{
    using namespace metrics_collector_detail;

    const auto parsed = parse_hccn_stat_output("mac_tx_pfc_pkt_num : 10\n"
                                               "mac_rx_pfc_pkt_num:20\n"
                                               "roce_tx_cnp_pkt_num: invalid\n"
                                               "roce_rx_cnp_pkt_num: 40\n"
                                               "line-without-colon\n");
    EXPECT_EQ(parsed.at("mac_tx_pfc_pkt_num"), 10U);
    EXPECT_EQ(parsed.at("mac_rx_pfc_pkt_num"), 20U);
    EXPECT_TRUE(parsed.find("roce_tx_cnp_pkt_num") == parsed.end());

    const auto values = collect_counter_values(3, parsed);
    EXPECT_EQ(values.size(), 6U);
    EXPECT_EQ(values[0], 3U);
    EXPECT_EQ(values[1], 10U);
    EXPECT_EQ(values[2], 20U);
    EXPECT_EQ(values[3], 0U);
    EXPECT_EQ(values[4], 40U);
    EXPECT_EQ(values[5], 0U);

    const auto complete = collect_counter_values(
        4, parse_hccn_stat_output("mac_tx_pfc_pkt_num:0\n"
                                  "mac_rx_pfc_pkt_num:0\n"
                                  "roce_tx_cnp_pkt_num:0\n"
                                  "roce_rx_cnp_pkt_num:0\n"));
    EXPECT_EQ(complete.size(), 6U);
    EXPECT_EQ(complete[5], 1U);

    const auto strict = parse_hccn_stat_output("mac_tx_pfc_pkt_num:12garbage\n"
                                               "mac_rx_pfc_pkt_num:-1\n");
    EXPECT_TRUE(strict.empty());

    const auto ids = normalize_device_ids({2, -1, 2, 0});
    EXPECT_EQ(ids.size(), 2U);
    EXPECT_EQ(ids[0], 2);
    EXPECT_EQ(ids[1], 0);
    EXPECT_EQ(normalize_device_ids({}).front(), 0);
    EXPECT_EQ(normalize_device_ids({-1}).front(), 0);

    const auto normal_delta = counter_delta(100, 135);
    EXPECT_EQ(normal_delta.value, 35U);
    EXPECT_FALSE(normal_delta.reset_or_wrap);
    const auto reset_delta = counter_delta(100, 7);
    EXPECT_EQ(reset_delta.value, 7U);
    EXPECT_TRUE(reset_delta.reset_or_wrap);

    return test::finish("T8 metrics collector parsing");
}
