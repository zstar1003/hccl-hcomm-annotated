/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "rank_info_types.h"
#include "eid_util.h"
#include <map>
#include <string>

extern int hex32_to_bin16(const char* hex_str, uint8_t* bin_out);

TEST(EID, test_ue_id)
{
    // 标卡全部EID
    std::map<std::string, int> port_map = {
        {"000000000000002000100000dfdf0020", 4}, {"000000000000002000100000dfdf0028", 5},
        {"000000000000002000100000dfdf0030", 6}, {"000000000000002000100000dfdf0021", 4},
        {"000000000000002000100000dfdf0029", 5}, {"000000000000002000100000dfdf0031", 6},
        {"000000000000002000100000dfdf0022", 4}, {"000000000000002000100000dfdf002a", 5},
        {"000000000000002000100000dfdf0032", 6}, {"000000000000002000100000dfdf0023", 4},
        {"000000000000002000100000dfdf002b", 5}, {"000000000000002000100000dfdf0033", 6},
    };
    for (auto it = port_map.begin(); it != port_map.end(); it++) {
        const char* eid = it->first.c_str();
        int port = it->second;
        int port1;
        EidGetPortId(eid, &port1);
        EXPECT_EQ(port1, port);
    }
}

TEST(EID, test_eid_decode)
{
    // 标卡全部EID

    dcmi_urma_eid_t eid1;
    dcmi_urma_eid_t eid2;
    hex32_to_bin16("000000000006030000100000df100700", eid1.raw);
    hex32_to_bin16("0000000000ff030000100000df100700", eid2.raw);
    int port1 = UrmaEidGetPortId(&eid1);
    int port2 = UrmaEidGetPortId(&eid2);
    int fe = UrmaEidGetFeId(&eid1);
    int die1 = UrmaEidGetDieId(&eid1);
    int die2 = UrmaEidGetDieId(&eid2);
    EXPECT_EQ(port1, 6);
    EXPECT_EQ(port2, 0x3f);
    EXPECT_EQ(fe, 3);
    EXPECT_EQ(die1, 0);
    EXPECT_EQ(die2, 3);
}

TEST(EID, test_uboe_ub_rtp)
{
    dcmi_urma_eid_t eid1;
    dcmi_urma_eid_t eid2;
    dcmi_urma_eid_t eid3;
    char ipaddr[32] = {0};
    hex32_to_bin16("00000000003f030000100000df100c00", eid1.raw);
    hex32_to_bin16("0000000000ff0ac0001000000a140200", eid2.raw);
    hex32_to_bin16("0000000000ff0a80001000000a140200", eid3.raw);
    EXPECT_EQ(UrmaEidIsUBOE(&eid1), 0);
    EXPECT_EQ(UrmaEidIsUbRtp(&eid1), 0);
    EXPECT_EQ(UrmaEidIsUBOE(&eid2), 1);
    EXPECT_EQ(UrmaEidIsUbRtp(&eid2), 0);
    EXPECT_EQ(UrmaEidIsUBOE(&eid3), 0);
    EXPECT_EQ(UrmaEidIsUbRtp(&eid3), 1);

    UrmaEid2CNA(&eid2, ipaddr, sizeof(ipaddr));
    EXPECT_STREQ(ipaddr, "10.10.20.2");
}
