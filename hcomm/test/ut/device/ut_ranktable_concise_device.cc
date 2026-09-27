/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include <mockcpp/mockcpp.hpp>
#define private public
#include "topoinfo_ranktableConcise.h"
#include <iostream>
#include "externalinput_pub.h"

using namespace std;
using namespace hccl;

class TopoinfoRanktableConciseTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "TopoinfoRanktableConciseTest SetUP" << std::endl; }
    static void TearDownTestCase() { std::cout << "TopoinfoRanktableConciseTest TearDown" << std::endl; }
    virtual void SetUp() { std::cout << "TopoinfoRanktableConciseTest Test SetUP" << std::endl; }
    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "TopoinfoRanktableConciseTest Test TearDown" << std::endl;
    }
};

TEST_F(TopoinfoRanktableConciseTest, Ut_GetSingleNicInfo_When_EmptyJsonArray_Expect_ReturnHCCL_E_PARA)
{
    std::string rankTableM = "";
    std::string identify = "test";
    TopoinfoRanktableConcise topoParser(rankTableM, identify);

    nlohmann::json serverListObj = nlohmann::json::array();
    u32 objIndex = 0;
    RankTable_t clusterInfo;
    RankInfo_t rankinfo;

    HcclResult ret = topoParser.GetSingleNicInfo(serverListObj, objIndex, rankinfo);
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);
}
