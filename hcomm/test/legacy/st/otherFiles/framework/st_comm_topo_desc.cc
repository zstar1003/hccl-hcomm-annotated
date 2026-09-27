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
#include <mockcpp/mokc.h>
#include <mockcpp/mockcpp.hpp>
#define private public
#include "comm_topo_desc_v2.h"
#undef private

using namespace Hccl;
class CommTopoDescTest : public testing::Test {
public:
    static void SetUpTestCase() { std::cout << "CommTopoDescTest SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "CommTopoDescTest TearDown" << std::endl; }

    virtual void SetUp()
    {
        CommTopoDesc::GetInstance().rankSizeMap_.clear();
        CommTopoDesc::GetInstance().l0TopoTypeMap_.clear();

        std::cout << "A Test case in CommTopoDescTest SetUp" << std::endl;
    }

    virtual void TearDown()
    {
        std::cout << "A Test case in CommTopoDescTest TearDown" << std::endl;
        CommTopoDesc::GetInstance().rankSizeMap_.clear();
        CommTopoDesc::GetInstance().l0TopoTypeMap_.clear();
        GlobalMockObject::verify();
    }
};

TEST_F(CommTopoDescTest, test_comm_topo_desc)
{
    std::string identifier("test");
    uint32_t rankSize = 8;
    CommTopo topoType = CommTopo::COMM_TOPO_1DMESH;

    CommTopoDesc::GetInstance().SaveRankSize(identifier, rankSize);
    CommTopoDesc::GetInstance().SaveL0TopoType(identifier, topoType);

    uint32_t rankSizeRet = 0;
    CommTopo topoTypeRet = CommTopo::COMM_TOPO_RESERVED;
    auto ret = CommTopoDesc::GetInstance().GetL0TopoType(identifier, &topoTypeRet);
    EXPECT_EQ(ret, 0);
    EXPECT_EQ(static_cast<uint32_t>(topoTypeRet), static_cast<uint32_t>(topoType));

    ret = CommTopoDesc::GetInstance().GetRankSize(identifier, &rankSizeRet);
    EXPECT_EQ(ret, 0);
    EXPECT_EQ(rankSizeRet, rankSize);
}

TEST_F(CommTopoDescTest, test_comm_topo_desc_fail)
{
    std::string identifier("test");
    uint32_t rankSizeRet = 8;
    CommTopo topoTypeRet = CommTopo::COMM_TOPO_RESERVED;
    auto ret = CommTopoDesc::GetInstance().GetL0TopoType(identifier, &topoTypeRet);
    EXPECT_EQ(ret, HCCL_E_PARA);

    CommTopoDesc::GetInstance().GetRankSize(identifier, &rankSizeRet);
    EXPECT_EQ(ret, HCCL_E_PARA);
}
