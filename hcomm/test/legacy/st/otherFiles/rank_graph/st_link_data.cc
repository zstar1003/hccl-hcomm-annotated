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
#include "virtual_topo_stub.h"
#include "virtual_topo.h"

using namespace Hccl;

class LinkDataTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "LinkDataTest SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "LinkDataTest TearDown" << std::endl; }

    virtual void SetUp() { std::cout << "A Test case in LinkDataTest SetUP" << std::endl; }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in LinkDataTest TearDown" << std::endl;
    }
};

TEST_F(LinkDataTest, linkData_get_uniqueId)
{
    LinkData linkData(
        PortDeploymentType::DEV_NET, LinkProtocol::UB_CTP, 0, 1, IpAddress("0.0.0.0"), IpAddress("0.0.0.0"));
    auto data = linkData.GetUniqueId();

    LinkData linkData1(data);

    EXPECT_EQ(linkData.Describe(), linkData1.Describe());
}
