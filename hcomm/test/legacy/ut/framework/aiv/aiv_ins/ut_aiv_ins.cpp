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
#define protected public
#include "aiv_ins.h"
#include "aiv_ins_preprocessor.h"
#include "hierarchical_queue.h"
#undef private
#undef protected

using namespace Hccl;

using namespace std;

class AivInsTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "AivInsTest SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "AivInsTest TearDown" << std::endl; }

    virtual void SetUp() { std::cout << "A Test case in AivInsTest SetUP" << std::endl; }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in AivInsTest TearDown" << std::endl;
    }
};

TEST_F(AivInsTest, should_return_success_when_calling_other)
{
    std::vector<LinkData> links;
    LinkData link(BasePortType(PortDeploymentType::P2P), 0, 1, 0, 1);
    links.push_back(link);
    AivOpArgs aivOpArgs{};
    std::unique_ptr<AivInstruction> ins = std::make_unique<AivInstruction>(links, aivOpArgs);

    EXPECT_EQ(ins->GetLinks().size(), 1);

    ins->Describe();
}
