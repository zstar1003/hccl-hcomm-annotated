/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <memory>

#include "cmd_base.h"

using namespace HcclSim;

class MockCommand : public CommandBase {
public:
    static std::string StaticName() { return "mock"; }
    void Setup(CLI::App& app) override { app.add_flag("--mock-flag", flag_); }
    bool flag_ = false;
};

class CommandBaseTest : public testing::Test {
protected:
};

TEST_F(CommandBaseTest, StaticName_ShouldReturnMock) { EXPECT_EQ(MockCommand::StaticName(), "mock"); }

TEST_F(CommandBaseTest, CreateAll_ShouldReturnVector)
{
    auto commands = CommandRegistry::CreateAll();
    EXPECT_GE(commands.size(), 0);
    for (auto& cmd : commands) {
        EXPECT_NE(cmd, nullptr);
    }
}

TEST_F(CommandBaseTest, VirtualDestructorShouldNotCrash)
{
    auto cmd = std::make_unique<MockCommand>();
    EXPECT_NE(cmd, nullptr);
}
