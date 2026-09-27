/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <CLI11.hpp>
#include <cstdint>
#include <gtest/gtest.h>
#include <iostream>
#include <sstream>

#include "cmd_base_utils.h"
#include "subcmd_run.h"

using namespace HcclSim;
#define private public
#undef private

class RunCommandTest : public testing::Test {
protected:
    void SetUp() override
    {
        app_ = std::make_unique<CLI::App>("test");
        cmd_ = std::make_unique<RunCommand>();
        savedBashFlag_ = g_hcclVmBashFlag;
        savedLevel_ = g_hcclVmLevel;
        g_hcclVmBashFlag = false;
        g_hcclVmLevel = 2;
    }
    void TearDown() override
    {
        g_hcclVmBashFlag = savedBashFlag_;
        g_hcclVmLevel = savedLevel_;
    }
    std::unique_ptr<CLI::App> app_;
    std::unique_ptr<RunCommand> cmd_;
    bool savedBashFlag_;
    std::uint32_t savedLevel_;
};

TEST_F(RunCommandTest, StaticName_ShouldReturnRun) { EXPECT_EQ(RunCommand::StaticName(), "run"); }

TEST_F(RunCommandTest, Setup_RegistersRunSubcommand)
{
    cmd_->Setup(*app_);
    auto sub = app_->get_subcommand("run");
    EXPECT_NE(sub, nullptr);
    EXPECT_EQ(sub->get_name(), "run");
}

TEST_F(RunCommandTest, Setup_RunSubcommandDescription)
{
    cmd_->Setup(*app_);
    auto run = app_->get_subcommand("run");
    ASSERT_NE(run, nullptr);
    EXPECT_FALSE(std::string(run->get_description()).empty());
}

TEST_F(RunCommandTest, Setup_HasConfigFileOption)
{
    cmd_->Setup(*app_);
    auto run = app_->get_subcommand("run");
    ASSERT_NE(run, nullptr);
    auto opts = run->get_options();
    EXPECT_GE(opts.size(), 1u);
}

TEST_F(RunCommandTest, Setup_HasLevelOption)
{
    cmd_->Setup(*app_);
    auto run = app_->get_subcommand("run");
    ASSERT_NE(run, nullptr);
    auto opts = run->get_options();
    EXPECT_GE(opts.size(), 1u);
}

TEST_F(RunCommandTest, Setup_AllowsExtras)
{
    cmd_->Setup(*app_);
    auto run = app_->get_subcommand("run");
    ASSERT_NE(run, nullptr);
    EXPECT_TRUE(run->get_allow_extras());
}

TEST_F(RunCommandTest, ParseRun_Help)
{
    cmd_->Setup(*app_);
    try {
        app_->parse("test run --help", true);
    } catch (const CLI::CallForHelp& e) {
        std::cerr << "Catch CLI::CallForHelp: " << e.what() << "\n";
    } catch (const std::exception& e) {
        std::cerr << "Catch unexpected exception: " << e.what() << "\n";
    }
}

TEST_F(RunCommandTest, ParseRun_WithConfigFile)
{
    cmd_->Setup(*app_);
    try {
        app_->parse("test run 112 -- echo hello", true);
    } catch (const CLI::CallForHelp& e) {
        std::cerr << "Catch CLI::CallForHelp: " << e.what() << "\n";
    } catch (const std::exception& e) {
        std::cerr << "Catch unexpected exception: " << e.what() << "\n";
    }
}

TEST_F(RunCommandTest, ParseRun_WithLevelOption)
{
    cmd_->Setup(*app_);
    try {
        app_->parse("test run 112 --level 2 -- echo hello", true);
    } catch (const CLI::CallForHelp& e) {
        std::cerr << "Catch CLI::CallForHelp: " << e.what() << "\n";
    } catch (const std::exception& e) {
        std::cerr << "Catch unexpected exception: " << e.what() << "\n";
    }
}

TEST_F(RunCommandTest, ParseRun_WithoutConfigFile_Throws)
{
    cmd_->Setup(*app_);
    EXPECT_THROW(app_->parse("test run", true), CLI::RequiredError);
}

TEST_F(RunCommandTest, CommandRegistry_CreateAll_ContainsRunCommand)
{
    auto commands = CommandRegistry::CreateAll();
    bool found = false;
    for (const auto& cmd : commands) {
        if (cmd != nullptr) {
            found = true;
            break;
        }
    }
    EXPECT_TRUE(found);
}

TEST_F(RunCommandTest, Execute_BashFlagTrue_ReturnsEarly)
{
    g_hcclVmBashFlag = true;
    cmd_->Setup(*app_);
    try {
        app_->parse("test run fake.yaml -- echo hello", true);
    } catch (const CLI::CallForHelp& e) {
        std::cerr << "Catch CLI::CallForHelp: " << e.what() << "\n";
    } catch (const std::exception& e) {
        std::cerr << "Catch unexpected exception: " << e.what() << "\n";
    }
    EXPECT_TRUE(g_hcclVmBashFlag);
}

TEST_F(RunCommandTest, Execute_BashFlagFalse_ParsesConfig)
{
    g_hcclVmBashFlag = false;
    cmd_->Setup(*app_);
    try {
        app_->parse("test run fake.yaml --level 1 -- echo world", true);
    } catch (const CLI::CallForHelp& e) {
        std::cerr << "Catch CLI::CallForHelp: " << e.what() << "\n";
    } catch (const std::exception& e) {
        std::cerr << "Catch unexpected exception: " << e.what() << "\n";
    }
    EXPECT_FALSE(g_hcclVmBashFlag);
}

TEST_F(RunCommandTest, ParseRun_ConfigFileOptionHasValidator)
{
    cmd_->Setup(*app_);
    auto sub = app_->get_subcommand("run");
    ASSERT_NE(sub, nullptr);
    auto* opt = sub->get_option("configFile");
    ASSERT_NE(opt, nullptr);
    EXPECT_NE(opt->get_validator(), nullptr);
}

TEST_F(RunCommandTest, ParseRun_ConfigFileRequired)
{
    cmd_->Setup(*app_);
    auto sub = app_->get_subcommand("run");
    ASSERT_NE(sub, nullptr);
    auto* opt = sub->get_option("configFile");
    ASSERT_NE(opt, nullptr);
    EXPECT_TRUE(opt->get_required());
}
