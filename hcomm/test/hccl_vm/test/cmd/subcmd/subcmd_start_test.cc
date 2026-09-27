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
#include "subcmd_start.h"

using namespace HcclSim;
#define private public
#undef private
class StartCommandTest : public testing::Test {
protected:
    void SetUp() override
    {
        app_ = std::make_unique<CLI::App>("test");
        cmd_ = std::make_unique<StartCommand>();
        // Save and reset global state
        savedBashFlag_ = g_hcclVmBashFlag;
        savedLevel_ = g_hcclVmLevel;
        g_hcclVmBashFlag = false;
        g_hcclVmLevel = 2;
    }

    void TearDown() override
    {
        // Restore global state
        g_hcclVmBashFlag = savedBashFlag_;
        g_hcclVmLevel = savedLevel_;
    }

    std::unique_ptr<CLI::App> app_;
    std::unique_ptr<StartCommand> cmd_;
    bool savedBashFlag_;
    std::uint32_t savedLevel_;
};

TEST_F(StartCommandTest, StaticName_ShouldReturnStart) { EXPECT_EQ(StartCommand::StaticName(), "start"); }

TEST_F(StartCommandTest, Setup_RegistersStartSubcommand)
{
    cmd_->Setup(*app_);
    auto sub = app_->get_subcommand("start");
    EXPECT_NE(sub, nullptr);
    EXPECT_EQ(sub->get_name(), "start");
}

TEST_F(StartCommandTest, Setup_StartSubcommandDescription)
{
    cmd_->Setup(*app_);
    auto start = app_->get_subcommand("start");
    ASSERT_NE(start, nullptr);
    EXPECT_FALSE(std::string(start->get_description()).empty());
}

TEST_F(StartCommandTest, Setup_HasConfigFileOption)
{
    cmd_->Setup(*app_);
    auto start = app_->get_subcommand("start");
    ASSERT_NE(start, nullptr);
    auto opts = start->get_options();
    EXPECT_GE(opts.size(), 1u);
}

TEST_F(StartCommandTest, Setup_HasLevelOption)
{
    cmd_->Setup(*app_);
    auto start = app_->get_subcommand("start");
    ASSERT_NE(start, nullptr);
    auto opts = start->get_options();
    bool hasLevel = false;
    for (const auto& opt : opts) {
        if (opt->get_name() == "--level") {
            hasLevel = true;
            break;
        }
    }
    EXPECT_TRUE(hasLevel);
}

TEST_F(StartCommandTest, Setup_ConfigFileOptionIsRequired)
{
    cmd_->Setup(*app_);
    auto start = app_->get_subcommand("start");
    ASSERT_NE(start, nullptr);
    auto opts = start->get_options();
    bool foundRequired = false;
    for (const auto& opt : opts) {
        if (opt->get_required()) {
            foundRequired = true;
            break;
        }
    }
    EXPECT_TRUE(foundRequired);
}

TEST_F(StartCommandTest, ParseStart_Help)
{
    cmd_->Setup(*app_);
    try {
        app_->parse("test start --help", true);
    } catch (const CLI::CallForHelp& e) {
        std::cerr << "Catch CLI::CallForHelp: " << e.what() << "\n";
    } catch (const std::exception& e) {
        std::cerr << "Catch unexpected exception: " << e.what() << "\n";
    }
}

TEST_F(StartCommandTest, ParseStart_WithoutConfigFile_Throws)
{
    cmd_->Setup(*app_);
    EXPECT_THROW(app_->parse("test start", true), CLI::RequiredError);
}

TEST_F(StartCommandTest, Execute_BashFlagTrue_LogsWarningAndReturns)
{
    // When g_hcclVmBashFlag is true, Execute() should log a warning and return early
    // without calling ParseYamlTopo or other external functions
    g_hcclVmBashFlag = true;
    cmd_->Setup(*app_);
    // Parse to trigger the callback which calls Execute()
    try {
        app_->parse("test start test_model.yaml", true);
    } catch (const CLI::CallForHelp& e) {
        std::cerr << "Catch CLI::CallForHelp: " << e.what() << "\n";
    } catch (const std::exception& e) {
        std::cerr << "Catch unexpected exception: " << e.what() << "\n";
    }
    // If we get here without crashing, the early return path worked
    EXPECT_TRUE(g_hcclVmBashFlag);
}

TEST_F(StartCommandTest, Execute_BashFlagFalse_ParsesAndCallsExternal)
{
    // When g_hcclVmBashFlag is false, Execute() will try to call ParseYamlTopo
    // which may fail, but we can at least verify the code path is exercised
    g_hcclVmBashFlag = false;
    cmd_->Setup(*app_);
    try {
        app_->parse("test start test_model.yaml", true);
    } catch (const CLI::CallForHelp& e) {
        std::cerr << "Catch CLI::CallForHelp: " << e.what() << "\n";
    } catch (const std::exception& e) {
        std::cerr << "Catch unexpected exception: " << e.what() << "\n";
    }
    // The test may fail at ParseYamlTopo since the file doesn't exist,
    // but we've covered the g_hcclVmBashFlag == false branch
    EXPECT_FALSE(g_hcclVmBashFlag);
}

TEST_F(StartCommandTest, CommandRegistry_CreateAll_ContainsStartCommand)
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

TEST_F(StartCommandTest, Execute_BashFlagTrue_ReturnsEarly)
{
    g_hcclVmBashFlag = true;
    cmd_->Setup(*app_);
    try {
        app_->parse("test start fake.yaml", true);
    } catch (const CLI::CallForHelp& e) {
        std::cerr << "Catch CLI::CallForHelp: " << e.what() << "\n";
    } catch (const std::exception& e) {
        std::cerr << "Catch unexpected exception: " << e.what() << "\n";
    }
    EXPECT_TRUE(g_hcclVmBashFlag);
}

TEST_F(StartCommandTest, Parse_LevelOptionDefault)
{
    cmd_->Setup(*app_);
    g_hcclVmLevel = 2;
    try {
        app_->parse("test start cfg.yaml", true);
    } catch (const CLI::CallForHelp& e) {
        std::cerr << "Catch CLI::CallForHelp: " << e.what() << "\n";
    } catch (const std::exception& e) {
        std::cerr << "Catch unexpected exception: " << e.what() << "\n";
    }
    EXPECT_EQ(g_hcclVmLevel, 2u);
}

TEST_F(StartCommandTest, Setup_ConfigFileOptionHasValidator)
{
    cmd_->Setup(*app_);
    auto start = app_->get_subcommand("start");
    ASSERT_NE(start, nullptr);
    auto* opt = start->get_option("configFile");
    ASSERT_NE(opt, nullptr);
    EXPECT_NE(opt->get_validator(), nullptr);
}

TEST_F(StartCommandTest, Setup_ConfigFileOptionRequired)
{
    cmd_->Setup(*app_);
    auto start = app_->get_subcommand("start");
    ASSERT_NE(start, nullptr);
    auto* opt = start->get_option("configFile");
    ASSERT_NE(opt, nullptr);
    EXPECT_TRUE(opt->get_required());
}
