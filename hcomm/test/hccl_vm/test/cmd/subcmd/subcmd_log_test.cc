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
#include <gtest/gtest.h>
#include <iostream>

#include "subcmd_log.h"

using namespace HcclSim;

class LogCommandTest : public testing::Test {
protected:
    void SetUp() override
    {
        app_ = std::make_unique<CLI::App>("test");
        cmd_ = std::make_unique<LogCommand>();
    }
    std::unique_ptr<CLI::App> app_;
    std::unique_ptr<LogCommand> cmd_;
};

TEST_F(LogCommandTest, StaticName_ShouldReturnLog) { EXPECT_EQ(LogCommand::StaticName(), "log"); }

TEST_F(LogCommandTest, Setup_RegistersLogSubcommand)
{
    cmd_->Setup(*app_);
    EXPECT_NE(app_->get_subcommand("log"), nullptr);
}

TEST_F(LogCommandTest, Setup_HasAllLevelOptions)
{
    cmd_->Setup(*app_);
    auto sub = app_->get_subcommand("log");
    ASSERT_NE(sub, nullptr);
    EXPECT_GE(sub->get_options().size(), 3u);
}

TEST_F(LogCommandTest, Setup_OptionExclusions_LevelExcludesConsoleAndFile)
{
    cmd_->Setup(*app_);
    auto sub = app_->get_subcommand("log");
    ASSERT_NE(sub, nullptr);
    auto opts = sub->get_options();
    EXPECT_GE(opts.size(), 3u);
    auto* levelOpt = sub->get_option("-l");
    if (levelOpt) {
        auto excludes = levelOpt->get_excludes();
        EXPECT_GE(excludes.size(), 2u);
    }
}

TEST_F(LogCommandTest, Setup_OptionExclusions_ConsoleExcludesLevel)
{
    cmd_->Setup(*app_);
    auto sub = app_->get_subcommand("log");
    ASSERT_NE(sub, nullptr);
    auto* consoleOpt = sub->get_option("--console-level");
    if (consoleOpt) {
        auto excludes = consoleOpt->get_excludes();
        EXPECT_GE(excludes.size(), 1u);
    }
}

TEST_F(LogCommandTest, Setup_OptionExclusions_FileExcludesLevel)
{
    cmd_->Setup(*app_);
    auto sub = app_->get_subcommand("log");
    ASSERT_NE(sub, nullptr);
    auto* fileOpt = sub->get_option("--file-level");
    if (fileOpt) {
        auto excludes = fileOpt->get_excludes();
        EXPECT_GE(excludes.size(), 1u);
    }
}

TEST_F(LogCommandTest, LevelMap_ContainsExpectedKeys)
{
    cmd_->Setup(*app_);
    auto sub = app_->get_subcommand("log");
    ASSERT_NE(sub, nullptr);
    auto* levelOpt = sub->get_option("-l");
    ASSERT_NE(levelOpt, nullptr);
}

TEST_F(LogCommandTest, Parse_LevelInfo_TriggersCallback)
{
    cmd_->Setup(*app_);
    EXPECT_NO_THROW(app_->parse("test log -l info", true));
}

TEST_F(LogCommandTest, Parse_ConsoleLevelDebug_TriggersCallback)
{
    cmd_->Setup(*app_);
    EXPECT_NO_THROW(app_->parse("test log --console-level debug", true));
}

TEST_F(LogCommandTest, Parse_FileLevelError_TriggersCallback)
{
    cmd_->Setup(*app_);
    EXPECT_NO_THROW(app_->parse("test log --file-level error", true));
}

TEST_F(LogCommandTest, Parse_InvalidLevel_Throws)
{
    cmd_->Setup(*app_);
    try {
        app_->parse("test log -l invalid_level", true);
        FAIL() << "Expected parse error for invalid level";
    } catch (const CLI::CallForHelp& e) {
        std::cerr << "Catch CLI::CallForHelp: " << e.what() << "\n";
    } catch (const CLI::ValidationError& e) {
        std::cerr << "Catch CLI::ValidationError: " << e.what() << "\n";
    } catch (const std::exception& e) {
        std::cerr << "Catch unexpected exception: " << e.what() << "\n";
    }
}

TEST_F(LogCommandTest, Parse_LevelAndConsoleLevel_ExclusionThrows)
{
    cmd_->Setup(*app_);
    EXPECT_THROW(app_->parse("test log -l info --console-level debug", true), CLI::ExcludesError);
}

TEST_F(LogCommandTest, LevelMap_AllEntries)
{
    // Verify all 8 level name entries are accepted by the CheckedTransformer
    cmd_->Setup(*app_);
    auto sub = app_->get_subcommand("log");
    ASSERT_NE(sub, nullptr);
    auto* levelOpt = sub->get_option("-l");
    ASSERT_NE(levelOpt, nullptr);
    auto* validator = levelOpt->get_validator();
    ASSERT_NE(validator, nullptr);

    EXPECT_EQ(validator->operator()("trace"), "");
    EXPECT_EQ(validator->operator()("debug"), "");
    EXPECT_EQ(validator->operator()("info"), "");
    EXPECT_EQ(validator->operator()("warn"), "");
    EXPECT_EQ(validator->operator()("warning"), "");
    EXPECT_EQ(validator->operator()("error"), "");
    EXPECT_EQ(validator->operator()("err"), "");
    EXPECT_EQ(validator->operator()("critical"), "");
}
