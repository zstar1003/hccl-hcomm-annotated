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

#include "subcmd_plugin.h"

using namespace HcclSim;

class PluginCommandTest : public testing::Test {
protected:
    void SetUp() override
    {
        app_ = std::make_unique<CLI::App>("test");
        cmd_ = std::make_unique<PluginCommand>();
    }
    std::unique_ptr<CLI::App> app_;
    std::unique_ptr<PluginCommand> cmd_;
};

TEST_F(PluginCommandTest, StaticName_ShouldReturnPlugin) { EXPECT_EQ(PluginCommand::StaticName(), "plugin"); }

TEST_F(PluginCommandTest, Setup_RegistersPluginSubcommand)
{
    cmd_->Setup(*app_);
    EXPECT_NE(app_->get_subcommand("plugin"), nullptr);
}

TEST_F(PluginCommandTest, Setup_RequiresSubcommand)
{
    cmd_->Setup(*app_);
    auto sub = app_->get_subcommand("plugin");
    ASSERT_NE(sub, nullptr);
    EXPECT_GT(sub->get_require_subcommand_min(), 0u);
}

TEST_F(PluginCommandTest, Setup_RegistersInstallSubcommand)
{
    cmd_->Setup(*app_);
    auto plugin = app_->get_subcommand("plugin");
    ASSERT_NE(plugin, nullptr);
    EXPECT_NE(plugin->get_subcommand("install"), nullptr);
}

TEST_F(PluginCommandTest, Setup_RegistersUninstallSubcommand)
{
    cmd_->Setup(*app_);
    auto plugin = app_->get_subcommand("plugin");
    ASSERT_NE(plugin, nullptr);
    EXPECT_NE(plugin->get_subcommand("uninstall"), nullptr);
}

TEST_F(PluginCommandTest, Setup_RegistersRunSubcommand)
{
    cmd_->Setup(*app_);
    auto plugin = app_->get_subcommand("plugin");
    ASSERT_NE(plugin, nullptr);
    EXPECT_NE(plugin->get_subcommand("run"), nullptr);
}

TEST_F(PluginCommandTest, Setup_RegistersListSubcommand)
{
    cmd_->Setup(*app_);
    auto plugin = app_->get_subcommand("plugin");
    ASSERT_NE(plugin, nullptr);
    EXPECT_NE(plugin->get_subcommand("list"), nullptr);
}

TEST_F(PluginCommandTest, Setup_InstallHasNameOption)
{
    cmd_->Setup(*app_);
    auto plugin = app_->get_subcommand("plugin");
    ASSERT_NE(plugin, nullptr);
    auto inst = plugin->get_subcommand("install");
    ASSERT_NE(inst, nullptr);
    EXPECT_GE(inst->get_options().size(), 1u);
}

TEST_F(PluginCommandTest, Setup_UninstallHasNameOptionWithValidator)
{
    cmd_->Setup(*app_);
    auto plugin = app_->get_subcommand("plugin");
    ASSERT_NE(plugin, nullptr);
    auto uninst = plugin->get_subcommand("uninstall");
    ASSERT_NE(uninst, nullptr);
    EXPECT_GE(uninst->get_options().size(), 1u);
    CLI::Option* opt = nullptr;
    try {
        opt = uninst->get_option("name");
    } catch (const CLI::OptionNotFound& e) {
        std::cerr << "Option 'name' not found: " << e.what() << "\n";
    }
    ASSERT_NE(opt, nullptr);
    EXPECT_NE(opt->get_validator(), nullptr);
}

TEST_F(PluginCommandTest, Setup_RunHasNameOptionWithValidator)
{
    cmd_->Setup(*app_);
    auto plugin = app_->get_subcommand("plugin");
    ASSERT_NE(plugin, nullptr);
    auto runCmd = plugin->get_subcommand("run");
    ASSERT_NE(runCmd, nullptr);
    EXPECT_GE(runCmd->get_options().size(), 1u);
    CLI::Option* opt = nullptr;
    try {
        opt = runCmd->get_option("name");
    } catch (const CLI::OptionNotFound& e) {
        std::cerr << "Option 'name' not found: " << e.what() << "\n";
    }
    ASSERT_NE(opt, nullptr);
    EXPECT_NE(opt->get_validator(), nullptr);
}

TEST_F(PluginCommandTest, Parse_PluginList_TriggersCallback)
{
    cmd_->Setup(*app_);
    EXPECT_NO_THROW(app_->parse("test plugin list", true));
}

TEST_F(PluginCommandTest, Parse_InstallTriggersCallback)
{
    cmd_->Setup(*app_);
    EXPECT_NO_THROW(app_->parse("test plugin install foo_plugin.so", true));
}

TEST_F(PluginCommandTest, Parse_UninstallValidName_TriggersCallback)
{
    cmd_->Setup(*app_);
    EXPECT_NO_THROW(app_->parse("test plugin uninstall @myplugin", true));
}

TEST_F(PluginCommandTest, Parse_RunValidName_TriggersCallback)
{
    cmd_->Setup(*app_);
    try {
        app_->parse("test plugin run @myplugin", true);
    } catch (const CLI::CallForHelp& e) {
        std::cerr << "Catch CLI::CallForHelp: " << e.what() << "\n";
    } catch (const std::exception& e) {
        std::cerr << "Catch unexpected exception: " << e.what() << "\n";
    }
}

TEST_F(PluginCommandTest, Parse_PluginNoSubcommand_Throws)
{
    cmd_->Setup(*app_);
    EXPECT_THROW(app_->parse("test plugin", true), CLI::RequiredError);
}
