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

#include "subcmd_table.h"

using namespace HcclSim;

class TableCommandTest : public testing::Test {
protected:
    void SetUp() override
    {
        app_ = std::make_unique<CLI::App>("test");
        cmd_ = std::make_unique<TableCommand>();
    }
    std::unique_ptr<CLI::App> app_;
    std::unique_ptr<TableCommand> cmd_;
};

TEST_F(TableCommandTest, StaticName_ShouldReturnTable) { EXPECT_EQ(TableCommand::StaticName(), "table"); }

TEST_F(TableCommandTest, Setup_RegistersTableSubcommand)
{
    cmd_->Setup(*app_);
    auto sub = app_->get_subcommand("table");
    ASSERT_NE(sub, nullptr);
    EXPECT_EQ(sub->get_name(), "table");
}

TEST_F(TableCommandTest, Setup_RequiresSubcommand)
{
    cmd_->Setup(*app_);
    auto sub = app_->get_subcommand("table");
    ASSERT_NE(sub, nullptr);
    EXPECT_GT(sub->get_require_subcommand_min(), 0u);
}

TEST_F(TableCommandTest, Setup_RegistersShowSubcommand)
{
    cmd_->Setup(*app_);
    auto table = app_->get_subcommand("table");
    ASSERT_NE(table, nullptr);
    EXPECT_NE(table->get_subcommand("show"), nullptr);
}

TEST_F(TableCommandTest, Setup_RegistersUpdateSubcommand)
{
    cmd_->Setup(*app_);
    auto table = app_->get_subcommand("table");
    ASSERT_NE(table, nullptr);
    EXPECT_NE(table->get_subcommand("update"), nullptr);
}

TEST_F(TableCommandTest, Setup_ShowHasNameOption)
{
    cmd_->Setup(*app_);
    auto table = app_->get_subcommand("table");
    ASSERT_NE(table, nullptr);
    auto show = table->get_subcommand("show");
    ASSERT_NE(show, nullptr);
    EXPECT_GE(show->get_options().size(), 1u);
}

TEST_F(TableCommandTest, Setup_UpdateHasRequiredOptions)
{
    cmd_->Setup(*app_);
    auto table = app_->get_subcommand("table");
    ASSERT_NE(table, nullptr);
    auto update = table->get_subcommand("update");
    ASSERT_NE(update, nullptr);
    EXPECT_GE(update->get_options().size(), 4u);
}

TEST_F(TableCommandTest, DefaultMembers_AreZeroInitialized)
{
    EXPECT_TRUE(cmd_->showStr.empty());
    EXPECT_TRUE(cmd_->tableName.empty());
    EXPECT_TRUE(cmd_->columnName.empty());
    EXPECT_TRUE(cmd_->newValue.empty());
    EXPECT_EQ(cmd_->rowId, 0u);
}

TEST_F(TableCommandTest, Parse_ShowSpecific_ShowStrSet)
{
    cmd_->Setup(*app_);
    try {
        app_->parse("test table show MyDeviceTable", true);
    } catch (const CLI::CallForHelp& e) {
        std::cerr << "Catch CLI::CallForHelp: " << e.what() << "\n";
    } catch (const std::exception& e) {
        std::cerr << "Catch unexpected exception: " << e.what() << "\n";
    }
    EXPECT_EQ(cmd_->showStr, "MyDeviceTable");
}

TEST_F(TableCommandTest, Parse_ShowAll_ShowStrSet)
{
    cmd_->Setup(*app_);
    try {
        app_->parse("test table show all", true);
    } catch (const CLI::CallForHelp& e) {
        std::cerr << "Catch CLI::CallForHelp: " << e.what() << "\n";
    } catch (const std::exception& e) {
        std::cerr << "Catch unexpected exception: " << e.what() << "\n";
    }
    EXPECT_EQ(cmd_->showStr, "all");
}

TEST_F(TableCommandTest, Parse_Update_SetsAllMembers)
{
    cmd_->Setup(*app_);
    try {
        app_->parse("test table update --table Device --id 42 --column soc_version --value V100", true);
    } catch (const CLI::CallForHelp& e) {
        std::cerr << "Catch CLI::CallForHelp: " << e.what() << "\n";
    } catch (const std::exception& e) {
        std::cerr << "Catch unexpected exception: " << e.what() << "\n";
    }
    EXPECT_EQ(cmd_->tableName, "Device");
    EXPECT_EQ(cmd_->rowId, 42u);
    EXPECT_EQ(cmd_->newValue, "V100");
}

TEST_F(TableCommandTest, Parse_Update_RequiredOptions)
{
    cmd_->Setup(*app_);
    EXPECT_THROW(app_->parse("test table update", true), CLI::RequiredError);
}
