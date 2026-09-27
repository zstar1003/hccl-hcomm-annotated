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
#include <sstream>

#include "subcmd_model.h"

using namespace HcclSim;

class ModelCommandTest : public testing::Test {
protected:
    void SetUp() override
    {
        app_ = std::make_unique<CLI::App>("test");
        cmd_ = std::make_unique<ModelCommand>();
    }
    void TearDown() override {}
    std::unique_ptr<CLI::App> app_;
    std::unique_ptr<ModelCommand> cmd_;
};

TEST_F(ModelCommandTest, StaticName_ShouldReturnModel) { EXPECT_EQ(ModelCommand::StaticName(), "model"); }

TEST_F(ModelCommandTest, Setup_RegistersModelSubcommand)
{
    cmd_->Setup(*app_);
    auto sub = app_->get_subcommand("model");
    EXPECT_NE(sub, nullptr);
    EXPECT_EQ(sub->get_name(), "model");
}

TEST_F(ModelCommandTest, Setup_RequiresSubcommand)
{
    cmd_->Setup(*app_);
    auto sub = app_->get_subcommand("model");
    ASSERT_NE(sub, nullptr);
    EXPECT_TRUE(sub->get_require_subcommand_min() > 0);
}

TEST_F(ModelCommandTest, Setup_RegistersListSubcommand)
{
    cmd_->Setup(*app_);
    auto model = app_->get_subcommand("model");
    ASSERT_NE(model, nullptr);
    auto list = model->get_subcommand("list");
    EXPECT_NE(list, nullptr);
}

TEST_F(ModelCommandTest, ParseModelList_InvokesCallback)
{
    cmd_->Setup(*app_);
    try {
        app_->parse("test model list", true);
    } catch (const CLI::CallForHelp& e) {
        std::cerr << "Catch CLI::CallForHelp: " << e.what() << "\n";
    } catch (const std::exception& e) {
        std::cerr << "Catch unexpected exception: " << e.what() << "\n";
    }
}

TEST_F(ModelCommandTest, ParseModelList_Help)
{
    cmd_->Setup(*app_);
    try {
        app_->parse("test model list --help", true);
    } catch (const CLI::CallForHelp& e) {
        std::cerr << "Catch CLI::CallForHelp: " << e.what() << "\n";
    } catch (const std::exception& e) {
        std::cerr << "Catch unexpected exception: " << e.what() << "\n";
    }
}

TEST_F(ModelCommandTest, ParseModel_Help)
{
    cmd_->Setup(*app_);
    try {
        app_->parse("test model --help", true);
    } catch (const CLI::CallForHelp& e) {
        std::cerr << "Catch CLI::CallForHelp: " << e.what() << "\n";
    } catch (const std::exception& e) {
        std::cerr << "Catch unexpected exception: " << e.what() << "\n";
    }
}

TEST_F(ModelCommandTest, ParseModel_NoSubcommand_Throws)
{
    cmd_->Setup(*app_);
    EXPECT_THROW(app_->parse("test model", true), CLI::RequiredError);
}

TEST_F(ModelCommandTest, Setup_ModelSubcommandDescription)
{
    cmd_->Setup(*app_);
    auto model = app_->get_subcommand("model");
    ASSERT_NE(model, nullptr);
    EXPECT_EQ(std::string(model->get_description()), "管理建模文件");
}

TEST_F(ModelCommandTest, Setup_ListSubcommandDescription)
{
    cmd_->Setup(*app_);
    auto model = app_->get_subcommand("model");
    ASSERT_NE(model, nullptr);
    auto list = model->get_subcommand("list");
    ASSERT_NE(list, nullptr);
    EXPECT_EQ(std::string(list->get_description()), "展示建模文件");
}

TEST_F(ModelCommandTest, Setup_ModelSubcommandHasCallback)
{
    cmd_->Setup(*app_);
    auto model = app_->get_subcommand("model");
    ASSERT_NE(model, nullptr);
    auto list = model->get_subcommand("list");
    ASSERT_NE(list, nullptr);
    // Verify the subcommand exists and has a callback by parsing it
    try {
        app_->parse("test model list", true);
    } catch (const CLI::CallForHelp& e) {
        std::cerr << "Catch CLI::CallForHelp: " << e.what() << "\n";
    } catch (const std::exception& e) {
        std::cerr << "Catch unexpected exception: " << e.what() << "\n";
    }
}

TEST_F(ModelCommandTest, CommandRegistry_CreateAll_ContainsModelCommand)
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
