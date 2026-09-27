/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <string>

#include "cmd_base_utils.h"
#include "sim_common_defs.h"
#include "sim_log.h"
#include "subcmd_plugin.h"

namespace HcclSim {
void PluginCommand::Setup(CLI::App& app)
{
    auto sub_plugin = app.add_subcommand("plugin", "插件管理子命令");
    sub_plugin->require_subcommand(1);
    // install
    auto plugin_install = sub_plugin->add_subcommand("install", "安装插件");
    plugin_install->add_option("name", plugName, "插件文件名")
        ->required()
        ->check([](const std::string& value) -> std::string {
            if (value.length() > 1 && value[0] == '@') {
                return ""; // 返回空串表示通过
            }
            return "[HVM] [ERROR] Install plugin : Invalid format! Plugin name must start with '@' (e.g., @myplugin).";
        });
    plugin_install->callback([&]() {
        auto ret = InstallUserPlugin(plugName);
    });
    // uninstall
    auto plugin_uninstall = sub_plugin->add_subcommand("uninstall", "卸载插件");
    plugin_uninstall->add_option("name", plugName, "插件文件名")
        ->required()
        ->check([](const std::string& value) -> std::string {
            if (value.length() > 1 && value[0] == '@') {
                return ""; // 返回空串表示通过
            }
            return "[HVM] [ERROR] Uninstall plugin : Invalid format! Plugin name must start with '@' (e.g., "
                   "@myplugin).";
        });
    plugin_uninstall->callback([&]() {
        auto ret = UninstallUserPlugin(plugName);
    });
    // run
    auto plugin_run = sub_plugin->add_subcommand("run", "运行插件"); // todo
    plugin_run->add_option("name", plugName, "插件文件名")
        ->required()
        ->check([](const std::string& value) -> std::string {
            if (value.length() > 1 && value[0] == '@') {
                return ""; // 返回空串表示通过
            }
            return "[HVM] [ERROR] Run plugin : Invalid format! Plugin name must start with '@' (e.g., @myplugin).";
        });
    plugin_run->callback([&]() {
        auto ret = RunUserPlugin(plugName);
    });
    // list
    auto plugin_list = sub_plugin->add_subcommand("list", "展示已安装插件");
    plugin_list->callback([&]() {
        ShowUserPlugin();
    });
}

static inline CommandAutoRegister<PluginCommand> g_plugin_cmd_reg{};
} // namespace HcclSim
