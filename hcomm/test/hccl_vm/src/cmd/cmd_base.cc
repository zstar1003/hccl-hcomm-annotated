/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cmd_base.h"

namespace HcclSim {
auto& CommandRegistry::GetCreators()
{
    static std::vector<std::pair<std::string, CommandCreator>> creators;
    return creators;
}

void CommandRegistry::RegisteCommand(const std::string& name, CommandCreator creator)
{
    GetCreators().emplace_back(name, std::move(creator));
}

std::vector<std::unique_ptr<CommandBase>> CommandRegistry::CreateAll()
{
    std::vector<std::unique_ptr<CommandBase>> commands;
    commands.reserve(GetCreators().size());
    for (const auto& [name, creator] : GetCreators()) {
        commands.push_back(creator());
    }
    return commands;
}
} // namespace HcclSim
