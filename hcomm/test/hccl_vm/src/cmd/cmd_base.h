/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_VM_COMMAND_BASE_H
#define HCCL_VM_COMMAND_BASE_H

#include <CLI11.hpp>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace HcclSim {
class CommandBase {
public:
    virtual ~CommandBase() = default;
    virtual void Setup(CLI::App& app) = 0;
};

// 命令注册器
class CommandRegistry {
public:
    using CommandCreator = std::function<std::unique_ptr<CommandBase>()>;

    static void RegisteCommand(const std::string& name, CommandCreator creator);
    static std::vector<std::unique_ptr<CommandBase>> CreateAll();

private:
    static auto& GetCreators();
};

// 命令自动注册
template <typename CommandType>
class CommandAutoRegister {
public:
    CommandAutoRegister()
    {
        CommandRegistry::RegisteCommand(CommandType::StaticName(), []() -> std::unique_ptr<CommandBase> {
            return std::make_unique<CommandType>();
        });
    }
};
} // namespace HcclSim

#endif
