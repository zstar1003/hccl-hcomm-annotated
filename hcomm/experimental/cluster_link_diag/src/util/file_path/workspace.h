/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once
#include <iostream>
#include <string>
#include <filesystem>
#define PROJECT_NAME "disp_probe"
static std::string DEFAULT_INFO_JSON = "./control_json/910b2_info.json";
// 获取可执行文件绝对路径
std::string get_exec_path();

// 根据工程名截取工程路径
std::string get_project_path(const std::string& projectName);
std::string get_project_path();
bool make_abs_path(std::string& path);

std::string get_output_path();
