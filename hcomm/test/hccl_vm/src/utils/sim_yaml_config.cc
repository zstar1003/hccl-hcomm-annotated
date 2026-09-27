/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "sim_yaml_config.h"

#include <fstream>
#include <string>

#if !defined(NO_YAML_CONFIG) && defined(HAVE_YAML_CPP)
#include "yaml-cpp/yaml.h"
#endif

#include "sim_log.h"

bool LoadYamlStringMap(
    const std::string& yamlPath, const std::string& nodeName, std::map<std::string, std::string>& out)
{
#if !defined(NO_YAML_CONFIG) && defined(HAVE_YAML_CPP)
    try {
        if (!std::ifstream(yamlPath).good()) {
            return false;
        }
        YAML::Node root = YAML::LoadFile(yamlPath);
        if (!root[nodeName]) {
            return false;
        }
        const YAML::Node& node = root[nodeName];
        for (auto it = node.begin(); it != node.end(); ++it) {
            out[it->first.as<std::string>()] = it->second.as<std::string>();
        }
        return true;
    } catch (const std::exception& e) {
        HCCL_VM_ERROR("Failed to parse YAML file {}: {}", yamlPath, e.what());
        return false;
    }
#else
    (void)yamlPath;
    (void)nodeName;
    (void)out;
    return false;
#endif
}
