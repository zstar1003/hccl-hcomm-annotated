/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cmd_cluster_model_utils.h"
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>

#include "sim_common_api.h"
#include "sim_log.h"
#include "yaml-cpp/yaml.h"

namespace {
constexpr const char* HVM_MODEL_ENV_KEY = "HCCL_VM_MODEL";
constexpr const char* HVM_POD_NUM_ENV_KEY = "HCCL_VM_POD_NUM";
constexpr const char* HVM_SERVER_COUNT_ENV_KEY = "HCCL_VM_SERVER_COUNT";
constexpr const char* HVM_RANK_NUM_ENV_KEY = "HCCL_VM_RANK_NUM";
constexpr const char* HVM_PROCESSES_PER_SERVER_ENV_KEY = "HCCL_VM_PROCESSES_PER_SERVER";

bool SetEnvValue(const char* key, const std::string& value)
{
    if (setenv(key, value.c_str(), 1) == 0) {
        return true;
    }
    HCCL_VM_ERROR("setenv failed: {}={}, errno={}, err={}", key, value, errno, std::strerror(errno));
    return false;
}

bool ParseYamlTopoImpl(const std::string& fileName, TopoMeta& topo)
{
    try {
        std::string filePath = InstallPath::ResolveToInstallRoot("config/topo_meta/" + fileName + ".yaml");
        YAML::Node root = YAML::LoadFile(filePath);

        if (!root["meta"]) {
            HCCL_VM_ERROR("YAML : 'meta' node not found.");
            return false;
        }
        uint32_t podNum = root["meta"]["podNum"].as<uint32_t>();
        uint32_t serNum = root["meta"]["serNum"].as<uint32_t>();
        uint32_t rankNum = root["meta"]["rankNum"].as<uint32_t>();
        HCCL_VM_DEBUG("PodNum: {}, SerNum: {}, RankNum: {}", podNum, serNum, rankNum);
        topo.reserve(podNum);
        if (podNum <= 0 || podNum > 1024 || serNum <= 0 || serNum > 1024 || rankNum <= 0 || rankNum > 1024) {
            HCCL_VM_ERROR("YAML : 'meta' number not surport, please check your config.yaml.");
            return false;
        }
        if (root["topology"] && root["topology"].IsSequence()) {
            for (const auto& pod : root["topology"]) {
                SuperPodMeta superPodMeta;
                if (pod["podId"] && pod["servers"] && pod["servers"].IsSequence()) {
                    for (const auto& ser : pod["servers"]) {
                        ServerMeta serverMeta;
                        uint32_t serverId = 0;
                        if (ser["serId"] && ser["ranks"] && ser["ranks"].IsSequence()) {
                            serverId = ser["serId"].as<uint32_t>();
                            serverMeta = ser["ranks"].as<std::vector<uint32_t>>();
                        }
                        superPodMeta.push_back(serverMeta);
                    }
                }
                topo.push_back(superPodMeta);
            }
        }
        return true;
    } catch (const YAML::Exception& e) {
        HCCL_VM_ERROR("Exception when parsing YAML: {}", e.what());
        return false;
    }
}
} // namespace

bool ParseYamlTopo(const std::string& fileName, TopoMeta& topo) { return ParseYamlTopoImpl(fileName, topo); }
