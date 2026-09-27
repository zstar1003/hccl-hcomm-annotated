/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "plf_debug_config.h"
#include "log.h"
#include <sstream>
#include <atomic>
#include <mutex>

namespace Hccl {

static std::atomic<u64> g_plfDebugConfig{0};
static std::once_flag g_parseOnce;

static u64 ParseDebugConfig(const char* envName, u64 domainMask)
{
    char* env = getenv(envName);
    if (env == nullptr) {
        return 0;
    }
    std::string configDup(env);

    bool invert = (!configDup.empty() && configDup.front() == '^');
    // 第一个字符是'^', 使用取反模式，用户配置的项关闭，未配置的项打开
    u64 result = invert ? domainMask : 0ULL;
    if (invert) {
        configDup.erase(configDup.begin()); // 去掉'^'符号
    }

    std::istringstream stream(configDup);
    std::string subConfig;
    while (std::getline(stream, subConfig, ',')) {
        if (subConfig.empty()) {
            continue;
        }
        u64 mask = 0;
        if (((domainMask & PLF_TASK) != 0) && strcasecmp(subConfig.c_str(), "TASK") == 0) {
            mask = PLF_TASK;
        } else if (((domainMask & PLF_ALG) != 0) && strcasecmp(subConfig.c_str(), "ALG") == 0) {
            mask = PLF_ALG;
        } else if (((domainMask & PLF_RES) != 0) && strcasecmp(subConfig.c_str(), "RESOURCE") == 0) {
            mask = PLF_RES;
        } else if (((domainMask & PLF_DATA_OP) != 0) && strcasecmp(subConfig.c_str(), "DATA_OP") == 0) {
            mask = PLF_DATA_OP;
        } else if (((domainMask & PLF_CHANNEL) != 0) && strcasecmp(subConfig.c_str(), "CHANNEL") == 0) {
            mask = PLF_CHANNEL;
        } else {
            HCCL_ERROR("%s:%s subConfig:%s is not supported", envName, env, subConfig.c_str());
            return 0;
        }
        result = invert ? (result & ~mask) : (result | mask);
    }
    HCCL_RUN_INFO("[HCCL_ENV] %s set by [%s] to [0x%llx]", envName, env, result);
    return result;
}

void EnvPlfDebugConfig::Parse()
{
    plfDebugConfig_ = ParseDebugConfig("HCCL_DEBUG_CONFIG", PLF_TASK | PLF_ALG | PLF_RES);
    plfDebugConfig_ |= ParseDebugConfig("HCOMM_DEBUG_CONFIG", PLF_TASK | PLF_DATA_OP | PLF_RES | PLF_CHANNEL);
    HCCL_RUN_INFO("[HCCL_ENV] plfDebugConfig set to [0x%llx]", plfDebugConfig_);
}

u64 EnvPlfDebugConfig::GetConfigValue() const { return plfDebugConfig_; }

u64 GetPlfDebugConfigValue()
{
    std::call_once(g_parseOnce, []() {
        EnvPlfDebugConfig cfg;
        cfg.Parse();
        g_plfDebugConfig.store(cfg.GetConfigValue(), std::memory_order_relaxed);
    });
    return g_plfDebugConfig.load(std::memory_order_relaxed);
}

void SetPlfDebugConfigValue(u64 value) { g_plfDebugConfig.store(value, std::memory_order_relaxed); }

} // namespace Hccl
