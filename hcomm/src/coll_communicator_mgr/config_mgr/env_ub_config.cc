/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "env_ub_config.h"

#include "log.h"

namespace hccl {

// EnvUbConfig

HcclResult EnvUbConfig::Parse()
{
    std::lock_guard<std::mutex> lock(parseMutex_);
    if (parsed_) {
        return parseRet_;
    }
    HcclResult ret = ubMultiChannelNum_.Parse();
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[HCCL_ENV] HCCL_UB_MULTI_CHANNEL_NUM parse failed, ret[%d].", ret);
        return ret;
    }
    HCCL_RUN_INFO(
        "[HCCL_ENV] HCCL_UB_MULTI_CHANNEL_NUM set by %s to [%u]", ubMultiChannelNum_.GetSource(),
        ubMultiChannelNum_.Get());
    parsed_ = true;
    parseRet_ = HCCL_SUCCESS;
    return HCCL_SUCCESS;
}

void EnvUbConfig::ResetParsed()
{
    std::lock_guard<std::mutex> lock(parseMutex_);
    parsed_ = false;
    parseRet_ = HCCL_SUCCESS;
}

EnvUbConfig& GetEnvUbConfig()
{
    static EnvUbConfig ubConfig;
    return ubConfig;
}

} // namespace hccl
