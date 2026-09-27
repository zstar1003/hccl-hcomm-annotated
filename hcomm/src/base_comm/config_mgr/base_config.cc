/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "base_config.h"
#include "log.h"
#include "hccl_log_keywords.h"

namespace hcomm {

// 环境变量错误上报：输出日志 + 上报故障码，返回错误码
HcclResult ReportEnvError(const char* envName, const std::string& envValue, const std::string& reason)
{
    std::string errMsg = std::string("Env config \"") + envName + "\" value \"" + envValue + "\" " + reason;
    HCCL_ERROR("[%s][%s] %s", LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_ENV_CONFIG.c_str(), errMsg.c_str());
    RPT_ENV_ERR(
        true, "EI0001", std::vector<std::string>({"value", "env", "expect"}),
        std::vector<std::string>({envValue, envName, errMsg}));
    return HCCL_E_PARA;
}

// EnvRdmaConfig

HcclResult EnvRdmaConfig::EnsureParsed()
{
    if (isParsed_.load(std::memory_order_acquire)) {
        return HCCL_SUCCESS;
    }
    std::lock_guard<std::mutex> lock(parseMutex_);
    if (isParsed_.load(std::memory_order_relaxed)) { // double-check
        return HCCL_SUCCESS;
    }
    CHK_RET(taCtpUbTimeOut_.Parse());
    HCCL_RUN_INFO(
        "[HCCL_ENV] HCOMM_TA_CTP_UB_TIMEOUT set by %s to [%u]", taCtpUbTimeOut_.GetSource(), taCtpUbTimeOut_.Get());

    CHK_RET(taRtpUbTimeOut_.Parse());
    HCCL_RUN_INFO(
        "[HCCL_ENV] HCOMM_TA_RTP_UB_TIMEOUT set by %s to [%u]", taRtpUbTimeOut_.GetSource(), taRtpUbTimeOut_.Get());

    CHK_RET(taRtpUboeTimeOut_.Parse());
    HCCL_RUN_INFO(
        "[HCCL_ENV] HCOMM_TA_RTP_UBOE_TIMEOUT set by %s to [%u]", taRtpUboeTimeOut_.GetSource(),
        taRtpUboeTimeOut_.Get());

    isParsed_.store(true, std::memory_order_release);
    return HCCL_SUCCESS;
}

void EnvRdmaConfig::ResetParsed()
{
    std::lock_guard<std::mutex> lock(parseMutex_);
    isParsed_.store(false, std::memory_order_release);
}

HcclResult EnvRdmaConfig::GetTaCtpUbTimeOut(uint32_t& value)
{
    CHK_RET(EnsureParsed());
    value = taCtpUbTimeOut_.Get();
    return HCCL_SUCCESS;
}

HcclResult EnvRdmaConfig::GetTaRtpUbTimeOut(uint32_t& value)
{
    CHK_RET(EnsureParsed());
    value = taRtpUbTimeOut_.Get();
    return HCCL_SUCCESS;
}

HcclResult EnvRdmaConfig::GetTaRtpUboeTimeOut(uint32_t& value)
{
    CHK_RET(EnsureParsed());
    value = taRtpUboeTimeOut_.Get();
    return HCCL_SUCCESS;
}

} // namespace hcomm
