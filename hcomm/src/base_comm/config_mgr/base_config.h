/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_BASE_CONFIG_H
#define HCOMM_BASE_CONFIG_H

#include <cstdint>
#include <cstdlib>
#include <functional>
#include <mutex>
#include <string>
#include <utility>
#include <exception>
#include <vector>
#include <algorithm>
#include <atomic>
#include <cctype>
#include <limits>
#include <stdexcept>
#include <type_traits>

#include "log.h"
#include "adapter_error_manager_pub.h"
#include "hccl_types.h"

namespace hcomm {

/**
 * @brief 获取环境变量并立即拷贝为 std::string，封装 getenv 的不可重入风险。
 */
inline std::string GetEnv(const char* name)
{
    if (name == nullptr) {
        return std::string();
    }
    const char* val = getenv(name);
    if (val == nullptr) {
        return std::string();
    }
    return std::string(val);
}

/*------------------- 通用解析器与校验器 -------------------
 *  新增环境变量字段时，直接复用这些模板，无需为每种类型手写 static 方法。
 *  例如：
 *    EnvField<uint32_t> myField{"MY_ENV", 10, StrToNum<uint32_t>, MakeRangeValidator(0U, 31U)};
 *    EnvField<uint8_t>  myField2{"MY_ENV2", 0, StrToNum<uint8_t>};
 *------------------------------------------------------------------------------------------------*/

/// 通用字符串→整数解析器：先检查全数字，再调用 std::stoul。
/// 对 EnvField<uint32_t>::Parser (返回 T, bool& parseOk) 签名适配。
template <typename T>
typename std::enable_if<std::is_integral<T>::value, T>::type StrToNum(const std::string& s, bool& parseOk)
{
    if (s.empty() || !std::all_of(s.begin(), s.end(), [](unsigned char c) {
            return ::isdigit(c) != 0;
        })) {
        parseOk = false;
        return T{};
    }
    try {
        unsigned long val = std::stoul(s);
        if (val > std::numeric_limits<T>::max()) {
            parseOk = false;
            return T{};
        }
        parseOk = true;
        return static_cast<T>(val);
    } catch (const std::exception&) {
        parseOk = false;
        return T{};
    }
}

/// 通用闭区间校验器工厂。
/// 返回 EnvField<T>::Validator 签名 (bool(const T&)) 的 lambda。
template <typename T>
typename std::function<bool(const T&)> MakeRangeValidator(T min, T max)
{
    return [min, max](const T& v) -> bool {
        return v >= min && v <= max;
    };
}

/// 环境变量错误上报（输出 HCCL_ERROR + RPT_ENV_ERR），返回错误码。
/// 不抛异常，由调用方逐层返回错误码。
HcclResult ReportEnvError(const char* envName, const std::string& envValue, const std::string& reason);

/**
 * @brief 轻量环境变量字段，独立实现。
 *
 * 每个字段自包含：环境变量名、默认值、解析器、校验器。
 * Parse() 负责 getenv + 解析 + 校验，失败时输出 HCCL_ERROR 日志并返回错误码。
 */
template <typename T>
class EnvField {
public:
    using Parser = std::function<T(const std::string&, bool&)>;
    using Validator = std::function<bool(const T&)>;

    EnvField(const char* name, T defaultValue, Parser parser, Validator validator = nullptr)
        : name_(name),
          value_(std::move(defaultValue)),
          defaultValue_(std::move(defaultValue)),
          parser_(std::move(parser)),
          validator_(std::move(validator))
    {}

    HcclResult Parse()
    {
        std::string envStr = GetEnv(name_);
        if (envStr.empty()) {
            isSetByEnv_ = false;
            value_ = defaultValue_;
            return HCCL_SUCCESS;
        }
        if (!parser_) {
            return ReportEnvError(name_, envStr, "no parser function is assigned.");
        }
        bool parseOk = false;
        T parsed = parser_(envStr, parseOk);
        if (!parseOk) {
            return ReportEnvError(name_, envStr, "is invalid, parse failed.");
        }
        if (validator_ && !validator_(parsed)) {
            return ReportEnvError(name_, envStr, "is out of range.");
        }
        isSetByEnv_ = true;
        value_ = std::move(parsed);
        return HCCL_SUCCESS;
    }

    const T& Get() const { return value_; }
    bool IsSetByEnv() const { return isSetByEnv_; }
    const char* GetSource() const { return isSetByEnv_ ? "environment" : "default"; }

private:
    const char* name_;
    T value_;
    T defaultValue_;
    bool isSetByEnv_{false};
    Parser parser_;
    Validator validator_;
};

/**
 * @brief RDMA 相关环境变量配置。
 *
 */
class EnvRdmaConfig {
public:
    HcclResult GetTaCtpUbTimeOut(uint32_t& value);
    HcclResult GetTaRtpUbTimeOut(uint32_t& value);
    HcclResult GetTaRtpUboeTimeOut(uint32_t& value);
    void ResetParsed();

private:
    HcclResult EnsureParsed();
    // 默认值与范围
    static constexpr uint32_t TA_CTP_UB_TIMEOUT_DEFAULT = 8;    // CTP UB默认TIMEOUT为8(对应4s)
    static constexpr uint32_t TA_RTP_UB_TIMEOUT_DEFAULT = 16;   // RTP UB默认TIMEOUT为16(对应8s)
    static constexpr uint32_t TA_RTP_UBOE_TIMEOUT_DEFAULT = 16; // RTP UBOE默认TIMEOUT为16(对应8s)
    static constexpr uint32_t UB_TIMEOUT_MIN = 0;               // UB/UBOE TIMEOUT最小值为0
    static constexpr uint32_t UB_TIMEOUT_MAX = 31;              // UB/UBOE TIMEOUT最大值为31

    // 环境变量字段（解析器与校验器复用通用模板）
    EnvField<uint32_t> taCtpUbTimeOut_{
        "HCOMM_TA_CTP_UB_TIMEOUT", TA_CTP_UB_TIMEOUT_DEFAULT, StrToNum<uint32_t>,
        MakeRangeValidator(UB_TIMEOUT_MIN, UB_TIMEOUT_MAX)};
    EnvField<uint32_t> taRtpUbTimeOut_{
        "HCOMM_TA_RTP_UB_TIMEOUT", TA_RTP_UB_TIMEOUT_DEFAULT, StrToNum<uint32_t>,
        MakeRangeValidator(UB_TIMEOUT_MIN, UB_TIMEOUT_MAX)};
    EnvField<uint32_t> taRtpUboeTimeOut_{
        "HCOMM_TA_RTP_UBOE_TIMEOUT", TA_RTP_UBOE_TIMEOUT_DEFAULT, StrToNum<uint32_t>,
        MakeRangeValidator(UB_TIMEOUT_MIN, UB_TIMEOUT_MAX)};

    std::atomic<bool> isParsed_{false};
    std::mutex parseMutex_;
};

} // namespace hcomm

#endif // HCOMM_BASE_CONFIG_H
