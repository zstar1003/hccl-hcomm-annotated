/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_CFG_FIELD_H
#define HCCLV2_CFG_FIELD_H

#include <string>
#include <utility>
#include <vector>
#include <functional>
#include "env_func.h"

#include "sal.h"
#include "log.h"
#include "exception_util.h"
#include "invalid_params_exception.h"
#include "adapter_error_manager_pub.h"
#include "hccl_log_keywords.h"

namespace Hccl {

template <typename T>
class CfgField {
public:
    CfgField(
        std::string name, const T& defaultValue, const std::function<T(const std::string&)> cast,
        const std::function<void(const T&)> validate = {}, const std::function<void(T&)> postProc = {})
        : name(std::move(name)),
          value(defaultValue),
          defaultBackup(defaultValue),
          cast(cast),
          validate(validate),
          postProc(postProc),
          isParsed(false),
          isSetByEnvironment_(false) {};

    void Parse()
    {
        std::string str = SalGetEnv(name.c_str());
        if (str.empty() || str == "EmptyString") {
            isParsed = true;
            value = defaultBackup;
            return;
        }
        isSetByEnvironment_ = true;
        // 类型转换
        if (cast) {
            try {
                value = cast(str);
            } catch (const InvalidParamsException& e) {
                // 有异常上报故障码EI0001
                RPT_ENV_ERR(
                    true, "EI0001", std::vector<std::string>({"value", "env", "expect"}),
                    std::vector<std::string>({str, name, e.what()}));
                HCCL_ERROR(
                    "[%s][%s] errNo[0x%016llx] Env config \"%s\" value \"%s\" is invalid. %s",
                    LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_ENV_CONFIG.c_str(),
                    HCOM_ERROR_CODE(HcclResult::HCCL_E_PARA), name.c_str(), str.c_str(), e.what());
                THROW<InvalidParamsException>(
                    StringFormat("[Init][EnvVarParam]Env config \"%s\" value is invalid.%s", name.c_str(), e.what()));
            } catch (const NotSupportException& e) { // 临时修改方案 HCCL_SOCKET_IFNAME等当前不支持配置 且需要报错
                THROW<NotSupportException>(StringFormat(
                    "[Init][EnvVarParam]Env config \"%s\" or its value is currently unsupported.%s", name.c_str(),
                    e.what()));
            }
        } else {
            THROW<InvalidParamsException>(
                StringFormat("[Init][EnvVarParam]Env config \"%s\" No cast function is assigned.", name.c_str()));
        }
        // 校验，环境变量取值范围异常时抛异获取到范围值
        if (validate) {
            try {
                validate(value);
            } catch (const InvalidParamsException& e) {
                // 有异常上报故障码EI0001
                RPT_ENV_ERR(
                    true, "EI0001", std::vector<std::string>({"value", "env", "expect"}),
                    std::vector<std::string>({str, name, e.what()}));
                HCCL_ERROR(
                    "[%s][%s] errNo[0x%016llx] Env config \"%s\" value \"%s\" is invalid. %s",
                    LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_ENV_CONFIG.c_str(),
                    HCOM_ERROR_CODE(HcclResult::HCCL_E_PARA), name.c_str(), str.c_str(), e.what());
                THROW<InvalidParamsException>(
                    StringFormat("[Init][EnvVarParam]Env config \"%s\" value is invalid.%s", name.c_str(), e.what()));
            }
        }
        // 后处理
        if (postProc) {
            postProc(value);
        }
        isParsed = true;
    }

    const std::string& GetEnvName() const { return name; }

    const T& Get() const
    {
        if (UNLIKELY(!isParsed)) {
            THROW<InvalidParamsException>(
                StringFormat("Env config %s is not parsed. Should not use it.", name.c_str()));
        }
        return value;
    }

    bool IsSetByEnvironment() const { return isSetByEnvironment_; }

    const char* GetSource() const { return isSetByEnvironment_ ? "environment" : "default"; }

private:
    std::string name;
    T value;
    T defaultBackup; // 将默认值备份一份，便于后续恢复
    std::function<T(const std::string&)> cast;
    std::function<void(const T&)> validate;
    std::function<void(T&)> postProc;
    bool isParsed;
    bool isSetByEnvironment_;
};

} // namespace Hccl

#endif // HCCLV2_CFG_FIELD_H
