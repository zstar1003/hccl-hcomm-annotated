/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_JSON_PARSER_H
#define HCCLV2_JSON_PARSER_H

#include <vector>
#include <string>
#include <linux/limits.h>
#include <iostream>
#include <fstream>
#include "nlohmann/json.hpp"
#include "exception_util.h"
#include "internal_exception.h"
#include "invalid_params_exception.h"

namespace Hccl {

std::string GetJsonProperty(const nlohmann::json& obj, const char* propName, bool required = true);
u32 GetJsonPropertyUInt(const nlohmann::json& obj, const char* propName, bool required = true, u32 defaultValue = 0);
s32 GetJsonPropertySInt(const nlohmann::json& obj, const char* propName, bool required = true, s32 defaultValue = 0);
void GetJsonPropertyList(const nlohmann::json& obj, const char* propName, nlohmann::json& listObj);
class JsonParser {
public:
    template <typename T>
    void ParseString(const std::string& jsonString, T& info) const;
    template <typename T>
    void ParseFile(const std::string& filePath, T& info) const;
    template <typename T>
    void GetString(const T& info, std::string& infoStr) const;
    void ParseFileToJson(const std::string& filePath, nlohmann::json& parseInformation) const;

private:
    template <typename T>
    void ParseInformation(nlohmann::json& parseInformation, T& information) const;
};

template <typename T>
void JsonParser::ParseString(const std::string& jsonString, T& info) const
{
    if (jsonString.empty()) {
        THROW<InvalidParamsException>(StringFormat(
            "[Load][JsonString]errNo[0x%016llx] json string length is zero", HCOM_ERROR_CODE(HcclResult::HCCL_E_PARA)));
    }
    HCCL_INFO("waiting for json string load complete");

    nlohmann::json json;
    ParseInformation(json, jsonString);

    try {
        info.Deserialize(json);
    } catch (nlohmann::json::exception& e) {
        THROW<InvalidParamsException>(e.what());
    }
}

template <typename T>
void JsonParser::ParseFile(const std::string& filePath, T& info) const
{
    // 校验文件是否存在
    char resolvedPath[PATH_MAX] = {0};
    if (realpath(filePath.c_str(), resolvedPath) == nullptr) {
        THROW<InvalidParamsException>(StringFormat(
            "[Get][RanktableRealPath]errNo[0x%016llx] path %s is not a valid real path",
            HCOM_ERROR_CODE(HcclResult::HCCL_E_PARA), filePath.c_str()));
    }

    HCCL_INFO("waiting for json file load complete");
    std::ifstream infoFile(resolvedPath, std::ifstream::in);
    if (!infoFile) {
        THROW<InternalException>(StringFormat(
            "[Read][File]errNo[0x%016llx],open file %s failed", HCOM_ERROR_CODE(HcclResult::HCCL_E_OPEN_FILE_FAILURE),
            resolvedPath));
    }

    nlohmann::json json;
    ParseInformation(json, infoFile);
    infoFile.close();

    try {
        info.Deserialize(json);
    } catch (nlohmann::json::exception& e) {
        THROW<InvalidParamsException>(e.what());
    }
}

template <typename T>
void JsonParser::GetString(const T& info, std::string& infoStr) const
{
    nlohmann::json json;
    info.Serialize(json);
    infoStr = json.dump();
}

template <typename T>
void JsonParser::ParseInformation(nlohmann::json& parseInformation, T& information) const
{
    try {
        parseInformation = nlohmann::json::parse(information);
    } catch (const nlohmann::json::parse_error& e) {
        HCCL_ERROR("JSON parse error: %s at byte %d", e.what(), e.byte);
        THROW<InvalidParamsException>(StringFormat(
            "[Parse][Information] errNo[0x%016llx] load allocated resource to "
            "json fail. JSON parse error: %s at byte %d"
            "please check json input!",
            HCOM_ERROR_CODE(HcclResult::HCCL_E_PARA), e.what(), e.byte));
    } catch (const nlohmann::json::exception& e) {
        HCCL_ERROR("JSON parse error: %s", e.what());
        THROW<InvalidParamsException>(StringFormat(
            "[Parse][Information] errNo[0x%016llx] load allocated resource to json fail. JSON parse error: %s"
            "please check json input!",
            HCOM_ERROR_CODE(HcclResult::HCCL_E_PARA), e.what()));
    } catch (...) {
        THROW<InvalidParamsException>(StringFormat(
            "[Parse][Information] errNo[0x%016llx] load allocated resource to json fail. "
            "please check json input!",
            HCOM_ERROR_CODE(HcclResult::HCCL_E_PARA)));
    };
}

} // namespace Hccl
#endif
