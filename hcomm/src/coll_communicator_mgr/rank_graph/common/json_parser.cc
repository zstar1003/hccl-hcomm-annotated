/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <climits>
#include "json_parser.h"
#include "adapter_error_manager_pub.h"
#include "hccl_log_keywords.h"

namespace Hccl {

std::string GetJsonProperty(const nlohmann::json& obj, const char* propName, bool required)
{
    if (propName == nullptr) {
        std::string msg = StringFormat("[%s] propName is nullptr", __func__);
        THROW<NullPtrException>(msg);
    }
    if (!required) {
        return obj.value<std::string>(propName, "");
    }

    if (!obj.contains(propName)) {
        THROW<InvalidParamsException>(
            StringFormat("[Get][JsonProperty] json object does not contain property[%s].", propName));
    }
    std::string value = obj.at(propName).get<std::string>();
    return value;
}

u32 GetJsonPropertyUInt(const nlohmann::json& obj, const char* propName, bool required, u32 defaultValue)
{
    if (!required) {
        return obj.value<u32>(propName, defaultValue);
    }

    if (!obj.contains(propName)) {
        THROW<InvalidParamsException>(
            StringFormat("[Get][JsonPropertyUInt] json object does not contain property[%s].", propName));
    }

    s64 value = obj.at(propName).get<s64>();
    if (value < 0 || value > UINT32_MAX) {
        THROW<InvalidParamsException>(StringFormat(
            "[Get][JsonPropertyUInt]errNo[0x%016llx]:json object "
            "property value of Name[%s] should be an unsigned 32-bit integer but actually not!",
            HCOM_ERROR_CODE(HcclResult::HCCL_E_PARA), propName));
    }
    return static_cast<u32>(value);
}

s32 GetJsonPropertySInt(const nlohmann::json& obj, const char* propName, bool required, s32 defaultValue)
{
    if (!required) {
        return obj.value<s32>(propName, defaultValue);
    }

    if (!obj.contains(propName)) {
        THROW<InvalidParamsException>(
            StringFormat("[Get][JsonPropertySInt] json object does not contain property[%s].", propName));
    }

    s64 value = obj.at(propName).get<s64>();
    if (value < INT_MIN || value > INT_MAX) {
        THROW<InvalidParamsException>(StringFormat(
            "[Get][JsonPropertySInt]errNo[0x%016llx]:json object "
            "property value of Name[%s] is not signed number!",
            HCOM_ERROR_CODE(HcclResult::HCCL_E_PARA), propName));
    }
    HCCL_INFO(
        "[Get][JsonPropertySInt]json object property value of Name [%s] is [%d]", propName, static_cast<s32>(value));
    return static_cast<s32>(value);
}

void GetJsonPropertyList(const nlohmann::json& obj, const char* propName, nlohmann::json& listObj)
{
    if (!obj.contains(propName)) {
        THROW<InvalidParamsException>(
            StringFormat("[Get][JsonPropertyList] json object does not contain property[%s].", propName));
    }

    listObj = obj.at(propName);
    if (!listObj.is_array()) {
        THROW<InvalidParamsException>(StringFormat(
            "[Get][GetJsonPropertyList]errNo[0x%016llx]:json object "
            "property value of Name \"%s\" is not list!",
            HCOM_ERROR_CODE(HcclResult::HCCL_E_PARA), propName));
    }
}

void JsonParser::ParseFileToJson(const std::string& filePath, nlohmann::json& parseInformation) const
{
    // 校验文件是否存在
    char resolvedPath[PATH_MAX] = {0};
    if (realpath(filePath.c_str(), resolvedPath) == nullptr) {
        RPT_INPUT_ERR(
            true, "EI0004", std::vector<std::string>({"error_reason", "ranktable_path"}),
            std::vector<std::string>(
                {filePath, "The rankTable file path does not exist, the permission is insufficient, or the JSON format "
                           "is incorrect."}));
        HCCL_ERROR(
            "[%s][%s] errNo[0x%016llx] path %s is not a valid real path", LOG_KEYWORDS_INIT_GROUP.c_str(),
            LOG_KEYWORDS_RANKTABLE_CONFIG.c_str(), HCOM_ERROR_CODE(HcclResult::HCCL_E_PARA), filePath.c_str());
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

    ParseInformation(parseInformation, infoFile);
    infoFile.close();
}

} // namespace Hccl
