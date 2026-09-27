/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <fstream>
#include "whitelist.h"
#include "internal_exception.h"
#include "invalid_params_exception.h"

namespace Hccl {
Whitelist& Whitelist::GetInstance()
{
    static Whitelist whitelist;
    return whitelist;
}

Whitelist::~Whitelist()
{
    std::unique_lock<std::mutex> lock(whiteListsMutex);
    whiteLists.clear();
}

void Whitelist::GetHostWhiteList(std::vector<IpAddress>& whiteList)
{
    std::unique_lock<std::mutex> lock(whiteListsMutex);
    whiteList.clear();
    auto iter = whiteLists.find(WhiteListType::HCCL_WHITELIST_HOST);
    if (iter == whiteLists.end()) {
        HCCL_INFO("GetHostWhiteList: white list is empty.");
        return;
    }
    whiteList = whiteLists[WhiteListType::HCCL_WHITELIST_HOST];
    HCCL_INFO("GetHostWhiteList: whitelist length is %zu.", whiteList.size());
}

void Whitelist::LoadConfigFile(const std::string& realName)
{
    if (realName.empty()) {
        HCCL_ERROR("Load ConfigFile whitelist file path is NULL.");
        THROW<InvalidParamsException>(StringFormat(
            "[Load][ConfigFile]errNo[0x%016llx] whitelist file path is NULL.",
            HCOM_ERROR_CODE(HcclResult::HCCL_E_PARA)));
    }
    std::unique_lock<std::mutex> lock(whiteListsMutex);
    whiteLists.clear();

    nlohmann::json fileContent;
    std::ifstream infile(realName.c_str(), std::ifstream::in);
    if (!infile) {
        HCCL_ERROR(
            "[Load][ConfigFile]errNo[0x%016llx] open file %s failed", HCOM_ERROR_CODE(HcclResult::HCCL_E_INTERNAL),
            realName.c_str());
        THROW<InternalException>(StringFormat(
            "[Load][ConfigFile]errNo[0x%016llx] open file %s failed", HCOM_ERROR_CODE(HcclResult::HCCL_E_INTERNAL),
            realName.c_str()));
        return;
    } else {
        fileContent.clear();
        try {
            infile >> fileContent; // 将文件内容读取到json对象内
        } catch (...) {
            HCCL_ERROR(
                "[Load][ConfigFile]errNo[0x%016llx] load file[%s] to json fail. please check json file format.",
                HCOM_ERROR_CODE(HcclResult::HCCL_E_INTERNAL), realName.c_str());
            infile.close();
            THROW<InternalException>(StringFormat(
                "[Load][ConfigFile]errNo[0x%016llx] load file[%s] to json fail. please check json file format.",
                HCOM_ERROR_CODE(HcclResult::HCCL_E_INTERNAL), realName.c_str()));
        }
    }
    infile.close();

    nlohmann::json hostWhitelist = GetHostIp(fileContent);

    for (auto& ipJson : hostWhitelist) {
        std::string ipStr;
        try {
            ipStr = ipJson.get<std::string>();
        } catch (...) {
            HCCL_ERROR(
                "[Load][ConfigFile]errNo[0x%016llx]get ipStr from ipJson failed, please check host white list",
                HCOM_ERROR_CODE(HcclResult::HCCL_E_INTERNAL));
            THROW<InternalException>(StringFormat(
                "[Load][ConfigFile]errNo[0x%016llx]get ipStr from ipJson failed, please check host white list",
                HCOM_ERROR_CODE(HcclResult::HCCL_E_INTERNAL)));
        }
        IpAddress ip(ipStr);
        whiteLists[WhiteListType::HCCL_WHITELIST_HOST].push_back(ip);
    }
}

nlohmann::json Whitelist::GetHostIp(nlohmann::json fileContent) const
{
    nlohmann::json hostWhitelist;
    if (fileContent.find("host_ip") == fileContent.end()) {
        HCCL_ERROR(
            "[Get][JsonProperty]errNo[0x%016llx] json object has no property called host_ip",
            HCOM_ERROR_CODE(HcclResult::HCCL_E_INTERNAL));
        THROW<InternalException>(StringFormat(
            "[Get][JsonProperty]errNo[0x%016llx] json object has no property called host_ip",
            HCOM_ERROR_CODE(HcclResult::HCCL_E_INTERNAL)));
    }
    hostWhitelist = fileContent["host_ip"];
    return hostWhitelist;
}

} // namespace Hccl
