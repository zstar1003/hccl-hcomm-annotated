/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ub_conn_lite_mgr.h"

namespace Hccl {

UbConnLiteMgr::UbConnLiteMgr() {}

UbConnLiteMgr::~UbConnLiteMgr() { ubConnLiteMap.clear(); }

std::string UbConnLiteMgr::GetKey(const UbConnLiteParam& liteParam) const
{
    // dieId + funcId + jettyId + tp + eid 可唯一确定一个connection
    std::string result;
    result += to_string(liteParam.dieId) + to_string(liteParam.funcId) + to_string(liteParam.jettyId)
              + to_string(liteParam.tpn);
    result += Bytes2hex(liteParam.rmtEid.raw, sizeof(liteParam.rmtEid.raw));
    return result;
}

bool UbConnLiteMgr::IsExist(const std::string& key)
{
    if (ubConnLiteMap.find(key) == ubConnLiteMap.end()) {
        return false;
    }
    return true;
}

UbConnLiteMgr& UbConnLiteMgr::GetInstance()
{
    static UbConnLiteMgr ubConnLiteMgr;
    return ubConnLiteMgr;
}

RmaConnLite* UbConnLiteMgr::Get(std::vector<char>& uniqueId)
{
    UbConnLiteParam liteParam(uniqueId);
    auto key = GetKey(liteParam);
    std::unique_lock<std::shared_mutex> lock(mtx_);
    if (IsExist(key)) {
        return ubConnLiteMap[key].get();
    }

    ubConnLiteMap[key] = make_unique<UbConnLite>(liteParam);
    return ubConnLiteMap[key].get();
}

void UbConnLiteMgr::Clear(std::vector<char>& uniqueId)
{
    UbConnLiteParam liteParam(uniqueId);
    auto key = GetKey(liteParam);
    std::unique_lock<std::shared_mutex> lock(mtx_);
    if (!IsExist(key)) {
        return;
    }

    ubConnLiteMap.erase(key);
}

} // namespace Hccl
