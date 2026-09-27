/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_comm_config_consistency.h"

namespace hccl {
CollCommConfigConsistency::CollCommConfigConsistency() {}

CollCommConfigConsistency::~CollCommConfigConsistency()
{
    HCCL_INFO("[CollCommConfigConsistency][~CollCommConfigConsistency] CollCommConfigConsistency deinit");
    remoteExchangeInfoMap_.clear();
    ResetExchangeInfo();
}

HcclResult CollCommConfigConsistency::AddExchangeInfo(const void* data, uint32_t length)
{
    CHK_PTR_NULL(data);
    if (length > 0 && length <= HCCL_EXCHANGE_INFO_LEN) {
        exchangeInfoBuf_.resize(length);
        s32 sRet = memcpy_s(exchangeInfoBuf_.data(), length, data, length);
        CHK_PRT_RET(sRet != EOK, HCCL_ERROR("[AddExchangeInfo] memcpy_s failed, ret[%d]", sRet), HCCL_E_MEMORY);
        HCCL_INFO("[AddExchangeInfo] success, length[%u].", length);
    } else {
        HCCL_ERROR("[AddExchangeInfo] length[%u] is illegal", length);
        return HCCL_E_PARA;
    }

    return HCCL_SUCCESS;
}

HcclResult
CollCommConfigConsistency::GetExchangeInfo(uint32_t remoteRank, uint32_t length, void* data, uint32_t* actualLength)
{
    CHK_PTR_NULL(data);
    auto iter = remoteExchangeInfoMap_.find(remoteRank);
    if (iter == remoteExchangeInfoMap_.end()) {
        *actualLength = 0;
        HCCL_INFO("[GetExchangeInfo] remoteExchangeInfoMap_ is null");
        return HCCL_SUCCESS;
    }
    *actualLength = static_cast<uint32_t>(iter->second.size());

    if (length == *actualLength || *actualLength == 0) {
        s32 sRet = memcpy_s(data, length, iter->second.data(), iter->second.size());
        CHK_PRT_RET(sRet != EOK, HCCL_ERROR("[GetExchangeInfo] memcpy_s failed, ret[%d]", sRet), HCCL_E_MEMORY);
    } else {
        HCCL_ERROR("[GetExchangeInfo] failed, length[%u] actualLength[%u].", length, *actualLength);
        return HCCL_E_PARA;
    }

    // 读后清除
    remoteExchangeInfoMap_.erase(iter);
    HCCL_INFO("[GetExchangeInfo] success, remoteRank[%u], actualLength[%u].", remoteRank, *actualLength);
    return HCCL_SUCCESS;
}

HcclResult CollCommConfigConsistency::StoreRemoteExchangeInfo(uint32_t remoteRank, std::vector<u8>& data)
{
    remoteExchangeInfoMap_[remoteRank] = std::move(data);
    HCCL_INFO(
        "[StoreRemoteExchangeInfo] success, remoteRank[%u], length[%zu].", remoteRank,
        remoteExchangeInfoMap_[remoteRank].size());
    return HCCL_SUCCESS;
}

HcclResult CollCommConfigConsistency::ResetExchangeInfo()
{
    exchangeInfoBuf_.clear();
    HCCL_INFO("[ResetExchangeInfo] exchange info state cleared.");
    return HCCL_SUCCESS;
}

HcclResult CollCommConfigConsistency::GetExchangeInfoBuf(std::vector<u8>& exchangeInfoBuf)
{
    exchangeInfoBuf = exchangeInfoBuf_;
    return HCCL_SUCCESS;
}

uint32_t CollCommConfigConsistency::GetExchangeInfoLen() const
{
    return static_cast<uint32_t>(exchangeInfoBuf_.size());
}

} // namespace hccl
