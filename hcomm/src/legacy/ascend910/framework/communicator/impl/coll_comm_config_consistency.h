/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COLL_COMM_CONFIG_CONSISTENCY_H
#define COLL_COMM_CONFIG_CONSISTENCY_H

#include <vector>
#include <memory>
#include <unordered_map>
#include "common.h"

namespace hccl {
class CollCommConfigConsistency {
public:
    CollCommConfigConsistency();
    ~CollCommConfigConsistency();

    // 交换信息管理接口
    HcclResult AddExchangeInfo(const void* data, uint32_t length);
    HcclResult GetExchangeInfo(uint32_t remoteRank, uint32_t length, void* data, uint32_t* actualLength);
    HcclResult StoreRemoteExchangeInfo(uint32_t remoteRank, std::vector<u8>& data);
    HcclResult ResetExchangeInfo();
    HcclResult GetExchangeInfoBuf(std::vector<u8>& exchangeInfoBuf);
    uint32_t GetExchangeInfoLen() const;

private:
    std::vector<u8> exchangeInfoBuf_;                                     // 本端待交换信息缓冲区
    std::unordered_map<uint32_t, std::vector<u8>> remoteExchangeInfoMap_; // 对端交换信息<remoteRank, data>
};
} // namespace hccl

#endif // COLL_COMM_CONFIG_CONSISTENCY_H
