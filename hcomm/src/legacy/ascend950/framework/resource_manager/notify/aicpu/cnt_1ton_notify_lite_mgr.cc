/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cnt_1ton_notify_lite_mgr.h"
#include "binary_stream.h"
#include "log.h"
#include "stl_util.h"

namespace Hccl {

Cnt1tonNotifyLite* Cnt1tonNotifyLiteMgr::Get(u32 postQId, u32 topicId)
{
    auto key = std::make_pair(postQId, topicId);
    if (!Contain(notifys, key)) {
        HCCL_WARNING("Cnt1tonNotifyLiteMgr::%s can not find key[%u][%u]", __func__, postQId, topicId);
        return nullptr;
    }
    HCCL_INFO("Cnt1tonNotifyLiteMgr::%s postQId [%u] topicId [%u]", __func__, postQId, topicId);
    return notifys[key].get();
}

void Cnt1tonNotifyLiteMgr::Reset() { notifys.clear(); }

void Cnt1tonNotifyLiteMgr::ParsePackedData(std::vector<char>& data)
{
    u32 poolSize;
    BinaryStream binaryStream(data);
    binaryStream >> poolSize;

    for (u32 idx = 0; idx < poolSize; idx++) {
        std::pair<u32, u32> key;
        binaryStream >> key.first;
        binaryStream >> key.second;
        std::vector<char> uniqueId;
        binaryStream >> uniqueId;
        if (!Contain(notifys, key)) {
            notifys[key] = std::make_unique<Cnt1tonNotifyLite>(uniqueId);
        }
    }
}

} // namespace Hccl
