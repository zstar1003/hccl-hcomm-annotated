/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "queue_notify_lite_mgr.h"
#include "binary_stream.h"
#include "log.h"
#include "stl_util.h"

namespace Hccl {

NotifyLite* QueueNotifyLiteMgr::Get(u32 postQId, u32 waitQId, u32 topicId)
{
    auto key = std::make_tuple(postQId, waitQId, topicId);
    if (!Contain(notifys, key)) {
        return nullptr;
    }
    HCCL_INFO("QueueNotifyLiteMgr::Get postQId [%u] waitQId [%u] topicId [%u]", postQId, waitQId, topicId);
    return notifys[key].get();
}

void QueueNotifyLiteMgr::Reset() { notifys.clear(); }

constexpr u8 QUEUE_NOTIFY_POST_QID_POS = 0;
constexpr u8 QUEUE_NOTIFY_WAIT_QID_POS = 1;
constexpr u8 QUEUE_NOTIFY_TOPIC_ID_POS = 2;

void QueueNotifyLiteMgr::ParsePackedData(std::vector<char>& data)
{
    u32 poolSize;
    BinaryStream binaryStream(data);
    binaryStream >> poolSize;

    for (u32 idx = 0; idx < poolSize; idx++) {
        std::tuple<u32, u32, u32> key;
        binaryStream >> std::get<QUEUE_NOTIFY_POST_QID_POS>(key);
        binaryStream >> std::get<QUEUE_NOTIFY_WAIT_QID_POS>(key);
        binaryStream >> std::get<QUEUE_NOTIFY_TOPIC_ID_POS>(key);
        std::vector<char> uniqueId;
        binaryStream >> uniqueId;
        if (!Contain(notifys, key)) {
            notifys[key] = std::make_unique<NotifyLite>(uniqueId);
        }
    }
}

} // namespace Hccl
