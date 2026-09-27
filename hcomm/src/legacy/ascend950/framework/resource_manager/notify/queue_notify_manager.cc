/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "queue_notify_manager_v2.h"
#include "communicator_impl.h"
namespace Hccl {

QueueNotifyManager::QueueNotifyManager(const CommunicatorImpl& comm) : comm(&comm) {}

QueueNotifyManager::~QueueNotifyManager() { DECTOR_TRY_CATCH("QueueNotifyManager", Destroy()); }

void QueueNotifyManager::ApplyFor(QId postQid, QId waitQid, u32 topicId)
{
    if (topicId > MAX_NUM_FOR_QPAIR) {
        string msg = StringFormat(
            "topicId=%d, Exceed max notify number %d for queue tuple {postQid=%d, waitQid=%d}", topicId,
            MAX_NUM_FOR_QPAIR, postQid, waitQid);
        THROW<InvalidParamsException>(msg);
    }
    const auto& tuple = std::make_tuple(postQid, waitQid, topicId);
    if (notifyPool[tuple] == nullptr) {
        notifyPool[tuple] = std::make_unique<RtsNotify>(comm->GetOpAiCpuTSFeatureFlag()); // 算子粒度
    }
}

bool QueueNotifyManager::Release(QId postQid, QId waitQid, u32 topicId)
{
    if (!IsExist(postQid, waitQid, topicId)) {
        HCCL_WARNING("Notify for postQid[%u] and waitQid[%u] does not exist, no need to release.", postQid, waitQid);
        return true;
    }

    notifyPool.erase(std::make_tuple(postQid, waitQid, topicId));
    return true;
}

bool QueueNotifyManager::Destroy()
{
    notifyPool.clear();
    return true;
}

RtsNotify* QueueNotifyManager::Get(QId postQid, QId waitQid, u32 topicId)
{
    if (!IsExist(postQid, waitQid, topicId)) {
        HCCL_WARNING("Notify for postQid[%u] and waitQid[%u] does not exist", postQid, waitQid);
        return nullptr;
    }

    return notifyPool[std::make_tuple(postQid, waitQid, topicId)].get();
}

bool QueueNotifyManager::IsExist(QId postQid, QId waitQid, u32 topicId)
{
    return notifyPool.count(std::make_tuple(postQid, waitQid, topicId)) != 0;
}

constexpr u8 QUEUE_NOTIFY_POST_QID_POS = 0;
constexpr u8 QUEUE_NOTIFY_WAIT_QID_POS = 1;
constexpr u8 QUEUE_NOTIFY_TOPIC_ID_POS = 2;

std::vector<char> QueueNotifyManager::GetPackedData()
{
    std::vector<char> result;
    BinaryStream binaryStream;

    u32 poolSize = notifyPool.size();
    binaryStream << poolSize;

    for (auto& it : notifyPool) {
        binaryStream << (std::get<QUEUE_NOTIFY_POST_QID_POS>(it.first));
        binaryStream << (std::get<QUEUE_NOTIFY_WAIT_QID_POS>(it.first));
        binaryStream << (std::get<QUEUE_NOTIFY_TOPIC_ID_POS>(it.first));
        binaryStream << it.second->GetUniqueId();
        HCCL_INFO(
            "QueueNotifyManager::GetPackedData: postQid=%u, waitQid=%u, topicId=%u, %s",
            (std::get<QUEUE_NOTIFY_POST_QID_POS>(it.first)), (std::get<QUEUE_NOTIFY_WAIT_QID_POS>(it.first)),
            (std::get<QUEUE_NOTIFY_TOPIC_ID_POS>(it.first)), it.second->Describe().c_str());
    }
    binaryStream.Dump(result);
    return result;
}

} // namespace Hccl
