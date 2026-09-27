/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "queue_wait_group_cnt_notify_manager.h"

namespace Hccl {

QueueWaitGroupCntNotifyManager::QueueWaitGroupCntNotifyManager() {}

QueueWaitGroupCntNotifyManager::~QueueWaitGroupCntNotifyManager()
{
    DECTOR_TRY_CATCH("QueueWaitGroupCntNotifyManager", Destroy());
}

void QueueWaitGroupCntNotifyManager::ApplyFor(QId qid, u32 topicId)
{
    HCCL_INFO("[QueueWaitGroupCntNotifyManager][%s] start, qid[%u] topicId[%u]", __func__, qid, topicId);
    const auto& pair = make_pair(qid, topicId);
    if (notifyPool[pair] == nullptr) {
        notifyPool[pair] = make_unique<RtsCntNotify>();
    }
}

RtsCntNotify* QueueWaitGroupCntNotifyManager::Get(QId qid, u32 topicId)
{
    if (!IsExist(qid, topicId)) {
        HCCL_WARNING("Count Notify for qid[%u] and topic Id[%u] does not exist", qid, topicId);
        return nullptr;
    }

    return notifyPool[make_pair(qid, topicId)].get();
}

bool QueueWaitGroupCntNotifyManager::Release(QId qid, u32 topicId)
{
    if (!IsExist(qid, topicId)) {
        HCCL_WARNING("Count Notify for qid[%u] and topic Id[%u] does not exist.", qid, topicId);
    }
    notifyPool.erase(make_pair(qid, topicId));
    return true;
}

bool QueueWaitGroupCntNotifyManager::IsExist(QId qid, u32 topicId)
{
    return notifyPool.count(make_pair(qid, topicId)) != 0;
}

void QueueWaitGroupCntNotifyManager::Destroy() { notifyPool.clear(); }

std::vector<char> QueueWaitGroupCntNotifyManager::GetPackedData()
{
    std::vector<char> result;
    BinaryStream binaryStream;

    u32 poolSize = notifyPool.size();
    binaryStream << poolSize;

    for (auto& it : notifyPool) {
        binaryStream << it.first.first;
        binaryStream << it.first.second;
        binaryStream << it.second->GetUniqueId();
    }
    binaryStream.Dump(result);
    return result;
}

} // namespace Hccl
