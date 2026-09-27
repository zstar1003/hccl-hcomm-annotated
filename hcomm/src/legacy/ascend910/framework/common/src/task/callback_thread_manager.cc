/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "callback_thread_manager.h"
#include "log.h"

namespace hccl {

bool ThreadStreamManager::StreamHasBeenReged(void* stream) { return streamTidMap_.find(stream) != streamTidMap_.end(); }

HcclResult ThreadStreamManager::RegTidAndStream(u64 tid, rtStream_t stream)
{
    CHK_PTR_NULL(stream);

    std::unique_lock<std::mutex> lock(mapMutex_);
    streamTidMap_[stream] = tid;

    return HCCL_SUCCESS;
}

HcclResult ThreadStreamManager::GetStreamByTid(u64 tid, rtStream_t& stream)
{
    std::unique_lock<std::mutex> lock(mapMutex_);
    std::map<rtStream_t, u64>::iterator it;
    for (it = streamTidMap_.begin(); it != streamTidMap_.end(); it++) {
        if (it->second == tid) {
            stream = it->first;
            return HCCL_SUCCESS;
        }
    }
    return HCCL_E_NOT_FOUND;
}

void ThreadStreamManager::ReleaseTidAndStream(rtStream_t stream)
{
    std::unique_lock<std::mutex> lock(mapMutex_);
    if (StreamHasBeenReged(stream)) {
        streamTidMap_.erase(stream);
    }
}

} // namespace hccl
