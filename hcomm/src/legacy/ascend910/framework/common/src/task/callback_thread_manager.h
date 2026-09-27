/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CALLBACK_THREAD_MANAGER_H
#define CALLBACK_THREAD_MANAGER_H

#include <map>
#include <mutex>
#include "hccl/base.h"

namespace hccl {

class ThreadStreamManager {
public:
    static ThreadStreamManager& Instance()
    {
        static ThreadStreamManager instance;
        return instance;
    }

    bool StreamHasBeenReged(rtStream_t stream);
    HcclResult RegTidAndStream(u64 tid, rtStream_t stream);
    HcclResult GetStreamByTid(u64 tid, rtStream_t& stream);
    void ReleaseTidAndStream(rtStream_t stream);

private:
    ThreadStreamManager() = default;
    ~ThreadStreamManager() = default;
    std::map<rtStream_t, u64> streamTidMap_; // stream被注册的map
    std::mutex mapMutex_;                    // 锁stream被注册的map
};
} // namespace hccl
#endif // CALLBACK_THREAD_MANAGER_H
