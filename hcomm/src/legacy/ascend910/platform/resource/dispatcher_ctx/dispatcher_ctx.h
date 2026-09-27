/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CTX_H
#define CTX_H

#include "hccl_common.h"
#include "hccl/base.h"
#include "notify_pool.h"

namespace hccl {
using FftsCounterCallBack = HcclResult (*)(const HcclDispatcher&, Stream&);
HcclResult FftsHeadCounter(const HcclDispatcher& dispatcher, Stream& stream);
HcclResult FftsTailCounter(const HcclDispatcher& dispatcher, Stream& stream);

enum class CtxDispatcherType {
    DISPATCHER_NORMAL = 0,
    DISPATCHER_VIRTURAL,
    DISPATCHER_AICPU,
    DISPATCHER_FFTS,
};

class DispatcherCtx {
public:
    explicit DispatcherCtx(u32 devicePhyId, u32 timeOut = INVALID_UINT)
        : devicePhyId_(devicePhyId),
          waitTimeOut_(timeOut) {};
    ~DispatcherCtx() { Destroy(); };
    HcclResult Init();
    HcclResult Destroy();
    HcclDispatcher GetDispatcher() const;
    u32 GetWaitTimeOut() const;
    HcclResult SetWaitTimeOut(u32 waitTimeOut);
    FftsCounterCallBack GetInitTaskCallback() const;
    FftsCounterCallBack GetLaunchTaskCallback() const;
    HcclResult SetDispatcherHcclQos(u32 hcclQos);

private:
    CtxDispatcherType dispatcherType_{CtxDispatcherType::DISPATCHER_NORMAL};
    HcclDispatcher dispatcher_{nullptr};

    u32 devicePhyId_ = INVALID_UINT;
    u32 deviceLogicId_ = INVALID_UINT;
    DevType deviceType_ = DevType::DEV_TYPE_COUNT;
    u32 waitTimeOut_ = INVALID_UINT;
    HcclResult DispatcherInit(CtxDispatcherType type, const s32 devicePhyId, HcclDispatcher* dispatcher);

    FftsCounterCallBack g_InitTaskCallback = nullptr;
    FftsCounterCallBack g_LaunchTaskCallback = nullptr;

    std::mutex destroyMutex_;
};
} // namespace hccl

#endif
