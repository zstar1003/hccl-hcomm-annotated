/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INTERFACE_CHANNEL_H
#define INTERFACE_CHANNEL_H
#include "hccl/hccl_res.h"
#include "hccl_types.h"
#include "hccl_mem_defs.h"

namespace Hccl {
class IChannel {
public:
    IChannel();
    virtual ~IChannel();

    // 数据面调用verbs接口
    virtual HcclResult NotifyRecord(uint32_t remoteNotifyIdx) = 0;
    virtual HcclResult NotifyWait(uint32_t localNotifyIdx, uint32_t timeout) = 0;
    virtual HcclResult WriteWithNotify(void* dst, const void* src, uint64_t len, uint32_t remoteNotifyIdx) = 0;
    virtual HcclResult ChannelFence() = 0;
    virtual HcclResult GetNotifyNum(uint32_t* notifyNum) = 0;
    virtual HcclResult GetHcclBuffer(void*& addr, uint64_t& size) = 0;
    virtual HcclResult Init() = 0;
    virtual HcclResult DeInit() = 0;
    virtual HcclResult GetDpuRemoteMem(HcclMem** remoteMem, uint32_t* memNum) = 0;
    virtual uint32_t GetStatus() = 0;
};

} // namespace Hccl
#endif // INTERFACE_CHANNEL_H
