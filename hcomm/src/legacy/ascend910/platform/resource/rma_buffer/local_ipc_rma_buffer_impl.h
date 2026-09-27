/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef LOCAL_IPC_RMA_BUFFER_IMPL_H
#define LOCAL_IPC_RMA_BUFFER_IMPL_H

#include "hccl_common.h"
#include "hccl_network_pub.h"
#include "rma_buffer.h"
#include "mem_name_repository_pub.h"

namespace hccl {
class LocalIpcRmaBufferImpl : public RmaBuffer {
public:
    LocalIpcRmaBufferImpl(const HcclNetDevCtx netDevCtx, void* addr, u64 size, const RmaMemType memType);

    // 别名构造函数：共享父buffer的IPC资源，不创建新的IPC内存名
    LocalIpcRmaBufferImpl(
        const HcclNetDevCtx netDevCtx, void* addr, u64 size, const RmaMemType memType,
        const LocalIpcRmaBufferImpl& parent);

    ~LocalIpcRmaBufferImpl() override;

    LocalIpcRmaBufferImpl(const LocalIpcRmaBufferImpl& that) = delete;
    LocalIpcRmaBufferImpl& operator=(const LocalIpcRmaBufferImpl& that) = delete;

    HcclResult Init();
    HcclResult Destroy();

    std::string& Serialize();

    HcclResult Grant(u32 remotePid, u32 remoteSdid);

private:
    s32 deviceLogicId{-1};
    SecIpcName_t memName;
    u64 memOffset{0};
    bool initialized_ = false;
    std::string serializeStr_;
};
} // namespace hccl
#endif //  LOCAL_IPC_RMA_BUFFER_IMPL_H
