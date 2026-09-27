/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef LOCAL_IPC_RMA_BUFFER_H
#define LOCAL_IPC_RMA_BUFFER_H

#include <memory>
#include <string>
#include "hccl_common.h"
#include "hccl_network_pub.h"
#include "rma_buffer.h"

namespace hccl {
class LocalIpcRmaBufferImpl;

class LocalIpcRmaBuffer : public RmaBuffer {
public:
    LocalIpcRmaBuffer(
        const HcclNetDevCtx netDevCtx, void* addr, u64 size, const RmaMemType memType = RmaMemType::DEVICE);

    // 别名构造函数：共享父buffer的IPC资源
    LocalIpcRmaBuffer(
        const HcclNetDevCtx netDevCtx, void* addr, u64 size, const RmaMemType memType, const LocalIpcRmaBuffer& parent);

    HcclResult Init();
    HcclResult Destroy();
    ~LocalIpcRmaBuffer() override;

    LocalIpcRmaBuffer(const LocalIpcRmaBuffer& that) = delete;
    LocalIpcRmaBuffer& operator=(const LocalIpcRmaBuffer& that) = delete;

    // Init成功后调用Serialize和获取属性接口
    std::string& Serialize();

    HcclResult Grant(u32 remotePid, u32 remoteSdid);

private:
    std::unique_ptr<LocalIpcRmaBufferImpl> pimpl_;
};
} // namespace hccl
#endif //  LOCAL_IPC_RMA_BUFFER_H
