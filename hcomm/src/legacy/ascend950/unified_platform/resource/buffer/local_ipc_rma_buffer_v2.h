/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_LOCAL_IPC_RMA_BUFFER_H
#define HCCLV2_LOCAL_IPC_RMA_BUFFER_H

#include "local_rma_buffer.h"
#include "orion_adapter_rts.h"

namespace Hccl {

class LocalIpcRmaBuffer : public LocalRmaBuffer {
public:
    explicit LocalIpcRmaBuffer(std::shared_ptr<Buffer> buf);

    // 别名构造函数：共享父buffer的IPC资源，不创建新的IPC内存名
    LocalIpcRmaBuffer(std::shared_ptr<Buffer> buf, const LocalIpcRmaBuffer& parent);

    ~LocalIpcRmaBuffer() override;

    LocalIpcRmaBuffer(const LocalIpcRmaBuffer& that) = delete;

    LocalIpcRmaBuffer& operator=(const LocalIpcRmaBuffer& that) = delete;

    std::string Describe() const override;

    void Grant(u32 pid);

    std::unique_ptr<Serializable> GetExchangeDto() override;
    std::pair<uintptr_t, u64> GetBufferInfo() { return std::make_pair(buf->GetAddr(), buf->GetSize()); }

    void* GetIpcPtr() const { return ipcPtr; }

protected:
    char name[RTS_IPC_MEM_NAME_LEN];
    void* ipcPtr{nullptr};
    u64 ipcOffset{0};
    u64 ipcSize{0};
};

} // namespace Hccl
#endif // HCCLV2_LOCAL_IPC_RMA_BUFFER_H
