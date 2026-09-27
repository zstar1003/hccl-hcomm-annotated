/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef LOCAL_RDMA_RMA_BUFFER_H
#define LOCAL_RDMA_RMA_BUFFER_H

#include <memory>
#include <string>
#include "hccl_common.h"
#include "hccl_network_pub.h"
#include "rma_buffer.h"

namespace hccl {
class LocalRdmaRmaBufferImpl;

class LocalRdmaRmaBuffer : public RmaBuffer {
public:
    LocalRdmaRmaBuffer(
        const HcclNetDevCtx netDevCtx, void* addr, u64 size, const RmaMemType memType = RmaMemType::DEVICE);

    // 别名构造函数：共享父buffer的RDMA注册资源
    LocalRdmaRmaBuffer(
        const HcclNetDevCtx netDevCtx, void* addr, u64 size, const RmaMemType memType,
        const LocalRdmaRmaBuffer& parent);

    HcclResult Init();
    HcclResult Destroy();
    ~LocalRdmaRmaBuffer() override;

    LocalRdmaRmaBuffer(const LocalRdmaRmaBuffer& that) = delete;
    LocalRdmaRmaBuffer& operator=(const LocalRdmaRmaBuffer& that) = delete;

    // Init成功后调用Serialize和获取属性接口
    std::string& Serialize();

    u32 GetKey() const;
    HcclResult Remap(void* addr, u64 length);

private:
    const LocalRdmaRmaBufferImpl& GetImpl() const { return *pimpl_; }
    std::unique_ptr<LocalRdmaRmaBufferImpl> pimpl_;
};
} // namespace hccl
#endif //  LOCAL_RDMA_RMA_BUFFER_H
