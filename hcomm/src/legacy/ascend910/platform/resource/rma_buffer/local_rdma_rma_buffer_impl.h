/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef LOCAL_RDMA_RMA_BUFFER_IMPL_H
#define LOCAL_RDMA_RMA_BUFFER_IMPL_H

#include "hccl_common.h"
#include "hccl_network_pub.h"
#include "rma_buffer.h"
#include "mem_name_repository_pub.h"

namespace hccl {
class LocalRdmaRmaBufferImpl : public RmaBuffer {
public:
    LocalRdmaRmaBufferImpl(const HcclNetDevCtx netDevCtx, void* addr, u64 size, const RmaMemType memType);

    // 别名构造函数：共享父buffer的RDMA注册资源，不调用hrtRaRegGlobalMr
    LocalRdmaRmaBufferImpl(
        const HcclNetDevCtx netDevCtx, void* addr, u64 size, const RmaMemType memType,
        const LocalRdmaRmaBufferImpl& parent);

    ~LocalRdmaRmaBufferImpl() override;

    LocalRdmaRmaBufferImpl(const LocalRdmaRmaBufferImpl& that) = delete;
    LocalRdmaRmaBufferImpl& operator=(const LocalRdmaRmaBufferImpl& that) = delete;

    HcclResult Init();
    HcclResult Destroy();

    std::string& Serialize();
    HcclResult Remap(void* addr, u64 length);

    inline u32 GetKey() const { return lkey; }

private:
    s32 deviceLogicId{-1};
    RdmaHandle rdmaHandle{nullptr};
    MrHandle mrHandle{nullptr};
    u32 lkey{0};
    std::string devAddrID{""};
    bool initialized_ = false;
    std::string serializeStr_;
};
} // namespace hccl
#endif //  LOCAL_RDMA_RMA_BUFFER_IMPL_H
