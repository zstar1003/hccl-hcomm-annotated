/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RMA_BUFFER_H
#define RMA_BUFFER_H

#include <memory>

#include "hccl_common.h"
#include "hccl_inner_common.h"
#include "transport_mem.h"

namespace hccl {
class RmaBuffer {
public:
    RmaBuffer(const HcclNetDevCtx netDevCtx, void* addr, u64 size, const RmaMemType memType, const RmaType rmaType)
        : netDevCtx(netDevCtx),
          addr(addr),
          size(size),
          memType(memType),
          rmaType(rmaType)
    {}

    RmaBuffer(
        const HcclNetDevCtx netDevCtx, void* addr, u64 size, const RmaMemType memType, const RmaType rmaType,
        bool isAlias)
        : netDevCtx(netDevCtx),
          addr(addr),
          size(size),
          memType(memType),
          rmaType(rmaType),
          isAlias_(isAlias)
    {}

    virtual ~RmaBuffer() = default;

    RmaBuffer(const RmaBuffer& that) = delete;

    RmaBuffer& operator=(const RmaBuffer& that) = delete;

    inline bool IsAlias() const { return isAlias_; }

    inline void* GetAddr() const { return addr; }

    inline u64 GetSize() const { return size; }

    inline RmaType GetRmaType() const // used for grant check
    {
        return rmaType;
    }

    inline RmaMemType GetMemType() const { return memType; }

    inline void* GetDevAddr() const { return devAddr; }

    inline HcclNetDevCtx GetNetDevCtx() const { return netDevCtx; }

protected:
    const HcclNetDevCtx netDevCtx{nullptr};
    void* addr{nullptr};
    u64 size{0};
    void* devAddr{nullptr};
    RmaMemType memType{RmaMemType::TYPE_NUM};
    RmaType rmaType{RmaType::RMA_TYPE_RESERVED};
    bool isAlias_{false};
};

struct RmaBufferSlice {
    std::shared_ptr<RmaBuffer> rmaBuffer{nullptr};
    void* addr{nullptr};
    u64 len{0};
    RmaMemType memType{RmaMemType::DEVICE};
};

inline HcclResult CheckHcclBuffer(const void* addr, const RmaBuffer* rmaBuffer)
{
    CHK_PTR_NULL(addr);
    CHK_PTR_NULL(rmaBuffer);
    if (UNLIKELY(
            reinterpret_cast<u64>(addr) < reinterpret_cast<u64>(rmaBuffer->GetAddr())
            || reinterpret_cast<u64>(addr) > (reinterpret_cast<u64>(rmaBuffer->GetAddr()) + rmaBuffer->GetSize()))) {
        HCCL_ERROR(
            "[CheckHcclBuffer]check buffer failed, hccl buffer addr[%p], "
            "ramBuffer addr[%p], rmaBuffer size[%u]",
            addr, rmaBuffer->GetAddr(), rmaBuffer->GetSize());
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}
} // namespace hccl
#endif //  RDMA_RMA_BUFFER_H
