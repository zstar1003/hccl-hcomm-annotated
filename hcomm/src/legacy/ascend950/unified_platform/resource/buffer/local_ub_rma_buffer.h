/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_LOCAL_UB_RMA_BUFFER_H
#define HCCLV2_LOCAL_UB_RMA_BUFFER_H

#include <cstring>

#include "local_rma_buffer.h"

#include "enum_factory.h"
#include "orion_adapter_hccp.h"
#include "orion_adapter_rts.h"
#include "net_device.h"

namespace Hccl {

MAKE_ENUM(UbBufferStatus, INIT, READY, RELEASED);

class LocalUbRmaBuffer : public LocalRmaBuffer {
public:
    LocalUbRmaBuffer(std::shared_ptr<Buffer> buf, RdmaHandle rdmaHandle);

    // 别名构造函数：共享父buffer的注册资源
    LocalUbRmaBuffer(std::shared_ptr<Buffer> buf, RdmaHandle rdmaHandle, const LocalUbRmaBuffer& parent);

    LocalUbRmaBuffer(std::shared_ptr<Buffer> buf, void* netDevice, bool flag);

    LocalUbRmaBuffer(std::shared_ptr<Buffer> buf);

    ~LocalUbRmaBuffer() override;

    LocalUbRmaBuffer(const LocalUbRmaBuffer& that) = delete;

    LocalUbRmaBuffer& operator=(const LocalUbRmaBuffer& that) = delete;

    string Describe() const override;

    std::unique_ptr<Serializable> GetExchangeDto() override;

    u32 GetTokenId() const;
    u32 GetTokenValue() const;
    TokenIdHandle GetTokenIdHandle() const;
    std::pair<uintptr_t, u64> GetBufferInfo() { return make_pair(buf->GetAddr(), buf->GetSize()); }
    u64 GetTargetSeg() const { return reqReg.targetSegVa; }

    void* GetMemRegOutParam() { return static_cast<void*>(&reqReg); }

    const void* GetMemRegOutParam() const { return static_cast<const void*>(&reqReg); }

    static bool IsSameMemRegOutParam(const void* lhs, const void* rhs)
    {
        if (lhs == nullptr || rhs == nullptr) {
            return false;
        }
        const auto* left = static_cast<const HrtRaUbLocalMemRegOutParam*>(lhs);
        const auto* right = static_cast<const HrtRaUbLocalMemRegOutParam*>(rhs);
        if (left->keySize > HRT_UB_MEM_KEY_MAX_LEN) {
            return false;
        }
        return left->handle == right->handle && left->keySize == right->keySize
               && left->targetSegVa == right->targetSegVa && memcmp(left->key, right->key, left->keySize) == 0;
    }

    std::vector<char> Desc;

private:
    RdmaHandle rdmaHandle{nullptr};
    HcclNetDevice* netDev{nullptr};
    u8 key[HRT_UB_MEM_KEY_MAX_LEN]{0};
    u32 tokenValue{0};
    u32 tokenId{0};
    TokenIdHandle tokenIdHandle{0};

    HrtRaUbLocalMemRegOutParam reqReg{};

    BufferKey<uintptr_t, u64> bufKey_{0, 0};
};
u32 GetUbToken(); // 生成伪随机数
} // namespace Hccl
#endif // HCCLV2_LOCAL_UB_RMA_BUFFER_H
