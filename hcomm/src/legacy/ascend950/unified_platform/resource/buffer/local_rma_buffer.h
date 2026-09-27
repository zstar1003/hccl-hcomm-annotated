/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_LOCAL_RMA_BUFFER_H
#define HCCLV2_LOCAL_RMA_BUFFER_H

#include <string>
#include <memory>
#include <vector>

#include "buffer.h"
#include "rma_type.h"

#include "serializable.h"
#include "rma_buffer_lite.h"

namespace Hccl {

class LocalRmaBuffer {
public:
    LocalRmaBuffer(std::shared_ptr<Buffer> buf, RmaType type) : buf(buf), rmaType(type) {}

    LocalRmaBuffer(std::shared_ptr<Buffer> buf, RmaType type, bool isAlias) : buf(buf), rmaType(type), isAlias_(isAlias)
    {}

    virtual ~LocalRmaBuffer() = default;

    virtual std::string Describe() const = 0;

    Buffer* GetBuf() const { return buf.get(); }

    RmaType GetRmaType() const // used for grant check
    {
        return rmaType;
    }

    bool IsAlias() const { return isAlias_; }

    u64 GetMemHandle() const { return memHandle; }

    size_t GetSize() const { return buf->GetSize(); }

    uintptr_t GetAddr() const { return buf->GetAddr(); }

    virtual std::unique_ptr<Serializable> GetExchangeDto()
    {
        HCCL_ERROR("this is base class, not support.");
        return nullptr;
    }

protected:
    std::shared_ptr<Buffer> buf;
    RmaType rmaType;
    u64 memHandle{0};
    bool isAlias_{false}; // 区分该RmaBuffer对象是否为某一注册Buffer的子集
};

} // namespace Hccl
#endif
