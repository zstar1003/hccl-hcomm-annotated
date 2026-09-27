/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "local_ub_rma_buffer.h"

namespace Hccl {

LocalUbRmaBuffer::LocalUbRmaBuffer(std::shared_ptr<Buffer> buf, RdmaHandle rdmaHandle)
    : LocalRmaBuffer(buf, RmaType::UB),
      rdmaHandle(rdmaHandle)
{
    ;
}

LocalUbRmaBuffer::~LocalUbRmaBuffer() { ; }

string LocalUbRmaBuffer::Describe() const { return ""; }

std::unique_ptr<Serializable> LocalUbRmaBuffer::GetExchangeDto() { return std::unique_ptr<Serializable>(nullptr); }

u32 LocalUbRmaBuffer::GetTokenId() const { return tokenId; }

u32 LocalUbRmaBuffer::GetTokenValue() const { return tokenValue; }

TokenIdHandle LocalUbRmaBuffer::GetTokenIdHandle() const { return tokenIdHandle; }

u32 GetUbToken()
{
    constexpr uint32_t tokenValue = 1;
    return tokenValue;
}

} // namespace Hccl
