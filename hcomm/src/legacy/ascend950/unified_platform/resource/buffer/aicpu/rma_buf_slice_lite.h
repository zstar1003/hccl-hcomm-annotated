/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_RMA_BUFFER_SLICE_LITE_H
#define HCCLV2_RMA_BUFFER_SLICE_LITE_H

#include <string>
#include "hccl/base.h"

namespace Hccl {
class RmaBufSliceLite {
public:
    RmaBufSliceLite(u64 addr, u64 size, u32 lkey, u32 tokenId);

    inline u64 GetAddr() const { return addr_; }

    inline u64 GetSize() const { return size_; }

    inline u32 GetLkey() const { return lkey_; }

    inline u32 GetTokenId() const { return tokenId_; }

    std::string Describe() const;

private:
    u64 addr_;
    u64 size_;
    u32 lkey_;
    u32 tokenId_;
};
} // namespace Hccl
#endif
