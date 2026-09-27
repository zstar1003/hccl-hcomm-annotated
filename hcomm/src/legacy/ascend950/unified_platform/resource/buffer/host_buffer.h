/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_HOST_BUFFER_H
#define HCCLV2_HOST_BUFFER_H

#include <cstdint>
#include <cstddef>
#include <string>
#include "buffer.h"
namespace Hccl {

class HostBuffer : public Buffer {
public:
    HostBuffer(uintptr_t devAddr, std::size_t devSize);

    explicit HostBuffer(std::size_t allocSize);

    ~HostBuffer() override;

    HostBuffer(const HostBuffer& that) = delete;

    HostBuffer& operator=(const HostBuffer& that) = delete;

    std::string Describe() const override;

    bool GetSelfOwned() const;

private:
    bool selfOwned{false}; // 是否为自己所有
};

} // namespace Hccl
#endif // HCCL_BUFFER_H
