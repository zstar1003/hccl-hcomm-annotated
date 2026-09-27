/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_DEV_BUFFER_H
#define HCCLV2_DEV_BUFFER_H

#include <cstddef>
#include <string>
#include <memory>
#include "buffer.h"

namespace Hccl {

struct PolicyTag {};

class DevBuffer : public Buffer {
public:
    static std::shared_ptr<DevBuffer> Create(uintptr_t devAddr, std::size_t devSize);

    explicit DevBuffer(std::size_t allocSize);

    DevBuffer(std::size_t allocSize, std::uint32_t policy, PolicyTag tag);

    static std::shared_ptr<DevBuffer> CreateHugePageBuf(std::size_t size);

    ~DevBuffer() override;

    DevBuffer(const DevBuffer& that) = delete;

    DevBuffer& operator=(const DevBuffer& that) = delete;

    std::string Describe() const override;

    bool GetSelfOwned() const;

private:
    DevBuffer(uintptr_t devAddr, std::size_t devSize);

    bool selfOwned{false}; // 是否为自己所有
};

} // namespace Hccl
#endif // HCCL_BUFFER_H
