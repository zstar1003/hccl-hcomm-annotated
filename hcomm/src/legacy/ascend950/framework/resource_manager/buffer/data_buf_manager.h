/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_DATA_BUFFER_MANAGER_H
#define HCCLV2_DATA_BUFFER_MANAGER_H

#include <string>
#include <memory>
#include <unordered_map>
#include "buffer.h"
#include "types.h"
#include "buffer_type.h"

namespace Hccl {
class CommunicatorImpl;
class DataBufManager {
public:
    ~DataBufManager();

    Buffer* Register(const std::string& opTag, BufferType type, std::shared_ptr<Buffer> buffer);

    void Deregister(const std::string& opTag, BufferType type);

    HcclResult Deregister(const std::string& opTag);

    Buffer* Get(const std::string& opTag, BufferType type);

    void Destroy();

private:
    bool IsExist(const std::string& opTag, BufferType type);

    std::unordered_map<std::string, std::unordered_map<BufferType, std::shared_ptr<Buffer>, std::EnumClassHash>> bufs;
};
} // namespace Hccl
#endif // HCCLV2_DATA_BUFFER_MANAGER_H
