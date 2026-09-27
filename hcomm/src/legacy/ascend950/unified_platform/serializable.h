/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_SERIALIZABLE_H
#define HCCL_SERIALIZABLE_H

#include <string>
#include "binary_stream.h"

namespace Hccl {
class Serializable {
public:
    virtual ~Serializable() = default;

    virtual void Serialize(BinaryStream& stream) = 0;

    virtual void Deserialize(BinaryStream& stream) = 0;

    virtual std::string Describe() const = 0;
};
} // namespace Hccl

#endif // HCCL_SERIALIZABLE_H
