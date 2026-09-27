/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_PRIM_QUEUE_H
#define HCCLV2_PRIM_QUEUE_H

#include <memory>
#include "primitive.h"
#include "hierarchical_queue.h"

namespace Hccl {

using namespace std;

class Primitive;

class PrimQueue : public HierarchicalQueue<Primitive, PrimQueue>, public enable_shared_from_this<PrimQueue> {
public:
    void Append(unique_ptr<Primitive> prim) override;
};

void PrintPrimQueue(const PrimQueue& queue);
} // namespace Hccl

#endif
