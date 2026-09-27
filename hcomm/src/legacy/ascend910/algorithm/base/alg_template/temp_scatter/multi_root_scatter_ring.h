/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MULTI_ROOT_SCATTER_RING_H
#define MULTI_ROOT_SCATTER_RING_H

#include "multi_root_scatter_ring_pub.h"

namespace hccl {
struct SliceSendRange {
    u32 sliceIdx = 0;
    u32 startRank = 0;
    u32 endRank = 0;
};

bool DscendSortWithSliceSendEnd(const SliceSendRange& a, const SliceSendRange& b);
} // namespace hccl

#endif
