/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <future>
#include <map>
#include <string>
#include <hccl/hccl_types.h>

#include "op_base.h"

using namespace std;
using namespace hccl;

HcclResult GetCaptureInfo(
    [[maybe_unused]] aclrtStream stream, [[maybe_unused]] aclmdlRICaptureStatus& captureStatus,
    [[maybe_unused]] uint64_t& modelId, [[maybe_unused]] bool& isCapture)
{
    HCCL_WARNING("[%s]Stream capture does not support!", __func__);
    return HCCL_SUCCESS;
}

HcclResult HcclAllReduceInner(
    [[maybe_unused]] void* sendBuf, [[maybe_unused]] void* recvBuf, [[maybe_unused]] uint64_t count,
    [[maybe_unused]] HcclDataType dataType, [[maybe_unused]] HcclReduceOp op, [[maybe_unused]] HcclComm comm,
    [[maybe_unused]] aclrtStream stream)
{
    HCCL_WARNING("[%s]HcclAllReduceInner does not support!", __func__);
    return HCCL_SUCCESS;
}
