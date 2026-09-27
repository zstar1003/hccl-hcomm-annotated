/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_ADAPTER_RTS_H
#define HCOMM_ADAPTER_RTS_H

#include "acl/acl_rt.h"
#include "rt_external.h"

#include "hccl_types.h"

namespace hcomm {

typedef struct {
    uint64_t va;
    uint64_t size;
    uint32_t tokenId;
    uint32_t tokenValue;
} rtMemUbTokenInfo;

HcclResult RtsUbDevQueryInfo(const rtUbDevQueryCmd cmd, rtMemUbTokenInfo& devInfo);

} // namespace hcomm
#endif // HCOMM_ADAPTER_RTS_H
