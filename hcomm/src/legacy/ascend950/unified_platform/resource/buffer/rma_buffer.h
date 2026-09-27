/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_RMA_BUFFER_H
#define HCCLV2_RMA_BUFFER_H

#include "types.h"
#include "orion_adapter_hccp.h"
#include "orion_adapter_rts.h"

namespace Hccl {
// 待删除 CCU 重构完成后不再需要该文件
struct IpcRmaBufferExchangeData {
    char_t name[RTS_IPC_MEM_NAME_LEN]{0};
    u64 addr{0};
    u64 size{0};
    u64 offset{0};
    u32 pid{0};
};

struct RdmaRmaBufferExchangeData {
    u64 addr{0};
    u64 size{0};
    u8 key[RDMA_MEM_KEY_MAX_LEN]{0};
    u32 rkey{0};
};

struct UbRmaBufferExchangeData {
    u64 addr{0};
    u64 size{0};
    u8 key[HRT_UB_MEM_KEY_MAX_LEN]{0};
    u32 tokenValue{0};
    u32 tokenId{0};
    u32 keySize{0};
};

} // namespace Hccl
#endif
