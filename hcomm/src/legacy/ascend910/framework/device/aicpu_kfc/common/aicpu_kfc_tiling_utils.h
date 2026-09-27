/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __MC2_TILING_UTILS_INTERFACE_H__
#define __MC2_TILING_UTILS_INTERFACE_H__

#include "hccl/base.h"
#include "kernel_tiling/kernel_tiling.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

inline uint32_t MC2TilingGetVer(void* tiling) { return *(uint32_t*)tiling; }

inline uint32_t MC2TilingGetHcommCnt(void* tiling) { return *(uint32_t*)((uint8_t*)tiling + sizeof(uint32_t)); }

inline Mc2HcommCfg* MC2TilingGetHcommCfg(void* tiling, uint32_t idx)
{
    uint32_t cnt = MC2TilingGetHcommCnt(tiling);
    if (idx >= cnt) {
        return nullptr;
    }
    return (Mc2HcommCfg*)((uint8_t*)tiling + sizeof(uint32_t) + sizeof(uint32_t) + sizeof(Mc2ServerCfg)
                          + idx * sizeof(Mc2HcommCfg));
}

inline Mc2ServerCfg* MC2TilingGetServerCfg(void* tiling)
{
    return (Mc2ServerCfg*)((uint8_t*)tiling + sizeof(uint32_t) + sizeof(uint32_t));
}

#ifdef __cplusplus
}
#endif // __cplusplus
#endif // __MC2_TILING_UTILS_INTERFACE_H__
