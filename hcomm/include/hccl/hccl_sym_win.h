/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_SYM_WIN_H
#define HCCL_SYM_WIN_H

#include <stdint.h>
#include <securec.h>
#include <arpa/inet.h>
#include "acl/acl_rt.h"
#include "hccl_types.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/**
 * @brief Get symmetric memory pointer.
 *
 * @param winHandle A pointer identifying the registered memory window handle.
 * @param offset A size_t identifying the offset of symmetric memory heap.
 * @param peerRank A u_integer identifying the identify for the peer rank.
 * @param ptr A pointer identifying the symmetric memory heap address.
 * @return HcclResult
 */
extern HcclResult HcclSymWinGetPeerPointer(HcclCommSymWindow winHandle, size_t offset, uint32_t peerRank, void** ptr);

/**
 * @brief Get remote symmetric memory pointer (for URMA).
 *
 * @param winHandle A pointer identifying the registered memory window handle.
 * @param offset A size_t identifying the offset of symmetric memory heap.
 * @param peerRank A u_integer identifying the identify for the peer rank.
 * @param ptr A pointer identifying the remote symmetric memory heap address.
 * @return HcclResult
 */
extern HcclResult HcclSymWinGetRemoteAddr(HcclCommSymWindow winHandle, size_t offset, uint32_t peerRank, void** ptr);

#ifdef __cplusplus
}
#endif // __cplusplus
#endif // HCCL_SYM_WIN_H
