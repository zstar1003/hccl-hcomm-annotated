/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TOPO_ADDR_INFO_ERR_H
#define TOPO_ADDR_INFO_ERR_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── TopoAddrResult 错误码枚举 ── */
/* 值对齐 include/hccl/hccl_types.h 中的 HcclResult，便于调用方识别 */
typedef enum {
    TOPO_SUCCESS = 0,
    TOPO_ERR_PARA = 1,       // 参数错误
    TOPO_ERR_PTR = 2,        // 空指针
    TOPO_ERR_MEMORY = 3,     // 内存错误
    TOPO_ERR_INTERNAL = 4,   // 内部错误
    TOPO_ERR_NOT_FOUND = 6,  // 未找到
    TOPO_ERR_SYSCALL = 8,    // 系统调用失败
    TOPO_ERR_OPEN_FILE = 10, // 文件打开失败
    TOPO_ERR_OOM = 24,       // 内存耗尽
} TopoAddrResult;

#ifdef __cplusplus
}
#endif

#endif /* TOPO_ADDR_INFO_ERR_H */
