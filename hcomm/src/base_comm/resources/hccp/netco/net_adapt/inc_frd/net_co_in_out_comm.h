/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef NET_CO_IN_OUT_COMM_H
#define NET_CO_IN_OUT_COMM_H

#include "bkf_comm.h"

#ifdef __cplusplus
extern "C" {
#endif
#define NET_CO_DFT_INST_ID (1)

/* 避免外部包含过多文件，typedef */
typedef struct BifrostCncoiSliceKey BifrostCncoiSliceKeyT;
extern BifrostCncoiSliceKeyT *g_NetCoDftSliceKey;

#ifdef __cplusplus
}
#endif

#endif
