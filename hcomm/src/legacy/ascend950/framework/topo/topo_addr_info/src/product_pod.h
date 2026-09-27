/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __PRODUCT_POD_H__
#define __PRODUCT_POD_H__

#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

int PodGetRootinfoLen(size_t* len);

int PodGetRootinfo(int npu_id, unsigned int mainboard_id, void* buf, size_t* len);

#ifdef __cplusplus
}
#endif

#endif // __PRODUCT_POD_H__
