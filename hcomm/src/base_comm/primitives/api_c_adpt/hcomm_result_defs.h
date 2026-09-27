/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_RESULT_DEFS_H
#define HCOMM_RESULT_DEFS_H

#ifdef __cplusplus
extern "C" {
#endif

enum HcommResultDef {
    HCOMM_SUCCESS = 0,
    HCOMM_E_PARA = 1,
    HCOMM_E_PTR = 2,
    HCOMM_E_MEMORY = 3,
    HCOMM_E_INTERNAL = 4,
    HCOMM_E_NOT_SUPPORT = 5,
    HCOMM_E_NOT_FOUND = 6
};

#ifdef __cplusplus
}
#endif

#endif
