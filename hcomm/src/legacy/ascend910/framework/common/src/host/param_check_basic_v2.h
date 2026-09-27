/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PARAM_CHECK_PUB_BASIC_V2_H
#define PARAM_CHECK_PUB_BASIC_V2_H

#if (!defined(HCCD)) && (!defined(CCL_KERNEL_AICPU))
#include "adapter_rts.h"
#endif

#define HCCLV2_FUNC_RUN(func, ...)                  \
    do {                                            \
        bool isSupportV2 = false;                   \
        CHK_RET(hrtGetHcclV2Support(&isSupportV2)); \
        if (isSupportV2) {                          \
            return func;                            \
        }                                           \
    } while (0)

#endif // PARAM_CHECK_PUB_BASIC_V2_H
