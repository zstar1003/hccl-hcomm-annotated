/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIV_AG_SUPERKERNEL_H
#define AIV_AG_SUPERKERNEL_H

#include "aiv_communication_base.h"
#include "aiv_all_gather_91093_smalldata.h"
#include "aiv_all_gather_910b_graph.h"

extern "C" __aicore__ void sk_allgather(SUPERKERNEL_LITE_ARGS_DEF)
{
    SUPERKERNEL_LITE_ARGS_EXTRACT;
    if (devType == DEV_TYPE_910_93) {
        return sk_all_gather_91093_smalldata(SUPERKERNEL_ARGS_CALL);
    } else if (devType == DEV_TYPE_910B) {
        return sk_all_gather_910b_bigdata(SUPERKERNEL_ARGS_CALL);
    }
}

#endif /* AIV_AG_SUPERKERNEL_H */
