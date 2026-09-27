/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIV_AR_SUPERKERNEL_H
#define AIV_AR_SUPERKERNEL_H

#include "aiv_communication_base.h"
#include "aiv_all_reduce_91093.h"
#include "aiv_all_reduce_910b_smalldata_graph.h"
#include "aiv_all_reduce_910b_bigdata_graph.h"

extern "C" __aicore__ void sk_allreduce(SUPERKERNEL_LITE_ARGS_DEF)
{
    SUPERKERNEL_LITE_ARGS_EXTRACT;
    if (devType == DEV_TYPE_910_93) {
        return sk_all_reduce_91093(SUPERKERNEL_ARGS_CALL);
    } else if (devType == DEV_TYPE_910B) {
        if (args->len * args->unitSize > UB_MAX_DATA_SIZE) {
            return sk_all_reduce_910b_bigdata_graph(SUPERKERNEL_ARGS_CALL);
        } else {
            return sk_all_reduce_910b_smalldata_graph(SUPERKERNEL_ARGS_CALL);
        }
    }
}

#endif /* AIV_AR_SUPERKERNEL_H */
