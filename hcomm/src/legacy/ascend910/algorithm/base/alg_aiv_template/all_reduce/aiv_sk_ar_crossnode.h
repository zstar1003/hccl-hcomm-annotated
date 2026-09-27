/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIV_SK_AR_CROSSNODE_H
#define AIV_SK_AR_CROSSNODE_H

#include "aiv_communication_base.h"
#include "aiv_all_reduce_crossnode_91093.h"
// aiv reducescatter

extern "C" __aicore__ void sk_all_reduce_crossnode(SUPERKERNEL_LITE_ARGS_DEF)
{
    SUPERKERNEL_LITE_ARGS_EXTRACT;
    return sk_allreduce_crossnode(SUPERKERNEL_ARGS_CALL);
}

#endif /* AIV_AR_SUPERKERNEL_DETER_H */
