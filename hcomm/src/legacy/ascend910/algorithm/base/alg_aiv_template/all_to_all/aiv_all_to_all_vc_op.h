/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIV_ALL_TO_ALL_VC_OP_H
#define AIV_ALL_TO_ALL_VC_OP_H

#include "aiv_communication_base.h"

#include "aiv_all_to_all_v_910b.h"
#include "aiv_all_to_all_vc_910b_no_loop.h"
#include "aiv_all_to_all_vc_910b_graph.h"
#include "aiv_all_to_all_91093_single.h"

#define AIV_ALL_TO_ALL_VC_KERNEL_BATCH_DEF(type)                                                 \
    extern "C" __global__ __aicore__ void aiv_all_to_all_vc_##type(EXTERN_KERNEL_ARGS_DEF)       \
    {                                                                                            \
        AIV_INFO_HINT;                                                                           \
        if (isOpBase) {                                                                          \
            if (extraArgs.maxCount * sizeof(type) > bufferSize) {                                \
                return aiv_all_to_all_vc_910b<type>(EXTERN_KERNEL_ARGS_CALL);                    \
            } else {                                                                             \
                return aiv_all_to_all_vc_910b_no_loop<type>(EXTERN_KERNEL_ARGS_CALL);            \
            }                                                                                    \
        } else {                                                                                 \
            if (devType == DEV_TYPE_910_93) {                                                    \
                return aiv_all_to_all_vc_91093_single_graph<type>(KERNEL_ARGS_CALL, &extraArgs); \
            } else {                                                                             \
                return aiv_all_to_all_vc_910b_graph<type>(EXTERN_KERNEL_ARGS_CALL);              \
            }                                                                                    \
        }                                                                                        \
    }                                                                                            \
    EXPORT_AIV_META_INFO(aiv_all_to_all_vc_##type)

// 定义算子各数据类型Kernel入口
AIV_COPY_DATA_TYPE_DEF(AIV_ALL_TO_ALL_VC_KERNEL_BATCH_DEF);

#endif /* AIV_ALL_TO_ALL_VC_OP_H */
