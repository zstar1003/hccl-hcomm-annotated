/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIV_REDUCE_SCATTER_OP_H
#define AIV_REDUCE_SCATTER_OP_H

#include "aiv_communication_base.h"

#include "aiv_reduce_scatter_910b_graph.h"
#include "aiv_reduce_scatter_91093_smalldata.h"
#include "aiv_reduce_scatter_910b_smalldata.h"
#include "aiv_reduce_scatter_910b_bigdata.h"
#include "aiv_reduce_scatter_910b_middata.h"

#include "aiv_reduce_scatter_910b_rdma.h"
#include "aiv_reduce_scatter_910b_rdma_graph.h"

#include "aiv_reduce_scatter_deter_910b_smalldata.h"
#include "aiv_reduce_scatter_deter_910b_middata.h"
#include "aiv_reduce_scatter_deter_910b_bigdata.h"

#include "aiv_reduce_scatter_crossnode_91093.h"
#include "aiv_reduce_scatter_crossnode_91093_graph.h"
#include "aiv_reduce_scatter_91093_deter.h"

#define AIV_REDUCE_SCATTER_KERNEL_DEF(type)                                                                   \
    __aicore__ inline void aiv_reduce_scatter_##type##_inner(KERNEL_ARGS_DEF)                                 \
    {                                                                                                         \
        AIV_INFO_HINT;                                                                                        \
        if (devType == DEV_TYPE_910B && deterministic == 1) {                                                 \
            if (rankSize * len * sizeof(type) < AIV_REDUCE_SCATTER_DETER_SMALL_SIZE) {                        \
                return aiv_reduce_scatter_deter_910b_smalldata<type>(KERNEL_ARGS_CALL);                       \
            } else if (rankSize * len * sizeof(type) <= AIV_REDUCE_SCATTER_DETER_MID_SIZE) {                  \
                return aiv_reduce_scatter_deter_910b_middata<type>(KERNEL_ARGS_CALL);                         \
            } else {                                                                                          \
                return aiv_reduce_scatter_deter_910b_bigdata<type>(KERNEL_ARGS_CALL);                         \
            }                                                                                                 \
        }                                                                                                     \
        if (isOpBase) {                                                                                       \
            if (aivRdmaStep >= 0) {                                                                           \
                return aiv_reduce_scatter_910b_rdma<type>(KERNEL_ARGS_CALL);                                  \
            } else if (                                                                                       \
                len * sizeof(type)                                                                            \
                > (devType == DEV_TYPE_910B ? AIV_REDUCE_SCATTER_MID_SIZE : AIV_REDUCE_SCATTER_SMALL_SIZE)) { \
                return aiv_reduce_scatter_910b_bigdata<type>(KERNEL_ARGS_CALL);                               \
            } else if (len * sizeof(type) > UB_MAX_DATA_SIZE) {                                               \
                return aiv_reduce_scatter_910b_middata<type>(KERNEL_ARGS_CALL);                               \
            } else {                                                                                          \
                return aiv_reduce_scatter_910b_smalldata<type>(KERNEL_ARGS_CALL);                             \
            }                                                                                                 \
        } else {                                                                                              \
            if (aivRdmaStep >= 0) {                                                                           \
                return aiv_reduce_scatter_910b_rdma_graph<type>(KERNEL_ARGS_CALL);                            \
            } else if (devType == DEV_TYPE_910B) {                                                            \
                return aiv_reduce_scatter_910b_bigdata_graph<type>(KERNEL_ARGS_CALL);                         \
            } else {                                                                                          \
                return aiv_reduce_scatter_91093_smalldata<type>(KERNEL_ARGS_CALL);                            \
            }                                                                                                 \
        }                                                                                                     \
    }

#define AIV_REDUCE_SCATTER_KERNEL_DEF_A3(type)                                      \
    __aicore__ inline void aiv_reduce_scatter_cn_##type##_inner(KERNEL_ARGS_DEF_A3) \
    {                                                                               \
        AIV_INFO_HINT;                                                              \
        if (deterministic == 1) {                                                   \
            return aiv_reduce_scatter_91093_deter<type>(KERNEL_ARGS_CALL_A3);       \
        }                                                                           \
        if (isOpBase) {                                                             \
            return aiv_reduce_scatter_crossnode_91093<type>(KERNEL_ARGS_CALL_A3);   \
        }                                                                           \
        return aiv_reduce_scatter_crossnode_91093_graph<type>(KERNEL_ARGS_CALL_A3); \
    }

#if defined(BUILD_SK_FUNC) && defined(SK_FUNC_ID)
#define AIV_REDUCE_SCATTER_KERNEL_BATCH_DEF(type) \
    AIV_REDUCE_SCATTER_KERNEL_DEF(type);          \
    SK_BIND_FUNC_DEF_A2(aiv_reduce_scatter_##type, SK_FUNC_ID)
#else
#define AIV_REDUCE_SCATTER_KERNEL_BATCH_DEF(type)  \
    AIV_REDUCE_SCATTER_KERNEL_DEF(type);           \
    GLOBAL_FUNC_DEF_A2(aiv_reduce_scatter_##type); \
    SuperKernelBindA2(aiv_reduce_scatter_##type)
#endif

#if defined(BUILD_SK_FUNC) && defined(SK_FUNC_ID)
#define AIV_REDUCE_SCATTER_KERNEL_BATCH_DEF_A3(type) \
    AIV_REDUCE_SCATTER_KERNEL_DEF_A3(type);          \
    SK_BIND_FUNC_DEF_A3(aiv_reduce_scatter_cn_##type, SK_FUNC_ID)
#else
#define AIV_REDUCE_SCATTER_KERNEL_BATCH_DEF_A3(type)  \
    AIV_REDUCE_SCATTER_KERNEL_DEF_A3(type);           \
    GLOBAL_FUNC_DEF_A3(aiv_reduce_scatter_cn_##type); \
    SuperKernelBindA3(aiv_reduce_scatter_cn_##type)
#endif

// 定义算子各数据类型Kernel入口
AIV_ATOMIC_DATA_TYPE_DEF(AIV_REDUCE_SCATTER_KERNEL_BATCH_DEF);
AIV_ATOMIC_DATA_TYPE_DEF(AIV_REDUCE_SCATTER_KERNEL_BATCH_DEF_A3);

#endif /* AIV_REDUCE_SCATTER_OP_H */
