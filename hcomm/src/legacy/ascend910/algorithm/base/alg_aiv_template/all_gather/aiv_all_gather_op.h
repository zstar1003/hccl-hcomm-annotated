/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIV_ALL_GATHER_OP_H
#define AIV_ALL_GATHER_OP_H

#include "aiv_communication_base.h"

#include "aiv_all_gather_910b_graph.h"
#include "aiv_all_gather_91093_smalldata.h"
#include "aiv_all_gather_910b_smalldata.h"
#include "aiv_all_gather_910b_bigdata.h"
#include "aiv_all_gather_910B_rdma.h"
#include "aiv_all_gather_910B_rdma_graph.h"
#include "aiv_all_gather_crossnode_91093.h"
#include "aiv_all_gather_crossnode_91093_graph.h"

#define AIV_ALL_GATHER_KERNEL_DEF(type)                                           \
    __aicore__ inline void aiv_all_gather_##type##_inner(KERNEL_ARGS_DEF)         \
    {                                                                             \
        AIV_INFO_HINT;                                                            \
        if (isOpBase) {                                                           \
            if (aivRdmaStep >= 0) {                                               \
                return aiv_all_gather_910b_rdma<type>(KERNEL_ARGS_CALL);          \
            } else if (len * sizeof(type) > AIV_ALL_GATHER_SMALL_SIZE) {          \
                return aiv_all_gather_910b_bigdata<type>(KERNEL_ARGS_CALL);       \
            } else {                                                              \
                return aiv_all_gather_910b_smalldata<type>(KERNEL_ARGS_CALL);     \
            }                                                                     \
        } else {                                                                  \
            if (aivRdmaStep >= 0) {                                               \
                return aiv_all_gather_910b_rdma_graph<type>(KERNEL_ARGS_CALL);    \
            } else if (devType == DEV_TYPE_910B) {                                \
                return aiv_all_gather_910b_bigdata_graph<type>(KERNEL_ARGS_CALL); \
            } else {                                                              \
                return aiv_all_gather_91093_smalldata<type>(KERNEL_ARGS_CALL);    \
            }                                                                     \
        }                                                                         \
    }

#define AIV_ALL_GATHER_KERNEL_DEF_A3(type)                                      \
    __aicore__ inline void aiv_all_gather_cn_##type##_inner(KERNEL_ARGS_DEF_A3) \
    {                                                                           \
        AIV_INFO_HINT;                                                          \
        if (isOpBase) {                                                         \
            return aiv_all_gather_crossnode_91093<type>(KERNEL_ARGS_CALL_A3);   \
        }                                                                       \
        return aiv_all_gather_crossnode_91093_graph<type>(KERNEL_ARGS_CALL_A3); \
    }

#if defined(BUILD_SK_FUNC) && defined(SK_FUNC_ID)
#define AIV_ALL_GATHER_KERNEL_BATCH_DEF(type) \
    AIV_ALL_GATHER_KERNEL_DEF(type);          \
    SK_BIND_FUNC_DEF_A2(aiv_all_gather_##type, SK_FUNC_ID)
#else
#define AIV_ALL_GATHER_KERNEL_BATCH_DEF(type)  \
    AIV_ALL_GATHER_KERNEL_DEF(type);           \
    GLOBAL_FUNC_DEF_A2(aiv_all_gather_##type); \
    SuperKernelBindA2(aiv_all_gather_##type)
#endif

#if defined(BUILD_SK_FUNC) && defined(SK_FUNC_ID)
#define AIV_ALL_GATHER_KERNEL_BATCH_DEF_A3(type) \
    AIV_ALL_GATHER_KERNEL_DEF_A3(type);          \
    SK_BIND_FUNC_DEF_A3(aiv_all_gather_cn_##type, SK_FUNC_ID)
#else
#define AIV_ALL_GATHER_KERNEL_BATCH_DEF_A3(type)  \
    AIV_ALL_GATHER_KERNEL_DEF_A3(type);           \
    GLOBAL_FUNC_DEF_A3(aiv_all_gather_cn_##type); \
    SuperKernelBindA3(aiv_all_gather_cn_##type)
#endif

// 定义算子各数据类型Kernel入口
AIV_COPY_DATA_TYPE_DEF(AIV_ALL_GATHER_KERNEL_BATCH_DEF);
AIV_COPY_DATA_TYPE_DEF(AIV_ALL_GATHER_KERNEL_BATCH_DEF_A3);

#endif /* AIV_ALL_GATHER_OP_H */
