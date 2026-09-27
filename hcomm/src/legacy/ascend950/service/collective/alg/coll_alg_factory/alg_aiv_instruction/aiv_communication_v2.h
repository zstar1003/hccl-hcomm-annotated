/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIV_COMMUNICATION_V2_H
#define AIV_COMMUNICATION_V2_H

#include "aiv_all_reduce_mesh_1D_twoshot.h"
#include "aiv_communication_base_v2.h"
#include "aiv_scatter_mesh_1d.h"
#include "aiv_broadcast_mesh_1d.h"
#include "aiv_all_reduce_mesh_1d_oneshot.h"
#include "aiv_all_gather_mesh_1d.h"
#include "aiv_all_to_all_mesh_1D.h"
#include "aiv_all_to_all_v_mesh_1D.h"
#include "aiv_reduce_mesh_1d.h"
#include "aiv_reduce_scatter_mesh_1d.h"
#include "aiv_reduce_scatter_mesh_1d_corectrl.h"
#include "aiv_reduce_scatter_mesh_1d_bigdata.h"
#include "aiv_send_mesh_1D.h"
#include "aiv_recv_mesh_1D.h"
#include "aiv_batchSendRecv.h"

using namespace AscendC;

#define EXPORT_AIV_META_INFO(kernel_name)                               \
    static const struct FunLevelKType kernel_name##_kernel_type_section \
        __attribute__((used, section(".ascend.meta." #kernel_name)))    \
        = {{F_TYPE_KTYPE, sizeof(unsigned int), K_TYPE_AIV}}

#define AIV_ALLGATHER_KERNEL_BATCH_DEF(type)                                               \
    extern "C" __global__ __aicore__ void aiv_all_gather_##type(EXTERN_KERNEL_ARGS_DEF_V2) \
    {                                                                                      \
        return AivAllGatherV2Mesh1D<type>(EXTERN_KERNEL_ARGS_CALL);                        \
    }                                                                                      \
    EXPORT_AIV_META_INFO(aiv_all_gather_##type)

#define AIV_SCATTER_KERNEL_BATCH_DEF(type)                                              \
    extern "C" __global__ __aicore__ void aiv_scatter_##type(EXTERN_KERNEL_ARGS_DEF_V2) \
    {                                                                                   \
        return AivScatterV2Mesh1D<type>(EXTERN_KERNEL_ARGS_CALL);                       \
    }                                                                                   \
    EXPORT_AIV_META_INFO(aiv_scatter_##type)

#define AIV_ALL_REDUCE_ONESHOT_KERNEL_BATCH_DEF(type)                                     \
    extern "C" __global__ __aicore__ void aiv_allreduce_##type(EXTERN_KERNEL_ARGS_DEF_V2) \
    {                                                                                     \
        return AivAllReduceV2Mesh1DOneShot<type>(EXTERN_KERNEL_ARGS_CALL);                \
    }                                                                                     \
    EXPORT_AIV_META_INFO(aiv_allreduce_##type)

#define AIV_BROADCAST_KERNEL_BATCH_DEF(type)                                              \
    extern "C" __global__ __aicore__ void aiv_broadcast_##type(EXTERN_KERNEL_ARGS_DEF_V2) \
    {                                                                                     \
        return AivBroadcastV2Mesh1D<type>(EXTERN_KERNEL_ARGS_CALL);                       \
    }                                                                                     \
    EXPORT_AIV_META_INFO(aiv_broadcast_##type)

#define AIV_ALLREDUCE_MESH1D_TWOSHOT_KERNEL_BATCH_DEF(type)                                              \
    extern "C" __global__ __aicore__ void aiv_allreduce_mesh1d_twoshot_##type(EXTERN_KERNEL_ARGS_DEF_V2) \
    {                                                                                                    \
        return AivAllReduceV2Mesh1DTwoShot<type>(EXTERN_KERNEL_ARGS_CALL);                               \
    }                                                                                                    \
    EXPORT_AIV_META_INFO(aiv_allreduce_mesh1d_twoshot_##type)

#define AIV_ALL_TO_ALL_KERNEL_BATCH_DEF(type)                                            \
    extern "C" __global__ __aicore__ void aiv_alltoall_##type(EXTERN_KERNEL_ARGS_DEF_V2) \
    {                                                                                    \
        return AivAlltoAllV2Mesh1D<type>(EXTERN_KERNEL_ARGS_CALL);                       \
    }                                                                                    \
    EXPORT_AIV_META_INFO(aiv_alltoall_##type)

#define AIV_REDUCE_KERNEL_BATCH_DEF(type)                                              \
    extern "C" __global__ __aicore__ void aiv_reduce_##type(EXTERN_KERNEL_ARGS_DEF_V2) \
    {                                                                                  \
        return AivReduceV2Mesh1D<type>(EXTERN_KERNEL_ARGS_CALL);                       \
    }                                                                                  \
    EXPORT_AIV_META_INFO(aiv_reduce_##type)

#define AIV_REDUCE_SCATTER_KERNEL_BATCH_DEF(type)                                              \
    extern "C" __global__ __aicore__ void aiv_reduce_scatter_##type(EXTERN_KERNEL_ARGS_DEF_V2) \
    {                                                                                          \
        return AivReduceScatterV2Mesh1DCoreCtrl<type>(EXTERN_KERNEL_ARGS_CALL);                \
    }                                                                                          \
    EXPORT_AIV_META_INFO(aiv_reduce_scatter_##type)

#define AIV_ALL_TO_ALL_V_KERNEL_BATCH_DEF(type)                                           \
    extern "C" __global__ __aicore__ void aiv_alltoallv_##type(EXTERN_KERNEL_ARGS_DEF_V2) \
    {                                                                                     \
        return AivAlltoAllVV2Mesh1D<type>(EXTERN_KERNEL_ARGS_CALL);                       \
    }                                                                                     \
    EXPORT_AIV_META_INFO(aiv_alltoallv_##type)

#define AIV_SEND_KERNEL_BATCH_DEF(type)                                              \
    extern "C" __global__ __aicore__ void aiv_send_##type(EXTERN_KERNEL_ARGS_DEF_V2) \
    {                                                                                \
        return AivSendV2Mesh1D<type>(EXTERN_KERNEL_ARGS_CALL);                       \
    }                                                                                \
    EXPORT_AIV_META_INFO(aiv_send_##type)

#define AIV_RECV_KERNEL_BATCH_DEF(type)                                              \
    extern "C" __global__ __aicore__ void aiv_recv_##type(EXTERN_KERNEL_ARGS_DEF_V2) \
    {                                                                                \
        return AivRecvV2Mesh1D<type>(EXTERN_KERNEL_ARGS_CALL);                       \
    }                                                                                \
    EXPORT_AIV_META_INFO(aiv_recv_##type)

#define AIV_BATCHSENDRECV_KERNEL_BATCH_DEF(type)                                              \
    extern "C" __global__ __aicore__ void aiv_batchSendRecv_##type(EXTERN_KERNEL_ARGS_DEF_V2) \
    {                                                                                         \
        return AivBatchSendRecvV2Mesh1D<type>(EXTERN_KERNEL_ARGS_CALL);                       \
    }                                                                                         \
    EXPORT_AIV_META_INFO(aiv_batchSendRecv_##type)

// 支持Reduce的数据类型
#define AIV_ATOMIC_DATA_TYPE_DEF(func) \
    func(float);                       \
    func(half);                        \
    func(int16_t);                     \
    func(int32_t);                     \
    func(int8_t);                      \
    func(bfloat16_t);                  \
    func(int64_t)

// 支持搬运的数据类型
#define AIV_COPY_DATA_TYPE_DEF(func) \
    func(half);                      \
    func(int16_t);                   \
    func(uint16_t);                  \
    func(float);                     \
    func(int32_t);                   \
    func(uint32_t);                  \
    func(int8_t);                    \
    func(uint8_t);                   \
    func(bfloat16_t);                \
    func(uint64_t);                  \
    func(int64_t);                   \
    func(fp8_e4m3fn_t);              \
    func(fp8_e5m2_t);                \
    func(fp8_e8m0_t);                \
    func(hifloat8_t)

// 定义各算子各数据类型Kernel入口
AIV_COPY_DATA_TYPE_DEF(AIV_ALLGATHER_KERNEL_BATCH_DEF);
AIV_COPY_DATA_TYPE_DEF(AIV_SCATTER_KERNEL_BATCH_DEF);
AIV_COPY_DATA_TYPE_DEF(AIV_ALL_TO_ALL_KERNEL_BATCH_DEF);
AIV_COPY_DATA_TYPE_DEF(AIV_ALL_TO_ALL_V_KERNEL_BATCH_DEF);
AIV_ATOMIC_DATA_TYPE_DEF(AIV_ALL_REDUCE_ONESHOT_KERNEL_BATCH_DEF);
AIV_COPY_DATA_TYPE_DEF(AIV_BROADCAST_KERNEL_BATCH_DEF);
AIV_ATOMIC_DATA_TYPE_DEF(AIV_ALLREDUCE_MESH1D_TWOSHOT_KERNEL_BATCH_DEF);
AIV_ATOMIC_DATA_TYPE_DEF(AIV_REDUCE_KERNEL_BATCH_DEF);
AIV_ATOMIC_DATA_TYPE_DEF(AIV_REDUCE_SCATTER_KERNEL_BATCH_DEF);
AIV_COPY_DATA_TYPE_DEF(AIV_SEND_KERNEL_BATCH_DEF);
AIV_COPY_DATA_TYPE_DEF(AIV_RECV_KERNEL_BATCH_DEF);
AIV_COPY_DATA_TYPE_DEF(AIV_BATCHSENDRECV_KERNEL_BATCH_DEF);

#endif /* AIV_COMMUNICATION_V2_H */
