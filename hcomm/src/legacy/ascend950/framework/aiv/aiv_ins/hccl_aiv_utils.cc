/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <mutex>
#include <vector>
#include <iostream>
#include <fstream>
#include "mmpa_api.h"
#include "acl/acl_rt.h"
#include "env_config_v2.h"
#include "hccl_aiv_utils.h"
#include "aicpu/launch_device_v2.h"

using namespace std;

namespace Hccl {
constexpr u32 SIG_MOVE_LEFT_BITS = 20;
constexpr u32 MAX_BIN_FILE_SIZE = 100 * 1024 * 1024;
constexpr s32 RESET_TAIL_SYNC_TAG = 2;
constexpr uint64_t MIN_NPU_TIMEOUT = 1;
constexpr uint64_t MAX_NPU_TIMEOUT = 254;

static bool g_init = false;
static mutex g_mut;
static aclrtBinHandle g_binHandle;
static std::unordered_map<const s8*, aclrtFuncHandle> g_aivFuncMap;
static std::unordered_map<const s8*, std::string> g_aivNameMap;

using AivKernelInfo = struct AivKernelInfoDef {
    const char* kernelName;
    HcclCMDType cmdType;
    DataType dataType;
    KernelArgsType argsType;

    AivKernelInfoDef(
        const char* kernelName, HcclCMDType cmdType, DataType dataType,
        KernelArgsType argsType = KernelArgsType::ARGS_TYPE_SERVER)
        : kernelName(kernelName),
          cmdType(cmdType),
          dataType(dataType),
          argsType(argsType)
    {}
};

static std::vector<AivKernelInfo> g_aivKernelInfoList = {
    // scatter
    {"aiv_scatter_half", HcclCMDType::HCCL_CMD_SCATTER, DataType::FP16},
    {"aiv_scatter_int16_t", HcclCMDType::HCCL_CMD_SCATTER, DataType::INT16},
    {"aiv_scatter_uint16_t", HcclCMDType::HCCL_CMD_SCATTER, DataType::UINT16},
    {"aiv_scatter_float", HcclCMDType::HCCL_CMD_SCATTER, DataType::FP32},
    {"aiv_scatter_uint64_t", HcclCMDType::HCCL_CMD_SCATTER, DataType::FP64},
    {"aiv_scatter_int32_t", HcclCMDType::HCCL_CMD_SCATTER, DataType::INT32},
    {"aiv_scatter_uint32_t", HcclCMDType::HCCL_CMD_SCATTER, DataType::UINT32},
    {"aiv_scatter_int8_t", HcclCMDType::HCCL_CMD_SCATTER, DataType::INT8},
    {"aiv_scatter_uint8_t", HcclCMDType::HCCL_CMD_SCATTER, DataType::UINT8},
    {"aiv_scatter_bfloat16_t", HcclCMDType::HCCL_CMD_SCATTER, DataType::BFP16},
    {"aiv_scatter_uint64_t", HcclCMDType::HCCL_CMD_SCATTER, DataType::INT64},
    {"aiv_scatter_int64_t", HcclCMDType::HCCL_CMD_SCATTER, DataType::UINT64},
    {"aiv_scatter_hifloat8_t", HcclCMDType::HCCL_CMD_SCATTER, DataType::HIF8},
    {"aiv_scatter_fp8_e4m3fn_t", HcclCMDType::HCCL_CMD_SCATTER, DataType::FP8E4M3},
    {"aiv_scatter_fp8_e5m2_t", HcclCMDType::HCCL_CMD_SCATTER, DataType::FP8E5M2},
    {"aiv_scatter_fp8_e8m0_t", HcclCMDType::HCCL_CMD_SCATTER, DataType::FP8E8M0},

    {"aiv_all_gather_half", HcclCMDType::HCCL_CMD_ALLGATHER, DataType::FP16},
    {"aiv_all_gather_int16_t", HcclCMDType::HCCL_CMD_ALLGATHER, DataType::INT16},
    {"aiv_all_gather_uint16_t", HcclCMDType::HCCL_CMD_ALLGATHER, DataType::UINT16},
    {"aiv_all_gather_float", HcclCMDType::HCCL_CMD_ALLGATHER, DataType::FP32},
    {"aiv_all_gather_uint64_t", HcclCMDType::HCCL_CMD_ALLGATHER, DataType::FP64},
    {"aiv_all_gather_int32_t", HcclCMDType::HCCL_CMD_ALLGATHER, DataType::INT32},
    {"aiv_all_gather_uint32_t", HcclCMDType::HCCL_CMD_ALLGATHER, DataType::UINT32},
    {"aiv_all_gather_int8_t", HcclCMDType::HCCL_CMD_ALLGATHER, DataType::INT8},
    {"aiv_all_gather_uint8_t", HcclCMDType::HCCL_CMD_ALLGATHER, DataType::UINT8},
    {"aiv_all_gather_bfloat16_t", HcclCMDType::HCCL_CMD_ALLGATHER, DataType::BFP16},
    {"aiv_all_gather_uint64_t", HcclCMDType::HCCL_CMD_ALLGATHER, DataType::INT64},
    {"aiv_all_gather_int64_t", HcclCMDType::HCCL_CMD_ALLGATHER, DataType::UINT64},
    {"aiv_all_gather_hifloat8_t", HcclCMDType::HCCL_CMD_ALLGATHER, DataType::HIF8},
    {"aiv_all_gather_fp8_e4m3fn_t", HcclCMDType::HCCL_CMD_ALLGATHER, DataType::FP8E4M3},
    {"aiv_all_gather_fp8_e5m2_t", HcclCMDType::HCCL_CMD_ALLGATHER, DataType::FP8E5M2},
    {"aiv_all_gather_fp8_e8m0_t", HcclCMDType::HCCL_CMD_ALLGATHER, DataType::FP8E8M0},
    // allreduce
    {"aiv_allreduce_half", HcclCMDType::HCCL_CMD_ALLREDUCE, DataType::FP16},
    {"aiv_allreduce_int16_t", HcclCMDType::HCCL_CMD_ALLREDUCE, DataType::INT16},
    {"aiv_allreduce_float", HcclCMDType::HCCL_CMD_ALLREDUCE, DataType::FP32},
    {"aiv_allreduce_int32_t", HcclCMDType::HCCL_CMD_ALLREDUCE, DataType::INT32},
    {"aiv_allreduce_int8_t", HcclCMDType::HCCL_CMD_ALLREDUCE, DataType::INT8},
    {"aiv_allreduce_bfloat16_t", HcclCMDType::HCCL_CMD_ALLREDUCE, DataType::BFP16},
    {"aiv_allreduce_int64_t", HcclCMDType::HCCL_CMD_ALLREDUCE, DataType::INT64},
    // broadcast
    {"aiv_broadcast_half", HcclCMDType::HCCL_CMD_BROADCAST, DataType::FP16},
    {"aiv_broadcast_int16_t", HcclCMDType::HCCL_CMD_BROADCAST, DataType::INT16},
    {"aiv_broadcast_uint16_t", HcclCMDType::HCCL_CMD_BROADCAST, DataType::UINT16},
    {"aiv_broadcast_float", HcclCMDType::HCCL_CMD_BROADCAST, DataType::FP32},
    {"aiv_broadcast_uint64_t", HcclCMDType::HCCL_CMD_BROADCAST, DataType::FP64},
    {"aiv_broadcast_int32_t", HcclCMDType::HCCL_CMD_BROADCAST, DataType::INT32},
    {"aiv_broadcast_uint32_t", HcclCMDType::HCCL_CMD_BROADCAST, DataType::UINT32},
    {"aiv_broadcast_int8_t", HcclCMDType::HCCL_CMD_BROADCAST, DataType::INT8},
    {"aiv_broadcast_uint8_t", HcclCMDType::HCCL_CMD_BROADCAST, DataType::UINT8},
    {"aiv_broadcast_bfloat16_t", HcclCMDType::HCCL_CMD_BROADCAST, DataType::BFP16},
    {"aiv_broadcast_uint64_t", HcclCMDType::HCCL_CMD_BROADCAST, DataType::INT64},
    {"aiv_broadcast_int64_t", HcclCMDType::HCCL_CMD_BROADCAST, DataType::UINT64},
    {"aiv_broadcast_hifloat8_t", HcclCMDType::HCCL_CMD_BROADCAST, DataType::HIF8},
    {"aiv_broadcast_fp8_e4m3fn_t", HcclCMDType::HCCL_CMD_BROADCAST, DataType::FP8E4M3},
    {"aiv_broadcast_fp8_e5m2_t", HcclCMDType::HCCL_CMD_BROADCAST, DataType::FP8E5M2},
    {"aiv_broadcast_fp8_e8m0_t", HcclCMDType::HCCL_CMD_BROADCAST, DataType::FP8E8M0},
    // allreduce two shot
    {"aiv_allreduce_mesh1d_twoshot_half", HcclCMDType::HCCL_CMD_ALLREDUCE, DataType::FP16,
     KernelArgsType::ARGS_TYPE_TWO_SHOT},
    {"aiv_allreduce_mesh1d_twoshot_int16_t", HcclCMDType::HCCL_CMD_ALLREDUCE, DataType::INT16,
     KernelArgsType::ARGS_TYPE_TWO_SHOT},
    {"aiv_allreduce_mesh1d_twoshot_float", HcclCMDType::HCCL_CMD_ALLREDUCE, DataType::FP32,
     KernelArgsType::ARGS_TYPE_TWO_SHOT},
    {"aiv_allreduce_mesh1d_twoshot_int32_t", HcclCMDType::HCCL_CMD_ALLREDUCE, DataType::INT32,
     KernelArgsType::ARGS_TYPE_TWO_SHOT},
    {"aiv_allreduce_mesh1d_twoshot_int8_t", HcclCMDType::HCCL_CMD_ALLREDUCE, DataType::INT8,
     KernelArgsType::ARGS_TYPE_TWO_SHOT},
    {"aiv_allreduce_mesh1d_twoshot_bfloat16_t", HcclCMDType::HCCL_CMD_ALLREDUCE, DataType::BFP16,
     KernelArgsType::ARGS_TYPE_TWO_SHOT},
    {"aiv_allreduce_mesh1d_twoshot_int64_t", HcclCMDType::HCCL_CMD_ALLREDUCE, DataType::INT64,
     KernelArgsType::ARGS_TYPE_TWO_SHOT},
    // alltoall
    {"aiv_alltoall_half", HcclCMDType::HCCL_CMD_ALLTOALL, DataType::FP16},
    {"aiv_alltoall_int16_t", HcclCMDType::HCCL_CMD_ALLTOALL, DataType::INT16},
    {"aiv_alltoall_uint16_t", HcclCMDType::HCCL_CMD_ALLTOALL, DataType::UINT16},
    {"aiv_alltoall_float", HcclCMDType::HCCL_CMD_ALLTOALL, DataType::FP32},
    {"aiv_alltoall_uint64_t", HcclCMDType::HCCL_CMD_ALLTOALL, DataType::FP64},
    {"aiv_alltoall_int32_t", HcclCMDType::HCCL_CMD_ALLTOALL, DataType::INT32},
    {"aiv_alltoall_uint32_t", HcclCMDType::HCCL_CMD_ALLTOALL, DataType::UINT32},
    {"aiv_alltoall_int8_t", HcclCMDType::HCCL_CMD_ALLTOALL, DataType::INT8},
    {"aiv_alltoall_uint8_t", HcclCMDType::HCCL_CMD_ALLTOALL, DataType::UINT8},
    {"aiv_alltoall_bfloat16_t", HcclCMDType::HCCL_CMD_ALLTOALL, DataType::BFP16},
    {"aiv_alltoall_uint64_t", HcclCMDType::HCCL_CMD_ALLTOALL, DataType::INT64},
    {"aiv_alltoall_int64_t", HcclCMDType::HCCL_CMD_ALLTOALL, DataType::UINT64},
    {"aiv_alltoall_hifloat8_t", HcclCMDType::HCCL_CMD_ALLTOALL, DataType::HIF8},
    {"aiv_alltoall_fp8_e4m3fn_t", HcclCMDType::HCCL_CMD_ALLTOALL, DataType::FP8E4M3},
    {"aiv_alltoall_fp8_e5m2_t", HcclCMDType::HCCL_CMD_ALLTOALL, DataType::FP8E5M2},
    {"aiv_alltoall_fp8_e8m0_t", HcclCMDType::HCCL_CMD_ALLTOALL, DataType::FP8E8M0},
    // alltoallv
    {"aiv_alltoallv_half", HcclCMDType::HCCL_CMD_ALLTOALLV, DataType::FP16},
    {"aiv_alltoallv_int16_t", HcclCMDType::HCCL_CMD_ALLTOALLV, DataType::INT16},
    {"aiv_alltoallv_uint16_t", HcclCMDType::HCCL_CMD_ALLTOALLV, DataType::UINT16},
    {"aiv_alltoallv_float", HcclCMDType::HCCL_CMD_ALLTOALLV, DataType::FP32},
    {"aiv_alltoallv_uint64_t", HcclCMDType::HCCL_CMD_ALLTOALLV, DataType::FP64},
    {"aiv_alltoallv_int32_t", HcclCMDType::HCCL_CMD_ALLTOALLV, DataType::INT32},
    {"aiv_alltoallv_uint32_t", HcclCMDType::HCCL_CMD_ALLTOALLV, DataType::UINT32},
    {"aiv_alltoallv_int8_t", HcclCMDType::HCCL_CMD_ALLTOALLV, DataType::INT8},
    {"aiv_alltoallv_uint8_t", HcclCMDType::HCCL_CMD_ALLTOALLV, DataType::UINT8},
    {"aiv_alltoallv_bfloat16_t", HcclCMDType::HCCL_CMD_ALLTOALLV, DataType::BFP16},
    {"aiv_alltoallv_uint64_t", HcclCMDType::HCCL_CMD_ALLTOALLV, DataType::INT64},
    {"aiv_alltoallv_int64_t", HcclCMDType::HCCL_CMD_ALLTOALLV, DataType::UINT64},
    {"aiv_alltoallv_hifloat8_t", HcclCMDType::HCCL_CMD_ALLTOALLV, DataType::HIF8},
    {"aiv_alltoallv_fp8_e4m3fn_t", HcclCMDType::HCCL_CMD_ALLTOALLV, DataType::FP8E4M3},
    {"aiv_alltoallv_fp8_e5m2_t", HcclCMDType::HCCL_CMD_ALLTOALLV, DataType::FP8E5M2},
    {"aiv_alltoallv_fp8_e8m0_t", HcclCMDType::HCCL_CMD_ALLTOALLV, DataType::FP8E8M0},
    // reduce
    {"aiv_reduce_half", HcclCMDType::HCCL_CMD_REDUCE, DataType::FP16},
    {"aiv_reduce_int16_t", HcclCMDType::HCCL_CMD_REDUCE, DataType::INT16},
    {"aiv_reduce_float", HcclCMDType::HCCL_CMD_REDUCE, DataType::FP32},
    {"aiv_reduce_int32_t", HcclCMDType::HCCL_CMD_REDUCE, DataType::INT32},
    {"aiv_reduce_int8_t", HcclCMDType::HCCL_CMD_REDUCE, DataType::INT8},
    {"aiv_reduce_bfloat16_t", HcclCMDType::HCCL_CMD_REDUCE, DataType::BFP16},
    {"aiv_reduce_int64_t", HcclCMDType::HCCL_CMD_REDUCE, DataType::INT64},
    // reducescatter
    {"aiv_reduce_scatter_half", HcclCMDType::HCCL_CMD_REDUCE_SCATTER, DataType::FP16},
    {"aiv_reduce_scatter_int16_t", HcclCMDType::HCCL_CMD_REDUCE_SCATTER, DataType::INT16},
    {"aiv_reduce_scatter_float", HcclCMDType::HCCL_CMD_REDUCE_SCATTER, DataType::FP32},
    {"aiv_reduce_scatter_int32_t", HcclCMDType::HCCL_CMD_REDUCE_SCATTER, DataType::INT32},
    {"aiv_reduce_scatter_int8_t", HcclCMDType::HCCL_CMD_REDUCE_SCATTER, DataType::INT8},
    {"aiv_reduce_scatter_bfloat16_t", HcclCMDType::HCCL_CMD_REDUCE_SCATTER, DataType::BFP16},
    {"aiv_reduce_scatter_int64_t", HcclCMDType::HCCL_CMD_REDUCE_SCATTER, DataType::INT64},
    // send
    {"aiv_send_int8_t", HcclCMDType::HCCL_CMD_SEND, DataType::INT8}, // hccl_types.h
    {"aiv_send_int16_t", HcclCMDType::HCCL_CMD_SEND, DataType::INT16},
    {"aiv_send_int32_t", HcclCMDType::HCCL_CMD_SEND, DataType::INT32},
    {"aiv_send_int64_t", HcclCMDType::HCCL_CMD_SEND, DataType::INT64},
    {"aiv_send_uint8_t", HcclCMDType::HCCL_CMD_SEND, DataType::UINT8},
    {"aiv_send_uint16_t", HcclCMDType::HCCL_CMD_SEND, DataType::UINT16},
    {"aiv_send_uint32_t", HcclCMDType::HCCL_CMD_SEND, DataType::UINT32},
    {"aiv_send_uint64_t", HcclCMDType::HCCL_CMD_SEND, DataType::UINT64},
    {"aiv_send_half", HcclCMDType::HCCL_CMD_SEND, DataType::FP16},
    {"aiv_send_float", HcclCMDType::HCCL_CMD_SEND, DataType::FP32},
    {"aiv_send_uint64_t", HcclCMDType::HCCL_CMD_SEND, DataType::FP64},
    {"aiv_send_bfloat16_t", HcclCMDType::HCCL_CMD_SEND, DataType::BFP16},
    {"aiv_send_hifloat8_t", HcclCMDType::HCCL_CMD_SEND, DataType::HIF8},
    {"aiv_send_fp8_e4m3fn_t", HcclCMDType::HCCL_CMD_SEND, DataType::FP8E4M3},
    {"aiv_send_fp8_e5m2_t", HcclCMDType::HCCL_CMD_SEND, DataType::FP8E5M2},
    {"aiv_send_fp8_e8m0_t", HcclCMDType::HCCL_CMD_SEND, DataType::FP8E8M0},
    // recv
    {"aiv_recv_int8_t", HcclCMDType::HCCL_CMD_RECEIVE, DataType::INT8},
    {"aiv_recv_int16_t", HcclCMDType::HCCL_CMD_RECEIVE, DataType::INT16},
    {"aiv_recv_int32_t", HcclCMDType::HCCL_CMD_RECEIVE, DataType::INT32},
    {"aiv_recv_int64_t", HcclCMDType::HCCL_CMD_RECEIVE, DataType::INT64},
    {"aiv_recv_uint8_t", HcclCMDType::HCCL_CMD_RECEIVE, DataType::UINT8},
    {"aiv_recv_uint16_t", HcclCMDType::HCCL_CMD_RECEIVE, DataType::UINT16},
    {"aiv_recv_uint32_t", HcclCMDType::HCCL_CMD_RECEIVE, DataType::UINT32},
    {"aiv_recv_uint64_t", HcclCMDType::HCCL_CMD_RECEIVE, DataType::UINT64},
    {"aiv_recv_half", HcclCMDType::HCCL_CMD_RECEIVE, DataType::FP16},
    {"aiv_recv_float", HcclCMDType::HCCL_CMD_RECEIVE, DataType::FP32},
    {"aiv_recv_uint64_t", HcclCMDType::HCCL_CMD_RECEIVE, DataType::FP64},
    {"aiv_recv_bfloat16_t", HcclCMDType::HCCL_CMD_RECEIVE, DataType::BFP16},
    {"aiv_recv_hifloat8_t", HcclCMDType::HCCL_CMD_RECEIVE, DataType::HIF8},
    {"aiv_recv_fp8_e4m3fn_t", HcclCMDType::HCCL_CMD_RECEIVE, DataType::FP8E4M3},
    {"aiv_recv_fp8_e5m2_t", HcclCMDType::HCCL_CMD_RECEIVE, DataType::FP8E5M2},
    {"aiv_recv_fp8_e8m0_t", HcclCMDType::HCCL_CMD_RECEIVE, DataType::FP8E8M0},
    // batchSendRecv
    {"aiv_batchSendRecv_int8_t", HcclCMDType::HCCL_CMD_BATCH_SEND_RECV, DataType::INT8},
    {"aiv_batchSendRecv_int16_t", HcclCMDType::HCCL_CMD_BATCH_SEND_RECV, DataType::INT16},
    {"aiv_batchSendRecv_int32_t", HcclCMDType::HCCL_CMD_BATCH_SEND_RECV, DataType::INT32},
    {"aiv_batchSendRecv_int64_t", HcclCMDType::HCCL_CMD_BATCH_SEND_RECV, DataType::INT64},
    {"aiv_batchSendRecv_uint8_t", HcclCMDType::HCCL_CMD_BATCH_SEND_RECV, DataType::UINT8},
    {"aiv_batchSendRecv_uint16_t", HcclCMDType::HCCL_CMD_BATCH_SEND_RECV, DataType::UINT16},
    {"aiv_batchSendRecv_uint32_t", HcclCMDType::HCCL_CMD_BATCH_SEND_RECV, DataType::UINT32},
    {"aiv_batchSendRecv_uint64_t", HcclCMDType::HCCL_CMD_BATCH_SEND_RECV, DataType::UINT64},
    {"aiv_batchSendRecv_half", HcclCMDType::HCCL_CMD_BATCH_SEND_RECV, DataType::FP16},
    {"aiv_batchSendRecv_float", HcclCMDType::HCCL_CMD_BATCH_SEND_RECV, DataType::FP32},
    {"aiv_batchSendRecv_uint64_t", HcclCMDType::HCCL_CMD_BATCH_SEND_RECV, DataType::FP64},
    {"aiv_batchSendRecv_bfloat16_t", HcclCMDType::HCCL_CMD_BATCH_SEND_RECV, DataType::BFP16},
    {"aiv_batchSendRecv_hifloat8_t", HcclCMDType::HCCL_CMD_BATCH_SEND_RECV, DataType::HIF8},
    {"aiv_batchSendRecv_fp8_e4m3fn_t", HcclCMDType::HCCL_CMD_BATCH_SEND_RECV, DataType::FP8E4M3},
    {"aiv_batchSendRecv_fp8_e5m2_t", HcclCMDType::HCCL_CMD_BATCH_SEND_RECV, DataType::FP8E5M2},
    {"aiv_batchSendRecv_fp8_e8m0_t", HcclCMDType::HCCL_CMD_BATCH_SEND_RECV, DataType::FP8E8M0},
};
using AivExtraKernelArgs = struct AivExtraKernelArgsDef {
    const void* buffersIn; // 注册的CCLIN地址，所有卡可访问
    u64 input;
    u64 output;
    u32 rank;
    u32 sendRecvRemoteRank;
    u32 rankSize;
    u64 xRankSize;
    u64 yRankSize;
    u64 zRankSize;
    u64 len;
    u32 dataType;
    u32 reduceOp;
    u32 root;
    u32 tag; // 第几次调用，定时重置成1
    u64 inputSliceStride;
    u64 outputSliceStride;
    u64 repeatNum;
    u64 inputRepeatStride;
    u64 outputRepeatStride;
    bool isOpBase;
    const void* headCountMem;
    const void* tailCountMem;
    const void* addOneMem;
    u32 counterMemSize;
    bool isEnableCounter;
    ExtraArgsA2A extraArgs;

    AivExtraKernelArgsDef(
        const void* buffIn, u64 input, u64 output, u32 rank, u32 sendRecvRemoteRank, u32 rankSize, u64 xRankSize,
        u64 yRankSize, u64 zRankSize, u64 len, u32 dataType, u32 reduceOp, u32 root, u32 tag, u64 inputSliceStride,
        u64 outputSliceStride, u64 repeatNum, u64 inputRepeatStride, u64 outputRepeatStride, bool isOpBase = true,
        const void* headCountMem = nullptr, const void* tailCountMem = nullptr, const void* addOneMem = nullptr,
        u32 counterMemSize = 0, const ExtraArgsA2A* extraArgsPtr = nullptr)
        : buffersIn(buffIn),
          input(input),
          output(output),
          rank(rank),
          sendRecvRemoteRank(sendRecvRemoteRank),
          rankSize(rankSize),
          xRankSize(xRankSize),
          yRankSize(yRankSize),
          zRankSize(zRankSize),
          len(len),
          dataType(dataType),
          reduceOp(reduceOp),
          root(root),
          tag(tag),
          inputSliceStride(inputSliceStride),
          outputSliceStride(outputSliceStride),
          repeatNum(repeatNum),
          inputRepeatStride(inputRepeatStride),
          outputRepeatStride(outputRepeatStride),
          isOpBase(isOpBase),
          headCountMem(headCountMem),
          tailCountMem(tailCountMem),
          addOneMem(addOneMem),
          counterMemSize(counterMemSize)
    {
        if (extraArgsPtr != nullptr) {
            extraArgs = *extraArgsPtr;
        }
    }
};

HcclResult GetAivOpBinaryPath(std::string& binaryPath)
{
    char* envValue = nullptr;
    MM_SYS_GET_ENV(MM_ENV_ASCEND_HOME_PATH, envValue);

    std::string libPath;
    if (envValue != nullptr) {
        libPath = envValue;
    } else {
        libPath = "/usr/local/Ascend/cann";
        HCCL_WARNING("[AIV][GetAivOpBinaryPath]ENV:ASCEND_HOME_PATH is not set, use default path[%s]", libPath.c_str());
    }

    binaryPath = libPath + "/lib64/hccl_aiv_op_910_95.o";

    HCCL_INFO("[AIV][GetAivOpBinaryPath]binaryPath: %s", binaryPath.c_str());

    return HCCL_SUCCESS;
}

s8* GetStubFunc(HcclCMDType cmdType, DataType dataType, KernelArgsType argsType = KernelArgsType::ARGS_TYPE_SERVER)
{
    return reinterpret_cast<s8*>(
        (((static_cast<u64>(cmdType) << SIG_MOVE_LEFT_BITS) + static_cast<u64>(dataType)) << SIG_MOVE_LEFT_BITS)
        + static_cast<u64>(argsType));
}

HcclResult RegisterBinaryKernel(const char* funcName, const aclrtBinHandle binHandle, const s8* stubFunc)
{
    if (stubFunc == nullptr) {
        return HCCL_E_PARA;
    }

    aclrtFuncHandle funcHandle;
    aclError aclRet = aclrtBinaryGetFunction(binHandle, funcName, &funcHandle);
    CHK_PRT_RET(
        aclRet != ACL_SUCCESS,
        HCCL_ERROR("[RegisterBinaryKernel]errNo[0x%016llx] get function from binary error.", aclRet), HCCL_E_NOT_FOUND);

    g_aivFuncMap[stubFunc] = funcHandle;
    g_aivNameMap[stubFunc] = funcName;

    return HCCL_SUCCESS;
}

HcclResult RegisterKernel()
{
    lock_guard<mutex> guard(g_mut);
    if (g_init) {
        return HCCL_SUCCESS;
    }

    HcclResult ret;
    string binFilePath;
    ret = GetAivOpBinaryPath(binFilePath);
    HCCL_INFO("[RegisterKernel] binFilePath: %s", binFilePath.c_str());
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[AIV][RegisterKernel] get aiv op binary path failed"), HCCL_E_RUNTIME);

    LoadBinaryFromFile(binFilePath.c_str(), ACL_RT_BINARY_LOAD_OPT_LAZY_LOAD, 1, g_binHandle);
    for (auto& aivKernelInfo : g_aivKernelInfoList) {
        ret = RegisterBinaryKernel(
            aivKernelInfo.kernelName, g_binHandle,
            GetStubFunc(aivKernelInfo.cmdType, aivKernelInfo.dataType, aivKernelInfo.argsType));

        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[AIV][RegisterKernel] register binary kernel for kernelName[%s] "
                "cmdType[%d] dataType[%d] argsType[%d] failed",
                aivKernelInfo.kernelName, aivKernelInfo.cmdType, aivKernelInfo.dataType, aivKernelInfo.argsType),
            HCCL_E_RUNTIME);
    }

    g_init = true;

    return HCCL_SUCCESS;
}

HcclResult UnRegisterAivKernel()
{
    lock_guard<mutex> guard(g_mut);
    if (g_init) {
        aclError aclRet = aclrtBinaryUnLoad(g_binHandle);
        CHK_PRT_RET(
            aclRet != ACL_SUCCESS, HCCL_ERROR("[UnRegisterAivKernel] aclrtBinaryUnLoad failed, ret[%d]", aclRet),
            HCCL_E_RUNTIME);
        g_aivFuncMap.clear();
        g_init = false;
    }
    return HCCL_SUCCESS;
}

HcclResult GetMinAndMaxNpuSchedTimeOut(u64& minNpuSchedTimeout, u64& maxNpuSchedTimeout)
{
    uint64_t interval = 0;
    aclError aclRet = aclrtGetOpTimeOutInterval(&interval);
    CHK_PRT_RET(
        aclRet != ACL_SUCCESS, HCCL_ERROR("aclrtGetOpTimeOutInterval get timeout interval failed, ret[%d]", aclRet),
        HCCL_E_RUNTIME);

    // NPU超时范围(1, 254) * interval
    minNpuSchedTimeout = MIN_NPU_TIMEOUT * interval;
    maxNpuSchedTimeout = MAX_NPU_TIMEOUT * interval;
    HCCL_INFO(
        "GetMinAndMaxNpuSchedTimeOut minNpuSchedTimeout[%u]us, maxNpuSchedTimeout[%u]us.", minNpuSchedTimeout,
        maxNpuSchedTimeout);
    return HCCL_SUCCESS;
}

u32 GetAivTimeout()
{
    constexpr u32 TIME_S_TO_US = 1000000;
    constexpr u32 AIV_TIMEOUT_DEFAULT_US = 1091 * TIME_S_TO_US;
    constexpr u32 AIV_TIMEOUT_MAX_US = 1091 * TIME_S_TO_US;
    u32 timeout = AIV_TIMEOUT_DEFAULT_US;
    double execTimeOut = EnvConfig::GetInstance().GetRtsConfig().GetAivExecTimeOut();

    double timeoutUs = execTimeOut * TIME_S_TO_US;
    if (timeoutUs > static_cast<double>(std::numeric_limits<u32>::max())) {
        HCCL_INFO("[GetAivTimeout]Get input timeout[%.2f us] is out of valid range.", timeoutUs);
        return AIV_TIMEOUT_MAX_US;
    }
    u32 timeoutUsInt = static_cast<u32>(timeoutUs);
    if (timeoutUsInt == 0) {
        timeoutUsInt = AIV_TIMEOUT_MAX_US;
    }
    u64 minNpuSchedTimeout = 0;
    u64 maxNpuSchedTimeout = 0;
    CHK_RET(GetMinAndMaxNpuSchedTimeOut(minNpuSchedTimeout, maxNpuSchedTimeout));
    timeout = (timeoutUsInt < minNpuSchedTimeout) ? minNpuSchedTimeout :
              (timeoutUsInt > maxNpuSchedTimeout) ? maxNpuSchedTimeout :
                                                    timeoutUsInt;
    HCCL_INFO(
        "[GetAivTimeout]timeout[%u]us, execTimeOut[%.2f]s, minNpuSchedTimeout[%u]us, maxNpuSchedTimeout[%u]us.",
        timeout, execTimeOut, minNpuSchedTimeout, maxNpuSchedTimeout);

    return timeout;
}

HcclResult GetKernelFunc(aclrtFuncHandle& funcHandle, const s8* stubFunc)
{
    if (stubFunc == nullptr || g_aivFuncMap.find(stubFunc) == g_aivFuncMap.end()) {
        HCCL_ERROR("[GetKernelFunc] stubFunc not found in g_aivFuncMap");
        return HCCL_E_PARA;
    }
    funcHandle = g_aivFuncMap[stubFunc];
    return HCCL_SUCCESS;
}

// KernelLaunch内部接口
HcclResult ExecuteKernelLaunchInner(const AivOpArgs& opArgs, void* args, u32 argsSize)
{
    constexpr u32 AIV_ATTRNUM_THREE = 3;
    HCCL_INFO(
        "[AIV][ExecuteKernelLaunch] sendbuff [%llu] recvbuff [%llu] rank [%u] sendRecvRemoteRank [%u] rankSize [%u] "
        "count [%llu] "
        "dataType [%d] reduceOp [%d] root [%u] tag [%u] isOpBase [%d] "
        "extraArgsPtr [%p] argsSize [%u] numBlocks [%u]",
        opArgs.input, opArgs.output, opArgs.rank, opArgs.sendRecvRemoteRank, opArgs.rankSize, opArgs.count,
        opArgs.dataType, opArgs.op, opArgs.root, opArgs.aivTag, opArgs.isOpBase, args, argsSize, opArgs.numBlocks);

    aclrtLaunchKernelCfg cfg;
    aclrtLaunchKernelAttr attr[AIV_ATTRNUM_THREE];

    u32 timeoutUs = GetAivTimeout();
    attr[0].id = ACL_RT_LAUNCH_KERNEL_ATTR_SCHEM_MODE;
    attr[0].value.schemMode = 1;
    attr[1].id = ACL_RT_LAUNCH_KERNEL_ATTR_TIMEOUT_US;
    attr[1].value.timeoutUs.timeoutLow = timeoutUs;
    attr[1].value.timeoutUs.timeoutHigh = 0;
    attr[2].id = ACL_RT_LAUNCH_KERNEL_ATTR_ENGINE_TYPE;
    attr[2].value.engineType = ACL_RT_ENGINE_TYPE_AIV;
    cfg.numAttrs = AIV_ATTRNUM_THREE;
    cfg.attrs = attr;

    HCCL_INFO(
        "[AIV][ExecuteKernelLaunch] KernelAttr attr[0]: id=[%u], schemMode=[%u]; attr[1]: id=[%u], "
        "timeoutLow=[%u us], timeoutHigh=[%u us]; attr[2]: id=[%u], engineType=[%u]; cfg: numAttrs=[%u]",
        attr[0].id, attr[0].value.schemMode, attr[1].id, attr[1].value.timeoutUs.timeoutLow,
        attr[1].value.timeoutUs.timeoutHigh, attr[2].id, attr[2].value.engineType, cfg.numAttrs);

    aclrtFuncHandle funcHandle;
    const s8* stubFunc = GetStubFunc(opArgs.cmdType, opArgs.dataType, opArgs.argsType);
    HcclResult ret = GetKernelFunc(funcHandle, stubFunc);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[ExecuteKernelLaunchInner] errNo[0x%016llx] GetKernelFunc failed, "
            "return[%d]",
            HCCL_ERROR_CODE(HCCL_E_RUNTIME), ret),
        HCCL_E_RUNTIME);

    aclError aclRet
        = aclrtLaunchKernelWithHostArgs(funcHandle, opArgs.numBlocks, opArgs.stream, &cfg, args, argsSize, nullptr, 0);
    if (aclRet == ACL_ERROR_RT_INVALID_HANDLE) {
        aclError aclGetRet = aclrtBinaryGetFunction(g_binHandle, g_aivNameMap[stubFunc].c_str(), &funcHandle);
        CHK_PRT_RET(
            aclGetRet != ACL_SUCCESS,
            HCCL_ERROR("[RegisterBinaryKernel]errNo[0x%016llx] get function from binary error.", aclRet),
            HCCL_E_NOT_FOUND);
        aclRet = aclrtLaunchKernelWithHostArgs(
            funcHandle, opArgs.numBlocks, opArgs.stream, &cfg, args, argsSize, nullptr, 0);
    }
    CHK_PRT_RET(
        aclRet != ACL_SUCCESS,
        HCCL_ERROR(
            "[ExecuteKernelLaunchInner]errNo[0x%016llx] aclrtLaunchKernelWithHostArgs error[%d].",
            HCCL_ERROR_CODE(HCCL_E_RUNTIME), aclRet),
        HCCL_E_RUNTIME);
    return HCCL_SUCCESS;
}

// Kernel单次调用Launch外部接口
HcclResult ExecuteKernelLaunch(const AivOpArgs& opArgs)
{
    AivExtraKernelArgs aivExtraKernelArgs{
        opArgs.buffersIn,
        opArgs.input,
        opArgs.output,
        opArgs.rank,
        opArgs.sendRecvRemoteRank,
        opArgs.rankSize,
        opArgs.xRankSize,
        opArgs.yRankSize,
        opArgs.zRankSize,
        opArgs.count,
        opArgs.dataType,
        opArgs.op,
        opArgs.root,
        opArgs.aivTag,
        opArgs.inputSliceStride,
        opArgs.outputSliceStride,
        opArgs.repeatNum,
        opArgs.inputRepeatStride,
        opArgs.outputRepeatStride,
        opArgs.isOpBase,
        reinterpret_cast<void*>(opArgs.counter.headCountMem),
        reinterpret_cast<void*>(opArgs.counter.tailCountMem),
        reinterpret_cast<void*>(opArgs.counter.addOneMem),
        opArgs.counter.memSize,
        &opArgs.extraArgs // 这里是真正发送的地方，device侧单个kernel不能超过32K，先写死1024个结构体
    };
    CHK_RET(ExecuteKernelLaunchInner(opArgs, &aivExtraKernelArgs, sizeof(aivExtraKernelArgs)));

    return HCCL_SUCCESS;
}

} // namespace Hccl
