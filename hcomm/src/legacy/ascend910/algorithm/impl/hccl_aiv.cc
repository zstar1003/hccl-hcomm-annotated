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
#include <unordered_map>
#include <cmath>
#include <limits>
#include "mmpa_api.h"
#include "adapter_rts_common.h"
#include "acl/acl_rt.h"
#include "launch_device.h"
#include "hccl_aiv.h"

using namespace std;

namespace hccl {
constexpr u32 SIG_MOVE_LEFT_BITS = 20;
constexpr u32 RANK_ZERO = 0;
constexpr u32 RANK_ONE = 1;
constexpr u32 RANK_TWO = 2;
constexpr u32 RANK_THREE = 3;
constexpr u32 RANK_FOUR = 4;
constexpr u32 RANK_FIVE = 5;
constexpr u32 RANK_SIX = 6;
constexpr u32 RANK_SEVEN = 7;
constexpr u32 MAX_ARGS_SIZE_A3_STRUCT = 9;

constexpr u32 AIV_BUFFER_PING_PONG_FACTOR = 2;

constexpr u32 MAX_BIN_FILE_SIZE = 100 * 1024 * 1024; // 最大读取100m的bin file到string中

constexpr s32 RESET_TAIL_SYNC_TAG = 2;
constexpr u32 AIV_FLAG_AREA_SIZE = 1024 * 1024;

constexpr u32 AIV_ATTRNUM_THREE = 3;

using AivKernelInfo = struct AivKernelInfoDef {
    const char* kernelName;
    HcclDataType dataType;
    KernelArgsType argsType;

    AivKernelInfoDef(
        const char* kernelName, HcclDataType dataType, KernelArgsType argsType = KernelArgsType::ARGS_TYPE_SERVER)
        : kernelName(kernelName),
          dataType(dataType),
          argsType(argsType)
    {}
};

static bool g_init = false;
static mutex g_mut;
static std::unordered_map<HcclCMDType, aclrtBinHandle> g_binHandleMap;
static std::unordered_map<s8*, aclrtFuncHandle> g_aivFuncMap;
static std::unordered_map<s8*, std::string> g_aivNameMap;

static std::vector<AivKernelInfo> g_allreduceAivKernelInfoList = {
    {"aiv_all_reduce_float", HcclDataType::HCCL_DATA_TYPE_FP32},
    {"aiv_all_reduce_half", HcclDataType::HCCL_DATA_TYPE_FP16},
    {"aiv_all_reduce_int16_t", HcclDataType::HCCL_DATA_TYPE_INT16},
    {"aiv_all_reduce_int32_t", HcclDataType::HCCL_DATA_TYPE_INT32},
    {"aiv_all_reduce_int8_t", HcclDataType::HCCL_DATA_TYPE_INT8},
    {"aiv_all_reduce_bfloat16_t", HcclDataType::HCCL_DATA_TYPE_BFP16},
    {"aiv_all_reduce_cn_float", HcclDataType::HCCL_DATA_TYPE_FP32, KernelArgsType::ARGS_TYPE_SIMPLE},
    {"aiv_all_reduce_cn_half", HcclDataType::HCCL_DATA_TYPE_FP16, KernelArgsType::ARGS_TYPE_SIMPLE},
    {"aiv_all_reduce_cn_int16_t", HcclDataType::HCCL_DATA_TYPE_INT16, KernelArgsType::ARGS_TYPE_SIMPLE},
    {"aiv_all_reduce_cn_int32_t", HcclDataType::HCCL_DATA_TYPE_INT32, KernelArgsType::ARGS_TYPE_SIMPLE},
    {"aiv_all_reduce_cn_int8_t", HcclDataType::HCCL_DATA_TYPE_INT8, KernelArgsType::ARGS_TYPE_SIMPLE},
    {"aiv_all_reduce_cn_bfloat16_t", HcclDataType::HCCL_DATA_TYPE_BFP16, KernelArgsType::ARGS_TYPE_SIMPLE},
};
static std::string g_allreduceAivBinaryName = "hccl_aiv_all_reduce_op.o";

static std::vector<AivKernelInfo> g_reducescatterAivKernelInfoList = {
    {"aiv_reduce_scatter_float", HcclDataType::HCCL_DATA_TYPE_FP32},
    {"aiv_reduce_scatter_half", HcclDataType::HCCL_DATA_TYPE_FP16},
    {"aiv_reduce_scatter_int16_t", HcclDataType::HCCL_DATA_TYPE_INT16},
    {"aiv_reduce_scatter_int32_t", HcclDataType::HCCL_DATA_TYPE_INT32},
    {"aiv_reduce_scatter_int8_t", HcclDataType::HCCL_DATA_TYPE_INT8},
    {"aiv_reduce_scatter_bfloat16_t", HcclDataType::HCCL_DATA_TYPE_BFP16},
    {"aiv_reduce_scatter_cn_float", HcclDataType::HCCL_DATA_TYPE_FP32, KernelArgsType::ARGS_TYPE_SIMPLE},
    {"aiv_reduce_scatter_cn_half", HcclDataType::HCCL_DATA_TYPE_FP16, KernelArgsType::ARGS_TYPE_SIMPLE},
    {"aiv_reduce_scatter_cn_int16_t", HcclDataType::HCCL_DATA_TYPE_INT16, KernelArgsType::ARGS_TYPE_SIMPLE},
    {"aiv_reduce_scatter_cn_int32_t", HcclDataType::HCCL_DATA_TYPE_INT32, KernelArgsType::ARGS_TYPE_SIMPLE},
    {"aiv_reduce_scatter_cn_int8_t", HcclDataType::HCCL_DATA_TYPE_INT8, KernelArgsType::ARGS_TYPE_SIMPLE},
    {"aiv_reduce_scatter_cn_bfloat16_t", HcclDataType::HCCL_DATA_TYPE_BFP16, KernelArgsType::ARGS_TYPE_SIMPLE},
};
static std::string g_reducescatterAivBinaryName = "hccl_aiv_reduce_scatter_op.o";

static std::vector<AivKernelInfo> g_allgatherAivKernelInfoList = {
    {"aiv_all_gather_half", HcclDataType::HCCL_DATA_TYPE_FP16},
    {"aiv_all_gather_int16_t", HcclDataType::HCCL_DATA_TYPE_INT16},
    {"aiv_all_gather_uint16_t", HcclDataType::HCCL_DATA_TYPE_UINT16},
    {"aiv_all_gather_float", HcclDataType::HCCL_DATA_TYPE_FP32},
    {"aiv_all_gather_int32_t", HcclDataType::HCCL_DATA_TYPE_INT32},
    {"aiv_all_gather_uint32_t", HcclDataType::HCCL_DATA_TYPE_UINT32},
    {"aiv_all_gather_int8_t", HcclDataType::HCCL_DATA_TYPE_INT8},
    {"aiv_all_gather_uint8_t", HcclDataType::HCCL_DATA_TYPE_UINT8},
    {"aiv_all_gather_bfloat16_t", HcclDataType::HCCL_DATA_TYPE_BFP16},
    {"aiv_all_gather_int64_t", HcclDataType::HCCL_DATA_TYPE_INT64},
    {"aiv_all_gather_uint64_t", HcclDataType::HCCL_DATA_TYPE_UINT64},
    {"aiv_all_gather_double", HcclDataType::HCCL_DATA_TYPE_FP64},
    {"aiv_all_gather_cn_half", HcclDataType::HCCL_DATA_TYPE_FP16, KernelArgsType::ARGS_TYPE_SIMPLE},
    {"aiv_all_gather_cn_int16_t", HcclDataType::HCCL_DATA_TYPE_INT16, KernelArgsType::ARGS_TYPE_SIMPLE},
    {"aiv_all_gather_cn_uint16_t", HcclDataType::HCCL_DATA_TYPE_UINT16, KernelArgsType::ARGS_TYPE_SIMPLE},
    {"aiv_all_gather_cn_float", HcclDataType::HCCL_DATA_TYPE_FP32, KernelArgsType::ARGS_TYPE_SIMPLE},
    {"aiv_all_gather_cn_int32_t", HcclDataType::HCCL_DATA_TYPE_INT32, KernelArgsType::ARGS_TYPE_SIMPLE},
    {"aiv_all_gather_cn_uint32_t", HcclDataType::HCCL_DATA_TYPE_UINT32, KernelArgsType::ARGS_TYPE_SIMPLE},
    {"aiv_all_gather_cn_int8_t", HcclDataType::HCCL_DATA_TYPE_INT8, KernelArgsType::ARGS_TYPE_SIMPLE},
    {"aiv_all_gather_cn_uint8_t", HcclDataType::HCCL_DATA_TYPE_UINT8, KernelArgsType::ARGS_TYPE_SIMPLE},
    {"aiv_all_gather_cn_bfloat16_t", HcclDataType::HCCL_DATA_TYPE_BFP16, KernelArgsType::ARGS_TYPE_SIMPLE},
    {"aiv_all_gather_cn_int64_t", HcclDataType::HCCL_DATA_TYPE_INT64, KernelArgsType::ARGS_TYPE_SIMPLE},
    {"aiv_all_gather_cn_uint64_t", HcclDataType::HCCL_DATA_TYPE_UINT64, KernelArgsType::ARGS_TYPE_SIMPLE},
    {"aiv_all_gather_cn_double", HcclDataType::HCCL_DATA_TYPE_FP64, KernelArgsType::ARGS_TYPE_SIMPLE},
};
static std::string g_allgatherAivBinaryName = "hccl_aiv_all_gather_op.o";

static std::vector<AivKernelInfo> g_broadcastAivKernelInfoList = {
    {"aiv_broadcast_half", HcclDataType::HCCL_DATA_TYPE_FP16},
    {"aiv_broadcast_int16_t", HcclDataType::HCCL_DATA_TYPE_INT16},
    {"aiv_broadcast_uint16_t", HcclDataType::HCCL_DATA_TYPE_UINT16},
    {"aiv_broadcast_float", HcclDataType::HCCL_DATA_TYPE_FP32},
    {"aiv_broadcast_int32_t", HcclDataType::HCCL_DATA_TYPE_INT32},
    {"aiv_broadcast_uint32_t", HcclDataType::HCCL_DATA_TYPE_UINT32},
    {"aiv_broadcast_int8_t", HcclDataType::HCCL_DATA_TYPE_INT8},
    {"aiv_broadcast_uint8_t", HcclDataType::HCCL_DATA_TYPE_UINT8},
    {"aiv_broadcast_bfloat16_t", HcclDataType::HCCL_DATA_TYPE_BFP16},
    {"aiv_broadcast_int64_t", HcclDataType::HCCL_DATA_TYPE_INT64},
    {"aiv_broadcast_uint64_t", HcclDataType::HCCL_DATA_TYPE_UINT64},
    {"aiv_broadcast_double", HcclDataType::HCCL_DATA_TYPE_FP64},
    {"aiv_broadcast_cn_half", HcclDataType::HCCL_DATA_TYPE_FP16, KernelArgsType::ARGS_TYPE_SIMPLE},
    {"aiv_broadcast_cn_int16_t", HcclDataType::HCCL_DATA_TYPE_INT16, KernelArgsType::ARGS_TYPE_SIMPLE},
    {"aiv_broadcast_cn_uint16_t", HcclDataType::HCCL_DATA_TYPE_UINT16, KernelArgsType::ARGS_TYPE_SIMPLE},
    {"aiv_broadcast_cn_float", HcclDataType::HCCL_DATA_TYPE_FP32, KernelArgsType::ARGS_TYPE_SIMPLE},
    {"aiv_broadcast_cn_int32_t", HcclDataType::HCCL_DATA_TYPE_INT32, KernelArgsType::ARGS_TYPE_SIMPLE},
    {"aiv_broadcast_cn_uint32_t", HcclDataType::HCCL_DATA_TYPE_UINT32, KernelArgsType::ARGS_TYPE_SIMPLE},
    {"aiv_broadcast_cn_int8_t", HcclDataType::HCCL_DATA_TYPE_INT8, KernelArgsType::ARGS_TYPE_SIMPLE},
    {"aiv_broadcast_cn_uint8_t", HcclDataType::HCCL_DATA_TYPE_UINT8, KernelArgsType::ARGS_TYPE_SIMPLE},
    {"aiv_broadcast_cn_bfloat16_t", HcclDataType::HCCL_DATA_TYPE_BFP16, KernelArgsType::ARGS_TYPE_SIMPLE},
    {"aiv_broadcast_cn_int64_t", HcclDataType::HCCL_DATA_TYPE_INT64, KernelArgsType::ARGS_TYPE_SIMPLE},
    {"aiv_broadcast_cn_uint64_t", HcclDataType::HCCL_DATA_TYPE_UINT64, KernelArgsType::ARGS_TYPE_SIMPLE},
    {"aiv_broadcast_cn_double", HcclDataType::HCCL_DATA_TYPE_FP64, KernelArgsType::ARGS_TYPE_SIMPLE},
};
static std::string g_broadcastAivBinaryName = "hccl_aiv_broadcast_op.o";

static std::vector<AivKernelInfo> g_alltoallAivKernelInfoList = {
    {"aiv_all_to_all_half", HcclDataType::HCCL_DATA_TYPE_FP16},
    {"aiv_all_to_all_int16_t", HcclDataType::HCCL_DATA_TYPE_INT16},
    {"aiv_all_to_all_uint16_t", HcclDataType::HCCL_DATA_TYPE_UINT16},
    {"aiv_all_to_all_float", HcclDataType::HCCL_DATA_TYPE_FP32},
    {"aiv_all_to_all_int32_t", HcclDataType::HCCL_DATA_TYPE_INT32},
    {"aiv_all_to_all_uint32_t", HcclDataType::HCCL_DATA_TYPE_UINT32},
    {"aiv_all_to_all_int8_t", HcclDataType::HCCL_DATA_TYPE_INT8},
    {"aiv_all_to_all_uint8_t", HcclDataType::HCCL_DATA_TYPE_UINT8},
    {"aiv_all_to_all_bfloat16_t", HcclDataType::HCCL_DATA_TYPE_BFP16},
    {"aiv_all_to_all_int64_t", HcclDataType::HCCL_DATA_TYPE_INT64},
    {"aiv_all_to_all_uint64_t", HcclDataType::HCCL_DATA_TYPE_UINT64},
    {"aiv_all_to_all_double", HcclDataType::HCCL_DATA_TYPE_FP64},
};
static std::string g_alltoallAivBinaryName = "hccl_aiv_all_to_all_op.o";

static std::vector<AivKernelInfo> g_alltoallvAivKernelInfoList = {
    {"aiv_all_to_all_v_half", HcclDataType::HCCL_DATA_TYPE_FP16},
    {"aiv_all_to_all_v_int16_t", HcclDataType::HCCL_DATA_TYPE_INT16},
    {"aiv_all_to_all_v_uint16_t", HcclDataType::HCCL_DATA_TYPE_UINT16},
    {"aiv_all_to_all_v_float", HcclDataType::HCCL_DATA_TYPE_FP32},
    {"aiv_all_to_all_v_int32_t", HcclDataType::HCCL_DATA_TYPE_INT32},
    {"aiv_all_to_all_v_uint32_t", HcclDataType::HCCL_DATA_TYPE_UINT32},
    {"aiv_all_to_all_v_int8_t", HcclDataType::HCCL_DATA_TYPE_INT8},
    {"aiv_all_to_all_v_uint8_t", HcclDataType::HCCL_DATA_TYPE_UINT8},
    {"aiv_all_to_all_v_bfloat16_t", HcclDataType::HCCL_DATA_TYPE_BFP16},
    {"aiv_all_to_all_v_int64_t", HcclDataType::HCCL_DATA_TYPE_INT64},
    {"aiv_all_to_all_v_uint64_t", HcclDataType::HCCL_DATA_TYPE_UINT64},
    {"aiv_all_to_all_v_double", HcclDataType::HCCL_DATA_TYPE_FP64},
    {"aiv_all_to_all_v_sp_half", HcclDataType::HCCL_DATA_TYPE_FP16, KernelArgsType::ARGS_TYPE_SUPERPOD},
    {"aiv_all_to_all_v_sp_int16_t", HcclDataType::HCCL_DATA_TYPE_INT16, KernelArgsType::ARGS_TYPE_SUPERPOD},
    {"aiv_all_to_all_v_sp_uint16_t", HcclDataType::HCCL_DATA_TYPE_UINT16, KernelArgsType::ARGS_TYPE_SUPERPOD},
    {"aiv_all_to_all_v_sp_float", HcclDataType::HCCL_DATA_TYPE_FP32, KernelArgsType::ARGS_TYPE_SUPERPOD},
    {"aiv_all_to_all_v_sp_int32_t", HcclDataType::HCCL_DATA_TYPE_INT32, KernelArgsType::ARGS_TYPE_SUPERPOD},
    {"aiv_all_to_all_v_sp_uint32_t", HcclDataType::HCCL_DATA_TYPE_UINT32, KernelArgsType::ARGS_TYPE_SUPERPOD},
    {"aiv_all_to_all_v_sp_int8_t", HcclDataType::HCCL_DATA_TYPE_INT8, KernelArgsType::ARGS_TYPE_SUPERPOD},
    {"aiv_all_to_all_v_sp_uint8_t", HcclDataType::HCCL_DATA_TYPE_UINT8, KernelArgsType::ARGS_TYPE_SUPERPOD},
    {"aiv_all_to_all_v_sp_bfloat16_t", HcclDataType::HCCL_DATA_TYPE_BFP16, KernelArgsType::ARGS_TYPE_SUPERPOD},
    {"aiv_all_to_all_v_sp_int64_t", HcclDataType::HCCL_DATA_TYPE_INT64, KernelArgsType::ARGS_TYPE_SUPERPOD},
    {"aiv_all_to_all_v_sp_uint64_t", HcclDataType::HCCL_DATA_TYPE_UINT64, KernelArgsType::ARGS_TYPE_SUPERPOD},
    {"aiv_all_to_all_v_sp_double", HcclDataType::HCCL_DATA_TYPE_FP64, KernelArgsType::ARGS_TYPE_SUPERPOD},
};
static std::string g_alltoallvAivBinaryName = "hccl_aiv_all_to_all_v_op.o";

static std::vector<AivKernelInfo> g_alltoallvcAivKernelInfoList = {
    {"aiv_all_to_all_vc_half", HcclDataType::HCCL_DATA_TYPE_FP16},
    {"aiv_all_to_all_vc_int16_t", HcclDataType::HCCL_DATA_TYPE_INT16},
    {"aiv_all_to_all_vc_uint16_t", HcclDataType::HCCL_DATA_TYPE_UINT16},
    {"aiv_all_to_all_vc_float", HcclDataType::HCCL_DATA_TYPE_FP32},
    {"aiv_all_to_all_vc_int32_t", HcclDataType::HCCL_DATA_TYPE_INT32},
    {"aiv_all_to_all_vc_uint32_t", HcclDataType::HCCL_DATA_TYPE_UINT32},
    {"aiv_all_to_all_vc_int8_t", HcclDataType::HCCL_DATA_TYPE_INT8},
    {"aiv_all_to_all_vc_uint8_t", HcclDataType::HCCL_DATA_TYPE_UINT8},
    {"aiv_all_to_all_vc_bfloat16_t", HcclDataType::HCCL_DATA_TYPE_BFP16},
    {"aiv_all_to_all_vc_int64_t", HcclDataType::HCCL_DATA_TYPE_INT64},
    {"aiv_all_to_all_vc_uint64_t", HcclDataType::HCCL_DATA_TYPE_UINT64},
    {"aiv_all_to_all_vc_double", HcclDataType::HCCL_DATA_TYPE_FP64},
};
static std::string g_alltoallvcAivBinaryName = "hccl_aiv_all_to_all_vc_op.o";

static std::vector<AivKernelInfo> g_reducescattervAivKernelInfoList = {
    {"aiv_reduce_scatter_v_float", HcclDataType::HCCL_DATA_TYPE_FP32},
    {"aiv_reduce_scatter_v_half", HcclDataType::HCCL_DATA_TYPE_FP16},
    {"aiv_reduce_scatter_v_int16_t", HcclDataType::HCCL_DATA_TYPE_INT16},
    {"aiv_reduce_scatter_v_int32_t", HcclDataType::HCCL_DATA_TYPE_INT32},
    {"aiv_reduce_scatter_v_int8_t", HcclDataType::HCCL_DATA_TYPE_INT8},
    {"aiv_reduce_scatter_v_bfloat16_t", HcclDataType::HCCL_DATA_TYPE_BFP16},
};
static std::string g_reducescattervAivBinaryName = "hccl_aiv_reduce_scatter_v_op.o";

static std::vector<AivKernelInfo> g_allgathervAivKernelInfoList = {
    {"aiv_all_gather_v_half", HcclDataType::HCCL_DATA_TYPE_FP16},
    {"aiv_all_gather_v_int16_t", HcclDataType::HCCL_DATA_TYPE_INT16},
    {"aiv_all_gather_v_uint16_t", HcclDataType::HCCL_DATA_TYPE_UINT16},
    {"aiv_all_gather_v_float", HcclDataType::HCCL_DATA_TYPE_FP32},
    {"aiv_all_gather_v_int32_t", HcclDataType::HCCL_DATA_TYPE_INT32},
    {"aiv_all_gather_v_uint32_t", HcclDataType::HCCL_DATA_TYPE_UINT32},
    {"aiv_all_gather_v_int8_t", HcclDataType::HCCL_DATA_TYPE_INT8},
    {"aiv_all_gather_v_uint8_t", HcclDataType::HCCL_DATA_TYPE_UINT8},
    {"aiv_all_gather_v_bfloat16_t", HcclDataType::HCCL_DATA_TYPE_BFP16},
    {"aiv_all_gather_v_int64_t", HcclDataType::HCCL_DATA_TYPE_INT64},
    {"aiv_all_gather_v_uint64_t", HcclDataType::HCCL_DATA_TYPE_UINT64},
    {"aiv_all_gather_v_double", HcclDataType::HCCL_DATA_TYPE_FP64},
};
static std::string g_allgathervAivBinaryName = "hccl_aiv_all_gather_v_op.o";

static std::map<HcclCMDType, std::pair<std::string, std::vector<AivKernelInfo>>> g_aivKernelInfoMap = {
    {HcclCMDType::HCCL_CMD_ALLREDUCE, {g_allreduceAivBinaryName, g_allreduceAivKernelInfoList}},
    {HcclCMDType::HCCL_CMD_REDUCE_SCATTER, {g_reducescatterAivBinaryName, g_reducescatterAivKernelInfoList}},
    {HcclCMDType::HCCL_CMD_ALLGATHER, {g_allgatherAivBinaryName, g_allgatherAivKernelInfoList}},
    {HcclCMDType::HCCL_CMD_BROADCAST, {g_broadcastAivBinaryName, g_broadcastAivKernelInfoList}},
    {HcclCMDType::HCCL_CMD_ALLTOALL, {g_alltoallAivBinaryName, g_alltoallAivKernelInfoList}},
    {HcclCMDType::HCCL_CMD_ALLTOALLV, {g_alltoallvAivBinaryName, g_alltoallvAivKernelInfoList}},
    {HcclCMDType::HCCL_CMD_ALLTOALLVC, {g_alltoallvcAivBinaryName, g_alltoallvcAivKernelInfoList}},
    {HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V, {g_reducescattervAivBinaryName, g_reducescattervAivKernelInfoList}},
    {HcclCMDType::HCCL_CMD_ALLGATHER_V, {g_allgathervAivBinaryName, g_allgathervAivKernelInfoList}},
};

using AivKernelArgs = struct AivKernelArgsDef {
    const void* buffersIn[MAX_RANK_SIZE] = {};  // 注册的CCLIN地址，所有卡可访问
    const void* buffersOut[MAX_RANK_SIZE] = {}; // 注册的CCLOUT地址，所有卡可访问
    const void* input;
    const void* output;
    u32 rank;
    u32 rankSize;
    u64 len;
    u32 dataType;
    u32 reduceOp;
    u32 root;
    s32 tag; // 第几次调用，定时重置成1
    u32 numBlocks;
    bool isOpBase;
    u64 bufferSize;
    s32 aivRdmaStep;      // 用于AIV与rdma组合的场景，在不同step中kernel完成多机通信的不同部分
    bool useAivRdmaSmall; // 使用aivRdma小数据量kernel，否则使用中数据量kernel
    u32 serverNum;
    u32 devType;
    const void* headCounterAddr;
    const void* tailCounterAddr;
    const void* addOneAddr;
    u32 counterMemSize;
    bool isEnableCounter;
    u32 deterministic;
    u64 rmaInfo;

    AivKernelArgsDef(
        void** buffIn, void** buffOut, const void* input, const void* output, u32 rank, u32 rankSize, u64 len,
        u32 dataType, u32 reduceOp, u32 root, s32 tag, u32 numBlocks, bool isOpBase = true,
        u64 bufferSize = 200 * 1024 * 1024, s32 aivRdmaStep = -1, bool useAivRdmaSmall = false, u32 serverNum = 1,
        u32 devType = 2, const void* headCounterAddr = nullptr, const void* tailCounterAddr = nullptr,
        const void* addOneAddr = nullptr, u32 counterMemSize = 0, bool isEnableCounter = false, u32 deterministic = 0,
        u64 rmaInfo = 0)
        : input(input),
          output(output),
          rank(rank),
          rankSize(rankSize),
          len(len),
          dataType(dataType),
          reduceOp(reduceOp),
          root(root),
          tag(tag),
          numBlocks(numBlocks),
          isOpBase(isOpBase),
          bufferSize(bufferSize),
          aivRdmaStep(aivRdmaStep),
          useAivRdmaSmall(useAivRdmaSmall),
          serverNum(serverNum),
          devType(devType),
          headCounterAddr(headCounterAddr),
          tailCounterAddr(tailCounterAddr),
          addOneAddr(addOneAddr),
          counterMemSize(counterMemSize),
          isEnableCounter(isEnableCounter),
          deterministic(deterministic),
          rmaInfo(rmaInfo)
    {
        for (u32 i = 0; i < MAX_RANK_SIZE; i++) {
            buffersIn[i] = (u8*)buffIn[i];
            buffersOut[i] = (u8*)buffOut[i];
        }
    }
};

using AivExtraKernelArgs = struct AivExtraKernelArgsDef {
    const void* buffersIn[MAX_RANK_SIZE] = {};  // 注册的CCLIN地址，所有卡可访问
    const void* buffersOut[MAX_RANK_SIZE] = {}; // 注册的CCLOUT地址，所有卡可访问
    const void* input;
    const void* output;
    u32 rank;
    u32 rankSize;
    u64 len;
    u32 dataType;
    u32 reduceOp;
    u32 root;
    s32 tag; // 第几次调用，定时重置成1
    u32 numBlocks;
    bool isOpBase;
    u64 bufferSize;
    s32 aivRdmaStep;      // 用于AIV与rdma组合的场景，在不同step中kernel完成多机通信的不同部分
    bool useAivRdmaSmall; // 使用aivRdma小数据量kernel，否则使用中数据量kernel
    u32 serverNum;
    u32 devType;
    const void* headCounterAddr;
    const void* tailCounterAddr;
    const void* addOneAddr;
    u32 counterMemSize;
    bool isEnableCounter;
    u32 deterministic;
    u64 rmaInfo;
    ExtraArgs extraArgs; // A2/A3单机

    AivExtraKernelArgsDef(
        void** buffIn, void** buffOut, const void* input, const void* output, u32 rank, u32 rankSize, u64 len,
        u32 dataType, u32 reduceOp, u32 root, s32 tag, u32 numBlocks, bool isOpBase = true,
        u64 bufferSize = 200 * 1024 * 1024, s32 aivRdmaStep = -1, bool useAivRdmaSmall = false, u32 serverNum = 1,
        u32 devType = 2, const void* headCounterAddr = nullptr, const void* tailCounterAddr = nullptr,
        const void* addOneAddr = nullptr, u32 counterMemSize = 0, bool isEnableCounter = false, u32 deterministic = 0,
        u64 rmaInfo = 0, const ExtraArgs* extraArgsPtr = nullptr)
        : input(input),
          output(output),
          rank(rank),
          rankSize(rankSize),
          len(len),
          dataType(dataType),
          reduceOp(reduceOp),
          root(root),
          tag(tag),
          numBlocks(numBlocks),
          isOpBase(isOpBase),
          bufferSize(bufferSize),
          aivRdmaStep(aivRdmaStep),
          useAivRdmaSmall(useAivRdmaSmall),
          serverNum(serverNum),
          devType(devType),
          headCounterAddr(headCounterAddr),
          tailCounterAddr(tailCounterAddr),
          addOneAddr(addOneAddr),
          counterMemSize(counterMemSize),
          isEnableCounter(isEnableCounter),
          deterministic(deterministic),
          rmaInfo(rmaInfo)
    {
        for (u32 i = 0; i < MAX_RANK_SIZE; i++) {
            buffersOut[i] = (u8*)buffOut[i];
            buffersIn[i] = (u8*)buffIn[i];
        }
        if (extraArgsPtr != nullptr) {
            extraArgs = *extraArgsPtr;
        }
    }
};

using AivExtraKernelArgsV2 = struct AivExtraKernelArgsV2Def {
    const void* buffersIn[MAX_RANK_SIZE] = {};  // 注册的CCLIN地址，所有卡可访问
    const void* buffersOut[MAX_RANK_SIZE] = {}; // 注册的CCLOUT地址，所有卡可访问
    const void* input;
    const void* output;
    u32 rank;
    u32 rankSize;
    u64 len;
    u32 dataType;
    u32 reduceOp;
    u32 root;
    s32 tag; // 第几次调用，定时重置成1
    u32 numBlocks;
    bool isOpBase;
    u64 bufferSize;
    s32 aivRdmaStep;      // 用于AIV与rdma组合的场景，在不同step中kernel完成多机通信的不同部分
    bool useAivRdmaSmall; // 使用aivRdma小数据量kernel，否则使用中数据量kernel
    u32 serverNum;
    u32 devType;
    const void* headCounterAddr;
    const void* tailCounterAddr;
    const void* addOneAddr;
    u32 counterMemSize;
    bool isEnableCounter;
    u32 deterministic;
    u64 rmaInfo;
    ExtraArgsV2 extraArgs; // A3超节点内多机

    AivExtraKernelArgsV2Def(
        void** buffIn, void** buffOut, const void* input, const void* output, u32 rank, u32 rankSize, u64 len,
        u32 dataType, u32 reduceOp, u32 root, s32 tag, u32 numBlocks, bool isOpBase = true,
        u64 bufferSize = 200 * 1024 * 1024, s32 aivRdmaStep = -1, bool useAivRdmaSmall = false, u32 serverNum = 1,
        u32 devType = 2, const void* headCounterAddr = nullptr, const void* tailCounterAddr = nullptr,
        const void* addOneAddr = nullptr, u32 counterMemSize = 0, bool isEnableCounter = false, u32 deterministic = 0,
        u64 rmaInfo = 0, const ExtraArgsV2* extraArgsPtr = nullptr)
        : input(input),
          output(output),
          rank(rank),
          rankSize(rankSize),
          len(len),
          dataType(dataType),
          reduceOp(reduceOp),
          root(root),
          tag(tag),
          numBlocks(numBlocks),
          isOpBase(isOpBase),
          bufferSize(bufferSize),
          aivRdmaStep(aivRdmaStep),
          useAivRdmaSmall(useAivRdmaSmall),
          serverNum(serverNum),
          devType(devType),
          headCounterAddr(headCounterAddr),
          tailCounterAddr(tailCounterAddr),
          addOneAddr(addOneAddr),
          counterMemSize(counterMemSize),
          isEnableCounter(isEnableCounter),
          deterministic(deterministic),
          rmaInfo(rmaInfo)
    {
        for (u32 i = 0; i < MAX_RANK_SIZE; i++) {
            buffersIn[i] = (u8*)buffIn[i];
            buffersOut[i] = (u8*)buffOut[i];
        }
        if (extraArgsPtr != nullptr) {
            extraArgs = *extraArgsPtr;
        }
    }
};

using AivKernelArgsV3 = struct AivKernelArgsV3Def {
    u64 massArgs[MAX_ARGS_SIZE_A3_STRUCT] = {};
    const void* input;
    const void* output;
    u32 rank;
    u32 rankSize;
    u64 len;
    u32 dataType;
    u32 reduceOp;
    u32 root;
    s32 tag;
    u32 numBlocks;
    bool isOpBase;
    s32 step;
    u32 deterministic;

    AivKernelArgsV3Def(
        void** buffIn, void** buffOut, const void* input, const void* output, u32 rank, u32 rankSize, u64 len,
        u32 dataType, u32 reduceOp, u32 root, s32 tag, u32 numBlocks, bool isOpBase = true,
        u64 bufferSize = 200 * 1024 * 1024, s32 step = 0, void* headCounterAddr = nullptr,
        void* tailCounterAddr = nullptr, void* addOneAddr = nullptr, bool isEnableCounter = false,
        u32 deterministic = 0)
        : input(input),
          output(output),
          rank(rank),
          rankSize(rankSize),
          len(len),
          dataType(dataType),
          reduceOp(reduceOp),
          root(root),
          tag(tag),
          numBlocks(numBlocks),
          isOpBase(isOpBase),
          step(step),
          deterministic(deterministic)
    {
        massArgs[0] = reinterpret_cast<u64>(buffIn[0]);
        massArgs[1] = reinterpret_cast<u64>(buffIn[1]);
        massArgs[2] = reinterpret_cast<u64>(buffOut[0]);
        massArgs[3] = reinterpret_cast<u64>(buffOut[1]);
        massArgs[4] = reinterpret_cast<u64>(bufferSize);
        massArgs[5] = reinterpret_cast<u64>(headCounterAddr);
        massArgs[6] = reinterpret_cast<u64>(tailCounterAddr);
        massArgs[7] = reinterpret_cast<u64>(addOneAddr);
        massArgs[8] = isEnableCounter ? 1 : 0;
    }
};

HcclResult GetAivOpBinaryPath(const std::string& aivBinaryName, std::string& binaryPath)
{
    // 获取二进制文件路径
    std::string libPath;
    char* getPath = nullptr;
    MM_SYS_GET_ENV(MM_ENV_ASCEND_HOME_PATH, getPath);
    if (getPath != nullptr) {
        libPath = getPath;
    } else {
        libPath = "/usr/local/Ascend/cann";
        HCCL_WARNING("[GetAivOpBinaryPath]ENV:ASCEND_HOME_PATH is not set");
    }
    binaryPath = libPath + "/lib64";

    // 拼接应该加载的文件
    binaryPath += "/" + aivBinaryName;
    HCCL_RUN_INFO("[GetAivOpBinaryPath]op binary file path[%s]", binaryPath.c_str());
    return HCCL_SUCCESS;
}

HcclResult ReadBinFile(const string& fileName, string& buffer)
{
    char realFile[PATH_MAX] = {0};
    if (realpath(fileName.c_str(), realFile) == nullptr) {
        HCCL_INFO("[AIV][ReadBinFile] Binfile path %s is not a valid real path.", realFile);
        return HCCL_E_NOT_FOUND;
    }
    std::ifstream filestr;
    filestr.open(realFile, std::ios::binary);
    if (!filestr) {
        HCCL_ERROR("[AIV][ReadBinFile]open file [%s] failed!", fileName.c_str());
        return HCCL_E_OPEN_FILE_FAILURE;
    }

    filestr.seekg(0, std::ios::end);
    std::streampos fileSize = filestr.tellg();
    filestr.seekg(0, std::ios::beg);

    if (fileSize == 0 || fileSize >= MAX_BIN_FILE_SIZE) {
        HCCL_ERROR("[AIV][ReadBinFile] file [%s] size is invalid, is [%d]!", fileName.c_str(), fileSize);
        filestr.close();
        return HCCL_E_OPEN_FILE_FAILURE;
    }
    buffer.resize(fileSize);
    filestr.read(&buffer[0], fileSize);

    filestr.close();
    return HCCL_SUCCESS;
}

s8* GetStubFunc(HcclCMDType cmdType, HcclDataType dataType, KernelArgsType argsType = KernelArgsType::ARGS_TYPE_SERVER)
{
    return reinterpret_cast<s8*>(
        (((static_cast<s64>(cmdType) << SIG_MOVE_LEFT_BITS) + static_cast<s64>(dataType)) << SIG_MOVE_LEFT_BITS)
        + static_cast<s64>(argsType));
}

HcclResult RegisterBinaryKernel(const char* funcName, const aclrtBinHandle binHandle, s8* stubFunc)
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

HcclResult GetKernelFunc(aclrtFuncHandle& funcHandle, s8* stubFunc)
{
    if (stubFunc == nullptr || g_aivFuncMap.find(stubFunc) == g_aivFuncMap.end()) {
        return HCCL_E_PARA;
    }
    funcHandle = g_aivFuncMap[stubFunc];
    return HCCL_SUCCESS;
}

// Kernel注册入口，全局只需要初始化一次
HcclResult RegisterKernel([[maybe_unused]] DevType deviceType)
{
    lock_guard<mutex> guard(g_mut);
    if (g_init) {
        return HCCL_SUCCESS;
    }
    HcclResult ret;

    for (auto& aivOpKernelInfo : g_aivKernelInfoMap) {
        HcclCMDType cmdType = aivOpKernelInfo.first;
        const std::string& aivBinaryName = aivOpKernelInfo.second.first;
        std::vector<AivKernelInfo>& aivKernelInfoList = aivOpKernelInfo.second.second;

        string binFilePath;
        ret = GetAivOpBinaryPath(aivBinaryName, binFilePath);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS, HCCL_ERROR("[AIV][RegisterKernel] get aiv op binary path failed"), HCCL_E_RUNTIME);

        ret = LoadBinaryFromFile(binFilePath.c_str(), ACL_RT_BINARY_LOAD_OPT_LAZY_LOAD, 1, g_binHandleMap[cmdType]);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS, HCCL_ERROR("[AIV][RegisterKernel] read aiv kernel bin file failed"), HCCL_E_RUNTIME);

        for (auto& aivKernelInfo : aivKernelInfoList) {
            ret = RegisterBinaryKernel(
                aivKernelInfo.kernelName, g_binHandleMap[cmdType],
                GetStubFunc(cmdType, aivKernelInfo.dataType, aivKernelInfo.argsType));
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR(
                    "[AIV][RegisterKernel] register binary kernel for kernelName[%s] "
                    "cmdType[%d] dataType[%s] argsType[%d] failed",
                    aivKernelInfo.kernelName, cmdType, GetDataTypeEnumStr(aivKernelInfo.dataType).c_str(),
                    aivKernelInfo.argsType),
                HCCL_E_RUNTIME);
        }
    }

    g_init = true;

    return HCCL_SUCCESS;
}

HcclResult UnRegisterAivKernel()
{
    lock_guard<mutex> guard(g_mut);
    if (g_init) {
        for (auto& binHandleItem : g_binHandleMap) {
            aclrtBinaryUnLoad(binHandleItem.second);
        }
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

    constexpr u64 MAX_INTERVAL = 254;
    // NPU超时范围(1, 254) * interval
    minNpuSchedTimeout = 1 * interval;
    maxNpuSchedTimeout = MAX_INTERVAL * interval;
    HCCL_INFO(
        "GetMinAndMaxNpuSchedTimeOut minNpuSchedTimeout[%llu]us, maxNpuSchedTimeout[%llu]us.", minNpuSchedTimeout,
        maxNpuSchedTimeout);
    return HCCL_SUCCESS;
}

u32 GetAivTimeout(s32 execTimeOut, bool isSetByConfig)
{
    u32 timeout = AIV_TIMEOUT_DEFAULT_US;
    if (GetExternalInputHcclExecTimeoutSet() != HcclExecTimeoutSet::HCCL_EXEC_TIMEOUT_NOT_SET || isSetByConfig) {
        // 配0时，使用最大超时时间
        if (execTimeOut == 0) {
            return AIV_TIMEOUT_MAX_US;
        }
        double timeoutUs = execTimeOut * TIME_S_TO_US;
        if (timeoutUs > static_cast<double>(std::numeric_limits<u32>::max())) {
            HCCL_INFO("[GetAivTimeout]Get input timeout[%.2f us] is out of valid range.", timeoutUs);
            return AIV_TIMEOUT_MAX_US;
        }
        u32 timeoutUsInt = static_cast<u32>(timeoutUs);
        u64 minNpuSchedTimeout = 0;
        u64 maxNpuSchedTimeout = 0;
        CHK_RET(GetMinAndMaxNpuSchedTimeOut(minNpuSchedTimeout, maxNpuSchedTimeout));
        timeout = (timeoutUsInt < minNpuSchedTimeout) ? minNpuSchedTimeout :
                  (timeoutUsInt > maxNpuSchedTimeout) ? maxNpuSchedTimeout :
                                                        timeoutUsInt;
        HCCL_INFO(
            "[GetAivTimeout]timeout[%u]us, minNpuSchedTimeout[%llu]us, maxNpuSchedTimeout[%llu]us.", timeout,
            minNpuSchedTimeout, maxNpuSchedTimeout);
    }

    return timeout < AIV_TIMEOUT_MAX_US ? timeout : AIV_TIMEOUT_MAX_US;
}

void TaskAivProfilerWrap(
    const AivOpArgs& opArgs, const AivTopoArgs& topoArgs, const AivResourceArgs& resourceArgs,
    const AivAlgArgs& algArgs, const AivProfilingInfo& aivProfilingInfo, void* flagMem)
{
    struct TaskParaGeneral taskParaGeneral;

    TaskParaAiv taskParaAiv(
        opArgs.cmdType, resourceArgs.aivTag, opArgs.count * SIZE_TABLE[opArgs.dataType], resourceArgs.numBlocks,
        topoArgs.rankSize, algArgs.step, flagMem, topoArgs.rank, opArgs.isOpBase);

    if (taskParaAiv.flagMem == nullptr) {
        taskParaAiv.flagMem = resourceArgs.buffersOut[topoArgs.rank];
    }

    taskParaGeneral.isMainStream = true;
    taskParaGeneral.stream = resourceArgs.stream;
    taskParaGeneral.beginTime = aivProfilingInfo.beginTime;
    taskParaGeneral.aiv = taskParaAiv;

    AlgWrap::GetInstance().TaskAivProfiler(topoArgs.identify, taskParaGeneral);
}

HcclResult CacheTaskOpInfo(rtStream_t stream, const std::string& identify)
{
    aclmdlRI rtModel = nullptr;
    aclmdlRICaptureStatus captureStatus = aclmdlRICaptureStatus::ACL_MODEL_RI_CAPTURE_STATUS_NONE;
    aclError aclRet = aclmdlRICaptureGetInfo(stream, &captureStatus, &rtModel);
    if (aclRet == ACL_ERROR_RT_FEATURE_NOT_SUPPORT) {
        HCCL_WARNING("[%s]Stream capture does not support!", __func__);
        return HCCL_SUCCESS;
    }
    CHK_PRT_RET(
        aclRet != ACL_SUCCESS, HCCL_ERROR("[%s]rtGet stream get capture status fail. return[%d]", __func__, aclRet),
        HCCL_E_RUNTIME);

    aclrtStreamAttrValue value;
    aclRet = aclrtGetStreamAttribute(stream, ACL_STREAM_ATTR_CACHE_OP_INFO, &value);
    CHK_PRT_RET(
        aclRet != ACL_SUCCESS, HCCL_ERROR("[%s]stream get attribute fail. return[%d]", __func__, aclRet),
        HCCL_E_RUNTIME);

    HCCL_INFO(
        "[CacheTaskOpInfo] cacheOpInfoSwitch[%u] captureStatus[%d] identify[%s]", value.cacheOpInfoSwitch,
        captureStatus, identify.c_str());
    if (value.cacheOpInfoSwitch == 1 && captureStatus == ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE) {
        HcclResult cacheRet = hrtCacheLastTaskExtendInfo(identify.c_str(), strlen(identify.c_str()));
        if (cacheRet == HCCL_E_NOT_SUPPORT) {
            HCCL_INFO("[%s] aclrtCacheLastTaskExtendInfo not supported", __func__);
        } else {
            CHK_PRT_RET(
                cacheRet != HCCL_SUCCESS,
                HCCL_ERROR("[%s] stream cache task op info fail. return[%d]", __func__, cacheRet), HCCL_E_RUNTIME);
        }
    }
    return HCCL_SUCCESS;
}

// KernelLaunch内部接口
HcclResult ExecuteKernelLaunchInner(
    const AivOpArgs& opArgs, const AivTopoArgs& topoArgs, const AivResourceArgs& resourceArgs,
    const AivAlgArgs& algArgs, void* args, u32 argsSize, AivProfilingInfo& aivProfilingInfo)
{
    HCCL_INFO(
        "[AIV][ExecuteKernelLaunchInner] sendbuff [%p] recvbuff [%p] rank [%d] rankSize [%d] count [%llu] "
        "dataType [%s] reduceOp [%s] root [%d] tag [%d] isOpBase [%d] bufferSize [%llu] step [%d] "
        "isSmallCount [%d] serverNum [%d] devType[%d] extraArgsPtr [%p] argsSize [%d], deterministic [%d].",
        opArgs.input, opArgs.output, topoArgs.rank, topoArgs.rankSize, opArgs.count,
        GetDataTypeEnumStr(opArgs.dataType).c_str(), GetReduceOpEnumStr(opArgs.op).c_str(), opArgs.root,
        resourceArgs.aivTag, opArgs.isOpBase, resourceArgs.bufferSize, algArgs.step, algArgs.isSmallCount,
        topoArgs.serverNum, topoArgs.devType, args, argsSize, algArgs.deterministic);

    HCCL_DEBUG(
        "[AIV][ExecuteKernelLaunchInner] buffersIn [%p] [%p] [%p] [%p] [%p] [%p] [%p] [%p] "
        "buffersOut [%p] [%p] [%p] [%p] [%p] [%p] [%p] [%p].",
        resourceArgs.buffersIn[RANK_ZERO], resourceArgs.buffersIn[RANK_ONE], resourceArgs.buffersIn[RANK_TWO],
        resourceArgs.buffersIn[RANK_THREE], resourceArgs.buffersIn[RANK_FOUR], resourceArgs.buffersIn[RANK_FIVE],
        resourceArgs.buffersIn[RANK_SIX], resourceArgs.buffersIn[RANK_SEVEN], resourceArgs.buffersOut[RANK_ZERO],
        resourceArgs.buffersOut[RANK_ONE], resourceArgs.buffersOut[RANK_TWO], resourceArgs.buffersOut[RANK_THREE],
        resourceArgs.buffersOut[RANK_FOUR], resourceArgs.buffersOut[RANK_FIVE], resourceArgs.buffersOut[RANK_SIX],
        resourceArgs.buffersOut[RANK_SEVEN]);

    KernelArgsType argsType = algArgs.argsType;

    HcclResult ret = HcclResult::HCCL_E_PARA;
    aclrtLaunchKernelCfg cfg;
    aclrtLaunchKernelAttr attr[AIV_ATTRNUM_THREE];
    attr[0].id = ACL_RT_LAUNCH_KERNEL_ATTR_SCHEM_MODE;
    attr[0].value.schemMode = 1;
    attr[1].id = ACL_RT_LAUNCH_KERNEL_ATTR_TIMEOUT_US;
    attr[1].value.timeoutUs.timeoutLow = GetAivTimeout(algArgs.execTimeOut, algArgs.execTimeOutSet);
    attr[1].value.timeoutUs.timeoutHigh = 0;
    attr[2].id = ACL_RT_LAUNCH_KERNEL_ATTR_ENGINE_TYPE;
    attr[2].value.engineType = ACL_RT_ENGINE_TYPE_AIV;
    cfg.numAttrs = AIV_ATTRNUM_THREE;
    cfg.attrs = attr;

    aclrtFuncHandle funcHandle;
    s8* stubFunc = GetStubFunc(opArgs.cmdType, opArgs.dataType, argsType);
    ret = GetKernelFunc(funcHandle, stubFunc);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[ExecuteKernelLaunchInner] errNo[0x%016llx] GetKernelFunc failed, "
            "return[%d]",
            HCCL_ERROR_CODE(HCCL_E_RUNTIME), ret),
        HCCL_E_RUNTIME);

    aclError aclRet = aclrtLaunchKernelWithHostArgs(
        funcHandle, resourceArgs.numBlocks, resourceArgs.stream, &cfg, args, argsSize, nullptr, 0);
    if (aclRet == ACL_ERROR_RT_INVALID_HANDLE) {
        aclError aclGetRet
            = aclrtBinaryGetFunction(g_binHandleMap[opArgs.cmdType], g_aivNameMap[stubFunc].c_str(), &funcHandle);
        CHK_PRT_RET(
            aclGetRet != ACL_SUCCESS,
            HCCL_ERROR("[RegisterBinaryKernel]errNo[0x%016llx] get function from binary error.", aclRet),
            HCCL_E_NOT_FOUND);
        aclRet = aclrtLaunchKernelWithHostArgs(
            funcHandle, resourceArgs.numBlocks, resourceArgs.stream, &cfg, args, argsSize, nullptr, 0);
    }
    CHK_PRT_RET(
        aclRet != ACL_SUCCESS,
        HCCL_ERROR(
            "[ExecuteKernelLaunchInner]errNo[0x%016llx] aclrtLaunchKernelWithHostArgs error[%d].",
            HCCL_ERROR_CODE(HCCL_E_RUNTIME), aclRet),
        HCCL_E_RUNTIME);

    CHK_RET(CacheTaskOpInfo(resourceArgs.stream, topoArgs.identify));

    TaskAivProfilerWrap(
        opArgs, topoArgs, resourceArgs, algArgs, aivProfilingInfo,
        (algArgs.argsType != KernelArgsType::ARGS_TYPE_SERVER) ? resourceArgs.buffersOut[0] :
                                                                 resourceArgs.buffersOut[topoArgs.rank]);

    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[AIV][ExecuteKernelLaunchInner] errNo[0x%016llx] rtKernelLaunch aiv fail, "
            "return[%d]",
            HCCL_ERROR_CODE(HCCL_E_RUNTIME), ret),
        HCCL_E_RUNTIME);
    return HCCL_SUCCESS;
}

// Kernel单次调用Launch外部接口
HcclResult ExecuteKernelLaunch(
    const AivOpArgs& opArgs, const AivTopoArgs& topoArgs, const AivResourceArgs& resourceArgs,
    const AivAlgArgs& algArgs, AivProfilingInfo& aivProfilingInfo)
{
    SetAivProfilingInfoBeginTime(aivProfilingInfo);
    CHK_PTR_NULL(resourceArgs.buffersIn);
    CHK_PTR_NULL(resourceArgs.buffersOut);

    if (algArgs.argsType == KernelArgsType::ARGS_TYPE_SIMPLE) {
        AivKernelArgsV3 aivKernelArgs{
            resourceArgs.buffersIn,
            resourceArgs.buffersOut,
            opArgs.input,
            opArgs.output,
            topoArgs.rank,
            topoArgs.rankSize,
            opArgs.count,
            opArgs.dataType,
            opArgs.op,
            opArgs.root,
            resourceArgs.aivTag,
            resourceArgs.numBlocks,
            opArgs.isOpBase,
            resourceArgs.bufferSize,
            algArgs.step,
            reinterpret_cast<void*>(aivProfilingInfo.counter.headCountMem),
            reinterpret_cast<void*>(aivProfilingInfo.counter.tailCountMem),
            reinterpret_cast<void*>(aivProfilingInfo.counter.addOneMem),
            aivProfilingInfo.counter.isEnableCounter,
            algArgs.deterministic};
        CHK_RET(ExecuteKernelLaunchInner(
            opArgs, topoArgs, resourceArgs, algArgs, &aivKernelArgs, sizeof(aivKernelArgs), aivProfilingInfo));
    } else {
        AivKernelArgs aivKernelArgs{
            resourceArgs.buffersIn,
            resourceArgs.buffersOut,
            opArgs.input,
            opArgs.output,
            topoArgs.rank,
            topoArgs.rankSize,
            opArgs.count,
            opArgs.dataType,
            opArgs.op,
            opArgs.root,
            resourceArgs.aivTag,
            resourceArgs.numBlocks,
            opArgs.isOpBase,
            resourceArgs.bufferSize,
            algArgs.step,
            algArgs.isSmallCount,
            topoArgs.serverNum,
            static_cast<u32>(topoArgs.devType),
            reinterpret_cast<void*>(aivProfilingInfo.counter.headCountMem),
            reinterpret_cast<void*>(aivProfilingInfo.counter.tailCountMem),
            reinterpret_cast<void*>(aivProfilingInfo.counter.addOneMem),
            aivProfilingInfo.counter.memSize,
            aivProfilingInfo.counter.isEnableCounter,
            algArgs.deterministic,
            algArgs.rmaInfo};
        CHK_RET(ExecuteKernelLaunchInner(
            opArgs, topoArgs, resourceArgs, algArgs, &aivKernelArgs, sizeof(aivKernelArgs), aivProfilingInfo));
    }

    return HCCL_SUCCESS;
}

// Kernel单次调用Launch外部接口
HcclResult ExecuteKernelLaunch(
    const AivOpArgs& opArgs, const AivTopoArgs& topoArgs, const AivResourceArgs& resourceArgs,
    const AivAlgArgs& algArgs, const ExtraArgs& extraArgs, AivProfilingInfo& aivProfilingInfo)
{
    SetAivProfilingInfoBeginTime(aivProfilingInfo);
    CHK_PTR_NULL(resourceArgs.buffersIn);
    CHK_PTR_NULL(resourceArgs.buffersOut);

    AivExtraKernelArgs aivExtraKernelArgs{
        resourceArgs.buffersIn,
        resourceArgs.buffersOut,
        opArgs.input,
        opArgs.output,
        topoArgs.rank,
        topoArgs.rankSize,
        opArgs.count,
        opArgs.dataType,
        opArgs.op,
        opArgs.root,
        resourceArgs.aivTag,
        resourceArgs.numBlocks,
        opArgs.isOpBase,
        resourceArgs.bufferSize,
        algArgs.step,
        algArgs.isSmallCount,
        topoArgs.serverNum,
        static_cast<u32>(topoArgs.devType),
        reinterpret_cast<void*>(aivProfilingInfo.counter.headCountMem),
        reinterpret_cast<void*>(aivProfilingInfo.counter.tailCountMem),
        reinterpret_cast<void*>(aivProfilingInfo.counter.addOneMem),
        aivProfilingInfo.counter.memSize,
        aivProfilingInfo.counter.isEnableCounter,
        algArgs.deterministic,
        algArgs.rmaInfo,
        &extraArgs};
    CHK_RET(ExecuteKernelLaunchInner(
        opArgs, topoArgs, resourceArgs, algArgs, &aivExtraKernelArgs, sizeof(aivExtraKernelArgs), aivProfilingInfo));

    return HCCL_SUCCESS;
}

// Kernel单次调用Launch外部接口
HcclResult ExecuteKernelLaunch(
    const AivOpArgs& opArgs, const AivTopoArgs& topoArgs, const AivResourceArgs& resourceArgs,
    const AivAlgArgs& algArgs, const ExtraArgsV2& extraArgs, AivProfilingInfo& aivProfilingInfo)
{
    SetAivProfilingInfoBeginTime(aivProfilingInfo);
    CHK_PTR_NULL(resourceArgs.buffersIn);
    CHK_PTR_NULL(resourceArgs.buffersOut);

    AivExtraKernelArgsV2 aivExtraKernelArgs{
        resourceArgs.buffersIn,
        resourceArgs.buffersOut,
        opArgs.input,
        opArgs.output,
        topoArgs.rank,
        topoArgs.rankSize,
        opArgs.count,
        opArgs.dataType,
        opArgs.op,
        opArgs.root,
        resourceArgs.aivTag,
        resourceArgs.numBlocks,
        opArgs.isOpBase,
        resourceArgs.bufferSize,
        algArgs.step,
        algArgs.isSmallCount,
        topoArgs.serverNum,
        static_cast<u32>(topoArgs.devType),
        reinterpret_cast<void*>(aivProfilingInfo.counter.headCountMem),
        reinterpret_cast<void*>(aivProfilingInfo.counter.tailCountMem),
        reinterpret_cast<void*>(aivProfilingInfo.counter.addOneMem),
        aivProfilingInfo.counter.memSize,
        aivProfilingInfo.counter.isEnableCounter,
        algArgs.deterministic,
        algArgs.rmaInfo,
        &extraArgs};
    CHK_RET(ExecuteKernelLaunchInner(
        opArgs, topoArgs, resourceArgs, algArgs, &aivExtraKernelArgs, sizeof(aivExtraKernelArgs), aivProfilingInfo));

    return HCCL_SUCCESS;
}

} // namespace hccl
