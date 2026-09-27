/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COLL_ALG_UTILS_H
#define COLL_ALG_UTILS_H

#include "externalinput_pub.h"
#include "hccl_common.h"
#include "common.h"
#include "device_capacity.h"
#include "coll_alg_param.h"
#include "op_context.h"
#include "env_config.h"

namespace hccl {
constexpr u64 MAX_ALLTOALL_MESH_ALGO_RANK_INTRA_MESH = 32;
constexpr u32 MAX_RING_PIPELINE_SERVER_NUM = 128; // 防止qp耗尽, Ring算法下Server间流水并行最多支持128 Server
constexpr u32 MIN_PER_LINK_DATA_SIZE = 4 * 1024 * 1024; // Server间流水并行分到每条链路上的最小数据量
constexpr u32 MIN_RING_DATA_SIZE = 64 * 1024;           // Ring算法下, Server间支持流水并行的最小数据量
constexpr u64 MAX_PIPELINE_SLICE_NUM = 4;               // 流水并行算法最大切分次数
constexpr u64 MIN_PIPELINE_SLICE_NUM = 2;               // 流水并行算法最小切分次数
constexpr u64 TINY_MEM_SIZE = 2 * 1024 * 1024;          // AlltoAll算子的tinyMem size
constexpr u64 MAX_ALLTOALLV_DIRECT_FULLMESH_RANKSIZE = 64;  // alltoallv在A2支持directfulmesh的最大ranksize
constexpr u64 MAX_ALLTOALLV_DIRECT_FULLMESH_SERVER_NUM = 8; // alltoallv在A2支持directfulmesh的最大server数

bool IsAlgTypeLevel0Mesh(const AlgTypeLevel0& originalAlgTypeLevel0);

bool IsSupportUnifiedMarch(const OpParam& param, const TopoType& topoType, u32 serverNum, u32 superPodNum);
bool IsAlltoAllvcSatisfyBufferSize(const OpParam& param, u32 userRankSize, u64 cclbufferSize);
bool IsSupportDirectFullmeshForAlltoallv(
    const OpParam& param, DevType deviceType, bool useSuperPodMode, u32 serverNum, bool isSingleMeshAggregation,
    u32 userRankSize, u64 cclbufferSize);
bool FullmeshPairwiseSatisfyHighPerfAlltoallMeshCondition(
    DevType deviceType, u32 rankSize, bool useSuperPodMode, std::vector<HcclAlgoType> algoConfig);
bool SatisfyIntraSuperPod(DevType deviceType, u32 rankSize, bool useSuperPodMode, u32 superPodNum = 1);
bool HcclOpInplaceDefaultCase(const OpParam& param, u8& isInplaceStatus);
bool IsInputOutputOverlap(const OpParam& param, u64 inputDataSize, u64 outputDataSize, u8& isInplaceStatus);
bool IsInputOutPtrNotNullPtr(const OpParam& param, u8& isInplaceStatus);
u32 InplaceDataUnitSize(const HcclCMDType& opType, const OpParam& param);
bool IsHcclOpInplace(
    const HcclCMDType& opType, const OpParam& param, u32 userRank, u32 userRankSize, u8& isInplaceStatus);
bool CheckUserInMemNotLargerThanCCLInMem(
    const HcclCMDType& opType, OpParam& param, u64 commInputSize, u32 userRankSize);
bool ExecutorOnlySupportDMAReduce(const std::string& algName);
bool ExecutorCanSupportDMAReduce(const std::string& algName);
bool ExecutorNoSupportDMAReduce(const std::string& algName);
bool ExecutorSupportInPlace(
    const OpParam& param, const std::string& algName, bool retryEnable,
    InplaceSupportRetryStatus& inPlaceSupportRetryStatus);
bool FitRetryConditionforInPlaceOp(
    const HcclCMDType& opType, OpParam& param, const std::string& algName, u64 commInputSize, u32 userRankSize,
    bool retryEnable, InplaceSupportRetryStatus& inPlaceSupportRetryStatus);
bool IsConfigAHCAlgo(std::map<HcclCMDType, std::vector<HcclAlgoType>> algoConfigMap);
template <typename keyType>
std::string GetAlgoString(const std::map<keyType, std::string>& levelMap, keyType key);
std::string AlgTypeToStr(const AlgType algType);
bool Is310P3Common(bool isHaveCpuRank, DevType deviceType);
u64 CalculatePiplineSliceNum(
    HcclCMDType opType, u64 dataSize, AlgType algType, DevType deviceType, u32 deviceNumPerAggregation, u32 moduleNum);
u32 CalGCD(std::vector<u32>& nums); // 计算n个数的最大公约数
u32 CalGCD(u32 a, u32 b);           // 计算2个数的最大公约数
} // namespace hccl
#endif
