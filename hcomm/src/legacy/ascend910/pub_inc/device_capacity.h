/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_DEVICE_CAPACITY_H
#define HCCL_DEVICE_CAPACITY_H

#include "hccl/base.h"
#include "dtype_common.h"
#include "hccl_common.h"

namespace hccl {
// 节点间RDMA发送数据单个WQE支持的最大数据量
const u64 RDMA_SEND_MAX_SIZE = 0x80000000;
// 节点内单个SDMA任务发送数据支持的最大数据量
const u64 SDMA_SEND_MAX_SIZE = 0x100000000;
constexpr u32 MAX_DEVICE_NUM_THIRTY_TWO = 32;
constexpr u32 MAX_DEVICE_NUM_SIXTY_FIVE = 65;
constexpr u32 MAX_DEVICE_NUM_SIXTEEN = 16;
bool IsSupportAIVCopy(HcclDataType dataType);
bool IsSupportAIVReduce(HcclDataType dataType, HcclReduceOp op);
bool IsSupportSDMAReduce(const void* inputPtr, const void* outputPtr, HcclDataType dataType, HcclReduceOp op);
bool IsSupportRDMAReduce(HcclDataType dataType, HcclReduceOp op);
HcclResult GetBandWidthPerNPU(u32 level, u32 userRankSize, u32 deviceNumPerAggregation, float& bandWidth);
HcclResult CheckDeviceType(const DevType deviceType);
bool IsOverFlowInfNanMode();
bool Is310PDevice();
bool IsUseSdidForDeviceId(const u32 superDeviceId = INVALID_UINT); // deprecated
HcclResult IsSuperPodMode(bool& useSuperPodMode);
bool IsSupportRDMALite(const s32 deviceLogicId); // 是否支持rdma lite
HcclResult IsSupportHccsAndSio(bool& flag);      // 是否支持hccs sio并发
HcclResult GetMemBlockNum(const u32 devicePhyId, u32& memBlockNum);
HcclResult IsSupportAicpuNormalQP(const u32& devicePhyId, bool& isSupportNormalQP); // 是否支持AICPU的Normal QP
HcclResult IsSupportAIVNormalQP(const u32& devicePhyId, bool& isSupport);           // 是否支持AIV直通ROCE
HcclResult GetMaxDevNum(u32& MaxDevNum);
u32 GetNotifyMaxWaitTime();
HcclResult IsSupportAtomicWrite(DevType deviceType, u32 devicePhyId, bool& isSupportAtomicWrite);
} // namespace hccl

#endif // end HCCL_DEVICE_CAPACITY_H
