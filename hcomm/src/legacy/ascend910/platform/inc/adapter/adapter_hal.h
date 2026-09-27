/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_INC_ADAPTER_HAL_H
#define HCCL_INC_ADAPTER_HAL_H

#include <hccl/hccl_types.h>
#include "adapter_hal_pub.h"
#include "driver/ascend_hal_define.h"
#include "hccl/base.h"
#include "dtype_common.h"

constexpr u32 HCCL_EVENT_RECV_REQUEST_MSG = static_cast<u32>(EVENT_ID::EVENT_USR_START);
constexpr u32 HCCL_EVENT_SEND_COMPLETION_MSG = static_cast<u32>(EVENT_ID::EVENT_USR_START) + 1U;
constexpr u32 HCCL_EVENT_RECV_COMPLETION_MSG = static_cast<u32>(EVENT_ID::EVENT_USR_START) + 2U;
constexpr u32 HCCL_EVENT_CONGESTION_RELIEF_MSG = static_cast<u32>(EVENT_ID::EVENT_USR_START) + 3U;

constexpr int HCCL_ESCHED_GROUP_ID = 6;
constexpr int HCCL_DRIVER_GROUP_NAME = 16;
constexpr int HCCL_RESERVED_DEV_ID = -1;

constexpr unsigned char SAFTY_STATE_SENSOR_TYTPE = 0xC3;

const std::unordered_map<std::string, DevType> SOC_VER_CONVERT{
    {"Ascend310P1", DevType::DEV_TYPE_310P3},
    {"Ascend310P3", DevType::DEV_TYPE_310P3},
    {"Ascend310P5", DevType::DEV_TYPE_310P3},
    {"Ascend310P7", DevType::DEV_TYPE_310P3},
    {"Ascend310B1", DevType::DEV_TYPE_310P3}, // 临时映射，临时当前Ascend310B1
                                              // torch_npu未与hccl的so解耦；计划20250630完成解耦，解耦后删除
    {"Ascend910", DevType::DEV_TYPE_910},
    {"Ascend910A", DevType::DEV_TYPE_910},
    {"Ascend910B", DevType::DEV_TYPE_910},
    {"Ascend910ProA", DevType::DEV_TYPE_910},
    {"Ascend910ProB", DevType::DEV_TYPE_910},
    {"Ascend910PremiumA", DevType::DEV_TYPE_910},
    {"Ascend910B1", DevType::DEV_TYPE_910B},
    {"Ascend910B2", DevType::DEV_TYPE_910B},
    {"Ascend910B2C", DevType::DEV_TYPE_910B},
    {"Ascend910B3", DevType::DEV_TYPE_910B},
    {"Ascend910B4", DevType::DEV_TYPE_910B},
    {"Ascend910B4-1", DevType::DEV_TYPE_910B},
    {"Ascend910_9391", DevType::DEV_TYPE_910_93},
    {"Ascend910_9381", DevType::DEV_TYPE_910_93},
    {"Ascend910_9392",
     DevType::DEV_TYPE_910_93}, // Ascend910_9392、Ascend910_9382为预留类型，当前版本暂不支持，待跟随后续版本节奏交付
    {"Ascend910_9382", DevType::DEV_TYPE_910_93},
    {"Ascend910_9372", DevType::DEV_TYPE_910_93},
    {"Ascend910_9362", DevType::DEV_TYPE_910_93},
    {"Ascend910_9363", DevType::DEV_TYPE_910_93},
    {"Ascend950PR_958b", DevType::DEV_TYPE_950},
    {"nosoc", DevType::DEV_TYPE_NOSOC}};

HcclResult hrtHalSubmitEvent(u32 devId, u32 eventId, u32 groupId = HCCL_ESCHED_GROUP_ID);
HcclResult hrtHalEschedAttachDevice(unsigned int devId);
HcclResult hrtHalEschedDettachDevice(unsigned int devId);
HcclResult hrtHalGetAPIVersion(int& apiVersion);
HcclResult hrtHalEschedCreateGrp(unsigned int devId, unsigned int grpId, GROUP_TYPE type);
HcclResult hrtHalEschedCreateGrpEx(unsigned int devId, unsigned int* grpId, unsigned int threadNum, GROUP_TYPE type);
HcclResult hrtHalEschedSubscribeEvent(
    unsigned int devId, unsigned int grpId, unsigned int threadId, unsigned long long eventBitmap);
HcclResult hrtHalEschedWaitEvent(
    unsigned int devId, unsigned int grpId, unsigned int threadId, int timeout, struct event_info* event);
HcclResult hrtHalEschedRegisterAckFunc(
    unsigned int eventId, void (*ackFunc)(unsigned int devId, unsigned int subeventId, u8* msg, unsigned int msgLen));

HcclResult hrtHalEschedRegisterAckFuncWithGrpid(
    unsigned int grpid, unsigned int eventId,
    void (*ackFunc)(unsigned int devId, unsigned int subeventId, u8* msg, unsigned int msgLen));

HcclResult hrtHalDrvOpenIpcNotify(const char* name, struct drvIpcNotifyInfo* notify);
HcclResult hrtHalDrvCloseIpcNotify(const char* name, struct drvIpcNotifyInfo* notify);
HcclResult hrtHalDrvIpcNotifyRecord(const char* name);
HcclResult hrtDrvMemCpy(void* dst, uint64_t destMax, const void* src, uint64_t count);
HcclResult hrtDrvGetDevNum(uint32_t* num_dev);
void hrtDrvDeviceGetBareTgid(s32& pid);
HcclResult hrtHalGrpQuery(GroupQueryCmdType cmd, void* inBuff, unsigned int inLen, void* outBuff, unsigned int* outLen);
HcclResult hrtEschedQueryInfo(
    unsigned int devId, ESCHED_QUERY_TYPE type, struct esched_input_info* inPut, struct esched_output_info* outPut);
HcclResult hrtHalBindCgroup(BIND_CGROUP_TYPE bindType);
HcclResult hrtDrvDeviceGetPhyIdByIndex(u32 deviceLogicId, u32& devicePhyId);
HcclResult hrtHalHostRegister(void* hostPtr, u64 size, u32 flag, u32 devid, void*& devPtr);
HcclResult hrtHalHostUnregister(void* hostPtr, u32 devid);
HcclResult hrtHalHostUnregisterEx(void* hostPtr, u32 devid, u32 flag);
HcclResult hrtHalMemCtl(int type, void* input, size_t inputSize, void* output, size_t* outputSize);

s32 hrtGetgrpId(int& groupId, int& devId);
void MemoryPreFetchImpl(u64 start, u64 end, u32 pageSize);
HcclResult MemoryPreFetch(u64 size, void* hostPtr);
HcclResult hrtDrvGetPlatformInfo(uint32_t* info);
HcclResult hrtHalGetChipInfo(uint32_t devId, std::string& chipName);
HcclResult GetRunSideIsDevice(bool& isDeviceSide);
HcclResult CheckRunSideIsDevice();
#endif
