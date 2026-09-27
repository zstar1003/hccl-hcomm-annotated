/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_INC_ADAPTER_RTS_COMMON_H
#define HCCL_INC_ADAPTER_RTS_COMMON_H

#include "hccl_common.h"
#include "dtype_common.h"
#include "acl/acl_rt.h"

#if T_DESC("stream管理", true)
HcclResult hcclStreamSynchronize(HcclRtStream stream, s32 execTimeOut = NOTIFY_DEFAULT_WAIT_TIME);
HcclResult hrtStreamSetMode(HcclRtStream stream, const uint64_t stmMode);
HcclResult hrtStreamGetMode(HcclRtStream const stream, uint64_t* const stmMode);
HcclResult hrtGetStreamId(HcclRtStream stream, s32& streamId);
HcclResult hrtStreamActive(HcclRtStream activeStream, HcclRtStream stream);
HcclResult hrtCacheLastTaskExtendInfo(const char* tag, size_t tagLen);
#endif

HcclResult hrtCtxGetCurrent(HcclRtContext* ctx);
HcclResult hrtCtxSetCurrent(HcclRtContext ctx);
HcclResult hrtEventCreateWithFlag(HcclRtEvent* evt);
HcclResult hrtGetEventID(HcclRtEvent event, uint32_t* eventId);
HcclResult hrtNotifyGetPhyInfo(HcclRtNotify notify, uint32_t* phyDevId, uint32_t* tsId);
HcclResult hrtGetNotifyID(HcclRtNotify signal, u32* notifyID);
HcclResult hrtNotifyReset(aclrtNotify notify);

#if T_DESC("Device管理", true)
HcclResult hrtResetDevice(s32 deviceLogicId);
HcclResult hrtSetDevice(s32 deviceLogicId);
HcclResult hrtGetPairDeviceLinkType(u32 phyDevId, u32 otherPhyDevId, LinkTypeInServer& linkType);

enum class HcclReduceType { HCCL_INLINE_REDUCE = 0, HCCL_TBE_REDUCE };

enum class HcclRtMemcpyKind {
    HCCL_RT_MEMCPY_KIND_HOST_TO_HOST = 0, /**< host to host */
    HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE,   /**< host to device */
    HCCL_RT_MEMCPY_KIND_DEVICE_TO_HOST,   /**< device to host */
    HCCL_RT_MEMCPY_KIND_DEVICE_TO_DEVICE, /**< device to device */
    HCCL_RT_MEMCPY_ADDR_DEVICE_TO_DEVICE, /**< Level-2 address copy, device to device */
    HCCL_RT_MEMCPY_KIND_RESERVED,
};

enum class HcclRtDeviceModuleType {
    HCCL_RT_MODULE_TYPE_SYSTEM = 0,  /**< system info*/
    HCCL_RT_MODULE_TYPE_AICORE,      /**< AI CORE info*/
    HCCL_RT_MODULE_TYPE_VECTOR_CORE, /**< VECTOR CORE info*/
    HCCL_RT_DEVICE_MOUDLE_RESERVED,
};

enum class HcclRtDeviceInfoType {
    HCCL_INFO_TYPE_CORE_NUM,
    HCCL_INFO_TYPE_PHY_CHIP_ID,
    HCCL_INFO_TYPE_SDID,
    HCCL_INFO_TYPE_SERVER_ID,
    HCCL_INFO_TYPE_SUPER_POD_ID,
    HCCL_INFO_TYPE_CUST_OP_ENHANCE,
    HCCL_RT_DEVICE_INFO_RESERVED,
};

enum class HcclRtStreamClearStep {
    HCCL_STREAM_STOP = 0,
    HCCL_STREAM_CLEAR,
};

#ifdef __cplusplus
extern "C" {
#endif
HcclResult hrtGetDevice(s32* deviceLogicId);
HcclResult hrtGetDeviceRefresh(s32* deviceLogicId);
HcclResult hrtGetDeviceCount(u32* count);
HcclResult
hrtGetDeviceInfo(u32 deviceId, HcclRtDeviceModuleType hcclModuleType, HcclRtDeviceInfoType hcclInfoType, s64& val);
HcclResult hrtGetDeviceType(DevType& devType);
HcclResult hrtSetlocalDeviceType(DevType devType);
HcclResult hrtSetlocalDevice(s32 deviceLogicId);
HcclResult hrtSetLocalDeviceSatMode(aclrtFloatOverflowMode floatOverflowMode);
HcclResult hrtSetWorkModeAicpu(bool workModeAicpu);
HcclResult hrtGetDeviceSatMode(aclrtFloatOverflowMode* floatOverflowMode);
HcclResult hrtGetDevicePhyIdByIndex(u32 deviceLogicId, u32& devicePhyId, bool isRefresh = false);
HcclResult hrtGetDeviceIndexByPhyId(u32 devicePhyId, u32& deviceLogicId);
HcclResult hrtGetLogicDevIdByUserDevId(s32 userDevId, s32& logicDevId);
HcclResult hrtGetPairDevicePhyId(u32 localDevPhyId, u32& pairDevPhyId);
HcclResult PrintMemoryAttr(const void* memAddr);
HcclResult hrtCtxGetOverflowAddr(void** overflowAddr);
HcclResult hrtGetDeviceTypeBySocVersion(std::string& socVersion, DevType& devType);

HcclResult hrtEventDestroy(HcclRtEvent event);
HcclResult hrtMalloc(void** devPtr, u64 size, bool level2Address = false);
HcclResult hrtFree(void* devPtr);
HcclResult hrtMemSet(void* dst, uint64_t destMax, uint64_t count);
HcclResult hrtMemSyncCopy(void* dst, uint64_t destMax, const void* src, uint64_t count, HcclRtMemcpyKind kind);
HcclResult
hrtMemAsyncCopy(void* dst, uint64_t destMax, const void* src, uint64_t count, HcclRtMemcpyKind kind, rtStream_t stream);
HcclResult hrtMemcpyAddrAsync(
    void* dst, uint64_t destMax, uint64_t destOffset, const void* src, uint64_t count, uint64_t srcOffset,
    rtStream_t stream);

#if T_DESC("RtsTaskCallBack", true)
HcclResult hrtSubscribeReport(u64 threadId, rtStream_t& stream);
HcclResult hrtProcessReport(s32 timeout);
HcclResult hrtTaskAbortHandleCallback(aclrtDeviceTaskAbortCallback callback, void* args);
HcclResult hrtResourceClean();
HcclResult hrtGetHccsPortNum(u32 deviceLogicId, s32& num);
#endif

HcclResult hrtGetTaskIdAndStreamID(u32& taskId, u32& streamId);

#if T_DESC("RtsTaskExceptionHandler", true)
HcclResult hrtRegTaskFailCallbackByModule(aclrtExceptionInfoCallback callback);
HcclResult hrtGetStreamAvailableNum(u32& maxStrCount);
#endif

#ifdef __cplusplus
} // extern "C"
#endif
#endif
#endif
