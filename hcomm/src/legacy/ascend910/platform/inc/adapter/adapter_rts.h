/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_INC_ADAPTER_RTS_H
#define HCCL_INC_ADAPTER_RTS_H

#include <hccl/hccl_types.h>

#include "adapter_rts_common.h"
#include "acl/acl_rt.h"
#include "rt_external.h"
#include "hccl/base.h"
#include "private_types.h"

#if T_DESC("test", true)
#if T_DESC("Device管理", true)

// 临时定义在此 依赖rt包
#define ACL_STREAM_DEVICE_USE_ONLY 0x00000020U

using rtGetPairPhyDevicesInfoPtr = rtError_t (*)(uint32_t, uint32_t, int32_t, int64_t*);

HcclResult hrtCtxCreate(aclrtContext* createCtx, uint32_t flags, int32_t devId);
HcclResult hrtCtxDestroy(aclrtContext destroyCtx);
HcclResult hrtDeviceGetBareTgid(s32* pid);

HcclResult hrtGetPairDevicesInfo(u32 phyDevId, u32 otherPhyDevId, s32 infoType, s64* pValue);
HcclResult hrtGetPairPhyDevicesInfo(
    u32 phyDevId, u32 otherPhyDevId, s32 infoType, s64* pValue, rtGetPairPhyDevicesInfoPtr funcPtr);
HcclResult hrtGetPhyDeviceInfo(u32 devicePhysicId, s32 moduleType, s32 infoType, s64& value);
HcclResult hrtGetPairDeviceLinkTypeRaw(u32 phyDevId, u32 otherPhyDevId, s32 infoType, s64* pValue);

#ifdef __cplusplus
extern "C" {
#endif
HcclResult hrtMemSyncCopyEx(void* dst, uint64_t destMax, const void* src, uint64_t count, HcclRtMemcpyKind kind);

#ifdef __cplusplus
} // extern "C"
#endif
constexpr u32 CHIP_VERSION_MAX_LEN = 32;
constexpr s32 HCCL_EXEC_TIME_OUT_OFFSET_S = 5; // 避免与notifywait timeout时间冲突，增加5s的偏移值
constexpr u32 NOTIFY_SIZE_BYTES_910A = 8;      // 910A每个notify占用的字节数
constexpr u32 NOTIFY_SIZE_BYTES_910B = 4;      // 910B/910_93每个notify占用的字节数

HcclResult hrtGetSocVer(std::string& socName);

HcclResult stubSetDevice(u32 deviceLogicId);
#endif

#if T_DESC("DeviceMemory管理", true)
HcclResult MemcpyKindTranslate(HcclRtMemcpyKind kind, aclrtMemcpyKind* rtKind);
HcclResult hrtMemAsyncCopyWithoutCheckKind(
    void* dst, uint64_t destMax, const void* src, uint64_t count, HcclRtMemcpyKind kind, rtStream_t stream);
HcclResult hrtGetPointAttr(HcclRtPointAttr ptrAttr, const void* ptr);
HcclResult hrtIpcSetMemoryName(void* ptr, u8* name, u64 ptrMaxLen = INVALID_U64, u32 nameMaxLen = INVALID_UINT);
HcclResult hrtIpcDestroyMemoryName(const u8* name);
HcclResult hrtIpcSetMemoryAttr(const u8* name, aclrtIpcMemAttrType type, u64 attr);
HcclResult hrtIpcOpenMemory(void** ptr, const u8* name);
HcclResult hrtIpcSetMemoryPid(const u8* name, int pid[], int num);
// 该接口仅用于910_93超节点模式
HcclResult hrtSetIpcMemorySuperPodPid(const u8* name, s32 peerSdid, s32 peerPid[], s32 pidNum);
HcclResult hrtDevMemAlignWithPage(void*& ptr, u64& size);
#endif

#if T_DESC("host memory管理", true)
HcclResult hrtMallocHost(void** hostPtr, u64 size);
HcclResult hrtFreeHost(void* hostPtr);

HcclResult HrtDevFree(void* devPtr);
HcclResult HrtDevMalloc(void** devPtr, u64 size);

#endif

#if T_DESC("stream管理", true)
HcclResult hrtStreamDestroy(rtStream_t stream);
HcclResult hrtStreamCreate(aclrtStream* stream);
HcclResult hrtStreamCreateWithFlags(aclrtStream* stream, int32_t priority, uint32_t flags);
s32 GetMsTimeFromExecTimeout(s32 execTimeOut);
#endif

#if T_DESC("event 同步机制", true)
HcclResult hrtEventCreate(aclrtEvent* event);
HcclResult hrtEventRecord(aclrtEvent event, aclrtStream stream);
HcclResult hrtStreamWaitEvent(aclrtStream stream, aclrtEvent event);
HcclResult hrtEventQuery(aclrtEvent event);

HcclResult hrtGetNotifySize(u32& notifySize);
HcclResult hrtNotifyGetOffset(HcclRtNotify notify, u64& offset);
HcclResult hrtNotifyWaitWithTimeOut(rtNotify_t notify, rtStream_t stream, uint32_t timeOut);
HcclResult hrtNotifyRecord(rtNotify_t notify, rtStream_t stream);
HcclResult hrtNotifyDestroy(rtNotify_t notify);
HcclResult hrtNotifyCreate(s32 deviceId, aclrtNotify* notify);
HcclResult hrtNotifyGetPhyInfoExt(rtNotify_t notify, rtNotifyPhyInfo* notifyInfo);
#endif

#if T_DESC("EnableP2P", true)

typedef enum tagRtPhyDeviceInfoType { RT_PHY_INFO_TYPE_CHIPTYPE = 0, RT_PHY_INFO_TYPE_MASTER_ID } rtPhyDeviceInfoType_t;

HcclResult hrtEnableP2P(u32 deviceLogicId, u32 devicePhyId);
HcclResult hrtDisableP2P(u32 deviceLogicId, u32 devicePhyId);
HcclResult hrtGetP2PStatus(u32 deviceLogicId, u32 devicePhyId, uint32_t* status);

#endif
#if T_DESC("RtsTaskCallBack", true)
HcclResult hrtUnSubscribeReport(uint64_t threadId, aclrtStream& stream);
#endif

HcclResult hrtSetIpcNotifyPid(aclrtNotify notify, int32_t pid[], int num);
// 该接口仅用于910_93超节点模式
HcclResult hrtSetIpcNotifySuperPodPid(aclrtNotify notify, s32 peerSdid, s32 peerPid[], s32 pidNum);
HcclResult hrtIpcOpenNotify(aclrtNotify* notify, const u8* name);
HcclResult hrtReduceAsync(
    void* dst, uint64_t destMax, const void* src, uint64_t count, aclrtReduceKind kind, aclDataType type,
    aclrtStream stream);
HcclResult hrtRDMASend(u32 qpn, u32 wqe_index, rtStream_t stream);
HcclResult hrtCallbackLaunch(aclrtCallback callBackFunc, void* fnData, aclrtStream stream, bool isBlock);
HcclResult hrtIpcSetNotifyName(aclrtNotify notify, u8* name, uint32_t len);
HcclResult hrtMemcpy(void* dst, uint64_t destMax, const void* src, uint64_t count, HcclRtMemcpyKind kind);
HcclResult hrtMemcpyEx(void* dst, uint64_t destMax, void* src, uint64_t count, rtMemcpyKind_t kind);
HcclResult hrtKernelLaunchWithFlagV2(
    const void* stubFunc, uint32_t numBlocks, rtArgsEx_t* argsInfo, rtSmDesc_t* smDesc, rtStream_t stream,
    uint32_t flags, const rtTaskCfgInfo_t* cfgInfo);
HcclResult hrtRDMADBSend(uint32_t dbindex, uint64_t dbinfo, rtStream_t stream);

HcclResult hrtNotifyGetAddr(rtNotify_t signal, u64* notifyAddr);

HcclResult hrtGetRdmaDoorbellAddr(u32 dbIndex, u64& dbAddr);
HcclResult hrtGetDevArgsAddr(rtStream_t stm, rtArgsEx_t* argsInfo, void** devArgsAddr, void** argsHandle);
HcclResult hrtNotifyCreateWithFlag(int32_t deviceId, aclrtNotify* notify);
HcclResult hrtIpcOpenNotifyWithFlag(rtNotify_t* notify, const u8* name, uint32_t flags);
HcclResult hrtNotifyImportByKey(rtNotify_t* notify, const u8* name);

HcclResult hrtStreamGetSqid(const rtStream_t stm, uint32_t* sqId);
HcclResult hrtStreamGetCqid(const rtStream_t stm, uint32_t* cqId, uint32_t* logicCqId);
HcclResult hrtTaskAbortHandleCallback(aclrtDeviceTaskAbortCallback callback, void* args);
HcclResult hrtResourceClean();
HcclResult hrtGetHccsPortNum(u32 deviceLogicId, s32& num);

HcclResult hrtGetHcclV2Support(bool* isSupport);
#endif
#endif
