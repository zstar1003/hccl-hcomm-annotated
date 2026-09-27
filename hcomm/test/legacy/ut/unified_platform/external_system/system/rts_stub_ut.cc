/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "rt_external.h"
#include "acl/acl_rt.h"
#include <mutex>

aclError aclrtDeviceGetBareTgid(int32_t* pid) { return RT_ERROR_NONE; }

const char* aclrtGetSocName() { return "Ascend950PR"; }

aclError aclrtGetDeviceCount(uint32_t* count) { return RT_ERROR_NONE; }

aclError aclrtGetStreamAttribute(aclrtStream stream, aclrtStreamAttr stmAttrType, aclrtStreamAttrValue* value)
{
    return ACL_SUCCESS;
}

int rtModelFake = 0;
aclError aclmdlRICaptureGetInfo(aclrtStream stream, aclmdlRICaptureStatus* status, aclmdlRI* modelRI)
{
    *modelRI = &rtModelFake;
    return ACL_SUCCESS;
}

aclError aclmdlRICaptureThreadExchangeMode(aclmdlRICaptureMode* mode) { return ACL_SUCCESS; }

aclError aclrtGetDeviceInfo(uint32_t deviceId, aclrtDevAttr attr, int64_t* value) { return ACL_SUCCESS; }

aclError aclrtSetDeviceTaskAbortCallback(const char* regName, aclrtDeviceTaskAbortCallback callback, void* args)
{
    return ACL_SUCCESS;
}

aclError aclrtCreateEventWithFlag(aclrtEvent* event, uint32_t flag) { return aclrtCreateEvent(event); }

aclError aclrtNotifyImportByKey(aclrtNotify* notify, const char* name, uint64_t flag) { return ACL_SUCCESS; }

aclError aclrtSetStreamAttribute(aclrtStream stream, aclrtStreamAttr stmAttrType, aclrtStreamAttrValue* value)
{
    return ACL_SUCCESS;
}

aclError aclrtMemcpy(void* dst, size_t destMax, const void* src, size_t count, aclrtMemcpyKind kind)
{
    return ACL_SUCCESS;
}

aclError aclrtPointerGetAttributes(const void* ptr, aclrtPtrAttributes* attributes)
{
    // 桩函数固定反回2M的页表大小
    attributes->pageSize = 1;
    attributes->location.type = aclrtMemLocationType::ACL_MEM_LOCATION_TYPE_HOST;
    return ACL_SUCCESS;
}

aclError aclrtCreateNotify(aclrtNotify* notify, uint64_t flag) { return ACL_SUCCESS; }

aclError aclrtNotifySetImportPid(aclrtNotify notify, int32_t* pid, size_t num) { return ACL_SUCCESS; }

aclError
aclrtMemcpyAsync(void* dst, size_t destMax, const void* src, size_t count, aclrtMemcpyKind kind, aclrtStream stream)
{
    return ACL_SUCCESS;
}

aclError aclrtReduceAsync(
    void* dst, const void* src, uint64_t count, aclrtReduceKind kind, aclDataType type, aclrtStream stream,
    void* reserve)
{
    return ACL_SUCCESS;
}

aclError aclrtCntNotifyCreate(aclrtCntNotify* const cntNotify, uint64_t flags) { return ACL_SUCCESS; }

aclError aclrtCntNotifyGetId(aclrtCntNotify cntNotify, uint32_t* notifyId) { return ACL_SUCCESS; }

aclError aclrtCntNotifyRecord(aclrtCntNotify cntNotify, aclrtStream stream, aclrtCntNotifyRecordInfo* info)
{
    return ACL_SUCCESS;
}

aclError aclrtCntNotifyWaitWithTimeout(aclrtCntNotify cntNotify, aclrtStream stream, aclrtCntNotifyWaitInfo* info)
{
    return ACL_SUCCESS;
}

aclError aclrtCreateStream(aclrtStream* stream)
{
    *stream = reinterpret_cast<void*>(0x12345678);
    return ACL_SUCCESS;
}

aclError aclrtCreateEvent(aclrtEvent* event) { return ACL_SUCCESS; }

aclError aclrtGetDevice(int32_t* device)
{
    *device = 0;
    return RT_ERROR_NONE;
}

rtError_t rtResetDevice(int32_t devId) { return RT_ERROR_NONE; }

rtError_t aclrtSetDevice(int32_t device) { return RT_ERROR_NONE; }

aclError aclrtResetDevice(int32_t deviceId) { return ACL_SUCCESS; }

rtError_t rtGetDeviceCount(int32_t* count) { return RT_ERROR_NONE; }

rtError_t rtGetDeviceIndexByPhyId(uint32_t phyId, uint32_t* devIndex) { return RT_ERROR_NONE; }

aclError aclrtGetPhyDevIdByLogicDevId(const int32_t logicDevId, int32_t* const phyDevId) { return RT_ERROR_NONE; }

aclError aclrtGetUserDevIdByPhyDevId(const int32_t phyDevId, int32_t* const userDevId) { return RT_ERROR_NONE; }
aclError aclrtGetPhyDevIdByUserDevId(const int32_t userDevId, int32_t* const phyDevId) { return RT_ERROR_NONE; }

rtError_t rtGetPhyDeviceInfo(uint32_t phyId, int32_t moduleType, int32_t infoType, int64_t* val)
{
    return RT_ERROR_NONE;
}

rtError_t rtGetVisibleDeviceIdByLogicDeviceId(const int32_t logicDeviceId, int32_t* const visibleDeviceId)
{
    return RT_ERROR_NONE;
}

rtError_t rtGetPairDevicesInfo(uint32_t devId, uint32_t otherDevId, int32_t infoType, int64_t* val)
{
    return RT_ERROR_NONE;
}

rtError_t rtGetDeviceMode(rtDeviceMode* deviceMode) { return RT_ERROR_NONE; }

rtError_t rtEnableP2P(uint32_t devIdDes, uint32_t phyIdSrc, uint32_t flag) { return RT_ERROR_NONE; }

rtError_t rtDisableP2P(uint32_t devIdDes, uint32_t phyIdSrc) { return RT_ERROR_NONE; }

rtError_t rtGetP2PStatus(uint32_t devIdDes, uint32_t phyIdSrc, uint32_t* status) { return RT_ERROR_NONE; }

rtError_t aclrtFree(void* devPtr) { return RT_ERROR_NONE; }

rtError_t aclrtMallocHost(void** hostPtr, size_t size) { return RT_ERROR_NONE; }

aclError aclrtMallocWithCfg(void** devPtr, size_t size, aclrtMemMallocPolicy policy, aclrtMallocConfig* cfg)
{
    return ACL_SUCCESS;
}

aclError aclrtIpcMemClose(const char_t* name) { return RT_ERROR_NONE; }

aclError aclrtIpcMemGetExportKey(void* devPtr, size_t size, char* key, size_t len, uint64_t flag)
{
    return ACL_SUCCESS;
}

aclError aclrtIpcMemImportByKey(void** devPtr, const char* key, uint64_t flag) { return ACL_SUCCESS; }

aclError aclrtIpcMemSetImportPid(const char* key, int32_t* pid, size_t num) { return ACL_SUCCESS; }

aclError aclrtIpcMemSetAttr(const char* key, aclrtIpcMemAttrType type, uint64_t attr) { return ACL_SUCCESS; }

aclError aclrtMallocHostWithCfg(void** ptr, uint64_t size, aclrtMallocConfig* cfg) { return ACL_SUCCESS; }

rtError_t aclrtFreeHost(void* hostPtr) { return RT_ERROR_NONE; }

rtError_t rtMemcpy(void* dst, uint64_t destMax, const void* src, uint64_t count, rtMemcpyKind_t kind)
{
    return RT_ERROR_NONE;
}

aclError aclrtMemset(void* devPtr, size_t maxCount, int32_t value, size_t count) { return ACL_SUCCESS; }

aclError aclrtStreamGetId(aclrtStream stream, int32_t* streamId) { return ACL_SUCCESS; }

rtError_t rtStreamCreate(rtStream_t* stream, int32_t priority) { return RT_ERROR_NONE; }

aclError aclrtActiveStream(aclrtStream activeStream, aclrtStream stream) { return ACL_SUCCESS; }

rtError_t rtGetTaskIdAndStreamID(uint32_t* taskId, uint32_t* streamId) { return RT_ERROR_NONE; }

rtError_t rtStreamSwitchEx(
    void* ptr, rtCondition_t condition, void* valuePtr, rtStream_t trueStream, rtStream_t stm,
    rtSwitchDataType_t dataType)
{
    return RT_ERROR_NONE;
}

rtError_t aclrtSynchronizeStreamWithTimeout(rtStream_t stream, int32_t timeout) { return RT_ERROR_NONE; }

rtError_t rtGetMaxStreamAndTask(uint32_t streamType, uint32_t* maxStrCount, uint32_t* maxTaskCount)
{
    return RT_ERROR_NONE;
}

aclError aclrtSubscribeReport(uint64_t threadId, aclrtStream stream) { return ACL_SUCCESS; }

aclError aclrtUnSubscribeReport(uint64_t threadId, aclrtStream stream) { return ACL_SUCCESS; }

aclError aclrtProcessReport(int32_t timeout) { return ACL_SUCCESS; }

aclError aclrtLaunchCallback(aclrtCallback fn, void* userData, aclrtCallbackBlockType blockType, aclrtStream stream)
{
    return ACL_SUCCESS;
}

rtError_t rtNotifyCreate(int32_t device, rtNotify_t* notify) { return RT_ERROR_NONE; }

aclError aclrtDestroyNotify(aclrtNotify notify) { return RT_ERROR_NONE; }

aclError aclrtNotifyGetExportKey(aclrtNotify notify, char* key, size_t len, uint64_t flag) { return ACL_SUCCESS; }

aclError aclrtGetNotifyId(aclrtNotify notify, uint32_t* notifyId) { return ACL_SUCCESS; }

rtError_t rtSetIpcNotifyPid(const char* name, int32_t pid[], int num) { return RT_ERROR_NONE; }

rtError_t rtIpcOpenNotify(rtNotify_t* notify, const char_t* name) { return ACL_SUCCESS; }

rtError_t rtNotifyGetAddrOffset(rtNotify_t notify, uint64_t* devAddrOffset) { return RT_ERROR_NONE; }

aclError aclrtRecordNotify(aclrtNotify notify, aclrtStream stream) { return ACL_SUCCESS; }

rtError_t rtNotifyWait(rtNotify_t notify, rtStream_t stream) { return RT_ERROR_NONE; }

aclError aclrtWaitAndResetNotify(aclrtNotify notify, aclrtStream stream, uint32_t timeout) { return ACL_SUCCESS; }

rtError_t rtMemcpyAsync(void* dst, uint64_t destMax, const void* src, uint64_t cnt, rtMemcpyKind_t kind, rtStream_t stm)
{
    return RT_ERROR_NONE;
}

rtError_t rtMemcpyAsyncWithCfgV2(
    void* dst, uint64_t destMax, const void* src, uint64_t cnt, rtMemcpyKind_t kind, rtStream_t stm,
    const rtTaskCfgInfo_t* cfgInfo)
{
    return RT_ERROR_NONE;
}

rtError_t rtRDMASend(uint32_t sqIndex, uint32_t wqeIndex, rtStream_t stm) { return RT_ERROR_NONE; }

rtError_t rtRDMADBSend(uint32_t dbIndex, uint64_t dbInfo, rtStream_t stm) { return RT_ERROR_NONE; }

rtError_t rtGetDeviceSatMode(rtFloatOverflowMode_t* floatOverflowMode) { return RT_ERROR_NONE; }

aclError aclrtLaunchKernelWithHostArgs(
    const void* stubFunc, uint32_t numBlocks, rtArgsEx_t* argsInfo, rtSmDesc_t* smDesc, rtStream_t stream,
    uint32_t flags, const rtTaskCfgInfo_t* cfgInfo)
{
    return RT_ERROR_NONE;
}

aclError aclrtBinaryGetFunctionByEntry(aclrtBinHandle binHandle, uint64_t funcEntry, aclrtFuncHandle* funcHandle)
{
    return ACL_SUCCESS;
}

rtError_t rtGetNotifyAddress(rtNotify_t notify, uint64_t* const notifyAddres) { return RT_ERROR_NONE; }

rtError_t rtKernelLaunch(
    const void* stubFunc, uint32_t numBlocks, void* args, uint32_t argsSize, rtSmDesc_t* smDesc, rtStream_t stm)
{
    return RT_ERROR_NONE;
}

rtError_t rtAicpuKernelLaunch(
    const rtKernelLaunchNames_t* launchNames, uint32_t numBlocks, const void* args, uint32_t argsSize,
    rtSmDesc_t* smDesc, rtStream_t stm)
{
    return RT_ERROR_NONE;
}

rtError_t aclrtStreamGetId(rtStream_t const stm, uint64_t* const stmMode) { return RT_ERROR_NONE; }

aclError aclrtDestroyStreamForce(aclrtStream stream) { return ACL_SUCCESS; }

rtError_t rtCntNotifyCreate(const int32_t deviceId, aclrtCntNotify* const cntNotify) { return RT_ERROR_NONE; }

rtError_t rtGetCntNotifyId(aclrtCntNotify inCntNotify, uint32_t* const notifyId) { return RT_ERROR_NONE; }

aclError aclrtCntNotifyDestroy(aclrtCntNotify cntNotify) { return ACL_SUCCESS; }

// rtError_t rtCntNotifyRecord(
//     rtCntNotify_t const inCntNotify, rtStream_t const stm, const rtCntNtyRecordInfo_t *const info)
// {
//     return RT_ERROR_NONE;
// }

// rtError_t rtCntNotifyWaitWithTimeout(
//     rtCntNotify_t const inCntNotify, rtStream_t const stm, const rtCntNtyWaitInfo_t *const info)
// {
//     return RT_ERROR_NONE;
// }

rtError_t rtCCULaunch(rtCcuTaskInfo_t* taskInfo, rtStream_t const stm) { return RT_ERROR_NONE; }

rtError_t rtUbDevQueryInfo(rtUbDevQueryCmd cmd, void* devInfo) { return RT_ERROR_NONE; }

rtError_t rtStreamGetSqid(const rtStream_t stm, uint32_t* sqId) { return RT_ERROR_NONE; }

rtError_t rtStreamGetCqid(const rtStream_t stm, uint32_t* cqId, uint32_t* logicCqId) { return RT_ERROR_NONE; }

aclError aclrtExceptionInfoCallbackRegister(aclrtExceptionInfoCallback callback) { return ACL_SUCCESS; }
aclError aclrtExceptionInfoCallbackUnregister(aclrtExceptionInfoCallback callback) { return ACL_SUCCESS; }

rtError_t rtNotifyCreateWithFlag(int32_t deviceId, rtNotify_t* notify, uint32_t flag) { return RT_ERROR_NONE; }

rtError_t rtIpcOpenNotifyWithFlag(rtNotify_t* notify, const char_t* name, uint32_t flag) { return RT_ERROR_NONE; }

rtError_t rtGetDevResAddress(rtDevResInfo* const resInfo, rtDevResAddrInfo* const addrInfo) { return RT_ERROR_NONE; }

rtError_t rtReleaseDevResAddress(rtDevResInfo* const resInfo) { return RT_ERROR_NONE; }

rtError_t rtEventCreateWithFlag(rtEvent_t* evt, uint32_t flag) { return RT_ERROR_NONE; }

rtError_t aclrtDestroyEvent(rtEvent_t evt) { return RT_ERROR_NONE; }

aclError aclrtRecordEvent(aclrtEvent event, aclrtStream stream) { return ACL_SUCCESS; }

aclError aclrtQueryEventWaitStatus(aclrtEvent event, aclrtEventWaitStatus* status) { return ACL_SUCCESS; }

rtError_t rtStreamAddToModel(rtStream_t stm, rtModel_t captureMdl) { return RT_ERROR_NONE; }

rtError_t rtModelGetId(rtModel_t mdl, uint32_t* modelId) { return RT_ERROR_NONE; }

aclError aclrtGetResInCurrentThread(aclrtDevResLimitType type, uint32_t* value)
{
    *value = 48;
    return ACL_SUCCESS;
}

aclError aclrtSetCurrentContext(aclrtContext ctx) { return ACL_SUCCESS; }
aclError aclrtGetCurrentContext(aclrtContext* ctx)
{
    void* tmp;
    *ctx = tmp;
    return ACL_SUCCESS;
}

aclError aclrtCreateStreamWithConfig(aclrtStream* stream, uint32_t priority, uint32_t flag)
{
    return aclrtCreateStream(stream);
}

aclError aclrtBinaryGetFunction(const aclrtBinHandle binHandle, const char* kernelName, aclrtFuncHandle* funcHandle)
{
    return ACL_SUCCESS;
}

aclError aclrtBinaryLoadFromFile(const char* binPath, aclrtBinaryLoadOptions* options, aclrtBinHandle* binHandle)
{
    return ACL_SUCCESS;
}

aclError aclrtLaunchKernelWithHostArgs(
    aclrtFuncHandle funcHandle, uint32_t numBlocks, aclrtStream stream, aclrtLaunchKernelCfg* cfg, void* hostArgs,
    size_t argsSize, aclrtPlaceHolderInfo* placeHolderArray, size_t placeHolderNum)
{
    return ACL_SUCCESS;
}

aclError aclrtGetOpTimeOutInterval(uint64_t* interval) { return ACL_SUCCESS; }

aclError aclrtBinaryUnLoad(aclrtBinHandle binHandle) { return ACL_SUCCESS; }

aclError aclrtMemP2PMap(void* devPtr, size_t size, int32_t dstDevId, uint64_t flags) { return ACL_SUCCESS; }

aclError aclrtGetLogicDevIdByUserDevId(const int32_t userDevid, int32_t* const logicDevId) { return ACL_SUCCESS; }

aclError aclrtCacheLastTaskExtendInfo(const char* tag, size_t tagLen) { return ACL_SUCCESS; }

aclError aclrtRegDeviceStateCallback(const char* regName, aclrtDeviceStateCallback callback, void* args)
{
    return ACL_SUCCESS;
}

namespace aicpu {
std::mutex g_sqeIdMtx;
constexpr uint32_t INITAL_SQE_ID = 0x80000000U;
uint32_t g_sqeId = INITAL_SQE_ID;
void GetSqeId(const uint32_t num, uint32_t& start, uint32_t& end)
{
    std::lock_guard<std::mutex> lk(g_sqeIdMtx);
    start = g_sqeId;
    g_sqeId += num;
    end = g_sqeId;
    if (start > end) {
        g_sqeId = INITAL_SQE_ID;
        start = g_sqeId;
        g_sqeId += num;
        if (start >= end) {
            g_sqeId = INITAL_SQE_ID;
            return;
        }
    }
    return;
}
} // namespace aicpu
