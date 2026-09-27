/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "rts_stub.h"
#include <string>
#include <cstring>
#include "FakeStreamMgr.h"
#include "../context/st_ctx.h"
#include "../fake_pub_stub.h"
/**
 * 设计思路：
 * 任务下发阶段：只接收task并存储
 * 任务执行阶段(外部触发stream的sync时)：轮询各个stream，执行未阻塞任务。
 * */

#ifdef __cplusplus
extern "C" {

FakeStreamMgr fakeStreamMgr;

// Notify related functions

rtError_t rtNotifyCreate(int32_t deviceId, rtNotify_t* notify)
{
    *notify = static_cast<void*>(fakeStreamMgr.GetFakeNotifyMgr()->CreateNotify(deviceId)); // deviceId is rankId in ST
    return 0;
}

aclError aclrtDestroyNotify(aclrtNotify notify)
{
    fakeStreamMgr.GetFakeNotifyMgr()->DestroyNotify(static_cast<int*>(notify));
    return 0;
}

aclError aclrtRecordNotify(aclrtNotify notify, aclrtStream stream)
{
    FakeSqe sqe{};
    sqe.type = FakeSqeType::NOTIFY_RECORD;
    sqe.notifyId = *(static_cast<int*>(notify));
    sqe.srcRank = fakeStreamMgr.GetRank(*(static_cast<int*>(stream)));
    sqe.dstRank = fakeStreamMgr.GetFakeNotifyMgr()->GetRank(sqe.notifyId);
    fakeStreamMgr.Append(*(static_cast<int*>(stream)), sqe);
    return 0;
}

rtError_t rtNotifyWait(rtNotify_t notify, rtStream_t stm)
{
    FakeSqe sqe{};
    sqe.type = FakeSqeType::NOTIFY_WAIT;
    sqe.notifyId = *(static_cast<int*>(notify));
    sqe.srcRank = fakeStreamMgr.GetRank(*(static_cast<int*>(stm)));
    fakeStreamMgr.Append(*(static_cast<int*>(stm)), sqe);
    return 0;
}

aclError aclrtWaitAndResetNotify(aclrtNotify notify, aclrtStream stream, uint32_t timeout)
{
    rtNotifyWait(notify, stream);
    return 0;
}

rtError_t rtIpcOpenNotify(rtNotify_t* notify, const char_t* name) { return 0; }

rtError_t rtNotifyGetAddrOffset(rtNotify_t notify, uint64_t* devAddrOffset) { return 0; }

aclError aclrtNotifyGetExportKey(aclrtNotify notify, char* key, size_t len, uint64_t flag) { return 0; }

aclError aclrtGetNotifyId(aclrtNotify notify, uint32_t* notifyId)
{
    *notifyId = *(static_cast<int*>(notify));
    return 0;
}

rtError_t rtSetIpcNotifyPid(const char_t* name, int32_t pid[], int32_t num) { return 0; }

rtError_t rtGetNotifyAddress(rtNotify_t notify, uint64_t* const notifyAddres) { return 0; }

rtError_t aclrtFree(void* devPtr)
{
    delete[] static_cast<char*>(devPtr);
    return 0;
}

aclError aclrtMallocHostWithCfg(void** hostPtr, uint64_t size, aclrtMallocConfig* cfg)
{
    *hostPtr = static_cast<void*>(new char[size]);
    return 0;
}

rtError_t aclrtFreeHost(void* hostPtr)
{
    aclrtFree(hostPtr);
    return 0;
}

rtError_t rtReduceAsync(
    void* dst, uint64_t destMax, const void* src, uint64_t cnt, rtRecudeKind_t kind, rtDataType_t type, rtStream_t stm)
{
    FakeSqe sqe{};
    sqe.type = FakeSqeType::SDMA_REDUCE;
    sqe.dst = dst;
    sqe.src = src;
    sqe.count = cnt;
    sqe.dataType = type;
    sqe.reduceOp = kind;
    sqe.srcRank = fakeStreamMgr.GetRank(*(static_cast<int*>(stm)));
    fakeStreamMgr.Append(*static_cast<int*>(stm), sqe);
    return 0;
}

rtError_t rtReduceAsyncV2(
    void* dst, uint64_t destMax, const void* src, uint64_t count, rtRecudeKind_t kind, rtDataType_t type,
    rtStream_t stm, void* overflowAddr)
{
    rtReduceAsync(dst, destMax, src, count, kind, type, stm);
    return 0;
}

rtError_t rtReduceAsyncWithCfgV2(
    void* dst, uint64_t destMax, const void* src, uint64_t cnt, rtRecudeKind_t kind, rtDataType_t type, rtStream_t stm,
    const rtTaskCfgInfo_t* cfgInfo)
{
    rtReduceAsync(dst, destMax, src, cnt, kind, type, stm);
    return 0;
}

rtError_t rtMemcpy(void* dst, uint64_t destMax, const void* src, uint64_t cnt, rtMemcpyKind_t kind)
{
    if (cnt > destMax) {
        return 1;
    }
    memcpy(dst, src, cnt);
    return 0;
}

rtError_t rtMemcpyAsync(void* dst, uint64_t destMax, const void* src, uint64_t cnt, rtMemcpyKind_t kind, rtStream_t stm)
{
    if (cnt > destMax) {
        return 1;
    }
    FakeSqe sqe{};
    sqe.type = FakeSqeType::MEM_CPY;
    sqe.dst = dst;
    sqe.src = src;
    sqe.count = cnt;
    sqe.srcRank = fakeStreamMgr.GetRank(*(static_cast<int*>(stm)));
    fakeStreamMgr.Append(*static_cast<int*>(stm), sqe);
    return 0;
}

rtError_t rtMemcpyAsyncWithCfgV2(
    void* dst, uint64_t destMax, const void* src, uint64_t cnt, rtMemcpyKind_t kind, rtStream_t stm,
    const rtTaskCfgInfo_t* cfgInfo)
{
    rtMemcpyAsync(dst, destMax, src, cnt, kind, stm);
    return 0;
}

aclError aclrtMemset(void* devPtr, size_t maxCount, int32_t value, size_t count)
{
    if (devPtr == nullptr) {
        return ACL_SUCCESS;
    }

    memset_s(devPtr, maxCount, value, count);
    return 0;
}

aclError aclrtIpcMemGetExportKey(void* devPtr, size_t size, char* key, size_t len, uint64_t flag) { return 0; }

aclError aclrtIpcMemSetImportPid(const char* key, int32_t* pid, size_t num) { return 0; }

aclError aclrtIpcMemClose(const char_t* name) { return RT_ERROR_NONE; }

aclError aclrtGetStreamAttribute(aclrtStream stream, aclrtStreamAttr stmAttrType, aclrtStreamAttrValue* value)
{
    return ACL_SUCCESS;
}

aclError aclrtCreateStreamWithConfig(aclrtStream* stream, uint32_t priority, uint32_t flag) { return ACL_SUCCESS; }

int rtModelFake = 0;
aclError aclmdlRICaptureGetInfo(aclrtStream stream, aclmdlRICaptureStatus* status, aclmdlRI* modelRI)
{
    *modelRI = &rtModelFake;
    return ACL_SUCCESS;
}

aclError aclmdlRICaptureThreadExchangeMode(aclmdlRICaptureMode* mode) { return ACL_SUCCESS; }

aclError aclrtGetDeviceInfo(uint32_t deviceId, aclrtDevAttr attr, int64_t* value) { return ACL_SUCCESS; }

aclError aclrtSetDeviceTaskAbortCallback(const char* regName, aclrtDeviceTaskAbortCallback callback, void* args);
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

typedef enum {
    RT_CNT_NOTIFY_WAIT_LESS_MODE = 0x0U,
    RT_CNT_NOTIFY_WAIT_EQUAL_MODE = 0x1U,
    RT_CNT_NOTIFY_WAIT_BIGGER_MODE = 0x2U,
    RT_CNT_NOTIFY_WAIT_BIGGER_OR_EQUAL_MODE = 0x3U,
    RT_CNT_NOTIFY_WAIT_EQUAL_WITH_BITMASK_MODE = 0x4U,
    RT_CNT_NOTIFY_WAIT_MODE_MAX
} rtCntNotifyWaitMode;
typedef struct {
    rtCntNotifyWaitMode mode;
    uint32_t value;
    uint32_t timeout;
    bool isClear;
    uint8_t rev[3U];
} rtCntNotifyWaitInfo_t;
aclError aclrtCntNotifyWaitWithTimeout(aclrtCntNotify cntNotify, aclrtStream stream, aclrtCntNotifyWaitInfo* info)
{
    return ACL_SUCCESS;
}

aclError aclrtCreateStream(aclrtStream* stream) { return ACL_SUCCESS; }

aclError aclrtCreateEvent(aclrtEvent* event) { return ACL_SUCCESS; }

aclError aclrtStreamGetId(aclrtStream stream, int32_t* streamId) { return ACL_SUCCESS; }

aclError aclrtGetPhyDevIdByLogicDevId(const int32_t logicDevId, int32_t* const phyDevId) { return RT_ERROR_NONE; }

aclError aclrtDeviceGetBareTgid(int32_t* pid) { return RT_ERROR_NONE; }

const char* aclrtGetSocName() { return "Ascend950"; }

aclError aclrtGetDeviceCount(uint32_t* count) { return RT_ERROR_NONE; }

rtError_t rtOpenNetService(const rtNetServiceOpenArgs* args) { return RT_ERROR_NONE; }

// device related functions
rtError_t rtGetDeviceIndexByPhyId(uint32_t phyId, uint32_t* devIndex) { return 0; }

rtError_t aclrtSetDevice(int32_t devId) { return 0; }

rtError_t rtGetPhyDeviceInfo(uint32_t phyId, int32_t moduleType, int32_t infoType, int64_t* val) { return 0; }

aclError aclrtResetDevice(int32_t deviceId) { return 0; }

aclError aclrtGetDevice(int32_t* devId)
{
    // *devId = GetCurrentThreadContext()->myRank;
    return 0;
}

aclError aclrtDestroyStreamForce(aclrtStream stream)
{
    fakeStreamMgr.DestroyStream(static_cast<int*>(stream));
    return 0;
}

rtError_t aclrtSynchronizeStreamWithTimeout(rtStream_t stm, int32_t timeout)
{
    fakeStreamMgr.Sync(*static_cast<int*>(stm));
    return 0;
}

rtError_t rtGetTaskIdAndStreamID(uint32_t* taskId, uint32_t* streamId) { return 0; }

aclError aclrtActiveStream(aclrtStream activeStream, aclrtStream stream) { return 0; }

// other functions
rtError_t rtEnableP2P(uint32_t devIdDes, uint32_t phyIdSrc, uint32_t flag) { return 0; }

rtError_t rtDisableP2P(uint32_t devIdDes, uint32_t phyIdSrc) { return 0; }

rtError_t rtDevBinaryUnRegister(void* hdl) { return 0; }

rtError_t rtGetDeviceCount(int32_t* cnt) { return 0; }

rtError_t rtAicpuKernelLaunch(
    const rtKernelLaunchNames_t* launchNames, uint32_t numBlocks, const void* args, uint32_t argsSize,
    rtSmDesc_t* smDesc, rtStream_t stm)
{
    return 0;
}

aclError aclrtIpcMemImportByKey(void** devPtr, const char* key, uint64_t flag) { return 0; }

rtError_t rtGetP2PStatus(uint32_t devIdDes, uint32_t phyIdSrc, uint32_t* status) { return 0; }

rtError_t rtGetVisibleDeviceIdByLogicDeviceId(const int32_t logicDeviceId, int32_t* const visibleDeviceId) { return 0; }

rtError_t rtRDMASend(uint32_t sqIndex, uint32_t wqeIndex, rtStream_t stm) { return 0; }

aclError aclrtBinaryGetFunctionByEntry(aclrtBinHandle binHandle, uint64_t funcEntry, aclrtFuncHandle* funcHandle)
{
    return 0;
}

rtError_t rtKernelLaunch(
    const void* stubFunc, uint32_t numBlocks, void* args, uint32_t argsSize, rtSmDesc_t* smDesc, rtStream_t stm)
{
    return 0;
}

rtError_t rtGetSocVersion(char_t* ver, const uint32_t maxLen)
{
    if (strlen(fakePubStubGetSocVersion()) >= 32) {
        throw std::exception();
    }
    strcpy(ver, fakePubStubGetSocVersion());
    return 0;
}

rtError_t rtGetDeviceSatMode(rtFloatOverflowMode_t* floatOverflowMode) { return 0; }

aclError aclrtPointerGetAttributes(aclrtPtrAttributes* attributes, const void* ptr) { return ACL_SUCCESS; }

rtError_t rtRDMADBSend(uint32_t dbIndex, uint64_t dbInfo, rtStream_t stm) { return 0; }

rtError_t rtCntNotifyCreate(const int32_t deviceId, aclrtCntNotify* const cntNotify) { return RT_ERROR_NONE; }

rtError_t rtGetCntNotifyId(aclrtCntNotify inCntNotify, uint32_t* const notifyId) { return RT_ERROR_NONE; }

aclError aclrtCntNotifyDestroy(aclrtCntNotify cntNotify) { return ACL_SUCCESS; }

// rtError_t rtCntNotifyRecord(rtCntNotify_t const inCntNotify, rtStream_t const stm,
//                             const rtCntNtyRecordInfo_t * const info)
// {
//     return RT_ERROR_NONE;
// }

// rtError_t rtCntNotifyWaitWithTimeout(rtCntNotify_t const inCntNotify, rtStream_t const stm,
//                                      const rtCntNtyWaitInfo_t * const info)
// {
//     return RT_ERROR_NONE;
// }

rtError_t rtStreamGetSqid(const rtStream_t stm, uint32_t* sqId) { return RT_ERROR_NONE; }

rtError_t rtStreamGetCqid(const rtStream_t stm, uint32_t* cqId, uint32_t* logicCqId) { return RT_ERROR_NONE; }

aclError aclrtExceptionInfoCallbackRegister(aclrtExceptionInfoCallback callback) { return ACL_SUCCESS; }
aclError aclrtExceptionInfoCallbackUnregister(aclrtExceptionInfoCallback callback) { return ACL_SUCCESS; }

rtError_t rtNotifyCreateWithFlag(int32_t deviceId, rtNotify_t* notify, uint32_t flag) { return RT_ERROR_NONE; }

rtError_t rtIpcOpenNotifyWithFlag(rtNotify_t* notify, const char_t* name, uint32_t flag) { return RT_ERROR_NONE; }

rtError_t rtCCULaunch(rtCcuTaskInfo_t* taskInfo, rtStream_t const stm) { return RT_ERROR_NONE; }

rtError_t rtUbDevQueryInfo(rtUbDevQueryCmd cmd, void* devInfo) { return RT_ERROR_NONE; }

rtError_t rtGetDevResAddress(rtDevResInfo* const resInfo, rtDevResAddrInfo* const addrInfo) { return RT_ERROR_NONE; }

rtError_t rtReleaseDevResAddress(rtDevResInfo* const resInfo) { return RT_ERROR_NONE; }

rtError_t rtEventCreateWithFlag(rtEvent_t* evt, uint32_t flag) { return RT_ERROR_NONE; }

rtError_t aclrtDestroyEvent(rtEvent_t evt) { return RT_ERROR_NONE; }

aclError aclrtRecordEvent(aclrtEvent event, aclrtStream stream) { return ACL_SUCCESS; }

aclError aclrtQueryEventWaitStatus(aclrtEvent event, aclrtEventWaitStatus* status) { return ACL_SUCCESS; }

rtError_t rtSetTaskAbortCallBack(const char* moduleName, rtTaskAbortCallBack callback, void* args)
{
    return RT_ERROR_NONE;
}

rtError_t rtStreamGetCaptureInfo(rtStream_t stm, rtStreamCaptureStatus* const status, rtModel_t* captureMdl)
{
    return RT_ERROR_NONE;
}

rtError_t rtStreamAddToModel(rtStream_t stm, rtModel_t captureMdl) { return RT_ERROR_NONE; }

rtError_t rtModelGetId(rtModel_t mdl, uint32_t* modelId) { return RT_ERROR_NONE; }
}

aclError aclrtLaunchKernelWithHostArgs(
    const void* stubFunc, uint32_t numBlocks, rtArgsEx_t* argsInfo, rtSmDesc_t* smDesc, rtStream_t stream,
    uint32_t flags, const rtTaskCfgInfo_t* cfgInfo)
{
    return ACL_SUCCESS;
}

aclError aclrtGetResInCurrentThread(aclrtDevResLimitType type, uint32_t* value)
{
    *value = 48;
    return ACL_SUCCESS;
}

aclError aclrtMemP2PMap(void* devPtr, size_t size, int32_t dstDevId, uint64_t flags) { return ACL_SUCCESS; }

aclError aclrtCacheLastTaskExtendInfo(const char* tag, size_t tagLen) { return ACL_SUCCESS; }

#endif
