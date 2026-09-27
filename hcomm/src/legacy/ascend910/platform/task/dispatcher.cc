/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dispatcher_pub.h"
#include "externalinput_pub.h"
#include "externalinput.h"
#include "adapter_rts.h"
#include "sal_pub.h"
#include "config_plf_log.h"
#include "hccl_tbe_task.h"
#ifndef HCCD
#include "graph_ctx_mgr_common.h"
#endif

using namespace hccl;

#if T_DESC("DispatcherPub", true)

namespace {
HcclResult g_callBackResult = HCCL_SUCCESS;
const std::map<HcclDataType, aclDataType> HCCL_RT_DATA_TYPE_MAP = {
    {HCCL_DATA_TYPE_INT8, ACL_INT8},    {HCCL_DATA_TYPE_INT16, ACL_INT16}, {HCCL_DATA_TYPE_INT32, ACL_INT32},
    {HCCL_DATA_TYPE_FP16, ACL_FLOAT16}, {HCCL_DATA_TYPE_FP32, ACL_FLOAT},  {HCCL_DATA_TYPE_BFP16, ACL_BF16},
};
const std::map<HcclReduceOp, aclrtReduceKind> HCCL_RT_REDUCE_OP_MAP = {
    {HCCL_REDUCE_SUM, ACL_RT_MEMCPY_SDMA_AUTOMATIC_SUM},
    {HCCL_REDUCE_MAX, ACL_RT_MEMCPY_SDMA_AUTOMATIC_MAX},
    {HCCL_REDUCE_MIN, ACL_RT_MEMCPY_SDMA_AUTOMATIC_MIN},
};
} // namespace

bool DispatcherPub::isForce_ = false;

DispatcherPub::DispatcherPub(const s32 deviceLogicId)
    : deviceLogicId_(deviceLogicId),
      notifyWaitMode_(SyncMode::DEFAULT_TIMEWAITSYNCMODE),
      hostNicTcpSendThreadState_(true),
      overflowAddr_(nullptr),
      setDeviceFlag_(false),
      execTimeOut_(NOTIFY_DEFAULT_WAIT_TIME),
      execTimeOutByConfig_(false)
{}

DispatcherPub::~DispatcherPub()
{
    HcclResult ret = HCCL_SUCCESS;
#ifndef HCCD
    std::map<int32_t, void*>::iterator devMemIter;
    std::unique_lock<std::mutex> lock(devMemMutex_);
    for (devMemIter = devMemMap_.begin(); devMemIter != devMemMap_.end(); devMemIter++) {
        if (devMemIter->second != nullptr) {
            if (hrtFree(devMemIter->second) != HCCL_SUCCESS) {
                HCCL_WARNING("free device memory failed");
            }
            devMemIter->second = nullptr;
        }
    }
    devMemMap_.clear();
    if (deviceLogicId_ != HOST_DEVICE_ID) {
        ret = HcclTbeTaskDeInit(deviceLogicId_);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("tbe task deinit failed. ret[%d] device id[%d]", ret, deviceLogicId_);
        }
    }
    lock.unlock();

    if (fftsPubInfo_ != nullptr) {
        GraphMgrDeInit(fftsPubInfo_);
        fftsPubInfo_ = nullptr;
    }
#endif

    // 清空task信息
    if (hostNicTcpSendThread_ != nullptr) {
        WaitHostNicTcpSendThreadComplete();
    }
    ClearHostNicRdmaParamsVec();
    ClearHostNicTcpSendParamsVec();
    ClearHostNicTcpRecvParamsVec();

    if (setDeviceFlag_) {
        ret = hrtResetDevice(deviceLogicId_);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR(
                "[DispatcherPub][Destroy]In dispatcher enhanced destruct, reset device failed.errno[%d] "
                "device id[%d]",
                ret, deviceLogicId_);
        }
    }
}

void DispatcherPub::JudgeOpBaseTcpSendComplete(bool& closeSendThreadFlag)
{
    bool hostNicTcpSendParamsVecIsEmpty = true;
    for (auto it = hostNicTcpSendParamsVec_.begin(); it != hostNicTcpSendParamsVec_.end(); it++) {
        if (it->second.size() != 0) {
            hostNicTcpSendParamsVecIsEmpty = false;
            HCCL_WARNING(
                "host nic TCP send task is not completed. streamID[%llu], size[%llu]", it->first, it->second.size());
        }
    }
    closeSendThreadFlag = (hostNicTcpSendThreadParam_ == nullptr) && hostNicTcpSendParamsVecIsEmpty;
}

void DispatcherPub::WaitHostNicTcpSendThreadComplete()
{
    // 等待tcp send线程join
    bool closeSendThreadFlag = true;
    while (true) {
        HcclWorkflowMode workflowMode = GetWorkflowMode();
        if (workflowMode == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
            JudgeOpBaseTcpSendComplete(closeSendThreadFlag);
        } else {
            closeSendThreadFlag = (hostNicTcpSendThreadParam_ == nullptr);
        }
        if (closeSendThreadFlag) {
            break;
        }
        HCCL_WARNING("host nic TCP send thread is not finished");
        SaluSleep(TCP_SEND_THREAD_SLEEP_TWO_HUNDRED_MICROSECOND);
    }
    hostNicTcpSendThreadState_ = false;
    if (hostNicTcpSendThread_ != nullptr && hostNicTcpSendThread_->joinable()) {
        hostNicTcpSendThread_->join(); // 等待线程执行完毕
    }
}

void DispatcherPub::ClearHostNicRdmaParamsVec()
{
    for (auto it = hostNicRdmaParamsVec_.begin(); it != hostNicRdmaParamsVec_.end(); it++) {
        if (it->second.size() != 0) {
            HCCL_WARNING(
                "host nic RDMA task is not completed. streamID[%llu], size[%llu]", it->first, it->second.size());
            while (!it->second.empty()) {
                it->second.pop();
            }
        }
    }
    hostNicRdmaParamsVec_.clear();
}

void DispatcherPub::ClearHostNicTcpSendParamsVec()
{
    for (auto it = hostNicTcpSendParamsVec_.begin(); it != hostNicTcpSendParamsVec_.end(); it++) {
        if (it->second.size() != 0) {
            HCCL_WARNING(
                "host nic TCP send task is not completed. streamID[%llu], size[%llu]", it->first, it->second.size());
            while (!it->second.empty()) {
                it->second.pop();
            }
        }
    }
    hostNicTcpSendParamsVec_.clear();
}

void DispatcherPub::ClearHostNicTcpRecvParamsVec()
{
    for (auto it = hostNicTcpRecvParamsVec_.begin(); it != hostNicTcpRecvParamsVec_.end(); it++) {
        if (it->second.size() != 0) {
            HCCL_WARNING(
                "host nic TCP recv task is not completed. streamID[%llu], size[%llu]", it->first, it->second.size());
            while (!it->second.empty()) {
                it->second.pop();
            }
        }
    }
    hostNicTcpRecvParamsVec_.clear();
}

void DispatcherPub::WaitHostNicTcpSendTaskDone()
{
    while (hostNicTcpSendThreadParam_ != nullptr) {
        SaluSleep(TCP_SEND_THREAD_SLEEP_TWO_HUNDRED_MICROSECOND);
    }
}

// 获取算子最大超时时间
HcclResult DispatcherPub::GetNotifyMaxWaitTime()
{
    DevType deviceType;
    CHK_RET(hrtGetDeviceType(deviceType));
    notifyMaxWaitTime_ = (deviceType == DevType::DEV_TYPE_910_93 || deviceType == DevType::DEV_TYPE_910B) ?
                             NOTIFY_MAX_WAIT_TIME_910_93 :
                             NOTIFY_MAX_WAIT_TIME;
    HCCL_INFO("[GetNotifyMaxWaitTime] notifyMaxWaitTime_ is %us", notifyMaxWaitTime_);
    return HCCL_SUCCESS;
}

s32 DispatcherPub::GetExecTimeOut() { return execTimeOut_; }
bool DispatcherPub::GetExecTimeOutSet() { return execTimeOutByConfig_; }

HcclResult DispatcherPub::Init()
{
#ifndef HCCD
    if (deviceLogicId_ == HOST_DEVICE_ID) {
        return HCCL_SUCCESS;
    }

    aclrtContext ctx = nullptr;
    CHK_RET(hrtCtxGetCurrent(&ctx));
    if (ctx == nullptr) {
        CHK_RET(hrtSetDevice(deviceLogicId_));
        setDeviceFlag_ = true;
    }

    CHK_RET(HcclTbeTaskInit(deviceLogicId_));

    fftsPubInfo_ = GraphMgrInit();
    CHK_PTR_NULL(fftsPubInfo_);

    if (GetExternalInputHcclExecTimeoutSet() != HcclExecTimeoutSet::HCCL_EXEC_TIMEOUT_NOT_SET || execTimeOutByConfig_) {
        notifyWaitMode_ = SyncMode::CONFIGURABLE_TIMEWAITSYNCMODE;
    }

    if (GetExternalInputHcclIsTcpMode()) {
        hostNicTcpSendThread_.reset(new (std::nothrow) std::thread(&DispatcherPub::HostNicTcpSendThreadTask, this));
    }

    CHK_RET(GetNotifyMaxWaitTime());
#else
    HCCL_ERROR("does not support this interface.");
    return HCCL_E_PARA;
#endif

    return HCCL_SUCCESS;
}

void DispatcherPub::SetupTaskParaDma(
    hccl::TaskPara& taskPara, hccl::TaskParaDMA& para, TaskType taskType, ProfilerType profilerType,
    hccl::Stream& stream, u64 beginTime, bool isMainStream) const
{
    taskPara.type = taskType;
    taskPara.profilerType = profilerType;
    taskPara.stream = stream.ptr();
    taskPara.beginTime = beginTime;
    taskPara.dma = para;
    taskPara.isMainStream = isMainStream;
}

void DispatcherPub::SetupTaskParaDma(
    hccl::TaskPara& taskPara, hccl::TaskParaDMA& para, TaskType taskType, HcclRtStream stream, u64 beginTime,
    bool isMainStream) const
{
    taskPara.type = taskType;
    taskPara.stream = stream;
    taskPara.beginTime = beginTime;
    taskPara.dma = para;
    taskPara.isMainStream = isMainStream;
}

HcclResult DispatcherPub::SignalRecord(
    HcclRtNotify signal, HcclRtStream stream, u32 userRank, u64 offset, s32 stage, bool isMainStream)
{
    uint64_t beginTime = GetMsprofSysCycleTime();
    CHK_RET(hrtNotifyRecord(static_cast<HcclRtNotify>(signal), stream));

    // 若没有输入offset， 则认为record的为本地notify，直接获取其offset
    u64 NotifyID = userRank;
    if (offset == INVALID_U64) {
        CHK_RET(hrtNotifyGetOffset(static_cast<HcclRtNotify>(signal), offset));
    }
    NotifyID = (NotifyID << 32) | (offset & 0x00000000FFFFFFFF); // 0x00000000FFFFFFFF用于取offset的低32位
    // 调用回调来保存task信息
    if (callback_ != nullptr) {
        hccl::TaskParaNotify para(NotifyID, stage);
        hccl::TaskPara taskPara(TaskType::TASK_NOTIFY_RECORD, para);
        taskPara.stream = stream;
        taskPara.beginTime = beginTime;
        taskPara.isMainStream = isMainStream;
        callback_(callBackUserPtr_, (void*)&taskPara, sizeof(struct TaskPara));
    }

    u32 taskID = 0;
    u32 streamID = 0;
    hrtGetTaskIdAndStreamID(taskID, streamID);
    PLF_CONFIG_INFO(
        PLF_TASK, "%s para: notifyId[0x%016llx] taskId[%u] streamID[%u] userRank[%u] offset[%llu] stage[%d]", __func__,
        NotifyID, taskID, streamID, userRank, offset, stage);
    return HCCL_SUCCESS;
}

u32 DispatcherPub::GetNotifyWaitTime(u32 timeOut)
{
    u32 notifyWaitTime = 0;
    if (timeOut > 0 && timeOut <= notifyMaxWaitTime_) {
        notifyWaitTime = timeOut;
    } else if (notifyWaitMode_ == SyncMode::CONFIGURABLE_TIMEWAITSYNCMODE) {
        notifyWaitTime = execTimeOut_;
    } else if (notifyWaitMode_ == SyncMode::UNLIMITED_TIMEWAITSYNCMODE) {
        notifyWaitTime = notifyMaxWaitTime_;
    } else {
        notifyWaitTime = NOTIFY_DEFAULT_WAIT_TIME;
    }
    return notifyWaitTime;
}

HcclResult DispatcherPub::SignalWait(
    HcclRtNotify signal, HcclRtStream stream, u32 userRank, u32 remoteUserRank, s32 stage, u32 timeOut,
    bool isMainStream)
{
    uint64_t beginTime = GetMsprofSysCycleTime();
    CHK_RET(hrtNotifyWaitWithTimeOut(static_cast<HcclRtNotify>(signal), stream, GetNotifyWaitTime(timeOut)));

    // 调用回调来保存task信息
    u64 NotifyID = userRank;
    u64 offset = 0;
    CHK_RET(hrtNotifyGetOffset(static_cast<HcclRtNotify>(signal), offset));

    NotifyID = (NotifyID << 32) | (offset & 0x00000000FFFFFFFF); // 0x00000000FFFFFFFF用于取offset的低32位
    if (callback_ != nullptr) {
        hccl::TaskParaNotify para(NotifyID, stage, remoteUserRank);
        hccl::TaskPara taskPara(TaskType::TASK_NOTIFY_WAIT, para);
        taskPara.stream = stream;
        taskPara.beginTime = beginTime;
        taskPara.isMainStream = isMainStream;
        callback_(callBackUserPtr_, (void*)&taskPara, sizeof(struct TaskPara));
    }

    u32 taskID = 0;
    u32 streamID = 0;
    hrtGetTaskIdAndStreamID(taskID, streamID);
    PLF_CONFIG_INFO(
        PLF_TASK,
        "%s para: notifyId[0x%016llx] taskId[%u] streamID[%u] userRank[%u] remoteUserRank[%u] stage[%d] timeOut[%u s]",
        __func__, NotifyID, taskID, streamID, userRank, remoteUserRank, stage, timeOut);
    return HCCL_SUCCESS;
}

HcclResult DispatcherPub::SetNotifyWaitMode(SyncMode notifyWaitMode)
{
    notifyWaitMode_ = notifyWaitMode;
    return HCCL_SUCCESS;
}

SyncMode DispatcherPub::GetNotifyWaitMode() { return notifyWaitMode_; }

HcclResult DispatcherPub::SetHcclExecTimeOut(s32 execTimeOut)
{
    execTimeOut_ = execTimeOut;
    execTimeOutByConfig_ = true;
    return HCCL_SUCCESS;
}

HcclResult
DispatcherPub::MemcpySync(void* dst, uint64_t destMax, const void* src, uint64_t count, HcclRtMemcpyKind kind)
{
    return hrtMemSyncCopy(dst, destMax, src, count, kind);
}

HcclResult DispatcherPub::MemcpyAsync(
    void* dst, uint64_t destMax, const void* src, u64 count, HcclRtMemcpyKind kind, Stream& stream, u32 remoteUserRank,
    hccl::LinkType inLinkType)
{
    uint64_t beginTime = GetMsprofSysCycleTime();

    // 参数有效性检查
    if (stream.ptr() == nullptr) {
        CHK_SAFETY_FUNC_RET(memcpy_s(dst, destMax, src, count));
        return HCCL_SUCCESS;
    }

    if (count == 0) {
        HCCL_DEBUG("count is 0, return success.");
        return HCCL_SUCCESS;
    }

    if (src == dst) {
        HCCL_DEBUG("src == dst, return success.");
        return HCCL_SUCCESS;
    }

    if (destMax < count) {
        HCCL_ERROR("The size of destMax is smaller than that of count. destMax[%llu], count[%llu]", destMax, count);
        return HCCL_E_PARA;
    }

    uint64_t spiltLoop = 0;
    uint64_t addrOffset = 0;
    uint64_t contSplit = 0;
    if (count > HCCL_SDMA_MAX_COUNT_4GB) {
        spiltLoop = (count % HCCL_SDMA_MAX_COUNT_4GB != 0) ? (count / HCCL_SDMA_MAX_COUNT_4GB) :
                                                             ((count / HCCL_SDMA_MAX_COUNT_4GB) - 1);
        HCCL_INFO("MemcpyAsync SDMA task countSize is bigger than 4GB and do segmentation splitloop[%llu]", spiltLoop);
    }
    /* SDMA任务拆分 */
    for (uint64_t index = 0; index <= spiltLoop; index++) {
        addrOffset = index * HCCL_SDMA_MAX_COUNT_4GB;
        contSplit = (index == spiltLoop) ? (count - index * HCCL_SDMA_MAX_COUNT_4GB) : (HCCL_SDMA_MAX_COUNT_4GB);
        void* srcSplit = static_cast<void*>(static_cast<char*>(const_cast<void*>(src)) + addrOffset);
        void* dstSplit = static_cast<void*>(static_cast<char*>(dst) + addrOffset);

        CHK_RET(hrtMemAsyncCopy(dstSplit, destMax, srcSplit, contSplit, kind, stream.ptr()));
        // 调用回调来保存task信息
        if (callback_ != nullptr) {
            hccl::TaskParaDMA para((const void*)srcSplit, dstSplit, contSplit, inLinkType, remoteUserRank);
            hccl::TaskPara taskPara;
            SetupTaskParaDma(
                taskPara, para, TaskType::TASK_SDMA, ProfilerType::TASK_ALL, stream, beginTime, stream.IsMainStream());
            callback_(callBackUserPtr_, (void*)&taskPara, sizeof(struct TaskPara));
        }
        u32 taskID = 0;
        u32 streamID = 0;
        hrtGetTaskIdAndStreamID(taskID, streamID);
        PLF_CONFIG_INFO(
            PLF_TASK,
            "%s para: dst[%p] destMax[%llu] src[%p] count[%llu] rtMemcpyKind[%d] taskID[%u] streamID[%u] "
            "remoteUserRank[%u] inLinkType[%d]",
            __func__, dstSplit, destMax, srcSplit, contSplit, kind, taskID, streamID, remoteUserRank, inLinkType);
    }

    return HCCL_SUCCESS;
}

HcclResult DispatcherPub::MemcpyAsync(hccl::HostMem& dst, const hccl::DeviceMem& src, hccl::Stream& stream)
{
    if (dst.size() < src.size()) {
        HCCL_ERROR(
            "The size of dst is smaller than that of src. dst addr[%p], dst size[%llu], src addr[%p], src size[%llu]",
            dst.ptr(), dst.size(), src.ptr(), src.size());
        return HCCL_E_PTR;
    }

    CHK_RET(MemcpyAsync(
        dst.ptr(), dst.size(), src.ptr(), src.size(), HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_DEVICE_TO_HOST, stream));

    return HCCL_SUCCESS;
}

HcclResult DispatcherPub::MemcpyAsync(hccl::HostMem& dst, const hccl::HostMem& src, hccl::Stream& stream)
{
    if (dst.size() < src.size()) {
        HCCL_ERROR(
            "The size of dst is smaller than that of src. dst addr[%p], dst size[%llu], src addr[%p], src size[%llu]",
            dst.ptr(), dst.size(), src.ptr(), src.size());
        return HCCL_E_PTR;
    }

    CHK_RET(MemcpyAsync(
        dst.ptr(), dst.size(), src.ptr(), src.size(), HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_HOST, stream));

    return HCCL_SUCCESS;
}

HcclResult DispatcherPub::MemcpyAsync(
    hccl::DeviceMem& dst, const hccl::DeviceMem& src, hccl::Stream& stream, u32 remoteUserRank,
    hccl::LinkType inLinkType)
{
    if (dst.size() < src.size()) {
        HCCL_ERROR(
            "The size of dst is smaller than that of src. dst addr[%p], dst size[%llu], src addr[%p], src size[%llu]",
            dst.ptr(), dst.size(), src.ptr(), src.size());
        return HCCL_E_PTR;
    }

    return MemcpyAsync(
        dst.ptr(), dst.size(), src.ptr(), src.size(), HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_DEVICE_TO_DEVICE, stream,
        remoteUserRank, inLinkType);
}

HcclResult DispatcherPub::MemcpyAsync(hccl::DeviceMem& dst, const hccl::HostMem& src, hccl::Stream& stream)
{
    if (dst.size() < src.size()) {
        HCCL_ERROR(
            "The size of dst is smaller than that of src. dst addr[%p], dst size[%llu], src addr[%p], src size[%llu]",
            dst.ptr(), dst.size(), src.ptr(), src.size());
        return HCCL_E_PTR;
    }

    CHK_RET(MemcpyAsync(
        dst.ptr(), dst.size(), src.ptr(), src.size(), HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE, stream));

    return HCCL_SUCCESS;
}

HcclResult DispatcherPub::MemcpyAsyncWithoutCheckKind(
    void* dst, uint64_t destMax, const void* src, u64 count, HcclRtMemcpyKind kind, Stream& stream, u32 remoteUserRank,
    hccl::LinkType inLinkType)
{
    uint64_t beginTime = GetMsprofSysCycleTime();

    // 参数有效性检查
    if (stream.ptr() == nullptr) {
        HCCL_DEBUG("stream ptr is null, use memcpy.");
        CHK_SAFETY_FUNC_RET(memcpy_s(dst, destMax, src, count));
        return HCCL_SUCCESS;
    }

    if (count == 0 || src == dst) {
        HCCL_DEBUG("count[%llu] is 0 or src is equal to dst, return success.", count);
        return HCCL_SUCCESS;
    }

    if (destMax < count) {
        HCCL_ERROR("The size of destMax is smaller than that of count. destMax[%llu], count[%llu]", destMax, count);
        return HCCL_E_PARA;
    }

    uint64_t spiltLoop = 0;
    uint64_t addrOffset = 0;
    uint64_t contSplit = 0;
    if (count > HCCL_SDMA_MAX_COUNT_4GB) {
        spiltLoop = (count % HCCL_SDMA_MAX_COUNT_4GB != 0) ? (count / HCCL_SDMA_MAX_COUNT_4GB) :
                                                             ((count / HCCL_SDMA_MAX_COUNT_4GB) - 1);
        HCCL_INFO("MemcpyAsync SDMA task countSize is bigger than 4GB and do segmentation splitloop[%llu]", spiltLoop);
    }
    /* SDMA任务拆分 */
    for (uint64_t index = 0; index <= spiltLoop; index++) {
        addrOffset = index * HCCL_SDMA_MAX_COUNT_4GB;
        contSplit = (index == spiltLoop) ? (count - index * HCCL_SDMA_MAX_COUNT_4GB) : (HCCL_SDMA_MAX_COUNT_4GB);
        void* srcSplit = static_cast<void*>(static_cast<char*>(const_cast<void*>(src)) + addrOffset);
        void* dstSplit = static_cast<void*>(static_cast<char*>(dst) + addrOffset);

        CHK_RET(hrtMemAsyncCopyWithoutCheckKind(dstSplit, destMax, srcSplit, contSplit, kind, stream.ptr()));

        // 调用回调来保存task信息
        if (callback_ != nullptr) {
            hccl::TaskParaDMA para((const void*)srcSplit, dstSplit, contSplit, inLinkType, remoteUserRank);
            hccl::TaskPara taskPara;
            SetupTaskParaDma(
                taskPara, para, TaskType::TASK_SDMA, ProfilerType::TASK_ALL, stream, beginTime, stream.IsMainStream());
            callback_(callBackUserPtr_, (void*)&taskPara, sizeof(struct TaskPara));
        }
        u32 taskID = 0;
        u32 streamID = 0;
        hrtGetTaskIdAndStreamID(taskID, streamID);
        PLF_CONFIG_INFO(
            PLF_TASK,
            "%s para: dst[%p] destMax[%llu] src[%p] count[%llu] rtMemcpyKind[%d] taskID[%u] streamID[%u] "
            "remoteUserRank[%u] inLinkType[%d]",
            __func__, dstSplit, destMax, srcSplit, contSplit, kind, taskID, streamID, remoteUserRank, inLinkType);
    }

    return HCCL_SUCCESS;
}

HcclResult DispatcherPub::DevMemMalloc(
    [[maybe_unused]] void* stream, [[maybe_unused]] void*& devMem1, [[maybe_unused]] void*& devMem2)
{
#ifndef HCCD
    int32_t streamId;
    u32 blockSize;
    CHK_RET(hrtGetStreamId(stream, streamId));
    CHK_RET(HcclGetVectorBlockSize(&blockSize, deviceLogicId_));

    std::unique_lock<std::mutex> lock(devMemMutex_);
    if (devMemMap_.find(streamId) == devMemMap_.end()) {
        u32 devMemSize = blockSize + blockSize;
        CHK_RET(hrtMalloc(&devMem1, devMemSize));
        CHK_PTR_NULL(devMem1);
        CHK_RET(hrtMemSet(devMem1, devMemSize, devMemSize));
        devMem2 = static_cast<char*>(devMem1) + blockSize;
        devMemMap_[streamId] = devMem1;
    } else {
        devMem1 = devMemMap_[streamId];
        devMem2 = static_cast<char*>(devMem1) + blockSize;
    }
#endif
    return HCCL_SUCCESS;
}

HcclResult DispatcherPub::JudgeIsTail(
    [[maybe_unused]] const void* src1, [[maybe_unused]] const void* src2, [[maybe_unused]] const void* dst,
    [[maybe_unused]] u64 count, [[maybe_unused]] const HcclDataType dataType, [[maybe_unused]] u64& headCount,
    [[maybe_unused]] u64& tailCount, [[maybe_unused]] void*& tailSrc1, [[maybe_unused]] void*& tailSrc2,
    [[maybe_unused]] void*& tailDst)
{
#ifndef HCCD
    CHK_PRT_RET(dataType >= HCCL_DATA_TYPE_RESERVED, HCCL_ERROR("dataType is failed."), HCCL_E_PARA);
    u32 blockSize = 0;
    CHK_RET(HcclGetVectorBlockSize(&blockSize, deviceLogicId_));
    // 获取总的数据量
    u64 dataSize = SIZE_TABLE[dataType] * count;                         // 计算总的字节数
    headCount = dataSize / blockSize * blockSize / SIZE_TABLE[dataType]; // 计算出32字节整倍数的数据数量
    tailCount = count - headCount;

    if (tailCount != 0) {
        tailSrc1 = static_cast<char*>(const_cast<void*>(src1)) + (headCount * SIZE_TABLE[dataType]);
        tailSrc2 = static_cast<char*>(const_cast<void*>(src2)) + (headCount * SIZE_TABLE[dataType]);
        tailDst = static_cast<char*>(const_cast<void*>(dst)) + (headCount * SIZE_TABLE[dataType]);
    }
#endif
    return HCCL_SUCCESS;
}

HcclResult DispatcherPub::DealTbeReduce(
    [[maybe_unused]] const void* src1, [[maybe_unused]] const void* src2, [[maybe_unused]] u64 count,
    [[maybe_unused]] const HcclDataType datatype, [[maybe_unused]] HcclReduceOp redOp, [[maybe_unused]] Stream& stream,
    [[maybe_unused]] const void* dst)
{
#ifndef HCCD
    HcclResult ret = HCCL_SUCCESS;
    void* tailSrc1 = nullptr;
    void* tailSrc2 = nullptr;
    void* tailDst = nullptr;
    u64 headCount = 0;
    u64 tailCount = 0;
    TbeReduceParam param;
    std::vector<void*> overflowAddrs;
    overflowAddrs.push_back(overflowAddr_);
    param.dataType = datatype;
    param.redOp = redOp;
    CHK_RET(JudgeIsTail(src1, src2, dst, count, datatype, headCount, tailCount, tailSrc1, tailSrc2, tailDst));
    if (headCount != 0) {
        param.src1 = const_cast<void*>(src1);
        param.src2 = const_cast<void*>(src2);
        param.dst = const_cast<void*>(dst);
        param.count = headCount;
        // 对满足32字节整倍数的数据进行reduce
        ret = HcclTbeReduce(&param, stream.ptr(), overflowAddrs.data(), overflowAddrs.size(), deviceLogicId_);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[DispatcherPub][ReduceAsync]errNo[0x%016llx] tbe vector Reduce fail,return[%d]. "
                "para: src1[%p] src2[%p] count_reduce[%llu] datatype[%s] op[%s] stream[%p] dst_reduce[%p].",
                HCCL_ERROR_CODE(ret), ret, src1, src2, count, GetDataTypeEnumStr(datatype).c_str(),
                GetReduceOpEnumStr(redOp).c_str(), stream.ptr(), dst),
            ret);
    }
    // 对不满足32字节整倍数的剩余数据进行reduce
    if (tailCount != 0) {
        void* devMem1 = nullptr;
        void* devMem2 = nullptr;
        u32 blockSize = 0;
        CHK_RET(HcclGetVectorBlockSize(&blockSize, deviceLogicId_));
        CHK_RET(DevMemMalloc(stream.ptr(), devMem1, devMem2));
        CHK_RET(hrtMemAsyncCopy(
            devMem1, blockSize, tailSrc1, tailCount * SIZE_TABLE[datatype],
            HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_DEVICE_TO_DEVICE, stream.ptr()));
        CHK_RET(hrtMemAsyncCopy(
            devMem2, blockSize, tailSrc2, tailCount * SIZE_TABLE[datatype],
            HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_DEVICE_TO_DEVICE, stream.ptr()));
        param.src1 = devMem1;
        param.src2 = devMem2;
        param.dst = devMem2;
        param.count = tailCount;
        ret = HcclTbeReduce(&param, stream.ptr(), overflowAddrs.data(), overflowAddrs.size(), deviceLogicId_);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[DispatcherPub][ReduceAsync]errNo[0x%016llx] tbe vector Reduce fail,return[%d]. "
                "para: src1[%p] src2[%p] count_reduce[%llu] datatype[%s] op[%s] stream[%p] dst_reduce[%p].",
                HCCL_ERROR_CODE(ret), ret, src1, src2, count, GetDataTypeEnumStr(datatype).c_str(),
                GetReduceOpEnumStr(redOp).c_str(), stream.ptr(), dst),
            ret);
        CHK_RET(hrtMemAsyncCopy(
            tailDst, tailCount * SIZE_TABLE[datatype], devMem2, tailCount * SIZE_TABLE[datatype],
            HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_DEVICE_TO_DEVICE, stream.ptr()));
    }
#endif
    return HCCL_SUCCESS;
}

HcclResult DispatcherPub::TbeReduceAsync(
    const void* src1, const void* src2, u64 count, const HcclDataType datatype, HcclReduceOp redOp, Stream& stream,
    const void* dst)
{
    HCCL_DEBUG(
        "Enter--para: src1[%p], src2[%p], count[%llu], datatype[%s], red_op[%s], dst[%p].", src1, src2, count,
        GetDataTypeEnumStr(datatype).c_str(), GetReduceOpEnumStr(redOp).c_str(), dst);
    uint64_t beginTime = GetMsprofSysCycleTime();

    if (count == 0) {
        HCCL_WARNING("count is 0, return success.");
        return HCCL_SUCCESS;
    }
#ifndef HCCD
    CHK_RET(DealTbeReduce(src1, src2, count, datatype, redOp, stream, dst));
#else
    HCCL_ERROR("[DispatcherPub][ReduceAsync] does not support this interface.");
    return HCCL_E_PARA;
#endif
    // 调用回调来保存task信息
    if (callback_ != nullptr) {
        hccl::TaskParaReduce para(src1, dst, count, redOp, datatype, hccl::LinkType::LINK_ONCHIP);
        hccl::TaskPara taskPara(TaskType::TASK_REDUCE_TBE, para);
        taskPara.stream = stream.ptr();
        taskPara.beginTime = beginTime;
        taskPara.isMainStream = stream.IsMainStream();
        callback_(callBackUserPtr_, (void*)&taskPara, sizeof(struct TaskPara));
    }

    return HCCL_SUCCESS;
}

HcclResult DispatcherPub::SetGlobalWorkSpace([[maybe_unused]] std::vector<void*>& globalWorkSpaceAddr)
{
#ifndef HCCD
    DevType devType;
    CHK_RET(hrtGetDeviceType(devType));
    if (devType != DevType::DEV_TYPE_910 && devType != DevType::DEV_TYPE_310P3) {
        return HCCL_SUCCESS;
    }

    void* overflowAddr = nullptr;
    CHK_RET(hrtCtxGetOverflowAddr(&overflowAddr));
    globalWorkSpaceAddr.push_back(overflowAddr);
    if (globalWorkSpaceAddr.size() != 0) {
        // 第0位代表溢出检测
        overflowAddr_ = globalWorkSpaceAddr[static_cast<u32>(GlobalWorkSpaceType::OVERFLOW_DETECT_MODE)];
    }
#else
    HCCL_ERROR("[DispatcherPub][SetGlobalWorkSpace] does not support this interface.");
    return HCCL_E_PARA;
#endif
    return HCCL_SUCCESS;
}

HcclResult DispatcherPub::InlineReduceAsync(
    const void* src, u64 count, const HcclDataType datatype, HcclReduceOp redOp, Stream& stream, void* dst,
    u32 remoteUserRank, hccl::LinkType inLinkType)
{
    if (count == 0) {
        HCCL_WARNING("count is 0, return success.");
        return HCCL_SUCCESS;
    }
    /* 注意：profiling数据任务时间仍提供切分前整个任务时间 */
    uint64_t beginTime = GetMsprofSysCycleTime();

    CHK_PTR_NULL(stream.ptr());

    aclDataType runtimeDataType = ACL_DT_UNDEFINED;
    aclrtReduceKind rtReduceOp = ACL_RT_MEMCPY_SDMA_AUTOMATIC_EQUAL;
    try {
        runtimeDataType = HCCL_RT_DATA_TYPE_MAP.at(datatype);
        rtReduceOp = HCCL_RT_REDUCE_OP_MAP.at(redOp);
    } catch (...) {
        HCCL_ERROR(
            "[DispatcherPub][ReduceAsync]data type[%s] or reduceOp[%s] is not support",
            GetDataTypeEnumStr(datatype).c_str(), GetReduceOpEnumStr(redOp).c_str());
        return HCCL_E_PARA;
    }

    DevType deviceType;
    CHK_RET(hrtGetDeviceType(deviceType));

    uint64_t spiltLoop = 0;
    uint64_t addr_offset = 0;
    uint64_t contSplit = 0;
    uint64_t countSize = count * SIZE_TABLE[datatype];
    if (countSize > HCCL_SDMA_MAX_COUNT_4GB) {
        spiltLoop = (countSize % HCCL_SDMA_MAX_COUNT_4GB != 0) ? (countSize / HCCL_SDMA_MAX_COUNT_4GB) :
                                                                 ((countSize / HCCL_SDMA_MAX_COUNT_4GB) - 1);
        HCCL_INFO(
            "InlineReduceAsync SDMA task countSize is bigger than 4GB and do segmentation splitloop[%llu]", spiltLoop);
    }
    for (uint64_t index = 0; index <= spiltLoop; index++) {
        addr_offset = index * HCCL_SDMA_MAX_COUNT_4GB;
        contSplit = (index == spiltLoop) ? (countSize - index * HCCL_SDMA_MAX_COUNT_4GB) : (HCCL_SDMA_MAX_COUNT_4GB);
        void* srcSplit = static_cast<void*>(static_cast<char*>(const_cast<void*>(src)) + addr_offset);
        void* dstSplit = static_cast<void*>(static_cast<char*>(dst) + addr_offset);

        CHK_RET(hrtReduceAsync(dstSplit, contSplit, srcSplit, contSplit, rtReduceOp, runtimeDataType, stream.ptr()));

        // 调用回调来保存 task 信息
        if (callback_ != nullptr) {
            hccl::TaskParaReduce para(srcSplit, dstSplit, contSplit, redOp, datatype, inLinkType, remoteUserRank);
            hccl::TaskPara taskPara(TaskType::TASK_REDUCE_INLINE, para);
            taskPara.stream = stream.ptr();
            taskPara.beginTime = beginTime;
            taskPara.isMainStream = stream.IsMainStream();
            callback_(callBackUserPtr_, (void*)&taskPara, sizeof(struct TaskPara));
        }

        u32 taskID = 0;
        u32 streamID = 0;
        hrtGetTaskIdAndStreamID(taskID, streamID);
        PLF_CONFIG_INFO(
            PLF_TASK,
            "%s para: dst[%p] src[%p] count[%llu] rtReduceOp[%d] runtimeDataType[%d] taskID[%u] streamID[%u] "
            "remoteUserRank[%u] inLinkType[%d]",
            __func__, dstSplit, srcSplit, contSplit / SIZE_TABLE[datatype], redOp, runtimeDataType, taskID, streamID,
            remoteUserRank, inLinkType);
    }

    return HCCL_SUCCESS;
}

HcclResult DispatcherPub::ReduceAsync(
    const void* src, void* dst, u64 dataCount, const HcclDataType datatype, HcclReduceOp redOp, Stream& stream,
    HcclReduceType reduceType)
{
    return (reduceType == HcclReduceType::HCCL_INLINE_REDUCE) ?
               InlineReduceAsync(src, dataCount, datatype, redOp, stream, dst) :
               TbeReduceAsync(src, dst, dataCount, datatype, redOp, stream, dst);
}

HcclResult DispatcherPub::SignalRecord(
    [[maybe_unused]] hccl::DeviceMem& dst, [[maybe_unused]] hccl::DeviceMem& src, [[maybe_unused]] hccl::Stream& stream,
    [[maybe_unused]] u32 remoteUserRank, [[maybe_unused]] hccl::LinkType inLinkType, [[maybe_unused]] u32 notifyId)
{
    HCCL_ERROR("does not support this interface.");
    return HCCL_E_NOT_SUPPORT;
}

HcclResult DispatcherPub::RdmaRecord(
    [[maybe_unused]] u32 dbindex, [[maybe_unused]] u64 dbinfo, [[maybe_unused]] const struct SendWr& wr,
    [[maybe_unused]] hccl::Stream& stream, [[maybe_unused]] RdmaType rdmaType, [[maybe_unused]] u32 userRank,
    [[maybe_unused]] u64 offset, [[maybe_unused]] u32 notifyId)
{
    HCCL_ERROR("does not support this interface.");
    return HCCL_E_NOT_SUPPORT;
}

HcclResult DispatcherPub::GetCallbackResult() { return g_callBackResult; }

void HostNicTcpCallBackProfiling(RaSocketParams* params, std::chrono::microseconds duration)
{
    hccl::TaskParaHost para(
        params->taskInfo.streamId, params->taskInfo.taskId, params->len, duration, params->taskInfo.tag);
    hccl::TaskPara taskPara(TaskType::TASK_HOST, para);
    taskPara.profilerType = ProfilerType::TASK_PROFILING;
    params->callback(params->callBackUserPtr, (void*)&taskPara, sizeof(struct TaskPara));
}

void HostNicCallbackSendWr(void* fnData)
{
    RaSendWrParams* params = static_cast<RaSendWrParams*>(fnData);
    unsigned int completeNum = 0;
    HcclUs startut = TIME_NOW();
    HcclResult ret = HrtRaSendWrlistExt(params->qpHandle, &params->wr, &params->opRsp, 1, &completeNum);
    HcclUs endtut = TIME_NOW();
    std::chrono::microseconds duration = DURATION_US(endtut - startut);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[Send][Wr]host nic hrtRaSendWrlist failed");
        g_callBackResult = ret;
    }

    hccl::TaskParaHost para(
        params->taskInfo.streamId, params->taskInfo.taskId, params->wr.memList.len, duration, params->taskInfo.tag);
    hccl::TaskPara taskPara(TaskType::TASK_HOST, para);
    taskPara.profilerType = ProfilerType::TASK_PROFILING;
    params->callback(params->callBackUserPtr, (void*)&taskPara, sizeof(struct TaskPara));

    // 单算子场景内存需要及时释放
    if (params->workMode == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        DispatcherPub* tmpDispatcherPtr = static_cast<DispatcherPub*>(params->dispatcherPtr);
        ret = tmpDispatcherPtr->DelHostNICRdmaTask(params->taskInfo.streamId, params->taskInfo.taskId);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("[Send][Wr]Del Host NIC Task failed");
            g_callBackResult = ret;
        }
    }
}

// 一次callback，多次收发
void HostNicCallbackTcpSend(void* fnData)
{
    RaSocketParams* params = static_cast<RaSocketParams*>(fnData);
    u64 bufferSize = params->socketBufferLen;
    u64 sendCount = params->len / bufferSize + (params->len % bufferSize != 0); // 要发送buffer的次数
    u64 totalSentSize = 0;                                                      // 已发送大小
    HcclResult ret = HCCL_SUCCESS;
    ret = hrtSetDevice(params->deviceLogicId);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[Socket][Send] set deviceId[%d] failed", params->deviceLogicId);
        g_callBackResult = ret;
        return;
    }
    HcclUs startut = TIME_NOW();
    for (u64 i = 0; i < sendCount; ++i) {
        u64 curSendSize = bufferSize;
        if (i == sendCount - 1 && totalSentSize + bufferSize > params->len) {
            curSendSize = params->len - totalSentSize;
        }
        ret = hrtMemSyncCopy(
            params->socketBufferPtr, curSendSize,
            static_cast<void*>(reinterpret_cast<char*>(params->ptr) + totalSentSize), curSendSize,
            (params->nicDeploy == NICDeployment::NIC_DEPLOYMENT_HOST) ?
                HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_DEVICE_TO_HOST :
                HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_DEVICE_TO_DEVICE);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR(
                "[Socket][Send]host nic hrtRaSocketBlockSend memcpy failed, tcp nicDeploy[%d]", params->nicDeploy);
            g_callBackResult = ret;
        }
        ret = hrtRaSocketBlockSend(params->socketFdHandle, params->socketBufferPtr, curSendSize);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("[Socket][Send]host nic hrtRaSocketBlockSend send failed");
            g_callBackResult = ret;
        }
        totalSentSize += curSendSize;
    }
    ret = hrtResetDevice(params->deviceLogicId);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[Socket][Send] reset deviceId[%d] failed", params->deviceLogicId);
        g_callBackResult = ret;
    }
    HostNicTcpCallBackProfiling(params, DURATION_US(TIME_NOW() - startut));
}

void HostNicCallbackTcpRecv(void* fnData)
{
    RaSocketParams* params = static_cast<RaSocketParams*>(fnData);
    u64 bufferSize = params->socketBufferLen;
    u64 recvCount = params->len / bufferSize + (params->len % bufferSize != 0); // 要接收buffer的次数
    u64 totalRecvSize = 0;                                                      // 已接收大小
    HcclResult ret = hrtSetDevice(params->deviceLogicId);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[Socket][Recv] set deviceId[%d] failed", params->deviceLogicId);
        g_callBackResult = ret;
    }
    HcclUs startut = TIME_NOW();
    for (u64 i = 0; i < recvCount; ++i) {
        u64 curRecvSize = bufferSize;
        if (i == recvCount - 1 && totalRecvSize + bufferSize > params->len) {
            curRecvSize = params->len - totalRecvSize;
        }
        ret = hrtRaSocketBlockRecv(params->socketFdHandle, params->socketBufferPtr, curRecvSize);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("[Socket][Recv]host nic hrtRaSocketBlockRecv recv failed");
            g_callBackResult = ret;
        }
        ret = hrtMemSyncCopy(
            static_cast<void*>(reinterpret_cast<char*>(params->ptr) + totalRecvSize), curRecvSize,
            params->socketBufferPtr, curRecvSize,
            (params->nicDeploy == NICDeployment::NIC_DEPLOYMENT_HOST) ?
                HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_DEVICE_TO_HOST :
                HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_DEVICE_TO_DEVICE);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR(
                "[Socket][Recv]host nic hrtRaSocketBlockRecv memcpy failed, tcp nicDeploy[%d]", params->nicDeploy);
            g_callBackResult = ret;
        }
        totalRecvSize += curRecvSize;
    }
    ret = hrtResetDevice(params->deviceLogicId);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[Socket][Reset] reset deviceId[%d] failed", params->deviceLogicId);
        g_callBackResult = ret;
    }
    HostNicTcpCallBackProfiling(params, DURATION_US(TIME_NOW() - startut));
    if (params->workMode == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        DispatcherPub* tmpDispatcherPtr = static_cast<DispatcherPub*>(params->dispatcherPtr);
        ret = tmpDispatcherPtr->DelHostNICTcpRecvTask(params->taskInfo.streamId, params->taskInfo.taskId);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("[Socket][Send]Del Host NIC Task failed");
            g_callBackResult = ret;
        }
    } // 单算子场景内存需要及时释放
}

void WaitHostNicTcpSendDone(void* dispatcher) { static_cast<DispatcherPub*>(dispatcher)->WaitHostNicTcpSendTaskDone(); }

void StartHostNicTcpSendThread(void* fnData)
{
    RaSocketParams* params = static_cast<RaSocketParams*>(fnData);
    DispatcherPub* tmpDispatcherPtr = static_cast<DispatcherPub*>(params->dispatcherPtr);
    HcclResult ret = tmpDispatcherPtr->SetHostNicTcpSendThreadPara(fnData);
    // 单算子场景内存需要及时释放
    if (params->workMode == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        tmpDispatcherPtr = static_cast<DispatcherPub*>(params->dispatcherPtr);
        ret = tmpDispatcherPtr->DelHostNICTcpSendTask(params->taskInfo.streamId, params->taskInfo.taskId);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("[Socket][Send]Del Host NIC Task failed");
            g_callBackResult = ret;
        }
    }
}

HcclResult DispatcherPub::HostNicRdmaSend(
    QpHandle qpHandle, SendWrlistDataExt& wr, SendWrRsp& opRsp, hccl::Stream& stream, u32 userRank, u64 offset)
{
    uint64_t beginTime = GetMsprofSysCycleTime();
    CHK_PTR_NULL(qpHandle);
    CHK_PTR_NULL(stream.ptr());
    (void)opRsp;

    if (wr.memList.len == 0) {
        // zero byte message 不需要进行通信
        return HCCL_SUCCESS;
    }

    u64 notifyID = userRank;
    notifyID = (notifyID << 32) | (offset & 0x00000000FFFFFFFF); // 0x00000000FFFFFFFF用于取offset的低32位
    u32 taskID = 0;
    u32 streamID = 0;
    CHK_RET(hrtGetTaskIdAndStreamID(taskID, streamID));

    std::unique_ptr<RaSendWrParams> params = nullptr;
    HcclWorkflowMode workflowMode = GetWorkflowMode();
    params.reset(new (std::nothrow) RaSendWrParams(
        qpHandle, wr, static_cast<void*>(this), streamID, taskID, notifyID, workflowMode, callback_, callBackUserPtr_));
    CHK_PTR_NULL(params);

    std::unique_lock<std::mutex> lock(hostNicMutex_);
    hostNicRdmaParamsVec_[streamID].push(move(params));
    lock.unlock();

    CHK_RET(hrtCallbackLaunch(HostNicCallbackSendWr, hostNicRdmaParamsVec_[streamID].back().get(), stream.ptr(), true));

    RdmaType rdmaType = (offset == 0xFFFFFFFFFFFFFFFF) ? RdmaType::RDMA_SEND_PAYLOAD : RdmaType::RDMA_SEND_NOTIFY;

    // 调用回调来保存task信息
    if (callback_ != nullptr) {
        hccl::TaskParaDMA para(
            reinterpret_cast<void*>(static_cast<uintptr_t>(wr.memList.addr)),
            reinterpret_cast<void*>(static_cast<uintptr_t>(wr.dstAddr)), wr.memList.len, notifyID,
            hccl::LinkType::LINK_ROCE, rdmaType);
        hccl::TaskPara taskPara;
        SetupTaskParaDma(taskPara, para, TaskType::TASK_RDMA, ProfilerType::TASK_EXCEPTION, stream, beginTime, false);
        callback_(callBackUserPtr_, (void*)&taskPara, sizeof(struct TaskPara));
    }

    return HCCL_SUCCESS;
}
HcclResult DispatcherPub::HostNicTcpSend(
    FdHandle socketFdHandle, const void* socketBufferPtr, u64 socketBufferLen, const void* src, u64 len,
    hccl::Stream& stream, const NICDeployment nicDeploy)
{
    uint64_t beginTime = GetMsprofSysCycleTime();
    CHK_PTR_NULL(socketFdHandle);
    CHK_PTR_NULL(stream.ptr());
    u32 taskID = 0;
    u32 streamID = 0;
    CHK_RET(hrtGetTaskIdAndStreamID(taskID, streamID));
    HcclWorkflowMode workflowMode = GetWorkflowMode();
    std::unique_ptr<RaSocketParams> params = nullptr;
    params.reset(new (std::nothrow) RaSocketParams(
        socketFdHandle, socketBufferPtr, socketBufferLen, src, len, static_cast<void*>(this), streamID, taskID,
        workflowMode, deviceLogicId_, nicDeploy, callback_, callBackUserPtr_));
    std::unique_lock<std::mutex> taskLock(hostNicMutex_);
    hostNicTcpSendParamsVec_[streamID].push(move(params));
    taskLock.unlock();

    // 下发callback task
    CHK_RET(hrtCallbackLaunch(
        StartHostNicTcpSendThread, hostNicTcpSendParamsVec_[streamID].back().get(), stream.ptr(), true));

    // 回调保存信息供profiling记录
    if (callback_ != nullptr) {
        hccl::TaskParaDMA para(
            src, socketBufferPtr, len, INVALID_U64, hccl::LinkType::LINK_ROCE, RdmaType::RDMA_TYPE_RESERVED);
        hccl::TaskPara taskPara;
        SetupTaskParaDma(taskPara, para, TaskType::TASK_RDMA, ProfilerType::TASK_EXCEPTION, stream, beginTime, false);
        callback_(callBackUserPtr_, (void*)&taskPara, sizeof(struct TaskPara));
    }
    return HCCL_SUCCESS;
}
HcclResult DispatcherPub::HostNicTcpRecv(
    FdHandle socketFdHandle, const void* socketBufferPtr, u64 socketBufferLen, const void* src, u64 len,
    hccl::Stream& stream, const NICDeployment nicDeploy)
{
    uint64_t beginTime = GetMsprofSysCycleTime();
    CHK_PTR_NULL(socketFdHandle);
    CHK_PTR_NULL(stream.ptr());

    u32 taskID = 0;
    u32 streamID = 0;
    CHK_RET(hrtGetTaskIdAndStreamID(taskID, streamID));
    HcclWorkflowMode workflowMode = GetWorkflowMode();
    std::unique_ptr<RaSocketParams> params = nullptr;
    params.reset(new (std::nothrow) RaSocketParams(
        socketFdHandle, socketBufferPtr, socketBufferLen, src, len, static_cast<void*>(this), streamID, taskID,
        workflowMode, deviceLogicId_, nicDeploy, callback_, callBackUserPtr_));
    CHK_SMART_PTR_NULL(params);

    std::unique_lock<std::mutex> taskLock(hostNicMutex_);
    hostNicTcpRecvParamsVec_[streamID].push(move(params));
    taskLock.unlock();

    // 下发callback task
    CHK_RET(
        hrtCallbackLaunch(HostNicCallbackTcpRecv, hostNicTcpRecvParamsVec_[streamID].back().get(), stream.ptr(), true));

    // 回调保存信息供profiling记录
    if (callback_ != nullptr) {
        hccl::TaskParaDMA para(
            src, socketBufferPtr, len, INVALID_U64, hccl::LinkType::LINK_ROCE, RdmaType::RDMA_TYPE_RESERVED);
        hccl::TaskPara taskPara;
        SetupTaskParaDma(taskPara, para, TaskType::TASK_RDMA, ProfilerType::TASK_EXCEPTION, stream, beginTime, false);
        callback_(callBackUserPtr_, (void*)&taskPara, sizeof(struct TaskPara));
    }

    return HCCL_SUCCESS;
}

HcclResult DispatcherPub::SetHostNicTcpSendThreadPara(void* fnData)
{
    std::unique_ptr<RaSocketParams> params = nullptr;
    auto tmpRaSocketParamsPtr = new (std::nothrow) RaSocketParams(*(static_cast<RaSocketParams*>(fnData)));
    CHK_PTR_NULL(tmpRaSocketParamsPtr);
    params.reset(tmpRaSocketParamsPtr);
    std::unique_lock<std::mutex> lock(hostNicMutex_);
    if (hostNicTcpSendThreadParam_ == nullptr) {
        hostNicTcpSendThreadParam_ = move(params);
    } else {
        HCCL_ERROR(
            "last send task is not finished! stream[%u] task[%u]", hostNicTcpSendThreadParam_->taskInfo.streamId,
            hostNicTcpSendThreadParam_->taskInfo.taskId);
    }
    return HCCL_SUCCESS;
}

void DispatcherPub::HostNicTcpSendThreadTask()
{
    // 给当前线程添加名字
    SetThreadName("Hccl_HostNicTcp");

    while (hostNicTcpSendThreadState_) {
        if (hostNicTcpSendThreadParam_ == nullptr) {
            SaluSleep(TCP_SEND_THREAD_SLEEP_TWO_HUNDRED_MICROSECOND);
        } else {
            void* fnData = hostNicTcpSendThreadParam_.get();
            if (fnData != nullptr) {
                HostNicCallbackTcpSend(fnData);
            }
            hostNicTcpSendThreadParam_ = nullptr;
        }
    }
}

HcclResult DispatcherPub::HostNicTcpWaitSendCompletion(hccl::Stream& stream)
{
    CHK_RET(hrtCallbackLaunch(WaitHostNicTcpSendDone, this, stream.ptr(), true));
    return HCCL_SUCCESS;
}
HcclResult DispatcherPub::DelHostNICRdmaTask(u32 streamID, u32 taskID)
{
    std::unique_lock<std::mutex> lock(hostNicMutex_);
    CHK_PRT_RET(
        (hostNicRdmaParamsVec_.find(streamID) == hostNicRdmaParamsVec_.end()),
        HCCL_ERROR(
            "[DispatcherPub][DelHostNICTask]errNo[0x%016llx] streamID[%u] not found in hostNicRdmaParamsVec_",
            HCCL_ERROR_CODE(HCCL_E_PARA), streamID),
        HCCL_E_PARA);

    CHK_PRT_RET(
        (hostNicRdmaParamsVec_[streamID].size() == 0),
        HCCL_ERROR(
            "[DispatcherPub][DelHostNICTask]"
            "errNo[0x%016llx] streamID[%u] task num is 0",
            HCCL_ERROR_CODE(HCCL_E_INTERNAL), streamID),
        HCCL_E_INTERNAL);

    CHK_PRT_RET(
        (hostNicRdmaParamsVec_[streamID].front()->taskInfo.taskId != taskID),
        HCCL_ERROR(
            "[DispatcherPub][DelHostNICTask]errNo[0x%016llx] streamID[%u] taskID[%u]"
            " is not equal to the front taskID[%u]",
            HCCL_ERROR_CODE(HCCL_E_INTERNAL), streamID, taskID,
            hostNicRdmaParamsVec_[streamID].front()->taskInfo.taskId),
        HCCL_E_INTERNAL);

    hostNicRdmaParamsVec_[streamID].pop();
    return HCCL_SUCCESS;
}
HcclResult DispatcherPub::DelHostNICTcpSendTask(u32 streamID, u32 taskID)
{
    std::unique_lock<std::mutex> lock(hostNicMutex_);
    CHK_PRT_RET(
        (hostNicTcpSendParamsVec_.find(streamID) == hostNicTcpSendParamsVec_.end()),
        HCCL_ERROR(
            "[DispatcherPub][DelHostNICTask]errNo[0x%016llx] streamID[%u] not found in hostNicTcpSendParamsVec_",
            HCCL_ERROR_CODE(HCCL_E_PARA), streamID),
        HCCL_E_PARA);

    CHK_PRT_RET(
        (hostNicTcpSendParamsVec_[streamID].size() == 0),
        HCCL_ERROR(
            "[DispatcherPub][DelHostNICTask]"
            "errNo[0x%016llx] streamID[%u] task num is 0",
            HCCL_ERROR_CODE(HCCL_E_INTERNAL), streamID),
        HCCL_E_INTERNAL);

    CHK_PRT_RET(
        (hostNicTcpSendParamsVec_[streamID].front()->taskInfo.taskId != taskID),
        HCCL_ERROR(
            "[DispatcherPub][DelHostNICTask]errNo[0x%016llx] streamID[%u] taskID[%u]"
            " is not equal to the front taskID[%u]",
            HCCL_ERROR_CODE(HCCL_E_INTERNAL), streamID, taskID,
            hostNicTcpSendParamsVec_[streamID].front()->taskInfo.taskId),
        HCCL_E_INTERNAL);

    hostNicTcpSendParamsVec_[streamID].pop();
    return HCCL_SUCCESS;
}
HcclResult DispatcherPub::DelHostNICTcpRecvTask(u32 streamID, u32 taskID)
{
    std::unique_lock<std::mutex> lock(hostNicMutex_);
    CHK_PRT_RET(
        (hostNicTcpRecvParamsVec_.find(streamID) == hostNicTcpRecvParamsVec_.end()),
        HCCL_ERROR(
            "[DispatcherPub][DelHostNICTask]errNo[0x%016llx] streamID[%u] not found in hostNicTcpRecvParamsVec_",
            HCCL_ERROR_CODE(HCCL_E_PARA), streamID),
        HCCL_E_PARA);

    CHK_PRT_RET(
        (hostNicTcpRecvParamsVec_[streamID].size() == 0),
        HCCL_ERROR(
            "[DispatcherPub][DelHostNICTask]"
            "errNo[0x%016llx] streamID[%u] task num is 0",
            HCCL_ERROR_CODE(HCCL_E_INTERNAL), streamID),
        HCCL_E_INTERNAL);

    CHK_PRT_RET(
        (hostNicTcpRecvParamsVec_[streamID].front()->taskInfo.taskId != taskID),
        HCCL_ERROR(
            "[DispatcherPub][DelHostNICTask]errNo[0x%016llx] streamID[%u] taskID[%u]"
            " is not equal to the front taskID[%u]",
            HCCL_ERROR_CODE(HCCL_E_INTERNAL), streamID, taskID,
            hostNicTcpRecvParamsVec_[streamID].front()->taskInfo.taskId),
        HCCL_E_INTERNAL);

    hostNicTcpRecvParamsVec_[streamID].pop();
    return HCCL_SUCCESS;
}
// 下沉模式下内部接口
HcclResult DispatcherPub::RdmaSend(
    u32 qpn, u32 wqeIndex, const struct SendWr& wr, HcclRtStream stream, RdmaType rdmaType, u64 notifyID,
    bool isMainStream)
{
    uint64_t beginTime = GetMsprofSysCycleTime();
    if ((qpn == INVALID_UINT) && (wqeIndex == INVALID_UINT)) {
        // zero byte message 不需要下发rdma send task
        return HCCL_SUCCESS;
    }

    CHK_RET(hrtRDMASend(qpn, wqeIndex, stream));

    // 调用回调来保存task信息
    if (callback_ != nullptr) {
        hccl::TaskParaDMA para(
            reinterpret_cast<void*>(static_cast<uintptr_t>(wr.bufList[0].addr)),
            reinterpret_cast<void*>(static_cast<uintptr_t>(wr.dstAddr)), wr.bufList[0].len, notifyID,
            hccl::LinkType::LINK_ROCE, rdmaType);
        hccl::TaskPara taskPara;
        SetupTaskParaDma(taskPara, para, TaskType::TASK_RDMA, stream, beginTime, isMainStream);
        callback_(callBackUserPtr_, (void*)&taskPara, sizeof(struct TaskPara));
    }

    u32 taskID = 0;
    u32 streamID = 0;
    hrtGetTaskIdAndStreamID(taskID, streamID);
    PLF_CONFIG_INFO(
        PLF_TASK, "%s para: qpn[%u] wqeIndex[%u] rdmaType[%d] notifyId[0x%016llx] taskID[%u] streamID[%u]", __func__,
        qpn, wqeIndex, rdmaType, notifyID, taskID, streamID);
    return HCCL_SUCCESS;
}

// 下沉模式下对外接口, 用于发送notify 信息
HcclResult
DispatcherPub::RdmaSend(u32 qpn, u32 wqeIndex, const struct SendWr& wr, hccl::Stream& stream, u32 userRank, u64 offset)
{
    u64 NotifyID
        = (static_cast<u64>(userRank) << 32) | (offset & 0x00000000FFFFFFFF); // 0x00000000FFFFFFFF用于取offset的低32位
    return RdmaSend(qpn, wqeIndex, wr, stream.ptr(), RdmaType::RDMA_SEND_NOTIFY, NotifyID, stream.IsMainStream());
}

// 下沉模式下对外接口, 用于发送payload 信息
HcclResult DispatcherPub::RdmaSend(u32 qpn, u32 wqeIndex, const struct SendWr& wr, hccl::Stream& stream, u32 userRank)
{
    u64 NotifyID
        = (static_cast<u64>(userRank) << 32) | (0x00000000FFFFFFFF); // 0x00000000FFFFFFFF usrrank位于notifyID的高32位
    return RdmaSend(qpn, wqeIndex, wr, stream.ptr(), RdmaType::RDMA_SEND_PAYLOAD, NotifyID, stream.IsMainStream());
}

// opbase 模式下内部接口
HcclResult DispatcherPub::RdmaSend(
    u32 dbindex, u64 dbinfo, const struct SendWr& wr, HcclRtStream stream, RdmaType rdmaType, u64 notifyID, u64 offset,
    bool isMainStream)
{
    uint64_t beginTime = GetMsprofSysCycleTime();
    if ((dbindex == INVALID_UINT) && (dbinfo == INVALID_U64)) {
        // zero byte message 不需要下发rdma send task
        return HCCL_SUCCESS;
    }

    CHK_RET(hrtRDMADBSend(dbindex, dbinfo, stream));

    // 调用回调来保存task信息
    if (callback_ != nullptr) {
        notifyID = (notifyID << 32) | (offset & 0x00000000FFFFFFFF); // 0x00000000FFFFFFFF用于取offset的低32位
        hccl::TaskParaDMA para(
            reinterpret_cast<void*>(static_cast<uintptr_t>(wr.bufList[0].addr)),
            reinterpret_cast<void*>(static_cast<uintptr_t>(wr.dstAddr)), wr.bufList[0].len, notifyID,
            hccl::LinkType::LINK_ROCE, rdmaType);
        hccl::TaskPara taskPara;
        SetupTaskParaDma(taskPara, para, TaskType::TASK_RDMA, stream, beginTime, isMainStream);
        callback_(callBackUserPtr_, (void*)&taskPara, sizeof(struct TaskPara));
    }

    u32 taskID = 0;
    u32 streamID = 0;
    hrtGetTaskIdAndStreamID(taskID, streamID);
    PLF_CONFIG_INFO(
        PLF_TASK,
        "%s para: dbindex[%u] dbinfo[%llu] rdmaType[%d] notifyId[0x%016llx] offset[%llu] taskID[%u] streamID[%u]",
        __func__, dbindex, dbinfo, rdmaType, notifyID, offset, taskID, streamID);
    return HCCL_SUCCESS;
}

// opbase 模式下对外接口，用于发送notify 信息
HcclResult DispatcherPub::RdmaSend(
    u32 dbindex, u64 dbinfo, const struct SendWr& wr, hccl::Stream& stream, u32 userRank, u64 offset,
    [[maybe_unused]] bool isCapture)
{
    CHK_RET(RdmaSend(
        dbindex, dbinfo, wr, stream.ptr(), RdmaType::RDMA_SEND_NOTIFY, userRank, offset, stream.IsMainStream()));

    return HCCL_SUCCESS;
}

// opbase 模式下对外接口，用于发送payload 信息
HcclResult DispatcherPub::RdmaSend(
    u32 dbindex, u64 dbinfo, const struct SendWr& wr, hccl::Stream& stream, u32 remoteUserRank,
    [[maybe_unused]] bool isCapture)
{
    u64 offset = 0;
    CHK_RET(RdmaSend(
        dbindex, dbinfo, wr, stream.ptr(), RdmaType::RDMA_SEND_PAYLOAD, remoteUserRank, offset, stream.IsMainStream()));

    return HCCL_SUCCESS;
}

HcclResult DispatcherPub::RdmaSend(
    [[maybe_unused]] u32 dbindex, [[maybe_unused]] u64 dbinfo, [[maybe_unused]] hccl::Stream& stream,
    [[maybe_unused]] RdmaTaskInfo& taskInfo)
{
    HCCL_ERROR("does not support this interface."); // host暂不使用此接口，待后续归一
    return HCCL_E_NOT_SUPPORT;
}

HcclResult DispatcherPub::SignalRecord(
    HcclRtNotify signal, Stream& stream, u32 userRank, u64 offset, s32 stage, [[maybe_unused]] bool inchip,
    [[maybe_unused]] u64 signalAddr, [[maybe_unused]] u32 notifyId)
{
    CHK_RET(SignalRecord(signal, stream.ptr(), userRank, offset, stage, stream.IsMainStream()));

    return HCCL_SUCCESS;
}

HcclResult DispatcherPub::SignalWait(
    HcclRtNotify signal, Stream& stream, u32 userRank, u32 remoteUserRank, s32 stage, [[maybe_unused]] bool inchip,
    u32 notifyId, u32 timeOut)
{
    (void)notifyId;
    CHK_RET(SignalWait(signal, stream.ptr(), userRank, remoteUserRank, stage, timeOut, stream.IsMainStream()));

    return HCCL_SUCCESS;
}

HcclResult DispatcherPub::AddRetryPreamble([[maybe_unused]] Stream& stream) { return HCCL_SUCCESS; }

HcclResult DispatcherPub::WaitValue(
    [[maybe_unused]] hccl::Stream& stream, [[maybe_unused]] u64 waitAddr, [[maybe_unused]] u64 valueAddr,
    [[maybe_unused]] bool reset)
{
    return HCCL_SUCCESS;
}
HcclResult DispatcherPub::WriteValue(
    [[maybe_unused]] hccl::Stream& stream, [[maybe_unused]] u64 writeAddr, [[maybe_unused]] u64 valueAddr)
{
    return HCCL_SUCCESS;
}

bool DispatcherPub::IsProfSubscribeAdditionInfo()
{
    u64 profConfig = GetProfConfig();
    if (((profConfig & PROF_TASK_TIME_L1_MASK) != 0) || ((profConfig & PROF_HCCL_TRACE_MASK) != 0) || isForce_) {
        return true;
    }
    return false;
}

HcclResult DispatcherPub::StreamSync([[maybe_unused]] Stream& stream)
{
    HCCL_INFO("StreamSync is not supported");
    return HCCL_SUCCESS;
}

void DispatcherPub::SetHcclQos(u32 hcclQos)
{
    HCCL_INFO("[DispatcherPub] [SetHcclQos] hcclQos = %u", hcclQos);
    // 按区间映射HCCL QOS到SDMA QOS
    if (hcclQos <= HCCL_QOS_LEVEL_1_LIMIT) {
        hcclQos_ = SDMA_QOS_LOW;
    } else if (hcclQos <= HCCL_QOS_LEVEL_2_LIMIT) {
        hcclQos_ = SDMA_QOS_MIDDLE;
    } else if (hcclQos <= HCCL_QOS_LEVEL_3_LIMIT) {
        hcclQos_ = SDMA_QOS_HIGH;
    } else {
        // 超出有效范围，使用默认值（包括hcclQos < HCCL_QOS_MIN的异常情况）
        hcclQos_ = SDMA_QOS_DEFAULT;
    }
}

void DispatcherPub::SetMpamid(u32 mPamid)
{
    HCCL_INFO("[DispatcherPub] [SetMpamid] mPamid[%u]", mPamid);
    mPamid_ = mPamid;
    return;
}
#endif
