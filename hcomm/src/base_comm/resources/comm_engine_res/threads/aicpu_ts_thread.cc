/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cast_utils.h"
#include "aicpu_ts_thread.h"
#include "hccl_common.h"
#include "aicpu/aicpu_hccl_sqcq.h"
#include "device_capacity.h"
#include "unified_platform/pub_inc/config_plf_log_v2.h"

namespace hccl {
AicpuTsThread::AicpuTsThread(StreamType streamType, uint32_t notifyNum, const NotifyLoadType notifyLoadType)
    : streamType_(streamType),
      notifyNum_(notifyNum),
      notifyLoadType_(notifyLoadType)
{}

AicpuTsThread::AicpuTsThread(const std::string& uniqueIdStr) : uniqueIdStr_(uniqueIdStr) {}

AicpuTsThread::~AicpuTsThread() { DeInitImpl(); }

HcclResult AicpuTsThread::Init()
{
    CHK_RET(GetRunSideIsDevice(isDeviceSide_));
    if (!isDeviceSide_) {
        // host侧申请资源
        HCCL_INFO("HcclThread::%s, is hostside", __func__);
        return HostInit();
    } else {
        // device侧反序列化，恢复资源
        HCCL_INFO("HcclThread::%s, is DeviceSide", __func__);
        return DeviceInit();
    }
}

HcclResult AicpuTsThread::DeInit() { return DeInitImpl(); }

HcclResult AicpuTsThread::DeInitImpl()
{
    streamType_ = StreamType::STREAM_TYPE_RESERVED;
    notifyNum_ = 0;
    if (stream_) {
        stream_->SetInvalidFlag();
    }
    stream_ = nullptr;
    notifys_.clear();
    uniqueIdStr_ = std::string();
    devType_ = DevType::DEV_TYPE_COUNT;
    return HCCL_SUCCESS;
}

std::string& AicpuTsThread::GetUniqueId()
{
    if (!uniqueIdStr_.empty()) {
        return uniqueIdStr_;
    }

    return UpdateUniqueId();
}

std::string& AicpuTsThread::UpdateUniqueId()
{
    // 序列化信息
    std::ostringstream oss;
    oss.write(ReinterpretAs<const char_t*>(&streamType_), sizeof(streamType_));
    oss.write(ReinterpretAs<const char_t*>(&notifyLoadType_), sizeof(notifyLoadType_));
    oss.write(ReinterpretAs<const char_t*>(&devId_), sizeof(devId_));
    oss.write(ReinterpretAs<const char_t*>(&notifyNum_), sizeof(notifyNum_));

    HcclStreamParam streamParam;
    streamParam.streamInfo.streamIds = stream_->id();
    streamParam.streamInfo.sqIds = stream_->sqId();
    streamParam.streamInfo.cqIds = stream_->cqId();
    streamParam.streamInfo.logicCqids = stream_->logicCqId();
    streamParam.sqCqContextAddr = ReinterpretAs<uint64_t>(sqCqeContext_.ptr());
    streamParam.sqCqContextSize = sqCqeContext_.size();
    oss.write(ReinterpretAs<const char_t*>(&streamParam), sizeof(streamParam));

    HcclResult ret = HCCL_SUCCESS;
    for (uint32_t idx = 0; idx < notifyNum_; idx++) {
        HcclSignalInfo notifyInfo;
        ret = notifys_[idx]->GetNotifyData(notifyInfo);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("[AicpuTsThread][UpdateUniqueId]GetNotifyData failed, ret[%d]", ret);
            uniqueIdStr_ = std::string();
            return uniqueIdStr_;
        }
        HCCL_INFO(
            "[AicpuTsThread][UpdateUniqueId]get local notify data success, resId[%u], tsId[%d], devId[%u]",
            notifyInfo.resId, notifyInfo.tsId, notifyInfo.devId);
        oss.write(ReinterpretAs<const char_t*>(&notifyInfo), sizeof(notifyInfo));
    }
    HCCL_DEBUG("[AicpuTsThread][UpdateUniqueId] stream[%p], notifyNum[%u]", stream_->ptr(), notifyNum_);

    uniqueIdStr_ = oss.str();
    return uniqueIdStr_;
}

#ifdef CCL_KERNEL_AICPU
HcclResult AicpuTsThread::BuildComStreamInfo(const HcclStreamInfo& streamInfo, HcclComStreamInfo& comStreamInfo) const
{
    comStreamInfo.sqId = streamInfo.sqIds;
    comStreamInfo.actualStreamId = streamInfo.streamIds;
    comStreamInfo.logicCqId = streamInfo.logicCqids;
    u64 sqAddr = 0;
    CHK_RET(QuerySqBaseAddr(devId_, streamInfo.sqIds, sqAddr));
    comStreamInfo.sqBaseAddr = ReinterpretAs<void*>(sqAddr);
    if (comStreamInfo.sqBaseAddr == nullptr) {
        HCCL_ERROR("[AicpuTsThread::InitStream] sqe base addr ptr is null.");
        return HCCL_E_PARA;
    }
    CHK_RET(QuerySqStatusByType(devId_, streamInfo.sqIds, DRV_SQCQ_PROP_SQ_DEPTH, comStreamInfo.sqDepth));
    HCCL_DEBUG(
        "[AicpuTsThread::InitStream] get stream data success, "
        "streamId[%d], sqId[%d], logicCqId[%u], sqDepth[%u]",
        comStreamInfo.actualStreamId, comStreamInfo.sqId, comStreamInfo.logicCqId, comStreamInfo.sqDepth);
    return HCCL_SUCCESS;
}
#endif

HcclResult AicpuTsThread::InitStream([[maybe_unused]] HcclStreamParam& streamParam)
{
#ifdef CCL_KERNEL_AICPU
    HcclStreamInfo& streamInfo = streamParam.streamInfo;

    static bool isCustom = false;
    static bool init = false;

    if (UNLIKELY(!init)) {
        uint32_t cpType = DEVDRV_PROCESS_CPTYPE_MAX;
        unsigned int hostpid = 0;
        CHK_RET(HrtHalDrvQueryProcessHostPid(getpid(), nullptr, nullptr, &hostpid, &cpType));
        isCustom = cpType == static_cast<uint32_t>(DEVDRV_PROCESS_CP2) ? true : false;
        init = true;
    }
    HcclResult ret = hrtHalResourceIdRestore(devId_, 0, DRV_STREAM_ID, streamInfo.streamIds, 0);
    // custom进程需要恢复stream资源, custom进程调用失败直接报错，aicpu进程调用失败做兼容性处理
    if (ret == HCCL_E_NOT_SUPPORT) {
        CHK_PRT_RET(
            isCustom,
            HCCL_ERROR(
                "%s hrtHalResourceIdRestore fail, drv not support, custom[%d], ret[%d]", __func__, isCustom, ret),
            HCCL_E_DRV);
    } else if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("%s hrtHalResourceIdRestore fail, ret[%d]", __func__, ret);
        return HCCL_E_DRV;
    }

    HcclComStreamInfo comStreamInfo{};
    CHK_RET(BuildComStreamInfo(streamInfo, comStreamInfo));

    stream_.reset(new (std::nothrow) Stream(comStreamInfo));
    CHK_SMART_PTR_NULL(stream_);

    // 初始化stream的sqeContext
    SqCqeContext* sqCqeContext = ReinterpretAs<SqCqeContext*>(streamParam.sqCqContextAddr);
    uint64_t sqCqContextSize = streamParam.sqCqContextSize;
    if (sqCqeContext == nullptr || sqCqContextSize != sizeof(SqCqeContext)) {
        HCCL_ERROR(
            "%s fail, sqCqeContext[%p] is null or size[%llu] is not equal to SqCqeContext size[%llu]", __func__,
            sqCqeContext, sqCqContextSize, sizeof(SqCqeContext));
        return HCCL_E_PARA;
    }
    sqCqeContext_ = DeviceMem::create(ReinterpretAs<void*>(sqCqeContext), sqCqContextSize);

    uint32_t sqTail = 0;
    uint32_t sqHead = 0;
    CHK_RET(QuerySqStatusByType(devId_, streamInfo.sqIds, DRV_SQCQ_PROP_SQ_TAIL, sqTail));
    CHK_RET(QuerySqStatusByType(devId_, streamInfo.sqIds, DRV_SQCQ_PROP_SQ_HEAD, sqHead));
    HCCL_DEBUG("[AicpuTsThread::InitStream] sqHead[%u], sqTail[%u]", sqHead, sqTail);

    ret = stream_->InitSqAndCqeContext(sqHead, sqTail, sqCqeContext);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("%s InitSqAndCqeContext failed", __func__), ret);
    HCCL_INFO("%s success, streamId[%d]", __func__, stream_->id());
#endif
    return HCCL_SUCCESS;
}

HcclResult AicpuTsThread::InitStreamLite(HcclStreamInfo& streamParam, uint32_t hostPhyId)
{
    // 在aicpu侧查询cqe时，需要使用logicCqids，而不是cqIds
    EXCEPTION_CATCH(
        pImpl_ = std::make_unique<Hccl::IAicpuTsThread>(
            streamParam.streamIds, streamParam.sqIds, hostPhyId, streamParam.logicCqids),
        return HCCL_E_PTR);
    return HCCL_SUCCESS;
}

uint32_t AicpuTsThread::GetNotifyNum() const { return notifyNum_; }

LocalNotify* AicpuTsThread::GetNotify(uint32_t index) const
{
    if (UNLIKELY(index >= notifyNum_ || index >= notifys_.size() || notifys_[index] == nullptr)) {
        HCCL_ERROR(
            "[AicpuTsThread][GetNotify] this[%p], streamId[%d], notifyNum[%u], index[%u], notifySize[%zu], "
            "index out of range[0, %u) or notify is null",
            this, stream_ ? stream_->id() : -1, notifyNum_, index, notifys_.size(), notifyNum_);
        return nullptr;
    }
    return notifys_[index].get();
}

// A3 Stream
Stream* AicpuTsThread::GetStream() const { return stream_.get(); }

// [中文导读] [AllReduce逐行 S213] AicpuTsThread::LaunchTask的接口声明：把发射委托给 ThreadImp 的底层执行队列；这些参数属于本函数调用边界。
void AicpuTsThread::LaunchTask() const
// [中文导读] [AllReduce逐行 S214] 进入AicpuTsThread::LaunchTask函数体：把发射委托给 ThreadImp 的底层执行队列。
{
    // [中文导读] [AllReduce逐行 S215] 将已生成的任务提交到具体RTSQ队列，不等同于全部任务完成；传入/处理AICPU_TS底层线程接口的LaunchTask字段。
    pImpl_->LaunchTask();
    // [中文导读] [AllReduce逐行 S216] 结束本次void调用，当前路径不再继续下发后续操作。
    return;
// [中文导读] [AllReduce逐行 S217] 结束AicpuTsThread::LaunchTask函数体；控制流返回外层。
}

void AicpuTsThread::TryLaunchTask() const
{
    pImpl_->TryLaunchTask();
    return;
}

// Local Data Plane Functions
HcclResult AicpuTsThread::LocalNotifyWait([[maybe_unused]] uint32_t notifyId) const
{
    HCCL_ERROR("[AicpuTsThread][%s] without timeout not support", __func__);
    return HCCL_E_NOT_SUPPORT;
}

// [中文导读] [AllReduce逐行 S232] AicpuTsThread::LocalNotifyRecord的接口声明：硬件通知ID；这些参数属于本函数调用边界。
HcclResult AicpuTsThread::LocalNotifyRecord(uint32_t notifyId) const
// [中文导读] [AllReduce逐行 S233] 进入AicpuTsThread::LocalNotifyRecord函数体：通过线程实现记录 notifyId 并填充本地通知观测槽。
{
    // [中文导读] [AllReduce逐行 S234] 设置设备轻量执行流为/按`GetStreamLitePtr()`；取得A5设备轻量执行流对象地址。
    void* streamLitePtr = GetStreamLitePtr();
    // [中文导读] [AllReduce逐行 S235] 设置设备轻量执行流为/按`static_cast<Hccl::StreamLite*>(streamLitePtr)`（设备轻量执行流）。
    Hccl::StreamLite* streamLite = static_cast<Hccl::StreamLite*>(streamLitePtr);
    // [中文导读] [AllReduce逐行 S236] 设置执行队列对象为/按`streamLite->GetRtsq()`（设备轻量执行流的GetRtsq字段）；返回当前StreamLite持有的具体执行队列。
    Hccl::RtsqBase* rtsq = streamLite->GetRtsq();
    // [中文导读] [AllReduce逐行 S237] 设置当前任务编号为/按`rtsq->GetTaskId()`（执行队列对象的GetTaskId字段）；读取当前队列taskId，用于该操作与观测信息关联。
    u32 taskId = rtsq->GetTaskId();

    // [中文导读] [AllReduce逐行 S239] 生成指定硬件ID的本地通知记录SQE；返回非成功时由检查宏立即向上传递。
    CHK_RET(pImpl_->NotifyRecordLoc(notifyId));

    // [中文导读] [AllReduce逐行 S241] 设置诊断任务信息槽为/按`streamLite->NextTaskSlot()`（设备轻量执行流的NextTaskSlot字段）；取得下一条诊断任务信息槽。
    auto* slot = streamLite->NextTaskSlot();
    // [中文导读] [AllReduce逐行 S242] 设置诊断任务信息槽的taskType字段为/按`Hccl::TaskParamTypeVal::TASK_NOTIFY_RECORD`。
    slot->taskType = Hccl::TaskParamTypeVal::TASK_NOTIFY_RECORD;
    // [中文导读] [AllReduce逐行 S243] 设置诊断任务信息槽的sqId字段为/按`streamLite->GetSqId()`（设备轻量执行流的GetSqId字段）；读取硬件SQ编号。
    slot->sqId = streamLite->GetSqId();
    // [中文导读] [AllReduce逐行 S244] 设置诊断任务信息槽的taskId字段为/按`taskId`（当前任务编号）。
    slot->taskId = taskId;
    // [中文导读] [AllReduce逐行 S245] 设置const void* notifyRecordOpInfo为/按`streamLite->GetLatestDfxOpInfo()`（设备轻量执行流的GetLatestDfxOpInfo字段）；取得当前算子诊断上下文供任务关联。
    const void* notifyRecordOpInfo = streamLite->GetLatestDfxOpInfo();
    // [中文导读] [AllReduce逐行 S246] 设置诊断任务信息槽的dfxOpInfo字段为/按`(notifyRecordOpInfo != nullptr) ? ReinterpretAs<u64>(notifyRecordOpInfo) : DFX_INVALID_U64`。
    slot->dfxOpInfo = (notifyRecordOpInfo != nullptr) ? ReinterpretAs<u64>(notifyRecordOpInfo) : DFX_INVALID_U64;
    // [中文导读] [AllReduce逐行 S247] 设置诊断任务信息槽的linkType字段为/按`Hccl::DfxLinkTypeVal::LINK_ONCHIP`。
    slot->linkType = Hccl::DfxLinkTypeVal::LINK_ONCHIP;
    // [中文导读] [AllReduce逐行 S248] 设置诊断任务信息槽的transportType字段为/按`static_cast<u8>(Hccl::DfxTransportType::DFX_TRANSPORT_TYPE_LOCAL)`。
    slot->transportType = static_cast<u8>(Hccl::DfxTransportType::DFX_TRANSPORT_TYPE_LOCAL);
    // [中文导读] [AllReduce逐行 S249] 设置诊断任务信息槽的channelHandle字段为/按`DFX_INVALID_U64`。
    slot->channelHandle = DFX_INVALID_U64;
    // [中文导读] [AllReduce逐行 S250] 设置诊断任务信息槽的taskPara.Notify.sqeAddr字段为/按`rtsq->GetSqeAddr()`（执行队列对象的GetSqeAddr字段）；取得刚生成SQE对应硬件环队列地址供DFX定位。
    slot->taskPara.Notify.sqeAddr = rtsq->GetSqeAddr();
    // [中文导读] [AllReduce逐行 S251] 记录AicpuTsThread::LocalNotifyRecord的状态/性能诊断，字段包含诊断任务信息槽的Describe字段；日志本身不执行传输。
    PLF_CONFIG_INFO(Hccl::PLF_TASK, "[%s] %s", __func__, slot->Describe().c_str());

    // [中文导读] [AllReduce逐行 S253] 当前路径返回成功状态；仅说明本函数处理/任务组织成功，完成语义由其具体调用职责决定。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S254] 结束AicpuTsThread::LocalNotifyRecord函数体；控制流返回外层。
}

HcclResult
AicpuTsThread::LocalNotifyRecord([[maybe_unused]] ThreadHandle dstThread, [[maybe_unused]] uint32_t dstNotifyIdx) const
{
    HCCL_ERROR("[AicpuTsThread][%s]not support", __func__);
    return HCCL_E_NOT_SUPPORT;
}

// [中文导读] [AllReduce逐行 S263] AicpuTsThread::LocalNotifyWait的接口声明：硬件通知ID、超时秒数；这些参数属于本函数调用边界。
HcclResult AicpuTsThread::LocalNotifyWait(uint32_t notifyId, uint32_t timeout) const
// [中文导读] [AllReduce逐行 S264] 进入AicpuTsThread::LocalNotifyWait函数体：通过线程实现排入有秒级超时的本地通知等待并填充观测槽。
{
    // [中文导读] [AllReduce逐行 S265] 设置设备轻量执行流为/按`GetStreamLitePtr()`；取得A5设备轻量执行流对象地址。
    void* streamLitePtr = GetStreamLitePtr();
    // [中文导读] [AllReduce逐行 S266] 设置设备轻量执行流为/按`static_cast<Hccl::StreamLite*>(streamLitePtr)`（设备轻量执行流）。
    Hccl::StreamLite* streamLite = static_cast<Hccl::StreamLite*>(streamLitePtr);
    // [中文导读] [AllReduce逐行 S267] 设置执行队列对象为/按`streamLite->GetRtsq()`（设备轻量执行流的GetRtsq字段）；返回当前StreamLite持有的具体执行队列。
    Hccl::RtsqBase* rtsq = streamLite->GetRtsq();
    // [中文导读] [AllReduce逐行 S268] 设置当前任务编号为/按`rtsq->GetTaskId()`（执行队列对象的GetTaskId字段）；读取当前队列taskId，用于该操作与观测信息关联。
    u32 taskId = rtsq->GetTaskId();

    // [中文导读] [AllReduce逐行 S270] 生成指定硬件通知ID的等待SQE；返回非成功时由检查宏立即向上传递。
    CHK_RET(pImpl_->NotifyWait(notifyId, timeout));

    // [中文导读] [AllReduce逐行 S272] 设置诊断任务信息槽为/按`streamLite->NextTaskSlot()`（设备轻量执行流的NextTaskSlot字段）；取得下一条诊断任务信息槽。
    auto* slot = streamLite->NextTaskSlot();
    // [中文导读] [AllReduce逐行 S273] 设置诊断任务信息槽的taskType字段为/按`Hccl::TaskParamTypeVal::TASK_NOTIFY_WAIT`。
    slot->taskType = Hccl::TaskParamTypeVal::TASK_NOTIFY_WAIT;
    // [中文导读] [AllReduce逐行 S274] 设置诊断任务信息槽的sqId字段为/按`streamLite->GetSqId()`（设备轻量执行流的GetSqId字段）；读取硬件SQ编号。
    slot->sqId = streamLite->GetSqId();
    // [中文导读] [AllReduce逐行 S275] 设置诊断任务信息槽的taskId字段为/按`taskId`（当前任务编号）。
    slot->taskId = taskId;
    // [中文导读] [AllReduce逐行 S276] 设置const void* notifyWaitOpInfo为/按`streamLite->GetLatestDfxOpInfo()`（设备轻量执行流的GetLatestDfxOpInfo字段）；取得当前算子诊断上下文供任务关联。
    const void* notifyWaitOpInfo = streamLite->GetLatestDfxOpInfo();
    // [中文导读] [AllReduce逐行 S277] 设置诊断任务信息槽的dfxOpInfo字段为/按`(notifyWaitOpInfo != nullptr) ? ReinterpretAs<u64>(notifyWaitOpInfo) : DFX_INVALID_U64`。
    slot->dfxOpInfo = (notifyWaitOpInfo != nullptr) ? ReinterpretAs<u64>(notifyWaitOpInfo) : DFX_INVALID_U64;
    // [中文导读] [AllReduce逐行 S278] 设置诊断任务信息槽的linkType字段为/按`Hccl::DfxLinkTypeVal::LINK_ONCHIP`。
    slot->linkType = Hccl::DfxLinkTypeVal::LINK_ONCHIP;
    // [中文导读] [AllReduce逐行 S279] 设置诊断任务信息槽的transportType字段为/按`static_cast<u8>(Hccl::DfxTransportType::DFX_TRANSPORT_TYPE_LOCAL)`。
    slot->transportType = static_cast<u8>(Hccl::DfxTransportType::DFX_TRANSPORT_TYPE_LOCAL);
    // [中文导读] [AllReduce逐行 S280] 设置诊断任务信息槽的channelHandle字段为/按`DFX_INVALID_U64`。
    slot->channelHandle = DFX_INVALID_U64;
    // [中文导读] [AllReduce逐行 S281] 设置诊断任务信息槽的taskPara.Notify.sqeAddr字段为/按`rtsq->GetSqeAddr()`（执行队列对象的GetSqeAddr字段）；取得刚生成SQE对应硬件环队列地址供DFX定位。
    slot->taskPara.Notify.sqeAddr = rtsq->GetSqeAddr();
    // [中文导读] [AllReduce逐行 S282] 记录AicpuTsThread::LocalNotifyWait的状态/性能诊断，字段包含诊断任务信息槽的Describe字段；日志本身不执行传输。
    PLF_CONFIG_INFO(Hccl::PLF_TASK, "[%s] %s", __func__, slot->Describe().c_str());

    // [中文导读] [AllReduce逐行 S284] 当前路径返回成功状态；仅说明本函数处理/任务组织成功，完成语义由其具体调用职责决定。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S285] 结束AicpuTsThread::LocalNotifyWait函数体；控制流返回外层。
}

HcclResult AicpuTsThread::LocalCopyReport(uint32_t taskId, Hccl::StreamLite* sl, Hccl::RtsqBase* rtsq) const
{
    auto* slot = sl->NextTaskSlot();
    slot->taskType = Hccl::TaskParamTypeVal::TASK_SDMA;
    slot->sqId = sl->GetSqId();
    slot->taskId = taskId;
    const void* copyOpInfo = sl->GetLatestDfxOpInfo();
    slot->dfxOpInfo = (copyOpInfo != nullptr) ? ReinterpretAs<u64>(copyOpInfo) : DFX_INVALID_U64;
    slot->linkType = Hccl::DfxLinkTypeVal::LINK_ONCHIP;
    slot->transportType = static_cast<u8>(Hccl::DfxTransportType::DFX_TRANSPORT_TYPE_LOCAL);
    slot->channelHandle = DFX_INVALID_U64;
    slot->taskPara.Dma.sqeAddr = rtsq->GetSqeAddr();
    PLF_CONFIG_INFO(Hccl::PLF_TASK, "[%s] %s", __func__, slot->Describe().c_str());
    return HCCL_SUCCESS;
}

HcclResult AicpuTsThread::LocalReduceReport(
    void* dst, const void* src, uint64_t size, HcommReduceOp reduceOp, uint32_t taskId, Hccl::StreamLite* sl,
    Hccl::RtsqBase* rtsq) const
{
    auto* slot = sl->NextTaskSlot();
    slot->taskType = Hccl::TaskParamTypeVal::TASK_REDUCE_INLINE;
    slot->sqId = sl->GetSqId();
    slot->taskId = taskId;
    const void* reduceOpInfo = sl->GetLatestDfxOpInfo();
    slot->dfxOpInfo = (reduceOpInfo != nullptr) ? ReinterpretAs<u64>(reduceOpInfo) : DFX_INVALID_U64;
    slot->linkType = Hccl::DfxLinkTypeVal::LINK_ONCHIP;
    slot->transportType = static_cast<u8>(Hccl::DfxTransportType::DFX_TRANSPORT_TYPE_LOCAL);
    slot->channelHandle = DFX_INVALID_U64;
    slot->taskPara.Reduce.sqeAddr = rtsq->GetSqeAddr();
    slot->taskPara.Reduce.srcAddr = ReinterpretAs<u64>(src);
    slot->taskPara.Reduce.dstAddr = ReinterpretAs<u64>(dst);
    slot->taskPara.Reduce.size = size;
    slot->taskPara.Reduce.notifyId = INVALID_U32;
    slot->taskPara.Reduce.reduceOp = static_cast<u8>(reduceOp);
    PLF_CONFIG_INFO(Hccl::PLF_TASK, "[%s] %s", __func__, slot->Describe().c_str());
    return HCCL_SUCCESS;
}

// [中文导读] [AllReduce逐行 S326] AicpuTsThread::LocalCopy的接口声明：操作目标地址、操作源地址、字节容量或单片字节数；这些参数属于本函数调用边界。
HcclResult AicpuTsThread::LocalCopy(void* dst, const void* src, uint64_t size) const
// [中文导读] [AllReduce逐行 S327] 进入AicpuTsThread::LocalCopy函数体：按 SDMA 最大字节长度分片并逐片构造本地拷贝任务与观测。
{
    // [中文导读] [AllReduce逐行 S328] 设置设备轻量执行流为/按`GetStreamLitePtr()`；取得A5设备轻量执行流对象地址。
    void* streamLitePtr = GetStreamLitePtr();
    // [中文导读] [AllReduce逐行 S329] 设置设备轻量执行流为/按`static_cast<Hccl::StreamLite*>(streamLitePtr)`（设备轻量执行流）。
    Hccl::StreamLite* streamLite = static_cast<Hccl::StreamLite*>(streamLitePtr);
    // [中文导读] [AllReduce逐行 S330] 设置执行队列对象为/按`streamLite->GetRtsq()`（设备轻量执行流的GetRtsq字段）；返回当前StreamLite持有的具体执行队列。
    Hccl::RtsqBase* rtsq = streamLite->GetRtsq();

    // [中文导读] [AllReduce逐行 S332] 设置目标地址整数表示为/按`ReinterpretAs<uint64_t>(dst)`（操作目标地址）。
    uint64_t dstAddr = ReinterpretAs<uint64_t>(dst);
    // [中文导读] [AllReduce逐行 S333] 设置源地址整数表示为/按`ReinterpretAs<uint64_t>(src)`（操作源地址）。
    uint64_t srcAddr = ReinterpretAs<uint64_t>(src);
    // [中文导读] [AllReduce逐行 S334] 设置未处理字节数为/按`size`（字节容量或单片字节数）。
    uint64_t remainSize = size;
    // [中文导读] [AllReduce逐行 S335] 设置已经处理的字节偏移为/按`0`。
    uint64_t doneSize = 0;

    // [中文导读] [AllReduce逐行 S337] 在`(remainSize > 0)`（未处理字节数）条件下重复执行后续等待或分片处理。
    while (remainSize > 0) {
        // [中文导读] [AllReduce逐行 S338] 设置当前SDMA分片字节数为/按`remainSize > SDMA_SEND_MAX_SIZE ? SDMA_SEND_MAX_SIZE : remainSize`（未处理字节数）。
        uint64_t realSize = remainSize > SDMA_SEND_MAX_SIZE ? SDMA_SEND_MAX_SIZE : remainSize;
        // [中文导读] [AllReduce逐行 S339] 设置当前任务编号为/按`rtsq->GetTaskId()`（执行队列对象的GetTaskId字段）；读取当前队列taskId，用于该操作与观测信息关联。
        u32 taskId = rtsq->GetTaskId();

        // [中文导读] [AllReduce逐行 S341] 按本端地址/字节长度生成SDMA复制SQE；返回非成功时由检查宏立即向上传递。
        CHK_RET(pImpl_->SdmaCopy(dstAddr + doneSize, srcAddr + doneSize, realSize));
        // [中文导读] [AllReduce逐行 S342] 登记当前本地复制片的任务观测信息；返回非成功时由检查宏立即向上传递。
        CHK_RET(LocalCopyReport(taskId, streamLite, rtsq));

        // [中文导读] [AllReduce逐行 S344] 增加已经处理的字节偏移为/按`realSize`（当前SDMA分片字节数）。
        doneSize += realSize;
        // [中文导读] [AllReduce逐行 S345] 减去未处理字节数为/按`realSize`（当前SDMA分片字节数）。
        remainSize -= realSize;
    // [中文导读] [AllReduce逐行 S346] 结束`while (remainSize > 0)`（未处理字节数）分支/循环；控制流返回外层。
    }
    // [中文导读] [AllReduce逐行 S347] 当前路径返回成功状态；仅说明本函数处理/任务组织成功，完成语义由其具体调用职责决定。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S348] 结束AicpuTsThread::LocalCopy函数体；控制流返回外层。
}

// [中文导读] [AllReduce逐行 S350] AicpuTsThread::LocalReduce的接口声明：按 SDMA 最大长度分片并按指定数据类型/归约操作构造本地归约任务；这些参数属于本函数调用边界。
HcclResult AicpuTsThread::LocalReduce(
    // [中文导读] [AllReduce逐行 S351] AicpuTsThread::LocalReduce的接口声明：操作目标地址、操作源地址、字节容量或单片字节数、元素数据类型、归约操作；这些参数属于本函数调用边界。
    void* dst, const void* src, uint64_t size, HcommDataType dataType, HcommReduceOp reduceOp) const
// [中文导读] [AllReduce逐行 S352] 进入AicpuTsThread::LocalReduce函数体：按 SDMA 最大长度分片并按指定数据类型/归约操作构造本地归约任务。
{
    // [中文导读] [AllReduce逐行 S353] 设置接口数据类型原始枚举为/按`static_cast<uint32_t>(dataType)`（元素数据类型）。
    uint32_t dataTypeRaw = static_cast<uint32_t>(dataType);
    // [中文导读] [AllReduce逐行 S354] 设置设备轻量执行流为/按`GetStreamLitePtr()`；取得A5设备轻量执行流对象地址。
    void* streamLitePtr = GetStreamLitePtr();
    // [中文导读] [AllReduce逐行 S355] 设置设备轻量执行流为/按`static_cast<Hccl::StreamLite*>(streamLitePtr)`（设备轻量执行流）。
    Hccl::StreamLite* streamLite = static_cast<Hccl::StreamLite*>(streamLitePtr);
    // [中文导读] [AllReduce逐行 S356] 设置执行队列对象为/按`streamLite->GetRtsq()`（设备轻量执行流的GetRtsq字段）；返回当前StreamLite持有的具体执行队列。
    Hccl::RtsqBase* rtsq = streamLite->GetRtsq();

    // [中文导读] [AllReduce逐行 S358] 设置目标地址整数表示为/按`ReinterpretAs<uint64_t>(dst)`（操作目标地址）。
    uint64_t dstAddr = ReinterpretAs<uint64_t>(dst);
    // [中文导读] [AllReduce逐行 S359] 设置源地址整数表示为/按`ReinterpretAs<uint64_t>(src)`（操作源地址）。
    uint64_t srcAddr = ReinterpretAs<uint64_t>(src);
    // [中文导读] [AllReduce逐行 S360] 设置未处理字节数为/按`size`（字节容量或单片字节数）。
    uint64_t remainSize = size;
    // [中文导读] [AllReduce逐行 S361] 设置已经处理的字节偏移为/按`0`。
    uint64_t doneSize = 0;

    // [中文导读] [AllReduce逐行 S363] 在`(remainSize > 0)`（未处理字节数）条件下重复执行后续等待或分片处理。
    while (remainSize > 0) {
        // [中文导读] [AllReduce逐行 S364] 设置当前SDMA分片字节数为/按`remainSize > SDMA_SEND_MAX_SIZE ? SDMA_SEND_MAX_SIZE : remainSize`（未处理字节数）。
        uint64_t realSize = remainSize > SDMA_SEND_MAX_SIZE ? SDMA_SEND_MAX_SIZE : remainSize;
        // [中文导读] [AllReduce逐行 S365] 设置当前任务编号为/按`rtsq->GetTaskId()`（执行队列对象的GetTaskId字段）；读取当前队列taskId，用于该操作与观测信息关联。
        u32 taskId = rtsq->GetTaskId();

        // [中文导读] [AllReduce逐行 S367] 按本端地址/字节长度和归约方式生成SDMA归约SQE；返回非成功时由检查宏立即向上传递。
        CHK_RET(pImpl_->SdmaReduce(dstAddr + doneSize, srcAddr + doneSize, realSize, dataTypeRaw, reduceOp));
        // [中文导读] [AllReduce逐行 S368] 登记当前本地归约片的任务观测信息；返回非成功时由检查宏立即向上传递。
        CHK_RET(LocalReduceReport(
            // [中文导读] [AllReduce逐行 S369] 为登记当前本地归约片的任务观测信息补入`static_cast<uint8_t*>(dst) + doneSize, static_cast<const uint8_t*>(src) + doneSize, realSize, reduceOp,`（操作目标地址、已经处理的字节偏移、操作源地址、当前SDMA分片字节数、归约操作）；本行是参数/结构化初始化续行。
            static_cast<uint8_t*>(dst) + doneSize, static_cast<const uint8_t*>(src) + doneSize, realSize, reduceOp,
            // [中文导读] [AllReduce逐行 S370] 为登记当前本地归约片的任务观测信息补入`taskId, streamLite, rtsq))`（当前任务编号、设备轻量执行流、执行队列对象）；本行是参数/结构化初始化续行。
            taskId, streamLite, rtsq));

        // [中文导读] [AllReduce逐行 S372] 增加已经处理的字节偏移为/按`realSize`（当前SDMA分片字节数）。
        doneSize += realSize;
        // [中文导读] [AllReduce逐行 S373] 减去未处理字节数为/按`realSize`（当前SDMA分片字节数）。
        remainSize -= realSize;
    // [中文导读] [AllReduce逐行 S374] 结束`while (remainSize > 0)`（未处理字节数）分支/循环；控制流返回外层。
    }
    // [中文导读] [AllReduce逐行 S375] 当前路径返回成功状态；仅说明本函数处理/任务组织成功，完成语义由其具体调用职责决定。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S376] 结束AicpuTsThread::LocalReduce函数体；控制流返回外层。
}

// Private functions
HcclResult AicpuTsThread::HostInit()
{
    CHK_PRT_RET(
        !uniqueIdStr_.empty(), HCCL_ERROR("[AicpuTsThread][Init]not support init with uniqueId on host"),
        HCCL_E_NOT_SUPPORT);
    s32 deviceLogicId;
    CHK_RET(hrtGetDevice(&deviceLogicId));
    CHK_RET(hrtGetDevicePhyIdByIndex(static_cast<uint32_t>(deviceLogicId), devId_));
    CHK_RET(hrtGetDeviceType(devType_));
    if (rtStream_ == nullptr) {
        stream_.reset(new (std::nothrow) Stream(streamType_));
        CHK_SMART_PTR_NULL(stream_);
        rtStream_ = stream_->ptr();
    }

    for (uint32_t idx = 0; idx < notifyNum_; idx++) {
        notifys_.emplace_back(nullptr);
        notifys_[idx].reset(new (std::nothrow) LocalNotify());
        CHK_SMART_PTR_NULL(notifys_[idx]);
        CHK_RET(notifys_[idx]->Init(notifyLoadType_));
        if (devType_ != DevType::DEV_TYPE_950 && devType_ != DevType::DEV_TYPE_960) {
            CHK_RET(notifys_[idx]->SetIpc());
        }
    }

    if (streamType_ == StreamType::STREAM_TYPE_DEVICE && devType_ != DevType::DEV_TYPE_950
        && devType_ != DevType::DEV_TYPE_960) {
        uint64_t size = sizeof(SqCqeContext);
        sqCqeContext_ = DeviceMem::alloc(size);
        CHK_PTR_NULL(sqCqeContext_.ptr());
        CHK_RET(hrtMemSet(sqCqeContext_.ptr(), size, size));
    }
    return HCCL_SUCCESS;
}

HcclResult AicpuTsThread::DeviceInit()
{
    CHK_PRT_RET(uniqueIdStr_.empty(), HCCL_ERROR("[AicpuTsThread][Init]uniqueIdStr is empty"), HCCL_E_INTERNAL);
    std::istringstream iss(uniqueIdStr_);
    CHK_RET(hrtGetDeviceType(devType_));
    uint32_t hostPhyId = 0;
    iss.read(ReinterpretAs<char_t*>(&streamType_), sizeof(streamType_));
    iss.read(ReinterpretAs<char_t*>(&notifyLoadType_), sizeof(notifyLoadType_));
    HCCL_INFO("[AicpuTsThread][Init]streamType[%d], notifyLoadType[%d].", streamType_, notifyLoadType_);
    iss.read(ReinterpretAs<char_t*>(&hostPhyId), sizeof(hostPhyId));
    CHK_RET(hrtDrvGetLocalDevIDByHostDevID(hostPhyId, &devId_));
    iss.read(ReinterpretAs<char_t*>(&notifyNum_), sizeof(notifyNum_));

    HcclStreamParam streamParam;
    iss.read(ReinterpretAs<char_t*>(&streamParam), sizeof(streamParam));
    // 91095初始化streamlite，初始化rtsq接口
    if (devType_ == DevType::DEV_TYPE_950 || devType_ == DevType::DEV_TYPE_960) {
        CHK_RET(InitStreamLite(streamParam.streamInfo, hostPhyId));
    } else {
        CHK_RET(InitStream(streamParam));
    }

    notifys_.reserve(notifyNum_);

    for (uint32_t idx = 0; idx < notifyNum_; idx++) {
        notifys_.emplace_back(nullptr);
        HcclSignalInfo notifyInfo;
        iss.read(ReinterpretAs<char_t*>(&notifyInfo), sizeof(notifyInfo));
        notifys_[idx].reset(new (std::nothrow) LocalNotify());
        CHK_SMART_PTR_NULL(notifys_[idx]);
        if (devType_ == DevType::DEV_TYPE_950 || devType_ == DevType::DEV_TYPE_960) {
            CHK_RET(notifys_[idx]->InitNotifyLite(notifyInfo));
            HCCL_INFO(
                "[AicpuTsThread][Init]local notifyLite init success, resId[%u], devId[%u]", notifyInfo.resId,
                notifyInfo.devId);
        } else {
            CHK_RET(notifys_[idx]->Init(notifyInfo, notifyLoadType_));
            HCCL_INFO(
                "[AicpuTsThread][Init]local notifyLite init success, resId[%u], tsId:%d, devId[%u]", notifyInfo.resId,
                notifyInfo.tsId, notifyInfo.devId);
        }
    }

    return HCCL_SUCCESS;
}

HcclResult AicpuTsThread::GetSqHeadAndTail([[maybe_unused]] uint32_t& sqHead, [[maybe_unused]] uint32_t& sqTail)
{
#ifdef CCL_KERNEL_AICPU

    uint32_t sqIds = pImpl_->GetSqId();

    CHK_RET(QuerySqStatusByType(devId_, sqIds, DRV_SQCQ_PROP_SQ_TAIL, sqTail));
    CHK_RET(QuerySqStatusByType(devId_, sqIds, DRV_SQCQ_PROP_SQ_HEAD, sqHead));
#endif
    return HCCL_SUCCESS;
}

bool AicpuTsThread::GetMaster() const { return isMaster_; }

void AicpuTsThread::SetIsMaster(bool isMaster) { isMaster_ = isMaster; }

HcclResult AicpuTsThread::SupplementNotify(uint32_t notifyNum)
{
    HCCL_INFO("[%s]supplement notifyNum[%u], notifyNum_[%u]", __func__, notifyNum, notifyNum_);
    const u32 beginIdx = notifyNum_;
    const u32 allNotifyNum = notifyNum_ + notifyNum;
    notifys_.resize(allNotifyNum);

    for (uint32_t idx = beginIdx; idx < allNotifyNum; idx++) {
        notifys_[idx].reset(new (std::nothrow) LocalNotify());
        CHK_SMART_PTR_NULL(notifys_[idx]);
        CHK_RET(notifys_[idx]->Init(notifyLoadType_));
        if (devType_ != DevType::DEV_TYPE_950 && devType_ != DevType::DEV_TYPE_960) {
            CHK_RET(notifys_[idx]->SetIpc());
        }
        notifyNum_++;
    }

    uniqueIdStr_.clear();
    UpdateUniqueId();
    return HCCL_SUCCESS;
}

HcclResult AicpuTsThread::GetNotifyByUniqueId(u32& notifyNum, std::string& notifyDesc)
{
    CHK_PRT_RET(
        uniqueIdStr_.empty(), HCCL_ERROR("[AicpuTsThread][GetNotifyByUniqueId]uniqueIdStr is empty"), HCCL_E_INTERNAL);
    std::istringstream iss(uniqueIdStr_);
    StreamType streamType = StreamType::STREAM_TYPE_RESERVED;
    NotifyLoadType notifyLoadType = NotifyLoadType::HOST_NOTIFY;
    uint32_t hostPhyId = 0;
    HcclStreamParam streamParam;
    iss.read(ReinterpretAs<char_t*>(&streamType), sizeof(streamType));
    iss.read(ReinterpretAs<char_t*>(&notifyLoadType), sizeof(notifyLoadType));
    iss.read(ReinterpretAs<char_t*>(&hostPhyId), sizeof(hostPhyId));
    iss.read(ReinterpretAs<char_t*>(&notifyNum), sizeof(notifyNum));
    iss.read(ReinterpretAs<char_t*>(&streamParam), sizeof(streamParam));

    // 序列化信息
    std::ostringstream oss;
    for (uint32_t idx = 0; idx < notifyNum; idx++) {
        HcclSignalInfo notifyInfo;
        iss.read(ReinterpretAs<char_t*>(&notifyInfo), sizeof(notifyInfo));
        HCCL_INFO(
            "[AicpuTsThread][%s]get local notify data success, resId[%u], tsId:%d, devId[%u]", __func__,
            notifyInfo.resId, notifyInfo.tsId, notifyInfo.devId);
        oss.write(ReinterpretAs<const char_t*>(&notifyInfo), sizeof(notifyInfo));
    }

    notifyDesc = oss.str();
    return HCCL_SUCCESS;
}

HcclResult AicpuTsThread::SupplementNotify(u32 notifyNum, const std::string& notifyDesc)
{
    if (notifyNum <= notifyNum_) {
        HCCL_WARNING("[%s]supplement notifyNum[%u], notifyNum_[%u]", __func__, notifyNum, notifyNum_);
        return HCCL_SUCCESS;
    }
    HCCL_INFO("[%s]supplement notifyNum[%u], notifyNum_[%u]", __func__, notifyNum, notifyNum_);

    std::istringstream iss(notifyDesc);
    const u32 beginIdx = notifyNum_;
    notifys_.resize(notifyNum);
    for (uint32_t idx = 0; idx < beginIdx; idx++) {
        HcclSignalInfo notifyInfo;
        iss.read(ReinterpretAs<char_t*>(&notifyInfo), sizeof(notifyInfo));
        HCCL_INFO(
            "[AicpuTsThread][SupplementNotify]skip init, resId[%u], tsId:%d, devId[%u]", notifyInfo.resId,
            notifyInfo.tsId, notifyInfo.devId);
    }

    for (uint32_t idx = beginIdx; idx < notifyNum; idx++) {
        HcclSignalInfo notifyInfo;
        iss.read(ReinterpretAs<char_t*>(&notifyInfo), sizeof(notifyInfo));
        notifys_[idx].reset(new (std::nothrow) LocalNotify());
        CHK_SMART_PTR_NULL(notifys_[idx]);
        if (devType_ == DevType::DEV_TYPE_950 || devType_ == DevType::DEV_TYPE_960) {
            CHK_RET(notifys_[idx]->InitNotifyLite(notifyInfo));
            HCCL_INFO(
                "[AicpuTsThread][SupplementNotify]local notifyLite init success, resId[%u], devId[%u]",
                notifyInfo.resId, notifyInfo.devId);
        } else {
            CHK_RET(notifys_[idx]->Init(notifyInfo, notifyLoadType_));
            HCCL_INFO(
                "[AicpuTsThread][SupplementNotify]local notifyLite init success, resId[%u], tsId:%d, devId[%u]",
                notifyInfo.resId, notifyInfo.tsId, notifyInfo.devId);
        }
        notifyNum_++;
    }
    return HCCL_SUCCESS;
}

HcclResult AicpuTsThread::SetCheckExecStatusCallback(std::function<HcclResult(bool)> callback)
{
    Hccl::StreamLite* streamLite = static_cast<Hccl::StreamLite*>(GetStreamLitePtr());
    CHK_PTR_NULL(streamLite);
    Hccl::RtsqBase* rtsq = streamLite->GetRtsq();
    CHK_PTR_NULL(rtsq);
    rtsq->SetCheckExecStatusCallback(callback);
    return HCCL_SUCCESS;
}

Hccl::TaskInfoCircularQueue* AicpuTsThread::GetTaskInfos() const
{
    Hccl::StreamLite* streamLite = static_cast<Hccl::StreamLite*>(GetStreamLitePtr());
    if (streamLite == nullptr) {
        return nullptr;
    }
    return streamLite->GetTaskInfos();
}

HcclResult AicpuTsThread::GetTaskInfoCount(u32& count) const
{
    Hccl::StreamLite* streamLite = static_cast<Hccl::StreamLite*>(GetStreamLitePtr());
    CHK_PTR_NULL(streamLite);
    Hccl::TaskInfoCircularQueue* taskInfos = streamLite->GetTaskInfos();
    count = static_cast<u32>(taskInfos->GetCount());
    return HCCL_SUCCESS;
}

void AicpuTsThread::SetReportStreamTaskCallback(std::function<void(Hccl::TaskInfoCircularQueue*)> callback) const
{
    Hccl::StreamLite* streamLite = static_cast<Hccl::StreamLite*>(GetStreamLitePtr());
    if (streamLite != nullptr) {
        streamLite->SetReportStreamTaskCallback(std::move(callback));
    }
}

void AicpuTsThread::SetGetLatestDfxOpInfoCallback(std::function<const void*()> callback) const
{
    Hccl::StreamLite* streamLite = static_cast<Hccl::StreamLite*>(GetStreamLitePtr());
    if (streamLite != nullptr) {
        streamLite->SetGetLatestDfxOpInfoCallback(std::move(callback));
    }
}
} // namespace hccl
