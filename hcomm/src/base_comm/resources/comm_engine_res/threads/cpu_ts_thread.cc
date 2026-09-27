/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cpu_ts_thread.h"
#include "hccl_common.h"
#include "adapter_rts.h"
#include "dfx_dlprof_function.h"

namespace hccl {
const std::unordered_map<HcclDataType, aclDataType> hccl2rtDataTypeMap = {
    {HCCL_DATA_TYPE_INT8, ACL_INT8},    {HCCL_DATA_TYPE_INT16, ACL_INT16}, {HCCL_DATA_TYPE_INT32, ACL_INT32},
    {HCCL_DATA_TYPE_FP16, ACL_FLOAT16}, {HCCL_DATA_TYPE_FP32, ACL_FLOAT},  {HCCL_DATA_TYPE_BFP16, ACL_BF16},
};

const std::unordered_map<HcclReduceOp, aclrtReduceKind> hccl2rtReduceOpMap = {
    {HCCL_REDUCE_SUM, ACL_RT_MEMCPY_SDMA_AUTOMATIC_SUM},
    {HCCL_REDUCE_MAX, ACL_RT_MEMCPY_SDMA_AUTOMATIC_MAX},
    {HCCL_REDUCE_MIN, ACL_RT_MEMCPY_SDMA_AUTOMATIC_MIN},
};

CpuTsThread::CpuTsThread(rtStream_t rtStream, uint32_t notifyNum, const NotifyLoadType notifyLoadType)
    : rtStream_(rtStream),
      notifyNum_(notifyNum),
      notifyLoadType_(notifyLoadType)
{}

CpuTsThread::CpuTsThread(StreamType streamType, uint32_t notifyNum, const NotifyLoadType notifyLoadType)
    : streamType_(streamType),
      notifyNum_(notifyNum),
      notifyLoadType_(notifyLoadType)
{}

CpuTsThread::~CpuTsThread() { DeInitImpl(); }

HcclResult CpuTsThread::Init()
{
    // Host 侧初始化
    CHK_RET(GetRunSideIsDevice(isDeviceSide_));
    CHK_RET(hrtGetDeviceType(devType_));
    if (!isDeviceSide_) {
        s32 deviceLogicId;
        CHK_RET(hrtGetDevice(&deviceLogicId));
        CHK_RET(hrtGetDevicePhyIdByIndex(static_cast<uint32_t>(deviceLogicId), devId_));
        if (streamType_ == StreamType::STREAM_TYPE_DEVICE || notifyLoadType_ == NotifyLoadType::DEVICE_NOTIFY) {
            return HCCL_E_NOT_SUPPORT;
        }
        if (rtStream_ == nullptr) {
            stream_.reset(new (std::nothrow) Stream(streamType_));
            CHK_SMART_PTR_NULL(stream_);
            rtStream_ = stream_->ptr();
        } else {
            stream_.reset(new (std::nothrow) Stream(rtStream_));
            CHK_SMART_PTR_NULL(stream_);
        }
        notifys_.reserve(notifyNum_);
        for (uint32_t idx = 0; idx < notifyNum_; idx++) {
            notifys_.emplace_back(nullptr);
            notifys_[idx].reset(new (std::nothrow) LocalNotify());
            CHK_SMART_PTR_NULL(notifys_[idx]);
            CHK_RET(notifys_[idx]->Init(notifyLoadType_));
            if (devType_ != DevType::DEV_TYPE_950 && devType_ != DevType::DEV_TYPE_960) {
                CHK_RET(notifys_[idx]->SetIpc());
            }
        }
        return HCCL_SUCCESS;
    } else {
        return HCCL_E_NOT_SUPPORT;
    }
}

HcclResult CpuTsThread::DeInit() { return DeInitImpl(); }

HcclResult CpuTsThread::DeInitImpl()
{
    streamType_ = StreamType::STREAM_TYPE_RESERVED;
    notifyNum_ = 0;
    stream_ = nullptr;
    notifys_.clear();
    return HCCL_SUCCESS;
}

std::string& CpuTsThread::GetUniqueId()
{
    if (!uniqueIdStr_.empty()) {
        return uniqueIdStr_;
    }
    return UpdateUniqueId();
}

std::string& CpuTsThread::UpdateUniqueId()
{
    // 序列化信息
    uniqueIdStr_ = std::string();
    std::ostringstream oss;
    StreamType streamType = StreamType::STREAM_TYPE_DEVICE;
    oss.write(reinterpret_cast<const char_t*>(&streamType), sizeof(streamType));
    oss.write(reinterpret_cast<const char_t*>(&notifyLoadType_), sizeof(notifyLoadType_));
    oss.write(reinterpret_cast<const char_t*>(&devId_), sizeof(devId_));
    oss.write(reinterpret_cast<const char_t*>(&notifyNum_), sizeof(notifyNum_));

    // 临时申请一条流，用于在device侧资源展开时initStream
    if (streamDevice_ == nullptr) {
        streamDevice_.reset(new (std::nothrow) Stream(streamType));
    }
    if (streamDevice_ == nullptr) {
        HCCL_ERROR("[CpuTsThread][%s]reset stream failed, stream type[%d]", __func__, streamType);
        return uniqueIdStr_;
    }

    uint64_t size = sizeof(SqCqeContext);
    if (sqCqeContext_.ptr() == nullptr) {
        sqCqeContext_ = DeviceMem::alloc(size);
    }
    if (sqCqeContext_.ptr() == nullptr) {
        HCCL_ERROR("[CpuTsThread][%s]alloc mem failed, mem size[%llu]", __func__, size);
        return uniqueIdStr_;
    }
    HcclResult ret = hrtMemSet(sqCqeContext_.ptr(), size, size);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[CpuTsThread][%s]mem set failed, mem size[%llu], ptr[%p]", __func__, size, sqCqeContext_.ptr());
        return uniqueIdStr_;
    }

    HcclStreamParam streamParam;
    streamParam.streamInfo.streamIds = streamDevice_->id();
    streamParam.streamInfo.sqIds = streamDevice_->sqId();
    streamParam.streamInfo.cqIds = streamDevice_->cqId();
    streamParam.streamInfo.logicCqids = streamDevice_->logicCqId();
    streamParam.sqCqContextAddr = reinterpret_cast<uint64_t>(sqCqeContext_.ptr());
    streamParam.sqCqContextSize = sqCqeContext_.size();
    oss.write(reinterpret_cast<const char_t*>(&streamParam), sizeof(streamParam));

    ret = HCCL_SUCCESS;
    for (uint32_t idx = 0; idx < notifyNum_; idx++) {
        HcclSignalInfo notifyInfo;
        ret = notifys_[idx]->GetNotifyData(notifyInfo);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("[AicpuTsThread][UpdateUniqueId]GetNotifyData failed, ret[%d]", ret);
            uniqueIdStr_ = std::string();
            return uniqueIdStr_;
        }
        HCCL_INFO(
            "[AicpuTsThread][UpdateUniqueId]get local notify data success, resId[%u], tsId:%d, devId[%u]",
            notifyInfo.resId, notifyInfo.tsId, notifyInfo.devId);
        oss.write(reinterpret_cast<const char_t*>(&notifyInfo), sizeof(notifyInfo));
    }
    HCCL_DEBUG("[AicpuTsThread][UpdateUniqueId] stream[%p], notifyNum[%u]", stream_->ptr(), notifyNum_);

    uniqueIdStr_ = oss.str();
    return uniqueIdStr_;
}

uint32_t CpuTsThread::GetNotifyNum() const { return notifyNum_; }

LocalNotify* CpuTsThread::GetNotify(uint32_t index) const
{
    if (index >= notifyNum_) {
        HCCL_ERROR(
            "[CpuTsThread][GetNotify] notifyNum[%u], index[%u] out of range[0, %u]", notifyNum_, index, notifyNum_ - 1);
        return nullptr;
    }
    return notifys_[index].get();
}

bool CpuTsThread::IsDeviceA5() const { return devType_ == DevType::DEV_TYPE_950 || devType_ == DevType::DEV_TYPE_960; }

// A3 Stream
Stream* CpuTsThread::GetStream() const { return stream_.get(); }

// A5 Stream
void* CpuTsThread::GetStreamLitePtr() const
{
    return nullptr; // Not implemented
}

void CpuTsThread::LaunchTask() const { return; }

void CpuTsThread::TryLaunchTask() const
{
    HCCL_DEBUG("[%s] CpuTsThread does not support TryLaunchTask, skip", __func__);
    return;
}

// Local Data Plane Functions
HcclResult CpuTsThread::LocalNotifyRecord([[maybe_unused]] uint32_t notifyId) const
{
    HCCL_ERROR("[CpuTsThread][%s]not support", __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult CpuTsThread::LocalNotifyWait([[maybe_unused]] uint32_t notifyId) const
{
    HCCL_ERROR("[CpuTsThread][%s]not support", __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult
CpuTsThread::LocalNotifyRecord([[maybe_unused]] ThreadHandle dstThread, [[maybe_unused]] uint32_t dstNotifyIdx) const
{
#ifndef CCL_KERNEL_AICPU
    u64 beginTime = Hccl::DfxDlProfFunction::GetInstance().dlMsprofSysCycleTime();
    HCCL_INFO("[%s]dstThread[0x%llu], dstNotifyIdx[%u].", __func__, dstThread, dstNotifyIdx);
    CHK_PRT_RET(
        !IsDeviceA5(), HCCL_ERROR("[CpuTsThread][%s]only support A5", __func__),
        HCCL_E_NOT_SUPPORT); // 只支持A5, 其他场景调用HcclLocalNotifyRecord

    Stream* stream = GetStream();
    CHK_PTR_NULL(stream);
    Thread* const dstThreadPtr = reinterpret_cast<Thread*>(dstThread);
    CHK_PTR_NULL(dstThreadPtr);
    LocalNotify* dstNotify = dstThreadPtr->GetNotify(dstNotifyIdx);
    CHK_PTR_NULL(dstNotify);

    HcclResult ret = dstNotify->Post(*stream);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[%s]fail, dstThread[0x%llx], dstNotifyIdx[%u].", __func__, dstThread, dstNotifyIdx), ret);

    HcclSignalInfo signalInfo;
    CHK_RET(dstNotify->GetNotifyData(signalInfo));
    CHK_RET(ReportHostNotifyRecordTask(signalInfo.resId, beginTime, isMaster_));
#endif
    return HCCL_SUCCESS;
}

HcclResult CpuTsThread::LocalNotifyWait([[maybe_unused]] uint32_t notifyIdx, [[maybe_unused]] uint32_t timeOut) const
{
#ifndef CCL_KERNEL_AICPU
    u64 beginTime = Hccl::DfxDlProfFunction::GetInstance().dlMsprofSysCycleTime();
    HCCL_INFO("[%s]notifyIdx[%u], timeOut[%u s].", __func__, notifyIdx, timeOut);
    CHK_PRT_RET(
        !IsDeviceA5(), HCCL_ERROR("[CpuTsThread][%s]only support A5", __func__),
        HCCL_E_NOT_SUPPORT); // 只支持A5, 其他场景调用HcclLocalNotifyWait

    Stream* stream = GetStream();
    CHK_PTR_NULL(stream);
    LocalNotify* notify = GetNotify(notifyIdx);
    CHK_PTR_NULL(notify);

    HcclResult ret = notify->Wait(*stream, timeOut);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS, HCCL_ERROR("[%s]fail, notifyIdx[%u], timeOut[%u s].", __func__, notifyIdx, timeOut), ret);

    HcclSignalInfo signalInfo;
    CHK_RET(notify->GetNotifyData(signalInfo));
    CHK_RET(ReportHostNotifyWaitTask(signalInfo.resId, beginTime, isMaster_));
#endif
    return HCCL_SUCCESS;
}

HcclResult CpuTsThread::LocalCopy(
    [[maybe_unused]] void* dst, [[maybe_unused]] const void* src, [[maybe_unused]] uint64_t sizeByte) const
{
#ifndef CCL_KERNEL_AICPU
    u64 beginTime = Hccl::DfxDlProfFunction::GetInstance().dlMsprofSysCycleTime();
    HCCL_INFO("[%s]dst[%p], src[%p], sizeByte[%llu].", __func__, dst, src, sizeByte);
    CHK_PRT_RET(
        !IsDeviceA5(), HCCL_ERROR("[CpuTsThread][%s]only support A5", __func__),
        HCCL_E_NOT_SUPPORT); // 只支持A5, 其他场景调用HcclLocalCopy

    if (sizeByte == 0 || src == dst) {
        HCCL_INFO("[CpuTsThread][%s]skip, dst[%p] equals src[%p] or len[%llu] equals 0", __func__, dst, src, sizeByte);
        return HCCL_SUCCESS;
    }

    Stream* stream = GetStream();
    CHK_PTR_NULL(stream);
    CHK_RET(hrtMemAsyncCopy(
        dst, sizeByte, src, sizeByte, HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_DEVICE_TO_DEVICE, stream->ptr()));

    CHK_RET(ReportHostLocalCopyTask(dst, src, sizeByte, beginTime, isMaster_));
#endif
    return HCCL_SUCCESS;
}

HcclResult CpuTsThread::LocalReduce(
    [[maybe_unused]] void* dst, [[maybe_unused]] const void* src, [[maybe_unused]] uint64_t sizeByte,
    [[maybe_unused]] HcommDataType dataType, [[maybe_unused]] HcommReduceOp reduceOp) const
{
#ifndef CCL_KERNEL_AICPU
    u64 beginTime = Hccl::DfxDlProfFunction::GetInstance().dlMsprofSysCycleTime();
    HCCL_INFO(
        "[%s]dst[%p], src[%p], sizeByte[%llu], dataType[%d], reduceOp[%d].", __func__, dst, src, sizeByte, dataType,
        reduceOp);
    CHK_PRT_RET(
        !IsDeviceA5(), HCCL_ERROR("[CpuTsThread][%s]only support A5", __func__),
        HCCL_E_NOT_SUPPORT); // 只支持A5, 其他场景调用HcclLocalCopyReduce

    auto dataTypeIt = hccl2rtDataTypeMap.find(static_cast<HcclDataType>(dataType));
    if (dataTypeIt == hccl2rtDataTypeMap.end()) {
        HCCL_ERROR(
            "[%s]data type[%s] is not supported", __func__,
            GetDataTypeEnumStr(static_cast<HcclDataType>(dataType)).c_str());
        return HCCL_E_PARA;
    }

    auto reduceOpIt = hccl2rtReduceOpMap.find(static_cast<HcclReduceOp>(reduceOp));
    if (reduceOpIt == hccl2rtReduceOpMap.end()) {
        HCCL_ERROR(
            "[%s]reduceOp[%s] is not supported", __func__,
            GetReduceOpEnumStr(static_cast<HcclReduceOp>(reduceOp)).c_str());
        return HCCL_E_PARA;
    }

    Stream* stream = GetStream();
    CHK_PTR_NULL(stream);
    CHK_RET(hrtReduceAsync(dst, sizeByte, src, sizeByte, reduceOpIt->second, dataTypeIt->second, stream->ptr()));
    CHK_RET(ReportHostLocalReduceTask(dst, src, sizeByte, dataType, reduceOp, beginTime, isMaster_));
#endif
    return HCCL_SUCCESS;
}
bool CpuTsThread::GetMaster() const { return isMaster_; }

void CpuTsThread::SetIsMaster(bool isMaster) { isMaster_ = isMaster; }

HcclResult CpuTsThread::SupplementNotify(uint32_t notifyNum)
{
    if (streamType_ == StreamType::STREAM_TYPE_DEVICE || notifyLoadType_ == NotifyLoadType::DEVICE_NOTIFY) {
        HCCL_ERROR("[%s]Does not support this interface.", __func__);
        return HCCL_E_NOT_SUPPORT;
    }
    HCCL_INFO("[%s]supplement notifyNum[%u], notifyNum_[%u]", __func__, notifyNum, notifyNum_);

    u32 currentNotifyNum = notifyNum_;
    notifyNum_ += notifyNum;
    notifys_.reserve(notifyNum_);
    for (uint32_t idx = currentNotifyNum; idx < notifyNum_; idx++) {
        notifys_.emplace_back(nullptr);
        notifys_[idx].reset(new (std::nothrow) LocalNotify());
        CHK_SMART_PTR_NULL(notifys_[idx]);
        CHK_RET(notifys_[idx]->Init(notifyLoadType_));
        if (devType_ != DevType::DEV_TYPE_950 && devType_ != DevType::DEV_TYPE_960) {
            CHK_RET(notifys_[idx]->SetIpc());
        }
    }

    uniqueIdStr_.clear();
    UpdateUniqueId();
    return HCCL_SUCCESS;
}
} // namespace hccl
