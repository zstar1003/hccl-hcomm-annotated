/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "adapter_rts.h"
#include "adapter_error_manager.h"
#include "sal.h"
#include "stream_pub.h"

namespace hccl {
// 默认构造函数只产生无效的Stream对象
Stream::Stream()
    : stream_(nullptr),
      device_id_(HCCL_DEVICE_NOT_SET),
      stream_owner_(false),
      streamId_(0),
      isMainStream_(true),
      modeGotFlag_(false),
      streamMode_(0),
      sqId_(0),
      ctx_(nullptr),
      cqId_(0),
      logicCqid_(0)
{}

Stream::Stream(const Stream& that)
    : stream_(that.ptr()),
      device_id_(that.device_id_),
      stream_owner_(false),
      streamId_(that.streamId_),
      isMainStream_(that.isMainStream_),
      modeGotFlag_(that.modeGotFlag_),
      streamMode_(that.streamMode_),
      sqId_(that.sqId_),
      ctx_(that.ctx_),
      cqId_(that.cqId_),
      logicCqid_(that.logicCqid_),
      sqeContext_(that.sqeContext_),
      cqeContext_(that.cqeContext_),
      streamInfo_(that.streamInfo_),
      invalidFlag_(that.invalidFlag_)
{} // 共享销毁标志, 副本与owner持有同一份

Stream::Stream(Stream&& that)
    : stream_(that.ptr()),
      device_id_(that.device_id_),
      stream_owner_(that.stream_owner_),
      streamId_(that.streamId_),
      isMainStream_(that.isMainStream_),
      modeGotFlag_(that.modeGotFlag_),
      streamMode_(that.streamMode_),
      sqId_(that.sqId_),
      ctx_(that.ctx_),
      cqId_(that.cqId_),
      logicCqid_(that.logicCqid_),
      sqeContext_(that.sqeContext_),
      cqeContext_(that.cqeContext_),
      streamInfo_(that.streamInfo_),
      invalidFlag_(std::move(that.invalidFlag_)) // 转移销毁标志所有权, 源置空避免源析构影响
{
    that.stream_ = nullptr;
    that.device_id_ = HCCL_DEVICE_NOT_SET;
    that.stream_owner_ = false;
    that.streamId_ = 0;
    that.isMainStream_ = true;
    that.modeGotFlag_ = false;
    that.streamMode_ = 0;
    that.sqId_ = 0;
    that.cqId_ = 0;
    that.logicCqid_ = 0;
    that.sqeContext_ = nullptr;
    that.cqeContext_ = nullptr;
    that.streamInfo_.actualStreamId = 0;
    that.streamInfo_.logicCqId = 0;
    that.streamInfo_.sqBaseAddr = nullptr;
    that.streamInfo_.sqDepth = 0;
    that.streamInfo_.sqId = 0;
}

Stream::Stream(const StreamType streamType, bool isMainStream)
    : stream_(nullptr),
      device_id_(HCCL_DEVICE_NOT_SET),
      stream_owner_(true),
      streamId_(0),
      isMainStream_(isMainStream),
      modeGotFlag_(false),
      streamMode_(0),
      sqId_(0),
      ctx_(nullptr),
      cqId_(0),
      logicCqid_(0)
{
    HcclResult ret;
    aclrtStream rtStream = nullptr;

    // 申请rtStream
    if (streamType == StreamType::STREAM_TYPE_ONLINE) {
        ret = hrtStreamCreateWithFlags(
            &rtStream, HCCL_STREAM_PRIORITY_HIGH, ACL_STREAM_FAST_LAUNCH | ACL_STREAM_FAST_SYNC);
    } else if (streamType == StreamType::STREAM_TYPE_DEVICE) {
        ret = hrtStreamCreateWithFlags(&rtStream, HCCL_STREAM_PRIORITY_HIGH, ACL_STREAM_DEVICE_USE_ONLY);
    } else {
        ret = hrtStreamCreateWithFlags(&rtStream, HCCL_STREAM_PRIORITY_LOW, ACL_STREAM_PERSISTENT);
    }

    if (ret == HCCL_SUCCESS) {
        HCCL_DEBUG("rtStreamCreate ok, streamType[%d]", streamType);
        stream_ = const_cast<void*>(rtStream);
        InitStream();
        HCCL_INFO(
            "Construct stream by stream type success, ptr[%p] ctx[%p], stream id[%d], cqId[%d], logicCqid[%d]",
            rtStream, ctx_, streamId_, cqId_, logicCqid_);
    } else {
        RPT_ENV_ERR(
            true, "EI0007", std::vector<std::string>({"resource_type", "resource_info"}),
            std::vector<std::string>(
                {"stream", std::string("StreamCreateWithFlags, streamType:") + std::to_string(uint32_t(streamType))}));
        HCCL_ERROR(
            "[%s][%s]Construct stream by stream type failed, errNo[0x%016llx] rtStreamCreate error, ret[%d]",
            LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_RESOURCE.c_str(), HCCL_ERROR_CODE(HCCL_E_RUNTIME), ret);
    }
    bool isSupportV2 = false;
    CHK_PRT_CONT(hrtGetHcclV2Support(&isSupportV2), HCCL_WARNING("[Stream] Can not check hccl version"));
    if (isSupportV2 && streamType == StreamType::STREAM_TYPE_ONLINE) {
        HcclResult setModeRet = hrtStreamSetMode(stream_, STREAM_MODE_STOP_ON_FAILURE);
        if (setModeRet != HCCL_SUCCESS) {
            HCCL_ERROR(
                "[Stream][SetMode]Failed to set stream mode, errNo[0x%016llx], ret[%d], stream id[%d]",
                HCCL_ERROR_CODE(setModeRet), setModeRet, streamId_);
        }
    }
}

Stream::Stream(const rtStream_t rtStream, bool isMainStream)
    : stream_(const_cast<void*>(rtStream)),
      device_id_(HCCL_DEVICE_NOT_SET),
      stream_owner_(false),
      streamId_(0),
      isMainStream_(isMainStream),
      modeGotFlag_(false),
      streamMode_(0),
      sqId_(0),
      ctx_(nullptr),
      cqId_(0),
      logicCqid_(0)
{
    InitStream();
}

Stream::Stream(const HcclComStreamInfo& streamInfo, bool isMainStream)
    : stream_(static_cast<void*>(streamInfo.sqBaseAddr)),
      device_id_(HCCL_DEVICE_NOT_SET),
      stream_owner_(false),
      streamId_(streamInfo.actualStreamId),
      isMainStream_(isMainStream),
      modeGotFlag_(false),
      streamMode_(0),
      sqId_(streamInfo.sqId),
      ctx_(nullptr)
{
    SetStreamInfo(streamInfo);
}

Stream::~Stream() { DestroyStream(); }

Stream& Stream::operator=(const Stream& that)
{
    if (&that != this) {
        stream_ = that.ptr();
        device_id_ = that.device_id_;
        stream_owner_ = false;
        taskLogicInfo_ = that.taskLogicInfo_;
        streamId_ = that.streamId_;
        isMainStream_ = that.isMainStream_;
        modeGotFlag_ = that.modeGotFlag_;
        streamMode_ = that.streamMode_;
        sqId_ = that.sqId_;
        ctx_ = that.ctx_;
        cqId_ = that.cqId_;
        logicCqid_ = that.logicCqid_;
        sqeContext_ = that.sqeContext_;
        cqeContext_ = that.cqeContext_;
        streamInfo_.actualStreamId = that.streamInfo_.actualStreamId;
        streamInfo_.logicCqId = that.streamInfo_.logicCqId;
        streamInfo_.sqBaseAddr = that.streamInfo_.sqBaseAddr;
        streamInfo_.sqDepth = that.streamInfo_.sqDepth;
        streamInfo_.sqId = that.streamInfo_.sqId;
        invalidFlag_ = that.invalidFlag_; // 共享销毁标志, 副本与owner持有同一份
    }
    return *this;
}

Stream Stream::operator=(Stream&& that)
{
    if (&that != this) {
        stream_ = that.stream_;
        device_id_ = that.device_id_;
        stream_owner_ = that.stream_owner_;
        taskLogicInfo_ = that.taskLogicInfo_;
        streamId_ = that.streamId_;
        isMainStream_ = that.isMainStream_;
        modeGotFlag_ = that.modeGotFlag_;
        streamMode_ = that.streamMode_;
        sqId_ = that.sqId_;
        ctx_ = that.ctx_;
        cqId_ = that.cqId_;
        logicCqid_ = that.logicCqid_;
        sqeContext_ = that.sqeContext_;
        cqeContext_ = that.cqeContext_;
        streamInfo_.actualStreamId = that.streamInfo_.actualStreamId;
        streamInfo_.logicCqId = that.streamInfo_.logicCqId;
        streamInfo_.sqBaseAddr = that.streamInfo_.sqBaseAddr;
        streamInfo_.sqDepth = that.streamInfo_.sqDepth;
        streamInfo_.sqId = that.streamInfo_.sqId;
        invalidFlag_ = std::move(that.invalidFlag_); // 转移销毁标志所有权
    }

    that.stream_ = nullptr;
    that.device_id_ = HCCL_DEVICE_NOT_SET;
    that.stream_owner_ = false;
    that.taskLogicInfo_ = taskLogicInfo_;
    that.streamId_ = 0;
    that.isMainStream_ = isMainStream_;
    that.modeGotFlag_ = modeGotFlag_;
    that.streamMode_ = streamMode_;
    that.sqId_ = sqId_;
    that.ctx_ = ctx_;
    that.cqId_ = cqId_;
    that.logicCqid_ = logicCqid_;
    that.sqeContext_ = nullptr;
    that.cqeContext_ = nullptr;
    that.streamInfo_.actualStreamId = streamInfo_.actualStreamId;
    that.streamInfo_.logicCqId = streamInfo_.logicCqId;
    that.streamInfo_.sqBaseAddr = nullptr;
    that.streamInfo_.sqDepth = streamInfo_.sqDepth;
    that.streamInfo_.sqId = streamInfo_.sqId;
    return *this;
}

void Stream::DestroyStream()
{
    // 销毁stream
    if (stream_owner_ && stream_ != nullptr) {
        // stream需要在原ctx上销毁
        aclrtContext ctxTmp = nullptr;
        HcclResult ret = hrtCtxGetCurrent(&ctxTmp);
        bool needChangeCtx = (ret == HCCL_SUCCESS && ctx_ != nullptr);
        if (needChangeCtx) {
            ret = hrtCtxSetCurrent(ctx_);
            HCCL_INFO("Switch Ctx ret[%d], curCtx[%p], setCtx[%p], stream id[%d]", ret, ctxTmp, ctx_, streamId_);
        }
        ret = hrtStreamDestroy(stream_);
        HCCL_RUN_INFO("[HCCL_TRACE]StreamDestroy, streamPtr[%p], stream id[%d]", stream_, streamId_);
        if (ret != HCCL_SUCCESS) {
            HCCL_WARNING("errNo[0x%016llx] hrtStreamDestroy error, ret[%d]", HCCL_ERROR_CODE(HCCL_E_RUNTIME), ret);
        }
        if (needChangeCtx) {
            ret = hrtCtxSetCurrent(ctxTmp);
            HCCL_INFO("Restore Ctx ret[%d], setCtx[%p], stream id[%d]", ret, ctxTmp, streamId_);
        }
    }
}

void Stream::SetEmpty()
{
    DestroyStream();
    stream_ = nullptr;
    device_id_ = HCCL_DEVICE_NOT_SET;
    stream_owner_ = false;
    streamId_ = 0;
    isMainStream_ = true;
    sqId_ = 0;
    ctx_ = nullptr;
    cqId_ = 0;
    logicCqid_ = 0;
}

HcclResult Stream::InitStream()
{
    if (stream_ != nullptr) {
        HcclResult ret = hrtGetStreamId(stream_, streamId_);
        if (ret != HCCL_SUCCESS) {
            SetEmpty();
            HCCL_ERROR(
                "[InitStream]Failed to get the streamId through the rtstream, errNo[0x%016llx] "
                "hrtGetStreamId error, ret[%d]",
                HCCL_ERROR_CODE(HCCL_E_RUNTIME), ret);
            return HCCL_E_INTERNAL;
        }

        ret = hrtStreamGetSqid(stream_, &(sqId_));
        if (ret != HCCL_SUCCESS) {
            SetEmpty();
            HCCL_ERROR(
                "[InitStream]Failed to get the sqId through the rtstream, errNo[0x%016llx] "
                "hrtStreamGetSqid error, ret[%d]",
                HCCL_ERROR_CODE(HCCL_E_RUNTIME), ret);
            return HCCL_E_INTERNAL;
        }
        (void)hrtCtxGetCurrent(&ctx_);

        ret = hrtStreamGetCqid(stream_, &(cqId_), &(logicCqid_));
        if (ret != HCCL_SUCCESS) {
            SetEmpty();
            HCCL_ERROR(
                "[InitStream]Failed to get the cqId through the rtstream, errNo[0x%016llx] "
                "hrtStreamGetCqid error, ret[%d]",
                HCCL_ERROR_CODE(HCCL_E_RUNTIME), ret);
            return HCCL_E_INTERNAL;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult Stream::SetMode(const uint64_t stmMode)
{
    HcclResult ret = hrtStreamSetMode(stream_, stmMode);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[Stream][SetMode]errNo[0x%016llx] hrtStreamSetMode error, ret[%d]", HCCL_ERROR_CODE(HCCL_E_RUNTIME), ret);
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

HcclResult Stream::GetMode(uint64_t* const stmMode)
{
    if (modeGotFlag_ == false) {
        HcclResult ret = hrtStreamGetMode(stream_, &streamMode_);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR(
                "[Stream][GetMode]errNo[0x%016llx] hrtStreamGetMode error, ret[%d]", HCCL_ERROR_CODE(HCCL_E_RUNTIME),
                ret);
            return HCCL_E_INTERNAL;
        }
    }
    *stmMode = streamMode_;
    return HCCL_SUCCESS;
}

void Stream::PushTaskLogicInfo(TaskLogicInfo& taskLogicInfo)
{
    taskLogicInfo_.push(taskLogicInfo);
    HCCL_INFO(
        "[PushTaskLogicInfo] stream[%p], taskLogicType[%d], taskLogicFuncType[%d], taskLogicInfo size[%d]", stream_,
        taskLogicInfo.taskLogicCmd.taskLogicType, taskLogicInfo.taskFuncType, taskLogicInfo_.size());
}

HcclResult Stream::PopTaskLogicInfo(TaskLogicInfo& taskLogicInfo)
{
    if (taskLogicInfo_.size() > 0) {
        taskLogicInfo = taskLogicInfo_.front();
        HCCL_INFO(
            "[PopTaskLogicInfo] stream[%p], taskLogicType[%d], taskLogicFuncType[%d], taskLogicInfo size[%d]", stream_,
            taskLogicInfo.taskLogicCmd.taskLogicType, taskLogicInfo.taskFuncType, taskLogicInfo_.size());
        taskLogicInfo_.pop();
        return HCCL_SUCCESS;
    }
    return HCCL_E_NOT_FOUND;
}

HcclResult
Stream::GetNextSqeBufferAddr(uint8_t*& sqeBufferAddr, uint8_t*& sqeTypeAddr, uint8_t*& sqeDfxInfoAddr, uint16_t& taskId)
{
    if (UNLIKELY(sqeContext_ == nullptr)) {
        HCCL_ERROR("[Stream][GetNextSqeBufferAddr] Sqe context is null");
        return HCCL_E_INTERNAL;
    }
    auto& buff = sqeContext_->buffer;
    if (UNLIKELY(buff.tailSqeIdx >= HCCL_SQE_MAX_CNT)) {
        HCCL_INFO("[Stream][GetNextSqeBufferAddr] Sqe index to 2048, need clear");
        if (buff.sqeCnt != 0) {
            HCCL_ERROR("[Stream][GetNextSqeBufferAddr] Sqe index to 2048, but sqeCnt is not 0");
            return HCCL_E_INTERNAL;
        }
        CHK_RET(ClearLocalBuff());
    }
    sqeBufferAddr = buff.localBuff + buff.tailSqeIdx * HCCL_SQE_SIZE;
    sqeTypeAddr = &buff.sqeType[buff.tailSqeIdx];
    sqeDfxInfoAddr = reinterpret_cast<uint8_t*>(&buff.dfxInfo[buff.tailSqeIdx]);

    buff.profTimestap[buff.tailSqeIdx] = ProfGetCurCpuTimestamp();
    taskId = buff.tailSqeTaskId;

    HCCL_DEBUG(
        "[Stream][GetNextSqeBufferAddr] streamId: %u Get next idx:%u, taskId:%u, flipNum:%u",
        streamInfo_.actualStreamId, buff.tailSqeIdx, taskId, buff.filpNum);
    if (UNLIKELY(buff.tailSqeTaskId == UINT16_MAX)) {
        buff.filpNum++;
        HCCL_WARNING("[Stream][GetNextSqeBufferAddr] Sqe context cur taskId is uint16_max");
    }
    buff.tailSqeTaskId++;
    buff.sqeCnt++;
    buff.tailSqeIdx++;
    return HCCL_SUCCESS;
}

HcclResult Stream::InitSqAndCqeContext(uint32_t sqHead, uint32_t sqTail, SqCqeContext* context)
{
    CHK_PTR_NULL(context);
    sqeContext_ = &context->sqContext;
    CHK_PTR_NULL(sqeContext_);
    cqeContext_ = &context->cqeContext;
    CHK_PTR_NULL(cqeContext_);

    auto& buff = sqeContext_->buffer;
    buff.sqHead = sqHead;
    buff.sqTail = sqTail;
    cqeContext_->cqeStatus = 0;
    HCCL_INFO(
        "%s success, streamId:%u, sqHead:%u, sqTail:%u, context:%p", __func__, streamId_, sqHead, sqTail, context);
    return HCCL_SUCCESS;
}

HcclResult Stream::ClearLocalBuff()
{
    CHK_PTR_NULL(sqeContext_);
    auto& buff = sqeContext_->buffer;
    if (memset_s(buff.localBuff, sizeof(buff.localBuff), 0, buff.tailSqeIdx * HCCL_SQE_SIZE) != EOK) {
        HCCL_ERROR("[Stream][ClearLocalBuff] clear local buff failed");
        return HCCL_E_MEMORY;
    }
    if (memset_s(buff.sqeType, sizeof(buff.sqeType), 0, buff.tailSqeIdx) != EOK) {
        HCCL_ERROR("[Stream][ClearLocalBuff] clear sqe type failed");
        return HCCL_E_MEMORY;
    }
    if (memset_s(buff.addInfo, sizeof(buff.addInfo), 0, buff.tailSqeIdx) != EOK) {
        HCCL_ERROR("[Stream][ClearLocalBuff] clear add info failed");
        return HCCL_E_MEMORY;
    }
    buff.sqeCnt = 0;
    buff.tailSqeIdx = 0;

    if (cqeContext_ != nullptr && memset_s(cqeContext_, sizeof(ErrCqeContext), 0, sizeof(ErrCqeContext)) != EOK) {
        HCCL_ERROR("[Stream][ClearLocalBuff] clear cqe context failed");
        return HCCL_E_MEMORY;
    }
    return HCCL_SUCCESS;
}

HcclResult Stream::SetCqeContext(const ErrCqeContext& cqeContext)
{
    CHK_PTR_NULL(cqeContext_);
    *cqeContext_ = cqeContext;
    return HCCL_SUCCESS;
}

HcclResult Stream::GetCqeContext(ErrCqeContext& cqeContext)
{
    CHK_PTR_NULL(cqeContext_);
    cqeContext = *cqeContext_;
    return HCCL_SUCCESS;
}

HcclResult Stream::GetStreamInfo(const HcclComStreamInfo*& streamInfo)
{
    streamInfo = &streamInfo_;
    return HCCL_SUCCESS;
}

HcclResult Stream::SetInvalidFlag()
{
    // owner销毁stream前先标记invalid, 让持有副本的dispatcher能感知并跳过, 避免访问悬空的sqeContext_
    // 必须在hrtStreamDestroy之前置位, 确保并发遍历streamMap_的线程在此期间读到true
    if (invalidFlag_ != nullptr) {
        invalidFlag_->store(true, std::memory_order_relaxed);
    }
    return HCCL_SUCCESS;
}
} // namespace hccl
