/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "plugin_runner.h"
#include "adapter_rts_common.h"
#include "externalinput_pub.h"

using namespace hccl;
PluginRunner::PluginRunner(ProfilerBase* profiler) : profiler_(profiler) {}

PluginRunner::~PluginRunner() {}

template <typename T>
void PluginRunner::operator()(rtStream_t stream, TaskType taskType, const T& para) const
{
    // capture模式下hrtGetStreamId获取的是原来的流对应ID，与实际执行流不是同一个
    // capture模式下hrtGetTaskIdAndStreamID获取实际执行的streamID和taskID
    u32 threadLastTaskID = 0;
    u32 threadLastStreamID = 0;
    s32 streamID = 0;
    bool isOneSideTask = false;
    bool isCapture = false;
    CHK_PRT(isStreamCapture(stream, isCapture));

    if (profiler_ == nullptr) {
        return;
    }
    CHK_PRT(hrtGetTaskIdAndStreamID(threadLastTaskID, threadLastStreamID));

    std::string tag;
    CHK_PRT(ProfilerBase::GetTagByStream(threadLastStreamID, tag));
    if (tag.find("BatchPut_") != std::string::npos || tag.find("BatchGet_") != std::string::npos) {
        isOneSideTask = true;
    }

    HcclResult ret;
    if (GetExternalInputHcclEnableFfts() && GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE
        && !isOneSideTask) {
        ret = hrtGetStreamId(stream, streamID);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[PluginRunner][Operator]rtGet stream id fail. return[%d]", ret), );

        u32 castStreamID = static_cast<u32>(streamID);
        if (isCapture) {
            ret = hrtGetTaskIdAndStreamID(threadLastTaskID, threadLastStreamID);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR("[PluginRunner][Operator]rtGet task id and stream id fail. return[%d]", ret), );
            profiler_->Save(castStreamID, threadLastStreamID, threadLastTaskID, taskType, para);
        } else {
            profiler_->Save(castStreamID, threadLastTaskID, taskType, para);
        }
    } else {
        ret = hrtGetTaskIdAndStreamID(threadLastTaskID, threadLastStreamID);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR("[PluginRunner][Operator]rtGet task id and stream id fail. return[%d]", ret), );

        if (isCapture) {
            ret = hrtGetStreamId(stream, streamID);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR("[PluginRunner][Operator]rtGet task id and stream id fail. return[%d]", ret), );
            u32 castStreamID = static_cast<u32>(streamID);
            profiler_->Save(castStreamID, threadLastStreamID, threadLastTaskID, taskType, para);
        } else {
            profiler_->Save(threadLastStreamID, threadLastTaskID, taskType, para);
        }
    }
}

template void PluginRunner::operator()<TaskParaDMA>(rtStream_t, TaskType, const TaskParaDMA&) const;
template void PluginRunner::operator()<TaskParaReduce>(rtStream_t, TaskType, const TaskParaReduce&) const;
template void PluginRunner::operator()<TaskParaNotify>(rtStream_t, TaskType, const TaskParaNotify&) const;

void PluginRunner::operator()(rtStream_t stream, const void* descBuf, size_t descBufLen) const
{
    u32 threadLastTaskID = 0;
    u32 threadLastStreamID = 0;
    s32 streamID = 0;
    HcclResult ret;
    bool isCapture = false;
    CHK_PRT(isStreamCapture(stream, isCapture));

    CHK_PRT_RET(profiler_ == nullptr, HCCL_WARNING("profiler_ is nullptr"), );
    ret = hrtGetTaskIdAndStreamID(threadLastTaskID, threadLastStreamID);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS, HCCL_ERROR("[PluginRunner][Operator]rtGet task id and stream id fail. return[%d]", ret), );

    if (isCapture) {
        ret = hrtGetStreamId(stream, streamID);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[PluginRunner][Operator]rtGet stream id fail. return[%d]", ret), );
        u32 castStreamID = static_cast<u32>(streamID);
        profiler_->Save(castStreamID, threadLastStreamID, threadLastTaskID, descBuf, descBufLen);
    } else {
        profiler_->Save(threadLastStreamID, threadLastTaskID, descBuf, descBufLen);
    }
}

void PluginRunner::operator()(const TaskParaHost& paraHost) const
{
    if (profiler_ != nullptr) {
        profiler_->SaveToLog(paraHost);
    }
}

void PluginRunner::operator()(rtStream_t stream, const TaskParaAiv& paraAiv) const
{
    u32 threadLastTaskID = 0;
    u32 threadLastStreamID = 0;
    s32 streamID = 0;
    HcclResult result;
    bool isCapture = false;
    CHK_PRT(isStreamCapture(stream, isCapture));
    result = hrtGetTaskIdAndStreamID(threadLastTaskID, threadLastStreamID);
    CHK_PRT_RET(
        result != HCCL_SUCCESS,
        HCCL_ERROR("[PluginRunner][Operator]rtGet task id and stream id fail. return[%d]", result), );
    CHK_PRT_RET(profiler_ == nullptr, HCCL_WARNING("profiler_ is nullptr"), );
    if (isCapture) {
        result = hrtGetStreamId(stream, streamID);
        CHK_PRT_RET(
            result != HCCL_SUCCESS, HCCL_ERROR("[PluginRunner][Operator]rtGet stream id fail. return[%d]", result), );
        u32 castStreamID = static_cast<u32>(streamID);
        profiler_->Save(castStreamID, threadLastStreamID, threadLastTaskID, paraAiv);
    } else {
        profiler_->Save(threadLastStreamID, threadLastTaskID, paraAiv);
    }
}
