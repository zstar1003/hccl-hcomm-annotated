/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "stream_lite.h"
#include "binary_stream.h"
#include "string_util.h"
#include "rtsq_a5.h"
namespace Hccl {

// 这里不进行构造时的特殊处理，因为RtsqA5的构造不会抛出异常，唯一的可能是申请内存失败，此时抛出标准异常
StreamLite::StreamLite(std::vector<char>& uniqueId)
{
    BinaryStream binaryStream(uniqueId);
    binaryStream >> id;
    binaryStream >> sqId;
    binaryStream >> devPhyId;
    binaryStream >> cqId;
    HCCL_INFO(
        "StreamLite::StreamLite:Start: %s, data=%s", Describe().c_str(),
        Bytes2hex(uniqueId.data(), uniqueId.size()).c_str());
    rtsq = std::make_unique<RtsqA5>(devPhyId, id, sqId);
    HCCL_INFO("StreamLite::StreamLite:End: %s", Describe().c_str());
}

StreamLite::StreamLite(u32 id, u32 sqIds, u32 phyId, u32 cqIds) : id(id), sqId(sqIds), devPhyId(phyId), cqId(cqIds)
{
    rtsq = std::make_unique<RtsqA5>(phyId, id, sqIds);
}

StreamLite::StreamLite(u32 id, u32 sqIds, u32 phyId, u32 cqIds, bool launchFlag)
    : id(id),
      sqId(sqIds),
      devPhyId(phyId),
      cqId(cqIds)
{
    rtsq = std::make_unique<RtsqA5>(phyId, id, sqIds, launchFlag);
}

RtsqBase* StreamLite::GetRtsq() const { return rtsq.get(); }

std::string StreamLite::Describe() const
{
    return StringFormat("StreamLite[id=%u, sqid=%u, devPhyId=%u]", id, sqId, devPhyId);
}

TaskInfoCircularQueue* StreamLite::GetTaskInfos() const { return &taskInfos_; }

DfxTaskInfo* StreamLite::NextTaskSlot() const
{
    if (taskInfos_.IsFull()) {
        ReportStreamTask();
    }
    return static_cast<DfxTaskInfo*>(taskInfos_.NextSlot());
}

void StreamLite::SetReportStreamTaskCallback(std::function<void(TaskInfoCircularQueue*)> callback)
{
    reportStreamTaskCallback_ = std::move(callback);
}

bool StreamLite::HasReportStreamTaskCallback() const { return reportStreamTaskCallback_ != nullptr; }

void StreamLite::ReportStreamTask() const
{
    if (reportStreamTaskCallback_ != nullptr) {
        reportStreamTaskCallback_(&taskInfos_);
    }
}

void StreamLite::SetGetLatestDfxOpInfoCallback(std::function<const void*()> callback)
{
    getLatestDfxOpInfoCallback_ = std::move(callback);
}

const void* StreamLite::GetLatestDfxOpInfo() const
{
    if (getLatestDfxOpInfoCallback_ != nullptr) {
        return getLatestDfxOpInfoCallback_();
    }
    return nullptr;
}

} // namespace Hccl
