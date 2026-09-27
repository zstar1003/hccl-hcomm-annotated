/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "stream.h"

namespace hccl {
Stream::Stream() {}

Stream::~Stream() {}

Stream::Stream(const Stream& that) : stream_(that.ptr()), isMainStream_(that.isMainStream_), streamId_(that.streamId_)
{}

Stream::Stream(Stream&& that) : stream_(that.ptr()), isMainStream_(that.isMainStream_), streamId_(that.streamId_)
{
    that.stream_ = nullptr;
}

Stream::Stream(const rtStream_t rtStream, bool isMainStream)
    : stream_(const_cast<void*>(rtStream)),
      isMainStream_(isMainStream)
{}

Stream::Stream(const StreamType streamType, bool isMainStream) {}

Stream Stream::operator=(Stream&& that)
{
    if (&that != this) {
        stream_ = that.stream_;
        isMainStream_ = that.isMainStream_;
        streamId_ = that.streamId_;
    }
    that.stream_ = nullptr;
    return *this;
}

Stream& Stream::operator=(const Stream& that)
{
    if (&that != this) {
        stream_ = that.ptr();
        isMainStream_ = that.isMainStream_;
        streamId_ = that.streamId_;
    }
    return *this;
}

HcclResult Stream::PopTaskLogicInfo(TaskLogicInfo& taskLogicInfo) { return HCCL_SUCCESS; }

StreamAddrRecorder* StreamAddrRecorder::Global()
{
    static StreamAddrRecorder* streamAddrRecorder = new StreamAddrRecorder;
    return streamAddrRecorder;
}

} // namespace hccl
