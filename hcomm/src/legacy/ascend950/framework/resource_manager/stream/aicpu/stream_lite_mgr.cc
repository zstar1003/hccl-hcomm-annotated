/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "stream_lite_mgr.h"
#include "binary_stream.h"
#include "task_exception_func.h"

namespace Hccl {
StreamLite* StreamLiteMgr::GetMaster()
{
    if (!streams.empty()) {
        return streams[0].get();
    }
    return nullptr;
}

StreamLite* StreamLiteMgr::GetSlave(u32 index)
{
    if (streams.size() > index + 1) {
        return streams[index + 1].get();
    }
    return nullptr;
}

u32 StreamLiteMgr::SizeOfSlaves() { return streams.size() > 1 ? streams.size() - 1 : 0; }

void StreamLiteMgr::Reset()
{
    for (auto& streamLite : streams) {
        TaskExceptionFunc::GetInstance().UnRegister(streamLite.get());
    }
    streams.clear();
}

void StreamLiteMgr::ParseLiteData(std::vector<char>& data, u32 num, u32 sizePerDto)
{
    u32 size = streams.size();
    for (u32 idx = 0; idx < num; idx++) {
        auto start = data.begin() + idx * sizePerDto;
        auto end = start + sizePerDto;
        std::vector<char> uniqueId(start, end);
        if (idx >= size) {
            HCCL_INFO("Make new Stream Lite idx=%u, size=%u", idx, size);
            auto stream = std::make_unique<StreamLite>(uniqueId);
            TaskExceptionFunc::GetInstance().Register(stream.get());
            streams.push_back(std::move(stream));
        }
    }
}

void StreamLiteMgr::ParsePackedData(std::vector<char>& givenData)
{
    u32 num;
    BinaryStream binaryStream(givenData);
    std::vector<char> data;
    binaryStream >> num;
    binaryStream >> data;
    HCCL_INFO("StreamLiteMgr, num=%u, data=%s", num, Bytes2hex(data.data(), data.size()).c_str());
    u32 sizePerDto = data.size() / num;
    if (streams.size() >= num) { // 已经解析出stream，并且已有stream数量大于需要解包的数量，则直接返回
        return;
    }

    ParseLiteData(data, num, sizePerDto);
}

StreamLiteMgr::~StreamLiteMgr()
{
    for (auto& streamLite : streams) {
        TaskExceptionFunc::GetInstance().UnRegister(streamLite.get());
    }
    streams.clear();
}

std::vector<StreamLite*> StreamLiteMgr::GetAllStreams()
{
    std::vector<StreamLite*> result;
    result.reserve(streams.size());
    std::transform(streams.begin(), streams.end(), std::back_inserter(result), [](const auto& ptr) {
        return ptr.get();
    });
    return result;
}

} // namespace Hccl
