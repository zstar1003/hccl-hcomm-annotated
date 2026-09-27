/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aicpu_stream_manager.h"
#include <algorithm>
#include "log.h"
#include "exception_util.h"
#include "binary_stream.h"
#include "internal_exception.h"
#include "stream_utils.h"

namespace Hccl {

AicpuStreamManager::~AicpuStreamManager() { DECTOR_TRY_CATCH("AicpuStreamManager", Clear()); }

void AicpuStreamManager::AllocStreams(u32 num)
{
    int size = streams.size();
    if (static_cast<int>(num) <= size) {
        return;
    }
    streams.resize(num);
    for (int i = size; i < static_cast<int>(num); i++) {
        streams[i] = std::make_unique<Stream>(true, false);
        stream_pointers.push_back(streams[i].get());
    }
}

void AicpuStreamManager::Clear() { streams.clear(); }

std::vector<char> AicpuStreamManager::GetPackedData()
{
    u32 num = streams.size();
    if (num == 0) {
        THROW<InternalException>("opbase stream num is 0.");
    }

    std::vector<char> data;
    for (auto& it : streams) {
        auto uniqueId = it->GetUniqueId();
        HCCL_INFO("AicpuStreamManager::GetPackedData:%s", it->Describe().c_str());
        data.insert(data.end(), uniqueId.begin(), uniqueId.end());
    }

    BinaryStream binaryStream;
    binaryStream << num;
    binaryStream << data;

    std::vector<char> result;
    binaryStream.Dump(result);
    HCCL_INFO("AicpuStreamManager::GetPackedData:%s", Bytes2hex(result.data(), result.size()).c_str());
    return result;
}

void AicpuStreamManager::AllocFreeStream()
{
    if (freeStream == nullptr) {
        freeStream = std::make_unique<Stream>(false, false);
        HCCL_RUN_INFO("AicpuStreamManager %s allocated: %s", __func__, freeStream->Describe().c_str());
    }
}

HcclResult AicpuStreamManager::CaptureFreeStream(const Stream* mainStream, const Stream* slaveStream) const
{
    HCCL_DEBUG(
        "[AicpuStreamManager][%s] mainStream[%u] slaveStream[%u]", __func__, mainStream->GetId(), slaveStream->GetId());
    rtModel_t rtModel = nullptr;
    bool isCapture = false;
    u32 modelId = 0;
    CHK_RET(GetStreamCaptureInfo(mainStream->GetPtr(), rtModel, isCapture));
    if (isCapture) {
        CHK_PTR_NULL(rtModel);
        CHK_RET(GetModelId(rtModel, modelId));
        CHK_RET(AddStreamToModel(slaveStream->GetPtr(), rtModel));
        HCCL_INFO(
            "[AicpuStreamManager][%s] Add freeStream[%u] to model[%u] success, mainStream[%u]", __func__,
            slaveStream->GetId(), modelId, mainStream->GetId());
    }
    return HCCL_SUCCESS;
}

void AicpuStreamManager::AclGraphCaptureFreeStream(const Stream* mainStream) const
{
    auto ret = CaptureFreeStream(mainStream, freeStream.get());
    if (ret != HCCL_SUCCESS) {
        THROW<InternalException>(StringFormat(
            "[AicpuStreamManager::%s] capture freeStream fail, "
            "error code:%d",
            __func__, ret));
    }
}

} // namespace Hccl
