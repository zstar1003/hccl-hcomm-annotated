/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "offload_stream_manager.h"
#include "log.h"
#include "exception_util.h"
#include "invalid_params_exception.h"
#include "stream_utils.h"

namespace Hccl {

void OffloadStreamManager::RegisterMaster(const std::string& opTag, std::unique_ptr<Stream> stream)
{
    HCCL_INFO("[OffloadStreamManager::%s] start.", __func__);

    if (masters.find(opTag) != masters.end()) {
        std::string msg = StringFormat("master stream of op[%s] has been registered.", opTag.c_str());
        THROW<InvalidParamsException>(msg);
    }
    // 判断是否为acl graph零拷贝切图模式，判断标志为主流是否被捕获
    bool isCapture = false;
    rtModel_t rtModel = nullptr;
    auto ret = GetStreamCaptureInfo(stream->GetPtr(), rtModel, isCapture);
    if (ret != HCCL_SUCCESS) {
        THROW<InvalidParamsException>(
            StringFormat("[OffloadStreamManager::%s] GetStreamCaptureInfo failed.", __func__));
    }
    if (!isCapture) {
        ActivateSlaveStreams(opTag, stream.get()); // 不是acl graph则维持原流程
    }
    masters[opTag] = std::move(stream);

    currOpTag = opTag;

    HCCL_INFO("[OffloadStreamManager::%s] end.", __func__);
}

void OffloadStreamManager::ActivateSlaveStreams(const std::string& opTag, const Stream* masterStream)
{
    HCCL_INFO("[OffloadStreamManager::%s] start.", __func__);

    const auto& slaveStreams = slaves[opTag];
    int slaveNum = slaveStreams.size();
    u32 mainStreamId = masterStream->GetId();
    auto& activeSlaveStreams = streamActiveManager_[mainStreamId];
    for (const auto& slave : slaveStreams) {
        u32 slaveId = slave->GetId();
        if (activeSlaveStreams.insert(slaveId).second) {
            HrtStreamActive(slave->GetPtr(), masterStream->GetPtr());
        }
    }
    HCCL_INFO("[OffloadStreamManager::%s] end, slaveNum[%d].", __func__, slaveNum);
}

void OffloadStreamManager::RegisterSlaves(const std::string& opTag, const std::vector<void*>& slaveStreams)
{
    HCCL_INFO("[OffloadStreamManager::%s] start.", __func__);

    if (slaves.find(opTag) != slaves.end()) {
        std::string msg = StringFormat("slave streams of op[%s] has been registered.", opTag.c_str());
        THROW<InvalidParamsException>(msg);
    }

    int slaveNum = slaveStreams.size();
    slaves[opTag].resize(slaveNum);
    for (int i = 0; i < slaveNum; i++) {
        slaves[opTag][i] = std::make_unique<Stream>(slaveStreams[i], false);
    }

    HCCL_INFO("[OffloadStreamManager::%s] end, slaveNum[%d].", __func__, slaveNum);
}

Stream* OffloadStreamManager::GetSlave(const std::string& opTag)
{
    HCCL_INFO("[OffloadStreamManager::%s] start, opTag[%s].", __func__, opTag.c_str());

    CheckOpTag(opTag);

    auto slavesIter = slaves.find(opTag);
    u32 slavesSize = slavesIter == slaves.end() ? 0 : slavesIter->second.size();
    HCCL_INFO("[OffloadStreamManager::%s] slavesSize[%u] slaveIndex[%u]", __func__, slavesSize, slaveIndex);
    if (slaveIndex >= slavesSize) {
        THROW<InvalidParamsException>(StringFormat("[OffloadStreamManager::%s] slave streams not enough.", __func__));
    }

    HCCL_INFO("[OffloadStreamManager::%s] end", __func__);
    return slaves[opTag][slaveIndex++].get();
}

Stream* OffloadStreamManager::GetMaster(const std::string& opTag)
{
    HCCL_INFO("[OffloadStreamManager::%s] start, opTag[%s].", __func__, opTag.c_str());

    CheckOpTag(opTag);

    if (masters.find(opTag) == masters.end()) {
        HCCL_WARNING("[OffloadStreamManager::%s] master stream of opTag[%s] not found.", __func__, opTag.c_str());
        return nullptr;
    }

    HCCL_INFO("[OffloadStreamManager::%s] end", __func__);
    return masters[opTag].get();
}

u32 OffloadStreamManager::GetSlaveIndex(const std::string& opTag) const
{
    CheckOpTag(opTag);
    return slaveIndex;
}

void OffloadStreamManager::ResetIndex(const std::string& opTag, u32 index)
{
    CheckOpTag(opTag);
    slaveIndex = index;
}

void OffloadStreamManager::CheckOpTag(const std::string& opTag) const
{
    if (opTag != currOpTag) {
        THROW<InvalidParamsException>(StringFormat(
            "[OffloadStreamManager::%s] opTag[%s] is not currOpTag[%s].", __func__, opTag.c_str(), currOpTag.c_str()));
    }
}

Stream* OffloadStreamManager::GetSlave(const std::string& opTag, u32 index) const
{
    CheckOpTag(opTag);
    if (index >= slaves.at(opTag).size()) {
        THROW<InvalidParamsException>(
            StringFormat("[OffloadStreamManager::%s] index[%u] is invalid.", __func__, index));
    }
    return slaves.at(opTag)[index].get();
}

HcclResult OffloadStreamManager::ClearOpStream(const std::string& opTag)
{
    if (masters.find(opTag) == masters.end()) {
        HCCL_WARNING("[OffloadStreamManager::%s] optag[%s] master stream not found.", __func__, opTag.c_str());
        return HCCL_SUCCESS;
    }
    if (slaves.find(opTag) == slaves.end()) {
        HCCL_WARNING("[OffloadStreamManager::%s] optag[%s] slave streams not found.", __func__, opTag.c_str());
        return HCCL_SUCCESS;
    }
    const auto& slaveStreams = slaves[opTag];
    u32 mainStreamId = masters[opTag]->GetId();
    auto& activeSlaveStreams = streamActiveManager_[mainStreamId];
    for (const auto& slave : slaveStreams) {
        u32 slaveId = slave->GetId();
        activeSlaveStreams.erase(slaveId);
    }
    masters.erase(opTag);
    slaves.erase(opTag);
    return HCCL_SUCCESS;
}

} // namespace Hccl
