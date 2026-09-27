/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "stream_manager.h"
#include "log.h"
#include "exception_util.h"
#include "communicator_impl.h"
#include "stream_utils.h"

namespace Hccl {

StreamManager::StreamManager(CommunicatorImpl* comm) : comm(comm)
{
    opbase = std::make_unique<OpbaseStreamManager>(comm);
    offload = std::make_unique<OffloadStreamManager>();
}

Stream* StreamManager::GetSlave() const
{
    HCCL_INFO("[StreamManager::%s] start.", __func__);

    Stream* stream = nullptr;
    auto op = comm->GetCurrentCollOperator();
    OpMode opMode = op->opMode;
    if (opMode == OpMode::OPBASE) {
        stream = comm->GetStreamManager().opbase->GetOrCreateSlave();
    } else if (opMode == OpMode::OFFLOAD) {
        stream = comm->GetStreamManager().offload->GetSlave(op->opTag);
    } else {
        THROW<NotSupportException>(StringFormat("Unsupported OpMode: %s", opMode.Describe().c_str()));
    }

    HCCL_INFO(
        "[StreamManager::%s] end, opMode[%s], slave stream[%u].", __func__, opMode.Describe().c_str(), stream->GetId());
    return stream;
}

Stream* StreamManager::GetSlaveByIndex(u32 index) const
{
    HCCL_INFO("[StreamManager::%s] start.", __func__);

    Stream* stream = nullptr;
    auto op = comm->GetCurrentCollOperator();
    OpMode opMode = op->opMode;
    if (opMode == OpMode::OPBASE) {
        stream = comm->GetStreamManager().opbase->GetSlave(index);
    } else if (opMode == OpMode::OFFLOAD) {
        stream = comm->GetStreamManager().offload->GetSlave(op->opTag, index);
    } else {
        THROW<NotSupportException>(StringFormat("Unsupported OpMode: %s", opMode.Describe().c_str()));
    }

    HCCL_INFO(
        "[StreamManager::%s] end, opMode[%s], slave stream[%u].", __func__, opMode.Describe().c_str(), stream->GetId());
    return stream;
}

Stream* StreamManager::GetMaster() const
{
    HCCL_INFO("[StreamManager::%s] start.", __func__);

    Stream* stream = nullptr;
    auto op = comm->GetCurrentCollOperator();
    OpMode opMode = op->opMode;
    if (opMode == OpMode::OPBASE) {
        stream = comm->GetStreamManager().opbase->GetMaster();
    } else if (opMode == OpMode::OFFLOAD) {
        stream = comm->GetStreamManager().offload->GetMaster(op->opTag);
    } else {
        THROW<NotSupportException>(StringFormat("Unsupported OpMode: %s", opMode.Describe().c_str()));
    }

    HCCL_INFO(
        "[StreamManager::%s] end, opMode[%s], master stream[%u].", __func__, opMode.Describe().c_str(),
        stream->GetId());
    return stream;
}

void StreamManager::CaptureSlaveStream(const Stream* masterStream, const Stream* slaveStream) const
{
    HCCL_RUN_INFO(
        "[StreamManager][%s] masterStream[%u] slaveStream[%u]", __func__, masterStream->GetId(), slaveStream->GetId());
    rtModel_t rtModel = nullptr;
    bool isCapture = false;
    u32 modelId = 0;
    auto op = comm->GetCurrentCollOperator();
    OpMode opMode = op->opMode;
    if (opMode == OpMode::OPBASE) {
        auto ret = GetStreamCaptureInfo(masterStream->GetPtr(), rtModel, isCapture);
        if (ret != HCCL_SUCCESS) {
            THROW<InternalException>(StringFormat(
                "[StreamManager::%s] Failed to obtain masterStream capture status, "
                "ret[%d]",
                __func__, ret));
        }

        if (isCapture) {
            if (rtModel == nullptr) {
                THROW<NullPtrException>(StringFormat("[StreamManager::%s] rtModel is NULL.", __func__));
            }

            ret = GetModelId(rtModel, modelId);
            if (ret != HCCL_SUCCESS) {
                THROW<InternalException>(StringFormat(
                    "[StreamManager::%s] Failed to obtain the modelId corresponding "
                    "to the masterStream rtModel, ret[%d]",
                    __func__, ret));
            }

            ret = AddStreamToModel(slaveStream->GetPtr(), rtModel);
            if (ret != HCCL_SUCCESS) {
                THROW<InternalException>(StringFormat(
                    "[StreamManager::%s] Adding the salveStream to the masterStream "
                    "failed, ret[%d]",
                    __func__, ret));
            }
            HCCL_RUN_INFO(
                "[StreamManager::%s] Add slaveStream[%u] to model[%u] success, masterStream[%u]", __func__,
                slaveStream->GetId(), modelId, masterStream->GetId());
        }
    }
}

u32 StreamManager::GetSlaveIndex() const
{
    HCCL_INFO("[StreamManager::%s] start.", __func__);

    u32 res = 0;
    auto op = comm->GetCurrentCollOperator();
    OpMode opMode = op->opMode;
    if (opMode == OpMode::OPBASE) {
        res = comm->GetStreamManager().opbase->GetSlaveIndex();
    } else if (opMode == OpMode::OFFLOAD) {
        res = comm->GetStreamManager().offload->GetSlaveIndex(op->opTag);
    } else {
        THROW<NotSupportException>(StringFormat("Unsupported OpMode: %s", opMode.Describe().c_str()));
    }

    HCCL_INFO("[StreamManager::%s] end, opMode[%s].", __func__, opMode.Describe().c_str());
    return res;
}

void StreamManager::ResetSlaveIndex(u32 index) const
{
    HCCL_INFO("[StreamManager::%s] start.", __func__);

    auto op = comm->GetCurrentCollOperator();
    OpMode opMode = op->opMode;
    if (opMode == OpMode::OPBASE) {
        comm->GetStreamManager().opbase->ResetIndex(index);
    } else if (opMode == OpMode::OFFLOAD) {
        comm->GetStreamManager().offload->ResetIndex(op->opTag, index);
    } else {
        THROW<NotSupportException>(StringFormat("Unsupported OpMode: %s", opMode.Describe().c_str()));
    }

    HCCL_INFO("[StreamManager::%s] end, opMode[%s].", __func__, opMode.Describe().c_str());
}

void StreamManager::RecordStreamIdToIndex(u32 streamId, u32 streamIndex)
{
    streamIdToIndexMap_[streamId] = streamIndex;
}

u32 StreamManager::GetStreamIndex(u32 streamId) { return streamIdToIndexMap_[streamId]; }

void StreamManager::InitBucket(u32 bucket) { streamBucket_[bucket] = std::vector<u32>{}; }

void StreamManager::RegisterBucket(u32 bucket, u32 subStreamIndex)
{
    streamBucket_[bucket].emplace_back(subStreamIndex);
}

std::vector<u32>& StreamManager::GetSubSlaveIndexes(u32 slaveIndex) { return streamBucket_[slaveIndex]; }

void StreamManager::DestroyRecords()
{
    streamIdToIndexMap_.clear();
    streamBucket_.clear();
}

} // namespace Hccl
