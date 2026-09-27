/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "op_base_stream_manager.h"

namespace hccl {
OpBaseStreamManager::OpBaseStreamManager() : master_()
{
    slaves_.reserve(MAX_SUBSTREAM_NUM);
    slaveDevices_.reserve(MAX_SUBSTREAM_NUM);
    HCCL_DEBUG("[OpBaseStreamManager]reserve slaves[%u]", MAX_SUBSTREAM_NUM);
}

OpBaseStreamManager::~OpBaseStreamManager() = default;

HcclResult OpBaseStreamManager::RegisterMaster(Stream stream)
{
    std::unique_lock<std::mutex> lock(masterMutex_);
    master_ = stream;
    HCCL_DEBUG("[OpBaseStreamManager][RegisterMaster]register master stream[%p] success.", master_.ptr());
    lock.unlock();
    return HCCL_SUCCESS;
}

HcclResult OpBaseStreamManager::RegisterMaster(rtStream_t rtStream)
{
    std::unique_lock<std::mutex> lock(masterMutex_);
    if (rtStream == master_.ptr()) {
        return HCCL_SUCCESS;
    }
    master_ = Stream(rtStream);
    if (!master_.ptr()) {
        HCCL_ERROR("[OpBaseStreamManager][RegisterMaster]register master stream by rtStream[%p] failed.", rtStream);
        return HCCL_E_INTERNAL;
    }
    HCCL_DEBUG("[OpBaseStreamManager][RegisterMaster]register master stream by rtStream[%p] success.", rtStream);
    lock.unlock();
    return HCCL_SUCCESS;
}

HcclResult OpBaseStreamManager::AllocMaster(const StreamType streamType)
{
    std::unique_lock<std::mutex> lock(masterMutex_);
    Stream stream(streamType);
    if (!stream.ptr()) {
        HCCL_ERROR("[OpBaseStreamManager][AllocMaster]alloc master stream of type[%d] failed.", streamType);
        return HCCL_E_INTERNAL;
    }
    master_ = stream;
    HCCL_INFO("[OpBaseStreamManager][AllocMaster]alloc master stream[%p] success.", master_.ptr());
    lock.unlock();
    return HCCL_SUCCESS;
}
std::vector<Stream> OpBaseStreamManager::AllocSlaves(const StreamType streamType, u32 num)
{
    HCCL_INFO("[OpBaseStreamManager][AllocSlaves]requesting for [%u] slave streams.", num);
    std::unique_lock<std::mutex> masterLock(masterMutex_);
    if (!master_ || !master_.ptr()) {
        HCCL_ERROR("[OpBaseStreamManager][AllocSlaves]master not found, alloc slave stream failed.");
        return std::vector<Stream>();
    }
    std::vector<Stream>* slavesPtr;
    slavesPtr = (streamType == StreamType::STREAM_TYPE_ONLINE) ? &slaves_ : &slaveDevices_;
    if (slavesPtr->capacity() < num) {
        HCCL_ERROR(
            "[OpBaseStreamManager][AllocSlaves]request number[%u] exceed max substream num[%u], alloc failed.", num,
            slavesPtr->capacity());
        return std::vector<Stream>();
    }
    std::unique_lock<std::mutex> slaveLock(slavesMutex_);
    if (slavesPtr->size() < num) {
        HCCL_INFO(
            "[OpBaseStreamManager][AllocSlaves]expanding slave streams, original size[%u], target size[%u].",
            slavesPtr->size(), num);
        for (u32 i = slavesPtr->size(); i < num; i++) {
            slavesPtr->emplace_back(Stream(streamType));
            if (!(*slavesPtr)[i].ptr()) {
                // 创建足够数量的slave stream失败，直接返回空vector
                HCCL_ERROR("[OpBaseStreamManager][AllocSlaves]alloc slave stream[%u] failed.", i);
                return std::vector<Stream>();
            }
            if (streamType != StreamType::STREAM_TYPE_DEVICE) {
                HcclResult ret = SetSlaveMode((*slavesPtr)[i]);
                if (ret != HCCL_SUCCESS) {
                    HCCL_ERROR("[OpBaseStreamManager][AllocSlaves]set mode to slave stream[%u] failed.", i);
                    return std::vector<Stream>();
                }
            }
        }
    }
    HCCL_INFO("[OpBaseStreamManager][AllocSlaves]find enough slave streams, return size[%u].", num);
    return std::vector<Stream>(slavesPtr->begin(), slavesPtr->begin() + num);
}

HcclResult OpBaseStreamManager::SetSlaveMode(Stream& slave)
{
    if (master_) {
        uint64_t streamMode = 0;
        HcclResult ret = HCCL_SUCCESS;
        ret = master_.GetMode(&streamMode);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR(
                "[OpBaseStreamManager][SetSlaveMode]errNo[0x%016llx], get master stream mode failed.",
                HCCL_ERROR_CODE(ret));
            return ret;
        }
        ret = slave.SetMode(streamMode);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR(
                "[OpBaseStreamManager][SetSlaveMode]errNo[0x%016llx], set slave stream mode failed.",
                HCCL_ERROR_CODE(ret));
            return ret;
        }
        return HCCL_SUCCESS;
    }
    HCCL_ERROR("[OpBaseStreamManager][SetSlaveMode]master not registered, slave mode will not be set.");
    return HCCL_E_INTERNAL;
}

Stream OpBaseStreamManager::GetMaster()
{
    std::unique_lock<std::mutex> lock(masterMutex_);
    HCCL_DEBUG("[OpBaseStreamManager][GetMaster]get master stream[%p].", master_.ptr());
    return master_;
}

HcclResult OpBaseStreamManager::ClearSlaves()
{
    std::unique_lock<std::mutex> lock(slavesMutex_);
    slaves_.clear();
    slaveDevices_.clear();
    HCCL_DEBUG("[OpBaseStreamManager][GetMaster]clear slave streams success.");
    return HCCL_SUCCESS;
}

} // namespace hccl
