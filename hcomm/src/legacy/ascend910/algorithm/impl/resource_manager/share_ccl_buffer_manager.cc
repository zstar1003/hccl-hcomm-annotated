/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "adapter_rts_common.h"
#include "share_ccl_buffer_manager.h"

namespace hccl {

ShareCCLbufferMgr& ShareCCLbufferMgr::GetInstance()
{
    static ShareCCLbufferMgr shareCCLbufferMgr[MAX_MODULE_DEVICE_NUM];
    s32 deviceLogicId;
    hrtGetDevice(&deviceLogicId);
    if (static_cast<u32>(deviceLogicId) >= MAX_MODULE_DEVICE_NUM) {
        HCCL_WARNING("[ShareCCLbufferMgr][GetInstance] deviceLogicID[%d] is invalid", deviceLogicId);
        return shareCCLbufferMgr[0];
    }
    return shareCCLbufferMgr[deviceLogicId];
}

HcclResult ShareCCLbufferMgr::CreateDevMem(u64 size, DeviceMem& buffer)
{
    CHK_PRT_RET(
        size == 0, HCCL_INFO("[ShareCCLbufferMgr][CreateDevMem]buffer size is zero. not need to malloc memory"),
        HCCL_SUCCESS);

    CHK_PRT_RET(
        (size > ULONG_MAX), HCCL_ERROR("[ShareCCLbufferMgr][CreateDevMem]buffer size is greater than %llu", ULONG_MAX),
        HCCL_E_PARA);

    buffer = DeviceMem::alloc(size);
    CHK_PRT_RET(
        buffer.ptr() == nullptr,
        HCCL_ERROR("[ShareCCLbufferMgr][CreateDevMem]Create ccl buffer fail,buffer ptr is nullptr"), HCCL_E_PTR);
    HCCL_INFO("[ShareCCLbufferMgr][CreateDevMem] buffer ptr[%p], size[%llu]", buffer.ptr(), buffer.size());
    CHK_PRT_RET(
        static_cast<bool>(size) && !buffer,
        HCCL_ERROR(
            "[ShareCCLbufferMgr][CreateDevMem]Create ccl buffer size[%llu] fail,"
            "please check environmental variable HCCL_BUFFSIZE.",
            size),
        HCCL_E_PTR);
    return HCCL_SUCCESS;
}

HcclResult ShareCCLbufferMgr::RecordShareCCLbuffer(const std::string& bufferName)
{
    CHK_PRT_RET(
        bufferName.empty(),
        HCCL_INFO("[ShareCCLbufferMgr][RecordShareCCLbuffer]: buffername is empty, no need record share buffer"),
        HCCL_SUCCESS);

    std::lock_guard<std::mutex> lock(lock_);
    auto memIter = memRecord_.find(bufferName);
    if (memIter == memRecord_.end()) {
        ShareCCLMem memResource;
        memResource.refCount = static_cast<uint64_t>(1);
        memRecord_.emplace(bufferName, std::move(memResource));
        HCCL_INFO("[ShareCCLbufferMgr][RecordShareCCLbuffer]: buffername=%s, refCount=1", bufferName.c_str());
    } else {
        memIter->second.refCount++;
        HCCL_INFO(
            "[ShareCCLbufferMgr][RecordShareCCLbuffer]: buffername=%s, refCount=%d", bufferName.c_str(),
            memIter->second.refCount);
    }
    return HCCL_SUCCESS;
}

HcclResult ShareCCLbufferMgr::CreateShareCCLbuffer(const std::string& bufferName, u64 bufferSize, DeviceMem& cclBuffer)
{
    CHK_PRT_RET(
        bufferName.empty(),
        HCCL_INFO("[ShareCCLbufferMgr][CreateShareCCLbuffer]: buffername is empty, no need create share buffer"),
        HCCL_SUCCESS);

    std::lock_guard<std::mutex> lock(lock_);
    CHK_PRT_RET(
        bufferSize == 0, HCCL_ERROR("[ShareCCLbufferMgr][CreateShareCCLbuffer]: ccl buffer size is abnormal!"),
        HCCL_E_PARA);
    if (shareBufferSize_ == 0) {
        shareBufferSize_ = bufferSize;
    }
    if (shareBufferSize_ != bufferSize) {
        HCCL_WARNING(
            "[ShareCCLbufferMgr][CreateShareCCLbuffer]: share ccLBuffer size [%d], expect buffsize [%d]",
            shareBufferSize_, bufferSize);
    }

    auto memIter = memRecord_.find(bufferName);
    // 申请inCCL,outCCL buffer
    if (memIter->second.cclBuffer.ptr() == nullptr) {
        CHK_RET(CreateDevMem(bufferSize, memIter->second.cclBuffer));
        CHK_RET(hrtMemSet(memIter->second.cclBuffer.ptr(), bufferSize, bufferSize));
    }
    cclBuffer = memIter->second.cclBuffer;
    HCCL_INFO(
        "[ShareCCLbufferMgr][CreateShareCCLbuffer]: buffername=%s, cclBuffer=%p", bufferName.c_str(), cclBuffer.ptr());
    return HCCL_SUCCESS;
}

HcclResult ShareCCLbufferMgr::FreeShareCCLbuffer(const std::string& bufferName)
{
    CHK_PRT_RET(
        bufferName.empty(),
        HCCL_INFO("[ShareCCLbufferMgr][FreeShareCCLbuffer]: buffername is empty, no need free share cclbuffer"),
        HCCL_SUCCESS);

    std::lock_guard<std::mutex> lock(lock_);
    auto it = memRecord_.find(bufferName);
    if (it == memRecord_.end()) {
        HCCL_ERROR(
            "[ShareCCLbufferMgr][FreeShareCCLbuffer] Cannot find the corresponding record of memory[%s].",
            bufferName.c_str());
        return HCCL_E_PARA;
    }
    int refCnt = --(it->second.refCount);
    if (refCnt == 0) {
        HCCL_INFO("[ShareCCLbufferMgr][FreeShareCCLbuffer]: free share cclbuffername=%s", bufferName.c_str());
        memRecord_.erase(it);
        streamIdMap_.erase(bufferName);
    }
    HCCL_INFO("[ShareCCLbufferMgr][FreeShareCCLbuffer]: buffername=%s, refCount=%d", bufferName.c_str(), refCnt);
    return HCCL_SUCCESS;
}

// 约束共享buffer的算子下发到同一条流
HcclResult ShareCCLbufferMgr::CheckCCLbuffConflict(const std::string& bufferName, s32 streamId)
{
    CHK_PRT_RET(
        bufferName.empty(),
        HCCL_INFO("[ShareCCLbufferMgr][CheckCCLbuffConflict]: buffername is empty, no need check CCLbuff conflict"),
        HCCL_SUCCESS);

    std::lock_guard<std::mutex> lock(lock_);
    auto streamIter = streamIdMap_.find(bufferName);
    if (streamIter == streamIdMap_.end()) {
        // 首次记录该缓冲区的stream ID
        streamIdMap_[bufferName] = streamId;
        HCCL_INFO("CheckCCLbuffConflict: sharebuffer[%s] bound to stream[%d]", bufferName.c_str(), streamId);
        return HCCL_SUCCESS;
    }
    const s32 recordedStreamId = streamIter->second;
    if (streamId != recordedStreamId) {
        HCCL_ERROR(
            "[CheckCCLbuffConflict] sharebuffer[%s] stream conflict: "
            "current %d vs recorded %d",
            bufferName.c_str(), streamId, recordedStreamId);
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

} // namespace hccl
