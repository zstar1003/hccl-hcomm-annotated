/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccl_buffer_manager.h"
#include "log.h"
#include "externalinput_pub.h"
#include "adapter_rts_common.h"
#include "share_ccl_buffer_manager.h"

namespace hccl {
CCLBufferManager::CCLBufferManager()
    : inCCLbuffer_(DeviceMem()),
      outCCLbuffer_(DeviceMem()),
      winExpBuffer_(DeviceMem()),
      inCCLbufferSize_(0),
      outCCLbufferSize_(0),
      winExpBufferSize_(0),
      inAlltoAllvParaBuffer_(DeviceMem()),
      outAlltoAllvParaBuffer_(DeviceMem())
{}

CCLBufferManager::~CCLBufferManager()
{
    if (!static_cast<bool>(isShareCCLbuffer_)) {
        ReleaseCommCCLbuffer();
    }
    ReleaseAlltoAllvParaBuffer();
    ReleaseCommAIVbuffer();
}

HcclResult CCLBufferManager::CreateCCLbuffer(u64 size, DeviceMem& buffer)
{
    CHK_PRT_RET(
        !size, HCCL_INFO("[CCLBufferManager][CreateCCLbuffer]buffer size is zero. not need to malloc memory"),
        HCCL_SUCCESS);

    CHK_PRT_RET(
        (size > ULONG_MAX),
        HCCL_ERROR("[CCLBufferManager][CreateCCLbuffer]buffer size is greater than %llu", ULONG_MAX), HCCL_E_PARA);

    CHK_RET(DeviceMem::alloc(buffer, size));
    HCCL_INFO("[CreateCCLbuffer] buffer ptr[%p], size[%llu]", buffer.ptr(), buffer.size());
    CHK_PRT_RET(
        size && !buffer,
        HCCL_ERROR(
            "[CCLBufferManager][CreateCCLbuffer]Create ccl buffer size[%llu] fail,"
            "please check environmental variable HCCL_BUFFSIZE.",
            size),
        HCCL_E_PTR);
    HCCL_RUN_INFO(
        "[HCCL_TRACE][CreateCCLbuffer]Create ccl buffer success. buffer ptr[%p], size[%llu]", buffer.ptr(),
        buffer.size());
    return HCCL_SUCCESS;
}

HcclResult CCLBufferManager::CreateCommCCLbuffer(const std::string& bufferName)
{
    if (inCCLbufferSize_ == 0) {
        inCCLbufferSize_ = GetExternalInputCCLBuffSize();
    }
    if (outCCLbufferSize_ == 0) {
        outCCLbufferSize_ = GetExternalInputCCLBuffSize();
    }
    if (winExpBufferSize_ == 0) {
        winExpBufferSize_ = EXP_BUFFER_SIZE;
    }

    if (cclBuffer_.ptr() == nullptr) {
        u64 totalSize = inCCLbufferSize_ + outCCLbufferSize_ + winExpBufferSize_;
        // buffername非空则申请共享cclbuffer
        if (!bufferName.empty()) {
            CHK_RET(ShareCCLbufferMgr::GetInstance().CreateShareCCLbuffer(bufferName, totalSize, cclBuffer_));
            isShareCCLbuffer_ = true;
        } else {
            CHK_RET(CreateCCLbuffer(totalSize, cclBuffer_));
            CHK_RET(hrtMemSet(cclBuffer_.ptr(), totalSize, totalSize));
        }
    }

    if (inCCLbuffer_.ptr() == nullptr) {
        inCCLbuffer_ = DeviceMem::create(cclBuffer_.ptr(), inCCLbufferSize_);
    }

    if (outCCLbuffer_.ptr() == nullptr) {
        outCCLbuffer_ = DeviceMem::create(static_cast<u8*>(cclBuffer_.ptr()) + inCCLbufferSize_, outCCLbufferSize_);
    }

    if (winExpBuffer_.ptr() == nullptr) {
        winExpBuffer_ = DeviceMem::create(
            static_cast<u8*>(cclBuffer_.ptr()) + inCCLbufferSize_ + outCCLbufferSize_, winExpBufferSize_);
    }
    HCCL_INFO(
        "[CreateCommCCLbuffer] create cclbuffer, inPtr[%p], outPtr[%p], winExpPtr[%p], isSharebuffer[%d]",
        inCCLbuffer_.ptr(), outCCLbuffer_.ptr(), winExpBuffer_.ptr(), isShareCCLbuffer_);
    return HCCL_SUCCESS;
}

HcclResult CCLBufferManager::CleanCCLbuffer()
{
    if (inCCLbuffer_.ptr() != nullptr) {
        CHK_RET(hrtMemSet(inCCLbuffer_.ptr(), inCCLbuffer_.size(), inCCLbuffer_.size()));
        HCCL_INFO("[CleanCCLbuffer] clean input buffer, ptr[%p], size[%llu]", inCCLbuffer_.ptr(), inCCLbuffer_.size());
    }

    if (outCCLbuffer_.ptr() != nullptr) {
        CHK_RET(hrtMemSet(outCCLbuffer_.ptr(), outCCLbuffer_.size(), outCCLbuffer_.size()));
        HCCL_INFO(
            "[CleanCCLbuffer] clean output buffer, ptr[%p], size[%llu]", outCCLbuffer_.ptr(), outCCLbuffer_.size());
    }
    return HCCL_SUCCESS;
}

HcclResult CCLBufferManager::CleanAIVbuffer(void* bufferPtr)
{
    // 将aiv的bufferPtr空间置于0
    if (bufferPtr != nullptr) {
        CHK_RET(hrtMemSet(bufferPtr, AIV_FLAG_SIZE, AIV_FLAG_SIZE));
        HCCL_INFO("[CleanAIVbuffer] clean aiv buffer, ptr[%p], size[%llu]", bufferPtr, AIV_FLAG_SIZE);
    }
    return HCCL_SUCCESS;
}

HcclResult CCLBufferManager::CreateCommAIVbuffer(bool useOpbaseFlag)
{
    if (useOpbaseFlag) {
        if (inAivOpbaseBuffer_.ptr() == nullptr) {
            CHK_RET(CreateCCLbuffer(AIV_DATA_SIZE, inAivOpbaseBuffer_));
            CHK_RET(CleanAIVbuffer(static_cast<u8*>(inAivOpbaseBuffer_.ptr()) + (AIV_DATA_SIZE - AIV_FLAG_SIZE)));
        }
        if (outAivOpbaseBuffer_.ptr() == nullptr) {
            CHK_RET(CreateCCLbuffer(AIV_FLAG_SIZE, outAivOpbaseBuffer_));
            CHK_RET(CleanAIVbuffer(outAivOpbaseBuffer_.ptr()));
            HCCL_RUN_INFO("[HCCL_TRACE][CreateCommAIVbuffer] OpbaseMode");
        }
    } else {
        if (inAivOffloadbuffer_.ptr() == nullptr) {
            CHK_RET(CreateCCLbuffer(AIV_DATA_SIZE, inAivOffloadbuffer_));
            CHK_RET(CleanAIVbuffer(static_cast<u8*>(inAivOffloadbuffer_.ptr()) + (AIV_DATA_SIZE - AIV_FLAG_SIZE)));
        }
        if (outAivOffloadbuffer_.ptr() == nullptr) {
            CHK_RET(CreateCCLbuffer(AIV_FLAG_SIZE, outAivOffloadbuffer_));
            CHK_RET(CleanAIVbuffer(outAivOffloadbuffer_.ptr()));
            HCCL_RUN_INFO("[HCCL_TRACE][CreateCommAIVbuffer] OffloadMode");
        }
    }
    return HCCL_SUCCESS;
}

HcclResult CCLBufferManager::CreateCommInfoAIVbuffer()
{
    if (aivCommInfoBuffer_.ptr() == nullptr) {
        CHK_RET(CreateCCLbuffer(AIV_COMM_INFO_SIZE, aivCommInfoBuffer_));
    }
    return HCCL_SUCCESS;
}

HcclResult CCLBufferManager::ReleaseCommCCLbuffer()
{
    if ((cclBuffer_.ptr() == nullptr) && (inCCLbuffer_.ptr() == nullptr) && (outCCLbuffer_.ptr() == nullptr)
        && (winExpBuffer_.ptr() == nullptr)) {
        HCCL_RUN_INFO("[HCCL_TRACE][ReleaseCCLbuffer]CCLBuffer is null, no need to release.");
        return HCCL_SUCCESS;
    }

    if (cclBuffer_.ptr() != nullptr) {
        HCCL_RUN_INFO(
            "[HCCL_TRACE][ReleaseCCLbuffer]Release cclBuffer. buffer ptr[%p], size[%llu]", cclBuffer_.ptr(),
            cclBuffer_.size());
        cclBuffer_.free();
    }

    if (inCCLbuffer_.ptr() != nullptr) {
        HCCL_RUN_INFO(
            "[HCCL_TRACE][ReleaseCCLbuffer]Release incclBuffer. buffer ptr[%p], size[%llu]", inCCLbuffer_.ptr(),
            inCCLbuffer_.size());
        inCCLbuffer_.free();
    }

    if (outCCLbuffer_.ptr() != nullptr) {
        HCCL_RUN_INFO(
            "[HCCL_TRACE][ReleaseCCLbuffer]Release outcclBuffer. buffer ptr[%p], size[%llu]", outCCLbuffer_.ptr(),
            outCCLbuffer_.size());
        outCCLbuffer_.free();
    }

    if (winExpBuffer_.ptr() != nullptr) {
        HCCL_RUN_INFO(
            "[HCCL_TRACE][ReleaseCCLbuffer]Release expcclBuffer. buffer ptr[%p], size[%llu]", winExpBuffer_.ptr(),
            winExpBuffer_.size());
        winExpBuffer_.free();
    }

    if ((cclBuffer_.ptr() == nullptr) && (inCCLbuffer_.ptr() == nullptr) && (outCCLbuffer_.ptr() == nullptr)
        && (winExpBuffer_.ptr() == nullptr)) {
        HCCL_RUN_INFO("[HCCL_TRACE][ReleaseCCLbuffer]Release CCLbuffer success.");
    }
    return HCCL_SUCCESS;
}

HcclResult CCLBufferManager::ReleaseCommAIVbuffer()
{
    HCCL_RUN_INFO(
        "[HCCL_TRACE][ReleaseAIVbuffer]Release inAivOpbaseBuffer. buffer ptr[%p], size[%llu]", inAivOpbaseBuffer_.ptr(),
        inAivOpbaseBuffer_.size());
    inAivOpbaseBuffer_.free();
    HCCL_RUN_INFO(
        "[HCCL_TRACE][ReleaseAIVbuffer]Release outAivOpbaseBuffer. buffer ptr[%p], size[%llu]",
        outAivOpbaseBuffer_.ptr(), outAivOpbaseBuffer_.size());
    outAivOpbaseBuffer_.free();
    HCCL_RUN_INFO(
        "[HCCL_TRACE][ReleaseAIVbuffer]Release inAivOffloadbuffer. buffer ptr[%p], size[%llu]",
        inAivOffloadbuffer_.ptr(), inAivOffloadbuffer_.size());
    inAivOffloadbuffer_.free();
    HCCL_RUN_INFO(
        "[HCCL_TRACE][ReleaseAIVbuffer]Release outAivOffloadbuffer. buffer ptr[%p], size[%llu]",
        outAivOffloadbuffer_.ptr(), outAivOffloadbuffer_.size());
    outAivOffloadbuffer_.free();
    HCCL_RUN_INFO(
        "[HCCL_TRACE][ReleaseAIVbuffer]Release aivCommInfoBuffer. buffer ptr[%p], size[%llu]", aivCommInfoBuffer_.ptr(),
        aivCommInfoBuffer_.size());
    aivCommInfoBuffer_.free();
    if (inAivOpbaseBuffer_.ptr() == nullptr && outAivOpbaseBuffer_.ptr() == nullptr
        && inAivOffloadbuffer_.ptr() == nullptr && outAivOffloadbuffer_.ptr() == nullptr
        && aivCommInfoBuffer_.ptr() == nullptr) {
        HCCL_RUN_INFO("[HCCL_TRACE][ReleaseAIVbuffer]Release AIV buffer success.");
    }
    return HCCL_SUCCESS;
}

HcclResult CCLBufferManager::ClearCommAIVbuffer()
{
    if (inAivOpbaseBuffer_.ptr() != nullptr) {
        CHK_RET(CleanAIVbuffer(static_cast<u8*>(inAivOpbaseBuffer_.ptr()) + (AIV_DATA_SIZE - AIV_FLAG_SIZE)));
    }
    if (outAivOpbaseBuffer_.ptr() != nullptr) {
        CHK_RET(CleanAIVbuffer(outAivOpbaseBuffer_.ptr()));
    }
    if (inAivOffloadbuffer_.ptr() != nullptr) {
        CHK_RET(CleanAIVbuffer(static_cast<u8*>(inAivOffloadbuffer_.ptr()) + (AIV_DATA_SIZE - AIV_FLAG_SIZE)));
    }
    if (outAivOffloadbuffer_.ptr() != nullptr) {
        CHK_RET(CleanAIVbuffer(outAivOffloadbuffer_.ptr()));
    }
    return HCCL_SUCCESS;
}

DeviceMem& CCLBufferManager::GetInAivOpbaseBuffer() { return inAivOpbaseBuffer_; }

DeviceMem& CCLBufferManager::GetOutAivOpbaseBuffer() { return outAivOpbaseBuffer_; }

DeviceMem& CCLBufferManager::GetInAivOffloadbuffer() { return inAivOffloadbuffer_; }

DeviceMem& CCLBufferManager::GetOutAivOffloadbuffer() { return outAivOffloadbuffer_; }

DeviceMem& CCLBufferManager::GetCommCCLBuffer() { return cclBuffer_; }

HcclResult CCLBufferManager::InitCCLbuffer(u64 inCCLbufferSize, u64 outCCLbufferSize)
{
    inCCLbufferSize_ = inCCLbufferSize;
    outCCLbufferSize_ = outCCLbufferSize;
    return HCCL_SUCCESS;
}

void* CCLBufferManager::GetCCLbufferAddr(const DeviceMem& buffer)
{
    if (buffer.ptr() == nullptr) {
        return nullptr;
    } else {
        return static_cast<void*>(reinterpret_cast<u8*>(buffer.ptr()));
    }
}

DeviceMem& CCLBufferManager::GetInCCLbuffer() { return inCCLbuffer_; }

DeviceMem& CCLBufferManager::GetCommExpBuffer() { return winExpBuffer_; }

DeviceMem& CCLBufferManager::GetAivCommInfoBuffer() { return aivCommInfoBuffer_; }

HcclResult CCLBufferManager::GetInCCLbuffer(void*& buffer, u64& size)
{
    buffer = GetCCLbufferAddr(inCCLbuffer_);
    size = inCCLbufferSize_;
    return HCCL_SUCCESS;
}

u64 CCLBufferManager::GetInCCLbufferSize() const { return inCCLbufferSize_; }

DeviceMem& CCLBufferManager::GetOutCCLbuffer() { return outCCLbuffer_; }

HcclResult CCLBufferManager::GetOutCCLbuffer(void*& buffer, u64& size)
{
    buffer = GetCCLbufferAddr(outCCLbuffer_);
    size = outCCLbufferSize_;
    return HCCL_SUCCESS;
}

u64 CCLBufferManager::GetOutCCLbufferSize() const { return outCCLbufferSize_; }

u64 CCLBufferManager::GetExpBufferSize() const { return winExpBufferSize_; }

DeviceMem CCLBufferManager::GetCommRegMem(const DeviceMem& mem, MemAttr memAttr, bool aivMode)
{
    u64 commMemSize = 0;
    if ((GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) && (!aivMode)) {
        // 单算子模式时，仅在第一次集合通信时创建子通信域，注册通信内存。需要将整个CCLbuffer注册进通信域。
        if (memAttr == MemAttr::IN_CCL_BUFFER) {
            commMemSize = inCCLbufferSize_;
        } else if (memAttr == MemAttr::OUT_CCL_BUFFER) {
            commMemSize = outCCLbufferSize_;
        }
    } else {
        commMemSize = mem.size();
    }
    DeviceMem commMem = DeviceMem::create(mem.ptr(), commMemSize);
    return commMem;
}

HcclResult CCLBufferManager::InitAlltoAllvParaBuffer(u64 inBufferSize, u64 outBufferSize)
{
    CHK_RET(CreateCCLbuffer(inBufferSize, inAlltoAllvParaBuffer_));
    CHK_RET(CreateCCLbuffer(outBufferSize, outAlltoAllvParaBuffer_));
    return HCCL_SUCCESS;
}

DeviceMem& CCLBufferManager::GetInAlltoAllvParaBuffer() { return inAlltoAllvParaBuffer_; }

DeviceMem& CCLBufferManager::GetOutAlltoAllvParaBuffer() { return outAlltoAllvParaBuffer_; }

void CCLBufferManager::ReleaseAlltoAllvParaBuffer()
{
    inAlltoAllvParaBuffer_.free();
    outAlltoAllvParaBuffer_.free();
}

HcclResult CCLBufferManager::GetIndependentOpCCLbuffer(void*& buffer, uint64_t& size)
{
    HCCL_INFO("[GetIndependentOpCCLbuffer] cclBuffer_[%p]", cclBuffer_.ptr());
    buffer = GetCCLbufferAddr(cclBuffer_);
    if (buffer == nullptr) {
        CHK_RET(CreateCommCCLbuffer());
        buffer = GetCCLbufferAddr(cclBuffer_);
    }
    // 大小在通信域初始化时调取InitCCLbuffer设置，MC1MB内存不对外暴露
    size = inCCLbufferSize_ + outCCLbufferSize_;
    return HCCL_SUCCESS;
}
} // namespace hccl
