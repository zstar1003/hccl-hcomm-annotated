/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCL_BUFFER_MANAGER_H
#define HCCL_CCL_BUFFER_MANAGER_H

#include "mem_device_pub.h"
#include "mem_host_pub.h"

namespace hccl {

enum class MemAttr { IN_CCL_BUFFER = 0, OUT_CCL_BUFFER = 1 };

constexpr s64 AIV_FLAG_SIZE = 4 * 1024 * 1024;  // aiv算子需要的flag区域大小
constexpr s64 AIV_DATA_SIZE = 36 * 1024 * 1024; // aiv算子需要的data区域大小
constexpr s64 AIV_COMM_INFO_SIZE = 32 * 1024; // aiv算子需要的通信域信息区域大小，当前最大768*2*8 Byte
constexpr u64 EXP_BUFFER_SIZE = 1 * 1024 * 1024; // 拓展内存, 供MC2使用

class CCLBufferManager {
public:
    CCLBufferManager();
    ~CCLBufferManager();
    HcclResult CreateCommCCLbuffer(const std::string& bufferName = "");
    HcclResult CreateCommAIVbuffer(bool useOpbaseFlag);
    HcclResult CreateCommInfoAIVbuffer();
    HcclResult ReleaseCommCCLbuffer();
    HcclResult ReleaseCommAIVbuffer();
    HcclResult InitCCLbuffer(u64 inCCLbufferSize, u64 outCCLbufferSize);
    DeviceMem& GetInCCLbuffer();
    DeviceMem& GetCommExpBuffer();
    HcclResult GetInCCLbuffer(void*& buffer, u64& size);
    u64 GetInCCLbufferSize() const;
    DeviceMem& GetOutCCLbuffer();
    HcclResult GetOutCCLbuffer(void*& buffer, u64& size);
    DeviceMem& GetCommCCLBuffer();
    u64 GetOutCCLbufferSize() const;
    u64 GetExpBufferSize() const;
    DeviceMem& GetInAivOpbaseBuffer();
    DeviceMem& GetOutAivOpbaseBuffer();
    DeviceMem& GetInAivOffloadbuffer();
    DeviceMem& GetOutAivOffloadbuffer();
    HcclResult ClearCommAIVbuffer();
    DeviceMem& GetAivCommInfoBuffer();
    DeviceMem GetCommRegMem(const DeviceMem& mem, MemAttr memAttr, bool aivMode);
    HcclResult InitAlltoAllvParaBuffer(u64 inBufferSize, u64 outBufferSize);
    DeviceMem& GetInAlltoAllvParaBuffer();
    DeviceMem& GetOutAlltoAllvParaBuffer();
    void ReleaseAlltoAllvParaBuffer();
    HcclResult CleanCCLbuffer();
    HcclResult CleanAIVbuffer(void* bufferPtr);
    HcclResult GetIndependentOpCCLbuffer(void*& buffer, uint64_t& size);

private:
    HcclResult CreateCCLbuffer(u64 size, DeviceMem& buffer);
    void* GetCCLbufferAddr(const DeviceMem& buffer);

    DeviceMem cclBuffer_;
    DeviceMem inCCLbuffer_;
    DeviceMem outCCLbuffer_;
    DeviceMem winExpBuffer_ = DeviceMem();
    u64 inCCLbufferSize_;
    u64 outCCLbufferSize_;
    u64 winExpBufferSize_;
    DeviceMem inAlltoAllvParaBuffer_;
    DeviceMem outAlltoAllvParaBuffer_;
    DeviceMem inAivOpbaseBuffer_ = DeviceMem();
    DeviceMem outAivOpbaseBuffer_ = DeviceMem();
    DeviceMem inAivOffloadbuffer_ = DeviceMem();
    DeviceMem outAivOffloadbuffer_ = DeviceMem();
    DeviceMem aivCommInfoBuffer_ = DeviceMem(); // 单算子使用固定内存如CCL建链，每个通信域只使用一块内存，不需要注册
    bool isShareCCLbuffer_ = false; // cclbuffer是否为通信域共享buffer
};
} // namespace hccl

#endif // HCCL_CCL_BUFFER_MANAGER_H
