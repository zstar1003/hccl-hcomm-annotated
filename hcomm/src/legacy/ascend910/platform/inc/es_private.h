/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ES_PRIVATE_H
#define ES_PRIVATE_H

#include <climits>
#include "private_types.h"
#include "hccl_common.h"

#define MEMBER_OFFSET(type, member) ((u64)(&(((type*)nullptr)->member)))

using MemoryStartAndSize = struct MemoryStartAndSizeDef {
    void* startAddr;
    u64 size;
    MemoryStartAndSizeDef() : startAddr(nullptr), size(0) {}
};

const std::string HCCL_KERNEL_OP_TYPE_GATHER = "HcomGather";

// ZEROCOPY
constexpr s32 ZERO_COPY_USED = 1;   // zerocopy使用标识
constexpr s32 ZERO_COPY_UNUSED = 0; // zerocopy未使用标识

constexpr u32 SERVICE_CANCEL_SIGNAL = 0xFFFFFFFF;

struct HcclRdmaSignalInfo {
    u32 type = INVALID_UINT;
    s32 mrRegFlag = INVALID_INT;
    void* notifyAddr;
    u64 len = INVALID_U64;
    hccl::MemType memType = hccl::MEM_TYPE_RESERVED;
    u32 lkey = INVALID_UINT;
};

struct HdcsEmbeddingServiceParam {
    s32 tag = 0;
    u32 devId = 0;
    void* keys = nullptr;
    u64 keyMaxNum = 0;
    u32 tableId{};
    HcclDataType keyType = HCCL_DATA_TYPE_RESERVED;
    s64* keyNumInput{};
    s32* uniqueIndices{};
    s32* keyCount{};
    void* values = nullptr;
    void* indices{};
    void* numUniqued{};
    void* psSeg{};
    void* psSegNum{};
    u64 valueCount = 0;
    HcclDataType valuesType = HCCL_DATA_TYPE_RESERVED;
    char groupName[GROUP_NAME_MAX_LEN] = {0};

    void* tableIdAddr = nullptr;
    s32 insertFlag{};
    s32 valueItemSize{};

    void* keyTransferMem = nullptr;
    u64 keyTransferMemSize = 0;
    void* valueTransferMem = nullptr;
    u64 valueTransferMemSize = 0;
    void* rdmaEnveInfosTransferMem{};
    u64 rdmaEnveInfosTransferMemSize{};
    u64 workerspaceMemSize{};
    s32 cubeIndex{0};
    u64 pipelineKeyNum = 0;
    void* errorMsg{};
    s32 intZerocpyFlag{};
    s32 outZerocpyFlag{};
    void* globalStepAddr{nullptr};

    bool usePipeline{};
    // pairedMode为true，表示采用了Paired接口
    bool pairedMode{false};
    bool enableKeyCounter{};
    bool disableUnique{};
    bool uniqued{false};
    bool haveRdmaConn{false};
};

struct PsBufferInfo {
    s64 offset{};
    s64 count{};
};

struct RdmaBuffer {
    u64 addr;
    u64 buffKey;
};

template <typename T>
T Align(T value, T alignment)
{
    return (value + alignment - 1) / alignment * alignment;
}

constexpr u32 IPC_MEM_ALIGNMENT_BYTE = 32;
#endif /* ES_PRIVATE_H */
