/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_TASK_H
#define OP_TASK_H

#include <memory>
#include "hccl_types.h"
#include "operator_param_type.h"
#include "es_private.h"

namespace hccl {
enum class OpTaskType {
    SEND,
    RECV,
    ISEND, // immediately return send
    IRECV, // immediately return send
    WAIT,  // 等待数据收发完成

    PROBE,      // 探测
    PROBE_RECV, // 探测后接收
    DROP_DUPLICATES,
    REMOTE_LOOKUP_PROCESS,

    DROP_DUPLICATES_KERNEL_LAUNCH,
    SEND_KEYS_KERNEL_LAUNCH,
    EVENT_SEND_RECORD,
    EVENT_WAIT_RECV_DONE,
    RESET_UNIQUE_HANDLE_KERNEL_LAUNCH,
    EVENT_WAIT_SEND_DONE,
    RECV_VALUES_KERNEL_LAUNCH,
    RECOVER_VALUE_KERNEL_LAUNCH,

    MEMCPY,
    UPDATE,
    KEYS_DUPLICATES,
    RECOVER_VALUE,
    RESET_UNIQUE_TASK,
    RESET_UNIQUE_HANDLE,
    WAIT_SEND_KEY_FINISH,
    REDUCE_SUM,
    KEY_REDUCE,
    SEND_REQUEST,
    RECV_RESPONSE
};

struct OpTask {
    OpTaskType taskType;

    explicit OpTask(OpTaskType taskTypeA) : taskType(taskTypeA) {}

    explicit OpTask() = default;
    virtual ~OpTask() = default;
};

using OpTaskPtr = std::unique_ptr<OpTask>;

struct TransTask : public OpTask {
    s32 tag;
    u32 dstRank = INVALID_VALUE_RANKID;
    u32 srcRank = INVALID_VALUE_RANKID;
    u64 count = 0;
    HcclDataType dataType = HCCL_DATA_TYPE_RESERVED;
    void* sendMem = nullptr;
    void* recvMem = nullptr;
    u64 remoteMemOffset = 0;
    void* stream = nullptr;
    std::vector<u32> improbeRanks;
    void* rankSendCount = nullptr;
    u64 offset = 0;
    void* handle = nullptr;
    HcclRequest* outRequest = nullptr;
    std::vector<MemoryStartAndSize>* addrInfo = nullptr;
    std::vector<void*> outputs;
    s32 outZerocpyFlag;
    bool errorFlag = false;

    struct RdmaBuffer responseBuffer {};
};

struct WaitSomeTask : public TransTask {
    s32 requestCount;
    HcclRequest* requestArray = nullptr;
    s32* compCount = nullptr;
    s32* compIndices = nullptr;
    HcclStatus* compStatus = nullptr;
};

struct MemTask : public OpTask {
    const void* srcMem = nullptr;
    void* dstMem = nullptr;
    u64 maxSize = 0;
    u64 offset = 0;
    u64 size = 0;
    void* stream = nullptr;
};

struct PartitionMapTask : public OpTask {
    s64* keys = nullptr;
    void* value = nullptr;
    s64* keyNumInput{};
    s32* uniqueIndices{};
    void* indices{};
    void* numUniqued{};
    void* psSeg{};
    void* psSegNum{};
    void* stream = nullptr;
    u64 valueItemSize = 0;
    std::map<u32, u64>* waitLookUpRanksPtr = nullptr;
    void* keyTransferMem = nullptr;
    u64 keyTransferMemSize = 0;
    void* valueTransferMem = nullptr;
    u64 valueTransferMemSize = 0;
    s32* keyCount{};

    void* rdmaEnveInfosTransferMem{};
    u64 rdmaEnveInfosTransferMemSize{};
    u64 workerspaceMemSize{};
    HcclComm comm = nullptr;
    void* outputTableId = nullptr;
    s32 insertFlag = 0;
    s32 intZerocpyFlag{};
    s32 outZerocpyFlag{};
    u32 cubeIndex{};                       // 表示流水执行其keys切分后的index
    u64 pipelineKeyNum = 0;                // 记录流水情况下的key max num，用于去重矩阵的reserve
    std::vector<rtStream_t> subStreamInfo; // 流水任务编排需要的从流
    std::vector<HcclRtNotify> notifyInfo;  // 流水任务编排需要的notify
    u64 keyMaxNum{};
    u32 tableId{};
    void* tableIdAddr{nullptr};
    s32 tag{};
    u32 userRank = 0;

    bool usePipeline{false};
    // pairedMode为true，表示采用了Paired接口
    bool pairedMode{false};
    bool enableKeyCounter{false};
    bool disableUnique{false};
    bool uniqued{false};
    bool haveRdmaConn{false};

    PartitionMapTask& operator=(const PartitionMapTask& other) = default;

    PartitionMapTask& operator=(const EmbeddingServiceParam& other) noexcept
    {
        tag = other.tag;
        pipelineKeyNum = other.keyMaxNum;
        keyMaxNum = other.keyMaxNum;
        keys = static_cast<s64*>(other.keys);
        value = other.values;
        stream = other.embeddingParam.stream;
        valueItemSize = other.embeddingParam.valueItemSize;
        keyTransferMem = other.embeddingParam.keyTransferMem;
        valueTransferMem = other.embeddingParam.valueTransferMem;
        rdmaEnveInfosTransferMem = other.embeddingParam.rdmaEnveInfosTransferMem;
        rdmaEnveInfosTransferMemSize = other.embeddingParam.rdmaEnveInfosTransferMemSize;

        workerspaceMemSize = other.embeddingParam.workerspaceMemSize;
        stream = other.embeddingParam.stream;
        subStreamInfo = other.embeddingParam.subStreamInfo;
        outputTableId = other.embeddingParam.outputTableId;
        notifyInfo = other.embeddingParam.notifyInfo;
        tableId = other.tableId;
        tableIdAddr = other.tableIdAddr;
        insertFlag = other.embeddingParam.insertFlag;
        pairedMode = other.pairedMode;
        uniqued = other.uniqued;
        keyNumInput = other.keyNumInput;
        uniqueIndices = other.uniqueIndices;
        keyCount = other.keyCount;
        disableUnique = other.disableUnique;
        usePipeline = other.embeddingParam.usePipeline;
        cubeIndex = other.embeddingParam.cubeIndex;

        indices = other.indices;
        numUniqued = other.numUniqued;
        psSeg = other.psSeg;
        psSegNum = other.psSegNum;

        comm = other.embeddingParam.comm;
        enableKeyCounter = other.embeddingParam.enableKeyCounter;
        intZerocpyFlag = other.intZerocpyFlag;
        outZerocpyFlag = other.outZerocpyFlag;
        userRank = other.userRank;
        haveRdmaConn = other.haveRdmaConn;

        return *this;
    }
};

struct ResetUniqueHandleTask : public OpTask {
    s32 tag = 0;
    s32 cubeIndex = 0;
    void* stream = nullptr;
    std::vector<rtStream_t> subStreamInfo; // 流水任务编排需要的从流
    std::vector<HcclRtNotify> notifyInfo;  // 流水任务编排需要的notify
    bool usePipeline{false};
};

struct ReduceSumTask : public OpTask {
    s32 tag = 0;
    s64* keys = nullptr;
    HcclDataType keyType = HCCL_DATA_TYPE_RESERVED;
    void* value = nullptr;
    HcclDataType valueType = HCCL_DATA_TYPE_RESERVED;
    u64 keyMaxNum = 0;
    u64 valueItemSize = 0;
    void* keyTransferMem = nullptr;
    void* valueTransferMem = nullptr;
    void* rdmaEnveInfosTransferMem{};
    HcclComm comm = nullptr;
    s32* actualKeyCount = nullptr;
    s32* actualValueCount = nullptr;
    std::vector<MemoryStartAndSize>* keyAddrInfo = nullptr;
    std::vector<MemoryStartAndSize>* valueAddrInfo = nullptr;
    u32* updateKeysSize = nullptr;
    u32* updateValuesSize = nullptr;
    bool pairedMode{false};
    bool haveRdmaConn{false};
    u32 tableId{};
    void* tableIdAddr{nullptr};
    void* globalStepAddr{nullptr};

    ReduceSumTask& operator=(const ReduceSumTask& other) = default;

    ReduceSumTask& operator=(const EmbeddingServiceParam& other) noexcept
    {
        tag = other.tag;
        keyMaxNum = other.keyMaxNum;
        keys = static_cast<s64*>(other.keys);
        value = other.values;
        valueItemSize = other.embeddingParam.valueItemSize;
        keyTransferMem = other.embeddingParam.keyTransferMem;
        valueTransferMem = other.embeddingParam.valueTransferMem;
        rdmaEnveInfosTransferMem = other.embeddingParam.rdmaEnveInfosTransferMem;

        pairedMode = other.pairedMode;
        haveRdmaConn = other.haveRdmaConn;
        comm = other.embeddingParam.comm;
        tableId = other.tableId;
        tableIdAddr = other.tableIdAddr;
        globalStepAddr = other.globalStepAddr;

        return *this;
    }
};

struct PsReduceSumTask : public ReduceSumTask {
    u32 workNum{};
};

struct KeyReduceTask : public ReduceSumTask {
    void* indices{};
    void* numUniqued{};
    void* psSeg{};
    void* psSegNum{};
    // pairedMode为true，表示采用了Paired接口
    bool pairedMode{false};
};

struct UpdateTransTask : public TransTask {
    u32 keyMaxNum = 0;
    u64 valueItemSize = 0;
    void* keyTransferMem = nullptr;
    void* valueTransferMem = nullptr;
    void* rdmaEnveInfosTransferMem{};
    HcclComm comm = nullptr;
    std::vector<s32>* rankTransCompPtr = nullptr;
    std::map<u32, bool> recvWaitFlagMap;
    bool needRecordFlag = true;
    void* transferMemBegin = nullptr;
    u64 transferMemLen = 0;

    std::unordered_map<u32, struct RdmaBuffer>* rdmaResponseAddrsPtr{};
};
} // namespace hccl
#endif
