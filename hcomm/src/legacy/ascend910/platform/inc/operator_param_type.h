/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPERATOR_PARAM_TYPE_H
#define OPERATOR_PARAM_TYPE_H

#include "hccl/base.h"
#include "mem_device_pub.h"
#include "es_private.h"

namespace hccl {

struct EmbeddingServiceParam {
    enum class OperatorType {
        COLL_REMOTE_LOOK_UP,
        COLL_REMOTE_LOOK_UP_PAIRED,
        COLL_REMOTE_LOOK_UP_UNIQUED_PAIRED,
        // 非流水task type start
        REMOTE_LOOK_UP,
        GET_LOOK_UP_REQUEST,
        WAIT_LOOK_UP_DATA,
        SET_LOOK_UP_RESPONSE,
        WAIT_SOME,
        ABORT_SELF,
        REMOTE_UPDATE,
        WAIT_UPDATE_DATA,
        COLL_REMOTE_UPDATE,
        COLL_REMOTE_UPDATE_PAIRED,
        RECV_UPDATE_REQUEST,
        SEND_UPDATE_RESPONSE,
        WAIT_UPDATE,
        SERVICE_CANCEL,
        // 非流水task type end

        // 流水task type start
        REMOTE_LOOKUP_KEYS_DUPLICATES,
        REMOTE_LOOKUP_SEND_KEYS,
        REMOTE_LOOKUP_RECV_VALUES,
        REMOTE_LOOKUP_RECOVER_VALUE,
        REMOTE_LOOKUP_RESET_UNIQUE_TASK,
        REMOTE_LOOKUP_RESET_UNIQUE_HANDLE,
        REMOTE_LOOKUP_WAIT_SEND_KEY_FINISH,
        REMOTE_UPDATE_REDUCE_SUM,
        REMOTE_UPDATE_KEY_REDUCE,
        REMOTE_UPDATE_SEND_REQUEST,
        REMOTE_UPDATE_RECV_RESPONSE,
        REMOTE_UPDATE_RESET_UNIQUE_HANDLE
        // 流水task type end
    } opType;

    s32 tag = 0;
    std::string sTag;
    u32 devId = 0;
    u32 tableId{};
    void* keys = nullptr;
    u64 keyMaxNum = 0;
    HcclDataType keyType = HCCL_DATA_TYPE_RESERVED;
    void* values = nullptr;
    void* indices{};
    void* numUniqued{};
    void* psSeg{};
    void* psSegNum{};
    u64 valueCount = 0;
    HcclDataType valuesType = HCCL_DATA_TYPE_RESERVED;
    void* tableIdAddr = nullptr;
    std::string groupName;
    s32 intZerocpyFlag{INVALID_INT};
    s32 outZerocpyFlag{INVALID_INT};
    void* algPtr = nullptr;
    void* opExecutorPtr = nullptr;
    u32 userRank = 0;
    s64* keyNumInput{};
    s32* uniqueIndices{};
    s32* keyCount{};
    s32 maxEmbeddingDim{};
    std::string uniqueTag;
    bool waitAny{false};
    void* globalStepAddr{nullptr};

    // pairedMode为true，表示采用了Paired接口
    bool pairedMode{false};
    bool disableUnique{false};
    bool uniqued{false};
    bool haveRdmaConn{false};
    struct {
        u32 tableId = 0;
        void* stream = nullptr;
        s32 insertFlag = 0;
        s32 valueItemSize = 0;
        std::map<u32, u32> psMap; // {psId, rankId}
        std::map<u32, u64> waitLookUpRanks;
        void* outputTableId = nullptr;
        void* keyTransferMem = nullptr;
        u64 keyTransferMemSize = 0;
        void* valueTransferMem = nullptr;
        u64 valueTransferMemSize = 0;
        void* rdmaEnveInfosTransferMem{};
        u64 rdmaEnveInfosTransferMemSize{};
        u64 workerspaceMemSize{};
        HcclComm comm = nullptr;
        std::vector<MemoryStartAndSize> keysAddrInfo;
        std::vector<MemoryStartAndSize> valuesAddrInfo;
        std::vector<rtStream_t> subStreamInfo;
        std::vector<HcclRtNotify> notifyInfo;
        s32 cubeIndex{0};
        s32 flags{};
        bool usePipeline{false};
        bool enableKeyCounter{};
    } embeddingParam;

    struct {
        std::vector<u32> workerList;
        std::map<u32, u32> psMap; // {psId, rankId}
        u32 count = 0;
        u32 rankId = 0;
        u32 workerRankId = 0;

        struct RdmaBuffer responseBuffer {};
        // rankId -> rdma < addr, key >
        std::unordered_map<u32, struct RdmaBuffer> rdmaResponseAddrs{};

        ReqStatus* reqStatus = nullptr;
        UpdateReqStatus* updateReqStatus = nullptr;
        LookupReqStatus* lookupReqStatus = nullptr;
        HcclRequest* outRequest = nullptr;
        std::map<u32, std::vector<DeviceMem>> transferMems;
        std::map<u32, std::vector<DeviceMem>> rankValueTransferMems;
        HcclComm commHandle = nullptr;
        u32 updateKeysSize = 0;
        u32 updateValuesSize = 0;
        std::vector<s32> rankTransKeysCompFlag;  // rank flag 默认 -1   完成 0   未完成 1
        std::vector<s32> rankTransValueCompFlag; // rank flag 默认 -1   完成 0   未完成 1
        u32 recvKeysCompleteNum = 0;
        u32 recvValueCompleteNum = 0;
        u32 lookupRecvRequestCount = 0;
        bool recvKeyWaitFlag = true;
        bool recvValueWaitFlag = true;
        u32 actualKeyCount = 0;
        u32 actualValueCount = 0;
    } getRequestParam;

    struct {
        s32 requestCount = 0;
        HcclRequest* requestArray = nullptr;
        s32* compCount = nullptr;
        s32* compIndices = nullptr;
        HcclStatus* compStatus = nullptr;
    } waitSomeParam;

    EmbeddingServiceParam& operator=(const EmbeddingServiceParam& other) = default;

    EmbeddingServiceParam& operator=(const HdcsEmbeddingServiceParam& other) noexcept
    {
        tag = other.tag;
        keyMaxNum = other.keyMaxNum;
        keys = other.keys;
        values = other.values;
        tableIdAddr = other.tableIdAddr;
        tableId = other.tableId;
        embeddingParam.insertFlag = other.insertFlag;
        embeddingParam.valueItemSize = other.valueItemSize;
        embeddingParam.keyTransferMem = other.keyTransferMem;
        embeddingParam.valueTransferMem = other.valueTransferMem;
        embeddingParam.rdmaEnveInfosTransferMem = other.rdmaEnveInfosTransferMem;
        haveRdmaConn = other.haveRdmaConn;
        embeddingParam.cubeIndex = other.cubeIndex;
        embeddingParam.usePipeline = other.usePipeline;
        embeddingParam.enableKeyCounter = other.enableKeyCounter;
        intZerocpyFlag = other.intZerocpyFlag;
        outZerocpyFlag = other.outZerocpyFlag;
        disableUnique = other.disableUnique;
        pairedMode = other.pairedMode;
        uniqued = other.uniqued;
        keyNumInput = other.keyNumInput;
        uniqueIndices = other.uniqueIndices;
        keyCount = other.keyCount;
        haveRdmaConn = other.haveRdmaConn;

        return *this;
    }
};

} // namespace hccl
#endif
