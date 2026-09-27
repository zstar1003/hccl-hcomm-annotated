/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef H_COM_PRIVATE_V2_H
#define H_COM_PRIVATE_V2_H

#include "hccl/base.h"

constexpr u32 ESTIMATE_CCU_TASK_PER_STREAM = 20;
#ifdef __cplusplus
extern "C" {
#endif // __cplusplus
HcclResult __attribute__((weak)) HcomAllGatherV2(
    const char* tag, void* inputPtr, void* outputPtr, u64 inputCount, HcclDataType dataType, const char* group,
    rtStream_t stream);
HcclResult __attribute__((weak)) HcomAllGatherVV2(
    const char* tag, void* sendBuf, u64 sendCount, void* recvBuf, void* recvCounts, void* rdispls,
    HcclDataType dataType, const char* group, rtStream_t stream);
HcclResult __attribute__((weak)) HcomAllReduceV2(
    const char* tag, void* inputPtr, void* outputPtr, u64 count, HcclDataType dataType, HcclReduceOp op,
    const char* group, rtStream_t stream);
HcclResult __attribute__((weak)) HcomReduceScatterV2(
    const char* tag, void* inputPtr, void* outputPtr, u64 count, HcclDataType dataType, HcclReduceOp op,
    const char* group, rtStream_t& stream);
HcclResult __attribute__((weak)) HcomReduceScatterVV2(
    const char* tag, void* sendBuf, void* sendCounts, void* sdispls, void* recvBuf, u64 recvCount,
    HcclDataType dataType, HcclReduceOp op, const char* group, rtStream_t stream);
HcclResult __attribute__((weak)) HcomSendV2(
    const char* tag, void* inputPtr, u64 count, HcclDataType dataType, u32 destRank, u32 srTag, const char* group,
    rtStream_t& stream);
HcclResult __attribute__((weak)) HcomReceiveV2(
    const char* tag, void* outputPtr, u64 count, HcclDataType dataType, u32 srcRank, u32 srTag, const char* group,
    rtStream_t& stream);
HcclResult __attribute__((weak)) HcomGetWorkspaceSubStreamNumV2(
    const char* group, u64& streamNum, u64 dataSize, HcclDataType dataType, HcclCMDType optype);
HcclResult __attribute__((weak))
HcomGetWorkspaceMemSizeV2(const std::string& opType, u64 count, HcclDataType dataType, const char* group, u64& memSize);
HcclResult __attribute__((weak)) HcomSetWorkspaceResourceV2(
    const std::string& tag, const char* group, std::vector<rtStream_t> stream, void* memPtr, u64 maxSize);
HcclResult __attribute__((weak)) HcomGetRankIdV2(const char* group, u32* rankId);
HcclResult __attribute__((weak)) HcomBroadcastV2(
    const char* tag, void* ptr, u64 count, HcclDataType dataType, u32 root, const char* group, rtStream_t stream);
HcclResult __attribute__((weak)) HcomReduceV2(
    const char* tag, void* inputPtr, void* outputPtr, u64 count, HcclDataType dataType, HcclReduceOp op, u32 root,
    const char* group, rtStream_t stream);
HcclResult __attribute__((weak)) HcomGetAlltoAllStagedWorkSpaceMemSizeV2(
    const char* group, u64* sendCounts, u64* sdispls, HcclDataType sendType, u64* recvCounts, u64* rdispls,
    HcclDataType recvType, u64& memSize);
HcclResult __attribute__((weak)) HcomGetAlltoAllvcStagedWorkSpaceMemSizeV2(const char* group, u64& memSize);
HcclResult __attribute__((weak)) HcomAlltoAllV2(
    const void* sendBuf, u64 sendCount, HcclDataType sendType, const void* recvBuf, u64 recvCount,
    HcclDataType recvType, const char* group, rtStream_t stream, const char* tag);
HcclResult __attribute__((weak)) HcomAlltoAllVV2(
    const void* sendBuf, const void* sendCounts, const void* sdispls, HcclDataType sendType, const void* recvBuf,
    const void* recvCounts, const void* rdispls, HcclDataType recvType, const char* group, rtStream_t stream,
    const char* tag);
HcclResult __attribute__((weak)) HcomAlltoAllVCV2(
    const void* sendBuf, const void* sendCountMatrix, HcclDataType sendType, const void* recvBuf, HcclDataType recvType,
    const char* group, rtStream_t stream, const char* tag);
HcclResult __attribute__((weak)) HcomGetLocalRankSizeV2(const char* group, u32* localRankSize);
HcclResult __attribute__((weak)) HcomGetLocalRankIdV2(const char* group, u32* localRankId);

HcclResult __attribute__((weak)) HcomCalcTaskNumV2(HcomOpParam* hcomOpParam, u32& taskNum);
HcclResult __attribute__((weak)) HcomGetTopoDescV2(const char* group, HcclTopoDescs* topoDescs, uint32_t topoSize);
HcclResult __attribute__((weak)) HcomGetCommHandleByGroupV2(const char* group, HcclComm* commHandle);

HcclResult __attribute__((weak)) HcomCreateCommCclBufV2(const char* group);
HcclResult __attribute__((weak)) HcomGetInCclBufV2(const char* group, void*& commInputPtr, u64& commInputSize);
HcclResult __attribute__((weak)) HcomGetOutCclBufV2(const char* group, void*& commOutputPtr, u64& commOutputSize);
HcclResult __attribute__((weak)) HcomGraphCreateCommCclBufV2(const int64_t& hcomComm);
HcclResult __attribute__((weak))
HcomGraphGetInCclBufV2(const int64_t& hcomComm, void*& commInputPtr, u64& commInputSize);
HcclResult __attribute__((weak))
HcomGraphGetOutCclBufV2(const int64_t& hcomComm, void*& commOutputPtr, u64& commOutputSize);
HcclResult __attribute__((weak)) HcclCommGraphGetRankIdV2(s64 opBaseHcom, u32* rankId);
HcclResult __attribute__((weak)) HcclCommGraphGetRankSizeV2(s64 opBaseHcom, u32* rankSize);
HcclResult __attribute__((weak)) HcclCommGraphAllGatherV2(
    const char* tag, void* inputPtr, void* outputPtr, u64 inputCount, HcclDataType dataType, s64 opBaseHcom,
    rtStream_t stream);
HcclResult __attribute__((weak)) HcomGraphAllReduceV2(
    const char* tag, void* inputPtr, void* outputPtr, u64 count, HcclDataType dataType, HcclReduceOp op, s64 opBaseHcom,
    rtStream_t stream);
HcclResult __attribute__((weak)) HcomGraphReduceV2(
    const char* tag, void* inputPtr, void* outputPtr, u64 count, HcclDataType dataType, HcclReduceOp op, u32 root,
    s64 opBaseHcom, rtStream_t stream);
HcclResult __attribute__((weak)) HcomGraphReduceScatterV2(
    const char* tag, void* inputPtr, void* outputPtr, u64 count, HcclDataType dataType, HcclReduceOp op, s64 opBaseHcom,
    rtStream_t& stream);
HcclResult __attribute__((weak)) HcomGraphSendV2(
    const char* tag, void* inputPtr, u64 count, HcclDataType dataType, u32 destRank, u32 srTag, s64 opBaseHcom,
    rtStream_t& stream);
HcclResult __attribute__((weak)) HcomGraphReceiveV2(
    const char* tag, void* outputPtr, u64 count, HcclDataType dataType, u32 srcRank, u32 srTag, s64 opBaseHcom,
    rtStream_t& stream);
HcclResult __attribute__((weak)) HcomGraphBroadcastV2(
    const char* tag, void* ptr, u64 count, HcclDataType dataType, u32 root, s64 opBaseHcom, rtStream_t stream);

HcclResult __attribute__((weak)) HcomGetIndirectInCclBufV2(const char* group, void*& commInputPtr, u64& commInputSize);
HcclResult __attribute__((weak))
HcomGetIndirectOutCclBufV2(const char* group, void*& commOutputPtr, u64& commOutputSize);

HcclResult __attribute__((weak)) HcomGetDevTypeV2(DevType& devType);
HcclResult __attribute__((weak)) HcomGetInitStatusV2(bool& initiated);
HcclResult __attribute__((weak)) HcomCheckCommValidityV2(const char* group);
HcclResult __attribute__((weak)) HcomSetGlobalWorkSpaceV2(const char* group, std::vector<void*>& globalWorkSpaceAddr);
HcclResult __attribute__((weak)) HcomSupportDeterministicOptimV2(const char* group, bool& isDeterministicOptim);
HcclResult __attribute__((weak)) HcomSetAivCoreLimitV2(const char* group, u32 aivCoreLimit);
HcclResult __attribute__((weak)) HcomSetQosCfgV2(const char* group, const u32 qosCfg);
HcclResult __attribute__((weak)) HcomSelectAlgV2(
    s64 comm, const char* group, HcclCMDType opType, u64 count, HcclDataType dataType, HcclReduceOp op,
    int32_t aivCoreLimit, bool& ifAiv, std::string& algName);
HcclResult __attribute__((weak)) HcomGraphSelectAlgV2(
    s64 comm, const char* group, HcclCMDType opType, u64 count, HcclDataType dataType, HcclReduceOp op,
    int32_t aivCoreLimit, bool& ifAiv, std::string& algName);
HcclResult __attribute__((weak)) HcomUnloadTaskV2(const std::string group, const char* tag);
HcclResult __attribute__((weak)) HcclCommGraphUnloadTaskV2(s64 opBaseHcom, const char* tag);
HcclResult __attribute__((weak)) HcomGetCommCCLBufferSizeV2();
HcclResult __attribute__((weak)) HcclCommResetQosCfgV2();
HcclResult __attribute__((weak)) HcomResetQosCfgV2();
HcclResult __attribute__((weak)) HcclCommSetQosCfgV2();
HcclResult __attribute__((weak)) HcomGetL0TopoTypeExV2(const char* group, CommTopo* topoType, uint32_t flag);
HcclResult __attribute__((weak)) HcomGetRankSizeExV2(const char* group, uint32_t* rankSize, uint32_t flag);
HcclResult __attribute__((weak))
HcomMc2AiCpuStreamAllocAndGetV2(const char* group, u32 streamMode, rtStream_t* aiCpuStream);
HcclResult __attribute__((weak)) HcomSetAivClearEnableV2(const char* group, bool aivClearEnable);
HcclResult __attribute__((weak)) HcomCalcNumBlocksV2(
    const char* group, HcclCMDType opType, u64 count, HcclDataType dataType, int32_t aivCoreLimit, std::string& algName,
    u32& numBlocks);
HcclResult __attribute__((weak)) HcclGetAlgExecParamV2(
    const std::string& tag, const char* group, u64 count, void* inputPtr, void* outputPtr, HcclCMDType opType,
    bool clearEnable, HcclDataType dataType, HcclReduceOp op, void*& commContext, u64& len, u32 aivCoreLimit);
HcclResult __attribute__((weak)) HcomGetDevIdV2(const char* group, s32* devId);
HcclResult __attribute__((weak)) HcomSetAttachedStreamV2();
HcclResult __attribute__((weak)) HcomReleaseSubCommsV2();
#ifdef __cplusplus
}
#endif // __cplusplus
#endif // H_COM_PRIVATE_V2_H
