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
#include <hccl/hccl_types.h>
#include <map>
#include <unordered_map>
#include "hccl/base.h"
#include "hccl/hcom.h"
#include "types/dev_type.h"
#include "hccl_types.h"
#include "types.h"
#include "log.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus
HcclResult HcomAllGatherV2(
    const char* tag, void* inputPtr, void* outputPtr, u64 inputCount, HcclDataType dataType, const char* group,
    rtStream_t stream);
HcclResult HcomAllGatherVV2(
    const char* tag, void* sendBuf, u64 sendCount, void* recvBuf, void* recvCounts, void* rdispls,
    HcclDataType dataType, const char* group, rtStream_t stream);
HcclResult HcomAllReduceV2(
    const char* tag, void* inputPtr, void* outputPtr, u64 count, HcclDataType dataType, HcclReduceOp op,
    const char* group, rtStream_t stream);
HcclResult HcomGetRankIdV2(const char* group, u32* rankId);
HcclResult HcomGetWorkspaceSubStreamNumV2(
    const char* group, u64& streamNum, u64 dataSize, HcclDataType dataType, HcclCMDType optype);
HcclResult
HcomGetWorkspaceMemSizeV2(const std::string& opType, u64 count, HcclDataType dataType, const char* group, u64& memSize);
HcclResult HcomSetWorkspaceResourceV2(
    const std::string& tag, const char* group, std::vector<rtStream_t> stream, void* memPtr, u64 maxSize);
HcclResult HcomReduceScatterV2(
    const char* tag, void* inputPtr, void* outputPtr, u64 count, HcclDataType dataType, HcclReduceOp op,
    const char* group, rtStream_t& stream);
HcclResult HcomReduceScatterVV2(
    const char* tag, void* sendBuf, void* sendCounts, void* sdispls, void* recvBuf, u64 recvCount,
    HcclDataType dataType, HcclReduceOp op, const char* group, rtStream_t stream);
HcclResult HcomSendV2(
    const char* tag, void* inputPtr, u64 count, HcclDataType dataType, u32 destRank, u32 srTag, const char* group,
    rtStream_t& stream);
HcclResult HcomReceiveV2(
    const char* tag, void* outputPtr, u64 count, HcclDataType dataType, u32 srcRank, u32 srTag, const char* group,
    rtStream_t& stream);
HcclResult HcomAlltoAllV2(
    const void* sendBuf, u64 sendCount, HcclDataType sendType, const void* recvBuf, u64 recvCount,
    HcclDataType recvType, const char* group, rtStream_t stream, const char* tag);
HcclResult HcomAlltoAllVV2(
    const void* sendBuf, const void* sendCounts, const void* sdispls, HcclDataType sendType, const void* recvBuf,
    const void* recvCounts, const void* rdispls, HcclDataType recvType, const char* group, rtStream_t stream,
    const char* tag);
HcclResult HcomAlltoAllVCV2(
    const void* sendBuf, const void* sendCountMatrix, HcclDataType sendType, const void* recvBuf, HcclDataType recvType,
    const char* group, rtStream_t stream, const char* tag);
HcclResult HcomBroadcastV2(
    const char* tag, void* ptr, u64 count, HcclDataType dataType, u32 root, const char* group, rtStream_t stream);
HcclResult HcomGetAlltoAllStagedWorkSpaceMemSizeV2(
    const char* group, u64* sendCounts, u64* sdispls, HcclDataType sendType, u64* recvCounts, u64* rdispls,
    HcclDataType recvType, u64& memSize);
HcclResult HcomGetAlltoAllvcStagedWorkSpaceMemSizeV2(const char* group, u64& memSize);
HcclResult HcomReduceV2(
    const char* tag, void* inputPtr, void* outputPtr, u64 count, HcclDataType dataType, HcclReduceOp op, u32 root,
    const char* group, rtStream_t stream);
HcclResult HcomGetLocalRankSizeV2(const char* group, u32* localRankSize);
HcclResult HcomGetLocalRankIdV2(const char* group, u32* localRankId);

HcclResult HcomCalcTaskNumV2(HcomOpParam* hcomOpParam, u32& taskNum);

HcclResult HcomGetTopoDescV2(const char* group, HcclTopoDescs* topoDescs, uint32_t topoSize);

HcclResult HcomGetCommHandleByGroupV2(const char* group, HcclComm* commHandle);

HcclResult HcomCreateCommCclBufV2(const char* group);
HcclResult HcomGetInCclBufV2(const char* group, void*& commInputPtr, u64& commInputSize);
HcclResult HcomGetOutCclBufV2(const char* group, void*& commOutputPtr, u64& commOutputSize);
HcclResult HcomGraphCreateCommCclBufV2(const int64_t& hcomComm);
HcclResult HcomGraphGetInCclBufV2(const int64_t& hcomComm, void*& commInputPtr, u64& commInputSize);
HcclResult HcomGraphGetOutCclBufV2(const int64_t& hcomComm, void*& commOutputPtr, u64& commOutputSize);

HcclResult HcclCommGraphGetRankIdV2(s64 opBaseHcom, u32* rankId);
HcclResult HcclCommGraphGetRankSizeV2(s64 opBaseHcom, u32* rankSize);

HcclResult HcclCommGraphAllGatherV2(
    const char* tag, void* inputPtr, void* outputPtr, u64 inputCount, HcclDataType dataType, s64 opBaseHcom,
    rtStream_t stream);
HcclResult HcomGraphAllReduceV2(
    const char* tag, void* inputPtr, void* outputPtr, u64 count, HcclDataType dataType, HcclReduceOp op, s64 opBaseHcom,
    rtStream_t stream);
HcclResult HcomGraphReduceV2(
    const char* tag, void* inputPtr, void* outputPtr, u64 count, HcclDataType dataType, HcclReduceOp op, u32 root,
    s64 opBaseHcom, rtStream_t stream);
HcclResult HcomGraphReduceScatterV2(
    const char* tag, void* inputPtr, void* outputPtr, u64 count, HcclDataType dataType, HcclReduceOp op, s64 opBaseHcom,
    rtStream_t& stream);
HcclResult HcomGraphSendV2(
    const char* tag, void* inputPtr, u64 count, HcclDataType dataType, u32 destRank, u32 srTag, s64 opBaseHcom,
    rtStream_t& stream);
HcclResult HcomGraphReceiveV2(
    const char* tag, void* outputPtr, u64 count, HcclDataType dataType, u32 srcRank, u32 srTag, s64 opBaseHcom,
    rtStream_t& stream);
HcclResult HcomGraphBroadcastV2(
    const char* tag, void* ptr, u64 count, HcclDataType dataType, u32 root, s64 opBaseHcom, rtStream_t stream);

HcclResult HcomGetIndirectInCclBufV2(const char* group, void*& commInputPtr, u64& commInputSize);
HcclResult HcomGetIndirectOutCclBufV2(const char* group, void*& commOutputPtr, u64& commOutputSize);

HcclResult HcomGetDevTypeV2(Hccl::DevType& devType);
HcclResult HcomSetGlobalWorkSpaceV2(const char* group, const std::vector<void*>& globalWorkSpaceAddr);
HcclResult HcomGetInitStatusV2(bool& initiated);
HcclResult HcomCheckCommValidityV2(const char* group);
HcclResult HcomSupportDeterministicOptimV2(const char* group, bool& isDeterministicOptim);
HcclResult HcomSetAivCoreLimitV2(const char* group, u32 aivCoreLimit);
HcclResult HcomSetQosCfgV2(const char* group, const u32 qosCfg);
HcclResult HcomSelectAlgV2(
    s64 comm, const char* group, HcclCMDType opType, u64 count, HcclDataType dataType, HcclReduceOp op,
    int32_t aivCoreLimit, bool& ifAiv, std::string& algName);
HcclResult HcomGraphSelectAlgV2(
    s64 comm, const char* group, HcclCMDType opType, u64 count, HcclDataType dataType, HcclReduceOp op,
    int32_t aivCoreLimit, bool& ifAiv, std::string& algName);
HcclResult HcomGetCommCCLBufferSizeV2();
HcclResult HcclCommResetQosCfgV2();
HcclResult HcomResetQosCfgV2();
HcclResult HcclCommSetQosCfgV2();

HcclResult HcomUnloadTaskV2(const std::string group, const char* tag);
HcclResult HcomSetAivClearEnableV2(const char* group, bool aivClearEnable);

HcclResult HcomCalcNumBlocksV2(
    const char* group, HcclCMDType opType, u64 count, HcclDataType dataType, int32_t aivCoreLimit, std::string& algName,
    u32& numBlocks);

HcclResult HcclGetAlgExecParamV2(
    const std::string& tag, const char* group, u64 count, void* inputPtr, void* outputPtr, HcclCMDType opType,
    bool clearEnable, HcclDataType dataType, HcclReduceOp op, void*& commContext, u64& len, u32 aivCoreLimit);

HcclResult HcomGetDevIdV2(const char* group, s32* devId);
HcclResult HcomGetL0TopoTypeExV2(const char* group, CommTopo* topoType, uint32_t flag);
HcclResult HcomGetRankSizeExV2(const char* group, uint32_t* rankSize, uint32_t flag);
HcclResult HcomMc2AiCpuStreamAllocAndGetV2(const char* group, u32 streamMode, rtStream_t* aiCpuStream);

HcclResult HcomSetAttachedStreamV2();
HcclResult HcomReleaseSubCommsV2();
#ifdef __cplusplus
}
#endif // __cplusplus
#endif // H_COM_PRIVATE_H
