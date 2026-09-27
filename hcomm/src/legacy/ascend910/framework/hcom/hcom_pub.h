/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef H_COM_PUB_H
#define H_COM_PUB_H

#include <map>
#include <mutex>

#include "hccl_comm_pub.h"
#include "hcom_common.h"
#include "workflow_pub.h"
#include "hcom.h"

bool HcomIsNormalComm(const char* group);

// 生成通信域标识符
HcclResult HcomGenerateCommId(hccl::HcclCommParams& params);
HcclResult HcomCheckCommValidity(const char* group);
HcclResult HcomGetCCLBufferAvailableSize(u64& size);
HcclResult HcomGetDevId(const char* group, s32* devId);
HcclResult HcclCommGraphGetDevId(s64 opBaseHcom, s32* devId);
HcclResult HcclCommGraphGetIdentifier(s64 opBaseHcom, std::string& identifier);
HcclResult HcclCommGraphGetIdentifierCpu(s64 opBaseHcom, std::string& identifier);
HcclResult CallMsprofReportHostApi(
    hccl::hcclComm* hcclComm, HcclCMDType cmdType, uint64_t beginTime, u64 count, HcclDataType dataType);
#ifdef __cplusplus
extern "C" {
#endif
HcclResult HcomInit(const char* rankTableM, const char* identify, WorkMode commWorkMode = WorkMode::HCCL_MODE_NORMAL);

HcclResult HcomDestroy(void);

// pytorch单算子通信域复用 start
HcclResult HcclCommGraphAllGather(
    const char* tag, void* inputPtr, void* outputPtr, u64 inputCount, HcclDataType dataType, s64 opBaseHcom,
    rtStream_t stream);
HcclResult HcclCommGraphAllReduce(
    const char* tag, void* inputPtr, void* outputPtr, u64 count, HcclDataType dataType, HcclReduceOp op, s64 opBaseHcom,
    rtStream_t stream);
HcclResult HcclCommGraphReduce(
    const char* tag, void* inputPtr, void* outputPtr, u64 count, HcclDataType dataType, HcclReduceOp op, u32 root,
    s64 opBaseHcom, rtStream_t stream);
HcclResult HcclCommGraphBroadcast(
    const char* tag, void* ptr, u64 count, HcclDataType dataType, u32 root, s64 opBaseHcom, rtStream_t stream);
HcclResult HcclCommGraphReduceScatter(
    const char* tag, void* inputPtr, void* outputPtr, u64 count, HcclDataType dataType, HcclReduceOp op, s64 opBaseHcom,
    rtStream_t stream);
HcclResult HcclCommGraphSend(
    const char* tag, void* inputPtr, u64 count, HcclDataType dataType, u32 destRank, u32 srTag, s64 opBaseHcom,
    rtStream_t stream);
HcclResult HcclCommGraphReceive(
    const char* tag, void* outputPtr, u64 count, HcclDataType dataType, u32 srcRank, u32 srTag, s64 opBaseHcom,
    rtStream_t stream);
HcclResult HcclCommGraphAlltoAllV(
    const void* sendBuf, const void* sendCounts, const void* sdispls, HcclDataType sendType, const void* recvBuf,
    const void* recvCounts, const void* rdispls, HcclDataType recvType, s64 opBaseHcom, rtStream_t stream,
    const char* tag);
HcclResult HcclCommGraphAlltoAllVC(
    const void* sendBuf, const void* sendCountMatrix, HcclDataType sendType, const void* recvBuf, HcclDataType recvType,
    s64 opBaseHcom, rtStream_t stream, const char* tag);
HcclResult HcclCommGraphGetAlltoAllStagedWorkSpaceMemSize(
    s64 opBaseHcom, u64* sendCounts, u64* sdispls, HcclDataType sendType, u64* recvCounts, u64* rdispls,
    HcclDataType recvType, u64& memSize);
HcclResult HcclCommGraphSetWorkspaceResource(
    const std::string& tag, s64 opBaseHcom, std::vector<rtStream_t> stream, void* memPtr, u64 maxSize);
HcclResult
HcclCommGraphGetAllReduceScratchSize(s64 opBaseHcom, const u64 count, const HcclDataType dataType, u64& outScratchSize);
HcclResult HcclCommGraphGetRankSize(s64 opBaseHcom, u32* rankSize);
HcclResult HcclCommGraphGetRankId(s64 opBaseHcom, u32* rankId);
HcclResult HcclCommGraphGetWorkspaceSubStreamNum(
    u64 count, HcclDataType dataType, HcclReduceOp op, std::string algName, s64 opBaseHcom, u64& streamNum,
    u64 dataSize = 0, bool ifAiv = false, HcclCMDType opType = HcclCMDType::HCCL_CMD_INVALID);
HcclResult HcclCommGraphSetIsPytorchComm();
// pytorch单算子通信域复用 end

HcclResult HcomGetWorkspaceSubStreamNum(
    const char* group, u64& streamNum, u64 dataSize = 0, HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_RESERVED,
    u32 aivCoreLimit = 0, HcclReduceOp reduceOp = HcclReduceOp::HCCL_REDUCE_SUM, u64 count = 0,
    HcclCMDType optype = HcclCMDType::HCCL_CMD_INVALID);
HcclResult
HcomGetWorkspaceMemSize(const std::string& opType, u64 count, HcclDataType dataType, const char* group, u64& memSize);
HcclResult HcomGetAlltoAllStagedWorkSpaceMemSize(
    const char* group, u64* sendCounts, u64* sdispls, HcclDataType sendType, u64* recvCounts, u64* rdispls,
    HcclDataType recvType, u64& memSize);
HcclResult HcomGetAlltoAllvcStagedWorkSpaceMemSize(
    const char* group, std::vector<SendRecvInfo>& allMeshAggregationSendRecvInfo, u64& memSize);
HcclResult
HcomGetAllReduceScratchSize(const char* group, const u64 count, const HcclDataType dataType, u64& outScratchSize);
HcclResult HcclCommSetAttachedStream(s64 opBaseHcom, u32 graphId, const std::vector<rtStream_t>& stream);
HcclResult HcomSetExecTimeOut(const char* execTimeOut);
HcclResult HcomSetAlgorithm(const char* algo);
HcclResult HcomSetDeterministic(u8 deterministic);
HcclResult HcclCommGraphUnloadTask(s64 opBaseHcom, const char* tag);

HcclResult HcomSetRankTableImpl(const char* rankTableStr);
HcclResult HcomGetActualRankSizeImpl(const char* group, u32* rankSize);
HcclResult HcclCommSetGlobalWorkSpace(s64 opBaseHcom, std::vector<void*>& globalWorkSpaceAddr);

HcclResult HcclCommGetandClearOverFlowTasks(s64 opBaseHcom, std::vector<hccl::HcclDumpInfo>& hcclDumpInfo);
HcclResult HcclCommSupportDeterministicOptim(s64 opBaseHcom, bool& isDeterministicOptim);
HcclResult HcomGetHccsLinkNum(const char* group, u32* numHccsLink);
HcclResult HcomSetQosCfg(const char* group, const u32 qosCfg);
HcclResult HcclCommSetQosCfg(s64 opBaseHcom, const u32 qosCfg);
HcclResult HcomResetQosCfg(const char* group);
HcclResult HcclCommResetQosCfg(s64 opBaseHcom);
HcclResult GenerateGroupHash(std::string& group, std::string& groupHash);

HcclResult HcclCommGraphClearAivSyncBuf(s64 comm, bool aivClearEnable);
HcclResult HcclCommGraphSetAivCoreLimit(s64 comm, u32 aivCoreLimit);

// hcce资源计算相关
HcclResult CalcTaskNum(
    HcomOpParam* hcomOpParam, const u64& streamNum, const s32& deviceNumPerServer, const s32& serverNum,
    bool multiModuleDiffDeviceNumMode, u32& taskNum, DevType devType);

HcclResult CalcTaskNumV2(HcomOpParam* hcomOpParam, u32& taskNum);
HcclResult HcomCalcTaskNum(HcomOpParam* hcomOpParam, u32& taskNum);

HcclResult GetInterComTaskNum(
    const std::string& sCollectiveType, s32 serverNum, s32 deviceNumPerServer, DevType devType, u32& taskNum,
    const std::string& group = HCCL_WORLD_GROUP);
HcclResult GetStreamNumOfflineComp(
    HcclCMDType hcclOpType, s32 serverNum, s32 deviceNumPerServer, bool ifAiv, DevType devType, u64& streamNum,
    const std::string& group = HCCL_WORLD_GROUP);
HcclResult GetStremNumOfflineByDev(
    const DevType& devType, HcclCMDType hcclOpType, s32 serverNum, s32 deviceNumPerServer, bool ifAiv, u64& streamNum,
    const std::string& group = HCCL_WORLD_GROUP);
HcclResult GetSubStreamNum(
    const DevType& devType, s32 deviceNum, u64& streamNum, const s32& serverNum,
    const std::string& group = HCCL_WORLD_GROUP);
HcclResult GetOffDeviceTypeWithoutDev(std::string socVersionStr, DevType& devType);
HcclResult GetServerAndDevNumFromGroupList(
    const u32* groupList, u32 groupListSize, const std::string rankTableString, DevType devType, s32& serverNum,
    s32& deviceNumPerServer, bool& multiModuleDiffDeviceNumMode);
HcclResult GetServerAndDevNumFromLogRanktable(
    const std::string rankTableString, const u32* groupList, u32 groupListSize, DevType devType, s32& serverNum,
    s32& deviceNum, bool& multiModuleDiffDeviceNumMode);
HcclResult GetOpWorkspaceMemSize(
    bool isOfflineCompilation, HcclCMDType hcclOpType, HcomOpParam* hcomOpParam, s32 serverNum, u64& opMemSize);
HcclResult GetOpScratchMemSize(
    bool isOfflineCompilation, HcclCMDType hcclOpType, HcomOpParam* hcomOpParam, u64& opMemSize, u32 dataTypeSize,
    s32 rankSize, s32 serverNum);
HcclResult GetAlltoAllvStagedScratchMemSize(HcomOpParam* hcomOpParam, u32 rankSize, u64& getMemSize);
HcclResult GetAlltoAllvcStagedScratchMemSize(HcomOpParam* hcomOpParam, u32 rankSize, u64& getMemSize);
HcclResult GetReduceScatterVScratchMemSize(HcomOpParam* hcomOpParam, u64& getMemSize);
HcclResult GetAllReduceScratchMemSize(
    bool isOfflineCompilation, HcomOpParam* hcomOpParam, s32 serverNum, s32 rankSize, u64& getMemSize);
HcclResult GetAllReduceScratchSizeWithoutDev(HcomOpParam* hcomOpParam, s32 serverNum, s32 rankSize, u64& scratchSize);
bool IsNeedCalTaskNum(HcclCMDType opType);
HcclResult GetDefaultAlgoLevel1(s32 serverNum, AlgTypeLevel1& algType);
HcclResult GetAlgoLevel1(s32 serverNum, std::string& opType, AlgTypeLevel1& algType);
HcclResult
SplitHcclOpTypeConfig(const std::string& algoConfig, const std::string& opType, std::string& specificAlgoConfig);
HcclResult GetDefaultAlgoLevel0Module(s32 deviceNumPerServer, AlgTypeLevel0& algType, std::string soc_version);
HcclResult
GetAlgType(s32 deviceNumPerServer, s32 serverNum, std::string opType, std::string socVersionStr, AlgType& algType);
HcclResult GetDfxTaskNum(const std::string& sCollectiveType, u32& taskNum);
HcclResult
GetToSlaveStreamTaskNum(const std::string& sCollectiveType, u64 streamNum, u64 pipelineSliceNum, u32& taskNum);
HcclResult GetToMasterStreamTaskNum(const std::string& sCollectiveType, u32& taskNum);
HcclResult GetCombineComTaskNum(
    const std::string& sCollectiveType, s32 serverNum, s32 deviceNumPerServer, u32& intraTaskNum, u32& interTaskNum);
HcclResult GetIntraComTaskNum(
    const std::string& sCollectiveType, s32 deviceNumPerServer, u64 streamNum, const AlgType& algType, u32& taskNum,
    u64 totalSize);
HcclResult GetBetweenServersStep(s32 serverNum, u32& commStep);
HcclResult
GetClusterInfoAndDeviceNum(const std::string rankTableString, hccl::RankTable_t& clusterInfo, s32& deviceNum);
HcclResult GetServerAndDevNumFromRanklist(
    const u32* groupList, u32 groupListSize, const std::vector<hccl::RankInfo_t>& rankList, DevType devType,
    s32& serverNum, s32& deviceNum, bool& multiModuleDiffDeviceNumMode);
HcclResult GetServerIdByRankId(const std::vector<hccl::RankInfo_t>& rankList, const u32& rankId, u32& serverId);
HcclResult
GetModuleInfo(DevType devType, const std::vector<hccl::RankInfo_t>& rankList, bool& multiModuleDiffDeviceNumMode);
HcclResult GetDeterministic(DevType devType, u8 geDetOption, u8& deterministic);
// end

// hcom.h内使用的非包间接口
HcclResult HcomGetLocalRankSize(const char* group, u32* localRankSize);
HcclResult HcomGetLocalRankId(const char* group, u32* localRankId);
HcclResult HcomGetWorldRankFromGroupRank(const char* group, u32 groupRank, u32* worldRank);
HcclResult HcomGetGroupRankFromWorldRank(u32 worldRank, const char* group, u32* groupRank);
extern HcclResult HcomSetGradFusionByIndex(const char* group, u32 segmentNum, const u32* IdxList);
extern HcclResult HcomSetGradFusionBySize(const char* group, u32 segmentNum, const float* sizeList);
HcclResult HcomGetandClearOverFlowTasks(const char* group, hccl::HcclDumpInfo** hcclDumpInfoPtr, s32* len);
// end

HcclResult GenerateCclOpTag(const std::string& opType, const int64_t& hcomComm, std::string& group, std::string& sTag);

HcclResult HcomExecSelectAlg(
    s64 comm, const char* group, HcclCMDType opType, u64 count, HcclDataType dataType, HcclReduceOp op,
    int32_t aivCoreLimit, bool& ifAiv, char* algName);

#ifdef __cplusplus
}
#endif
#endif // H_COM_PUB_H
