/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_BASE_V2_H
#define OP_BASE_V2_H
#include <vector>
#include <memory>
#include <functional>
#include <hccl/hccl_types.h>
#include "comm_manager.h"
#include "binary_stream.h"
#include "snap_shot_parse.h"
#include "param_check_v2.h"
#include "task_param.h"
#include "hostdpu/task_service.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus
HcclCommInfoV2& GetOpbasedCommInfoV2(void);

HcclResult HcclCommDestroyV2(HcclComm comm);

HcclResult HcclCommInitClusterInfoV2(const char* clusterInfo, uint32_t rank, HcclComm* comm);

HcclResult
HcclCommInitClusterInfoConfigV2(const char* clusterInfo, uint32_t rank, HcclCommConfig* config, HcclComm* comm);

HcclResult HcclCommInitAllV2(uint32_t ndev, int32_t* devices, HcclComm* comms);

HcclResult HcclAlltoAllV2(
    const void* sendBuf, uint64_t sendCount, HcclDataType sendType, const void* recvBuf, uint64_t recvCount,
    HcclDataType recvType, HcclComm comm, aclrtStream stream);
HcclResult HcclAlltoAllVV2(
    const void* sendBuf, const void* sendCounts, const void* sdispls, HcclDataType sendType, const void* recvBuf,
    const void* recvCounts, const void* rdispls, HcclDataType recvType, HcclComm comm, aclrtStream stream);

HcclResult HcclCreateSubCommConfigV2(
    const HcclComm* comm, uint32_t rankNum, uint32_t* rankIds, uint64_t subCommId, uint32_t subCommRankId,
    HcclCommConfig* config, HcclComm* subComm);

HcclResult
HcclCommInitClusterInfoMemConfigV2(const char* rankTableString, uint32_t rank, HcclCommConfig* config, HcclComm* comm);

HcclResult HcclGetRankIdV2(HcclComm comm, uint32_t* rank);

HcclResult HcclGetRootInfoV2(HcclRootInfo* rootInfo);

HcclResult HcclGetCommNameV2(HcclComm commHandle, char* commName);

HcclResult HcclCommInitRootInfoV2(
    uint32_t nRanks, const HcclRootInfo* rootInfo, uint32_t rank, HcclComm* comm, std::string& identifier);

HcclResult HcclCommInitRootInfoConfigV2(
    uint32_t nRanks, const HcclRootInfo* rootInfo, uint32_t rank, const HcclCommConfig* config, HcclComm* comm);

HcclResult HcclGetRankSize(HcclComm comm, uint32_t* rankSize);

HcclResult HcclAlltoAllVCV2(
    const void* sendBuf, const void* sendCountMatrix, HcclDataType sendType, const void* recvBuf, HcclDataType recvType,
    HcclComm comm, rtStream_t stream);

HcclResult HcclReduceV2(
    void* sendBuf, void* recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op, uint32_t root, HcclComm comm,
    aclrtStream stream);

HcclResult HcclAllReduceV2(
    void* sendBuf, void* recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op, HcclComm comm,
    aclrtStream stream);

HcclResult
HcclBroadcastV2(void* buf, uint64_t count, HcclDataType dataType, uint32_t root, HcclComm comm, aclrtStream stream);

HcclResult HcclGetTopoDescV2();

HcclResult HcclScatterV2(
    void* sendBuf, void* recvBuf, uint64_t recvCount, HcclDataType dataType, uint32_t root, HcclComm comm,
    aclrtStream stream);

HcclResult HcclCommSuspend(HcclComm comm);

HcclResult HcclReduceScatterV2(
    void* sendBuf, void* recvBuf, uint64_t recvCount, HcclDataType dataType, HcclReduceOp op, HcclComm comm,
    aclrtStream stream);

HcclResult HcclReduceScatterVV2(
    void* sendBuf, void* sendCounts, void* sendDispls, void* recvBuf, uint64_t recvCount, HcclDataType dataType,
    HcclReduceOp op, HcclComm comm, aclrtStream stream);

HcclResult HcclAllGatherV2(
    void* sendBuf, void* recvBuf, uint64_t sendCount, HcclDataType dataType, HcclComm comm, aclrtStream stream);

HcclResult HcclAllGatherVV2(
    void* sendBuf, uint64_t sendCount, void* recvBuf, void* recvCounts, void* recvDispls, HcclDataType dataType,
    HcclComm comm, aclrtStream stream);

HcclResult
HcclSendV2(void* sendBuf, uint64_t count, HcclDataType dataType, uint32_t destRank, HcclComm comm, aclrtStream stream);

HcclResult
HcclRecvV2(void* recvBuf, uint64_t count, HcclDataType dataType, uint32_t srcRank, HcclComm comm, aclrtStream stream);

HcclResult HcclBatchSendRecvV2(HcclSendRecvItem* sendRecvInfo, uint32_t itemNum, HcclComm comm, aclrtStream stream);

HcclResult HcclGetRankSizeV2(HcclComm comm, uint32_t* rankSize);

HcclResult HcclAllocComResourceByTilingV2(HcclComm comm, const void* stream, void* mc2Tiling, void** commContext);

HcclResult HcclGetOpArgsV2(void** opArgs);

HcclResult HcclFreeOpArgsV2(void* opArgs);

HcclResult HcclSetOpSrcDataTypeV2(void* opArgs, uint8_t srcDataType);

HcclResult HcclSetOpDstDataTypeV2(void* opArgs, uint8_t dstDataType);

HcclResult HcclSetOpReduceTypeV2(void* opArgs, uint32_t reduceType);

HcclResult HcclSetOpCountV2(void* opArgs, uint64_t count);

HcclResult HcclSetOpAlgConfigV2(void* opArgs, char* algConfig);

HcclResult HcclSetOpCommEngineV2(void* opArgs, uint8_t commEngine);

HcclResult HcclCommResPrepareV2(HcclComm comm, char* opName, void* opArgs, void** addr);

HcclResult HcclDevMemAcquireV2(HcclComm comm, const char* memTag, uint64_t* size, void** addr, bool* newCreated);

HcclResult HcclGetHcclBufferV2(HcclComm comm, void** addr, uint64_t* size);

HcclResult HcclGetRemoteIpcHcclBufV2(HcclComm comm, uint64_t remoteRank, void** addr, uint64_t* size);

HcclResult HcclGetAicpuOpStreamAndNotifyV2(HcclComm comm, rtStream_t* opstream, u8 aicpuNotifyNum, void** aicpuNotify);

HcclResult HcclCommSuspendV2(HcclComm comm);

HcclResult HcclCommResumeV2(HcclComm comm);

HcclResult HcclCommResumeImplV2(HcclComm comm);

HcclResult HcclGetRawCommHandle(const char* commName, HcclComm* commHandle);
HcclResult HcclGetCcuTaskInfoLegacy(HcclComm comm, void* tilingData, void* ccuTaskGroup);
HcclResult HcclSnapshotSave(void* snapshotBuf, uint32_t size, uint32_t step);
void RecoverSnapshotCcuStatus(const std::shared_ptr<Hccl::SnapShotBuf>& savedSnapshotBuf);
HcclResult HcclSnapshotRecoverAllComms(
    const char* clusterInfo, const char* changedInfo, void* snapshotBuf, uint32_t snapshotBufSize);
void GetSnapShotCcuStatusBuf(Hccl::BinaryStream& buf);
HcclResult SnapshotGenerate(
    const std::shared_ptr<Hccl::HcclCommunicator>& pComm,
    const std::map<std::string, std::shared_ptr<Hccl::HcclCommunicator>>& hcclGroupMap, uint32_t step, uint32_t* size);
HcclResult HcclSnapshotGetBufSize(uint32_t step, uint32_t* size);
HcclResult HcclGetCommAsyncErrorV2();

HcclResult HcclSetConfigV2(HcclConfig config, HcclConfigValue configValue);
HcclResult HcclGetConfigV2(HcclConfig config, HcclConfigValue* configValue);

HcclResult HcclGetRankGraphV2(const HcclComm* comm, void** rankGraph);

HcclResult HcclBarrierV2(HcclComm comm, aclrtStream stream);
HcclResult HcclGetHeterogModeV2(HcclComm comm, HcclHeterogMode* mode);
HcclResult WaitAllCommReady(s32 deviceLogicId);

HcclResult CommGetCCLBufSizeCfgV2(HcclComm comm, uint64_t* cclBufSize);

HcclResult HcclGetNetLayersV2(HcclComm comm, uint32_t** netLayers, uint32_t* netLayerNum);
HcclResult HcclGetInstSizeByNetLayerV2(HcclComm comm, uint32_t netLayer, uint32_t* rankNum);
HcclResult HcclGetInstTopoTypeByNetLayerV2(HcclComm comm, uint32_t netLayer, uint32_t* topoType);
HcclResult HcclGetInstRanksByNetLayerV2(HcclComm comm, uint32_t netLayer, uint32_t** ranks, uint32_t* rankNum);
HcclResult HcclGetNetTypeByLayerV2(HcclComm comm, uint32_t netLayer, CommTopo* netType);
HcclResult
HcclGetInstSizeListByNetLayerV2(HcclComm comm, uint32_t netLayer, uint32_t** instSizeList, uint32_t* listSize);
HcclResult HcclGetLinksV2(
    HcclComm comm, uint32_t netLayer, uint32_t srcRank, uint32_t dstRank, CommLink** linkList, uint32_t* listSize);
HcclResult HcclGetTopoInstsByLayerV2(HcclComm comm, uint32_t netLayer, uint32_t** topoInsts, uint32_t* topoInstNum);
HcclResult HcclGetTopoTypeV2(HcclComm comm, uint32_t netLayer, uint32_t topoInstId, CommTopo* topoType);
HcclResult
HcclGetRanksByTopoInstV2(HcclComm comm, uint32_t netLayer, uint32_t topoInstId, uint32_t** ranks, uint32_t* rankNum);
HcclResult HcommFlushV2();
HcclResult
HcclGetCclBuffer(HcclComm comm, uintptr_t& cclBufferAddr, size_t& cclBufferSize, HcclMemType& cclBufferMemType);

HcclResult HcclRankGraphGetEndpointNumV2(HcclComm comm, uint32_t layer, uint32_t topoInstId, uint32_t* num);

HcclResult HcclRankGraphGetEndpointDescV2(
    HcclComm comm, uint32_t layer, uint32_t topoInstId, uint32_t* descNum, EndpointDesc* endpointDesc);

HcclResult HcclRankGraphGetEndpointInfoV2(
    HcclComm comm, uint32_t rankId, const EndpointDesc* endpointDesc, EndpointAttr endpointAttr, uint32_t infoLen,
    void* info);
typedef int32_t(Callback)(uint64_t, int32_t);
HcclResult HcclTaskRegisterV2(HcclComm comm, const char* msgTag, Callback cb);
HcclResult
HcclTaskRegisterProfV2(HcclComm comm, std::function<HcclResult(const Hccl::TaskParam&, uint64_t)> profCallback);
HcclResult HcclTaskReportRegisterV2(HcclComm comm, std::function<HcclResult()> reportCallback);
HcclResult HcclGetDpuSteamIdV2(HcclComm comm, u32& dpuStreamId);
HcclResult HcclTaskUnRegisterV2(HcclComm comm, const char* msgTag);
HcclResult HcclCheckTaskServiceExist(const std::string& commId, s32 deviceId);

HcclResult HcclCommWorkingDevNicSetV2(const HcclComm comm, uint32_t* ranks, bool* useBackup, uint32_t nRanks);

HcclResult
HcclCommSetMemoryRangeV2(const HcclComm comm, void* baseVirPtr, size_t size, size_t alignment, uint64_t flags);

HcclResult HcclCommUnsetMemoryRangeV2(const HcclComm comm, void* baseVirPtr);

HcclResult HcclCommActivateCommMemoryV2(
    const HcclComm comm, void* virPtr, size_t size, size_t offset, void* handle, uint64_t flags);

HcclResult HcclCommDeactivateCommMemoryV2(const HcclComm comm, void* virPtr);

uint32_t HcclGetCommConfigCapabilityV2();

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // OP_BASE_V2_H
