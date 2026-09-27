/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_BASE_H
#define OP_BASE_H

#include <functional>
#include <string>
#include <vector>
#include <hccl/hccl_comm.h>
#include <hccl/hccl_inner.h>
#include <hccl/hccl_types.h>

#include "hccl/base.h"
#include "hccl/hccl_rank_graph.h"
#include "hccl_mem_defs.h"
#include "hcomm_res_defs.h"
#include "acl/acl_rt.h"
#include "task_param.h"
#include "op_base_pub.h"
#include "hccl_comm_pub.h"
// ltm指定config路径
#include "common/src/config.h"
#include "../common/src/topo/topoinfo_detect.h"
#include "legacy_op_hcom_info.h"

namespace Hccl {
using ProfCallback = std::function<HcclResult(const TaskParam&, uint64_t)>;
using ReportCallback = std::function<HcclResult()>;
} // namespace Hccl

#ifdef __cplusplus
extern "C" {
#endif

HcclResult HcclCommDestroyV2(HcclComm comm);

HcclResult __attribute__((weak)) HcclCommInitClusterInfoV2(const char* clusterInfo, uint32_t rank, HcclComm* comm);

HcclResult __attribute__((weak))
HcclCommInitClusterInfoConfigV2(const char* clusterInfo, uint32_t rank, HcclCommConfig* config, HcclComm* comm);

HcclResult __attribute__((weak)) HcclCommInitAllV2(uint32_t ndev, int32_t* devices, HcclComm* comms);

HcclResult __attribute__((weak))
HcclCommInitClusterInfoMemConfigV2(const char* rankTableString, uint32_t rank, HcclCommConfig* config, HcclComm* comm);

HcclResult __attribute__((weak)) HcclAlltoAllV2(
    const void* sendBuf, uint64_t sendCount, HcclDataType sendType, const void* recvBuf, uint64_t recvCount,
    HcclDataType recvType, HcclComm comm, aclrtStream stream);

HcclResult __attribute__((weak)) HcclAlltoAllVV2(
    const void* sendBuf, const void* sendCounts, const void* sdispls, HcclDataType sendType, const void* recvBuf,
    const void* recvCounts, const void* rdispls, HcclDataType recvType, HcclComm comm, aclrtStream stream);

HcclResult __attribute__((weak)) HcclCreateSubCommConfigV2(
    HcclComm* comm, uint32_t rankNum, uint32_t* rankIds, uint64_t subCommId, uint32_t subCommRankId,
    HcclCommConfig* config, HcclComm* subComm);

HcclResult __attribute__((weak)) HcclGetRankIdV2(HcclComm comm, uint32_t* rank);

HcclResult __attribute__((weak)) HcclGetRootInfoV2(HcclRootInfo* rootInfo);

HcclResult __attribute__((weak)) HcclGetCommNameV2(HcclComm commHandle, char* commName);

HcclResult __attribute__((weak)) HcclCommInitRootInfoV2(
    uint32_t nRanks, const HcclRootInfo* rootInfo, uint32_t rank, HcclComm* comm, std::string& identifier);

HcclResult __attribute__((weak)) HcclCommInitRootInfoConfigV2(
    uint32_t nRanks, const HcclRootInfo* rootInfo, uint32_t rank, const HcclCommConfig* config, HcclComm* comm);

HcclResult __attribute__((weak)) HcclGetRankSize(HcclComm comm, uint32_t* rankSize);

HcclResult __attribute__((weak)) HcclAlltoAllVCV2(
    const void* sendBuf, const void* sendCountMatrix, HcclDataType sendType, const void* recvBuf, HcclDataType recvType,
    HcclComm comm, rtStream_t stream);

HcclResult __attribute__((weak)) HcclReduceV2(
    void* sendBuf, void* recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op, uint32_t root, HcclComm comm,
    aclrtStream stream);

HcclResult __attribute__((weak)) HcclAllReduceV2(
    void* sendBuf, void* recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op, HcclComm comm,
    aclrtStream stream);

HcclResult __attribute__((weak))
HcclBroadcastV2(void* buf, uint64_t count, HcclDataType dataType, uint32_t root, HcclComm comm, aclrtStream stream);

HcclResult __attribute__((weak)) HcclGetTopoDescV2();

HcclResult __attribute__((weak)) HcclScatterV2(
    void* sendBuf, void* recvBuf, uint64_t recvCount, HcclDataType dataType, uint32_t root, HcclComm comm,
    aclrtStream stream);

HcclResult __attribute__((weak)) HcclCommSuspend(HcclComm comm);

HcclResult __attribute__((weak)) HcclReduceScatterV2(
    void* sendBuf, void* recvBuf, uint64_t recvCount, HcclDataType dataType, HcclReduceOp op, HcclComm comm,
    aclrtStream stream);

HcclResult __attribute__((weak)) HcclReduceScatterVV2(
    void* sendBuf, void* sendCounts, void* sendDispls, void* recvBuf, uint64_t recvCount, HcclDataType dataType,
    HcclReduceOp op, HcclComm comm, aclrtStream stream);

HcclResult __attribute__((weak)) HcclAllGatherV2(
    void* sendBuf, void* recvBuf, uint64_t sendCount, HcclDataType dataType, HcclComm comm, aclrtStream stream);

HcclResult __attribute__((weak)) HcclAllGatherVV2(
    void* sendBuf, uint64_t sendCount, void* recvBuf, void* recvCounts, void* recvDispls, HcclDataType dataType,
    HcclComm comm, aclrtStream stream);

HcclResult __attribute__((weak))
HcclSendV2(void* sendBuf, uint64_t count, HcclDataType dataType, uint32_t destRank, HcclComm comm, aclrtStream stream);

HcclResult __attribute__((weak))
HcclRecvV2(void* recvBuf, uint64_t count, HcclDataType dataType, uint32_t srcRank, HcclComm comm, aclrtStream stream);

HcclResult __attribute__((weak))
HcclBatchSendRecvV2(HcclSendRecvItem* sendRecvInfo, uint32_t itemNum, HcclComm comm, aclrtStream stream);

HcclResult __attribute__((weak)) HcclGetRankSizeV2(HcclComm comm, uint32_t* rankSize);

HcclResult __attribute__((weak))
HcclAllocComResourceByTilingV2(HcclComm comm, const void* stream, void* mc2Tiling, void** commContext);

HcclResult __attribute__((weak)) HcclCommSuspendV2(HcclComm comm);

HcclResult __attribute__((weak)) HcclCommResumeV2(HcclComm comm);

HcclResult __attribute__((weak)) HcclCommResumeImplV2(HcclComm comm);

HcclResult __attribute__((weak)) HcclGetCommAsyncErrorV2();

HcclResult __attribute__((weak)) HcclGetRawCommHandle(const char* commName, HcclComm* commHandle);

HcclResult __attribute__((weak)) HcclSetConfigV2(HcclConfig config, HcclConfigValue configValue);

HcclResult __attribute__((weak)) HcclGetConfigV2(HcclConfig config, HcclConfigValue* configValue);

HcclResult __attribute__((weak)) HcclBarrierV2(HcclComm comm, aclrtStream stream);

HcclResult __attribute__((weak)) HcclGetHeterogModeV2(HcclComm comm, HcclHeterogMode* mode);

HcclResult __attribute__((weak)) HcclGetRankGraphV2(const HcclComm* comm, void** rankGraph);

HcclResult __attribute__((weak))
HcclGetCclBuffer(HcclComm comm, uintptr_t& cclBufferAddr, size_t& cclBufferSize, HcclMemType& cclBufferMemType);

HcclResult __attribute__((weak))
HcclCommWorkingDevNicSetV2(const HcclComm comm, uint32_t* ranks, bool* useBackup, uint32_t nRanks);

HcclResult __attribute__((weak))
HcclCommSetMemoryRangeV2(const HcclComm comm, void* baseVirPtr, size_t size, size_t alignment, uint64_t flags);

HcclResult __attribute__((weak)) HcclCommUnsetMemoryRangeV2(const HcclComm comm, void* baseVirPtr);

HcclResult __attribute__((weak)) HcclCommActivateCommMemoryV2(
    const HcclComm comm, void* virPtr, size_t size, size_t offset, void* handle, uint64_t flags);

HcclResult __attribute__((weak)) HcclCommDeactivateCommMemoryV2(const HcclComm comm, void* virPtr);

HcclResult __attribute__((weak)) HcommFlushV2();

uint32_t __attribute__((weak)) HcclGetCommConfigCapabilityV2();

#if (!defined(HCCD)) && (!defined(CCL_KERNEL_AICPU))
HcclResult __attribute__((weak)) HcclGetCcuTaskInfoLegacy(HcclComm comm, void* tilingData, void* ccuTaskGroup);

HcclResult __attribute__((weak)) HcclGetNetLayersV2(HcclComm comm, uint32_t** netLayers, uint32_t* netLayerNum);

HcclResult __attribute__((weak)) HcclGetInstSizeByNetLayerV2(HcclComm comm, uint32_t netLayer, uint32_t* rankNum);

HcclResult __attribute__((weak)) HcclGetInstTopoTypeByNetLayerV2(HcclComm comm, uint32_t netLayer, uint32_t* topoType);

HcclResult __attribute__((weak)) CommGetCCLBufSizeCfgV2(HcclComm comm, uint64_t* cclBufSize);

HcclResult __attribute__((weak))
HcclGetInstRanksByNetLayerV2(HcclComm comm, uint32_t netLayer, uint32_t** ranks, uint32_t* rankNum);

HcclResult __attribute__((weak))
HcclGetInstSizeListByNetLayerV2(HcclComm comm, uint32_t netLayer, uint32_t** instSizeList, uint32_t* listSize);

HcclResult __attribute__((weak)) HcclGetLinksV2(
    HcclComm comm, uint32_t netLayer, uint32_t srcRank, uint32_t dstRank, CommLink** linkList, uint32_t* listSize);

HcclResult __attribute__((weak))
HcclGetTopoInstsByLayerV2(HcclComm comm, uint32_t netLayer, uint32_t** topoInsts, uint32_t* topoInstNum);

HcclResult __attribute__((weak))
HcclGetTopoTypeV2(HcclComm comm, uint32_t netLayer, uint32_t topoInstId, CommTopo* topoType);

HcclResult __attribute__((weak))
HcclGetRanksByTopoInstV2(HcclComm comm, uint32_t netLayer, uint32_t topoInstId, uint32_t** ranks, uint32_t* rankNum);

HcclResult __attribute__((weak))
HcclRankGraphGetEndpointNumV2(HcclComm comm, uint32_t layer, uint32_t topoInstId, uint32_t* num);

HcclResult __attribute__((weak)) HcclRankGraphGetEndpointDescV2(
    HcclComm comm, uint32_t layer, uint32_t topoInstId, uint32_t* descNum, EndpointDesc* endpointDesc);

HcclResult __attribute__((weak)) HcclRankGraphGetEndpointInfoV2(
    HcclComm comm, uint32_t rankId, const EndpointDesc* endpointDesc, EndpointAttr endpointAttr, uint32_t infoLen,
    void* info);

HcclResult __attribute__((weak)) HcclGetOpArgsV2(void** opArgs);

HcclResult __attribute__((weak)) HcclFreeOpArgsV2(void* opArgs);

HcclResult __attribute__((weak)) HcclSetOpSrcDataTypeV2(void* opArgs, uint8_t srcDataType);

HcclResult __attribute__((weak)) HcclSetOpDstDataTypeV2(void* opArgs, uint8_t dstDataType);

HcclResult __attribute__((weak)) HcclSetOpReduceTypeV2(void* opArgs, uint32_t reduceType);

HcclResult __attribute__((weak)) HcclSetOpCountV2(void* opArgs, uint64_t count);

HcclResult __attribute__((weak)) HcclSetOpAlgConfigV2(void* opArgs, char* algConfig);

HcclResult __attribute__((weak)) HcclSetOpCommEngineV2(void* opArgs, uint8_t commEngine);

HcclResult __attribute__((weak)) HcclCommResPrepareV2(HcclComm comm, char* opName, void* opArgs, void** addr);

HcclResult __attribute__((weak))
HcclDevMemAcquireV2(HcclComm comm, const char* memTag, uint64_t* size, void** addr, bool* newCreated);

HcclResult __attribute__((weak)) HcclGetHcclBufferV2(HcclComm comm, void** addr, uint64_t* size);

HcclResult __attribute__((weak))
HcclGetRemoteIpcHcclBufV2(HcclComm comm, uint64_t remoteRank, void** addr, uint64_t* size);

HcclResult __attribute__((weak))
HcclGetAicpuOpStreamAndNotifyV2(HcclComm comm, rtStream_t* opstream, u8 aicpuNotifyNum, void** aicpuNotify);

typedef int32_t(Callback)(uint64_t, int32_t);
HcclResult __attribute__((weak)) HcclTaskRegisterV2(HcclComm comm, const char* msgTag, Callback cb);
HcclResult __attribute__((weak)) HcclTaskUnRegisterV2(HcclComm comm, const char* msgTag);
HcclResult __attribute__((weak)) HcclTaskRegisterProfV2(HcclComm comm, Hccl::ProfCallback profCallback);
HcclResult __attribute__((weak)) HcclTaskReportRegisterV2(HcclComm comm, Hccl::ReportCallback reportCallback);
HcclResult __attribute__((weak)) HcclGetDpuSteamIdV2(HcclComm comm, u32& dpuStreamId);
HcclResult __attribute__((weak)) HcclCheckTaskServiceExist(const std::string& commId, s32 deviceId);
#endif

#ifdef __cplusplus
}
#endif

constexpr uint32_t MAX_HCOM_NUM = 3U;

HcclResult InitOtherInfo(hccl::HcclCommParams& params, const char* rankTable);

HcclResult CallMsprofReportHostApi(
    hccl::hcclComm* hcclComm, HcclCMDType cmdType, uint64_t beginTime, u64 count, HcclDataType dataType,
    const std::string& tag);

HcclResult ReduceScatterLoop(
    const std::string& tag, void* inputPtr, void* outputPtr, const u64& count, HcclDataType dataType, HcclReduceOp op,
    hccl::hcclComm* hcclComm, rtStream_t stream);

HcclResult HcclGetOpBasedMemSize(const HcclCMDType& opType, u64& size, const HcomCollOpInfo& opInfo);

HcclResult ReduceLoop(
    const std::string& tag, void* inputPtr, void* outputPtr, const u64 count, HcclDataType dataType, HcclReduceOp op,
    const u32 root, hccl::hcclComm* hcclComm, rtStream_t stream);

HcclResult HcclGatherAlltoAllV(HcomGatherAllToAllVParams params, HcclComm comm, aclrtStream stream);

HcclResult RunGather(u64* sendCounts, u64* sdispls, void* sendDevBuf, GatherPara& gatherPara);

void GatherMemCopyThread(void* baseAddr, u64 offset, std::vector<u64>& addrInfo, OpBaseMemPara memCpyPara);

HcclResult HcclGetCommAll(uint32_t ndev, int32_t* devices, HcclComm* comms);

HcclResult
GetDeviceComm(uint32_t ndev, const HcclRootInfo& rootHandle, const s32 rank, const s32 logicDeviceId, HcclComm& comm);

HcclResult SetOverFlowAddr(hccl::hcclComm* hcclComm);

HcclResult HcclGetCommHandle(const char* commName, std::shared_ptr<hccl::hcclComm>& comm);

HcclResult CheckScatterInputPara(const HcclComm comm, const void* recvBuf);

HcclResult HcclMc2ComResourceByTiling(HcclComm comm, uint32_t* pVersion, void* mc2Tiling, rtStream_t& aicpuStream);

HcclResult HcclCreateComResourceByComm(
    HcclComm comm, u32 streamMode, bool isOpbaseMode, void** commContext, bool isMC2 = false,
    void* mc2Tiling = nullptr);

HcclResult HcclDeviceRefresh(s32& deviceLogicId);

HcclResult HcclBatchSendRecvGroup(HcclSendRecvItem* sendRecvInfo, uint32_t itemNum, HcclComm comm, aclrtStream stream);

HcclResult HcclSetIfProfile(void);

void HcclResetIfProfile(void);

void PrintCountsAndDispls(const u32 length, const void* counts, const void* displs, const std::string& tag);

void CheckCountsAndDispls(const u32 length, const void* counts, const void* displs, const std::string& tag);

HcclResult GetCaptureInfo(aclrtStream stream, aclmdlRICaptureStatus& captureStatus, uint64_t& modelId, bool& isCapture);

HcclResult HcclGetInitTilingList(const void* mc2Tiling, const void* p[], uint32_t& cnt);

HcclResult HcclCommInitCollComm(uint32_t rank, void** commV2, const HcclCommConfig* config, HcclComm* comm);
HcclResult HcclMc2ComOpResCtx(
    HcclComm comm, uint8_t opType, HcclDataType srcDataType, HcclDataType dstDataType, HcclReduceOp reduceType,
    uint64_t count, char* algConfig, uint32_t commEngine, rtStream_t& aicpuStream);

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

HcclResult
HcclCommInitClusterInfoMemConfig(const char* rankTableString, uint32_t rank, HcclCommConfig* config, HcclComm* comm);

#if (!defined(HCCD)) && (!defined(CCL_KERNEL_AICPU))
HcclResult HcclCommStateNotify(HcclComm comm, HcclCommStatePhase state);

static inline HcclResult HcclCommResumePostCallback(HcclComm comm)
{
    return HcclCommStateNotify(comm, HcclCommStatePhase::HCCL_COMM_STATE_PHASE_RESUME_POST);
}
#endif
#ifdef __cplusplus
}
#endif // __cplusplus
#endif // OP_BASE_H
