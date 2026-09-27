/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_COMMUNICATOR_PUB_H
#define HCCLV2_COMMUNICATOR_PUB_H

#include <memory>
#include <mutex>
#include "hccl_params_pub.h"
#include "hccl_types.h"
#include "comm_type.h"
#include "rank_table_info.h"
#include "hccl_rank_graph.h"
#include "hccl_mem_defs.h"
#include "trace.h"
#include "socket_manager.h"

namespace Hccl {
class CommunicatorImpl;
class HcclOneSidedService;
class DevBuffer;
class HcclCommunicator {
public:
    explicit HcclCommunicator(const CommParams& commParams);
    HcclCommunicator(const CommParams& commParams, const HcclCommConfig* config);
    ~HcclCommunicator();

    HcclResult Init(const std::string& ranktableM);
    HcclResult Init(const RankTableInfo& ranktable);

    HcclResult CreateSubComm(
        const CommParams& subCommParams, const std::vector<u32>& rankIds,
        std::shared_ptr<HcclCommunicator>& subHcclComm);
    HcclResult CreateSubComm(
        const CommParams& subCommParams, const std::vector<u32>& rankIds,
        std::shared_ptr<HcclCommunicator>& subHcclComm, HcclCommConfig& subConfig);
    void DeInit() const;

    HcclResult LoadOpbasedCollOp(const CollOpParams& opParams, void* stream);
    HcclResult AllocCollOpResource(const CollOpParams& opParams, void** addr);

    const std::string& GetId() const;
    HcclResult
    CalcCollOffloadOpRes(const OpType opType, u64 dataSize, HcclDataType dataType, CollOffloadOpResReq& resReq);
    HcclResult SetCollOffloadSlaveStreams(const std::string& opTag, std::vector<void*> slaveStreams);
    HcclResult SetCollOffloadScratchBuf(const std::string& opTag, void* scratchMemPtr, u64 requiredScratchMemSize);
    HcclResult LoadOffloadCollOp(std::string& opTag, const CollOpParams& opParams, void* stream);
    HcclResult Suspend();
    HcclResult Clean();
    HcclResult Resume();

    void RegistTaskAbortHandler();
    void UnRegistTaskAbortHandler();

    // MC2 流程专用
    HcclResult GetRankSize(uint32_t* rankSize);

    HcclResult GetRankId(uint32_t& rankId);

    HcclResult AllocCommResource(void* mc2Tiling, void** commContext);

    HcclResult GetCcuTaskInfo(void* tilingData, void* ccuTaskGroup);

    u32 GetCcuMc2ServerNum();
    /*
     * @brief MC2获取topoDes, 包含各拓扑层级的可选算法信息和rank数量等
     */
    HcclResult GetTopoDesc(HcclTopoDescs* topoDescs, uint32_t topoSize);

    bool IsWorldGroup() const;

    bool IsCommReady();

    HcclResult RecoverComm(void* snapShotComm, u32 step, const char* changeInfo);

    HcclResult RecoverSubComm(const void* snapShotSubComm, std::shared_ptr<HcclCommunicator>& subComm, u32 step);

    // 获取建链邻居信息
    HcclResult GetSnapShotDynamicBuf(void* buf);
    void* GetStaticBinaryInfo();

    HcclResult GetOneSidedService(HcclOneSidedService** oneSidedService);

    // 获取Channel信息，ccu专用
    u32 GetUsedChannelCount(u32 dieId);
    void RegisterPrintChannelInfoCallback(std::function<void()> callback);
    CommStatus GetCommStatus() const;
    // 设置加速模式
    HcclResult SetAccelerator(HcclAccelerator hcclAccelerator, bool isCcuMsAvailable);
    HcclResult SetAccelerator(int32_t accelerator, bool isCcuMsAvailable);
    HcclResult GetAccelerator(int32_t* accelerator) const;
    bool IsUsingCcuMs() const;
    bool IsUsingCcuSched() const;
    void RegisterAcceStateCallBack(
        std::function<HcclResult(const std::string& commId, bool isUsingCcuMs, bool isUsingCcuSched)> callback);
    HcclResult CalcTaskNum(OpType opType, DataType dataType, u64 count, u32& taskNum);
    HcclResult CreateCommCclBuf();
    HcclResult GetLocalCclBuffer(void** addr, uint64_t* size);
    HcclResult GetDevMemWorkSpace(const std::string& memTag, uint64_t* size, void** addr, bool* newCreated);
    HcclResult GetInCclBuf(void*& commInputPtr, u64& commInputSize);
    HcclResult GetOutCclBuf(void*& commOutputPtr, u64& commOutputSize);
    HcclResult GetIndirectInputCclBuf(void*& commIndirectInputPtr, u64& commIndirectInputSize);
    HcclResult GetIndirectOutputCclBuf(void*& commIndirectOutputPtr, u64& commIndirectOutputSize);
    HcclResult GetDevType(DevType& devType);
    HcclResult SetGlobalWorkSpace() const;
    HcclResult ExecAlgSelect(const CollOpParams& opParams, int32_t aivCoreLimit, bool& ifAiv, std::string& algName);
    HcclResult CreateBarrierMemory(void*& sendBuf, void*& recvBuf, uint64_t count);
    HcclResult SetAivClearEnable(bool aivClearEnable);
    HcclResult SetAivCoreLimit(u32 aivCoreLimit);
    HcclResult GetAicpuOpStreamNotify(rtStream_t* opStream, u8 aicpuNotifyNum, void** aicpuNotify);
    HcclResult GetKFCWorkSpace(const char* memTag, uint64_t* size, void** addr, bool* newCreated);

    HcclResult GetRankGraphV2(void*& rankGraph);
    HcclResult HcclGetCclBuffer(uintptr_t& cclBufferAddr, size_t& cclBufferSize, HcclMemType& cclBufferMemType);
    // 获取cclBuffer的shared_ptr，用于延长其生命周期
    HcclResult GetCclBufferSharedPtr(std::shared_ptr<DevBuffer>& cclBuffer);
    HcclResult GetConfigInCCLbufferSize(uint64_t* cclBufSize);
    HcclResult GetNetLayers(uint32_t** netLayers, uint32_t* netLayerNum);
    HcclResult GetInstSizeByNetLayer(uint32_t netLayer, uint32_t* rankNum);
    HcclResult GetInstTopoTypeByNetLayer(uint32_t netLayer, uint32_t* topoType);
    HcclResult GetInstRanksByNetLayer(uint32_t netLayer, uint32_t** ranks, uint32_t* rankNum);
    HcclResult GetInstSizeListByNetLayer(uint32_t netLayer, uint32_t** instSizeList, uint32_t* listSize);
    HcclResult GetLinks(uint32_t netLayer, uint32_t srcRank, uint32_t dstRank, CommLink** linkList, uint32_t* listSize);
    HcclResult GetTopoInstsByLayer(uint32_t netLayer, uint32_t** topoInsts, uint32_t* topoInstNum);
    HcclResult GetTopoType(uint32_t netLayer, uint32_t topoInstId, CommTopo* topoType);
    HcclResult GetRanksByTopoInst(uint32_t netLayer, uint32_t topoInstId, uint32_t** ranks, uint32_t* rankNum);
    HcclResult CalcNumBlocks(const CollOpParams& opParams, int32_t aivCoreLimit, std::string& algName, u32& numBlocks);
    HcclResult
    GetAlgExecParam(const CollOpParams& opParams, bool clearEnable, void*& commContext, u64& len, u32 aivCoreLimit);
    HcclResult ClearOpResource(const std::string& opTag);
    HcclResult GetEndpointNum(uint32_t layer, uint32_t topoInstId, uint32_t* num);
    HcclResult GetEndpointDesc(uint32_t layer, uint32_t topoInstId, uint32_t* descNum, EndpointDesc* endpointDesc);
    HcclResult GetEndpointInfo(
        uint32_t rankId, const EndpointDesc* endpointDesc, EndpointAttr endpointAttr, uint32_t infoLen, void* info);
    Trace& GetTrace() const;

    u32 GetDeviceLogicId() const;
    u32 GetRankInParentComm();
    u8 GetCommDepth() const;
    HcclResult Mc2AiCpuStreamAllocAndGetV2(rtStream_t* aiCpuStream);
    HcclResult GetStreamId(u32& streamId);
    HcclResult GetRankIpPortMap(RankIpPortMapPtr& rankIpPortMap);

private:
    CommParams commParams;
    HcclCommConfig config{};
    std::unique_ptr<CommunicatorImpl> pimpl;
    CommunicatorImpl* GetCommImpl();
    std::mutex serialMutex;
};

/*
 * @brief 通知ccu做mission task kill
 */
HcclResult HcclCcuTaskKillPreProcess(u32 deviceLogicId);

/*
 * @brief ccu task kill之后通知ccu清除寄存器和配置表项
 */
HcclResult HcclCcuTaskKillPostProcess(u32 deviceLogicId);

/*
 * @brief 恢复通信域时配置ccu Pfe表项
 */
HcclResult HcclCcuResumePfeTableProcess(u32 deviceLogicId);

} // namespace Hccl

#endif
