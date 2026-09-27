/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl_communicator.h"

#include <memory>
#include <utility>
#include "communicator_impl.h"
#include "snap_shot_parse.h"
#include "task_abort_handler_v2.h"
#include "ccu_dev_mgr.h"
#include "communicator_callback.h"
#include "comm_manager.h"
#include "orion_adapter_rts.h"

namespace Hccl {

HcclCommunicator::HcclCommunicator(const CommParams& commParams) : commParams(std::move(commParams))
{
    pimpl = std::make_unique<CommunicatorImpl>();
    config.hcclBufferSize = 0;
    config.hcclDeterministic = 0;
    RegistTaskAbortHandler();
}

HcclCommunicator::HcclCommunicator(const CommParams& commParams, const HcclCommConfig* config)
    : commParams(std::move(commParams)),
      config(*config)
{
    pimpl = std::make_unique<CommunicatorImpl>();
    RegistTaskAbortHandler();
}

HcclCommunicator::~HcclCommunicator()
{
    DECTOR_TRY_CATCH("HcclCommunicator", {
        UnRegistTaskAbortHandler();
        u32 devLogicId = GetDeviceLogicId();
        pimpl = nullptr;
        CommManager::GetInstance(static_cast<s32>(devLogicId)).DeinitCcuDriver();
    });
}

HcclResult HcclCommunicator::Init(const std::string& ranktableM) { return pimpl->Init(commParams, ranktableM, config); }

HcclResult HcclCommunicator::Init(const RankTableInfo& ranktable) { return pimpl->Init(commParams, ranktable, config); }

HcclResult HcclCommunicator::CreateSubComm(
    const CommParams& subCommParams, const std::vector<u32>& rankIds, std::shared_ptr<HcclCommunicator>& subHcclComm)
{
    subHcclComm = std::make_shared<Hccl::HcclCommunicator>(subCommParams);
    return pimpl->CreateSubComm(subCommParams, rankIds, subHcclComm->GetCommImpl());
}

HcclResult HcclCommunicator::CreateSubComm(
    const CommParams& subCommParams, const std::vector<u32>& rankIds, std::shared_ptr<HcclCommunicator>& subHcclComm,
    HcclCommConfig& subConfig)
{
    subHcclComm = std::make_shared<Hccl::HcclCommunicator>(subCommParams);
    config.hcclBufferSize = 0;
    config.hcclDeterministic = 0;
    return pimpl->CreateSubComm(subCommParams, rankIds, subHcclComm->GetCommImpl(), subConfig);
}

CommunicatorImpl* HcclCommunicator::GetCommImpl() { return pimpl.get(); }

void HcclCommunicator::DeInit() const {}

HcclResult HcclCommunicator::LoadOpbasedCollOp(const CollOpParams& opParams, void* stream)
{
    return pimpl->LoadOpbasedCollOp(opParams, stream);
}

HcclResult HcclCommunicator::AllocCollOpResource(const CollOpParams& opParams, void** addr)
{
    return pimpl->AllocCollOpResource(opParams, addr);
}

HcclResult HcclCommunicator::CalcCollOffloadOpRes(
    const OpType opType, u64 dataSize, HcclDataType dataType, CollOffloadOpResReq& resReq)
{
    std::lock_guard<std::mutex> lock(serialMutex);
    auto ret = pimpl->CalcCollOffloadOpRes(opType, dataSize, dataType, resReq);
    return ret;
}

HcclResult HcclCommunicator::SetCollOffloadSlaveStreams(const std::string& opTag, std::vector<void*> slaveStreams)
{
    std::lock_guard<std::mutex> lock(serialMutex);
    auto ret = pimpl->SetCollOffloadSlaveStreams(opTag, slaveStreams);
    return ret;
}

HcclResult
HcclCommunicator::SetCollOffloadScratchBuf(const std::string& opTag, void* scratchMemPtr, u64 requiredScratchMemSize)
{
    std::lock_guard<std::mutex> lock(serialMutex);
    auto ret = pimpl->SetCollOffloadScratchBuf(opTag, scratchMemPtr, requiredScratchMemSize);
    return ret;
}

HcclResult HcclCommunicator::LoadOffloadCollOp(std::string& opTag, const CollOpParams& opParams, void* stream)
{
    std::lock_guard<std::mutex> lock(serialMutex);
    auto ret = pimpl->LoadOffloadCollOp(opTag, opParams, stream);
    return ret;
}

HcclResult HcclCommunicator::GetRankSize(uint32_t* rankSize)
{
    if (rankSize == nullptr) {
        HCCL_ERROR("Parameter rank size is nullptr.");
        return HcclResult::HCCL_E_PARA;
    }

    *rankSize = pimpl->GetRankSize();

    return HcclResult::HCCL_SUCCESS;
}

HcclResult
HcclCommunicator::HcclGetCclBuffer(uintptr_t& cclBufferAddr, size_t& cclBufferSize, HcclMemType& cclBufferMemType)
{
    auto commImpl = GetCommImpl();
    if (UNLIKELY(commImpl == nullptr)) {
        HCCL_ERROR("[HcclCommunicator][%s] commImpl is null.", __func__);
        return HcclResult::HCCL_E_PTR;
    }

    // GetCclBuffer接口不合理，应返回裸指针
    // 本次性能整改暂时最小化修改，避免重复分配内存和拷贝
    const auto& hcclBuffer = commImpl->GetCclBuffer();
    if (UNLIKELY(hcclBuffer == nullptr)) {
        cclBufferSize = 0;
        cclBufferAddr = 0;
        cclBufferMemType = HcclMemType::HCCL_MEM_TYPE_DEVICE;
        return HcclResult::HCCL_SUCCESS;
    }

    cclBufferSize = commImpl->GetBufferSize();
    cclBufferAddr = hcclBuffer->GetAddr();
    cclBufferMemType = hcclBuffer->GetMemType();
    return HCCL_SUCCESS;
}

// 后续会整改把cclbuffer等资源的申请放到collcomm中
HcclResult HcclCommunicator::GetCclBufferSharedPtr(std::shared_ptr<DevBuffer>& cclBuffer)
{
    auto commImpl = GetCommImpl();
    if (commImpl == nullptr) {
        HCCL_ERROR("[HcclCommunicator][%s] commImpl is null.", __func__);
        return HcclResult::HCCL_E_PTR;
    }
    cclBuffer = commImpl->GetCclBuffer();
    if (cclBuffer == nullptr) {
        HCCL_WARNING("[HcclCommunicator][%s] cclBuffer is nullptr, ranksize is %u.", __func__, commImpl->GetRankSize());
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult HcclCommunicator::GetRankId(uint32_t& rankId)
{
    rankId = pimpl->GetMyRank();
    return HcclResult::HCCL_SUCCESS;
}

HcclResult HcclCommunicator::AllocCommResource(void* mc2Tiling, void** commContext)
{
    std::lock_guard<std::mutex> lock(serialMutex);
    auto ret = pimpl->AllocCommResource(mc2Tiling, commContext);
    return ret;
}

HcclResult HcclCommunicator::GetCcuTaskInfo(void* tilingData, void* ccuTaskGroup)
{
    return pimpl->GetCcuTaskInfo(tilingData, ccuTaskGroup);
}

u32 HcclCommunicator::GetCcuMc2ServerNum() { return pimpl->GetCcuMc2ServerNum(); }

const std::string& HcclCommunicator::GetId() const { return pimpl->GetId(); }

HcclResult HcclCommunicator::Suspend()
{
    std::lock_guard<std::mutex> lock(serialMutex);
    auto ret = pimpl->Suspend();
    return ret;
}

HcclResult HcclCommunicator::Clean()
{
    std::lock_guard<std::mutex> lock(serialMutex);
    auto ret = pimpl->Clean();
    return ret;
}

HcclResult HcclCommunicator::Resume()
{
    std::lock_guard<std::mutex> lock(serialMutex);
    auto ret = pimpl->Resume();
    return ret;
}

bool HcclCommunicator::IsWorldGroup() const { return pimpl->IsWorldGroup(); }

HcclResult HcclCcuTaskKillPreProcess(u32 deviceLogicId)
{
    // 有没有使能ccu都尝试执行
    return CcuSetTaskKill(deviceLogicId);
}

HcclResult HcclCcuTaskKillPostProcess(u32 deviceLogicId) { return CcuSetTaskKillDone(deviceLogicId); }

HcclResult HcclCcuResumePfeTableProcess([[maybe_unused]] u32 deviceLogicId)
{
    // 待修改
    return HcclResult::HCCL_SUCCESS;
}

HcclResult HcclCommunicator::GetSnapShotDynamicBuf(void* buf)
{
    std::lock_guard<std::mutex> lock(serialMutex);
    CHK_RET(pimpl->GetSnapShotDynamicBuf(*(static_cast<BinaryStream*>(buf))));
    return HcclResult::HCCL_SUCCESS;
}

HcclResult HcclCommunicator::RecoverComm(void* snapShotComm, u32 step, const char* changeInfo)
{
    std::lock_guard<std::mutex> lock(serialMutex);
    return pimpl->RecoverComm(*(static_cast<SnapShotComm*>(snapShotComm)), step, changeInfo);
}

HcclResult
HcclCommunicator::RecoverSubComm(const void* snapShotSubComm, std::shared_ptr<HcclCommunicator>& subComm, u32 step)
{
    const SnapShotSubComm* snapShotSubCommTemp = static_cast<const SnapShotSubComm*>(snapShotSubComm);
    std::lock_guard<std::mutex> lock(serialMutex);
    subComm = std::make_shared<Hccl::HcclCommunicator>(snapShotSubCommTemp->commParams);
    subComm->RegisterAcceStateCallBack(CommunicatorCallback());
    return pimpl->RecoverSubComm(*snapShotSubCommTemp, subComm->GetCommImpl(), step);
}

void* HcclCommunicator::GetStaticBinaryInfo()
{
    std::lock_guard<std::mutex> lock(serialMutex);
    return static_cast<void*>(&pimpl->GetStaticBinaryInfo());
}

bool HcclCommunicator::IsCommReady() { return pimpl->IsCommReady(); }

void HcclCommunicator::RegistTaskAbortHandler() { TaskAbortHandler::GetInstance().Register(this); }

void HcclCommunicator::UnRegistTaskAbortHandler() { TaskAbortHandler::GetInstance().UnRegister(this); }

HcclResult HcclCommunicator::GetOneSidedService(HcclOneSidedService** oneSidedService)
{
    HCCL_INFO("HcclCommunicator::GetOneSidedService begin");
    CHK_RET(pimpl->GetOneSidedService(oneSidedService));
    HCCL_INFO("HcclCommunicator::GetOneSidedService end");
    return HCCL_SUCCESS;
}

u32 HcclCommunicator::GetUsedChannelCount(u32 dieId) { return pimpl->GetUsedChannelCount(dieId); }

void HcclCommunicator::RegisterPrintChannelInfoCallback(std::function<void()> callback)
{
    pimpl->RegisterPrintChannelInfoCallback(callback);
}

CommStatus HcclCommunicator::GetCommStatus() const { return pimpl->GetCommStatus(); }

HcclResult HcclCommunicator::CreateCommCclBuf()
{
    HCCL_INFO("HcclCommunicator::CreateCommCclBuf start");
    return pimpl->CreateCommCclBuf();
}

HcclResult HcclCommunicator::GetInCclBuf(void*& commInputPtr, u64& commInputSize)
{
    return pimpl->GetInCclBuf(commInputPtr, commInputSize);
}

HcclResult HcclCommunicator::GetOutCclBuf(void*& commOutputPtr, u64& commOutputSize)
{
    return pimpl->GetOutCclBuf(commOutputPtr, commOutputSize);
}

HcclResult HcclCommunicator::GetLocalCclBuffer(void** addr, uint64_t* size)
{
    return pimpl->GetLocalCclBuffer(addr, size);
}

HcclResult
HcclCommunicator::GetDevMemWorkSpace(const std::string& memTag, uint64_t* size, void** addr, bool* newCreated)
{
    return pimpl->GetDevMemWorkSpace(memTag, size, addr, newCreated);
}

HcclResult HcclCommunicator::GetAicpuOpStreamNotify(rtStream_t* opStream, u8 aicpuNotifyNum, void** aicpuNotify)
{
    return pimpl->GetAicpuOpStreamNotify(opStream, aicpuNotifyNum, aicpuNotify);
}

HcclResult HcclCommunicator::GetIndirectInputCclBuf(void*& commIndirectInputPtr, u64& commIndirectInputSize)
{
    return pimpl->GetIndirectInCclBuf(commIndirectInputPtr, commIndirectInputSize);
}

HcclResult HcclCommunicator::GetIndirectOutputCclBuf(void*& commIndirectOutputPtr, u64& commIndirectOutputSize)
{
    return pimpl->GetIndirectOutCclBuf(commIndirectOutputPtr, commIndirectOutputSize);
}

HcclResult HcclCommunicator::SetAccelerator(HcclAccelerator hcclAccelerator, bool isCcuMsAvailable)
{
    CHK_RET(pimpl->SetAccelerator(hcclAccelerator, isCcuMsAvailable));
    return HcclResult::HCCL_SUCCESS;
}

HcclResult HcclCommunicator::SetAccelerator(int32_t accelerator, bool isCcuMsAvailable)
{
    if (accelerator < static_cast<int32_t>(HcclAccelerator::DEFAULT)
        || accelerator > static_cast<int32_t>(HcclAccelerator::AICPU)) {
        HCCL_ERROR(
            "[HcclCommunicator][SetAccelerator] Invalid accelerator value [%d], valid range is [0,7]", accelerator);
        return HCCL_E_NOT_SUPPORT;
    }
    HcclAccelerator hcclAccelerator = static_cast<HcclAccelerator::Value>(accelerator);
    CHK_RET(SetAccelerator(hcclAccelerator, isCcuMsAvailable));
    return HcclResult::HCCL_SUCCESS;
}

HcclResult HcclCommunicator::GetAccelerator(int32_t* accelerator) const
{
    CHK_RET(pimpl->GetAccelerator(accelerator));
    return HcclResult::HCCL_SUCCESS;
}

bool HcclCommunicator::IsUsingCcuMs() const
{
    return pimpl->IsCommUsingCcuMs(); // 通信域粒度
}

bool HcclCommunicator::IsUsingCcuSched() const
{
    return pimpl->IsCommUsingCcuSched(); // 通信域粒度
}

void HcclCommunicator::RegisterAcceStateCallBack(
    std::function<HcclResult(const std::string& commId, bool isUsingCcuMs, bool isUsingCcuSched)> callback)
{
    pimpl->RegisterAcceStateCallBack(callback);
}

HcclResult HcclCommunicator::CalcTaskNum(OpType opType, DataType dataType, u64 count, u32& taskNum)
{
    HCCL_INFO("HcclCommunicator::CalcTaskNum begin");
    return pimpl->CalcTaskNum(opType, dataType, count, taskNum);
}

HcclResult HcclCommunicator::GetTopoDesc(HcclTopoDescs* topoDescs, uint32_t topoSize)
{
    return pimpl->GetTopoDesc(topoDescs, topoSize);
}

HcclResult HcclCommunicator::GetDevType(DevType& devType)
{
    devType = pimpl->GetDevType();
    HCCL_INFO("HcclCommunicator::GetDevTyp, devtype is %s", devType.Describe().c_str());
    return HcclResult::HCCL_SUCCESS;
}

HcclResult HcclCommunicator::SetGlobalWorkSpace() const
{
    HCCL_WARNING("set global work space not support at A5");
    return HCCL_SUCCESS;
}

HcclResult
HcclCommunicator::ExecAlgSelect(const CollOpParams& opParams, int32_t aivCoreLimit, bool& ifAiv, std::string& algName)
{
    return pimpl->HcomSelectAlg(opParams, aivCoreLimit, ifAiv, algName);
}

HcclResult HcclCommunicator::GetRankGraphV2(void*& rankGraph)
{
    CHK_SMART_PTR_NULL(pimpl);
    shared_ptr<RankGraph> rankGraphShPtr = pimpl->GetRankGraph();
    CHK_SMART_PTR_NULL(rankGraphShPtr);
    rankGraph = static_cast<void*>(rankGraphShPtr.get());
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::CreateBarrierMemory(void*& sendBuf, void*& recvBuf, uint64_t count)
{
    return pimpl->CreateBarrierMemory(sendBuf, recvBuf, count);
}

HcclResult HcclCommunicator::SetAivClearEnable(bool aivClearEnable)
{
    pimpl->SetAivClearEnable(aivClearEnable);
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::SetAivCoreLimit(u32 newAivCoreLimit)
{
    pimpl->SetAivCoreLimit(newAivCoreLimit);
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::GetNetLayers(uint32_t** netLayers, uint32_t* netLayerNum)
{
    return pimpl->GetNetLayers(netLayers, netLayerNum);
}

HcclResult HcclCommunicator::GetInstSizeByNetLayer(uint32_t netLayer, uint32_t* rankNum)
{
    return pimpl->GetInstSizeByNetLayer(netLayer, rankNum);
}

HcclResult HcclCommunicator::GetConfigInCCLbufferSize(uint64_t* cclBufSize)
{
    *cclBufSize = static_cast<uint64_t>(pimpl->GetBufferSize());
    return HCCL_SUCCESS;
}
HcclResult HcclCommunicator::GetKFCWorkSpace(const char* memTag, uint64_t* size, void** addr, bool* newCreated)
{
    HCCL_INFO("HcclCommunicator::GetKFCWorkSpace start");
    CHK_RET(pimpl->CreateWorkspaceBuf(memTag, size, newCreated));
    shared_ptr<DevBuffer> buff = pimpl->GetKFCWorkSpace(memTag);
    *addr = reinterpret_cast<void*>(buff.get()->GetAddr());
    if (*size != static_cast<uint64_t>(buff.get()->GetSize())) {
        HCCL_ERROR(
            "HcclCommunicator::GetKFCWorkSpace, The size of mem is non-consistent. [%u->%u]", *size,
            buff.get()->GetSize());
        return HCCL_E_PARA;
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult HcclCommunicator::GetInstRanksByNetLayer(uint32_t netLayer, uint32_t** ranks, uint32_t* rankNum)
{
    return pimpl->GetInstRanksByNetLayer(netLayer, ranks, rankNum);
}

HcclResult HcclCommunicator::GetInstTopoTypeByNetLayer(uint32_t netLayer, uint32_t* topoType)
{
    return pimpl->GetInstTopoTypeByNetLayer(netLayer, topoType);
}

HcclResult HcclCommunicator::GetInstSizeListByNetLayer(uint32_t netLayer, uint32_t** instSizeList, uint32_t* listSize)
{
    return pimpl->GetInstSizeListByNetLayer(netLayer, instSizeList, listSize);
}

HcclResult HcclCommunicator::GetLinks(
    uint32_t netLayer, uint32_t srcRank, uint32_t dstRank, CommLink** linkList, uint32_t* listSize)
{
    return pimpl->GetLinks(netLayer, srcRank, dstRank, linkList, listSize);
}

HcclResult HcclCommunicator::GetTopoInstsByLayer(uint32_t netLayer, uint32_t** topoInsts, uint32_t* topoInstNum)
{
    return pimpl->GetTopoInstsByLayer(netLayer, topoInsts, topoInstNum);
}

HcclResult HcclCommunicator::GetTopoType(uint32_t netLayer, uint32_t topoInstId, CommTopo* topoType)
{
    return pimpl->GetTopoType(netLayer, topoInstId, topoType);
}

HcclResult
HcclCommunicator::GetRanksByTopoInst(uint32_t netLayer, uint32_t topoInstId, uint32_t** ranks, uint32_t* rankNum)
{
    return pimpl->GetRanksByTopoInst(netLayer, topoInstId, ranks, rankNum);
}

HcclResult HcclCommunicator::CalcNumBlocks(
    const CollOpParams& opParams, int32_t aivCoreLimit, std::string& algName, u32& numBlocks)
{
    return pimpl->CalcNumBlocks(opParams, aivCoreLimit, algName, numBlocks);
}

HcclResult HcclCommunicator::GetAlgExecParam(
    const CollOpParams& opParams, bool clearEnable, void*& commContext, u64& len, u32 aivCoreLimit)
{
    return pimpl->GetAlgExecParam(opParams, clearEnable, commContext, len, aivCoreLimit);
}

HcclResult HcclCommunicator::ClearOpResource(const std::string& opTag) { return pimpl->ClearOpResource(opTag); }

u32 HcclCommunicator::GetDeviceLogicId() const { return pimpl->GetDeviceLogicId(); }

HcclResult HcclCommunicator::GetEndpointNum(uint32_t layer, uint32_t topoInstId, uint32_t* num)
{
    return pimpl->GetEndpointNum(layer, topoInstId, num);
}

HcclResult
HcclCommunicator::GetEndpointDesc(uint32_t layer, uint32_t topoInstId, uint32_t* descNum, EndpointDesc* endpointDesc)
{
    return pimpl->GetEndpointDesc(layer, topoInstId, descNum, endpointDesc);
}

HcclResult HcclCommunicator::GetEndpointInfo(
    uint32_t rankId, const EndpointDesc* endpointDesc, EndpointAttr endpointAttr, uint32_t infoLen, void* info)
{
    return pimpl->GetEndpointInfo(rankId, endpointDesc, endpointAttr, infoLen, info);
}

Trace& HcclCommunicator::GetTrace() const { return pimpl->GetTrace(); }

u32 HcclCommunicator::GetRankInParentComm() { return pimpl->GetRankInParentComm(); }

u8 HcclCommunicator::GetCommDepth() const { return pimpl->GetCommDepth(); }

HcclResult HcclCommunicator::Mc2AiCpuStreamAllocAndGetV2(rtStream_t* aiCpuStream)
{
    return pimpl->Mc2AiCpuStreamAllocAndGetV2(aiCpuStream);
}

HcclResult HcclCommunicator::GetStreamId(u32& streamId)
{
    streamId = pimpl->GetDpuStreamId();
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::GetRankIpPortMap(RankIpPortMapPtr& rankIpPortMap)
{
    CHK_RET(pimpl->GetRankIpPortMap(rankIpPortMap));
    return HCCL_SUCCESS;
}

} // namespace Hccl
