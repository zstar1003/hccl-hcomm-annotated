/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <atomic>
#include <algorithm>
#include <arpa/inet.h>
#include <fstream>
#include <unistd.h>
#include <hccl/hccl_types.h>
#include "hccl_communicator.h"
#include "hccl_comm_pub.h"
#include "task_abort_handler_pub.h"
#include "i_hccl_one_sided_service.h"
#include "comm_configer.h"
#include "launch_aicpu.h"
#include "launch_device.h"
#include "sal_pub.h"
#include "coll_comm_config.h"
#include "coll_comm_mgr.h"
#include "env_config/env_config_v2.h"
#include "config_plf_log.h"
#include "plf_debug_config.h"
#include "dlprof_function.h"
#include "dfx_dlprof_function.h"

namespace hccl {
HcclResult hcclComm::AllReduce(
    const std::string& tag, void* inputPtr, void* outputPtr, u64 count, HcclDataType dataType, HcclReduceOp op,
    HcclRtStream stream, SyncMode syncMode)
{
    /* 增加输出日志关键字 */
    HCCL_INFO(
        "HCCL_KEY_INFO: tag[%s], input_ptr[%p], output_ptr[%p], count[%llu], data_type[%s], op[%s]", tag.c_str(),
        inputPtr, outputPtr, count, GetDataTypeEnumStr(dataType).c_str(), GetReduceOpEnumStr(op).c_str());

    /* * 入参检查 */
    CHK_PTR_NULL(stream);
    CHK_PTR_NULL(inputPtr);
    CHK_PTR_NULL(outputPtr);

    CHK_PRT_RET(
        tag.empty(),
        HCCL_ERROR("[HcclComm][AllReduce]errNo[0x%016llx] AllReduce tag length is 0", HCCL_ERROR_CODE(HCCL_E_PARA)),
        HCCL_E_PARA);

    CHK_RET(communicator_->CheckCount(count));
    CHK_RET(communicator_->CheckDataType(dataType, true));
    CHK_RET(communicator_->CheckReduceDataType(dataType, op));
    CHK_RET(communicator_->CheckReductionOp(op));
    HcclResult ret = communicator_->AllReduce(tag, inputPtr, outputPtr, count, dataType, op, stream, syncMode);
    if (ret != HCCL_SUCCESS) {
        PrintSubmittedOpCnt(tag, ret);
        return ret;
    }

    return HCCL_SUCCESS;
}

HcclResult hcclComm::AllReduceOutPlace(
    const std::string& tag, void* inputPtr, void* outputPtr, u64 count, HcclDataType dataType, HcclReduceOp op,
    HcclRtStream stream, SyncMode syncMode)
{
    /* 增加输出日志关键字 */
    HCCL_INFO(
        "HCCL_KEY_INFO: tag[%s], input_ptr[%p], output_ptr[%p], count[%llu], data_type[%s], op[%s]", tag.c_str(),
        inputPtr, outputPtr, count, GetDataTypeEnumStr(dataType).c_str(), GetReduceOpEnumStr(op).c_str());

    /* * 入参检查 */
    CHK_RET(communicator_->CheckDataType(dataType, true));
    CHK_RET(communicator_->CheckReduceDataType(dataType, op));
    HcclResult ret = communicator_->AllReduceOutPlace(tag, inputPtr, outputPtr, count, dataType, op, stream, syncMode);
    if (ret != HCCL_SUCCESS) {
        PrintSubmittedOpCnt(tag, ret);
        return ret;
    }

    return HCCL_SUCCESS;
}

HcclResult hcclComm::GetOneSidedService(IHcclOneSidedService** service)
{
    CHK_RET(communicator_->GetOneSidedService(service));

    return HCCL_SUCCESS;
}

HcclResult hcclComm::InitOneSidedServiceNetDevCtx(u32 remoteRankId)
{
    CHK_RET(communicator_->InitOneSidedServiceNetDevCtx(remoteRankId));
    return HCCL_SUCCESS;
}

HcclResult hcclComm::OneSidedServiceStartListen(NicType nicType, HcclNetDevCtx netDevCtx)
{
    CHK_SMART_PTR_NULL(communicator_);
    CHK_RET(communicator_->OneSidedServiceStartListen(nicType, netDevCtx));
    return HCCL_SUCCESS;
}

HcclResult hcclComm::GetOneSidedServiceDevIpAndPort(NicType nicType, HcclIpAddress& ipAddress, u32& port)
{
    CHK_SMART_PTR_NULL(communicator_);
    CHK_RET(communicator_->GetOneSidedServiceDevIpAndPort(nicType, ipAddress, port));
    return HCCL_SUCCESS;
}

HcclResult hcclComm::DeinitOneSidedService()
{
    CHK_SMART_PTR_NULL(communicator_);
    CHK_RET(communicator_->DeinitOneSidedService());
    return HCCL_SUCCESS;
}

HcclResult hcclComm::RegistTaskAbortHandler() const
{
    HCCL_RUN_INFO("RegistTaskAbortHandler begin, group[%s]", identifier_.c_str());
    CHK_RET(TaskAbortHandler::Init(communicator_.get()));
    return HCCL_SUCCESS;
}

HcclResult hcclComm::UnRegistTaskAbortHandler() const
{
    HCCL_RUN_INFO("UnRegistTaskAbortHandler begin, group[%s]", identifier_.c_str());
    CHK_RET(TaskAbortHandler::DeInit(communicator_.get()));
    return HCCL_SUCCESS;
}

HcclResult hcclComm::RegisterCommUserMem(void* addr, u64 size, void** handle)
{
    CHK_SMART_PTR_NULL(communicator_);
    CHK_RET(communicator_->RegisterCommUserMem(addr, size, handle));
    return HCCL_SUCCESS;
}

HcclResult hcclComm::DeregisterCommUserMem(void* handle)
{
    CHK_SMART_PTR_NULL(communicator_);
    CHK_RET(communicator_->DeregisterCommUserMem(handle));
    return HCCL_SUCCESS;
}

HcclResult hcclComm::ExchangeCommUserMem(void* handle, std::vector<u32>& peerRanks)
{
    CHK_SMART_PTR_NULL(communicator_);
    return communicator_->ExchangeCommUserMem(handle, peerRanks);
}

HcclResult hcclComm::SetIndependentOpConfig(const CommConfig& commConfig, const RankTable_t& rankTable)
{
    CHK_SMART_PTR_NULL(communicator_);
    HcclTopoAttr topoAttr = communicator_->GetTopoAttr();
    aclrtBinHandle binHandle = communicator_->GetBinHandle();
    HDCommunicateParams kfcControlTransferH2DParams;
    HDCommunicateParams kfcStatusTransferD2HParams;
    std::function<bool()> getAicpuCommState = [this]() {
        return this->GetIndependentOp().GetAicpuCommState();
    };
    CHK_RET(communicator_->GetHDCommunicate(kfcControlTransferH2DParams, kfcStatusTransferD2HParams));
    CHK_RET(communicator_->SetGetAicpuCommState(getAicpuCommState));
    CHK_RET(GetIndependentOp().SetIndependentOpConfig(
        commConfig, rankTable, topoAttr, binHandle, kfcControlTransferH2DParams, kfcStatusTransferD2HParams,
        communicator_->GetCCLbufferManager()));
    return HCCL_SUCCESS;
}

HcclResult hcclComm::ReleaseChannel() { return independentOp_.GetChannelManager().ReleaseChannel(); }

HcclResult hcclComm::InitIndependentOp()
{
    if (communicator_ != nullptr) {
        communicator_->SetReleaseChannel([this]() -> HcclResult {
            return this->ReleaseChannel();
        });
    }
    ChannelManagerCallbacks channelCallbacks;
    channelCallbacks.indOpTransportAlloc
        = [this](
              const std::string& tag, OpCommTransport& opCommTransport, bool isAicpuModeEn,
              const HcclMemHandle* memHandles, uint32_t memHandleNum) -> HcclResult {
        return this->IndOpTransportAlloc(tag, opCommTransport, isAicpuModeEn, memHandles, memHandleNum);
    };
    channelCallbacks.getRankLists = [this]() -> std::vector<RankInfo> {
        return this->GetRankLists();
    };
    return independentOp_.SetChannelCallbacks(channelCallbacks);
}

IndependentOp& hcclComm::GetIndependentOp() { return independentOp_; }
HcclResult hcclComm::PrepareChannelMem(
    [[maybe_unused]] const std::string& tag, TransportIOMem& transMem, const HcclMemHandle* memHandles,
    uint32_t memHandleNum)
{
    // 获取本地cclbuffer
    CommBuffer commBuffer;
    CHK_RET(GetIndependentOp().GetCommMemMgr().GetHcclBuffer(&commBuffer));
    DeviceMem cclbuffer = DeviceMem::create(commBuffer.addr, commBuffer.size);
    CHK_PTR_NULL(cclbuffer.ptr());

    // 获取通信域内存
    IndOpMem indOpMem{};
    std::vector<HcclMem> localMemVec{};
    CHK_RET(GetIndependentOp().GetCommMemMgr().CommGetLocalRegMemByHandles(memHandles, memHandleNum, localMemVec));
    for (const HcclMem& mem : localMemVec) {
        if (mem.type == HCCL_MEM_TYPE_HOST) {
            indOpMem.userHostMem.push_back(HostMem::create(mem.addr, mem.size));
            CHK_PTR_NULL(indOpMem.userHostMem.back().ptr());
        } else if (mem.type == HCCL_MEM_TYPE_DEVICE) {
            indOpMem.userDeviceMem.push_back(DeviceMem::create(mem.addr, mem.size));
            CHK_PTR_NULL(indOpMem.userDeviceMem.back().ptr());
        }
    }
    transMem.indOpMem = indOpMem;
    transMem.cclInputMem = cclbuffer;
    transMem.cclOutputMem = cclbuffer;
    return HCCL_SUCCESS;
}
HcclResult hcclComm::IndOpTransportAlloc(
    const std::string& tag, OpCommTransport& opCommTransport, bool isAicpuModeEn, const HcclMemHandle* memHandles,
    uint32_t memHandleNum)
{
    CHK_SMART_PTR_NULL(communicator_);
    TransportIOMem transMem;
    CHK_RET(PrepareChannelMem(tag, transMem, memHandles, memHandleNum));
    std::string commId = GetIdentifier();
    return communicator_->IndOpTransportAlloc(tag, opCommTransport, transMem, isAicpuModeEn);
}
HcclResult hcclComm::CommGetNetLayers(uint32_t** netLayers, uint32_t* netLayerNum)
{
    return communicator_->CommGetNetLayers(netLayers, netLayerNum);
}

HcclResult hcclComm::CommGetInstSizeByNetLayer(uint32_t netLayer, uint32_t* rankNum)
{
    return communicator_->CommGetInstSizeByNetLayer(netLayer, rankNum);
}

HcclResult hcclComm::CommGetInstTopoTypeByNetLayer(uint32_t netLayer, u32* topoType)
{
    return communicator_->CommGetInstTopoTypeByNetLayer(netLayer, topoType);
}
HcclResult hcclComm::GetNetLayers(uint32_t** netLayers, uint32_t* netLayerNum)
{
    return communicator_->GetNetLayers(netLayers, netLayerNum);
}

HcclResult hcclComm::GetInstSizeByNetLayer(uint32_t netLayer, uint32_t* rankNum)
{
    return communicator_->GetInstSizeByNetLayer(netLayer, rankNum);
}

HcclResult hcclComm::GetInstTopoTypeByNetLayer(uint32_t netLayer, CommTopo* topoType)
{
    return communicator_->GetInstTopoTypeByNetLayer(netLayer, topoType);
}

HcclResult hcclComm::GetInstRanksByNetLayer(uint32_t netLayer, uint32_t** rankList, uint32_t* rankNum)
{
    return communicator_->GetInstRanksByNetLayer(netLayer, rankList, rankNum);
}

HcclResult hcclComm::GetInstSizeListByNetLayer(uint32_t netLayer, uint32_t** instSizeList, uint32_t* listSize)
{
    return communicator_->GetInstSizeListByNetLayer(netLayer, instSizeList, listSize);
}

HcclResult hcclComm::GetTopoInstsByLayer(uint32_t netLayer, uint32_t** topoInsts, uint32_t* topoInstNum)
{
    return communicator_->GetTopoInstsByLayer(netLayer, topoInsts, topoInstNum);
}

HcclResult hcclComm::GetTopoType(uint32_t netLayer, uint32_t topoInstId, CommTopo* topoType)
{
    return communicator_->GetTopoType(netLayer, topoInstId, topoType);
}

HcclResult hcclComm::GetRanksByTopoInst(uint32_t netLayer, uint32_t topoInstId, uint32_t** ranks, uint32_t* rankNum)
{
    return communicator_->GetRanksByTopoInst(netLayer, topoInstId, ranks, rankNum);
}

HcclResult hcclComm::GetEndpointNum(uint32_t netLayer, uint32_t topoInstId, uint32_t* num)
{
    return communicator_->GetEndpointNum(netLayer, topoInstId, num);
}

HcclResult
hcclComm::GetEndpointDesc(uint32_t netLayer, uint32_t topoInstId, uint32_t* descNum, EndpointDesc* endpointDesc)
{
    return communicator_->GetEndpointDesc(netLayer, topoInstId, descNum, endpointDesc);
}

HcclResult hcclComm::GetEndpointInfo(
    uint32_t rankId, const EndpointDesc* endPointDesc, EndpointAttr endpointAttr, uint32_t infoLen, void* info)
{
    return communicator_->GetEndpointInfo(rankId, endPointDesc, endpointAttr, infoLen, info);
}

HcclResult hcclComm::GetRankGraph(GraphType type, void** graph, uint32_t* len)
{
    return communicator_->GetRankGraph(type, graph, len);
}

uint32_t hcclComm::GetConnectMode() { return communicator_->GetConnectMode(); }
HcclResult hcclComm::GetDevMemWorkSpace(const std::string& memTag, uint64_t* size, void** addr, bool* newCreated)
{
    return communicator_->GetDevMemWorkSpace(memTag, size, addr, newCreated);
}
HcclResult
hcclComm::GetLinks(uint32_t netLayer, uint32_t srcRank, uint32_t dstRank, CommLink** linkList, uint32_t* listSize)
{
    return communicator_->GetLinks(netLayer, srcRank, dstRank, linkList, listSize);
}

HcclResult hcclComm::GetHeterogMode(HcclHeterogMode* mode) { return communicator_->GetHeterogMode(mode); }

inline uint32_t GetCollCommOpExpansionMode(CollComm* collComm)
{
    auto* myRank = collComm->GetMyRank();
    CHK_PTR_NULL(myRank);
    return myRank->GetOpExpansionMode();
}

HcclResult hcclComm::InitCollComm(
    void* commV2, void* rankGraph, uint32_t userRank, HcclMem cclBuffer, const std::string& commName,
    const HcclCommConfig* config, CollCommInitMode initMode)
{
    // 不校验config，为空时配置默认加速模式

    // aicpu侧初始化状态的回调函数
    ManagerCallbacks callbacks;
    callbacks.getAicpuCommState = [this]() {
        return this->GetAicpuCommState();
    };
    callbacks.setAicpuCommState = [this](bool state) {
        this->SetAicpuCommState(state);
    };
    callbacks.kernelLaunchAicpuCommInit = [this]() {
        return this->KernelLaunchAicpuCommInit();
    };
    callbacks.reportProfilingKernel = [this](uint64_t beginTime, std::string kernelName) {
        return this->ReportProfilingKernel(beginTime, kernelName);
    };

    // Aicpu通信域初始化参数
    auto ret = snprintf_s(commAicpuParam_.hcomId, HCOMID_MAX_SIZE, HCOMID_MAX_SIZE - 1, "%s", commName.c_str());
    if (ret < 0) {
        HCCL_ERROR(
            "[InitCollComm]comm id snprintf_s fail, commId: %s, commId maxSize: %u", commName.c_str(), HCOMID_MAX_SIZE);
        return HCCL_E_PARA;
    }

    CHK_RET(hrtGetDevice(&(commAicpuParam_.deviceLogicId)));
    CHK_RET(hrtGetDevicePhyIdByIndex(static_cast<u32>(commAicpuParam_.deviceLogicId), commAicpuParam_.devicePhyId));
    CHK_RET(hrtGetDeviceType(devType_));
    commAicpuParam_.deviceType = static_cast<u32>(devType_);
    CHK_RET(InitBinHandle());

    EXCEPTION_CATCH(
        collComm_ = std::make_unique<CollComm>(commV2, userRank, commName, callbacks, initMode), return HCCL_E_PTR);

    uint32_t configOpExpansionMode = 0;
    CHK_RET(ApplyHcclCommConfig(config, collComm_->GetCommConfig(), configOpExpansionMode));
    CHK_RET(collComm_->Init(rankGraph, binHandle_, cclBuffer, configOpExpansionMode));
    if (initMode == CollCommInitMode::simpleMode) { /* hccl::CommunicatorV1支持CollComm简易流程 */
        return HCCL_SUCCESS;
    }

    // 注册通信域到 CollCommMgr，由 owner(hcclComm) 负责注册/注销，避免 CollComm 反向依赖 CollCommMgr
    CollCommMgr::GetInstance().RegisterCollComm(collComm_.get());

    CHK_RET(collComm_->GetHDCommunicate(
        commAicpuParam_.kfcControlTransferH2DParams, commAicpuParam_.kfcStatusTransferD2HParams));
    commAicpuParam_.userRank = collComm_->GetMyRankId();
    commAicpuParam_.userRankSize = collComm_->GetRankSize();
    commAicpuParam_.commConfig.taskExceptionEnable
        = Hccl::EnvConfig::GetInstance().GetLogConfig().GetDfsConfig().taskExceptionEnable;
    commAicpuParam_.commConfig.notifyWaitTimeout = Hccl::EnvConfig::GetInstance().GetRtsConfig().GetExecTimeOut();
    commAicpuParam_.commConfig.plfDebugConfig = Hccl::GetPlfDebugConfigValue();
    const auto opExpansionMode = GetCollCommOpExpansionMode(collComm_.get());
    HCCL_RUN_INFO(
        "[%s]success, commId[%s], deviceLogicId[%u], devicePhyId[%u], devType[%u], "
        "userRank[%u], userRankSize[%u], opExpansionMode[%u], taskExceptionEnable[%d], notifyWaitTimeout[%u], "
        "plfDebugConfig[0x%llx].",
        __func__, collComm_->GetCommId().c_str(), commAicpuParam_.deviceLogicId, commAicpuParam_.devicePhyId,
        commAicpuParam_.deviceType, commAicpuParam_.userRank, commAicpuParam_.userRankSize, opExpansionMode,
        commAicpuParam_.commConfig.taskExceptionEnable, commAicpuParam_.commConfig.notifyWaitTimeout,
        commAicpuParam_.commConfig.plfDebugConfig);

    // 当前需要支持coll comm与legacy comm混跑，coll comm确定加速模式后，需要设置comm加速模式
    auto* commImplV2 = static_cast<Hccl::HcclCommunicator*>(commV2);
    constexpr bool isCcuMsAvailable = false; // 禁止legacy通信域使用ms模式，避免抢占过多coll comm ccu可用资源
    CHK_RET(commImplV2->SetAccelerator(static_cast<int32_t>(opExpansionMode), isCcuMsAvailable));

    return HCCL_SUCCESS;
}

HcclResult hcclComm::InitCollCommInner(uint32_t userRank)
{
    if (GetConnectMode() == 0) {
        return HCCL_SUCCESS;
    }

    CHK_PRT_RET(
        userRank == INVALID_VALUE_RANKID, HCCL_ERROR("[%s] invalid userRank[%u]", __func__, userRank), HCCL_E_PARA);

    std::string commName = GetIdentifier();
    HCCL_INFO("[%s]Init CollComm start, comm[%s], userRank[%u]", __func__, commName.c_str(), userRank);
    HcclCommunicator* hcclComm = GetHcclCommunicator();
    if (hcclComm == nullptr) {
        HCCL_WARNING("[%s] HcclCommunicator NULL, skip CollComm init", __func__);
        return HCCL_SUCCESS;
    }

    void* rankGraphV1 = hcclComm->GetRankGraphV1();
    if (rankGraphV1 == nullptr) {
        HCCL_WARNING("[%s] rankGraphV1 is nullptr, skip CollComm init, comm[%s]", __func__, commName.c_str());
        return HCCL_SUCCESS;
    }

    void* cclBufferAddr = nullptr;
    u64 cclBufferSize = 0;
    CHK_RET(CreateCommCCLbuffer());
    HcclResult ret = hcclComm->GetInCCLbuffer(cclBufferAddr, cclBufferSize);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[%s] GetInCCLbuffer failed, comm[%s], ret=%d", __func__, commName.c_str(), ret);
        return ret;
    }

    HcclMem cclBuffer{};
    cclBuffer.size = static_cast<uint64_t>(cclBufferSize);
    cclBuffer.addr = cclBufferAddr;
    cclBuffer.type = HcclMemType::HCCL_MEM_TYPE_DEVICE;
    constexpr const HcclCommConfig* config = nullptr;

    ret = InitCollComm(nullptr, rankGraphV1, userRank, cclBuffer, commName, config, CollCommInitMode::simpleMode);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[%s] InitCollComm failed, comm[%s], ret=%d", __func__, commName.c_str(), ret);
        return ret;
    }

    HCCL_INFO("[%s] CollComm init success for V1, comm[%s]", __func__, commName.c_str());
    return HCCL_SUCCESS;
}

HcclResult hcclComm::InitBinHandle()
{
    std::string jsonPath;
    CHK_RET(GetKernelFilePath(jsonPath));
    jsonPath += "ccl_kernel.json";

    HcclResult retCode = LoadBinaryFromFile(jsonPath.c_str(), ACL_RT_BINARY_LOAD_OPT_CPU_KERNEL_MODE, 0, binHandle_);
    CHK_PRT_RET(
        retCode != HCCL_SUCCESS,
        HCCL_ERROR(
            "[InitCollComm]errNo[0x%016llx]load aicpu file fail, path[%s] optionType[%u] "
            "cpuKernelMode[%u].",
            retCode, jsonPath.c_str(), ACL_RT_BINARY_LOAD_OPT_CPU_KERNEL_MODE, 0),
        retCode);
    return HCCL_SUCCESS;
}

void hcclComm::BinaryUnLoad()
{
    if (binHandle_ != nullptr) {
        HCCL_INFO("[BinaryUnLoad]aclrtBinaryUnLoad binHandle");
        aclError ret = aclrtBinaryUnLoad(binHandle_);
        if (ret != 0) {
            HCCL_RUN_WARNING("[BinaryUnLoad]aclrtBinaryUnLoad binHandle failed");
        }
        binHandle_ = nullptr;
    }
}

bool hcclComm::GetAicpuCommState() const { return isAicpuCommInit_; }

void hcclComm::SetAicpuCommState(bool aicpuCommState)
{
    isAicpuCommInit_ = aicpuCommState;
    return;
}

HcclResult hcclComm::KernelLaunchAicpuCommInit()
{
    // 创建局部流
    u64 beginTime = Hccl::DfxDlProfFunction::GetInstance().dlMsprofSysCycleTime();
    Stream localStream(StreamType::STREAM_TYPE_ONLINE);
    constexpr u32 aicpuStreamMode = 1;
    CHK_RET(hrtStreamSetMode(localStream.ptr(), aicpuStreamMode));

    // 下kernel进行自定义算子aicpu侧通信域的公共初始化
    std::string kernelName = "RunAicpuCommInit";
    HCCL_INFO("AicpuAclKernelLaunch start");
    s32 timeout = 1836;
    if (IsCommunicatorV2()) {
        timeout = Hccl::EnvConfig::GetInstance().GetRtsConfig().GetExecTimeOut() + 25; // 多25s，避免超时
    } else {
        timeout = CommConfiger::GetInstance().GetCommConfigExecTimeOut("") + 25; // 多25s，避免超时
    }
    CHK_RET(AicpuAclKernelLaunch(
        localStream.ptr(), static_cast<void*>(&commAicpuParam_), sizeof(commAicpuParam_), binHandle_, kernelName, true,
        timeout));
    HCCL_INFO("AicpuAclKernelLaunch end, hcclStreamSynchronize start");
    CHK_RET(hcclStreamSynchronize(localStream.ptr(), timeout));
    HCCL_INFO("[KernelLaunchAicpuCommInit] ReportAicpuCommKernel begin");
    CHK_PTR_NULL(collComm_);
    HcclCommDfx* hcclComDfx = collComm_->GetHcclCommDfx();
    CHK_PTR_NULL(hcclComDfx);
    // 通信域初始化在op注册之前，这个地方一定是false，因为还不知道是不是图模式
    CHK_RET(hcclComDfx->ReportKernel(beginTime, identifier_, kernelName, SalGetTid(), false));
    HCCL_INFO("[KernelLaunchAicpuCommInit] ReportAicpuCommKernel end");
    // 打印增加初始化对应的参数
    HCCL_RUN_INFO("[%s] KernelLaunchAicpuCommInit Success", __func__);
    return HCCL_SUCCESS;
}

HcclResult hcclComm::ReportProfilingKernel(uint64_t beginTime, std::string kernelName)
{
    CHK_PTR_NULL(collComm_);
    HcclCommDfx* hcclComDfx = collComm_->GetHcclCommDfx();
    CHK_PTR_NULL(hcclComDfx);
    // 通信域初始化在op注册之前，这个地方一定是false，因为还不知道是不是图模式
    CHK_RET(hcclComDfx->ReportKernel(beginTime, identifier_, kernelName, SalGetTid(), false));
    return HCCL_SUCCESS;
}

HcclComm hcclComm::GetCommunicatorV2()
{
    if (collComm_ == nullptr) {
        return nullptr;
    }
    return collComm_->GetCommunicatorV2();
}

HcclCommunicator* hcclComm::GetHcclCommunicator() { return communicator_.get(); }

CollComm* hcclComm::GetCollComm() { return collComm_ != nullptr ? collComm_.get() : nullptr; }

HcclResult hcclComm::Resume()
{
    if (IsCommunicatorV2()) {
        CHK_RET(collComm_->Resume());
    } else {
        CHK_RET(communicator_->Resume());
    }

    return HCCL_SUCCESS;
}
HcclResult hcclComm::GetCommStatus(HcclCommStatus& status)
{
    if (IsCommunicatorV2()) {
        status = collComm_->GetCommStatus();
    } else if (devType_ == DevType::DEV_TYPE_910B && collComm_ != nullptr) {
        status = collComm_->GetCommStatus();
    } else {
        HCCL_ERROR("[%s] deviceType is not supported", __func__);
        return HCCL_E_NOT_SUPPORT;
    }
    return HCCL_SUCCESS;
}

} // namespace hccl
