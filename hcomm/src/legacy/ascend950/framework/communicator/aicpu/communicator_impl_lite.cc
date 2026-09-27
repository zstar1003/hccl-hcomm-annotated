/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "communicator_impl_lite.h"
#include "aicpu_res_package_helper.h"
#include "alg_topo_package_helper.h"
#include "sal.h"
#include "suspending_exception.h"
#include "exception_util.h"
#include "task_info.h"
namespace Hccl {

constexpr int KERNEL_SUCCESS_CODE = 0;
constexpr int KERNEL_ERROR_CODE = 1;
constexpr int ALLTOALLV_DATA_INDEX_2 = 2; // sdispls在sendRecvInfos数组中的偏移
constexpr int ALLTOALLV_DATA_INDEX_3 = 3; // rdispls在sendRecvInfos数组中的偏移

HcclResult CommunicatorImplLite::InitProfilingReporterLite()
{
    CHK_RET(Hccl::ProfilingHandlerLite::GetInstance().Init());
    CHK_RET(profilingReporterLite->Init());
    return HCCL_SUCCESS;
}

int CommunicatorImplLite::LoadWithOpBasedMode(HcclKernelParamLite* kernelParam)
{
    try {
        // 设定devType，初始化能力，算法及其他模块通过Get获取能力
        DevCapability::GetInstance().Init(kernelParam->comm.devType);
        HcclResult ret = InitProfilingReporterLite();
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("InitProfilingReporterLite failed, ret[%d]", ret);
            return KERNEL_ERROR_CODE;
        }
        UnfoldOp(kernelParam);
    } catch (HcclException& e) {
        HCCL_ERROR("Hccl exception %s was caught.", e.what());
        return KERNEL_ERROR_CODE;
    } catch (std::exception& e) {
        HCCL_ERROR("Std exception %s was caught.", e.what());
        return KERNEL_ERROR_CODE;
    } catch (...) {
        HCCL_ERROR("Some unknown error occurred.");
        return KERNEL_ERROR_CODE;
    }

    return KERNEL_SUCCESS_CODE;
}

int CommunicatorImplLite::UpdateComm(HcclKernelParamLite* kernelParam)
{
    if (!isSuspended) {
        HCCL_ERROR("CommunicatorImplLite is not suspended");
        return KERNEL_ERROR_CODE;
    }
    try {
        // 设定devType，初始化能力，算法及其他模块通过Get获取能力
        DevCapability::GetInstance().Init(kernelParam->comm.devType);
        UpdateTransports(kernelParam);

        auto id = GetHostDeviceSyncNotifyLiteMgr()->GetDeviceWaitNotify()->GetId();
        CHECK_NULLPTR(streamLiteMgr->GetMaster(), "[UpdateComm] master stream is nullptr!");
        streamLiteMgr->GetMaster()->GetRtsq()->NotifyWait(id);
        id = GetHostDeviceSyncNotifyLiteMgr()->GetHostWaitNotify()->GetId();
        streamLiteMgr->GetMaster()->GetRtsq()->NotifyRecordLoc(id);
        streamLiteMgr->GetMaster()->GetRtsq()->LaunchTask();
        HCCL_INFO("[NsRecovery] UpdateComm: task launched.");
    } catch (HcclException& e) {
        HCCL_ERROR("CommunicatorImplLite::UpdateComm Hccl exception %s was caught.", e.what());
        return KERNEL_ERROR_CODE;
    } catch (std::exception& e) {
        HCCL_ERROR("CommunicatorImplLite::UpdateComm Std exception %s was caught.", e.what());
        return KERNEL_ERROR_CODE;
    } catch (...) {
        HCCL_ERROR("CommunicatorImplLite::UpdateComm Some unknown error occurred.");
        return KERNEL_ERROR_CODE;
    }
    isSuspended = false;
    return KERNEL_SUCCESS_CODE;
}

std::shared_ptr<InsQueue> CommunicatorImplLite::GetInsQueue(HcclKernelParamLite* kernelParam)
{
    if (kernelParam->oneSidedComm) {
        HCCL_INFO("CommunicatorImplLite::GetInsQueue oneSidedComm begin");
        CreateOneSidedComponentLite();
        return GetOneSidedInsQueue(kernelParam);
    }

    CreateCollAlgComponentLite();
    HCCL_INFO("CommunicatorImplLite::GetInsQueue begin kernelParam->algName = %s", kernelParam->algName);
    std::shared_ptr<InsQueue> queue = std::make_shared<InsQueue>();
    auto it = algTopoInfoMap.find(kernelParam->tagKey);
    auto ret = algComponentLite->Orchestrate(kernelParam->op.algOperator, kernelParam->algName, it->second, queue);
    if (ret == HCCL_E_PARA) {
        return nullptr;
    }
    return queue;
}

void CommunicatorImplLite::CreateCollAlgComponentLite()
{
    if (algComponentLite.get() == nullptr) {
        algComponentLite = make_unique<CollAlgComponentLite>(
            myRank, rankSize, devType, scratchSize, connectedLinkMgr.get(), rmtDataBufferMgr.get());
        HCCL_INFO("CommunicatorImplLite::CreateCollAlgComponentLite is null");
    } else {
        algComponentLite->UpdateScratchBufferSize(scratchSize);
        HCCL_INFO("CommunicatorImplLite::CreateCollAlgComponentLite, bufferSize %llu", scratchSize);
    }
}

void CommunicatorImplLite::UnfoldOp(HcclKernelParamLite* kernelParam)
{
    opIndex_ = kernelParam->comm.opIndex_;
    uint64_t beginTime = ProfGetCurCpuTimestamp();
    profilingReporterLite->UpdateProfStat();
    UpdateCommParam(kernelParam);
    UpdateLocBuffer(kernelParam);
    UpdateUserStreamId(kernelParam);
    UpdateRes(kernelParam);
    UpdateDynamicOpData(kernelParam);
    SetDfxOpInfo(beginTime);

    UpdateHDCommnicate(kernelParam);
    RegisterRtsqCallback();

    isCommReady = true;
    HCCL_INFO("CommunicatorImplLite::UnfoldOpBase isCommReady is set to true.");
    std::shared_ptr<InsQueue> insQueue = GetInsQueue(kernelParam);
    if (insQueue == nullptr) {
        THROW<NullPtrException>(StringFormat("CommunicatorImplLite::UnfoldOpBase insQueue is nullptr."));
    }
    if (devType == DevType::DEV_TYPE_950 || devType == DevType::DEV_TYPE_960) {
        HCCL_INFO("CommunicatorImplLite::UnfoldOpBase DevType is DEV_TYPE_950 or DEV_TYPE_960.");
        insExecutor->ExecuteV82(*insQueue);
        profilingReporterLite->ReportAllTasks();
        ProfilingHandlerLite::GetInstance().ReportHcclOpInfo(*mirrorTaskMgrLite->GetCurrDfxOpInfo());
    } else if (devType == DevType::DEV_TYPE_910A2) {
        HCCL_INFO("CommunicatorImplLite::UnfoldOpBase DevType is DEV_TYPE_910A2.");
        insExecutor->Execute(*insQueue);
    } else {
        HCCL_WARNING("CommunicatorImplLite::UnfoldOpBase DevType is not support.");
    }
    kernelParam->op.algOperator.scratchMem = nullptr;
}

void CommunicatorImplLite::RegisterRtsqCallback()
{
    auto checkOpExecStatusCallback = [this]() {
        this->CheckOpExecStatus();
    };
    CHECK_NULLPTR(streamLiteMgr->GetMaster(), "[RegisterRtsqCallback]master stream is nullptr!");
    streamLiteMgr->GetMaster()->GetRtsq()->SetOpExecStatusCallback(checkOpExecStatusCallback);
    for (u32 i = 0; i < streamLiteMgr->SizeOfSlaves(); ++i) {
        streamLiteMgr->GetSlave(i)->GetRtsq()->SetOpExecStatusCallback(checkOpExecStatusCallback);
    }
}

void CommunicatorImplLite::CheckOpExecStatus() const
{
    if (isSuspended) {
        HCCL_INFO("hccl aicpu stop wait finish, for recv stop launch cmd");
        THROW<SuspendingException>(
            StringFormat("[CheckOpExecStatus] recv stop launch command, coll service is suspended."));
    }
}

void CommunicatorImplLite::UpdateCommParam(HcclKernelParamLite* kernelParam)
{
    if (isUpdateComm) {
        return;
    }
    myRank = kernelParam->comm.myRank;
    rankSize = kernelParam->comm.rankSize;
    devPhyId = kernelParam->comm.devPhyId;
    devType = kernelParam->comm.devType;
    opCounterAddr = kernelParam->comm.opCounterAddr;
    hcclExecTimeout = kernelParam->envConfig.hcclExecTimeout;
    if (rmtDataBufferMgr == nullptr) {
        collAlgInfo = std::make_unique<CollAlgInfo>(kernelParam->op.algOperator.opMode, kernelParam->opTag);
        rmtDataBufferMgr = std::make_unique<RmtDataBufferMgr>(transportLiteMgr.get(), collAlgInfo.get());
    }
    commId = kernelParam->comm.commId;
    HCCL_INFO(
        "CommunicatorImplLite::UpdateCommParam myRank [%u] rankSize[%u] devPhyId[%u] devType[%d] scratchSize [%llu] "
        "scratchaddress[%llx] opCounterAddr[%llx] commId[%s]",
        myRank, rankSize, devPhyId, devType, scratchSize, kernelParam->comm.opBaseScratch.addr, opCounterAddr,
        commId.c_str());
    isUpdateComm = true;
}

void CommunicatorImplLite::UpdateLocBuffer(HcclKernelParamLite* kernelParam)
{
    locBuffer[BufferType::INPUT] = kernelParam->op.input.addr;
    locBuffer[BufferType::OUTPUT] = kernelParam->op.output.addr;

    rmaBufferLiteVec.clear();
    rmaBufferLiteVec.resize(BufferType::__COUNT__);

    InitRmaBufferLite(kernelParam->op.input, BufferType::INPUT);
    InitRmaBufferLite(kernelParam->op.output, BufferType::OUTPUT);

    if (kernelParam->op.algOperator.opMode == OpMode::OPBASE) {
        scratchSize = kernelParam->comm.opBaseScratch.size;
        locBuffer[BufferType::SCRATCH] = kernelParam->comm.opBaseScratch.addr;
        InitRmaBufferLite(kernelParam->comm.opBaseScratch, BufferType::SCRATCH);
    } else {
        scratchSize = kernelParam->op.scratch.size;
        locBuffer[BufferType::SCRATCH] = kernelParam->op.scratch.addr;
        InitRmaBufferLite(kernelParam->op.scratch, BufferType::SCRATCH);
    }

    if (kernelParam->oneSidedComm) {
        if ((kernelParam->op.batchPutGetLocalAddr == nullptr) || (kernelParam->op.batchPutGetRemoteAddr == nullptr)) {
            THROW<InternalException>("batchPutGetAddr is nullptr");
        }
    }

    InitCurrentOp(kernelParam);
    HCCL_INFO(
        "CommunicatorImplLite::UpdateLocBuffer locBuffer[BufferType::INPUT] %llx, locBuffer[BufferType::OUTPUT] %llx",
        locBuffer[BufferType::INPUT], locBuffer[BufferType::OUTPUT]);
    HCCL_INFO(
        "CommunicatorImplLite::UpdateLocBuffer locBuffer[BufferType::SCRATCH] %llx", locBuffer[BufferType::SCRATCH]);
}

void CommunicatorImplLite::UpdateTransports(HcclKernelParamLite* kernelParam)
{
    HCCL_INFO("[NsRecovery] RestoreAllTransports start");
    RestoreAllTransports(kernelParam->binaryResAddr, kernelParam->binaryResSize);
    HCCL_INFO("[NsRecovery] RestoreAllTransports end");
}

void CommunicatorImplLite::RestoreAllTransports(u64 addr, u64 bufSize)
{
    std::vector<char> data;
    data.resize(bufSize);
    int ret = memcpy_s(data.data(), bufSize, reinterpret_cast<void*>(addr), bufSize);
    if (ret != 0) {
        THROW<InternalException>(
            StringFormat("[NsRecovery] CommunicatorImplLite::RestoreAllTransports: memcpy_s failed, ret = %d", ret));
    }
    HCCL_INFO(
        "[NsRecovery] CommunicatorImplLite::RestoreAllTransports: RestoreData %s",
        Bytes2hex(data.data(), data.size()).c_str());
    AicpuResPackageHelper helper;
    auto dataVec = helper.ParsePackedData(data);

    AicpuResMgrType resType = AicpuResMgrType::TRANSPORT;
    GetTransportLiteMgr()->ParseAllPackedData(dataVec[resType].data);
    HCCL_INFO("[NsRecovery] CommunicatorImplLite::RestoreAllTransports: GetResMgr %s Data", resType.Describe().c_str());
}

bool CommunicatorImplLite::CheckNeedUpdateRes(HcclKernelParamLite* kernelParam)
{
    std::string tagKey = kernelParam->tagKey;
    auto it = loadedOpSet.find(tagKey);
    if (it != loadedOpSet.end()) {
        HCCL_INFO("[CheckNeedUpdateRes] Corresponding resources of tag[%s] have been loaded", tagKey.c_str());
        return false;
    }
    loadedOpSet.insert(tagKey);
    return true;
}

void CommunicatorImplLite::UpdateRes(HcclKernelParamLite* kernelParam)
{
    if (CheckNeedUpdateRes(kernelParam)) {
        HCCL_INFO("[UpdateRes] start, opMode[%s]", kernelParam->op.algOperator.opMode.Describe().c_str());
        RestoreOpRes(kernelParam->opTag, kernelParam->tagKey, kernelParam->binaryResAddr, kernelParam->binaryResSize);
        HCCL_INFO("[UpdateRes] end");
    }
}

void CommunicatorImplLite::UpdateHDCommnicate(HcclKernelParamLite* kernelParam)
{
    CHK_RET_THROW(
        InternalException,
        StringFormat("[CommunicatorImplLite][%s] failed to init kfcControlTransferH2DParams", __func__),
        kfcControlTransferH2D->Init(kernelParam->kfcControlTransferH2DParams));
    CHK_RET_THROW(
        InternalException,
        StringFormat("[CommunicatorImplLite][%s] failed to init kfcControlTransferD2HParams", __func__),
        kfcStatusTransferD2H->Init(kernelParam->kfcControlTransferD2HParams));
    std::unique_lock<std::mutex> lock(hdcShmLock_);
    hdcHandler = make_unique<AicpuHdcHandler>(*kfcControlTransferH2D, *kfcStatusTransferD2H);
}

void CommunicatorImplLite::UpdateDynamicOpData(HcclKernelParamLite* kernelParam) const
{
    HCCL_INFO(
        "[CommunicatorImplLite][UpdateDynamicOpData] OpType[%s]",
        kernelParam->op.algOperator.opType.Describe().c_str());
    u8* dynamicDataPtr = reinterpret_cast<u8*>(kernelParam) + sizeof(struct HcclKernelParamLite);
    if (kernelParam->op.algOperator.opType == OpType::BATCHSENDRECV) {
        struct BatchSendRecvDataDes* batchSendRecvDataPtr
            = reinterpret_cast<struct BatchSendRecvDataDes*>(dynamicDataPtr);
        kernelParam->op.algOperator.batchSendRecvDataDes.itemNum = batchSendRecvDataPtr->itemNum;
        kernelParam->op.algOperator.batchSendRecvDataDes.sendRecvItemsPtr = batchSendRecvDataPtr->batchSendRecvItem;
        HcclSendRecvItem* itemPtr
            = reinterpret_cast<HcclSendRecvItem*>(kernelParam->op.algOperator.batchSendRecvDataDes.sendRecvItemsPtr);
        for (u32 i = 0; i < kernelParam->op.algOperator.batchSendRecvDataDes.itemNum; i++) {
            HCCL_INFO(
                "[CommunicatorImplLite][UpdateDynamicOpData] batchSendRecvDataDes remoteRank[%u]",
                (itemPtr + i)->remoteRank);
        }
    } else if (kernelParam->op.algOperator.opType == OpType::ALLTOALLV) {
        struct AllToAllvDataDes* alltoallvDataPtr = reinterpret_cast<struct AllToAllvDataDes*>(dynamicDataPtr);
        kernelParam->op.algOperator.all2AllVDataDes.sendType = static_cast<DataType::Value>(alltoallvDataPtr->sendType);
        kernelParam->op.algOperator.all2AllVDataDes.recvType = static_cast<DataType::Value>(alltoallvDataPtr->recvType);
        u64 rankSize = kernelParam->comm.rankSize;
        kernelParam->op.algOperator.all2AllVDataDes.sendCounts = static_cast<void*>(alltoallvDataPtr->sendRecvInfos);
        kernelParam->op.algOperator.all2AllVDataDes.recvCounts
            = static_cast<void*>(static_cast<u64*>(alltoallvDataPtr->sendRecvInfos) + rankSize);
        kernelParam->op.algOperator.all2AllVDataDes.sdispls = static_cast<void*>(
            static_cast<u64*>(alltoallvDataPtr->sendRecvInfos) + ALLTOALLV_DATA_INDEX_2 * rankSize);
        kernelParam->op.algOperator.all2AllVDataDes.rdispls = static_cast<void*>(
            static_cast<u64*>(alltoallvDataPtr->sendRecvInfos) + ALLTOALLV_DATA_INDEX_3 * rankSize);
        for (u32 i = 0; i < rankSize; i++) {
            HCCL_INFO(
                "[CommunicatorImplLite][UpdateDynamicOpData] alltoallv sendCounts[%llu], recvCounts[%llu]",
                *(static_cast<const u64*>(kernelParam->op.algOperator.all2AllVDataDes.sendCounts) + i),
                *(static_cast<const u64*>(kernelParam->op.algOperator.all2AllVDataDes.recvCounts) + i));
        }
    } else if (kernelParam->op.algOperator.opType == OpType::ALLTOALLVC) {
        struct AllToAllvcDataDes* alltoallvcDataPtr = reinterpret_cast<struct AllToAllvcDataDes*>(dynamicDataPtr);
        kernelParam->op.algOperator.all2AllVCDataDes.sendType
            = static_cast<DataType::Value>(alltoallvcDataPtr->sendType);
        kernelParam->op.algOperator.all2AllVCDataDes.recvType
            = static_cast<DataType::Value>(alltoallvcDataPtr->recvType);
        kernelParam->op.algOperator.all2AllVCDataDes.sendCountMatrix
            = static_cast<void*>(alltoallvcDataPtr->sendCountMatrix);
    }
}

HostDeviceSyncNotifyLiteMgr* CommunicatorImplLite::GetHostDeviceSyncNotifyLiteMgr()
{
    return hostDeviceSyncNotifyLiteMgr.get();
}

StreamLiteMgr* CommunicatorImplLite::GetStreamLiteMgr() { return streamLiteMgr.get(); }

QueueNotifyLiteMgr* CommunicatorImplLite::GetQueueNotifyLiteMgr() { return queueNotifyLiteMgr.get(); }

Cnt1tonNotifyLiteMgr* CommunicatorImplLite::GetCnt1tonNotifyLiteMgr() { return cnt1tonNotifyLiteMgr.get(); }

CntNto1NotifyLiteMgr* CommunicatorImplLite::GetCntNto1NotifyLiteMgr() { return cntNto1NotifyLiteMgr.get(); }

ConnectedLinkMgr* CommunicatorImplLite::GetConnectedLinkMgr() { return connectedLinkMgr.get(); }

DevId CommunicatorImplLite::GetDevPhyId() { return devPhyId; }

u32 CommunicatorImplLite::GetExecTimeOut() { return hcclExecTimeout; }

KfcCommand CommunicatorImplLite::BackGroundGetCmd()
{
    std::unique_lock<std::mutex> lock(hdcShmLock_);
    return hdcHandler->GetKfcCommand();
}

void CommunicatorImplLite::BackGroundSetStatus(KfcStatus status, KfcErrType errorCode)
{
    std::unique_lock<std::mutex> lock(hdcShmLock_);
    hdcHandler->SetKfcExecStatus(status, errorCode);
}

// 从 buffer中解析出算子需要的信息 ，对应 Host侧的 PackOpData
void CommunicatorImplLite::RestoreOpRes(const string& opTag, const string& tagKey, u64 addr, u64 bufSize)
{
    std::vector<char> data;
    data.resize(bufSize);
    (void)memcpy_s(data.data(), bufSize, reinterpret_cast<void*>(addr), bufSize);
    AicpuResPackageHelper helper;
    auto dataVec = helper.ParsePackedData(data);

    AicpuResMgrType resType = AicpuResMgrType::ALG_COMP_INFO;
    CreateCollAlgComponentLite();
    if (dataVec[resType].data.size() != 0) {
        algComponentLite->ParsePackedData(dataVec[resType].data);
    }

    resType = AicpuResMgrType::STREAM;
    GetStreamLiteMgr()->ParsePackedData(dataVec[resType].data);
    HCCL_INFO(
        "CommunicatorImplLite::RestoreOpRes: opTag %s GetResMgr %s Data", opTag.c_str(), resType.Describe().c_str());

    resType = AicpuResMgrType::QUEUE_NOTIFY;
    GetQueueNotifyLiteMgr()->ParsePackedData(dataVec[resType].data);
    HCCL_INFO(
        "CommunicatorImplLite::RestoreOpRes: opTag %s GetResMgr %s Data", opTag.c_str(), resType.Describe().c_str());

    resType = AicpuResMgrType::QUEUE_WAIT_GROUP_CNT_NOTIFY;
    GetCntNto1NotifyLiteMgr()->ParsePackedData(dataVec[resType].data);
    HCCL_INFO(
        "CommunicatorImplLite::RestoreOpRes: opTag %s GetResMgr %s Data", opTag.c_str(), resType.Describe().c_str());

    resType = AicpuResMgrType::QUEUE_BCAST_POST_CNT_NOTIFY;
    GetCnt1tonNotifyLiteMgr()->ParsePackedData(dataVec[resType].data);
    HCCL_INFO(
        "CommunicatorImplLite::RestoreOpRes: opTag %s GetResMgr %s Data", opTag.c_str(), resType.Describe().c_str());

    resType = AicpuResMgrType::HOST_DEV_SYNC_NOTIFY;
    GetHostDeviceSyncNotifyLiteMgr()->ParsePackedData(dataVec[resType].data);
    HCCL_INFO(
        "CommunicatorImplLite::RestoreOpRes: opTag %s GetResMgr %s Data", opTag.c_str(), resType.Describe().c_str());

    resType = AicpuResMgrType::TRANSPORT;
    if (currentOp.opMode == OpMode::OPBASE) { // 单算子模式
        GetTransportLiteMgr()->ParseOpbasePackedData(dataVec[resType].data);
    } else if (currentOp.opMode == OpMode::OFFLOAD) { // 图下沉模式
        GetTransportLiteMgr()->ParseOffloadPackedData(opTag, dataVec[resType].data);
    } else {
        THROW<InternalException>(StringFormat("opMode=%s failed", currentOp.opMode.Describe().c_str()));
    }
    HCCL_INFO(
        "CommunicatorImplLite::RestoreOpRes: opTag %s GetResMgr %s Data, %s", opTag.c_str(), resType.Describe().c_str(),
        currentOp.opMode.Describe().c_str());

    resType = AicpuResMgrType::ALG_TOPO;
    AlgTopoPackageHelper algTopoHelper;
    algTopoInfoMap[tagKey] = algTopoHelper.GetAlgTopoInfo(dataVec[resType].data);
    HCCL_INFO(
        "CommunicatorImplLite::RestoreOpRes: opTag %s GetResMgr %s Data, tagKey=%s", opTag.c_str(),
        resType.Describe().c_str(), tagKey.c_str());

    resType = AicpuResMgrType::CONNECTD_MGR;
    GetConnectedLinkMgr()->ParsePackedData(dataVec[resType].data);
    HCCL_INFO(
        "CommunicatorImplLite::RestoreOpRes: opTag %s GetResMgr %s Data", opTag.c_str(), resType.Describe().c_str());
}

CommunicatorImplLite::CommunicatorImplLite(u32 idIndex) : idIndex_(idIndex) {}

void CommunicatorImplLite::InitRmaBufferLite(HcclAicpuLocBufLite& bufLite, BufferType type)
{
    rmaBufferLiteVec[type]
        = std::make_unique<RmaBufferLite>(bufLite.addr, bufLite.size, bufLite.tokenId, bufLite.tokenValue);
}

void CommunicatorImplLite::InitCurrentOp(HcclKernelParamLite* kernelParam)
{
    currentOp.opTag = kernelParam->opTag; // opTag的赋值
    HCCL_INFO("CommunicatorImplLite::InitCurrentOp opTag[%s]", currentOp.opTag.c_str());

    currentOp.opMode = kernelParam->op.algOperator.opMode;
    currentOp.opType = kernelParam->op.algOperator.opType;
    currentOp.reduceOp = kernelParam->op.algOperator.reduceOp;
    currentOp.dataType = kernelParam->op.algOperator.dataType;
    currentOp.outputDataType = kernelParam->op.algOperator.outputDataType;
    currentOp.dataCount = kernelParam->op.algOperator.dataCount;
    currentOp.root = kernelParam->op.algOperator.root;
    currentOp.sendRecvRemoteRank = kernelParam->op.algOperator.sendRecvRemoteRank;

    if (kernelParam->op.algOperator.opType != OpType::BATCHSENDRECV) {
        currentOp.inputMem = std::make_shared<Buffer>(kernelParam->op.input.addr, kernelParam->op.input.size);
        currentOp.outputMem = std::make_shared<Buffer>(kernelParam->op.output.addr, kernelParam->op.output.size);
    }
    currentOp.scratchMem = std::make_shared<Buffer>(
        rmaBufferLiteVec[BufferType::SCRATCH]->GetAddr(), rmaBufferLiteVec[BufferType::SCRATCH]->GetSize());
    kernelParam->op.algOperator.scratchMem = currentOp.scratchMem;
    if (kernelParam->op.algOperator.scratchMem != nullptr) {
        HCCL_INFO(
            "CommunicatorImplLite::InitCurrentOp scratchMem addr %llx, size %llu",
            kernelParam->op.algOperator.scratchMem->GetAddr(), kernelParam->op.algOperator.scratchMem->GetSize());
    }
    HCCL_INFO("CommunicatorImplLite::InitCurrentOp end");
}

constexpr u32 TAILADDR_OFFSET_MULTIPLIER = 2;
void CommunicatorImplLite::SetDfxOpInfo(uint64_t beginTime)
{
    u64 size = 4;
    auto dfxopInfo = std::make_shared<DfxOpInfo>();
    dfxopInfo->op_ = currentOp;
    dfxopInfo->tag_ = currentOp.opTag;
    dfxopInfo->algType_ = AlgType{AlgType::MESH}.Describe();
    dfxopInfo->commIndex_ = idIndex_;
    dfxopInfo->beginTime_ = beginTime;
    dfxopInfo->comm_ = this;
    dfxopInfo->commId_ = commId;
    dfxopInfo->opIndex_ = opIndex_;
    dfxopInfo->headOpCounterAddr_ = opCounterAddr + size;
    dfxopInfo->tailOpCounterAddr_ = opCounterAddr + size * TAILADDR_OFFSET_MULTIPLIER;
    CHECK_NULLPTR(streamLiteMgr->GetMaster(), "[SetDfxOpInfo]master stream is nullptr!");
    mirrorTaskMgrLite->SetCurrDfxOpInfo(dfxopInfo);
}

void CommunicatorImplLite::CreateOneSidedComponentLite()
{
    if (oneSidedComponentLite.get() == nullptr) {
        oneSidedComponentLite = make_unique<OneSidedComponentLite>(
            myRank, rankSize, devType, scratchSize, connectedLinkMgr.get(), rmtDataBufferMgr.get());
        HCCL_INFO("CommunicatorImplLite::CreateOneSidedComponentLite is null");
    }
}

std::shared_ptr<InsQueue> CommunicatorImplLite::GetOneSidedInsQueue(HcclKernelParamLite* kernelParam)
{
    HCCL_INFO("CommunicatorImplLite::GetOneSidedInsQueue begin kernelParam->algName = %s", kernelParam->algName);
    std::shared_ptr<InsQueue> queue = std::make_shared<InsQueue>();

    auto ret = oneSidedComponentLite->Orchestrate(kernelParam->op, queue);
    if (ret == HCCL_E_PARA) {
        HCCL_ERROR("CommunicatorImplLite::GetOneSidedInsQueue ret[HCCL_E_PARA]");
        THROW<InternalException>(StringFormat("CommunicatorImplLite::GetOneSidedInsQueue ret[HCCL_E_PARA]"));
    }
    return queue;
}

HcclResult CommunicatorImplLite::SendErrorMessageReportToHost(ErrorMessageReport& errMsgInfo)
{
    if (kfcStatusTransferD2H == nullptr) {
        return HCCL_E_PTR;
    }
    CHK_RET(kfcStatusTransferD2H->Put(
        sizeof(KfcStatus) + sizeof(KfcErrType), sizeof(errMsgInfo), reinterpret_cast<uint8_t*>(&errMsgInfo)));

    return HCCL_SUCCESS;
}

void CommunicatorImplLite::UpdateUserStreamId(HcclKernelParamLite* kernelParam)
{
    userStreamId_ = kernelParam->op.userStreamId;
}

} // namespace Hccl
