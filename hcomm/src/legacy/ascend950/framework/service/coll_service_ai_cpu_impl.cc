/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_service_ai_cpu_impl.h"
#include <memory>
#include "communicator_impl.h"
#include "internal_exception.h"
#include "env_config_v2.h"
#include "stl_util.h"
#include "stream_utils.h"

#include "aicpu_res_package_helper.h"
#include "alg_topo_package_helper.h"
#include "dlprof_function_v2.h"
#include "task_exception_handler.h"
#include "exception_util.h"
#include "runtime_api_exception.h"

namespace Hccl {

template <class T, class U>
u16 CalcFieldOffset(T* target, U* base)
{
    return static_cast<u16>(reinterpret_cast<const char*>(target) - reinterpret_cast<const char*>(base));
}

void CollServiceAiCpuImpl::Init()
{
    // 算子执行次数计数器buffer申请
    AddOpCounterMems();
}

static std::string GetTagKey(CollOperator& op, std::string algName, u32 bsrRemoteRanksHashValue)
{
    std::string tmp{};
    tmp = (op.opMode == OpMode::OPBASE) ? algName : op.opTag;
    if (op.opType == OpType::BATCHSENDRECV) {
        tmp = tmp + std::to_string(bsrRemoteRanksHashValue);
    } else if (op.opType == OpType::SEND || op.opType == OpType::RECV) {
        tmp = tmp + std::to_string(op.sendRecvRemoteRank);
    }
    return tmp;
}

DevBuffer* CollServiceAiCpuImpl::OpBasedCollProcess(CollOperator& op, const std::string& algName)
{
    auto req = comm->GetCollAlgComponent()->GetCollAlgOpReq(op, algName);
    HCCL_INFO(
        "CollServiceAiCpuImpl::OpBasedCollProcess GetCollAlgOpReq OrchestMode::INSTRUCTION, algName %s",
        req.algName.c_str());

    for (auto& it : req.resReq.levelRankPairs) {
        HCCL_INFO("CollServiceAiCpuImpl::level=%u, rank=%d", it.first, it.second);
    }

    HCCL_INFO("CollServiceAiCpuImpl::OpBasedCollProcess req.resReq.primQueueNum =%u", req.resReq.primQueueNum);
    HCCL_INFO(
        "CollServiceAiCpuImpl::OpBasedCollProcess req.resReq.queueNotifys size =%zu", req.resReq.queueNotifys.size());
    HCCL_INFO(
        "CollServiceAiCpuImpl::OpBasedCollProcess req.resReq.localBcastPostCntNotify size =%zu",
        req.resReq.localBcastPostCntNotify.size());
    HCCL_INFO(
        "CollServiceAiCpuImpl::OpBasedCollProcess req.resReq.localWaitGroupCntNotify size =%zu",
        req.resReq.localWaitGroupCntNotify.size());

    AllocWorkStream(req.resReq.primQueueNum);
    AllocQueueNotify(req.resReq.queueNotifys);
    AllocBcastPostCntNotify(req.resReq.localBcastPostCntNotify);
    AllocWaitGroupCntNotify(req.resReq.localWaitGroupCntNotify);

    HCCL_INFO("CollServiceAiCpuImpl::OpBasedCollProcess req.resReq.links size =%zu", req.resReq.links.size());
    HCCL_INFO("CollServiceAiCpuImpl::OpBasedCollProcess op.opTag %s", op.opTag.c_str());

    RegisterCclBuffer(req.resReq.links);

    // Socket建链
    comm->GetSocketManager().BatchCreateSockets(req.resReq.links);
    // 建立RmaConnection并建链
    auto connBuilderPair = connectionsBuilders.emplace(comm->GetId(), make_unique<ConnectionsBuilder>(*comm));
    connBuilderPair.first->second->BatchBuild(comm->GetId(), req.resReq.links);

    AllocNotifies(req.resReq.links);

    if (op.opMode == OpMode::OPBASE) {
        comm->GetMemTransportManager()->BatchBuildOpbasedTransports(req.resReq.links);
        WaitOpbasedTransportReady();
    } else if (op.opMode == OpMode::OFFLOAD) {
        comm->GetMemTransportManager()->BatchBuildOffloadTransports(op.opTag, req.resReq.links);
        WaitOffloadTransportReady(op.opTag);
    }

    u32 bsrRemoteRanksHashValue = 0;
    if (op.opType == OpType::BATCHSENDRECV) {
        bsrRemoteRanksHashValue = GetRemoteRankIdsHashValue(op);
    }

    curTagKey = GetTagKey(op, req.algName, bsrRemoteRanksHashValue);

    auto it = collOpLoadedMap.find(curTagKey);
    if (it != collOpLoadedMap.end()) { // 已经向Device Mem写过资源
        HCCL_INFO("[OpBasedCollProcess] tag[%s] devMem has been allocated, reuse it", curTagKey.c_str());
        return it->second.get();
    }

    auto buffer = PackOpData(op.opTag, req);
    shared_ptr<DevBuffer> devMem = make_shared<DevBuffer>(buffer.size()); // 申请device内存
    TRY_CATCH_THROW(
        InternalException, StringFormat("HrtMemcpy to device memory failed, size[%zu]", buffer.size()),
        HrtMemcpy(
            reinterpret_cast<void*>(devMem->GetAddr()), devMem->GetSize(), buffer.data(), buffer.size(),
            RT_MEMCPY_HOST_TO_DEVICE)); // H2D拷贝，将资源拷贝到device内存

    collOpLoadedMap[curTagKey] = devMem;
    return devMem.get();
}

void CollServiceAiCpuImpl::LoadWithOpBasedModeNoRegister(CollOperator& op)
{
    RegisterOpbasedLocalRmaBuf(op.opTag);

    comm->GetAicpuStreamManager().AllocFreeStream();
    Stream* lanchStream = comm->GetAicpuStreamManager().GetFreeStream();
    comm->GetAicpuStreamManager().AclGraphCaptureFreeStream(comm->GetStreamManager().opbase->GetMaster());
    DevBuffer* mem = nullptr;
    comm->SetCommStatus(CommStatus::COMM_BUILDING);
    mem = OpBasedCollProcess(op, comm->GetCurAlgName());
    std::vector<Stream*>& stream_pointers = comm->GetAicpuStreamManager().GetStreams();
    comm->ReportHcclMC2Info(
        *lanchStream, *comm->GetStreamManager().opbase->GetMaster(), stream_pointers); // 上报MC2信息
    AllocOpMem(op);

    SaveMirrorDfxOpInfo();
    AicpuKernelEntranceLaunch(*comm->GetStreamManager().opbase->GetMaster(), op, comm->GetCurAlgName(), mem);
}

void CollServiceAiCpuImpl::LoadWithOpBasedMode(CollOperator& op, unique_ptr<Stream> stream)
{
    RegisterOpBufToBufMgr(op);
    RegisterOpbasedStream(std::move(stream));

    LoadWithOpBasedModeNoRegister(op);
}

void CollServiceAiCpuImpl::LoadWithOffloadModeNoRegister(CollOperator& op)
{
    RegisterOffloadLocalRmaBuf(op.opTag);

    // 将通讯域设置为transport建链中状态
    comm->SetCommStatus(CommStatus::COMM_BUILDING);

    DevBuffer* mem = nullptr;
    mem = OpBasedCollProcess(op, comm->GetCurAlgName());
    AllocOpMem(op);

    SaveMirrorDfxOpInfo();
    AicpuKernelEntranceLaunch(*comm->GetStreamManager().offload->GetMaster(op.opTag), op, comm->GetCurAlgName(), mem);
}

void CollServiceAiCpuImpl::LoadWithOffloadMode(CollOperator& op, std::unique_ptr<Stream> stream)
{
    RegisterOpBufToBufMgr(op);
    RegisterOffloadMasterStream(op.opTag, std::move(stream));

    LoadWithOffloadModeNoRegister(op);
}

HcclResult
CollServiceAiCpuImpl::AllocCollOpResourceNoRegister(CollOperator& op, const std::string& opAlgTag, void** addr)
{
    RegisterOpbasedLocalRmaBuf(op.opTag);
    comm->GetAicpuStreamManager().AllocFreeStream();
    DevBuffer* mem = nullptr;
    comm->SetCommStatus(CommStatus::COMM_BUILDING);
    mem = OpBasedCollProcess(op, comm->GetCurAlgName());
    CHK_RET(AicpuMc2CommResourcePrepare(op, comm->GetCurAlgName(), mem, opAlgTag, addr));
    return HCCL_SUCCESS;
}

HcclResult CollServiceAiCpuImpl::AllocCollOpResource(CollOperator& op, const std::string& opAlgTag, void** addr)
{
    auto iter = aicpuMc2CommResourceMap_.find(opAlgTag);
    if (iter != aicpuMc2CommResourceMap_.end()) {
        HCCL_INFO("[AllocCollOpResource] has existed comm %s, return for %p.", opAlgTag.c_str(), addr);
        shared_ptr<DevBuffer> oldDevBuf = iter->second;
        *addr = reinterpret_cast<void*>(oldDevBuf->GetAddr());
        return HCCL_SUCCESS;
    }
    RegisterOpBufToBufMgr(op);
    CHK_RET(AllocCollOpResourceNoRegister(op, opAlgTag, addr));
    return HCCL_SUCCESS;
}

HcclResult CollServiceAiCpuImpl::AicpuMc2CommResourcePrepare(
    const CollOperator& op, const string& algName, const DevBuffer* mem, const std::string& opAlgTag, void** addr)
{
    HCCL_INFO(
        "CollServiceAiCpuImpl::AicpuMc2CommResourcePrepare entry, algName: %s, opAlgTag: %s", algName.c_str(),
        opAlgTag.c_str());
    HcclKernelLaunchParam param{};
    s32 ret = strcpy_s(param.kernel.algName, sizeof(param.kernel.algName), algName.data());
    if (ret != EOK) {
        HCCL_ERROR(
            "CollServiceAiCpuImpl::AicpuMc2CommResourcePrepare, strcpy_s algName failed! ret: %d, algName: %s", ret,
            algName.c_str());
        return HCCL_E_INTERNAL;
    }

    ret = strcpy_s(param.kernel.opTag, sizeof(param.kernel.opTag), op.opTag.data());
    if (ret != EOK) {
        HCCL_ERROR(
            "CollServiceAiCpuImpl::AicpuMc2CommResourcePrepare, strcpy_s opTag failed! ret: %d, op.opTag: %s", ret,
            op.opTag.c_str());
        return HCCL_E_INTERNAL;
    }

    HCCL_INFO(
        "CollServiceAiCpuImpl::AicpuMc2CommResourcePrepare param.kernel.algName: %s, op.opTag: %s",
        param.kernel.algName, op.opTag.c_str());
    param.kernel.binaryResAddr = mem->GetAddr();
    param.kernel.binaryResSize = mem->GetSize();

    SetHcclKernelLaunchParam(param, comm, false);

    comm->SetAicpuKernelLaunched(true);
    comm->GetStreamManager().ResetSlaveIndex(0);

    shared_ptr<DevBuffer> newDevMem = make_shared<DevBuffer>(sizeof(HcclKernelParamLite));
    HrtMemcpy(
        reinterpret_cast<void*>(newDevMem->GetAddr()), sizeof(HcclKernelParamLite), static_cast<void*>(&param.kernel),
        sizeof(HcclKernelParamLite), RT_MEMCPY_HOST_TO_DEVICE);
    aicpuMc2CommResourceMap_.insert(make_pair(opAlgTag, newDevMem));
    *addr = reinterpret_cast<void*>(newDevMem->GetAddr());
    HCCL_INFO(
        "CollServiceAiCpuImpl::AicpuMc2CommResourcePrepare alloc %s kernel param success, set addr %p value is %p",
        opAlgTag.c_str(), addr, *addr);
    return HCCL_SUCCESS;
}

void InitAicpuLocBufLite(HcclAicpuLocBufLite& lite, u64 addr, u64 size, const string& desc)
{
    auto tokenPair = HrtUbDevQueryToken(addr, size);
    lite.addr = addr;
    lite.size = size;
    lite.tokenId = tokenPair.first;
    lite.tokenValue = tokenPair.second;
    HCCL_INFO("InitAicpuLocBufLite %s, addr=0x%llx, size=0x%llx", desc.c_str(), addr, size);
}

void CollServiceAiCpuImpl::SetOpbaseBufferParam(
    HcclKernelLaunchParam& param, CommunicatorImpl* comm, CollOperator& op) const
{
    auto buffer = comm->GetCclBuffer();
    param.kernel.comm.opBaseScratch.addr = buffer->GetAddr();
    param.kernel.comm.opBaseScratch.size = buffer->GetSize();
    InitAicpuLocBufLite(param.kernel.comm.opBaseScratch, buffer->GetAddr(), buffer->GetSize(), "opBaseScratch");
    if (op.inputMem != nullptr) {
        InitAicpuLocBufLite(param.kernel.op.input, op.inputMem->GetAddr(), op.inputMem->GetSize(), "inputMem");
    }

    if (op.outputMem != nullptr) {
        InitAicpuLocBufLite(param.kernel.op.output, op.outputMem->GetAddr(), op.outputMem->GetSize(), "outputMem");
    }
    HCCL_INFO(
        "SetOpbaseBufferParam param.kernel.comm.opBaseScratch.addr %llu, param.kernel.comm.opBaseScratch.size %llu",
        param.kernel.comm.opBaseScratch.addr, param.kernel.comm.opBaseScratch.size);
}

void CollServiceAiCpuImpl::SetOffloadBufferParam(
    HcclKernelLaunchParam& param, CommunicatorImpl* comm, CollOperator& op) const
{
    HCCL_INFO("SetOffloadBufferParam");
    auto offloadInput = comm->GetDataBufferManager().Get(op.opTag, BufferType::INPUT);
    if (offloadInput != nullptr) {
        InitAicpuLocBufLite(param.kernel.op.input, op.inputMem->GetAddr(), op.inputMem->GetSize(), "inputMem");
    }

    auto offloadOuput = comm->GetDataBufferManager().Get(op.opTag, BufferType::OUTPUT);
    if (offloadOuput != nullptr) {
        InitAicpuLocBufLite(param.kernel.op.output, op.outputMem->GetAddr(), op.outputMem->GetSize(), "outputMem");
    }

    auto offloadScartch = comm->GetDataBufferManager().Get(op.opTag, BufferType::SCRATCH);
    if (offloadScartch != nullptr) {
        InitAicpuLocBufLite(param.kernel.op.scratch, op.scratchMem->GetAddr(), op.scratchMem->GetSize(), "scratchMem");
    }
}

void CollServiceAiCpuImpl::SetHcclKernelLaunchParam(HcclKernelLaunchParam& param, CommunicatorImpl* comm, bool isLaunch)
{
    CollOperator op = *comm->GetCurrentCollOperator();

    param.kernel.comm.idIndex = comm->GetIdIndex();
    param.kernel.comm.myRank = comm->GetMyRank();
    param.kernel.comm.rankSize = comm->GetRankSize();
    param.kernel.comm.devType = comm->GetDevType();
    param.kernel.comm.devPhyId = comm->GetDevicePhyId();
    param.kernel.comm.opIndex_ = comm->GetOpIndex();
    param.kernel.comm.opCounterAddr = static_cast<u64>(counterBuf->GetAddr());
    auto ret = strcpy_s(param.kernel.comm.commId, sizeof(param.kernel.comm.commId), comm->GetId().data());
    if (ret != EOK) {
        THROW<InternalException>(
            StringFormat("CollServiceAiCpuImpl::SetHcclKernelLaunchParam, strcpy_s commId failed! ret[%d]", ret));
    }
    if (op.opMode == OpMode::OPBASE) {
        SetOpbaseBufferParam(param, comm, op);
    } else {
        SetOffloadBufferParam(param, comm, op);
    }

    param.kernel.op.algOperator.opMode = op.opMode;
    param.kernel.op.algOperator.opType = op.opType;
    param.kernel.op.algOperator.reduceOp = op.reduceOp;
    param.kernel.op.algOperator.dataType = op.dataType;
    param.kernel.op.algOperator.dataCount = op.dataCount;
    param.kernel.op.algOperator.root = op.root;
    HcclResult hcclRet = HCCL_SUCCESS;
    if (op.opType == OpType::ALLTOALL && isLaunch) {
        param.kernel.op.algOperator.all2AllDataDes = op.all2AllDataDes;
    } else if (op.opType == OpType::ALLTOALLV && isLaunch) {
        hcclRet = FillAllToAllvData(op);
    } else if (op.opType == OpType::ALLTOALLVC) {
        hcclRet = FillAllToAllvcData(op);
    } else if (op.opType == OpType::BATCHSENDRECV) {
        hcclRet = FillBatchSendRecvData(op);
    } else if (op.opType == OpType::SEND || op.opType == OpType::RECV) {
        param.kernel.op.algOperator.sendRecvRemoteRank = op.sendRecvRemoteRank;
    }
    if (hcclRet != HCCL_SUCCESS) {
        HCCL_ERROR("[CollServiceAiCpuImpl][SetHcclKernelLaunchParam] fill op data failed!");
        THROW<InternalException>(
            StringFormat("CollServiceAiCpuImpl::SetHcclKernelLaunchParam, fill op data failed! ret [%d]", hcclRet));
    }
    param.kernel.op.sendRecvRemoteRank = op.sendRecvRemoteRank;
    Stream* streamPtr = nullptr;
    if (op.opMode == OpMode::OPBASE) {
        streamPtr = comm->GetStreamManager().opbase->GetMaster();
    } else {
        streamPtr = comm->GetStreamManager().offload->GetMaster(op.opTag);
    }
    if (streamPtr != nullptr) {
        param.kernel.op.userStreamId = streamPtr->GetId();
    } else {
        HCCL_WARNING(
            "CollServiceAiCpuImpl::%s userStream is nullptr, userStreamId id in kernel param is invalid.", __func__);
    }
    param.kernel.kfcControlTransferH2DParams = comm->GetKfcControlTransferH2D().GetCommunicateParams();
    param.kernel.kfcControlTransferD2HParams = comm->GetKfcStatusTransferD2H().GetCommunicateParams();

    SetDeviceEnvConfigParam(param);
}

void CollServiceAiCpuImpl::SetDeviceEnvConfigParam(HcclKernelLaunchParam& param) const
{
    param.kernel.envConfig.hcclExecTimeout = EnvConfig::GetInstance().GetRtsConfig().GetExecTimeOut();
    param.kernel.envConfig.taskExceptionEnable
        = EnvConfig::GetInstance().GetLogConfig().GetDfsConfig().taskExceptionEnable;
}

void CollServiceAiCpuImpl::AicpuKernelEntranceLaunch(
    Stream& stream, const CollOperator& op, const string& algName, const DevBuffer* mem)
{
    HcclKernelLaunchParam param;

    s32 ret = strcpy_s(param.kernel.algName, sizeof(param.kernel.algName), algName.data());
    if (ret != EOK) {
        THROW<InternalException>(
            StringFormat("CollServiceAiCpuImpl::AicpuKernelEntranceLaunch, strcpy_s algName failed! ret[%d]", ret));
    }

    ret = strcpy_s(param.kernel.opTag, sizeof(param.kernel.opTag), op.opTag.data());
    if (ret != EOK) {
        THROW<InternalException>(
            StringFormat("CollServiceAiCpuImpl::AicpuKernelEntranceLaunch, strcpy_s opTag failed! ret[%d]", ret));
    }

    ret = strcpy_s(param.kernel.tagKey, sizeof(param.kernel.tagKey), curTagKey.data());
    if (ret != EOK) {
        THROW<InternalException>(
            StringFormat("CollServiceAiCpuImpl::AicpuKernelEntranceLaunch, strcpy_s tagKey failed! ret[%d]", ret));
    }

    HCCL_INFO(
        "CollServiceAiCpuImpl::AicpuKernelEntranceLaunch param.kernel.algName: %s, op.opTag %s", param.kernel.algName,
        op.opTag.c_str());

    param.kernel.binaryResAddr = mem->GetAddr();
    param.kernel.binaryResSize = mem->GetSize();

    SetHcclKernelLaunchParam(param, comm);
    AicpuKernelLaunch(param, stream, op.opMode);
    comm->SetAicpuKernelLaunched(true);

    comm->GetStreamManager().ResetSlaveIndex(0);
}

void CollServiceAiCpuImpl::AicpuKernelLaunch(HcclKernelLaunchParam& param, Stream& stream, OpMode opMode)
{
    param.kernel.op.userStreamId = stream.GetId();
    aclrtLaunchKernelCfg cfg;
    aclrtLaunchKernelAttr attr;
    attr.id = ACL_RT_LAUNCH_KERNEL_ATTR_TIMEOUT;
    auto timeoutCheck = EnvConfig::GetInstance().GetRtsConfig().GetExecTimeOut();
    attr.value.timeout
        = static_cast<u16>((timeoutCheck == 0) ? timeoutCheck : (timeoutCheck + 30)); // aicpu kernel超时时间: X+30s
    cfg.numAttrs = 1;
    cfg.attrs = &attr;
    TRY_CATCH_THROW(
        InternalException, std::string("HrtMemcpy kernelParamBuf failed"),
        HrtMemcpy(
            reinterpret_cast<void*>(kernelParamBuf_.get()->GetAddr()), sizeof(HcclKernelParamLite), &param,
            sizeof(HcclKernelParamLite), RT_MEMCPY_HOST_TO_HOST));

    HCCL_INFO("[CollServiceAiCpuImpl][%s] args timeout[%u]s", __func__, attr.value.timeout);

    AddPostToUserStream(stream);
    TaskParam taskParam{};
    taskParam.beginTime = DlProfFunction::GetInstance().dlMsprofSysCycleTime();

    HCCL_INFO(
        "[CollServiceAiCpuImpl::AicpuKernelLaunch] RegisterGetAicpuTaskExceptionCallBack streamId[%u], devLogicId[%u]",
        stream.GetId(), comm->GetDeviceLogicId());
    auto getAicpuTaskExceptionCallBack = [this]() {
        return this->comm->GetAicpuTaskException();
    };
    Hccl::RegisterGetAicpuTaskExceptionCallBackV2(
        stream.GetId(), comm->GetDeviceLogicId(), getAicpuTaskExceptionCallBack);

    HCCL_INFO(
        "[CollServiceAiCpuImpl][%s] param.soName: %s, param.kernelName: %s", __func__, param.soName, param.kernelName);
    const aclrtFuncHandle funcHandle = comm->GetAicpuKernelFuncHandle(param.kernelName);

    bool isCapture = false;
    rtModel_t rtModel = nullptr;
    CHK_RET_THROW(
        RuntimeApiException,
        StringFormat("[CollServiceAiCpuImpl][%s] GetStreamCaptureInfo fail, streamId[%u]", __func__, stream.GetId()),
        GetStreamCaptureInfo(stream.GetPtr(), rtModel, isCapture));
    Stream* mStreamPtr = nullptr;
    if (opMode == OpMode::OPBASE || isCapture) {
        comm->GetAicpuStreamManager().AllocFreeStream();
        mStreamPtr = comm->GetAicpuStreamManager().GetFreeStream();
        comm->GetAicpuStreamManager().AclGraphCaptureFreeStream(&stream);
    } else {
        mStreamPtr = &stream;
    }
    auto& mStream = *mStreamPtr;

    std::string mode = (opMode == OpMode::OPBASE) ? "OPBASE" : "OFFLOAD";
    constexpr u32 numBlocks = 1;
    HrtAicpuLaunchKernelWithHostArgs(
        funcHandle, numBlocks, mStream.GetPtr(), &cfg, reinterpret_cast<void*>(kernelParamBuf_.get()->GetAddr()),
        sizeof(HcclKernelParamLite) + dynamicDataSize);
    HCCL_INFO(
        "[AicpuKernelLauncher][AicpuKernelLaunch] param.kernel.algName: %s, %s mode, %s "
        "HrtAicpuLaunchKernelWithHostArgs end!",
        param.kernel.algName, mode.c_str(), mStream.Describe().c_str());
    taskParam.taskType = TaskParamType::TASK_AICPU_KERNEL;
    taskParam.endTime = DlProfFunction::GetInstance().dlMsprofSysCycleTime();

    SaveDfxTaskInfo(taskParam, -1, mStream.IsMaster());
    AddWaitToUserStream(stream);
}

constexpr u8 QUEUE_NOTIFY_POST_QID_POS = 0;
constexpr u8 QUEUE_NOTIFY_WAIT_QID_POS = 1;
constexpr u8 QUEUE_NOTIFY_TOPIC_ID_POS = 2;

void CollServiceAiCpuImpl::AllocQueueNotify(std::vector<std::tuple<QId, QId, u32>>& queueNotifyReq) const
{
    QueueNotifyManager& queueNotifyMgr = comm->GetAicpuQueueNotifyManager();

    std::for_each(queueNotifyReq.begin(), queueNotifyReq.end(), [&queueNotifyMgr](auto item) {
        queueNotifyMgr.ApplyFor(
            std::get<QUEUE_NOTIFY_POST_QID_POS>(item), std::get<QUEUE_NOTIFY_WAIT_QID_POS>(item),
            std::get<QUEUE_NOTIFY_TOPIC_ID_POS>(item));
    });
}

void CollServiceAiCpuImpl::AllocBcastPostCntNotify(std::vector<std::pair<QId, u32>>& bcastPostCntNotifyReq) const
{
    QueueBcastPostCntNotifyManager& bcastPostCntNotifyMgr = comm->GetBcastPostCntNotifyManager();

    std::for_each(bcastPostCntNotifyReq.begin(), bcastPostCntNotifyReq.end(), [&bcastPostCntNotifyMgr](auto item) {
        bcastPostCntNotifyMgr.ApplyFor(item.first, item.second);
        HCCL_INFO("[CollServiceAiCpuImpl][%s] qid[%u] topicId[%u]", __func__, item.first, item.second);
    });
}

void CollServiceAiCpuImpl::AllocWaitGroupCntNotify(std::vector<std::pair<QId, u32>>& waitGroupCntNotifyReq) const
{
    QueueWaitGroupCntNotifyManager& waitGroupCntNotifyMgr = comm->GetQueueWaitGroupCntNotifyManager();

    std::for_each(waitGroupCntNotifyReq.begin(), waitGroupCntNotifyReq.end(), [&waitGroupCntNotifyMgr](auto item) {
        waitGroupCntNotifyMgr.ApplyFor(item.first, item.second);
        HCCL_INFO("[CollServiceAiCpuImpl][%s] qid[%u] topicId[%u]", __func__, item.first, item.second);
    });
}

void CollServiceAiCpuImpl::AddPostToUserStream(const Stream& stream)
{
    TaskParam taskParam{};
    taskParam.beginTime = DlProfFunction::GetInstance().dlMsprofSysCycleTime();

    auto postNotify = comm->GetHostDeviceSyncNotifyManager().GetDeviceWaitNotify();
    HCCL_INFO(
        "[CollServiceAiCpuImpl][AddPostToUserStream] notify id[%u] DevPhyId[%u], streamId[%u]", postNotify->GetId(),
        postNotify->GetDevPhyId(), stream.GetId());
    postNotify->Post(stream);

    taskParam.taskType = TaskParamType::TASK_NOTIFY_RECORD;
    taskParam.endTime = DlProfFunction::GetInstance().dlMsprofSysCycleTime();
    taskParam.taskPara.Notify.notifyID = postNotify->GetId();
    taskParam.taskPara.Notify.value = 1;

    SaveDfxTaskInfo(taskParam, -1, stream.IsMaster());
}

void CollServiceAiCpuImpl::AddWaitToUserStream(const Stream& stream)
{
    TaskParam taskParam{};
    taskParam.beginTime = DlProfFunction::GetInstance().dlMsprofSysCycleTime();

    auto waitNotify = comm->GetHostDeviceSyncNotifyManager().GetHostWaitNotify();
    HCCL_INFO(
        "[CollServiceAiCpuImpl][AddWaitToUserStream] notify id[%u] DevPhyId[%u], streamId[%u]", waitNotify->GetId(),
        waitNotify->GetDevPhyId(), stream.GetId());

    auto timeoutCheck = EnvConfig::GetInstance().GetRtsConfig().GetExecTimeOut();
    u32 timeout = (timeoutCheck == 0) ? timeoutCheck : (timeoutCheck + 50); // 主流notifywait超时时间: X+50s
    HCCL_INFO("[CollServiceAiCpuImpl][%s] notify wait timeout[%u]s", __func__, timeout);
    waitNotify->Wait(stream, timeout);

    taskParam.taskType = TaskParamType::TASK_NOTIFY_WAIT;
    taskParam.endTime = DlProfFunction::GetInstance().dlMsprofSysCycleTime();
    taskParam.taskPara.Notify.notifyID = waitNotify->GetId();
    taskParam.taskPara.Notify.value = 1;
    SaveDfxTaskInfo(taskParam, -1, stream.IsMaster());
}

void CollServiceAiCpuImpl::AllocWorkStream(u32 primQueueNum) const
{
    comm->GetAicpuStreamManager().AllocStreams(primQueueNum);
}

CollServiceAiCpuImpl::CollServiceAiCpuImpl(CommunicatorImpl* comm) : CollServiceBase(comm) {}

void CollServiceAiCpuImpl::AllocNotifies(const vector<LinkData>& links)
{
    vector<LinkData> pendingLinks;
    for (auto& link : links) {
        if (Contain(availableLinks, link)) {
            continue;
        }
        pendingLinks.emplace_back(link);
    }
    HCCL_INFO(
        "CollServiceAiCpuImpl::AllocNotifies links size %zu, pendingLinks size %zu", links.size(), pendingLinks.size());
    if (pendingLinks.empty()) {
        return;
    }

    for (auto& link : pendingLinks) {
        // 待修改: 申请数量
        comm->GetConnLocalNotifyManager().ApplyFor(link.GetRemoteRankId(), link);
    }
    HCCL_INFO("[CollServiceAiCpuImpl][AllocNotifies] end");

    availableLinks.insert(pendingLinks.begin(), pendingLinks.end());
}

void CollServiceAiCpuImpl::AllocOpMem(const CollOperator& op)
{
    dynamicDataSize = CalcOpDynamicDataSize(op, op.opType, comm->GetRankSize());
    u64 kernelParamSize = sizeof(HcclKernelParamLite) + dynamicDataSize;
    if (kernelParamBuf_ == nullptr) {
        kernelParamBuf_ = make_shared<HostBuffer>(KERNEL_PARAM_BUF_SIZE);
        if (kernelParamBuf_ == nullptr) {
            HCCL_ERROR("[CollServiceAiCpuImpl][AllocOpMem] Alloc kernelParamBuf failed !");
            THROW<InternalException>(
                StringFormat("[CollServiceAiCpuImpl][AllocOpMem] Alloc kernelParamBuf failed len"));
        }
    }

    if (kernelParamBuf_ != nullptr && kernelParamSize > kernelParamBuf_.get()->GetSize()) {
        kernelParamBuf_ = make_shared<HostBuffer>(kernelParamSize);
        if (kernelParamBuf_ == nullptr) {
            HCCL_ERROR("[CollServiceAiCpuImpl][AllocOpMem] Alloc kernelParamBuf len[%llu] failed !", kernelParamSize);
            THROW<InternalException>(StringFormat(
                "[CollServiceAiCpuImpl][AllocOpMem] Alloc kernelParamBuf failed len[%llu]", kernelParamSize));
        }
    }
    HCCL_INFO("[AllocOpMem] op.opType[%d]", op.opType);
}

u64 CollServiceAiCpuImpl::CalcOpDynamicDataSize(const CollOperator& op, const OpType& opType, const u32& rankSize) const
{
    u64 dynamicDataSize = 0ULL;
    switch (opType) {
        case OpType::BATCHSENDRECV:
            dynamicDataSize
                = sizeof(struct BatchSendRecvDataDes) + op.batchSendRecvDataDes.itemNum * sizeof(HcclSendRecvItem);
            break;
        case OpType::ALLTOALLV:
            dynamicDataSize = sizeof(struct AllToAllvDataDes) + rankSize * ALLTOALL_INFO_SIZE * sizeof(u64);
            break;
        case OpType::ALLTOALLVC:
            dynamicDataSize = sizeof(struct AllToAllvcDataDes) + rankSize * rankSize * sizeof(u64);
            break;
        default:
            break;
    }
    HCCL_INFO("HcclCommunicator::CalcOpDynamicDataSize dynamicDataSize[%llu]", dynamicDataSize);
    return dynamicDataSize;
}

// 功能说明：根据输入的LinkData信息，恢复Transport对象
// 输入说明：vector<LinkData> &links：linkData数据
void CollServiceAiCpuImpl::RecoverTransport(
    vector<LinkData>& links, [[maybe_unused]] vector<std::pair<LinkGroup, u32>> linkGroupPair) // communicatorLinkData
{
    HCCL_INFO("[CollServiceAiCpuImpl][RecoverTransport] start");
    RegisterCclBuffer(links);
    // 创建TransPort所需的Socket
    comm->GetSocketManager().BatchCreateSockets(links);

    // 创建TransPort所需的RmaConnection
    auto connBuilderPair = connectionsBuilders.emplace(comm->GetId(), make_unique<ConnectionsBuilder>(*comm));
    connBuilderPair.first->second->BatchBuild(comm->GetId(), links);
    // 创建TransPort所需的Notify资源
    AllocNotifies(links);
    // 重新构造TransPort
    auto op = comm->GetCurrentCollOperator();
    if (op->opMode == OpMode::OPBASE) {
        comm->GetMemTransportManager()->BatchRecoverOpbasedTransports(links);
    } else if (op->opMode == OpMode::OFFLOAD) {
        comm->GetMemTransportManager()->BatchRecoverOffloadTransports(op->opTag, links);
    }

    HCCL_INFO("[CollServiceAiCpuImpl][RecoverTransport] RecoverTransport success!");
    return;
}

HcclResult CollServiceAiCpuImpl::GetSnapShotDynamicBuf(CollOperator& op, BinaryStream& buf)
{
    auto req = comm->GetCollAlgComponent()->GetCollAlgOpReq(op, comm->GetCurAlgName());
    HCCL_INFO(
        "CollServiceAiCpuImpl::GetSnapShotDynamicBuf GetCollAlgOpReq OrchestMode::INSTRUCTION, algName %s",
        req.algName.c_str());
    buf << req.resReq.levelRankPairs.size();
    HCCL_INFO("LeveRankPairs size is %zu", req.resReq.levelRankPairs.size());
    for (auto levelRankPair : req.resReq.levelRankPairs) {
        buf << levelRankPair.first << levelRankPair.second;
        HCCL_INFO("levelRankPair.first is %u, levelRankPair.second is %d", levelRankPair.first, levelRankPair.second);
    }
    // 保证快照一致性，使用0占位
    size_t linkGroupPairCount{0};
    buf << linkGroupPairCount;
    HCCL_DEBUG("[%s], linkGroupPairCount[%u]", __func__, linkGroupPairCount);
    return HcclResult::HCCL_SUCCESS;
}

static void SetModuleDataName(ModuleData& module, const std::string& name)
{
    int ret = strcpy_s(module.name, sizeof(module.name), name.c_str());
    if (ret != 0) {
        THROW<InternalException>(StringFormat("strcpy_s name %s failed. ret[%d]", name.c_str(), ret));
    }
}

std::vector<char> CollServiceAiCpuImpl::PackOpData(const std::string& opTag, const CollAlgOpReq& req) const
{
    std::vector<ModuleData> dataVec;
    dataVec.resize(AicpuResMgrType::__COUNT__);

    AicpuResMgrType resType = AicpuResMgrType::ALG_COMP_INFO;
    dataVec[resType].data = comm->GetCollAlgComponent()->GetPackedData();
    HCCL_INFO("CollServiceAiCpuImpl::PackOpData: GetResMgr %s Data", resType.Describe().c_str());

    resType = AicpuResMgrType::STREAM;
    SetModuleDataName(dataVec[resType], "StreamManager");
    dataVec[resType].data = comm->GetAicpuStreamManager().GetPackedData();
    HCCL_INFO("CollServiceAiCpuImpl::PackOpData: GetResMgr %s Data", resType.Describe().c_str());

    resType = AicpuResMgrType::QUEUE_NOTIFY;
    SetModuleDataName(dataVec[resType], "QueueNotifyManager");
    dataVec[resType].data = comm->GetAicpuQueueNotifyManager().GetPackedData();
    HCCL_INFO("CollServiceAiCpuImpl::PackOpData: GetResMgr %s Data", resType.Describe().c_str());

    resType = AicpuResMgrType::QUEUE_WAIT_GROUP_CNT_NOTIFY;
    SetModuleDataName(dataVec[resType], "QueueWaitGroupCntNotifyManager");
    dataVec[resType].data = comm->GetQueueWaitGroupCntNotifyManager().GetPackedData();
    HCCL_INFO("CollServiceAiCpuImpl::PackOpData: GetResMgr %s Data", resType.Describe().c_str());

    resType = AicpuResMgrType::QUEUE_BCAST_POST_CNT_NOTIFY;
    SetModuleDataName(dataVec[resType], "GetBcastPostCntNotifyManager");
    dataVec[resType].data = comm->GetBcastPostCntNotifyManager().GetPackedData();
    HCCL_INFO("CollServiceAiCpuImpl::PackOpData: GetResMgr %s Data", resType.Describe().c_str());

    resType = AicpuResMgrType::HOST_DEV_SYNC_NOTIFY;
    SetModuleDataName(dataVec[resType], "HostDeviceSyncNotifyManager");
    dataVec[resType].data = comm->GetHostDeviceSyncNotifyManager().GetPackedData();
    HCCL_INFO("CollServiceAiCpuImpl::PackOpData: GetResMgr %s Data", resType.Describe().c_str());

    resType = AicpuResMgrType::TRANSPORT;
    SetModuleDataName(dataVec[resType], "MemTransportManager");
    auto op = comm->GetCurrentCollOperator();
    if (op->opMode == OpMode::OPBASE) { // 单算子模式
        dataVec[resType].data = comm->GetMemTransportManager()->GetOpbasedPackedData();
    } else if (op->opMode == OpMode::OFFLOAD) { // 图下沉模式
        dataVec[resType].data = comm->GetMemTransportManager()->GetOffloadPackedData(opTag);
    } else {
        THROW<InternalException>(StringFormat("opMode=%s failed", op->opMode.Describe().c_str()));
    }
    HCCL_INFO("CollServiceAiCpuImpl::PackOpData: GetResMgr %s Data", resType.Describe().c_str());

    resType = AicpuResMgrType::ALG_TOPO;
    SetModuleDataName(dataVec[resType], req.algName);
    AlgTopoPackageHelper algTopoHelper;
    dataVec[resType].data = algTopoHelper.GetPackedData(req.resReq.topoInfo);
    HCCL_INFO("CollServiceAiCpuImpl::PackOpData: GetResMgr %s Data", resType.Describe().c_str());

    resType = AicpuResMgrType::CONNECTD_MGR;
    SetModuleDataName(dataVec[resType], "ConnectedManager");
    dataVec[resType].data = comm->GetRankGraph()->GetPackedData(req.resReq.levelRankPairs);
    HCCL_INFO("CollServiceAiCpuImpl::PackOpData: GetResMgr %s Data", resType.Describe().c_str());

    AicpuResPackageHelper helper;
    return helper.GetPackedData(dataVec);
}

void CollServiceAiCpuImpl::SaveDfxTaskInfo(
    const TaskParam& taskParam, const RankId remoteRankId, const bool isMaster) const
{
    u32 taskId;
    u32 streamId;
    HrtGetTaskIdAndStreamID(taskId, streamId);

    std::unique_ptr<TaskInfo> taskInfo = std::make_unique<TaskInfo>(
        streamId, taskId, remoteRankId, taskParam, comm->GetMirrorTaskManager().GetCurrDfxOpInfo(), isMaster);

    comm->GetMirrorTaskManager().AddTaskInfo(std::move(taskInfo));
}

void CollServiceAiCpuImpl::Resume()
{
    // 从connectionsBuilder中获取linkData vector
    auto it = connectionsBuilders.find(comm->GetId());
    if (it == connectionsBuilders.end()) {
        HCCL_WARNING(
            "[NsRecovery][Resume] CollServiceAiCpuImpl::Resume, cannot find connectionsBuilder, maybe no op has been "
            "loaded before, commId[%s].",
            comm->GetId().c_str());
        return;
    }
    vector<LinkData> links = it->second->GetAvailableLinksVec();
    HCCL_INFO("[NsRecovery][Resume] CollServiceAiCpuImpl::Resume, links size is [%zu].", links.size());

    // 基于当前已使用的LinkData创建connection
    comm->GetRmaConnManager().BatchCreate(links);
    HCCL_INFO("[NsRecovery][Resume] CollServiceAiCpuImpl::Resume, create connections end.");

    // 创建单算子transport, transport恢复建链，并等待完成
    comm->GetMemTransportManager()->BatchBuildOpbasedTransports(links);
    HCCL_INFO("[NsRecovery][Resume] CollServiceAiCpuImpl::Resume, BatchBuildOpbasedTransports end.");
    WaitOpbasedTransportReady();
    HCCL_INFO("[NsRecovery][Resume] CollServiceAiCpuImpl::Resume, WaitOpbasedTransportReady end.");

    // 基于opTag将connection注入到图模式的transport中
    comm->GetMemTransportManager()->UpdateOffloadTransports();
    HCCL_INFO("[NsRecovery][Resume] CollServiceAiCpuImpl::Resume, UpdateOffloadTransports end.");

    // 将通信域打包
    auto buffer = PackAllTransportData();
    HCCL_INFO("[NsRecovery][Resume] CollServiceAiCpuImpl::Resume, PackAllTransportData end.");

    shared_ptr<DevBuffer> devMem = make_shared<DevBuffer>(buffer.size()); // 申请device内存
    HCCL_INFO(
        "[NsRecovery][Resume] devMem->GetAddr(): 0x%llx, devMem->GetSize(): %llu", devMem->GetAddr(),
        devMem->GetSize());
    HrtMemcpy(
        reinterpret_cast<void*>(devMem->GetAddr()), devMem->GetSize(), buffer.data(), buffer.size(),
        RT_MEMCPY_HOST_TO_DEVICE); // H2D拷贝，将资源拷贝到device内存

    // 组新增的kernelLaunch命令、将打包数据下发到AICPU侧
    auto op = comm->GetCurrentCollOperator();
    if (op->opMode == OpMode::OPBASE) {
        AicpuUpdateCommLaunch(*comm->GetStreamManager().opbase->GetMaster(), devMem.get());
    } else if (op->opMode == OpMode::OFFLOAD) {
        comm->GetAicpuStreamManager().AllocFreeStream();
        AicpuUpdateCommLaunch(*comm->GetAicpuStreamManager().GetFreeStream(), devMem.get());
        HcclStreamSynchronize(comm->GetAicpuStreamManager().GetFreeStream()->GetPtr());
        HCCL_INFO("[NsRecovery][CollServiceAiCpuImpl] HcclUpdateCommKernelEntrance Stream Synchronize finished.");
    } else {
        THROW<InternalException>(StringFormat("[NsRecovery][Resume] opMode=%s failed", op->opMode.Describe().c_str()));
    }
}

void CollServiceAiCpuImpl::AicpuUpdateCommLaunch(Stream& stream, const DevBuffer* mem)
{
    HcclKernelLaunchParam param;

    param.kernel.binaryResAddr = mem->GetAddr();
    param.kernel.binaryResSize = mem->GetSize();

    SetHcclKernelLaunchParam(param, comm);

    s32 ret = strcpy_s(param.kernelName, sizeof(param.kernelName), "HcclUpdateCommKernelEntrance");
    if (ret != EOK) {
        THROW<InternalException>(
            StringFormat("CollServiceAiCpuImpl::AicpuUpdateCommLaunch, strcpy_s kernelName failed! ret[%d]", ret));
    }
    auto op = comm->GetCurrentCollOperator();
    AicpuKernelLaunch(param, stream, op->opMode);
    HCCL_INFO("[NsRecovery][CollServiceAiCpuImpl] HcclUpdateCommKernelEntrance launched.");
}

std::vector<char> CollServiceAiCpuImpl::PackAllTransportData() const
{
    std::vector<ModuleData> dataVec;
    dataVec.resize(AicpuResMgrType::__COUNT__);

    AicpuResMgrType resType = AicpuResMgrType::TRANSPORT;
    SetModuleDataName(dataVec[resType], "MemTransportManager");
    dataVec[resType].data = comm->GetMemTransportManager()->GetPackedAllTransportData();
    HCCL_INFO("CollServiceAiCpuImpl::PackTransportData: GetResMgr %s Data", resType.Describe().c_str());

    AicpuResPackageHelper helper;
    return helper.GetPackedData(dataVec);
}

void CollServiceAiCpuImpl::ReLoadWithOpBasedMode(CollOperator& op)
{
    HCCL_INFO("[CollServiceAiCpuImpl::%s] start.", __func__);
    LoadWithOpBasedModeNoRegister(op);
    HCCL_INFO("[CollServiceAiCpuImpl::%s] end.", __func__);
}

void CollServiceAiCpuImpl::ReLoadWithOffloadMode(CollOperator& op)
{
    HCCL_INFO("[CollServiceAiCpuImpl::%s] start.", __func__);
    LoadWithOffloadModeNoRegister(op);
    HCCL_INFO("[CollServiceAiCpuImpl::%s] end.", __func__);
}

void CollServiceAiCpuImpl::AllocQueueNotify([[maybe_unused]] const InsQueue& insQueue)
{
    // 重写基类接口，AICPU下不支持InsQueue传参
    THROW<InternalException>(
        "Should never use this method in AiCpu, use AllocQueueNotify(std::vector<std::tuple<QId, QId, "
        "u32>> &queueNotifyReq) Instead.");
}

void CollServiceAiCpuImpl::AllocQNotifyForSingleQ([[maybe_unused]] const InsQueue& insQueue) const
{
    THROW<InternalException>("AllocQNotifyForSingleQ is not support in AiCpu mode");
}

HcclResult CollServiceAiCpuImpl::ClearOpLoadedInfo(const std::string& opTag)
{
    if (collOpLoadedMap.find(opTag) == collOpLoadedMap.end()) {
        HCCL_WARNING(
            "[LocalRmaBufManager::%s] opTag[%s] Cannot find Transport in collOpLoadedMap.", __func__, opTag.c_str());
        return HCCL_SUCCESS;
    }
    collOpLoadedMap.erase(opTag);
    return HCCL_SUCCESS;
}

u32 CollServiceAiCpuImpl::GetRemoteRankIdsHashValue(const CollOperator& op) const
{
    vector<RankId> tempRankIds;
    HcclSendRecvItem* itemPtr = reinterpret_cast<HcclSendRecvItem*>(op.batchSendRecvDataDes.sendRecvItemsPtr);
    u32 itemNum = op.batchSendRecvDataDes.itemNum;
    CHK_PTR_NULL(itemPtr);
    for (u32 i = 0; i < itemNum; i++) {
        u32 remoteRankId = (itemPtr + i)->remoteRank;
        tempRankIds.push_back(remoteRankId);
        HCCL_INFO(
            "[CollServiceAiCpuImpl][GetRemoteRankIdsHashValue] insert remoteUserRank[%u] to vector", remoteRankId);
    }
    std::sort(tempRankIds.begin(), tempRankIds.end());

    u32 seed = tempRankIds.size();
    const u32 goldRatio = 0x9e3779b9;
    for (u32 rankId : tempRankIds) {
        seed ^= std::hash<uint32_t>()(rankId) + goldRatio + (seed << 6) + (seed >> 2);
    }
    return seed;
}

HcclResult CollServiceAiCpuImpl::FillBatchSendRecvData(const CollOperator& op)
{
    if (dynamicDataSize == 0) {
        HCCL_ERROR("CollServiceAiCpuImpl::FillBatchSendRecvData dynamicDataSize is 0");
        THROW<InternalException>(StringFormat("CollServiceAiCpuImpl::FillBatchSendRecvData dynamicDataSize is 0"));
        return HCCL_E_PARA;
    }
    Buffer dynamicDataMem = kernelParamBuf_.get()->Range(sizeof(struct HcclKernelParamLite), dynamicDataSize);
    struct BatchSendRecvDataDes* batchSendRecvDataPtr
        = reinterpret_cast<struct BatchSendRecvDataDes*>(dynamicDataMem.GetAddr());
    batchSendRecvDataPtr->itemNum = op.batchSendRecvDataDes.itemNum;
    for (u32 i = 0; i < op.batchSendRecvDataDes.itemNum; i++) {
        CHK_PTR_NULL(static_cast<HcclSendRecvItem*>(op.batchSendRecvDataDes.sendRecvItemsPtr) + i);
        batchSendRecvDataPtr->batchSendRecvItem[i]
            = *(static_cast<HcclSendRecvItem*>(op.batchSendRecvDataDes.sendRecvItemsPtr) + i);
        HCCL_INFO(
            "[CollServiceAiCpuImpl][FillBatchSendRecvData] alloc batchSendRecvItem remoteRank[%u]",
            batchSendRecvDataPtr->batchSendRecvItem[i].remoteRank);
    }

    return HCCL_SUCCESS;
}

HcclResult CollServiceAiCpuImpl::FillAllToAllvData(const CollOperator& op)
{
    if (dynamicDataSize == 0) {
        HCCL_ERROR("CollServiceAiCpuImpl::FillAllToAllvData dynamicDataSize is 0");
        THROW<InternalException>(StringFormat("CollServiceAiCpuImpl::FillAllToAllvData dynamicDataSize is 0"));
        return HCCL_E_PARA;
    }
    Buffer dynamicDataMem = kernelParamBuf_.get()->Range(sizeof(struct HcclKernelParamLite), dynamicDataSize);
    struct AllToAllvDataDes* alltoallvDataPtr = reinterpret_cast<struct AllToAllvDataDes*>(dynamicDataMem.GetAddr());
    alltoallvDataPtr->sendType = static_cast<u8>(op.all2AllVDataDes.sendType);
    alltoallvDataPtr->recvType = static_cast<u8>(op.all2AllVDataDes.recvType);
    u32 rankSize = comm->GetRankSize();
    u64* sendCountsPtr = static_cast<u64*>(alltoallvDataPtr->sendRecvInfos);
    u64* recvCountsPtr = sendCountsPtr + rankSize;
    u64* sdisplsPtr = recvCountsPtr + rankSize;
    u64* rdisplsPtr = sdisplsPtr + rankSize;
    for (u32 i = 0; i < rankSize; i++) {
        CHK_PTR_NULL(static_cast<const u64*>(op.all2AllVDataDes.sendCounts) + i);
        sendCountsPtr[i] = *(static_cast<const u64*>(op.all2AllVDataDes.sendCounts) + i);
        CHK_PTR_NULL(static_cast<const u64*>(op.all2AllVDataDes.recvCounts) + i);
        recvCountsPtr[i] = *(static_cast<const u64*>(op.all2AllVDataDes.recvCounts) + i);
        CHK_PTR_NULL(static_cast<const u64*>(op.all2AllVDataDes.sdispls) + i);
        sdisplsPtr[i] = *(static_cast<const u64*>(op.all2AllVDataDes.sdispls) + i);
        CHK_PTR_NULL(static_cast<const u64*>(op.all2AllVDataDes.rdispls) + i);
        rdisplsPtr[i] = *(static_cast<const u64*>(op.all2AllVDataDes.rdispls) + i);
        HCCL_INFO(
            "[CollServiceAiCpuImpl][FillAllToAllvData] sendCounts[%llu], recvCounts[%llu], sdispls[%llu], "
            "rdispls[%llu]",
            sendCountsPtr[i], recvCountsPtr[i], sdisplsPtr[i], rdisplsPtr[i]);
    }
    HCCL_INFO(
        "[CollServiceAiCpuImpl][FillAllToAllvData] fill alltoallv data success, alltoallvDataPtr[%p]",
        alltoallvDataPtr);
    return HCCL_SUCCESS;
}

HcclResult CollServiceAiCpuImpl::FillAllToAllvcData(const CollOperator& op)
{
    if (dynamicDataSize == 0) {
        HCCL_ERROR("CollServiceAiCpuImpl::FillAllToAllvcData dynamicDataSize is 0");
        THROW<InternalException>(StringFormat("CollServiceAiCpuImpl::FillAllToAllvcData dynamicDataSize is 0"));
        return HCCL_E_PARA;
    }
    Buffer dynamicDataMem = kernelParamBuf_.get()->Range(sizeof(struct HcclKernelParamLite), dynamicDataSize);
    struct AllToAllvcDataDes* alltoallvcDataPtr = reinterpret_cast<struct AllToAllvcDataDes*>(dynamicDataMem.GetAddr());
    alltoallvcDataPtr->sendType = static_cast<u8>(op.all2AllVCDataDes.sendType);
    alltoallvcDataPtr->recvType = static_cast<u8>(op.all2AllVCDataDes.recvType);
    u32 rankSize = comm->GetRankSize();
    for (u64 i = 0; i < rankSize * rankSize; i++) {
        CHK_PTR_NULL(static_cast<const u64*>(op.all2AllVCDataDes.sendCountMatrix) + i);
        alltoallvcDataPtr->sendCountMatrix[i] = *(static_cast<const u64*>(op.all2AllVCDataDes.sendCountMatrix) + i);
    }
    return HCCL_SUCCESS;
}

} // namespace Hccl
