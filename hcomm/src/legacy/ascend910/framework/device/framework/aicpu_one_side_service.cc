/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aicpu_one_side_service.h"
#include "log.h"
#include "common/aicpu_sqe_context.h"
#include "aicpu_hccl_common.h"
#include "transport_pub.h"
#include "adapter_hal_pub.h"
#include "dispatcher.h"
#include "executor_tracer.h"
#include "aicpu_hccl_process.h"

namespace hccl {
constexpr u64 MAX_RDMA_WQE_SIZE = 2ULL * 1024 * 1024 * 1024; // RDMA最大WQE限制是2GB

std::shared_mutex HcclOneSideServiceAicpu::serviceMapMutex_;
std::unordered_map<std::string, std::shared_ptr<HcclOneSideServiceAicpu>> HcclOneSideServiceAicpu::services_;

HcclOneSideServiceAicpu::HcclOneSideServiceAicpu() {}

HcclOneSideServiceAicpu::~HcclOneSideServiceAicpu()
{
    CHK_PRT_CONT(ReportHcclTaskInfo() != HCCL_SUCCESS, HCCL_WARNING("[~] ReportHcclTaskInfo failed"));
    CHK_PRT_CONT(ClearStreamLocalBuff() != HCCL_SUCCESS, HCCL_WARNING("[~] ClearStreamLocalBuff failed"));
    rdmaLinks_.clear();
    if (dispatcher_ != nullptr) {
        HcclDispatcherDestroy(dispatcher_);
        dispatcher_ = nullptr;
    }
}

HcclResult HcclOneSideServiceAicpu::Process(const OpTilingData* tilingData)
{
    std::string tag = tilingData->tag;
    const HcclCMDType cmdType = static_cast<HcclCMDType>(tilingData->opType);
    HCCL_DEBUG("[Process] Entry, tag[%s] cmdType[%u]", tag.c_str(), cmdType);

    const u8* dynamicDataPtr = reinterpret_cast<const u8*>(tilingData) + sizeof(OpTilingData);
    CHK_PRT_RET(
        tilingData->length < sizeof(OpTilingOneSideCommDataDes),
        HCCL_ERROR(
            "[Process] dynamicDataSize[%llu] should be greater than or equal to "
            "OpTilingOneSideCommDataDes[%llu]",
            tilingData->length, sizeof(OpTilingOneSideCommDataDes)),
        HCCL_E_PARA);
    const auto* vDataPtr = reinterpret_cast<const OpTilingOneSideCommDataDes*>(dynamicDataPtr);
    if (vDataPtr->finalize) {
        std::unique_lock<std::shared_mutex> rwlock(serviceMapMutex_);
        services_.erase(tag);
        HCCL_INFO("[Finalize] tag[%s], services[%u]", tag.c_str(), services_.size());
        return HCCL_SUCCESS;
    }

    auto service = GetService(tag, tilingData);
    CHK_PRT_RET(service == nullptr, HCCL_ERROR("[Process] Service not found, tag[%s].", tag.c_str()), HCCL_E_INTERNAL);
    return service->DoProcess(tag, tilingData);
}

std::shared_ptr<HcclOneSideServiceAicpu>
HcclOneSideServiceAicpu::GetService(const std::string& tag, const OpTilingData* tilingData)
{
    {
        std::shared_lock<std::shared_mutex> rwlock(serviceMapMutex_);
        auto serviceIter = services_.find(tag);
        if (serviceIter != services_.cend()) {
            return serviceIter->second;
        }
    }
    std::shared_ptr<HcclOneSideServiceAicpu> service;
    EXCEPTION_CATCH(service = std::make_shared<HcclOneSideServiceAicpu>(), return nullptr);
    CHK_PRT_RET(service == nullptr, HCCL_ERROR("[GetService] Alloc failed, tag[%s].", tag.c_str()), nullptr);
    HcclResult ret = service->Init(tag, tilingData);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[GetService] Init failed, tag[%s].", tag.c_str()), nullptr);
    {
        std::unique_lock<std::shared_mutex> rwlock(serviceMapMutex_);
        services_[tag] = service;
    }
    AicpuComContext* ctx = AicpuGetComContext();
    AicpuHcclProcess::CallMC2MaintenanceThread(ctx);
    return service;
}

HcclResult HcclOneSideServiceAicpu::Init(const std::string& tag, const OpTilingData* tilingData)
{
    if (isInited_) {
        HCCL_WARNING("[Init] Already inited, tag[%s]", tag.c_str());
        return HCCL_SUCCESS;
    }

    const u8* dynamicDataPtr = reinterpret_cast<const u8*>(tilingData) + sizeof(OpTilingData);
    CHK_PRT_RET(
        tilingData->length < sizeof(OpTilingOneSideCommDataDes),
        HCCL_ERROR(
            "[Init] dynamicDataSize[%llu] should be greater than or equal to OpTilingOneSideCommDataDes[%llu]",
            tilingData->length, sizeof(OpTilingOneSideCommDataDes)),
        HCCL_E_PARA);
    const auto* vDataPtr = reinterpret_cast<const OpTilingOneSideCommDataDes*>(dynamicDataPtr);
    CHK_PRT_RET(
        vDataPtr->commResParaSize != sizeof(HcclOneSideCommResParam),
        HCCL_ERROR(
            "[Init] commResParaSize[%llu] should be equal to HcclOneSideCommResParam[%llu]", vDataPtr->commResParaSize,
            sizeof(HcclOneSideCommResParam)),
        HCCL_E_PARA);
    commResParaPtr_ = reinterpret_cast<const HcclOneSideCommResParam*>(vDataPtr->commResParaAddr);
    CHK_PTR_NULL(commResParaPtr_);

    identifier_ = tag;
    rankId_ = tilingData->srcRank;
    rankSize_ = vDataPtr->rankSize;

    const u32 hostDevId = commResParaPtr_->aicpuOpNotify[0].devId;
    CHK_RET(hrtDrvGetLocalDevIDByHostDevID(hostDevId, &devId_));
    CHK_RET(hrtHalGetDeviceType(devId_, devType_));
    CHK_PRT_RET(
        devType_ != DevType::DEV_TYPE_910_93 && devType_ != DevType::DEV_TYPE_910B,
        HCCL_ERROR("[Init] Expect devType[%u] is A2 or A3", devType_), HCCL_E_NOT_SUPPORT);
    CHK_RET(hrtHalGetDeviceInfo(devId_, MODULE_TYPE_SYSTEM, INFO_TYPE_PHY_CHIP_ID, &chipId_));

    s32 devLogicId = INVALID_INT;
    CHK_RET(hrtGetDevice(&devLogicId));
    if (devLogicId == INVALID_INT) { // standalone mode, run without ccl op
        CHK_RET(hrtSetlocalDevice(hostDevId));
        CHK_RET(hrtSetlocalDeviceType(devType_));
        CHK_RET(hrtSetLocalDeviceSatMode(static_cast<aclrtFloatOverflowMode>(tilingData->floatOverflowMode)));
    }
    logicDevId_ = hostDevId;

    CHK_RET(HcclDispatcherAicpuInit(&dispatcher_, devId_, SDMA_QOS_DEFAULT, DispatcherType::DISPATCHER_AICPU));

    CHK_RET(InitOpNotifyObj());
    CHK_RET(InitStream(execStream_, execComStreamInfo_, commResParaPtr_->execStreamParam, tag));

    CHK_RET(InitProfiling());

    isInited_ = true;

    HCCL_RUN_INFO(
        "[Init] End. rankId[%u] hostDevId[%u] chipId[%u] devId[%u] streamId[%u] commResPara[%p]", rankId_, hostDevId,
        chipId_, devId_, execStream_.id(), commResParaPtr_);

    return HCCL_SUCCESS;
}

HcclResult HcclOneSideServiceAicpu::InitOpNotifyObj()
{
    for (u32 i = 0; i < AICPU_OP_NOTIFY_MAX_NUM; i++) {
        const HcclSignalInfo& signalInfo = commResParaPtr_->aicpuOpNotify[i];
        CHK_PRT_RET(
            signalInfo.resId == INVALID_U64, HCCL_ERROR("[InitOpNotifyObj] resId[%llu] is invalid", signalInfo.resId),
            HCCL_E_PARA);

        std::shared_ptr<LocalNotify> notify;
        EXCEPTION_CATCH((notify = std::make_shared<LocalNotify>()), return HCCL_E_PTR);
        CHK_SMART_PTR_NULL(notify);
        CHK_RET(notify->Init(signalInfo, NotifyLoadType::DEVICE_NOTIFY));
        opNotifies_.push_back(notify);
        HCCL_INFO(
            "[InitOpNotifyObj] tag[%s] resId[%llu] tsId[%u] devId[%u]", identifier_.c_str(), signalInfo.resId,
            signalInfo.tsId, signalInfo.devId);
    }
    return HCCL_SUCCESS;
}

HcclResult HcclOneSideServiceAicpu::InitStream(
    Stream& stream, HcclComStreamInfo& comStreamInfo, const HcclStreamParam& streamParam, const std::string& tag)
{
    const HcclStreamInfo& streamInfo = streamParam.streamInfo;
    comStreamInfo.sqId = streamInfo.sqIds;
    comStreamInfo.actualStreamId = streamInfo.streamIds;
    comStreamInfo.logicCqId = streamInfo.logicCqids;

    u64 sqAddr = 0;
    CHK_RET(QuerySqBaseAddr(devId_, streamInfo.sqIds, sqAddr));
    comStreamInfo.sqBaseAddr = reinterpret_cast<void*>(sqAddr);
    CHK_PRT_RET(comStreamInfo.sqBaseAddr == nullptr, HCCL_ERROR("[Init] sqe base addr is nullptr."), HCCL_E_PTR);
    CHK_RET(QuerySqStatusByType(devId_, streamInfo.sqIds, DRV_SQCQ_PROP_SQ_DEPTH, comStreamInfo.sqDepth));
    u32 sqHead = 0;
    CHK_RET(QuerySqStatusByType(devId_, streamInfo.sqIds, DRV_SQCQ_PROP_SQ_HEAD, sqHead));
    u32 sqTail = 0;
    CHK_RET(QuerySqStatusByType(devId_, streamInfo.sqIds, DRV_SQCQ_PROP_SQ_TAIL, sqTail));
    HCCL_DEBUG(
        "[Init] get stream data success, tag[%s], streamId[%d], sqId[%d], logicCqId[%u], sqDepth[%u], "
        "sqHead[%u], sqTail[%u]",
        tag.c_str(), comStreamInfo.actualStreamId, comStreamInfo.sqId, comStreamInfo.logicCqId, comStreamInfo.sqDepth,
        sqHead, sqTail);
    stream = Stream(comStreamInfo);
    u64 sqCqeContextSize = streamParam.sqCqContextSize;
    CHK_PRT_RET(
        sqCqeContextSize != sizeof(SqCqeContext),
        HCCL_ERROR(
            "[%s] sqCqeContextSize[%llu] is not equal to sizeof(SqCqeContext)[%llu], tag[%s]", __func__,
            sqCqeContextSize, sizeof(SqCqeContext), tag.c_str()),
        HCCL_E_PARA);
    SqCqeContext* sqCqeContext = reinterpret_cast<SqCqeContext*>(streamParam.sqCqContextAddr);
    CHK_PRT_RET(
        sqCqeContext == nullptr,
        HCCL_ERROR("[%s] sqCqeContext[%llu] is nullptr, tag[%s]", __func__, streamParam.sqCqContextAddr, tag.c_str()),
        HCCL_E_PARA);
    CHK_RET(stream.InitSqAndCqeContext(sqHead, sqTail, sqCqeContext));
    HCCL_INFO(
        "[%s] Create stream success, tag[%s], streamId[%u], devId[%u]", __func__, tag.c_str(), stream.id(), devId_);

    return HCCL_SUCCESS;
}

HcclResult HcclOneSideServiceAicpu::FillMemDetails(
    MemDetails& localMems, MemDetails& remoteMems, const HcclOneSideOpDescParam* descPtr, u32 index)
{
    CHK_PTR_NULL(descPtr);
    const HcclDataType dataType = static_cast<HcclDataType>(descPtr[index].dataType);
    if (dataType_ == HcclDataType::HCCL_DATA_TYPE_RESERVED) {
        dataType_ = dataType;
    }
    const u32 perDataSize = DataUnitSize(dataType);
    CHK_PRT_RET(perDataSize == 0, HCCL_ERROR("[FillMemDetails] dataType[%u] DataUnitSize is 0", dataType), HCCL_E_PARA);
    const u64 count = descPtr[index].count;
    const u64 buffSize = count * perDataSize;
    totalCount_ += count;
    localMems.addr = descPtr[index].localAddr;
    localMems.size = buffSize;
    localMems.key = descPtr[index].lkey;
    remoteMems.addr = descPtr[index].remoteAddr;
    remoteMems.size = buffSize;
    remoteMems.key = descPtr[index].rkey;
    HCCL_DEBUG(
        "[FillMemDetails] local addr[%#llx], remote addr[%#llx], size[%llu]", localMems.addr, remoteMems.addr,
        localMems.size);
    return HCCL_SUCCESS;
}

HcclResult HcclOneSideServiceAicpu::PrepareRdmaLink(u32 remoteRankId, const struct HcclQpInfoV2& qpInfo)
{
    if (rdmaLinks_.find(remoteRankId) == rdmaLinks_.end()) {
        const int UNIT_CONVERSION = 1000;
        linkTimeout_
            = 4096ULL * (1 << qpInfo.retryTime) * (qpInfo.retryCnt + 1) / UNIT_CONVERSION; // RDMA超时基数是4.096us
        TransportMem::AttrInfo attrInfo{};
        attrInfo.localRankId = rankId_;
        attrInfo.remoteRankId = remoteRankId;
        attrInfo.timeout = linkTimeout_;
        std::shared_ptr<TransportMem> link;
        EXCEPTION_CATCH(
            link = TransportMem::Create(TransportMem::TpType::ROCE_DEVICE, qpInfo, dispatcher_, attrInfo),
            return HCCL_E_MEMORY);
        CHK_SMART_PTR_NULL(link);
        rdmaLinks_[remoteRankId] = link;
        HCCL_INFO(
            "[Init] PrepareRdmaLink. rankId[%u] chipId[%u] devId[%u] remoteRankId[%u] linkTimeout[%u us]"
            "retryTime[%u] retryCnt[%u]",
            rankId_, chipId_, devId_, remoteRankId, linkTimeout_, qpInfo.retryTime, qpInfo.retryCnt);
    }
    return HCCL_SUCCESS;
}

HcclResult HcclOneSideServiceAicpu::DoProcess(const std::string& tag, const OpTilingData* tilingData)
{
    const HcclCMDType cmdType = static_cast<HcclCMDType>(tilingData->opType);
    const u32 remoteRankId = tilingData->dstRank;
    HCCL_DEBUG("[DoProcess] Entry. tag[%s] cmdType[%u] remoteRankId[%u]", tag.c_str(), cmdType, remoteRankId);

    const u8* dynamicDataPtr = reinterpret_cast<const u8*>(tilingData) + sizeof(OpTilingData);
    CHK_PRT_RET(
        tilingData->length < sizeof(OpTilingOneSideCommDataDes),
        HCCL_ERROR(
            "[DoProcess] dynamicDataSize[%llu] should be greater than or equal to "
            "OpTilingOneSideCommDataDes[%llu]",
            tilingData->length, sizeof(OpTilingOneSideCommDataDes)),
        HCCL_E_PARA);
    const auto* vDataPtr = reinterpret_cast<const OpTilingOneSideCommDataDes*>(dynamicDataPtr);
    CHK_PRT_RET(
        vDataPtr->commResParaSize != sizeof(HcclOneSideCommResParam),
        HCCL_ERROR(
            "[DoProcess] commResParaSize[%llu] should be equal to HcclOneSideCommResParam[%llu]",
            vDataPtr->commResParaSize, sizeof(HcclOneSideCommResParam)),
        HCCL_E_PARA);
    const auto* commResParaPtr = reinterpret_cast<const HcclOneSideCommResParam*>(vDataPtr->commResParaAddr);
    CHK_PRT_RET(
        commResParaPtr != commResParaPtr_,
        HCCL_ERROR("[DoProcess] not support commResParaPtr[%p/%p] address update", commResParaPtr, commResParaPtr_),
        HCCL_E_PARA);
    LinkType linkType = static_cast<LinkType>(vDataPtr->linkType);
    CHK_PRT_RET(
        (linkType != LinkType::LINK_ROCE) && (linkType != LinkType::LINK_HCCS),
        HCCL_ERROR("[DoProcess] not support linkType[%u]", vDataPtr->linkType), HCCL_E_PARA);
    const u32 descNum = vDataPtr->descNum; // Batch descNum + 1(signal)
    CHK_PRT_RET(
        vDataPtr->descDataLen != descNum * sizeof(HcclOneSideOpDescParam),
        HCCL_ERROR(
            "[DoProcess] descDataLen[%llu] should be equal to "
            "descNum[%u] * sizeof(HcclOneSideOpDescParam)[%llu]",
            vDataPtr->descDataLen, descNum, sizeof(HcclOneSideOpDescParam)),
        HCCL_E_PARA);
    const auto* desc
        = reinterpret_cast<const HcclOneSideOpDescParam*>(dynamicDataPtr + sizeof(OpTilingOneSideCommDataDes));

    CHK_RET(WorkStart(cmdType, remoteRankId));

    if (linkType == LinkType::LINK_ROCE) {
        CHK_RET(DoRdmaProcess(cmdType, remoteRankId, vDataPtr, desc, descNum));
    } else {
        CHK_RET(DoSdmaProcess(cmdType, remoteRankId, vDataPtr, desc, descNum));
    }

    CHK_RET(WorkEnd(cmdType, remoteRankId));
    return HCCL_SUCCESS;
}

HcclResult HcclOneSideServiceAicpu::DoRdmaProcess(
    HcclCMDType cmdType, u32 remoteRankId, const OpTilingOneSideCommDataDes* vDataPtr,
    const HcclOneSideOpDescParam* desc, u32 descNum)
{
    CHK_PRT_RET(
        vDataPtr->transportDataSize != sizeof(TransportDeviceNormalData),
        HCCL_ERROR(
            "[DoProcess] transportDataSize[%llu] should be equal to TransportDeviceNormalData[%llu]",
            vDataPtr->transportDataSize, sizeof(TransportDeviceNormalData)),
        HCCL_E_PARA);
    const auto* transportDataPtr = reinterpret_cast<const TransportDeviceNormalData*>(vDataPtr->transportDataAddr);
    const TransportDeviceNormalData& ibvData = *transportDataPtr;

    CHK_RET(PrepareRdmaLink(remoteRankId, ibvData.qpInfo));
    auto link = rdmaLinks_[remoteRankId];
    CHK_SMART_PTR_NULL(link);

    const u32 userDescNum = descNum - 1;
    std::vector<MemDetails> localMems(userDescNum);
    std::vector<MemDetails> remoteMems(userDescNum);
    for (u32 index = 0; index < userDescNum; ++index) {
        CHK_RET(FillMemDetails(localMems[index], remoteMems[index], desc, index));
    }
    const bool isRead = (cmdType == HcclCMDType::HCCL_CMD_BATCH_GET);
    if (isRead) {
        CHK_RET(link->BatchRead(localMems, remoteMems, execStream_));
    } else {
        CHK_RET(link->BatchWrite(remoteMems, localMems, execStream_));
    }

    // fence signal at last
    MemDetails localFenceMem{};
    MemDetails remoteFenceMem{};
    CHK_RET(FillMemDetails(localFenceMem, remoteFenceMem, desc, descNum - 1));
    CHK_RET(link->AddOpFence(localFenceMem, remoteFenceMem, execStream_));

    HCCL_DEBUG(
        "[DoProcess] End. tag[%s] cmdType[%u] remoteRankId[%u] transportData[%p] rdma desc[%p] "
        "descNum[%u]",
        identifier_.c_str(), cmdType, remoteRankId, transportDataPtr, desc, descNum);
    return HCCL_SUCCESS;
}

HcclResult HcclOneSideServiceAicpu::DoSdmaProcess(
    HcclCMDType cmdType, u32 remoteRankId, [[maybe_unused]] const OpTilingOneSideCommDataDes* vDataPtr,
    const HcclOneSideOpDescParam* desc, u32 descNum)
{
    CHK_PTR_NULL(desc);
    u32 userDescNum = descNum - 1; // fence signal at last, but sdma needn't fence, so keep reserve
    for (u32 index = 0; index < userDescNum; ++index) {
        HcclDataType dataType = static_cast<HcclDataType>(desc[index].dataType);
        u64 dataSize = desc[index].count * DataUnitSize(dataType);
        DeviceMem localMem = DeviceMem::create(reinterpret_cast<void*>(desc[index].localAddr), dataSize);
        DeviceMem remoteMem = DeviceMem::create(reinterpret_cast<void*>(desc[index].remoteAddr), dataSize);
        if (cmdType == HcclCMDType::HCCL_CMD_BATCH_GET) {
            CHK_RET(
                HcclD2DMemcpyAsync(dispatcher_, localMem, remoteMem, execStream_, remoteRankId, LinkType::LINK_HCCS));
        } else {
            CHK_RET(
                HcclD2DMemcpyAsync(dispatcher_, remoteMem, localMem, execStream_, remoteRankId, LinkType::LINK_HCCS));
        }
    }

    CHK_RET(LocalNotify::Post(execStream_, dispatcher_, opNotifies_[1]));
    CHK_RET(LaunchTask(dispatcher_, const_cast<Stream&>(execStream_)));

    HCCL_DEBUG(
        "[DoProcess] End. tag[%s] cmdType[%u] remoteRankId[%u] sdma desc[%p] descNum[%u]", identifier_.c_str(), cmdType,
        remoteRankId, desc, descNum);
    return HCCL_SUCCESS;
}

HcclResult HcclOneSideServiceAicpu::InitProfiling()
{
    CHK_RET(RegisterLoadTaskCallBack(dispatcher_, nullptr, dfx::TaskProfilingCallBack));
    groupHashId_ = dfx::ProfilingManager::GetProfHashId(identifier_.c_str(), identifier_.length());
    HCCL_INFO("[InitProfiling]group[%s], groupHashId[%llu].", identifier_.c_str(), groupHashId_);
    dfx::ProfCommInfo profInfo{groupHashId_, rankSize_, rankId_};
    CHK_RET(dfx::ProfilingManager::AddProfInfoByStreamId(execStream_.id(), identifier_, profInfo));
    dfx::ProfilingExtendInfoHelper::InitProfItemId();
    return HCCL_SUCCESS;
}

HcclResult HcclOneSideServiceAicpu::WorkStart(HcclCMDType cmdType, u32 remoteRankId)
{
    CHK_RET(ReportMainStreamTask(HEAD_TASK));
    CHK_RET(UpdateProfReportStartSqeIdx());
    // 刷新profiling开关, 支持profiling从中间迭代采集
    const bool profL0Open = dfx::ProfilingManager::IsProfL0On();
    const bool profL1Open = dfx::ProfilingManager::IsProfL1On();
    HCCL_INFO(
        "[WorkStart] streamId[%u] tag[%s] cmdType[%u] remoteRankId[%u] profL0Open[%u/%u] profL1Open[%u/%u]",
        execStream_.id(), identifier_.c_str(), cmdType, remoteRankId, profL0Open,
        dfx::ProfilingManager::GetProfL0State(), profL1Open, dfx::ProfilingManager::GetProfL1State());
    return HCCL_SUCCESS;
}

HcclResult HcclOneSideServiceAicpu::WorkEnd(HcclCMDType cmdType, u32 remoteRankId)
{
    CHK_RET(CombineReportOpInfo(cmdType, dataType_, totalCount_));
    dataType_ = HcclDataType::HCCL_DATA_TYPE_RESERVED;
    totalCount_ = 0;
    HCCL_DEBUG(
        "[WorkEnd] streamId[%u] tag[%s] cmdType[%u] remoteRankId[%u]", execStream_.id(), identifier_.c_str(), cmdType,
        remoteRankId);
    CHK_RET(ReportMainStreamTask(TAIL_TASK));
    return HCCL_SUCCESS;
}

HcclResult HcclOneSideServiceAicpu::ReportMainStreamTask(u16 type)
{
    HcclSqeContext* sqeContext = execStream_.GetSqeContextPtr();
    const SqeRingBuffer& sqeBuffer = sqeContext->buffer;
    u16 taskId = (type == TAIL_TASK) ? (sqeBuffer.tailSqeTaskId - 1) : sqeBuffer.tailSqeTaskId;
    return dfx::ProfilingManager::ReportMainStreamTask(execStream_, taskId, type);
}

HcclResult HcclOneSideServiceAicpu::UpdateProfReportStartSqeIdx()
{
    if (dfx::ProfilingManager::IsL1fromOffToOn()) {
        HcclSqeContext* sqeContext = execStream_.GetSqeContextPtr();
        const SqeRingBuffer& sqeBuffer = sqeContext->buffer;
        CHK_RET(dfx::ProfilingManager::UpdateStartReportSqeIdx(execStream_.id(), sqeBuffer.tailSqeIdx));
    }
    return HCCL_SUCCESS;
}

HcclResult HcclOneSideServiceAicpu::CombineReportOpInfo(HcclCMDType cmdType, u8 dataType, u64 count)
{
    MsprofAicpuHCCLOPInfo hcclOpInfo{};
    hcclOpInfo.dataType = dataType;
    hcclOpInfo.count = count;
    hcclOpInfo.groupName = groupHashId_;
    hcclOpInfo.ranksize = rankSize_;
    std::string typeStr = (cmdType == HcclCMDType::HCCL_CMD_BATCH_GET) ? "BatchGet" : "BatchPut";
    CHK_RET(dfx::ProfilingManager::ReportHcclOpInfo(hcclOpInfo, typeStr));
    return HCCL_SUCCESS;
}

HcclResult HcclOneSideServiceAicpu::ReportHcclTaskInfo()
{
    return dfx::ProfilingManager::ReportTaskInfo(execStream_.id(), execStream_.GetSqeContextPtr());
}

HcclResult HcclOneSideServiceAicpu::ClearStreamLocalBuff()
{
    CHK_RET(execStream_.ClearLocalBuff());
    return dfx::ProfilingManager::UpdateStartReportSqeIdx(execStream_.id(), 0);
}

HcclResult HcclOneSideServiceAicpu::CleanStreamFunc()
{
    if (execStreamEnable_) {
        return HCCL_SUCCESS;
    }
    HCCL_RUN_INFO("Entry HcclOneSideServiceAicpu::CleanStreamFunc tag[%s]", identifier_.c_str());
    const HcclComStreamInfo& streamInfo = execStream_.GetHcclStreamInfo();
    CHK_RET(ConfigSqStatusByType(devId_, streamInfo.sqId, DRV_SQCQ_PROP_SQ_DISABLE_TO_ENABLE, 1));
    CHK_RET(CleanStream(execStream_));
    execStreamEnable_ = true;
    HCCL_RUN_INFO(
        "Entry HcclOneSideServiceAicpu::CleanStreamFunc reset stream sq buffer success, "
        "SetStreamEnable streamid[%d]",
        streamInfo.actualStreamId);
    return HCCL_SUCCESS;
}

HcclResult HcclOneSideServiceAicpu::CleanAllStreamFunc()
{
    HCCL_INFO("Entry HcclOneSideServiceAicpu::CleanAllStreamFunc");
    std::shared_lock<std::shared_mutex> rwlock(serviceMapMutex_);
    for (auto& serviceIter : services_) {
        HcclResult ret = serviceIter.second->CleanStreamFunc();
        if (ret != HCCL_SUCCESS) {
            return ret;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult HcclOneSideServiceAicpu::DisableStreamFunc()
{
    HCCL_INFO("Entry HcclOneSideServiceAicpu::DisableStreamFunc tag[%s]", identifier_.c_str());
    execStreamEnable_ = false;
    return HCCL_SUCCESS;
}

HcclResult HcclOneSideServiceAicpu::DisableAllStreamFunc()
{
    HCCL_INFO("Entry HcclOneSideServiceAicpu::DisableAllStreamFunc");
    std::shared_lock<std::shared_mutex> rwlock(serviceMapMutex_);
    for (auto& serviceIter : services_) {
        HcclResult ret = serviceIter.second->DisableStreamFunc();
        if (ret != HCCL_SUCCESS) {
            return ret;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult HcclOneSideServiceAicpu::CleanStream(Stream& stream)
{
    CHK_RET(stream.ClearLocalBuff());
    CHK_RET(UpdateSqStatus(stream));
    HCCL_INFO("Entry HcclOneSideServiceAicpu::CleanStream %u success tag[%s]", stream.sqId(), identifier_.c_str());
    return HCCL_SUCCESS;
}

HcclResult HcclOneSideServiceAicpu::UpdateSqStatus(Stream& stream)
{
    HcclSqeContext* sqeContext = stream.GetSqeContextPtr();
    CHK_PTR_NULL(sqeContext);
    SqeRingBuffer* sqeContextBuffer = &(sqeContext->buffer);
    auto& head = sqeContextBuffer->sqHead;
    auto& tail = sqeContextBuffer->sqTail;

    CHK_RET(QuerySqStatusByType(devId_, stream.sqId(), DRV_SQCQ_PROP_SQ_TAIL, head));
    CHK_RET(QuerySqStatusByType(devId_, stream.sqId(), DRV_SQCQ_PROP_SQ_HEAD, tail));
    HCCL_INFO(
        "Entry HcclOneSideServiceAicpu::UpdateSqStatus, sqid:%u head:%u tail:%u tag[%s]", stream.sqId(), head, tail,
        identifier_.c_str());
    return HCCL_SUCCESS;
}

HcclResult HcclOneSideServiceAicpu::HandleErrCqe()
{
    std::shared_lock<std::shared_mutex> rwlock(serviceMapMutex_);
    for (auto& serviceIter : services_) {
        serviceIter.second->HandleCqeMessage(true);
    }
    return HCCL_SUCCESS;
}

void HcclOneSideServiceAicpu::HandleCqeMessage(bool isReadClear)
{
    rtLogicCqReport_t cqeException;
    CqeStatus cqeStatus = CqeStatus::kDefault;
    PollCqeException(execStream_, isReadClear, cqeException, cqeStatus);
}

void HcclOneSideServiceAicpu::PollCqeException(
    Stream& stream, bool isReadClear, rtLogicCqReport_t& cqeException, CqeStatus& cqeStatus)
{
    const HcclComStreamInfo& streamInfo = stream.GetHcclStreamInfo();
    bool isPollCqe = isReadClear;
    while (isPollCqe) {
        CqeQueryInput cqeQueryInput;
        dfx_tracer::ExecutorTracer::SetCqeQueryInput(devId_, streamInfo, cqeQueryInput);
        constexpr u32 reportSize = 256;
        rtLogicCqReport_t streamReport[reportSize];
        cqeQueryInput.cqeAddr = reinterpret_cast<uint8_t*>(streamReport);
        cqeStatus = CqReportRecv(cqeQueryInput, cqeException);
        isPollCqe = (cqeStatus == dfx::CqeStatus::kCqeException);
    }
}

bool HcclOneSideServiceAicpu::isAllDestroy()
{
    std::shared_lock<std::shared_mutex> rwlock(serviceMapMutex_);
    bool isEmpty = services_.empty();
    return isEmpty;
}
} // namespace hccl
