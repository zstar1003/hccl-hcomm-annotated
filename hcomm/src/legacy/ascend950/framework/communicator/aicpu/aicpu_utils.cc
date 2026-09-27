/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <shared_mutex>
#include "inc/aicpu_utils.h"
#include "log.h"
#include "aicpu_comm_destroy_func.h"
#include "communicator_impl_lite_manager.h"
#include "ub_conn_lite_mgr.h"
#include "aicpu_daemon_service.h"
#include "task_exception_func.h"
#include "task_exception_handler_lite.h"
#include "coll_operator.h"
#include "profiling_handler_lite.h"
using namespace Hccl;

AicpuUtils::AicpuUtils() {}

AicpuUtils& AicpuUtils::GetInstance()
{
    static AicpuUtils instance_;
    return instance_;
}

void AicpuUtils::CreateSingleInstance(void* args) const
{
    auto* kernelParam = reinterpret_cast<HcclKernelParamLite*>(args);
    UbConnLiteMgr::GetInstance();
    AicpuDaemonService::GetInstance();
    TaskExceptionFunc::GetInstance().SetEnable(
        kernelParam->envConfig.taskExceptionEnable); // 根据环境变量使能TaskException
    AicpuCommDestroyFunc::GetInstance();
    TaskExceptionHandlerLite::GetInstance();
    ProfilingHandlerLite::GetInstance();
    DevCapability::GetInstance();
    CommunicatorImplLiteMgr::GetInstance().SetEnvConfig(kernelParam->envConfig); // 初始化并设置Device侧环境变量
}

HcclResult AicpuUtils::Init() const
{
    CHK_RET(ProfilingHandlerLite::GetInstance().Init());
    return HCCL_SUCCESS;
}

HcclResult AicpuUtils::WaitCommFree(CommunicatorImplLite* communicatorImplLite, const char* funcName) const
{
    auto startTime = std::chrono::steady_clock::now();
    constexpr uint32_t pollIntervalUs = 10; // 轮询间隔10us
    constexpr uint32_t pollTimeoutMs = 10;  // 轮询超时时间10ms
    auto waitPollTimeOutMs = std::chrono::milliseconds(pollTimeoutMs);
    unique_lock<std::mutex> aicpuLock(communicatorImplLite->GetAicpuMc2Mutex());
    while (true) {
        if (communicatorImplLite->IsUsed()) {
            if ((std::chrono::steady_clock::now() - startTime) >= waitPollTimeOutMs) {
                HCCL_ERROR(
                    "%s poll timeout, comm id [%u] has been used", funcName, communicatorImplLite->GetCommIdIndex());
                return HCCL_E_TIMEOUT;
            }
            aicpuLock.unlock();
            usleep(pollIntervalUs);
            aicpuLock.lock();
        } else {
            communicatorImplLite->SetIsUsed(true);
            aicpuLock.unlock();
            break;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult AicpuUtils::GetCommHandle(CommunicatorImplLite* communicatorImplLite, void** opHandle) const
{
    // 启动计时，一直获取不到comm.isUsed会退出
    CHK_RET(WaitCommFree(communicatorImplLite, __func__));

    // 默认执行反序列化
    auto reporter = communicatorImplLite->GetProfilingReporterLite();
    CHK_PTR_NULL(reporter);
    reporter->UpdateProfStat();
    if (kernelParam_->op.algOperator.opMode == OpMode::OPBASE) {
        communicatorImplLite->SetCurrentOpMode(kernelParam_->op.algOperator.opMode);
        communicatorImplLite->UpdateCommParam(kernelParam_);
        EXCEPTION_CATCH(communicatorImplLite->UpdateRes(kernelParam_), return HCCL_E_INTERNAL);
    } else {
        HCCL_ERROR(
            "[%s]%s only support opbase, but get opMode %s.", __func__, __func__,
            kernelParam_->op.algOperator.opMode.Describe().c_str());
        return HCCL_E_PARA;
    }

    *opHandle = reinterpret_cast<void*>(communicatorImplLite);
    return HCCL_SUCCESS;
}

int AicpuUtils::GetException(
    StreamLite* curStream, uint32_t flag, CommunicatorImplLite* communicatorImplLite, string additionInfo) const
{
    // 遍历主从流的状态
    auto recvInfo = make_shared<halReportRecvInfo>();
    constexpr uint32_t cqeSize = MAX_REPORT_CNT * sizeof(rtLogicCqReport_t);
    uint8_t tmpAddr[cqeSize] = {}; // cqe byte size
    recvInfo->cqe_addr = tmpAddr;  // 外部保证是有效的地址

    const char* typeStr = (flag == GET_TASK_STATUS) ? "HcclGetTaskStatus" : "HcclPrintTaskExceptionAllComm";

    if (TaskExceptionFunc::GetInstance().GetReporterInfo(curStream, recvInfo) == 1) {
        HCCL_WARNING("[%s]GetReporterInfo execute failed", typeStr);
        return 1;
    }
    uint32_t reportNum = recvInfo->report_cqe_num;
    if (reportNum > MAX_REPORT_CNT) {
        HCCL_WARNING("[%s]report cqe num %u should not be bigger than %u", typeStr, reportNum, MAX_REPORT_CNT);
        return 1;
    }

    if (flag == GET_TASK_STATUS) {
        HCCL_INFO(
            "[%s]Status info:stream %u, head %u, tail %u", __func__, curStream->GetId(),
            curStream->GetRtsq()->GetHead(), curStream->GetRtsq()->GetTail());
        for (uint32_t idx = 0U; idx < reportNum; ++idx) {
            auto& reportOfOne
                = *((reinterpret_cast<rtLogicCqReport_t*>(recvInfo->cqe_addr)) + idx); // 外部保证是有效的地址
            if (TaskExceptionFunc::GetInstance().IsExceptionCqe(reportOfOne)) {
                return 1;
            }
        }
    } else {
        for (uint32_t idx = 0U; idx < reportNum; ++idx) {
            auto& reportOfOne
                = *((reinterpret_cast<rtLogicCqReport_t*>(recvInfo->cqe_addr)) + idx); // 外部保证是有效的地址
            if (TaskExceptionFunc::GetInstance().IsExceptionCqe(reportOfOne)) {
                if (additionInfo != "") {
                    HCCL_ERROR("%s", additionInfo.c_str());
                }
                TaskExceptionHandlerLite::Process(communicatorImplLite, &reportOfOne);
            }
        }
    }
    return 0;
}

void AicpuUtils::GetStreamException(
    StreamLite* curStream, string nullInfo, CommunicatorImplLite* communicatorImplLite, string additionInfo) const
{
    if (curStream == nullptr) {
        HCCL_WARNING("[%s]%s", __func__, nullInfo.c_str());
        return;
    }
    if (communicatorImplLite == nullptr) {
        HCCL_WARNING("[%s]communicatorImplLite is nullptr", __func__);
        return;
    }
    auto* curRtsq = curStream->GetRtsq();
    if (curRtsq == nullptr) {
        HCCL_WARNING("[%s]Stream[%u] rtsq is nullptr.", __func__, curStream->GetId());
        return;
    }
    auto curSqHead = curRtsq->QuerySqHead();
    auto curSqTail = curRtsq->QuerySqTail();

    string finishInfo = "finished";
    if (curSqHead != curSqTail) {
        finishInfo = "unfinished";
        GetException(curStream, GET_EXCEPTION_INFO, communicatorImplLite, additionInfo);
    }
    HCCL_INFO(
        "[%s]Stream %u %s, sq id %u, head %u, tail %u.", __func__, curStream->GetId(), finishInfo.c_str(),
        curStream->GetSqId(), curSqHead, curSqTail);
    return;
}

HcclResult AicpuUtils::HcclLaunchCcore(
    void* opHandle, uint64_t dstAddr, uint32_t turnNum, uint64_t turnNumAddr, bool isLast, int ccoreType) const
{
    const char* typeStr = (ccoreType == CCORE_NOTIFY_TYPE) ? "HcclLaunchCcoreWait" : "HcclLaunchCcorePost";
    HCCL_INFO(
        "[%s]opHandle %p, dstAddr %llu, turnNum %u, turnNumAddr %llu, isLast %u, type %s.", __func__, opHandle, dstAddr,
        turnNum, turnNumAddr, isLast, typeStr);
    if (ccoreType != CCORE_WAIT_TYPE && ccoreType != CCORE_NOTIFY_TYPE) {
        HCCL_ERROR("[%s]Args type %d is not in CCORE_WAIT_TYPE(0) or CCORE_NOTIFY_TYPE(1).", __func__, ccoreType);
        return HCCL_E_PARA;
    }

    CommunicatorImplLite* communicatorImplLite = reinterpret_cast<CommunicatorImplLite*>(opHandle);
    auto* streamLiteMgr = communicatorImplLite->GetStreamLiteMgr();
    CHK_PTR_NULL(streamLiteMgr);

    auto* master = streamLiteMgr->GetMaster();
    CHK_PTR_NULL(master);

    auto* rtsq = master->GetRtsq();
    CHK_PTR_NULL(rtsq);

    if (ccoreType == CCORE_NOTIFY_TYPE) {
        rtsq->CCoreNotifyRecord(dstAddr, turnNumAddr + turnNum * sizeof(uint32_t));
    } else {
        rtsq->CCoreNotifyWait(dstAddr, turnNumAddr + turnNum * sizeof(uint32_t), isLast);
    }
    rtsq->LaunchTask();
    return HCCL_SUCCESS;
}

void AicpuUtils::CalcA2ASendRecvMem(const CollAlgOperator& algOperator, uint64_t& sendSize, uint64_t& recvSize) const
{
    uint64_t sendCount = 0;
    uint64_t recvCount = 0;
    uint32_t sendTypeSize = 0;
    uint32_t recvTypeSize = 0;

    if (algOperator.opType == OpType::ALLTOALLV) {
        for (uint32_t i = 0; i < rankSize_; i++) {
            uint64_t curSendCount = *(static_cast<const uint64_t*>(algOperator.all2AllVDataDes.sendCounts) + i)
                                    + *(static_cast<const uint64_t*>(algOperator.all2AllVDataDes.sdispls) + i);
            sendCount = std::max(sendCount, curSendCount);
            uint64_t curRecvCount = *(static_cast<const uint64_t*>(algOperator.all2AllVDataDes.recvCounts) + i)
                                    + *(static_cast<const uint64_t*>(algOperator.all2AllVDataDes.rdispls) + i);
            recvCount = std::max(recvCount, curRecvCount);
        }
        sendTypeSize = DataTypeSizeGet(algOperator.all2AllVDataDes.sendType);
        recvTypeSize = DataTypeSizeGet(algOperator.all2AllVDataDes.recvType);
    } else if (algOperator.opType == OpType::ALLTOALLVC) {
        for (uint32_t i = 0; i < rankSize_; i++) {
            sendCount += *(
                static_cast<const uint64_t*>(algOperator.all2AllVCDataDes.sendCountMatrix) + myRank_ * rankSize_ + i);
            recvCount += *(
                static_cast<const uint64_t*>(algOperator.all2AllVCDataDes.sendCountMatrix) + myRank_ + rankSize_ * i);
        }
        sendTypeSize = DataTypeSizeGet(algOperator.all2AllVCDataDes.sendType);
        recvTypeSize = DataTypeSizeGet(algOperator.all2AllVCDataDes.recvType);
    } else {
        sendCount = algOperator.all2AllDataDes.sendCount * rankSize_;
        recvCount = algOperator.all2AllDataDes.recvCount * rankSize_;
        sendTypeSize = DataTypeSizeGet(algOperator.all2AllDataDes.sendType);
        recvTypeSize = DataTypeSizeGet(algOperator.all2AllDataDes.recvType);
    }
    sendSize = sendCount * sendTypeSize;
    recvSize = recvCount * recvTypeSize;
    HCCL_INFO(
        "[%s]CalcA2ASendRecvMem finish, algOperator %s, sendCount %llu, sendTypeSize %u, "
        "recvCount %llu, recvTypeSize %u, sendSize %llu, recvSize %llu",
        __func__, algOperator.opType.Describe().c_str(), sendCount, sendTypeSize, recvCount, recvTypeSize, sendSize,
        recvSize);
}
HcclResult
AicpuUtils::ConvertCollOperatorMemV(CollAlgOperator& algOperator, HcclAicpuOpLite& op, const HcclOpData* data) const
{
    auto dataType = HcclDataTypeToDataType(data->dataType);
    CHECK_DATA_TYPE(dataType);
    uint64_t size = DataTypeSizeGet(dataType) * data->dataCount;
    uint64_t* counts = static_cast<uint64_t*>(data->vDataDes.counts);
    uint64_t totalCount = 0;
    for (size_t index = 0; index < rankSize_; index++) {
        totalCount += counts[index];
    }
    uint64_t totalSize = DataTypeSizeGet(dataType) * totalCount;

    if (algOperator.opType == OpType::REDUCESCATTERV) {
        algOperator.inputMem = make_shared<Buffer>(data->input, totalSize);
        op.input.size = totalSize;
    } else {
        algOperator.inputMem = make_shared<Buffer>(data->input, size);
        op.input.size = size;
    }
    if (algOperator.opType == OpType::ALLGATHERV) {
        algOperator.outputMem = make_shared<Buffer>(data->output, totalSize);
        op.output.size = totalSize;
    } else {
        algOperator.outputMem = make_shared<Buffer>(data->output, size);
        op.output.size = size;
    }

    HCCL_INFO(
        "[%s] finish, opType[%s], inputSize[%llu], outputSize[%llu]", __func__, algOperator.opType.Describe().c_str(),
        op.input.size, op.output.size);
    return HCCL_SUCCESS;
}

void AicpuUtils::ConvertCollOperatorMem(
    CollAlgOperator& algOperator, HcclAicpuOpLite& op, const HcclOpData* data, const uint64_t& size) const
{
    if (algOperator.opType == OpType::REDUCESCATTER || algOperator.opType == OpType::SCATTER) {
        algOperator.inputMem = make_shared<Buffer>(data->input, size * rankSize_);
        op.input.size = size * rankSize_;
    } else {
        algOperator.inputMem = make_shared<Buffer>(data->input, size);
        op.input.size = size;
    }
    if (algOperator.opType == OpType::ALLGATHER || algOperator.opType == OpType::GATHER) {
        algOperator.outputMem = make_shared<Buffer>(data->output, size * rankSize_);
        op.output.size = size * rankSize_;
    } else {
        algOperator.outputMem = make_shared<Buffer>(data->output, size);
        op.output.size = size;
    }

    HCCL_INFO(
        "[%s] finish, opType[%s], inputSize[%llu], outputSize[%llu]", __func__, algOperator.opType.Describe().c_str(),
        op.input.size, op.output.size);
}

HcclResult
AicpuUtils::FillCollOperatorMemInfo(CollAlgOperator& algOperator, HcclAicpuOpLite& op, const HcclOpData* data) const
{
    op.input.addr = data->input;
    op.input.tokenId = 0;
    op.input.tokenValue = 0;
    op.output.addr = data->output;
    op.output.tokenId = 0;
    op.output.tokenValue = 0;
    if (algOperator.opType == OpType::ALLTOALL || algOperator.opType == OpType::ALLTOALLV
        || algOperator.opType == OpType::ALLTOALLVC) {
        uint64_t sendSize = 0, recvSize = 0;
        CalcA2ASendRecvMem(algOperator, sendSize, recvSize);
        algOperator.inputMem = make_shared<Buffer>(data->input, sendSize);
        algOperator.outputMem = make_shared<Buffer>(data->output, recvSize);
        op.input.size = sendSize;
        op.output.size = recvSize;
    } else if (algOperator.opType == OpType::BATCHSENDRECV) {
        HCCL_INFO("[%s] OpType::BATCHSENDRECV item = %llu", __func__, algOperator.batchSendRecvDataDes.itemNum);
    } else {
        if (algOperator.opType == OpType::REDUCESCATTERV || algOperator.opType == OpType::ALLGATHERV) {
            return ConvertCollOperatorMemV(algOperator, op, data);
        } else {
            auto tmp = HcclDataTypeToDataType(data->dataType);
            CHECK_DATA_TYPE(tmp);
            uint64_t size = DataTypeSizeGet(tmp) * data->dataCount;
            if (size != 0) {
                HCCL_INFO("[%s] size is %llu", __func__, size);
                ConvertCollOperatorMem(algOperator, op, data, size);
            } else {
                HCCL_WARNING("[%s] data size is 0", __func__);
            }
        }
    }
    HCCL_INFO(
        "[%s]opType %s, op.input.addr %llu, op.input.size %llu, op.output.addr %llu, "
        "op.output.size %llu",
        __func__, algOperator.opType.Describe().c_str(), op.input.addr, op.input.size, op.output.addr, op.output.size);
    return HCCL_SUCCESS;
}

HcclResult AicpuUtils::FillKernelParam(HcclOpData* data) const
{
    kernelParam_->op.algOperator.reduceOp = HcclReduceOpToReduceOp(HCCL_REDUCE_RESERVED);
    if (data->opType == HCCL_CMD_ALLREDUCE || data->opType == HCCL_CMD_REDUCE || data->opType == HCCL_CMD_REDUCE_SCATTER
        || data->opType == HCCL_CMD_REDUCE_SCATTER_V) {
        kernelParam_->op.algOperator.reduceOp = HcclReduceOpToReduceOp(data->reduceOp);
    }
    kernelParam_->op.algOperator.dataType = HcclDataTypeToDataType(data->dataType);
    CHECK_DATA_TYPE(kernelParam_->op.algOperator.dataType);
    kernelParam_->op.algOperator.outputDataType = HcclDataTypeToDataType(data->outputDataType);
    CHECK_DATA_TYPE(kernelParam_->op.algOperator.outputDataType);
    kernelParam_->op.algOperator.dataCount = data->dataCount;
    kernelParam_->op.algOperator.root = data->root;
    kernelParam_->op.algOperator.sendRecvRemoteRank = data->sendRecvRemoteRank;
    HCCL_INFO(
        "[%s]opType=%s, reduceOp=%u, dataType=%u, outputDataType=%u, dataCount=%llu, root=%u, sendRecvRemoteRank=%u",
        __func__, kernelParam_->op.algOperator.opType.Describe().c_str(), data->reduceOp, data->dataType,
        data->outputDataType, data->dataCount, data->root, data->sendRecvRemoteRank);
    if (kernelParam_->op.algOperator.opType == OpType::ALLTOALL) {
        kernelParam_->op.algOperator.all2AllDataDes.recvType = HcclDataTypeToDataType(data->all2AllDataDes.recvType);
        CHECK_DATA_TYPE(kernelParam_->op.algOperator.all2AllDataDes.recvType);
        kernelParam_->op.algOperator.all2AllDataDes.sendType = HcclDataTypeToDataType(data->all2AllDataDes.sendType);
        CHECK_DATA_TYPE(kernelParam_->op.algOperator.all2AllDataDes.sendType);
        kernelParam_->op.algOperator.all2AllDataDes.sendCount = data->all2AllDataDes.sendCount;
        kernelParam_->op.algOperator.all2AllDataDes.recvCount = data->all2AllDataDes.recvCount;
    } else if (kernelParam_->op.algOperator.opType == OpType::ALLTOALLV) {
        kernelParam_->op.algOperator.all2AllVDataDes.sendType = HcclDataTypeToDataType(data->all2AllVDataDes.sendType);
        CHECK_DATA_TYPE(kernelParam_->op.algOperator.all2AllVDataDes.sendType);
        kernelParam_->op.algOperator.all2AllVDataDes.recvType = HcclDataTypeToDataType(data->all2AllVDataDes.recvType);
        CHECK_DATA_TYPE(kernelParam_->op.algOperator.all2AllVDataDes.recvType);
        CHK_PTR_NULL(data->all2AllVDataDes.sendCounts);
        kernelParam_->op.algOperator.all2AllVDataDes.sendCounts = data->all2AllVDataDes.sendCounts;
        CHK_PTR_NULL(data->all2AllVDataDes.recvCounts);
        kernelParam_->op.algOperator.all2AllVDataDes.recvCounts = data->all2AllVDataDes.recvCounts;
        CHK_PTR_NULL(data->all2AllVDataDes.sdispls);
        kernelParam_->op.algOperator.all2AllVDataDes.sdispls = data->all2AllVDataDes.sdispls;
        CHK_PTR_NULL(data->all2AllVDataDes.rdispls);
        kernelParam_->op.algOperator.all2AllVDataDes.rdispls = data->all2AllVDataDes.rdispls;
    } else if (kernelParam_->op.algOperator.opType == OpType::ALLTOALLVC) {
        kernelParam_->op.algOperator.all2AllVCDataDes.sendType
            = HcclDataTypeToDataType(data->all2AllVCDataDes.sendType);
        CHECK_DATA_TYPE(kernelParam_->op.algOperator.all2AllVCDataDes.sendType);
        kernelParam_->op.algOperator.all2AllVCDataDes.recvType
            = HcclDataTypeToDataType(data->all2AllVCDataDes.recvType);
        CHECK_DATA_TYPE(kernelParam_->op.algOperator.all2AllVCDataDes.recvType);
        CHK_PTR_NULL(data->all2AllVCDataDes.sendCountMatrix);
        kernelParam_->op.algOperator.all2AllVCDataDes.sendCountMatrix = data->all2AllVCDataDes.sendCountMatrix;
    } else if (
        kernelParam_->op.algOperator.opType == OpType::ALLGATHERV
        || kernelParam_->op.algOperator.opType == OpType::REDUCESCATTERV) {
        CHK_PTR_NULL(data->vDataDes.counts);
        kernelParam_->op.algOperator.vDataDes.counts = data->vDataDes.counts;
        CHK_PTR_NULL(data->vDataDes.displs);
        kernelParam_->op.algOperator.vDataDes.displs = data->vDataDes.displs;
        kernelParam_->op.algOperator.vDataDes.dataType = HcclDataTypeToDataType(data->vDataDes.dataType);
        CHECK_DATA_TYPE(kernelParam_->op.algOperator.vDataDes.dataType);
    } else if (kernelParam_->op.algOperator.opType == OpType::BATCHSENDRECV) {
        CHK_PTR_NULL(data->batchSendRecvDataDes.sendRecvItemsPtr);
        kernelParam_->op.algOperator.batchSendRecvDataDes.sendRecvItemsPtr
            = data->batchSendRecvDataDes.sendRecvItemsPtr;
        kernelParam_->op.algOperator.dataType = HcclDataTypeToDataType(
            static_cast<HcclSendRecvItem*>(data->batchSendRecvDataDes.sendRecvItemsPtr)->dataType);
        CHECK_DATA_TYPE(kernelParam_->op.algOperator.dataType);
        kernelParam_->op.algOperator.batchSendRecvDataDes.itemNum = data->batchSendRecvDataDes.itemNum;
    } else {
        kernelParam_->op.algOperator.dataDes.dataType = HcclDataTypeToDataType(data->dataDes.dataType);
        CHECK_DATA_TYPE(kernelParam_->op.algOperator.dataDes.dataType);
        kernelParam_->op.algOperator.dataDes.dataCount = data->dataDes.dataCount;
        kernelParam_->op.algOperator.dataDes.strideCount = data->dataDes.strideCount;
    }
    return HCCL_SUCCESS;
}

HcclResult AicpuUtils::RecoverKernelParam(CommunicatorImplLite* communicatorImplLite, HcclOpData* data)
{
    unique_lock<std::shared_timed_mutex> handlerLock(handlerMutex_);
    uint32_t commIdIndex = communicatorImplLite->GetCommIdIndex();
    auto kernelParamIter = kernelParamMap_.find(commIdIndex);
    if (kernelParamIter == kernelParamMap_.end()) {
        HCCL_ERROR(
            "[%s]KernelParam is not found, commId %u, please execute HcclGetCommHandleByCtx first.", __func__,
            commIdIndex);
        return HCCL_E_PTR;
    }
    kernelParam_ = kernelParamIter->second;
    rankSize_ = communicatorImplLite->GetRankSize();
    myRank_ = communicatorImplLite->GetMyRank();

    // 恢复op算子信息，buffer
    if (OP_TYPE_MAP.find(data->opType) == OP_TYPE_MAP.end()) {
        HCCL_ERROR(
            "[%s]Args OP_TYPE_MAP not find data->opType %u, commId %u.", __func__, data->opType,
            communicatorImplLite->GetCommIdIndex());
        return HCCL_E_PARA;
    }
    if (kernelParam_->op.algOperator.opType != OP_TYPE_MAP.at(data->opType)) {
        HCCL_ERROR(
            "[%s]Args kernelParam_->op.algOperator.opType %s is not equal to data->opType %s, commId %u.", __func__,
            kernelParam_->op.algOperator.opType.Describe().c_str(), OP_TYPE_MAP.at(data->opType).Describe().c_str(),
            communicatorImplLite->GetCommIdIndex());
        return HCCL_E_PARA;
    }
    HCCL_INFO(
        "[%s]opHandle %p, commId %u, rankSize_ %u, myRank_ %u, opType %u", __func__, communicatorImplLite,
        communicatorImplLite->GetCommIdIndex(), rankSize_, myRank_, data->opType);
    auto ret = FillKernelParam(data);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[%s]FillKernelParam execute failed, ret %u, commId %u", __func__, ret,
            communicatorImplLite->GetCommIdIndex());
        return ret;
    }
    ret = FillCollOperatorMemInfo(kernelParam_->op.algOperator, kernelParam_->op, data);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[%s]FillCollOperatorMemInfo execute failed, ret %u, commId %u", __func__, ret,
            communicatorImplLite->GetCommIdIndex());
        return ret;
    }
    return HCCL_SUCCESS;
}

HcclResult AicpuUtils::RestoreOpRes(CommunicatorImplLite* communicatorImplLite)
{
    std::shared_lock<std::shared_timed_mutex> sharedLock(handlerMutex_);
    communicatorImplLite->UpdateLocBuffer(kernelParam_);

    uint64_t beginTime = ProfGetCurCpuTimestamp();
    communicatorImplLite->SetDfxOpInfo(beginTime);

    // 使用op信息分配input，output
    communicatorImplLite->UpdateHDCommnicate(kernelParam_);
    communicatorImplLite->RegisterRtsqCallback();
    communicatorImplLite->SetIsCommReady(true);
    return HCCL_SUCCESS;
}

HcclResult AicpuUtils::ExecuteOp(CommunicatorImplLite* communicatorImplLite)
{
    std::shared_lock<std::shared_timed_mutex> sharedLock(handlerMutex_);
    // 修改Orchestrate编排入参
    std::shared_ptr<InsQueue> insQueue = communicatorImplLite->GetInsQueue(kernelParam_);
    sharedLock.unlock();
    CHK_PTR_NULL(insQueue);

    // 执行算子指令队列&&报告任务信息&&报告算子信息
    HCCL_INFO("[%s]DevType is DEV_TYPE_950 or DEV_TYPE_960.", __func__);
    auto* executor = communicatorImplLite->GetInsExecutor();
    CHK_PTR_NULL(executor);
    executor->ExecuteV82(*insQueue, true);

    auto* reporter = communicatorImplLite->GetProfilingReporterLite();
    CHK_PTR_NULL(reporter);
    reporter->ReportAllTasks();

    auto* taskMgr = communicatorImplLite->GetMirrorTaskMgrLite();
    CHK_PTR_NULL(taskMgr);
    ProfilingHandlerLite::GetInstance().ReportHcclOpInfo(*(taskMgr->GetCurrDfxOpInfo()));
    return HCCL_SUCCESS;
}
