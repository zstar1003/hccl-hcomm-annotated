/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cast_utils.h"
#include "ub_mem_transport.h"
#include "serializable.h"
#include "exchange_ub_buffer_dto.h"
#include "exchange_ub_conn_dto.h"
#include "local_ub_rma_buffer.h"
#include "dev_capability.h"
#include "dev_buffer.h"
#include "../../common/dlprof_func_v2.h"
#include "user_remote_mem_getter.h"
#include "exception_util.h"
#include "env_config/env_config_v2.h"

namespace Hccl {
constexpr u32 FINISH_MSG_SIZE = 128;
constexpr char_t FINISH_MSG[FINISH_MSG_SIZE] = "Ub Comm Pipe ready!";
constexpr u32 ONE_MILLISECOND_OF_USLEEP = 1000;

UbMemTransport::UbMemTransport(
    CommonLocRes& commonLocRes, Attribution& attr, const LinkData& linkData, const Socket& socket,
    RdmaHandle rdmaHandle1, LocCntNotifyRes& locCntNotifyRes1, bool isRecvFirst)
    : BaseMemTransport(commonLocRes, attr, linkData, socket, TransportType::UB),
      rdmaHandle(rdmaHandle1),
      locCntNotifyRes(locCntNotifyRes1),
      isRecvFirst_(isRecvFirst)
{
    HCCL_INFO("source: %s", locCntNotifyRes.Describe().c_str());
}

UbMemTransport::UbMemTransport(
    CommonLocRes& commonLocRes, Attribution& attr, const LinkData& linkData, const Socket& socket,
    RdmaHandle rdmaHandle1, LocCntNotifyRes& locCntNotifyRes1,
    std::function<void(u32 streamId, u32 taskId, const TaskParam& taskParam)> callback)
    : BaseMemTransport(commonLocRes, attr, linkData, socket, TransportType::UB, callback),
      rdmaHandle(rdmaHandle1),
      locCntNotifyRes(locCntNotifyRes1)
{
    HCCL_INFO("source: %s", locCntNotifyRes.Describe().c_str());
}

std::string UbMemTransport::Describe() const
{
    string msg = StringFormat(
        "UbMemTransport=[commonLocRes=%s, locCntNotifyRes=%s, ubStatus=%s, ", commonLocRes.Describe().c_str(),
        locCntNotifyRes.Describe().c_str(), ubStatus.Describe().c_str());
    msg += StringFormat("exchangeDataSize=%u, ", exchangeDataSize);
    msg += StringFormat("rmtNotifyNum=%zu, rmtCntNotifyVecNum=%zu]", rmtNotifyVec.size(), rmtCntNotifyVec.size());
    return msg;
}

HcclResult UbMemTransport::BuildDrainResource()
{
    // notify作为read的落点
    if (drainNotify_ == nullptr && !isHost_) {
        EXCEPTION_CATCH(drainNotify_ = std::make_unique<Hccl::UbLocalNotify>(rdmaHandle, !isHost_), return HCCL_E_PTR);
        HCCL_INFO("[UbMemTransport][%s] drain notify created: %s", __func__, drainNotify_->Describe().c_str());
    }

    // 常量1内存供远端读取
    if (drainBuffer_ == nullptr && !isHost_) {
        u32 notifySize = Hccl::DevCapability::GetInstance().GetNotifySize();

        std::shared_ptr<Hccl::DevBuffer> constMem;
        EXCEPTION_CATCH(constMem = std::make_shared<Hccl::DevBuffer>(notifySize), return HCCL_E_PTR);

        Hccl::HrtMemcpy(
            ReinterpretAs<void*>(constMem->GetAddr()), constMem->GetSize(), &NORMAL_NOTIFY_VAL,
            sizeof(NORMAL_NOTIFY_VAL), RT_MEMCPY_HOST_TO_DEVICE);

        EXCEPTION_CATCH(
            drainBuffer_ = std::make_unique<Hccl::LocalUbRmaBuffer>(constMem, rdmaHandle), return HCCL_E_PTR);
        HCCL_INFO(
            "[UbMemTransport][%s] drain buffer created: addr[0x%llx], size[%zu]", __func__,
            static_cast<unsigned long long>(drainBuffer_->GetAddr()), drainBuffer_->GetSize());
    }

    return HCCL_SUCCESS;
}

HcclResult UbMemTransport::Describe(std::string& dfxMsg)
{
    HCCL_INFO("UbMemTransport Describe connNum[%u]", connNum);
    for (u32 i = 0; i < connNum; i++) {
        CHK_RET(commonLocRes.connVec[i]->Describe(dfxMsg));
    }
    return HCCL_SUCCESS;
}

MemoryBuffer UbMemTransport::GetLocMemBuffer(const RmaBufferSlice& locSlice) const
{
    return MemoryBuffer(locSlice.addr, locSlice.size, locSlice.buf->GetMemHandle());
}

MemoryBuffer UbMemTransport::GetRmtMemBuffer(const RmtRmaBufferSlice& rmtSlice) const
{
    return MemoryBuffer(rmtSlice.addr, rmtSlice.size, rmtSlice.buf->GetMemHandle());
}

MemoryBuffer UbMemTransport::GetRmtNotifyMemBuffer(u32 index)
{
    return MemoryBuffer(
        rmtNotifyVec[index]->GetAddr(), rmtNotifyVec[index]->GetSize(), rmtNotifyVec[index]->GetMemHandle());
}

MemoryBuffer UbMemTransport::GetRmtCntNotifyMemBuffer(const WithNotifyIn& withNotify)
{
    auto index = withNotify.index_;
    return MemoryBuffer(
        rmtCntNotifyVec[index]->GetAddr(), rmtCntNotifyVec[index]->GetSize(), rmtCntNotifyVec[index]->GetMemHandle());
}

static void SubmitTask(const TaskUbDbSend& ubSend, const Stream& stream)
{
    HCCL_INFO("SubmitTask UbDbSend ");
    HrtUbDbInfo info;
    info.dbNum = 1;
    info.wrCqe = 0; // 默认值是0 不会cqe  如果传1，驱动分发，会给hccl cqe，用于维护ci指针。
    info.info[0].functionId = ubSend.GetFuncId();
    info.info[0].dieId = ubSend.GetDieId();
    info.info[0].jettyId = ubSend.GetJettyId();
    info.info[0].piValue = ubSend.GetPiVal();
    HrtUbDbSend(info, stream.GetPtr());
}

static void SubmitTask(const TaskUbDirectSend& ubDirectSend, const Stream& stream)
{
    HCCL_INFO("SubmitTask UbDirectSend");
    if (ubDirectSend.GetDwqeSize() != DWQE_SIZE_64 && ubDirectSend.GetDwqeSize() != DWQE_SIZE_128) {
        std::string msg
            = StringFormat("dwqe size is not valid, cannot submit task, dwqeSize=%u", ubDirectSend.GetDwqeSize());
        THROW<InternalException>(msg);
    }
    HrtUbWqeInfo info;
    info.wrCqe = 0;
    info.functionId = ubDirectSend.GetFuncId();
    info.dieId = ubDirectSend.GetDieId();
    info.jettyId = ubDirectSend.GetJettyId();
    info.wqe = const_cast<u8*>(ubDirectSend.GetDwqePtr());
    info.wqePtrLen = ubDirectSend.GetDwqeSize();
    info.wqeSize = info.wqePtrLen == DWQE_SIZE_64 ? 0 : 1;
    HrtUbDirectSend(info, stream.GetPtr());
}

static void SubmitTask(const TaskWriteValue& taskWriteValue, const Stream& stream)
{
    HCCL_INFO("begin HrtWriteValue");
    HrtWriteValue(taskWriteValue.GetDbAddr(), taskWriteValue.GetPiVal(), stream.GetPtr());
    HCCL_INFO("finished HrtWriteValue");
}

template <typename TaskType>
std::function<void(const BaseTask&, const Stream&)> GetSubmitUbTaskFunction()
{
    return [](const BaseTask& task, const Stream& stream) {
        SubmitTask(static_cast<const TaskType&>(task), stream);
    };
}

std::map<TaskType, std::function<void(const BaseTask&, const Stream&)>> g_ubTaskSubmitRuleMap
    = {{TaskType::UB_SEND, GetSubmitUbTaskFunction<TaskUbDbSend>()},
       {TaskType::UB_DIRECT_SEND, GetSubmitUbTaskFunction<TaskUbDirectSend>()},
       {TaskType::WRITE_VALUE, GetSubmitUbTaskFunction<TaskWriteValue>()}};

static void SubmitUbTask(unique_ptr<BaseTask> task, const Stream& stream)
{
    if (task != nullptr) {
        g_ubTaskSubmitRuleMap.at(task->GetType())(*task.get(), stream);
    }
}

void UbMemTransport::SubmitNotify(const MemoryBuffer& rmtNotify, u64 data, const Stream& stream)
{
    SqeConfig config;
    SubmitUbTask(commonLocRes.connVec[0]->PrepareInlineWrite(rmtNotify, data, config), stream);
}

void UbMemTransport::Post(u32 index, const Stream& stream)
{
    TaskParam taskParam{};
    taskParam.beginTime = DlProfFunc::GetInstance().dlMsprofSysCycleTime();

    SubmitNotify(GetRmtNotifyMemBuffer(index), NORMAL_NOTIFY_VAL, stream);

    taskParam.taskType = TaskParamType::TASK_NOTIFY_RECORD;
    taskParam.endTime = DlProfFunc::GetInstance().dlMsprofSysCycleTime();
    ;
    taskParam.taskPara.Notify.notifyID = rmtNotifyVec[index]->GetAddr();
    taskParam.taskPara.Notify.value = NORMAL_NOTIFY_VAL;

    SaveDfxTaskInfo(taskParam);
}

void UbMemTransport::Wait(u32 index, const Stream& stream, u32 timeout)
{
    TaskParam taskParam{};
    taskParam.beginTime = DlProfFunc::GetInstance().dlMsprofSysCycleTime();

    commonLocRes.notifyVec[index]->Wait(stream, timeout);

    taskParam.taskType = TaskParamType::TASK_NOTIFY_WAIT;
    taskParam.endTime = DlProfFunc::GetInstance().dlMsprofSysCycleTime();
    taskParam.taskPara.Notify.notifyID = commonLocRes.notifyVec[index]->GetNotify()->GetId();
    taskParam.taskPara.Notify.value = NORMAL_NOTIFY_VAL;

    SaveDfxTaskInfo(taskParam);
}

void UbMemTransport::Read(const RmaBufferSlice& locSlice, const RmtRmaBufferSlice& rmtSlice, const Stream& stream)
{
    TaskParam taskParam{};
    taskParam.beginTime = DlProfFunc::GetInstance().dlMsprofSysCycleTime();

    SqeConfig config;
    config.wqeMode = WqeMode::DWQE;
    SubmitUbTask(
        commonLocRes.connVec[0]->PrepareRead(GetRmtMemBuffer(rmtSlice), GetLocMemBuffer(locSlice), config), stream);

    taskParam.taskType = TaskParamType::TASK_RDMA;
    taskParam.endTime = DlProfFunc::GetInstance().dlMsprofSysCycleTime();
    taskParam.taskPara.DMA.src = ReinterpretAs<const void*>(locSlice.addr);
    taskParam.taskPara.DMA.dst = ReinterpretAs<const void*>(rmtSlice.addr);
    taskParam.taskPara.DMA.size = rmtSlice.size;
    taskParam.taskPara.DMA.notifyID = INVALID_VALUE_NOTIFYID;
    taskParam.taskPara.DMA.linkType = DfxLinkType::UB;
    taskParam.taskPara.DMA.dmaOp = DmaOp::HCCL_DMA_READ;
    SaveDfxTaskInfo(taskParam);
}

void UbMemTransport::ReadReduce(
    const RmaBufferSlice& locSlice, const RmtRmaBufferSlice& rmtSlice, const ReduceIn& reduceIn, const Stream& stream)
{
    TaskParam taskParam{};
    taskParam.beginTime = DlProfFunc::GetInstance().dlMsprofSysCycleTime();

    SqeConfig config;
    config.wqeMode = WqeMode::DWQE;
    SubmitUbTask(
        commonLocRes.connVec[0]->PrepareReadReduce(
            GetRmtMemBuffer(rmtSlice), GetLocMemBuffer(locSlice), reduceIn.dataType, reduceIn.reduceOp, config),
        stream);

    taskParam.taskType = TaskParamType::TASK_UB_REDUCE_INLINE;
    taskParam.endTime = DlProfFunc::GetInstance().dlMsprofSysCycleTime();
    taskParam.taskPara.DMA.src = ReinterpretAs<const void*>(locSlice.addr);
    taskParam.taskPara.DMA.dst = ReinterpretAs<const void*>(rmtSlice.addr);
    taskParam.taskPara.DMA.size = rmtSlice.size;
    taskParam.taskPara.DMA.notifyID = INVALID_VALUE_NOTIFYID;
    taskParam.taskPara.DMA.linkType = DfxLinkType::UB;
    taskParam.taskPara.DMA.dmaOp = DmaOp::HCCL_DMA_READ;

    SaveDfxTaskInfo(taskParam);
}

void UbMemTransport::Write(const RmaBufferSlice& locSlice, const RmtRmaBufferSlice& rmtSlice, const Stream& stream)
{
    TaskParam taskParam{};
    taskParam.beginTime = DlProfFunc::GetInstance().dlMsprofSysCycleTime();

    SqeConfig config;
    config.wqeMode = WqeMode::DWQE;
    SubmitUbTask(
        commonLocRes.connVec[0]->PrepareWrite(GetRmtMemBuffer(rmtSlice), GetLocMemBuffer(locSlice), config), stream);
    taskParam.taskType = TaskParamType::TASK_RDMA;
    taskParam.endTime = DlProfFunc::GetInstance().dlMsprofSysCycleTime();
    taskParam.taskPara.DMA.src = ReinterpretAs<const void*>(locSlice.addr);
    taskParam.taskPara.DMA.dst = ReinterpretAs<const void*>(rmtSlice.addr);
    taskParam.taskPara.DMA.size = locSlice.size;
    taskParam.taskPara.DMA.notifyID = INVALID_VALUE_NOTIFYID;
    taskParam.taskPara.DMA.linkType = DfxLinkType::UB;
    taskParam.taskPara.DMA.dmaOp = DmaOp::HCCL_DMA_WRITE;

    SaveDfxTaskInfo(taskParam);
}

void UbMemTransport::WriteReduce(
    const RmaBufferSlice& locSlice, const RmtRmaBufferSlice& rmtSlice, const ReduceIn& reduceIn, const Stream& stream)
{
    TaskParam taskParam{};
    taskParam.beginTime = DlProfFunc::GetInstance().dlMsprofSysCycleTime();

    SqeConfig config;
    config.wqeMode = WqeMode::DWQE;
    SubmitUbTask(
        commonLocRes.connVec[0]->PrepareWriteReduce(
            GetRmtMemBuffer(rmtSlice), GetLocMemBuffer(locSlice), reduceIn.dataType, reduceIn.reduceOp, config),
        stream);

    taskParam.taskType = TaskParamType::TASK_UB_REDUCE_INLINE;
    taskParam.endTime = DlProfFunc::GetInstance().dlMsprofSysCycleTime();
    taskParam.taskPara.DMA.src = ReinterpretAs<const void*>(locSlice.addr);
    taskParam.taskPara.DMA.dst = ReinterpretAs<const void*>(rmtSlice.addr);
    taskParam.taskPara.DMA.size = locSlice.size;
    taskParam.taskPara.DMA.notifyID = INVALID_VALUE_NOTIFYID;
    taskParam.taskPara.DMA.linkType = DfxLinkType::UB;
    taskParam.taskPara.DMA.dmaOp = DmaOp::HCCL_DMA_WRITE;

    SaveDfxTaskInfo(taskParam);
}

void UbMemTransport::WriteWithNotify(
    const RmaBufferSlice& locSlice, const RmtRmaBufferSlice& rmtSlice, const WithNotifyIn& withNotify,
    const Stream& stream)
{
    if (locSlice.size == 0) {
        return SubmitWriteEmptyWithNotify(withNotify, stream);
    }

    if (withNotify.notifyType_ == TransportNotifyType::NORMAL) {
        return SubmitWriteWithNotify(
            GetRmtMemBuffer(rmtSlice), GetLocMemBuffer(locSlice), NORMAL_NOTIFY_VAL,
            GetRmtNotifyMemBuffer(withNotify.index_), stream);
    } else if (withNotify.notifyType_ == TransportNotifyType::COUNT) {
        return SubmitWriteWithNotify(
            GetRmtMemBuffer(rmtSlice), GetLocMemBuffer(locSlice), withNotify.userData_,
            GetRmtCntNotifyMemBuffer(withNotify), stream);
    } else {
        std::string msg = StringFormat("%s error", withNotify.Describe().c_str());
        THROW<InternalException>(msg);
    }
}

void UbMemTransport::WriteReduceWithNotify(
    const RmaBufferSlice& locSlice, const RmtRmaBufferSlice& rmtSlice, const ReduceIn& reduceIn,
    const WithNotifyIn& withNotify, const Stream& stream)
{
    if (locSlice.size == 0) {
        return SubmitWriteEmptyWithNotify(withNotify, stream);
    }

    if (withNotify.notifyType_ == TransportNotifyType::NORMAL) {
        SubmitWriteReduceWithNotify(
            GetRmtMemBuffer(rmtSlice), GetLocMemBuffer(locSlice), reduceIn, NORMAL_NOTIFY_VAL,
            GetRmtNotifyMemBuffer(withNotify.index_), stream);
    } else if (withNotify.notifyType_ == TransportNotifyType::COUNT) {
        SubmitWriteReduceWithNotify(
            GetRmtMemBuffer(rmtSlice), GetLocMemBuffer(locSlice), reduceIn, withNotify.userData_,
            GetRmtCntNotifyMemBuffer(withNotify), stream);
    } else {
        std::string msg = StringFormat("%s error", withNotify.Describe().c_str());
        THROW<InternalException>(msg);
    }
}

void UbMemTransport::SubmitWriteEmptyWithNotify(const WithNotifyIn& withNotify, const Stream& stream)
{
    TaskParam taskParam{};
    taskParam.beginTime = DlProfFunc::GetInstance().dlMsprofSysCycleTime();
    u32 value = NORMAL_NOTIFY_VAL;

    if (withNotify.notifyType_ == TransportNotifyType::NORMAL) {
        SubmitNotify(GetRmtNotifyMemBuffer(withNotify.index_), NORMAL_NOTIFY_VAL, stream);
    } else if (withNotify.notifyType_ == TransportNotifyType::COUNT) {
        SubmitNotify(GetRmtCntNotifyMemBuffer(withNotify), withNotify.userData_, stream);
        value = withNotify.userData_;
    } else {
        std::string msg = StringFormat("%s error", withNotify.Describe().c_str());
        THROW<InternalException>(msg);
    }

    taskParam.taskType = TaskParamType::TASK_NOTIFY_WAIT;
    taskParam.endTime = DlProfFunc::GetInstance().dlMsprofSysCycleTime();
    taskParam.taskPara.Notify.notifyID = INVALID_VALUE_NOTIFYID;
    taskParam.taskPara.Notify.value = value;

    SaveDfxTaskInfo(taskParam);
}

void UbMemTransport::SubmitWriteWithNotify(
    const MemoryBuffer& rmt, const MemoryBuffer& loc, u64 data, const MemoryBuffer& rmtNotify, const Stream& stream)
{
    TaskParam taskParam{};
    taskParam.beginTime = DlProfFunc::GetInstance().dlMsprofSysCycleTime();

    SqeConfig config;
    config.wqeMode = WqeMode::DWQE;
    SubmitUbTask(commonLocRes.connVec[0]->PrepareWriteWithNotify(rmt, loc, data, rmtNotify, config), stream);

    taskParam.taskType = TaskParamType::TASK_WRITE_WITH_NOTIFY;
    taskParam.endTime = DlProfFunc::GetInstance().dlMsprofSysCycleTime();
    taskParam.taskPara.DMA.src = ReinterpretAs<const void*>(loc.addr);
    taskParam.taskPara.DMA.dst = ReinterpretAs<const void*>(rmt.addr);
    taskParam.taskPara.DMA.size = loc.size;
    taskParam.taskPara.DMA.notifyID = INVALID_VALUE_NOTIFYID;
    taskParam.taskPara.DMA.linkType = DfxLinkType::UB;
    taskParam.taskPara.DMA.dmaOp = DmaOp::HCCL_DMA_WRITE;

    SaveDfxTaskInfo(taskParam);
}

void UbMemTransport::SubmitWriteReduceWithNotify(
    const MemoryBuffer& rmt, const MemoryBuffer& loc, const ReduceIn& reduceIn, u64 data, const MemoryBuffer& rmtNotify,
    const Stream& stream)
{
    TaskParam taskParam{};
    taskParam.beginTime = DlProfFunc::GetInstance().dlMsprofSysCycleTime();

    SqeConfig config;
    config.wqeMode = WqeMode::DWQE;
    SubmitUbTask(
        commonLocRes.connVec[0]->PrepareWriteReduceWithNotify(
            rmt, loc, reduceIn.dataType, reduceIn.reduceOp, data, rmtNotify, config),
        stream);

    taskParam.taskType = TaskParamType::TASK_WRITE_REDUCE_WITH_NOTIFY;
    taskParam.endTime = DlProfFunc::GetInstance().dlMsprofSysCycleTime();
    taskParam.taskPara.DMA.src = ReinterpretAs<const void*>(loc.addr);
    taskParam.taskPara.DMA.dst = ReinterpretAs<const void*>(rmt.addr);
    taskParam.taskPara.DMA.size = loc.size;
    taskParam.taskPara.DMA.notifyID = INVALID_VALUE_NOTIFYID;
    taskParam.taskPara.DMA.linkType = DfxLinkType::UB;
    taskParam.taskPara.DMA.dmaOp = DmaOp::HCCL_DMA_WRITE;

    SaveDfxTaskInfo(taskParam);
}

bool UbMemTransport::IsResReady()
{
    for (auto& it : commonLocRes.connVec) {
        CHECK_NULLPTR(it, StringFormat("[UbMemTransport::%s] failed, connection pointer is nullptr", __func__));

        RmaConnType connType = it->GetRmaConnType();
        if (connType != RmaConnType::UB) {
            THROW<InternalException>(
                "[UbMemTransport::%s] connection type[%s] is not ub", __func__, connType.Describe().c_str());
        }

        auto status = it->GetStatus();
        if (status != RmaConnStatus::EXCHANGEABLE && status != RmaConnStatus::READY) {
            return false;
        }
    }

    HCCL_INFO("[UbMemTransport::IsResReady] all resources ready.");
    return true;
}

bool UbMemTransport::IsConnsReady()
{
    for (u32 i = 0; i < connNum; i++) {
        if (commonLocRes.connVec[i]->GetStatus() != RmaConnStatus::READY) {
            return false;
        }
    }
    HCCL_INFO("conns are ready.");
    return true;
}

HcclResult UbMemTransport::StatusMachine()
{
    TRY_CATCH_RETURN(
        if (socket == nullptr) {
            HCCL_ERROR("[UbMemTransport][StatusMachine]socket is nullptr, please check");
            return HcclResult::HCCL_E_INTERNAL;
        } SocketStatus socketStatus
        = isHost_ ? socket->GetStatus() : socket->GetAsyncStatus();
        if (socketStatus == Hccl::SocketStatus::INIT || socketStatus == Hccl::SocketStatus::TIMEOUT) {
            HCCL_ERROR("[UbMemTransport][StatusMachine]socket timeout or no link, please check");
            return HcclResult::HCCL_E_INTERNAL;
        }

        if (socketStatus != Hccl::SocketStatus::OK) {
            SaluSleep(ONE_MILLISECOND_OF_USLEEP); // 防止get sockets冲高CtrlCPU
            return HcclResult::HCCL_SUCCESS;      // 操作成功，保持当前状态
        } switch (ubStatus) {
            case UbStatus::INIT:
                CHK_RET(HandleInitStatus());
                break;
            case UbStatus::SEND_DATA:
                CHK_RET(HandleSendAllStatus());
                break;
            case UbStatus::RECV_SIZE:
                CHK_RET(HandleRecvSizeStatus());
                break;
            case UbStatus::RECV_DATA:
                CHK_RET(HandleRecvDataStatus());
                break;
            case UbStatus::PROCESS_DATA:
                CHK_RET(HandleProcessDataStatus());
                break;
            case UbStatus::SEND_FIN:
                CHK_RET(HandleSendFinStatus());
                break;
            case UbStatus::RECV_FIN:
                CHK_RET(HandleRecvFinStatus());
                break;
            case UbStatus::SET_READY:
                CHK_RET(HandleSetReadyStatus());
                break;
            default:
                break;
        });
    return HCCL_SUCCESS;
}

HcclResult UbMemTransport::HandleInitStatus()
{
    ubStatus = isRecvFirst_ ? UbStatus::RECV_SIZE : UbStatus::SEND_DATA;
    baseStatus = TransportStatus::SOCKET_OK;
    return HCCL_SUCCESS;
}

HcclResult UbMemTransport::HandleSendAllStatus()
{
    if (IsResReady()) {
        CHK_RET(SendAll());
        ubStatus = isRecvFirst_ ? UbStatus::PROCESS_DATA : UbStatus::RECV_SIZE;
    }
    return HCCL_SUCCESS;
}

HcclResult UbMemTransport::HandleRecvSizeStatus()
{
    CHK_RET(RecvDataSize());
    ubStatus = UbStatus::RECV_DATA;
    return HCCL_SUCCESS;
}

HcclResult UbMemTransport::HandleRecvDataStatus()
{
    CHK_RET(RecvExchangeData());
    ubStatus = isRecvFirst_ ? UbStatus::SEND_DATA : UbStatus::PROCESS_DATA;
    return HCCL_SUCCESS;
}

HcclResult UbMemTransport::HandleProcessDataStatus()
{
    bool needSendFinish = false;
    CHK_RET(RecvDataProcess(needSendFinish));
    if (needSendFinish) {
        ubStatus = UbStatus::SEND_FIN;
    } else {
        SetBaseStatusReady();
        ubStatus = UbStatus::READY;
    }
    return HCCL_SUCCESS;
}

HcclResult UbMemTransport::HandleSendFinStatus()
{
    if (IsConnsReady()) {
        CHK_RET(SendFinish());
        ubStatus = UbStatus::RECV_FIN;
    }
    return HCCL_SUCCESS;
}

HcclResult UbMemTransport::HandleRecvFinStatus()
{
    CHK_RET(RecvFinish());
    ubStatus = UbStatus::SET_READY;
    return HCCL_SUCCESS;
}

HcclResult UbMemTransport::HandleSetReadyStatus()
{
    SetBaseStatusReady();
    ubStatus = UbStatus::READY;
    return HCCL_SUCCESS;
}

TransportStatus UbMemTransport::GetStatus()
{
    if (baseStatus == TransportStatus::READY || baseStatus == TransportStatus::CONNECT_FAILED
        || baseStatus == TransportStatus::SOCKET_TIMEOUT) {
        return baseStatus;
    }

    HcclResult ret = StatusMachine();
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[UbMemTransport::GetStatus] StatusMachine failed, ret=%d", ret);
        baseStatus = TransportStatus::CONNECT_FAILED;
    }
    return baseStatus;
}

HcclResult UbMemTransport::SendAll()
{
    notifyNum = commonLocRes.notifyVec.size();
    bufferNum = commonLocRes.bufferVec.size();
    connNum = commonLocRes.connVec.size();
    cntNotifyNum = locCntNotifyRes.vec.size();

    cntNotifyDescSize = locCntNotifyRes.desc.size();

    HCCL_INFO(
        "notifyNum=%u, bufferNum=%u, connNum=%u, cntNotifyNum=%u, cntNotifyDescSize=%u, isHost_[%d]", notifyNum,
        bufferNum, connNum, cntNotifyNum, cntNotifyDescSize, isHost_);

    BinaryStream binaryStream;
    HandshakeMsgPack(binaryStream);
    NotifyVecPack(binaryStream);
    BufferVecPack(binaryStream, commonLocRes.bufferVec);
    CntNotifyVecPack(binaryStream);
    CntNotifyDescPack(binaryStream);
    CHK_RET(DrainBufPack(binaryStream));
    ConnVecPack(binaryStream);

    sendDataPack_.resize(sizeof(u32));
    binaryStream.Dump(sendDataPack_);
    u32 dataSize = sendDataPack_.size() - sizeof(u32);
    CHK_SAFETY_FUNC_RET(memcpy_s(sendDataPack_.data(), sizeof(u32), &dataSize, sizeof(u32)));

    bool ret = false;
    if (isHost_) {
        ret = socket->Send(sendDataPack_.data(), sendDataPack_.size());
    } else {
        socket->SendAsync(sendDataPack_.data(), sendDataPack_.size());
        ret = true;
    }
    if (!ret) {
        HCCL_ERROR("[UbMemTransport::SendAll] Send failed");
        return HCCL_E_INTERNAL;
    }
    HCCL_INFO("[UbMemTransport::%s] Send size[%zu] of data success.", __func__, sendDataPack_.size());
    return HCCL_SUCCESS;
}

HcclResult UbMemTransport::RecvDataSize()
{
    // 接收数据包尺寸
    bool ret = false;
    HCCL_DEBUG("Starting to recv message size[%zu] bytes, isHost_[%d]", sizeof(exchangeDataSize), isHost_);
    if (isHost_) {
        ret = socket->Recv(&exchangeDataSize, sizeof(exchangeDataSize));
    } else {
        socket->RecvAsync(ReinterpretAs<u8*>(&exchangeDataSize), sizeof(exchangeDataSize));
        ret = true;
    }
    if (!ret) {
        HCCL_ERROR("[UbMemTransport::RecvDataSize] Recv size failed");
        return HCCL_E_INTERNAL;
    }
    HCCL_INFO(
        "[UbMemTransport::%s] Receive size[%u] of data success. [%zu] bytes received.", __func__, exchangeDataSize,
        sizeof(exchangeDataSize));
    return HCCL_SUCCESS;
}

HcclResult UbMemTransport::SendExchangeData()
{
    bool ret = false;
    HCCL_DEBUG("Starting to send message size[%zu] bytes, isHost_[%d]", sendData.size(), isHost_);
    if (isHost_) {
        ret = socket->Send(sendData.data(), sendData.size());
    } else {
        socket->SendAsync(sendData.data(), sendData.size());
        ret = true;
    }
    if (!ret) {
        HCCL_ERROR("[UbMemTransport::SendExchangeData] Send data failed");
        return HCCL_E_INTERNAL;
    }
    HCCL_INFO("send data %s, size=%zu", GetLinkDescInfo().c_str(), sendData.size());
    return HCCL_SUCCESS;
}

HcclResult UbMemTransport::RecvExchangeData()
{
    recvData.resize(exchangeDataSize);
    bool ret = false;
    HCCL_DEBUG("Starting to recv message size[%zu] bytes, isHost_[%d]", recvData.size(), isHost_);
    if (isHost_) {
        ret = socket->Recv(recvData.data(), recvData.size());
    } else {
        socket->RecvAsync(ReinterpretAs<u8*>(recvData.data()), recvData.size());
        ret = true;
    }
    if (!ret) {
        HCCL_ERROR("[UbMemTransport::RecvExchangeData] Recv data failed");
        return HCCL_E_INTERNAL;
    }

    HCCL_INFO("recv data %s, size=%zu", GetLinkDescInfo().c_str(), recvData.size());
    return HCCL_SUCCESS;
}

HcclResult UbMemTransport::RecvDataProcess(bool& needSendFinish)
{
    HCCL_INFO(
        "RecvDataProcess: link=%s, size=%zu, exchangeDataSize=%u", GetLinkDescInfo().c_str(), recvData.size(),
        exchangeDataSize);
    BinaryStream binaryStream(recvData);
    HcclResult ret = HandshakeMsgUnpack(binaryStream);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[UbMemTransport::RecvDataProcess] HandshakeMsgUnpack failed, ret=%d", ret);
        return ret;
    }

    ret = RmtBufferVecUnpackProc(notifyNum, binaryStream, rmtNotifyVec, UbRmtBufType::NOTIFY);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[UbMemTransport::RecvDataProcess] RmtBufferVecUnpackProc notify failed, ret=%d", ret);
        return ret;
    }

    ret = RmtBufferVecUnpackProc(bufferNum, binaryStream, rmtBufferVec, UbRmtBufType::BUFFER);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[UbMemTransport::RecvDataProcess] RmtBufferVecUnpackProc buffer failed, ret=%d", ret);
        return ret;
    }

    ret = RmtBufferVecUnpackProc(cntNotifyNum, binaryStream, rmtCntNotifyVec, UbRmtBufType::CNT_NOTIFY);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[UbMemTransport::RecvDataProcess] RmtBufferVecUnpackProc cntNotify failed, ret=%d", ret);
        return ret;
    }

    ret = CntNotifyDescUnpack(binaryStream);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[UbMemTransport::RecvDataProcess] CntNotifyDescUnpack failed, ret=%d", ret);
        return ret;
    }

    ret = DrainBufUnpack(binaryStream);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[UbMemTransport::RecvDataProcess] DrainBufUnpack failed, ret=%d", ret);
        return ret;
    }

    ret = ConnVecUnpackProc(binaryStream, needSendFinish);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[UbMemTransport::RecvDataProcess] ConnVecUnpackProc failed, ret=%d", ret);
        return ret;
    }

    return HCCL_SUCCESS;
}

void UbMemTransport::BufferVecPack(BinaryStream& binaryStream, std::vector<LocalRmaBuffer*>& bufferVec)
{
    binaryStream << static_cast<u32>(bufferVec.size());
    HCCL_INFO("start pack %s bufferVec", transportType.Describe().c_str());
    u32 pos = 0;
    for (auto& it : bufferVec) {
        binaryStream << pos;
        if (it != nullptr) { // 非空的buffer，从buffer中获取 dto
            std::unique_ptr<Serializable> dto = it->GetExchangeDto();
            dto->Serialize(binaryStream);
            HCCL_INFO("pack buffer pos=%u dto %s", pos, dto->Describe().c_str());
        } else { // 空的buffer，dto所有字段为0(size=0)
            ExchangeUbBufferDto exchangeDto;
            exchangeDto.Serialize(binaryStream);
            HCCL_INFO("pack buffer pos=%u, dto is null %s", pos, exchangeDto.Describe().c_str());
        }
        pos++;
    }
}

void UbMemTransport::CntNotifyVecPack(BinaryStream& binaryStream)
{
    binaryStream << cntNotifyNum;
    HCCL_INFO("pack UB cntNotify num=%u, %s", cntNotifyNum, GetLinkDescInfo().c_str());
    u32 pos = 0;
    for (auto& it : locCntNotifyRes.vec) {
        binaryStream << pos;
        std::unique_ptr<Serializable> dto = it->GetExchangeDto();
        dto->Serialize(binaryStream);
        HCCL_INFO("pack cntNotify pos=%u, dto %s", pos, dto->Describe().c_str());
        pos++;
    }
}

void UbMemTransport::CntNotifyDescPack(BinaryStream& binaryStream)
{
    binaryStream << cntNotifyDescSize;
    HCCL_INFO("pack cntNotify desc size=%u %s", cntNotifyDescSize, GetLinkDescInfo().c_str());
    HCCL_INFO("pack cntNotify desc =%s", Bytes2hex(locCntNotifyRes.desc.data(), locCntNotifyRes.desc.size()).c_str());
    for (auto& it : locCntNotifyRes.desc) {
        binaryStream << it;
    }
}

HcclResult UbMemTransport::DrainBufPack(BinaryStream& binaryStream)
{
    // 打包交换信息前，进行资源创建
    CHK_RET(BuildDrainResource());

    // 只需交换 常量buffer信息 供远端读
    HCCL_INFO("start pack drain buffer");
    if (drainBuffer_ != nullptr) { // 非空的buffer，从buffer中获取 dto
        std::unique_ptr<Serializable> dto = drainBuffer_->GetExchangeDto();
        dto->Serialize(binaryStream);
        HCCL_INFO("pack drain buffer dto %s", dto->Describe().c_str());
    } else { // 空的buffer，dto所有字段为0(size=0)
        ExchangeUbBufferDto exchangeDto;
        exchangeDto.Serialize(binaryStream);
        HCCL_INFO("pack drain buffer, dto is null %s", exchangeDto.Describe().c_str());
    }

    return HCCL_SUCCESS;
}

HcclResult UbMemTransport::DrainBufUnpack(BinaryStream& binaryStream)
{
    HCCL_INFO("start unpack drain buffer");
    ExchangeUbBufferDto dto;
    dto.Deserialize(binaryStream);

    if (dto.size == 0) {
        rmtDrainBuffer_ = nullptr;
        HCCL_WARNING("unpack drain buffer dto is null");
    } else {
        rmtDrainBuffer_ = std::make_unique<RemoteUbRmaBuffer>(rdmaHandle, dto);
        HCCL_INFO("unpack drain buffer rmtDrainBuffer=%s", rmtDrainBuffer_->Describe().c_str());
    }

    return HCCL_SUCCESS;
}

HcclResult UbMemTransport::CntNotifyDescUnpack(BinaryStream& binaryStream)
{
    u32 descSize;
    binaryStream >> descSize;
    if (descSize != cntNotifyDescSize) {
        HCCL_ERROR(
            "[UbMemTransport::CntNotifyDescUnpack] size=%u is not equal to rmtNum=%u", descSize, cntNotifyDescSize);
        return HCCL_E_PARA;
    }
    rmtCntNotifyDesc.clear();
    u32 pos = 0;
    for (pos = 0; pos < descSize; pos++) {
        char c;
        binaryStream >> c;
        rmtCntNotifyDesc.push_back(c);
    }
    HCCL_INFO("unpack cntNotify Desc=%s", Bytes2hex(rmtCntNotifyDesc.data(), rmtCntNotifyDesc.size()).c_str());
    return HCCL_SUCCESS;
}

HcclResult UbMemTransport::RmtBufferVecUnpackProc(
    u32 locNum, BinaryStream& binaryStream, RemoteBufferVec& bufferVec, UbRmtBufType type)
{
    u32 rmtNum;
    binaryStream >> rmtNum;
    if (UNLIKELY(type == UbRmtBufType::BUFFER && rmtNum > MAX_BUFFER_NUM)) {
        HCCL_ERROR("[UbMemTransport][RmtBufferVecUnpackProc] rmtNum[%u] exceeds limit[%u]", rmtNum, MAX_BUFFER_NUM);
        return HCCL_E_PARA;
    }

    // 允许本端和远端交换内存数量不一致
    HCCL_INFO("unpack %s %s, locNum=%u, rmtNum=%u", type.Describe().c_str(), GetLinkDescInfo().c_str(), locNum, rmtNum);

    for (u32 i = 0; i < rmtNum; i++) {
        u32 pos;
        binaryStream >> pos;
        ExchangeUbBufferDto dto;
        dto.Deserialize(binaryStream);
        if (bufferVec.size() > pos) {
            // 对于之前已经加过的资源，无需追加
            continue;
        }

        HCCL_INFO("unpack %s pos=%u, dto %s", type.Describe().c_str(), pos, dto.Describe().c_str());
        if (dto.size == 0) { // size为0，则为 remote 空buffer
            HCCL_INFO("unpack nullptr, pos=%u", pos);
            bufferVec.push_back(nullptr);
            FillRmtRmaBufferVec(nullptr, type);
        } else { // size非0，则构造一个remote buffer
            bufferVec.push_back(make_unique<RemoteUbRmaBuffer>(rdmaHandle, dto));
            FillRmtRmaBufferVec(bufferVec.back().get(), type);
            HCCL_INFO("unpack buffer pos=%u, rmtRmaBuffer=%s", pos, bufferVec.back()->Describe().c_str());
        }
    }

    return HCCL_SUCCESS;
}

HcclResult UbMemTransport::ConnVecUnpackProc(BinaryStream& binaryStream, bool& needSendFinish)
{
    u32 rmtConnNum;
    binaryStream >> rmtConnNum;
    HCCL_INFO("start unpack conn %s connNum=%u, rmtConnNum=%u", GetLinkDescInfo().c_str(), connNum, rmtConnNum);
    if (connNum != rmtConnNum) {
        HCCL_ERROR("[UbMemTransport::ConnVecUnpackProc] connNum=%u is not equal to rmtConnNum=%u", connNum, rmtConnNum);
        return HCCL_E_PARA;
    }

    needSendFinish = false; // 不需要发送 finish
    for (u32 i = 0; i < rmtConnNum; i++) {
        u32 pos;
        binaryStream >> pos;
        ExchangeUbConnDto rmtDto;
        rmtDto.Deserialize(binaryStream);
        HCCL_INFO("unpack connection pos=%u dto %s", pos, rmtDto.Describe().c_str());
        if (commonLocRes.connVec[i]->GetStatus() != RmaConnStatus::READY) {
            HCCL_INFO(
                "parse and import pos=%u, rmt dto to connection[%s]", pos, commonLocRes.connVec[i]->Describe().c_str());
            commonLocRes.connVec[i]->ParseRmtExchangeDto(rmtDto);
            commonLocRes.connVec[i]->ImportRmtDto();
            needSendFinish = true; // connection 建链，需要发送finish
        }
    }
    return HCCL_SUCCESS;
}

void UbMemTransport::FillRmtRmaBufferVec(RemoteRmaBuffer* rmaBuffer, UbRmtBufType type)
{
    if (type == UbRmtBufType::BUFFER) {
        rmtRmaBufferVec.push_back(rmaBuffer);
    }
}

HcclResult UbMemTransport::SendFinish()
{
    HCCL_INFO("start send Finish Msg %s [%s], isHost_[%d]", GetLinkDescInfo().c_str(), FINISH_MSG, isHost_);
    sendFinishMsg = std::vector<char>(FINISH_MSG, FINISH_MSG + FINISH_MSG_SIZE);
    bool ret = false;
    if (isHost_) {
        ret = socket->Send(sendFinishMsg.data(), FINISH_MSG_SIZE);
    } else {
        socket->SendAsync(sendFinishMsg.data(), FINISH_MSG_SIZE);
        ret = true;
    }
    if (!ret) {
        HCCL_ERROR("[UbMemTransport::SendFinish] Send finish msg failed");
        return HCCL_E_INTERNAL;
    }
    HCCL_INFO("end send Finish Msg %s [%s]", GetLinkDescInfo().c_str(), FINISH_MSG);
    return HCCL_SUCCESS;
}

HcclResult UbMemTransport::RecvFinish()
{
    recvFinishMsg.resize(FINISH_MSG_SIZE);
    HCCL_INFO("start recv Finish Msg %s [%s], isHost_[%d]", GetLinkDescInfo().c_str(), FINISH_MSG, isHost_);
    bool ret = false;
    if (isHost_) {
        ret = socket->Recv(recvFinishMsg.data(), FINISH_MSG_SIZE);
    } else {
        socket->RecvAsync(ReinterpretAs<u8*>(recvFinishMsg.data()), FINISH_MSG_SIZE);
        ret = true;
    }
    if (!ret) {
        HCCL_ERROR("[UbMemTransport::RecvFinish] Recv finish msg failed");
        return HCCL_E_INTERNAL;
    }
    HCCL_INFO("end recv Finish Msg %s [%s]", GetLinkDescInfo().c_str(), FINISH_MSG);
    return HCCL_SUCCESS;
}

std::vector<char> UbMemTransport::GetUniqueId()
{
    if (baseStatus != TransportStatus::READY) {
        MACRO_THROW(InternalException, StringFormat("transport status is not ready, please check"));
    }
    u32 type = static_cast<u32>(transportType);
    BinaryStream binaryStream;
    binaryStream << type;
    binaryStream << notifyNum;
    binaryStream << bufferNum;
    binaryStream << static_cast<u32>(rmtBufferVec.size());
    binaryStream << connNum;

    // [header...][notifyUniqueId...][rmtNotifyUniqueId...][rmtBufferUniqueIds...]
    auto notifyUniqueIds = GetNotifyUniqueIds();
    binaryStream << notifyUniqueIds;

    auto rmtNotifyUniqueIds = GetRmtBufferUniqueIds(rmtNotifyVec, UbRmtBufType::NOTIFY);
    binaryStream << rmtNotifyUniqueIds;

    auto rmtBufferUniqueIds = GetRmtBufferUniqueIds(rmtBufferVec, UbRmtBufType::BUFFER);
    binaryStream << rmtBufferUniqueIds;

    auto connUniqueIds = GetConnUniqueIds();
    binaryStream << connUniqueIds;

    std::vector<char> result;
    binaryStream.Dump(result);
    return result;
}

std::vector<char> UbMemTransport::GetUniqueIdV2()
{
    if (baseStatus != TransportStatus::READY) {
        MACRO_THROW(
            InternalException,
            StringFormat("transport status[%d] is not ready[%d], please check.", baseStatus, TransportStatus::READY));
    }
    u32 type = static_cast<u32>(transportType);
    BinaryStream binaryStream;
    binaryStream << type;
    binaryStream << notifyNum;
    binaryStream << bufferNum;
    binaryStream << static_cast<u32>(rmtBufferVec.size());
    binaryStream << connNum;

    auto notifyUniqueIds = GetNotifyUniqueIds();
    binaryStream << notifyUniqueIds;

    auto rmtNotifyUniqueIds = GetRmtBufferUniqueIds(rmtNotifyVec, UbRmtBufType::NOTIFY);
    binaryStream << rmtNotifyUniqueIds;

    for (auto& it : commonLocRes.bufferVec) {
        locBufferVec.emplace_back(ReinterpretAs<LocalUbRmaBuffer*>(it));
    }

    auto locBufferUniqueIds = GetLocBufferUniqueIds(locBufferVec, UbRmtBufType::BUFFER);
    binaryStream << locBufferUniqueIds;

    auto rmtBufferUniqueIds = GetRmtBufferUniqueIds(rmtBufferVec, UbRmtBufType::BUFFER);
    binaryStream << rmtBufferUniqueIds;

    auto drainUniqueIds = GetDrainUniqueIds();
    binaryStream << drainUniqueIds;

    auto connUniqueIds = GetConnUniqueIds();
    binaryStream << connUniqueIds;

    std::vector<char> result;
    binaryStream.Dump(result);
    return result;
}

std::vector<char> UbMemTransport::PackConnData()
{
    if (baseStatus != TransportStatus::READY) {
        MACRO_THROW(
            InternalException,
            StringFormat("transport status[%d] is not ready[%d], please check.", baseStatus, TransportStatus::READY));
    }
    u32 type = static_cast<u32>(transportType);
    BinaryStream binaryStream;
    binaryStream << type;
    binaryStream << connNum;

    auto connUniqueIds = GetConnUniqueIds();
    binaryStream << connUniqueIds;

    std::vector<char> result;
    binaryStream.Dump(result);
    return result;
}

std::vector<char>
UbMemTransport::GetSingleRmtBufferUniqueId(u64 addr, u64 size, u32 tokenId, u32 tokenValue, u32 notifyId) const
{
    BinaryStream binaryStream;
    binaryStream << addr;
    binaryStream << size;
    binaryStream << tokenId;
    binaryStream << tokenValue;
    binaryStream << notifyId;
    HCCL_INFO("UbMemTransport RmtBuffer[addr=0x%llx, size=0x%llx, notifyId=%u]", addr, size, notifyId);
    std::vector<char> result;
    binaryStream.Dump(result);
    return result;
}

std::vector<char> UbMemTransport::GetSingleLocBufferUniqueId(u64 addr, u64 size, u32 tokenId, u32 tokenValue) const
{
    BinaryStream binaryStream;
    binaryStream << addr;
    binaryStream << size;
    binaryStream << tokenId;
    binaryStream << tokenValue;
    HCCL_INFO("UbMemTransport LocBuffer[addr=0x%llx, size=0x%llx]", addr, size);
    std::vector<char> result;
    binaryStream.Dump(result);
    return result;
}

std::vector<char> UbMemTransport::GetNotifyUniqueIds()
{
    HCCL_INFO("start packing all notify uniqueIds");
    std::vector<char> result(0);
    for (auto& it : commonLocRes.notifyVec) {
        HCCL_INFO("ubMemTransport Notify %s", it->Describe().c_str());
        auto uniqueId = it->GetUniqueId();
        result.insert(result.end(), uniqueId.begin(), uniqueId.end());
    }
    return result;
}

std::vector<char> UbMemTransport::GetDrainUniqueIds() const
{
    HCCL_INFO("start packing drain resources uniqueIds");
    std::vector<char> result(0);
    std::vector<char> uniqueId;

    // pack drain notify
    if (drainNotify_ != nullptr) {
        auto dto = drainNotify_->GetExchangeDto();
        ExchangeUbBufferDto* rawDto = static_cast<ExchangeUbBufferDto*>(dto.get());
        uniqueId = GetSingleRmtBufferUniqueId(
            rawDto->addr, rawDto->size, rawDto->tokenId, rawDto->tokenValue, rawDto->notifyId);
        HCCL_INFO("UbMemTransport::GetDrainUniqueIds, %s", drainNotify_->Describe().c_str());
    } else {
        uniqueId = GetSingleRmtBufferUniqueId(0, 0, 0, 0, UINT32_MAX); // 填充一个空的buffer
        HCCL_INFO("UbMemTransport::GetDrainUniqueIds, drainNotify_ null buffer");
    }
    result.insert(result.end(), uniqueId.begin(), uniqueId.end());

    // pack drain rmtMem
    if (rmtDrainBuffer_ != nullptr) {
        uniqueId = GetSingleRmtBufferUniqueId(
            rmtDrainBuffer_->GetAddr(), rmtDrainBuffer_->GetSize(), rmtDrainBuffer_->GetTokenId(),
            rmtDrainBuffer_->GetTokenValue(), rmtDrainBuffer_->GetNotifyId());
        HCCL_INFO("UbMemTransport::GetDrainUniqueIds, %s", rmtDrainBuffer_->Describe().c_str());
    } else {
        uniqueId = GetSingleRmtBufferUniqueId(0, 0, 0, 0, UINT32_MAX); // 填充一个空的buffer
        HCCL_INFO("UbMemTransport::GetDrainUniqueIds, rmtDrainBuffer_ null buffer");
    }
    result.insert(result.end(), uniqueId.begin(), uniqueId.end());

    return result;
}

std::vector<char> UbMemTransport::GetRmtBufferUniqueIds(RemoteBufferVec& bufferVec, UbRmtBufType type) const
{
    HCCL_INFO("start packing all remote buffer %s uniqueIds", type.Describe().c_str());
    std::vector<char> result(0);
    for (auto& it : bufferVec) {
        std::vector<char> uniqueId;
        if (it != nullptr) {
            uniqueId = GetSingleRmtBufferUniqueId(
                it->GetAddr(), it->GetSize(), it->GetTokenId(), it->GetTokenValue(), it->GetNotifyId());
            HCCL_INFO("UbMemTransport::GetRmtBufferUniqueIds, %s", it->Describe().c_str());
        } else {
            uniqueId = GetSingleRmtBufferUniqueId(0, 0, 0, 0, UINT32_MAX); // 填充一个空的buffer
            HCCL_INFO("UbMemTransport::GetRmtBufferUniqueIds, null buffer");
        }
        result.insert(result.end(), uniqueId.begin(), uniqueId.end());
    }
    return result;
}

std::vector<char> UbMemTransport::GetLocBufferUniqueIds(LocalBufferVec& bufferVec, UbRmtBufType type) const
{
    HCCL_INFO("start packing all local buffer %s uniqueIds", type.Describe().c_str());
    std::vector<char> result(0);
    for (auto& it : bufferVec) {
        std::vector<char> uniqueId;
        if (it != nullptr) {
            uniqueId = GetSingleLocBufferUniqueId(it->GetAddr(), it->GetSize(), it->GetTokenId(), it->GetTokenValue());
            HCCL_INFO("UbMemTransport::GetLocBufferUniqueIds, %s", it->Describe().c_str());
        } else {
            uniqueId = GetSingleLocBufferUniqueId(0, 0, 0, 0); // 填充一个空的buffer
            HCCL_INFO("UbMemTransport::GetLocBufferUniqueIds, null buffer");
        }
        result.insert(result.end(), uniqueId.begin(), uniqueId.end());
    }
    return result;
}

std::vector<char> UbMemTransport::GetConnUniqueIds()
{
    HCCL_INFO("start packing all conn uniqueIds");
    std::vector<char> result(0);
    for (auto& it : commonLocRes.connVec) {
        HCCL_INFO("[UbMemTransport::%s] conn[%s]", __func__, it->Describe().c_str());
        auto uniqueId = it->GetUniqueId();
        result.insert(result.end(), uniqueId.begin(), uniqueId.end());
    }
    return result;
}

void UbMemTransport::SaveDfxTaskInfo(const TaskParam& taskParam)
{
    u32 taskId;
    u32 streamId;
    HrtGetTaskIdAndStreamID(taskId, streamId);

    callback(streamId, taskId, taskParam);
}

HcclResult UbMemTransport::GetRemoteMems(uint32_t* memNum, CommMem** remoteMem, char*** memInfos)
{
    std::lock_guard<std::mutex> lock(remoteMemsMutex_);
    Hccl::RemoteMemCtx<std::unique_ptr<RemoteUbRmaBuffer>> remoteMemCtx{
        cacheValid_, rmtBufferVec, remoteUserMems_, memInfoCopies_, memInfoPointers_, remoteMem, memInfos, memNum};
    CHK_RET(GetRemoteUserMems(remoteMemCtx));
    return HCCL_SUCCESS;
}

HcclResult UbMemTransport::CheckSocketStatus(std::string socketOpreator)
{
    CHK_PTR_NULL(socket);
    auto timeout = std::chrono::seconds(Hccl::EnvConfig::GetInstance().GetSocketConfig().GetLinkTimeOut());
    auto startTime = std::chrono::steady_clock::now();
    uint32_t retryCount = 0;
    while (true) {
        SocketStatus socketStatus = socket->GetAsyncStatus();
        if (socketStatus == SocketStatus::OK) {
            auto elapsed
                = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startTime)
                      .count();
            HCCL_INFO(
                "[UbMemTransport][%s] socket transport operation[%s] success, elapsed[%lld]ms, retryCount[%u]",
                __func__, socketOpreator.c_str(), elapsed, retryCount);
            break;
        }
        if ((std::chrono::steady_clock::now() - startTime) >= timeout || socketStatus == Hccl::SocketStatus::TIMEOUT) {
            auto elapsed
                = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startTime)
                      .count();
            HCCL_ERROR(
                "[UbMemTransport][%s] socket transport operation[%s] timeout after %lld sec, elapsed[%lld]ms, "
                "retryCount[%u]",
                __func__, socketOpreator.c_str(), timeout, elapsed, retryCount);
            return HCCL_E_TIMEOUT;
        }
        retryCount++;
    }
    return HCCL_SUCCESS;
}

HcclResult UbMemTransport::UpdateMemInfo(std::vector<LocalRmaBuffer*>& bufferVecTemp)
{
    if (bufferVecTemp.size() == 0) {
        HCCL_WARNING("[UbMemTransport][UpdateMemInfo] bufferNum is 0.");
        return HCCL_SUCCESS;
    }
    HCCL_INFO("[UbMemTransport][UpdateMemInfo] bufferNum[%zu]", bufferVecTemp.size());
    sendData.clear();
    BinaryStream sendStream;
    std::vector<std::unique_ptr<RemoteUbRmaBuffer>> rmtBufferTemp{};
    TRY_CATCH_RETURN([&]() -> void {
        BufferVecPack(sendStream, bufferVecTemp);
        sendStream.Dump(sendData);
        u32 sendSize = sendData.size();
        socket->SendAsync(&sendSize, sizeof(sendSize));
        HCCL_INFO(
            "[UbMemTransport][UpdateMemInfo] Send size[%u] of data success. [%zu] bytes sent.", sendSize,
            sizeof(sendSize));
        HcclResult result = CheckSocketStatus("SendDataSize");
        CHK_RET_THROW(
            InternalException, StringFormat("[UbMemTransport][UpdateMemInfo] failed to send dataSize."), result);
        RecvDataSize();
        result = CheckSocketStatus("RecvDataSize");
        CHK_RET_THROW(
            InternalException, StringFormat("[UbMemTransport][UpdateMemInfo] failed to receive dataSize."), result);
        SendExchangeData();
        result = CheckSocketStatus("SendExchangeData");
        CHK_RET_THROW(InternalException, StringFormat("[UbMemTransport][UpdateMemInfo] failed to send data."), result);
        RecvExchangeData();
        result = CheckSocketStatus("RecvExchangeData");
        CHK_RET_THROW(
            InternalException, StringFormat("[UbMemTransport][UpdateMemInfo] failed to receive data."), result);
        BinaryStream recvStream(recvData);
        RmtBufferVecUnpackProc(bufferNum, recvStream, rmtBufferTemp, UbRmtBufType::BUFFER);
    }());
    rmtBufferVec.insert(
        rmtBufferVec.end(), std::make_move_iterator(rmtBufferTemp.begin()),
        std::make_move_iterator(rmtBufferTemp.end()));
    commonLocRes.bufferVec.insert(commonLocRes.bufferVec.end(), bufferVecTemp.begin(), bufferVecTemp.end());
    cacheValid_ = false;
    return HCCL_SUCCESS;
}

HcclResult UbMemTransport::Init()
{
    for (auto& ubConn : commonLocRes.connVec) {
        TRY_CATCH_RETURN(ubConn->Connect());
    }

    return HCCL_SUCCESS;
}

HcclResult UbMemTransport::DeInit() const
{
    socket->Destroy();
    return HCCL_SUCCESS;
}

HcclResult UbMemTransport::GetRemoteSeg(const void* addr, u64 len, u64* seg)
{
    if (rmtBufferVec.empty()) {
        HCCL_ERROR("[UbMemTransport::%s] rmtBufferVec is empty.", __func__);
        return HCCL_E_INTERNAL;
    }

    bool isAddrInRange = false;
    for (auto& it : rmtBufferVec) {
        Buffer iterBuf(it->GetAddr(), it->GetSize());
        if (iterBuf.Contains(ReinterpretAs<uintptr_t>(addr), len)) {
            *seg = it->GetSegVa();
            isAddrInRange = true;
            break;
        }
    }

    if (!isAddrInRange) {
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

} // namespace Hccl
