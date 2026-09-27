/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "transport_base.h"
#include "adapter_rts.h"
#include "externalinput_pub.h"
#include "device_capacity.h"
#include "new/hccl_dispatcher_ctx.h"
#include "dispatcher_ctx.h"

namespace hccl {
struct SuperPodInfo {
    s32 pid = 0;
    s32 sdid = INVALID_INT;         // super Pod device id
    s32 serverPhyIdx = INVALID_INT; // 超节点server id
};

TransportBase::TransportBase(
    DispatcherPub* dispatcher, const std::unique_ptr<NotifyPool>& notifyPool, MachinePara& machinePara,
    std::chrono::milliseconds timeout)
    : exchangeDataTotalSize_(0),
      dispatcher_(dispatcher),
      notifyPool_(notifyPool),
      defaultSocket_(nullptr),
      machinePara_(machinePara),
      timeout_(timeout),
      recvPid_(0),
      recvSdid_(INVALID_INT),
      nicDeploy_(NICDeployment::NIC_DEPLOYMENT_RESERVED),
      useOneDoorbell_(false),
      notifyNum_(machinePara.notifyNum)
{
    if (machinePara_.sockets.size() > 0) {
        defaultSocket_ = machinePara_.sockets[0];
    }
}

TransportBase::~TransportBase() {}

HcclResult TransportBase::Init()
{
    CHK_SMART_PTR_NULL(dispatcher_);
    CHK_RET(CheckExchangeData());

    return HCCL_SUCCESS;
}

HcclResult TransportBase::CheckDeviceId()
{
    u32 maxDeviceNum;
    CHK_RET(GetMaxDevNum(maxDeviceNum));
    bool invalidDevId
        = machinePara_.deviceLogicId < 0 || (static_cast<u32>(machinePara_.deviceLogicId) >= maxDeviceNum);
    CHK_PRT_RET(
        invalidDevId,
        HCCL_ERROR("[TransportBase][CheckDeviceId] deviceLogicId[%d] is invalid", machinePara_.deviceLogicId),
        HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

HcclResult TransportBase::DeInit() { return HCCL_SUCCESS; }

HcclResult TransportBase::Stop() { return HCCL_SUCCESS; }

HcclResult TransportBase::Resume() { return HCCL_SUCCESS; }

TransportAttr TransportBase::GetTransportAttr() { return transportAttr_; }

HcclResult TransportBase::TxDataSignal(Stream& stream)
{
    static_cast<void>(stream);
    return HCCL_SUCCESS;
}

HcclResult TransportBase::RxDataSignal(Stream& stream)
{
    static_cast<void>(stream);
    return HCCL_SUCCESS;
}

HcclResult TransportBase::TxData(UserMemType dstMemType, u64 dstOffset, const void* src, u64 len, Stream& stream)
{
    static_cast<void>(dstMemType);
    static_cast<void>(dstOffset);
    static_cast<void>(src);
    static_cast<void>(len);
    static_cast<void>(stream);
    return HCCL_SUCCESS;
}

HcclResult TransportBase::RxData(UserMemType srcMemType, u64 srcOffset, void* dst, u64 len, Stream& stream)
{
    static_cast<void>(srcMemType);
    static_cast<void>(srcOffset);
    static_cast<void>(dst);
    static_cast<void>(len);
    static_cast<void>(stream);
    return HCCL_SUCCESS;
}

HcclResult TransportBase::TxAsync(UserMemType dstMemType, u64 dstOffset, const void* src, u64 len, Stream& stream)
{
    static_cast<void>(dstMemType);
    static_cast<void>(dstOffset);
    static_cast<void>(src);
    static_cast<void>(len);
    static_cast<void>(stream);
    return HCCL_SUCCESS;
}

HcclResult TransportBase::TxAsync(std::vector<TxMemoryInfo>& txMems, Stream& stream)
{
    static_cast<void>(txMems);
    static_cast<void>(stream);
    return HCCL_SUCCESS;
}

HcclResult TransportBase::RxAsync(UserMemType srcMemType, u64 srcOffset, void* dst, u64 len, Stream& stream)
{
    static_cast<void>(srcMemType);
    static_cast<void>(srcOffset);
    static_cast<void>(dst);
    static_cast<void>(len);
    static_cast<void>(stream);
    return HCCL_SUCCESS;
}

HcclResult TransportBase::RxAsync(std::vector<RxMemoryInfo>& rxMems, Stream& stream)
{
    static_cast<void>(rxMems);
    static_cast<void>(stream);
    return HCCL_SUCCESS;
}

HcclResult TransportBase::DataReceivedAck(Stream& stream)
{
    static_cast<void>(stream);
    return HCCL_SUCCESS;
}

HcclResult TransportBase::TxAck(Stream& stream)
{
    static_cast<void>(stream);
    return HCCL_SUCCESS;
}

HcclResult TransportBase::RxAck(Stream& stream)
{
    static_cast<void>(stream);
    return HCCL_SUCCESS;
}

HcclResult TransportBase::TxPrepare(Stream& stream)
{
    static_cast<void>(stream);
    return HCCL_SUCCESS;
}

HcclResult TransportBase::RxPrepare(Stream& stream)
{
    static_cast<void>(stream);
    return HCCL_SUCCESS;
}

HcclResult TransportBase::TxDone(Stream& stream)
{
    static_cast<void>(stream);
    return HCCL_SUCCESS;
}

HcclResult TransportBase::RxDone(Stream& stream)
{
    static_cast<void>(stream);
    return HCCL_SUCCESS;
}

HcclResult TransportBase::TxWaitDone(Stream& stream)
{
    static_cast<void>(stream);
    return HCCL_SUCCESS;
}

HcclResult TransportBase::RxWaitDone(Stream& stream)
{
    static_cast<void>(stream);
    return HCCL_SUCCESS;
}

HcclResult TransportBase::Post(u32 notifyIdx, Stream& stream)
{
    static_cast<void>(notifyIdx);
    static_cast<void>(stream);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TransportBase::Wait(u32 notifyIdx, Stream& stream, const u32 timeOut)
{
    static_cast<void>(notifyIdx);
    static_cast<void>(stream);
    static_cast<void>(timeOut);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult
TransportBase::TxEnv([[maybe_unused]] const void* ptr, [[maybe_unused]] const u64 len, [[maybe_unused]] Stream& stream)
{
    return HCCL_SUCCESS;
}

HcclResult TransportBase::RxEnv([[maybe_unused]] Stream& stream) { return HCCL_SUCCESS; }

HcclResult TransportBase::TxWithReduce(
    UserMemType dstMemType, u64 dstOffset, const void* src, u64 len, const HcclDataType datatype, HcclReduceOp redOp,
    Stream& stream)
{
    static_cast<void>(dstMemType);
    static_cast<void>(dstOffset);
    static_cast<void>(src);
    static_cast<void>(len);
    static_cast<void>(datatype);
    static_cast<void>(redOp);
    static_cast<void>(stream);
    return HCCL_SUCCESS;
}

HcclResult TransportBase::TxWithReduce(
    const std::vector<TxMemoryInfo>& txWithReduceMems, const HcclDataType datatype, HcclReduceOp redOp, Stream& stream)
{
    static_cast<void>(txWithReduceMems);
    static_cast<void>(datatype);
    static_cast<void>(redOp);
    static_cast<void>(stream);
    return HCCL_SUCCESS;
}

HcclResult TransportBase::RxWithReduce(
    UserMemType recvSrcMemType, u64 recvSrcOffset, void* recvDst, u64 recvLen, void* reduceSrc, void* reduceDst,
    u64 reduceDataCount, HcclDataType reduceDatatype, HcclReduceOp reduceOp, Stream& stream, const u64 reduceAttr)
{
    static_cast<void>(recvSrcMemType);
    static_cast<void>(recvSrcOffset);
    static_cast<void>(recvDst);
    static_cast<void>(recvLen);
    static_cast<void>(reduceSrc);
    static_cast<void>(reduceDst);
    static_cast<void>(reduceDataCount);
    static_cast<void>(reduceDatatype);
    static_cast<void>(reduceOp);
    static_cast<void>(stream);
    static_cast<void>(reduceAttr);
    return HCCL_SUCCESS;
}

HcclResult TransportBase::RxWithReduce(
    const std::vector<RxWithReduceMemoryInfo>& rxWithReduceMems, HcclDataType reduceDatatype, HcclReduceOp reduceOp,
    Stream& stream, const u64 reduceAttr)
{
    static_cast<void>(rxWithReduceMems);
    static_cast<void>(reduceDatatype);
    static_cast<void>(reduceOp);
    static_cast<void>(stream);
    static_cast<void>(reduceAttr);
    return HCCL_SUCCESS;
}

bool TransportBase::IsSupportTransportWithReduce() { return false; }

HcclResult
TransportBase::GetIndOpRemoteMemDetails(MemDetails** remoteMem, uint32_t* memNum, [[maybe_unused]] HcclMemType memType)
{
    static_cast<void>(remoteMem);
    static_cast<void>(memNum);
    return HCCL_E_PARA;
}

HcclResult TransportBase::GetIndOpRemoteMem(HcclMem** remoteMem, uint32_t* memNum)
{
    static_cast<void>(remoteMem);
    static_cast<void>(memNum);
    return HCCL_E_PARA;
}

HcclResult TransportBase::GetRemoteMem(UserMemType memType, void** remotePtr)
{
    static_cast<void>(memType);
    static_cast<void>(remotePtr);
    return HCCL_E_PARA;
}

HcclResult TransportBase::GetRemoteMem(std::vector<void*>* remotePtrVec)
{
    static_cast<void>(remotePtrVec);
    return HCCL_SUCCESS;
}

HcclResult TransportBase::GetRemoteMemKey(UserMemType memType, uint32_t* remoteMemKey)
{
    static_cast<void>(memType);
    static_cast<void>(remoteMemKey);
    return HCCL_E_PARA;
}

HcclResult TransportBase::GetRemoteMemSize(UserMemType memType, u64& size)
{
    static_cast<void>(memType);
    static_cast<void>(size);
    return HCCL_E_PARA;
}

HcclResult TransportBase::GetLocalRdmaNotify(std::vector<HcclSignalInfo>& rdmaNotify)
{
    static_cast<void>(rdmaNotify);
    return HCCL_E_PARA;
}

HcclResult TransportBase::GetDrainLocalDataNotify(void*& localAddr, uint32_t& lkey, HcclSignalInfo& dataNotify)
{
    static_cast<void>(localAddr);
    static_cast<void>(lkey);
    static_cast<void>(dataNotify);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TransportBase::GetRemoteRdmaNotifyAddrKey(std::vector<AddrKey>& rdmaNotifyAddr)
{
    static_cast<void>(rdmaNotifyAddr);
    return HCCL_E_PARA;
}

HcclResult TransportBase::GetLocalNotifyValueAddrKey(std::vector<AddrKey>& notifyValue)
{
    static_cast<void>(notifyValue);
    return HCCL_E_PARA;
}

HcclResult TransportBase::GetLocalMemDetails(UserMemType memType, MemDetails& memDetails)
{
    static_cast<void>(memType);
    static_cast<void>(memDetails);
    return HCCL_E_PARA;
}

HcclResult TransportBase::GetLocalNotify(std::vector<HcclSignalInfo>& localNotify)
{
    static_cast<void>(localNotify);
    return HCCL_E_PARA;
}

HcclResult TransportBase::GetRemoteNotify(std::vector<HcclSignalInfo>& localNotify)
{
    static_cast<void>(localNotify);
    return HCCL_E_PARA;
}

HcclResult TransportBase::GetAiQpInfo(std::vector<HcclQpInfoV2>& aiQpInfo)
{
    static_cast<void>(aiQpInfo);
    return HCCL_E_PARA;
}
HcclResult TransportBase::GetTransportId(u32& id)
{
    static_cast<void>(id);
    return HCCL_E_PARA;
}

HcclResult TransportBase::GetAiRMAQueueInfo(std::vector<HcclAiRMAQueueInfo>& aiRMAQueueInfo)
{
    static_cast<void>(aiRMAQueueInfo);
    return HCCL_E_PARA;
}

HcclResult TransportBase::FillExchangeDataTotalSize()
{
    exchangeDataTotalSize_ = 0;
    return HCCL_E_PARA; // this function should not be called in normal process
}

HcclResult TransportBase::ConstructExchangeForSend()
{
    return HCCL_E_PARA; // this function should not be called in normal process
}

HcclResult TransportBase::ParseReceivedExchangeData()
{
    return HCCL_E_PARA; // this function should not be called in normal process
}

HcclResult TransportBase::GetChipId(s64& chipId)
{
    CHK_RET(hrtGetDeviceInfo(
        machinePara_.deviceLogicId, HcclRtDeviceModuleType::HCCL_RT_MODULE_TYPE_SYSTEM,
        HcclRtDeviceInfoType::HCCL_INFO_TYPE_PHY_CHIP_ID, chipId));
    HCCL_DEBUG("[GetChipId]chipId: %ld", chipId);
    return HCCL_SUCCESS;
}

HcclResult TransportBase::ExchangeTgidMesg()
{
    SuperPodInfo sendInfo;
    CHK_RET(SalGetBareTgid(&sendInfo.pid)); // 当前进程id
    if (machinePara_.deviceType == DevType::DEV_TYPE_910_93) {
        s64 sdid = 0;
        CHK_RET(hrtGetDeviceInfo(
            machinePara_.deviceLogicId, HcclRtDeviceModuleType::HCCL_RT_MODULE_TYPE_SYSTEM,
            HcclRtDeviceInfoType::HCCL_INFO_TYPE_SDID, sdid));
        sendInfo.sdid = static_cast<s32>(sdid);

        s64 serverPhyIdx = 0;
        CHK_RET(hrtGetDeviceInfo(
            machinePara_.deviceLogicId, HcclRtDeviceModuleType::HCCL_RT_MODULE_TYPE_SYSTEM,
            HcclRtDeviceInfoType::HCCL_INFO_TYPE_SERVER_ID, serverPhyIdx));
        sendInfo.serverPhyIdx = static_cast<s32>(serverPhyIdx);
    }

    HcclResult ret = HCCL_SUCCESS;
    CHK_SMART_PTR_NULL(defaultSocket_);
    ret = defaultSocket_->Send(reinterpret_cast<u8*>(&sendInfo), sizeof(SuperPodInfo));
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[Exchange][TgidMesg]errNo[0x%016llx] In exchange tgid mesg, send pid failed. "
            "remote userrank[%u] pid[%d] sdid[%016llx] local rank[%u]",
            HCCL_ERROR_CODE(ret), machinePara_.remoteUserrank, sendInfo.pid, sendInfo.sdid, machinePara_.localUserrank),
        ret);

    SuperPodInfo recvInfo = {};
    ret = defaultSocket_->Recv(reinterpret_cast<u8*>(&recvInfo), sizeof(SuperPodInfo));
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[Exchange][TgidMesg]errNo[0x%016llx] In exchange tgid mesg, recv pid failed. "
            "remote userrank[%u] pid[%d] sdid[%016llx] local rank[%u]",
            HCCL_ERROR_CODE(ret), machinePara_.remoteUserrank, recvInfo.pid, recvInfo.sdid, machinePara_.localUserrank),
        ret);

    recvPid_ = recvInfo.pid;
    // sdid同时满足以下条件时使用: 1.跨server场景 2.使能HCCS 3.超节点内(默认满足, 链路选择时保证)
    recvSdid_ = (sendInfo.serverPhyIdx != recvInfo.serverPhyIdx && !GetExternalInputInterHccsDisable()) ?
                    recvInfo.sdid :
                    INVALID_INT;
    HCCL_INFO(
        "[Exchange][TgidMesg]local: rank[%u], pid[%d], sdid[%016llx], serverPhyIdx[%016llx], "
        "remote: rank[%u], pid[%d], sdid[%016llx], serverPhyIdx[%016llx], recvSdid[%016llx]",
        machinePara_.localUserrank, sendInfo.pid, sendInfo.sdid, sendInfo.serverPhyIdx, machinePara_.remoteUserrank,
        recvInfo.pid, recvInfo.sdid, recvInfo.serverPhyIdx, recvSdid_);

    return HCCL_SUCCESS;
}

HcclResult TransportBase::SendNotifyReadyMesg()
{
    HCCL_DEBUG(
        "[Send][NotifyReadyMesg]recvSDID[%016llx], remoteRank[%016llx], recvPid[%016llx]", recvSdid_,
        machinePara_.remoteUserrank, recvPid_);
    RemoteRankInfo info(machinePara_.remoteDeviceId, machinePara_.remoteWorldRank, recvPid_, recvSdid_);
    CHK_SMART_PTR_NULL(notifyPool_);
    CHK_RET(notifyPool_->Alloc(machinePara_.tag, info, localSendReadyNotify_));

    std::vector<u8> data(NOTIFY_INFO_LENGTH, 0);
    CHK_RET(localSendReadyNotify_->Serialize(data));
    CHK_SMART_PTR_NULL(defaultSocket_);
    HcclResult ret = defaultSocket_->Send(&data[0], data.size());
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[Send][IpcNotifyReadyMesg]errNo[0x%016llx]In send notify ready mesg, send read msg failed. remote "
            "userrank[%u] notify local rank[%u]",
            HCCL_ERROR_CODE(ret), machinePara_.remoteUserrank, machinePara_.localUserrank),
        ret);

    HCCL_DEBUG(
        "local_send_ready_notify send rank[%u] to rank[%u]", machinePara_.localUserrank, machinePara_.remoteUserrank);
    return HCCL_SUCCESS;
}

HcclResult TransportBase::SendNotifyDoneMesg()
{
    HCCL_DEBUG(
        "[Send][NotifyDoneMesg]recvSDID[%016llx], remoteRank[%016llx], recvPid[%016llx]", recvSdid_,
        machinePara_.remoteUserrank, recvPid_);
    RemoteRankInfo info(machinePara_.remoteDeviceId, machinePara_.remoteWorldRank, recvPid_, recvSdid_);
    CHK_RET(notifyPool_->Alloc(machinePara_.tag, info, localSendDoneNotify_));

    std::vector<u8> data(NOTIFY_INFO_LENGTH, 0);
    CHK_RET(localSendDoneNotify_->Serialize(data));
    CHK_SMART_PTR_NULL(defaultSocket_);
    HcclResult ret = defaultSocket_->Send(&data[0], data.size());
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[Send][IpcNotifyDoneMesg]errNo[0x%016llx] In send notify done mesg, send done msg "
            "failed. remote userrank[%u] local rank[%u]",
            HCCL_ERROR_CODE(ret), machinePara_.remoteUserrank, machinePara_.localUserrank),
        ret);

    HCCL_DEBUG("send_done_notify send rank[%u] to rank[%u]", machinePara_.localUserrank, machinePara_.remoteUserrank);
    return HCCL_SUCCESS;
}

HcclResult TransportBase::SendDeviceIpcNotifyReadyMesg()
{
    HCCL_DEBUG(
        "[Send][DeviceIpcNotifyReadyMesg]recvSDID[%016llx], remoteRank[%016llx], recvPid[%016llx]", recvSdid_,
        machinePara_.remoteUserrank, recvPid_);
    RemoteRankInfo info(machinePara_.remoteDeviceId, machinePara_.remoteWorldRank, recvPid_, recvSdid_);
    CHK_RET(notifyPool_->Alloc(machinePara_.tag, info, localSendReadyDeviceNotify_, NotifyLoadType::DEVICE_NOTIFY));

    std::vector<u8> data(NOTIFY_INFO_LENGTH, 0);
    CHK_RET(localSendReadyDeviceNotify_->Serialize(data));
    CHK_SMART_PTR_NULL(defaultSocket_);
    HcclResult ret = defaultSocket_->Send(&data[0], data.size());
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[Send][IpcNotifyReadyMesg]errNo[0x%016llx]In send notify ready mesg, send read msg failed. remote "
            "userrank[%u] notify local rank[%u]",
            HCCL_ERROR_CODE(ret), machinePara_.remoteUserrank, machinePara_.localUserrank),
        ret);

    HCCL_DEBUG(
        "send_device_ready_notify send rank[%u] to rank[%u]", machinePara_.localUserrank, machinePara_.remoteUserrank);
    return HCCL_SUCCESS;
}

HcclResult TransportBase::SendDeviceIpcNotifyDoneMesg()
{
    HCCL_DEBUG(
        "[Send][DeviceIpcNotifyDoneMesg]recvSDID[%016llx], remoteRank[%016llx], recvPid[%016llx]", recvSdid_,
        machinePara_.remoteUserrank, recvPid_);
    RemoteRankInfo info(machinePara_.remoteDeviceId, machinePara_.remoteWorldRank, recvPid_, recvSdid_);
    CHK_RET(notifyPool_->Alloc(machinePara_.tag, info, localSendDoneDeviceNotify_, NotifyLoadType::DEVICE_NOTIFY));

    std::vector<u8> data(NOTIFY_INFO_LENGTH, 0);
    CHK_RET(localSendDoneDeviceNotify_->Serialize(data));
    CHK_SMART_PTR_NULL(defaultSocket_);
    HcclResult ret = defaultSocket_->Send(&data[0], data.size());
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[Send][IpcNotifyReadyMesg]errNo[0x%016llx]In send notify ready mesg, send read msg failed. remote "
            "userrank[%u] notify local rank[%u]",
            HCCL_ERROR_CODE(ret), machinePara_.remoteUserrank, machinePara_.localUserrank),
        ret);

    HCCL_DEBUG(
        "send_device_done_notify send rank[%u] to rank[%u]", machinePara_.localUserrank, machinePara_.remoteUserrank);
    return HCCL_SUCCESS;
}

HcclResult TransportBase::RecvNotifyReadyMesg()
{
    // 获取ready notify data
    std::vector<u8> data(NOTIFY_INFO_LENGTH, 0);
    CHK_SMART_PTR_NULL(defaultSocket_);
    HcclResult ret = defaultSocket_->Recv(&data[0], NOTIFY_INFO_LENGTH);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[Recv][NotifyReadyMesg]errNo[0x%016llx]receive remote send ready notify data failed. remote "
            "user rank[%u], receive local rank[%u]",
            HCCL_ERROR_CODE(ret), machinePara_.remoteUserrank, machinePara_.localUserrank),
        ret);

    CHK_RET(OpenRemoteNotify(data, remoteSendReadyNotify_));
    return HCCL_SUCCESS;
}

HcclResult TransportBase::RecvNotifyDoneMesg()
{
    // 获取done notify data
    std::vector<u8> data(NOTIFY_INFO_LENGTH, 0);
    CHK_SMART_PTR_NULL(defaultSocket_);
    HcclResult ret = defaultSocket_->Recv(&data[0], NOTIFY_INFO_LENGTH);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[Recv][RecvNotifyDoneMesg]errNo[0x%016llx]receive remote send ready notify data failed. remote "
            "user rank[%u], receive local rank[%u]",
            HCCL_ERROR_CODE(ret), machinePara_.remoteUserrank, machinePara_.localUserrank),
        ret);
    HCCL_DEBUG(
        "send_done_notify rank[%u] receive from rank[%u]", machinePara_.localUserrank, machinePara_.remoteUserrank);

    CHK_RET(OpenRemoteNotify(data, remoteSendDoneNotify_));

    HCCL_DEBUG(
        "remote_send_done_notify send rank[%u] to rank[%u]", machinePara_.localUserrank, machinePara_.remoteUserrank);

    return HCCL_SUCCESS;
}

HcclResult TransportBase::RecvDeviceIpcNotifyReadyMesg()
{
    // 获取ready notify data
    std::vector<u8> data(NOTIFY_INFO_LENGTH, 0);
    CHK_SMART_PTR_NULL(defaultSocket_);
    HcclResult ret = defaultSocket_->Recv(&data[0], NOTIFY_INFO_LENGTH);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[Recv][DeviceIpcNotifyReadyMesg]errNo[0x%016llx]receive remote send ready notify data failed. "
            "remote user rank[%u], receive local rank[%u]",
            HCCL_ERROR_CODE(ret), machinePara_.remoteUserrank, machinePara_.localUserrank),
        ret);
    HCCL_DEBUG(
        "send_ready_device_notify rank[%u] receive from rank[%u]", machinePara_.localUserrank,
        machinePara_.remoteUserrank);

    CHK_RET(OpenRemoteNotify(data, remoteSendReadyDeviceNotify_));

    HCCL_DEBUG(
        "remote_send_ready_device_notify send rank[%u] to rank[%u]", machinePara_.localUserrank,
        machinePara_.remoteUserrank);
    return HCCL_SUCCESS;
}

HcclResult TransportBase::RecvDeviceIpcNotifyDoneMesg()
{
    // 获取ready notify data
    std::vector<u8> data(NOTIFY_INFO_LENGTH, 0);
    CHK_SMART_PTR_NULL(defaultSocket_);
    HcclResult ret = defaultSocket_->Recv(&data[0], NOTIFY_INFO_LENGTH);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[Recv][DeviceIpcNotifyDoneMesg]errNo[0x%016llx]receive remote send ready notify data failed. remote"
            " user rank[%u], receive local rank[%u]",
            HCCL_ERROR_CODE(ret), machinePara_.remoteUserrank, machinePara_.localUserrank),
        ret);
    HCCL_DEBUG(
        "send_done_device_notify rank[%u] receive from rank[%u]", machinePara_.localUserrank,
        machinePara_.remoteUserrank);

    CHK_RET(OpenRemoteNotify(data, remoteSendDoneDeviceNotify_));

    HCCL_DEBUG(
        "remote_send_done_device_notify send rank[%u] to rank[%u]", machinePara_.localUserrank,
        machinePara_.remoteUserrank);
    return HCCL_SUCCESS;
}

HcclResult TransportBase::CheckLinkStatus()
{
    HcclResult ret;
    /* link状态 */
    std::string localLinkStatus = "true";
    CHK_SMART_PTR_NULL(defaultSocket_);
    ret = defaultSocket_->Send(localLinkStatus);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[Check][LinkStatus]errNo[0x%016llx]In check link status, send link status failed. "
            "remote userrank[%u] local rank[%u]",
            HCCL_ERROR_CODE(ret), machinePara_.remoteUserrank, machinePara_.localUserrank),
        ret);

    HCCL_DEBUG(
        "local_link_status send rank[%u] to rank[%u] message[%s]", machinePara_.localUserrank,
        machinePara_.remoteUserrank, localLinkStatus.c_str());

    // 获取remote_link_status
    std::string remoteLinkStatus;
    CHK_SMART_PTR_NULL(defaultSocket_);
    ret = defaultSocket_->Recv(remoteLinkStatus);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[Check][LinkStatus]errNo[0x%016llx]In check link status, receive remote link status failed. "
            "remote user rank[%u] local rank[%u]",
            HCCL_ERROR_CODE(ret), machinePara_.remoteUserrank, machinePara_.localUserrank),
        ret);

    HCCL_DEBUG(
        "remote_link_status rank[%u] receive from rank[%u] message[%s]", machinePara_.localUserrank,
        machinePara_.remoteUserrank, remoteLinkStatus.c_str());
    return HCCL_SUCCESS;
}

HcclResult TransportBase::CheckLinkMode()
{
    bool bErr = (machinePara_.linkMode != LinkMode::LINK_SIMPLEX_MODE)
                && (machinePara_.linkMode != LinkMode::LINK_DUPLEX_MODE);
    CHK_PRT_RET(
        bErr,
        HCCL_ERROR(
            "[Check][LinkMode]errNo[0x%016llx] check LinkMode[%d] fail", HCCL_ERROR_CODE(HCCL_E_PARA),
            machinePara_.linkMode),
        HCCL_E_PARA);
    return HCCL_SUCCESS;
}

HcclResult TransportBase::LinkSendNotifyMesg()
{
    s32 sendPid = 0;
    CHK_RET(SalGetBareTgid(&sendPid)); // 当前进程id
    HCCL_INFO("LinkSendNotifyMesg, sendPid[%d], recvPid[%d]", sendPid, recvPid_);
    if (machinePara_.linkMode != LinkMode::LINK_SIMPLEX_MODE
        || machinePara_.machineType == MachineType::MACHINE_CLIENT_TYPE) {
        /* 发送IPC notify Ready 信息 */
        CHK_RET(SendNotifyReadyMesg());
    }

    if (machinePara_.linkMode != LinkMode::LINK_SIMPLEX_MODE
        || machinePara_.machineType == MachineType::MACHINE_SERVER_TYPE) {
        /* 发送IPC notify Done 信息 */
        CHK_RET(SendNotifyDoneMesg());
    }

    if ((machinePara_.linkMode != LinkMode::LINK_SIMPLEX_MODE
         || machinePara_.machineType == MachineType::MACHINE_CLIENT_TYPE)
        && machinePara_.isAicpuModeEn == true) {
        /* 发送Device上使用的IPC notify ready信息 */
        CHK_RET(SendDeviceIpcNotifyReadyMesg());
    }

    if ((machinePara_.linkMode != LinkMode::LINK_SIMPLEX_MODE
         || machinePara_.machineType == MachineType::MACHINE_SERVER_TYPE)
        && machinePara_.isAicpuModeEn == true) {
        /* 发送Device上使用的IPC notify ready信息 */
        CHK_RET(SendDeviceIpcNotifyDoneMesg());
    }
    return HCCL_SUCCESS;
}

HcclResult TransportBase::LinkRecvNotifyMesg()
{
    s32 sendPid = 0;
    CHK_RET(SalGetBareTgid(&sendPid)); // 当前进程id
    HCCL_INFO("LinkRecvNotifyMesg, sendPid[%d], recvPid[%d]", sendPid, recvPid_);

    if (machinePara_.linkMode != LinkMode::LINK_SIMPLEX_MODE
        || machinePara_.machineType == MachineType::MACHINE_SERVER_TYPE) {
        /* 接收IPC ready 信息 */
        CHK_RET(RecvNotifyReadyMesg());
    }
    if (machinePara_.linkMode != LinkMode::LINK_SIMPLEX_MODE
        || machinePara_.machineType == MachineType::MACHINE_CLIENT_TYPE) {
        /* 接收IPC ready 信息 */
        CHK_RET(RecvNotifyDoneMesg());
    }

    if ((machinePara_.linkMode != LinkMode::LINK_SIMPLEX_MODE
         || machinePara_.machineType == MachineType::MACHINE_SERVER_TYPE)
        && machinePara_.isAicpuModeEn == true) {
        /* 接收IPC ready 信息 */
        CHK_RET(RecvDeviceIpcNotifyReadyMesg());
    }
    if ((machinePara_.linkMode != LinkMode::LINK_SIMPLEX_MODE
         || machinePara_.machineType == MachineType::MACHINE_CLIENT_TYPE)
        && machinePara_.isAicpuModeEn == true) {
        /* 接收IPC ready 信息 */
        CHK_RET(RecvDeviceIpcNotifyDoneMesg());
    }
    return HCCL_SUCCESS;
}

HcclResult TransportBase::SetNotify()
{
    HcclSignalInfo signalInfo;
    CHK_PTR_NULL(remoteSendReadyNotify_);
    CHK_PTR_NULL(remoteSendDoneNotify_);
    CHK_PTR_NULL(localSendReadyNotify_);
    CHK_PTR_NULL(localSendDoneNotify_);

    CHK_RET(remoteSendReadyNotify_->GetNotifyData(signalInfo));
    remoteSendReadyAddress_ = signalInfo.addr;

    CHK_RET(remoteSendDoneNotify_->GetNotifyData(signalInfo));
    remoteSendDoneAddress_ = signalInfo.addr;

    remoteSendReadyNotify_->GetNotifyOffset(remoteSendReadyOffset_);
    remoteSendDoneNotify_->GetNotifyOffset(remoteSendDoneOffset_);

    bool bRet = !(notifyNum_ == userLocalNotify_.size() && notifyNum_ == userRemoteNotify_.size());
    CHK_PRT_RET(
        bRet,
        HCCL_ERROR(
            "[TransportBase][SetNotify]NotifyNumber of userLocalNotify_/userRemoteNotify_ doesn't equal to "
            "notifyNum_[%u]",
            notifyNum_),
        HCCL_E_INTERNAL);

    for (u32 i = 0; i < notifyNum_; i++) {
        CHK_PTR_NULL(userLocalNotify_[i]);
        CHK_PTR_NULL(userRemoteNotify_[i]);
        CHK_RET(userRemoteNotify_[i]->GetNotifyData(signalInfo));
        userRemoteNotifyAddr_[i] = signalInfo.addr;
        userRemoteNotify_[i]->GetNotifyOffset(userRemoteNotifyOffset_[i]);
    }

    return HCCL_SUCCESS;
}

HcclResult
TransportBase::SignalInit(const std::shared_ptr<LocalNotify>& notify, std::shared_ptr<LocalIpcNotify>& ipcNotify)
{
    CHK_SMART_PTR_NULL(notify);
    HcclSignalInfo signalInfo;
    CHK_RET(notify->GetNotifyData(signalInfo));
    EXCEPTION_CATCH((ipcNotify = std::make_shared<LocalIpcNotify>()), return HCCL_E_PTR);
    CHK_RET(ipcNotify->Init(signalInfo, NotifyLoadType::DEVICE_NOTIFY));
    HCCL_INFO("%s notifyId_ [%u]", __func__, ipcNotify->notifyId_);
    return HCCL_SUCCESS;
}

HcclResult TransportBase::SetNotifyPtr(const TransportDeviceP2pData& transDevP2pData)
{
    CHK_RET(SignalInit(transDevP2pData.ipcPreWaitNotify, localSendReadyNotify_));
    CHK_RET(SignalInit(transDevP2pData.ipcPostWaitNotify, localSendDoneNotify_));
    remoteSendReadyNotify_ = transDevP2pData.ipcPreRecordNotify;
    remoteSendDoneNotify_ = transDevP2pData.ipcPostRecordNotify;

    // 校验notifyNum_数量
    bool bRet = !(
        notifyNum_ == transDevP2pData.userLocalNotify.size() && notifyNum_ == transDevP2pData.userRemoteNotify.size()
        && notifyNum_ == userLocalNotify_.size() && notifyNum_ == userRemoteNotify_.size());
    CHK_PRT_RET(
        bRet,
        HCCL_ERROR(
            "[TransportBase][SetNotifyPtr]NotifyNum of userLocalNotify/userRemoteNotify doesn't equal to "
            "notifyNum_[%u]",
            notifyNum_),
        HCCL_E_INTERNAL);

    for (u32 i = 0; i < notifyNum_; i++) {
        CHK_RET(SignalInit(transDevP2pData.userLocalNotify[i], userLocalNotify_[i]));
        userRemoteNotify_[i] = transDevP2pData.userRemoteNotify[i];
    }

    return HCCL_SUCCESS;
}

void TransportBase::DestroyHostSignal()
{
    s32 sendPid = 0;
    SalGetBareTgid(&sendPid); // 当前进程id
    HCCL_INFO("SignalDestroy, sendPid[%d], recvPid[%d]", sendPid, recvPid_);

    if (machinePara_.linkMode != LinkMode::LINK_SIMPLEX_MODE
        || machinePara_.machineType == MachineType::MACHINE_SERVER_TYPE) {
        if ((remoteSendReadyNotify_ != nullptr)) {
            remoteSendReadyNotify_->Close();
            remoteSendReadyNotify_ = nullptr;
        }
        /* 销毁create的signal资源 */
        localSendDoneNotify_ = nullptr;
    }
    if (machinePara_.linkMode != LinkMode::LINK_SIMPLEX_MODE
        || machinePara_.machineType == MachineType::MACHINE_CLIENT_TYPE) {
        /* 关闭open的signal资源, destroy支持close */
        if ((remoteSendDoneNotify_ != nullptr)) {
            remoteSendDoneNotify_->Close();
            remoteSendDoneNotify_ = nullptr;
        }
        localSendReadyNotify_ = nullptr;
    }
}

void TransportBase::DestroyDeviceSignal()
{
    if ((machinePara_.linkMode != LinkMode::LINK_SIMPLEX_MODE
         || machinePara_.machineType == MachineType::MACHINE_SERVER_TYPE)
        && machinePara_.isAicpuModeEn == true) {
        if ((remoteSendReadyDeviceNotify_ != nullptr)) {
            remoteSendReadyDeviceNotify_->Close();
            remoteSendReadyDeviceNotify_ = nullptr;
        }
        /* 销毁create的signal资源 */
        localSendDoneDeviceNotify_ = nullptr;
    }
    if ((machinePara_.linkMode != LinkMode::LINK_SIMPLEX_MODE
         || machinePara_.machineType == MachineType::MACHINE_CLIENT_TYPE)
        && machinePara_.isAicpuModeEn == true) {
        /* 关闭open的signal资源, destroy支持close */
        if ((remoteSendDoneDeviceNotify_ != nullptr)) {
            remoteSendDoneDeviceNotify_->Close();
            remoteSendDoneDeviceNotify_ = nullptr;
        }
        localSendReadyDeviceNotify_ = nullptr;
    }
}

void TransportBase::SignalDestroy()
{
    DestroyHostSignal();
    DestroyDeviceSignal();
}

HcclResult TransportBase::GetTxAckDevNotifyInfo(HcclSignalInfo& notifyInfo)
{
    CHK_SMART_PTR_NULL(remoteSendDoneDeviceNotify_);
    CHK_RET(remoteSendDoneDeviceNotify_->GetNotifyData(notifyInfo));

    return HCCL_SUCCESS;
}

HcclResult TransportBase::GetRxAckDevNotifyInfo(HcclSignalInfo& notifyInfo)
{
    CHK_SMART_PTR_NULL(localSendDoneDeviceNotify_);
    CHK_RET(localSendDoneDeviceNotify_->GetNotifyData(notifyInfo));

    return HCCL_SUCCESS;
}

HcclResult TransportBase::GetTxDataSigleDevNotifyInfo(HcclSignalInfo& notifyInfo)
{
    CHK_SMART_PTR_NULL(remoteSendReadyDeviceNotify_);
    CHK_RET(remoteSendReadyDeviceNotify_->GetNotifyData(notifyInfo));

    return HCCL_SUCCESS;
}

HcclResult TransportBase::GetRxDataSigleDevNotifyInfo(HcclSignalInfo& notifyInfo)
{
    CHK_SMART_PTR_NULL(localSendReadyDeviceNotify_);
    CHK_RET(localSendReadyDeviceNotify_->GetNotifyData(notifyInfo));

    return HCCL_SUCCESS;
}

HcclResult TransportBase::ConstructExchangeDataForSend(u8*& exchangeDataPtr, u64& exchangeDataBlankSize)
{
    u64 dataLength = machinePara_.exchangeInfo.size();
    if (dataLength == 0) {
        HCCL_DEBUG("[Construct][ExchangeData]exchangeInfo size is 0.");
        return HCCL_SUCCESS;
    }

    HCCL_DEBUG("[Construct][ExchangeData]exchangeInfo size[%llu].", dataLength);
    CHK_SAFETY_FUNC_RET(memcpy_s(exchangeDataPtr, exchangeDataBlankSize, &machinePara_.exchangeInfo[0], dataLength));
    exchangeDataPtr += dataLength;
    exchangeDataBlankSize -= dataLength;
    return HCCL_SUCCESS;
}

HcclResult TransportBase::ParseExchangeData(u8*& exchangeDataPtr, u64& exchangeDataBlankSize)
{
    u64 dataLength = machinePara_.exchangeInfo.size();
    if (dataLength == 0) {
        HCCL_DEBUG("[Parse][ExchangeData]exchangeInfo size is 0.");
        return HCCL_SUCCESS;
    }
    exchangeMsg_.resize(dataLength);
    CHK_SAFETY_FUNC_RET(memcpy_s(&exchangeMsg_[0], exchangeMsg_.size(), exchangeDataPtr, dataLength));
    exchangeDataPtr += dataLength;
    exchangeDataBlankSize -= dataLength;

    return HCCL_SUCCESS;
}

HcclResult TransportBase::SendExchangeData(void)
{
    u64 dataLength = machinePara_.exchangeInfo.size();
    if (dataLength == 0) {
        HCCL_DEBUG("[Send][ExchangeData]exchangeInfo size is 0.");
        return HCCL_SUCCESS;
    }

    HCCL_DEBUG("[Send][ExchangeData]exchangeInfo size[%llu].", dataLength);
    CHK_SMART_PTR_NULL(defaultSocket_);
    HcclResult ret = defaultSocket_->Send(machinePara_.exchangeInfo.data(), dataLength);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[Send][ExchangeData]failed to send custom exchange data size [%llu].", dataLength), ret);

    return HCCL_SUCCESS;
}
HcclResult TransportBase::RecvAndCheckExchangeData(void)
{
    u64 dataLength = machinePara_.exchangeInfo.size();
    if (dataLength == 0) {
        HCCL_DEBUG("[Check][ExchangeData]exchangeInfo size is 0.");
        return HCCL_SUCCESS;
    }
    exchangeMsg_.resize(dataLength);

    CHK_SMART_PTR_NULL(defaultSocket_);
    HcclResult ret = defaultSocket_->Recv(exchangeMsg_.data(), dataLength);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[Check][ExchangeData]failed to recv custom exchange data size [%llu].", dataLength), ret);

    return HCCL_SUCCESS;
}

HcclResult
TransportBase::OpenRemoteNotify(const std::vector<u8>& byteVector, std::shared_ptr<RemoteNotify>& remoteNotify)
{
    EXCEPTION_CATCH((remoteNotify = std::make_shared<RemoteNotify>()), return HCCL_E_PTR);
    CHK_SMART_PTR_NULL(remoteNotify);

    HcclResult ret = HCCL_SUCCESS;
    bool errorFlag = false;
    do {
        ret = remoteNotify->Init(byteVector);
        CHK_PRT_BREAK(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[TransportBase][OpenRemoteNotify]remoteNotify init failed, "
                "ret[%d]",
                ret),
            errorFlag = true);

        ret = remoteNotify->Open();
        CHK_PRT_BREAK(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[TransportBase][OpenRemoteNotify]remoteNotify open failed, "
                "ret[%d]",
                ret),
            errorFlag = true);
    } while (0);

    if (errorFlag) {
        HCCL_ERROR("[TransportBase][OpenRemoteNotify]remoteNotify open failed ,ret[%d]", ret);
        remoteNotify = nullptr;
        return ret;
    }
    return HCCL_SUCCESS;
}

HcclResult TransportBase::PostReady(Stream& stream)
{
    static_cast<void>(stream);
    return HCCL_SUCCESS;
}

HcclResult TransportBase::WaitReady(Stream& stream)
{
    static_cast<void>(stream);
    return HCCL_SUCCESS;
}

HcclResult TransportBase::PostFin(Stream& stream)
{
    static_cast<void>(stream);
    return HCCL_SUCCESS;
}

HcclResult TransportBase::WaitFin(Stream& stream)
{
    static_cast<void>(stream);
    return HCCL_SUCCESS;
}

HcclResult TransportBase::PostFinAck(Stream& stream)
{
    static_cast<void>(stream);
    return HCCL_SUCCESS;
}

HcclResult TransportBase::WaitFinAck(Stream& stream)
{
    static_cast<void>(stream);
    return HCCL_SUCCESS;
}

HcclResult TransportBase::SetStopFlag(bool value)
{
    stopFlag_.store(value);
    return HCCL_SUCCESS;
}

bool TransportBase::GetStopFlag() { return stopFlag_.load(); }

HcclResult TransportBase::UpdateRemoteAddr(void* remoteIn, void* remoteOut)
{
    static_cast<void>(remoteIn);
    static_cast<void>(remoteOut);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult
TransportBase::WriteAsync(struct Transport::Buffer& remoteBuf, struct Transport::Buffer& localBuf, Stream& stream)
{
    static_cast<void>(remoteBuf);
    static_cast<void>(localBuf);
    static_cast<void>(stream);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult
TransportBase::WriteSync(struct Transport::Buffer& remoteBuf, struct Transport::Buffer& localBuf, Stream& stream)
{
    static_cast<void>(remoteBuf);
    static_cast<void>(localBuf);
    static_cast<void>(stream);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TransportBase::WriteReduceAsync(
    struct Transport::Buffer& remoteBuf, struct Transport::Buffer& localBuf, const HcclDataType datatype,
    HcclReduceOp redOp, Stream& stream)
{
    static_cast<void>(remoteBuf);
    static_cast<void>(localBuf);
    static_cast<void>(datatype);
    static_cast<void>(redOp);
    static_cast<void>(stream);

    return HCCL_E_NOT_SUPPORT;
}

HcclResult
TransportBase::ReadAsync(struct Transport::Buffer& localBuf, struct Transport::Buffer& remoteBuf, Stream& stream)
{
    static_cast<void>(localBuf);
    static_cast<void>(remoteBuf);
    static_cast<void>(stream);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult
TransportBase::ReadSync(struct Transport::Buffer& localBuf, struct Transport::Buffer& remoteBuf, Stream& stream)
{
    static_cast<void>(localBuf);
    static_cast<void>(remoteBuf);
    static_cast<void>(stream);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TransportBase::ReadReduceSync(
    struct Transport::Buffer& localBuf, struct Transport::Buffer& remoteBuf, const HcclDataType datatype,
    HcclReduceOp redOp, Stream& stream)
{
    static_cast<void>(remoteBuf);
    static_cast<void>(localBuf);
    static_cast<void>(datatype);
    static_cast<void>(redOp);
    static_cast<void>(stream);

    return HCCL_E_NOT_SUPPORT;
}

HcclResult
TransportBase::BatchTransferAsync(const HcommBatchTransferDesc* transferDescs, uint32_t descNum, Stream& stream)
{
    static_cast<void>(transferDescs);
    static_cast<void>(descNum);
    static_cast<void>(stream);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TransportBase::Fence() { return HCCL_E_NOT_SUPPORT; }

HcclResult TransportBase::Drain(Stream& stream)
{
    static_cast<void>(stream);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TransportBase::InitDrainNotifyInfo() { return HCCL_E_NOT_SUPPORT; }

HcclResult TransportBase::GetDrainRemSrcMem(void*& remoteAddr, uint32_t& remoteKey, uint32_t& size)
{
    (void)remoteAddr;
    (void)remoteKey;
    (void)size;
    return HCCL_E_NOT_SUPPORT;
}
} // namespace hccl
