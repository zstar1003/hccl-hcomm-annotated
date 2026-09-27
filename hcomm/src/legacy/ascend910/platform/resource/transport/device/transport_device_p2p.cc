/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "transport_device_p2p.h"
#include <securec.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <arpa/inet.h>
#include <unistd.h>
#include "adapter_hal_pub.h"

namespace hccl {
TransportDeviceP2p::TransportDeviceP2p(
    DispatcherPub* dispatcher, const std::unique_ptr<NotifyPool>& notifyPool, MachinePara& machinePara,
    std::chrono::milliseconds timeout, const TransportDeviceP2pData& transDevP2pData)
    : TransportP2p(dispatcher, notifyPool, machinePara, timeout)
{
    this->remoteInputPtr_ = transDevP2pData.inputBufferPtr;
    this->remoteOutputPtr_ = transDevP2pData.outputBufferPtr;
    this->transportAttr_ = transDevP2pData.transportAttr;

    if (!machinePara.isNewOneSide) {
        this->SetNotifyPtr(transDevP2pData);
    }
}

TransportDeviceP2p::~TransportDeviceP2p() { HCCL_DEBUG("[TransportDeviceP2p] ~TransportDeviceP2p Success!"); }

HcclResult TransportDeviceP2p::Init()
{
    HCCL_INFO(
        "[TransportDeviceP2p][Init] machineType=[%d], serverId=[%s], localDeviceId=[%d] remoteDeviceId=[%d], "
        "localRank=[%u], localUserRank=[%u], remoteRank=[%u], remoteUserRank=[%u], deviceType=[%d], "
        "input_ptr=[%p], output_ptr=[%p], linkAttribute=[0x%x], linkMode=[%d], notifyNum[%u]",
        machinePara_.machineType, machinePara_.serverId.c_str(), machinePara_.localDeviceId,
        machinePara_.remoteDeviceId, machinePara_.localUserrank, machinePara_.localWorldRank,
        machinePara_.remoteUserrank, machinePara_.remoteWorldRank, machinePara_.deviceType, machinePara_.inputMem.ptr(),
        machinePara_.outputMem.ptr(), machinePara_.linkAttribute, machinePara_.linkMode, machinePara_.notifyNum);

    HCCL_DEBUG(
        "[TransportDeviceP2p][Init] transport attr: linktype[0x%x], relationship[0x%x], "
        "signalRecordBuff.addr[%p], signalRecordBuff.length[%llu]",
        transportAttr_.linkType, transportAttr_.relationship, transportAttr_.signalRecordBuff.address,
        transportAttr_.signalRecordBuff.length);
    HcclUs startut = TIME_NOW();

    SetUseSdmaToSignalRecord();
    if (!machinePara_.isNewOneSide) {
        CHK_RET(ConfigUseSdmaCopyToSignalRecord());
        CHK_RET(this->SetNotify());
    } else {
        // init localHcclMemExMgr_ and remoteHcclMemExMgr_ from machinePara_.localBufMem and machinePara_.remoteBufMem
        CHK_RET(InitHcclMemExMgr(machinePara_));
    }

    HcclUs endut = TIME_NOW();
    HCCL_INFO("[TransportDeviceP2p][Init] take time:%lld us", DURATION_US(endut - startut));

    HCCL_USER_CRITICAL_LOG(
        "create hccl transport:communicator[%s], local rank[%u], remote rank[%u], "
        "transporttype[%s]",
        machinePara_.tag.c_str(), machinePara_.localUserrank, machinePara_.remoteUserrank,
        GetLinkTypeEnumStr(GetLinkType()).c_str());

    return HCCL_SUCCESS;
}

HcclResult TransportDeviceP2p::UpdateRemoteAddr(void* remoteIn, void* remoteOut)
{
    this->remoteInputPtr_ = remoteIn;
    this->remoteOutputPtr_ = remoteOut;
    return HCCL_SUCCESS;
}

extern "C" {
drvError_t __attribute__((weak))
halResAddrMap(unsigned int devId, struct res_addr_info* res_info, unsigned long* va, unsigned int* len);
};

HcclResult TransportDeviceP2p::GetNotifyAddr(s32 deviceId, const HcclSignalInfo& signalInfo, u64& addr)
{
    if (halResAddrMap == nullptr) {
        HCCL_ERROR("driver package does not support function [halResAddrMap], please update the package.");
        return HCCL_E_DRV;
    }

    unsigned int drvDevid = 0;
    CHK_RET(hrtDrvGetLocalDevIDByHostDevID(static_cast<uint32_t>(deviceId), &drvDevid));

    res_addr_info resInfo = {};
    resInfo.id = signalInfo.tsId;
    resInfo.target_proc_type = PROCESS_CP1;
    resInfo.res_type = RES_ADDR_TYPE_STARS_NOTIFY_RECORD;
    resInfo.res_id = static_cast<uint32_t>(signalInfo.resId);
    resInfo.flag = signalInfo.flag;
    resInfo.rudevid = signalInfo.devId;
    resInfo.rsv[0] = 0; // 0 is reserved array idx
    resInfo.rsv[1] = 0; // 1 is reserved array idx

    unsigned int len = 0;
    int ret = halResAddrMap(drvDevid, &resInfo, reinterpret_cast<uint64_t*>(&addr), &len);
    if (ret != 0 || len != transportAttr_.signalRecordBuff.length || len == 0) {
        HCCL_ERROR(
            "[drv api]res get addr failed, result:%d, devid:%d, resType:%d, resId:%u, tsId:%d, ruDevId:%d, "
            "flag:%d, addr:%p, notify len:%u",
            ret, drvDevid, resInfo.res_type, resInfo.res_id, resInfo.id, resInfo.rudevid, resInfo.flag, addr, len);
        return HCCL_E_DRV;
    }
    HCCL_DEBUG(
        "get notify address success, devid:%d, drvDevid:%u, resType:%d, resId:%u, tsId:%d, ruDevId:%d, flag:%d, "
        "addr:%p",
        deviceId, drvDevid, resInfo.res_type, resInfo.res_id, resInfo.id, resInfo.rudevid, resInfo.flag, addr);
    return HCCL_SUCCESS;
}

template <typename T>
HcclResult TransportDeviceP2p::ModifySignalAddrToVA(s32 deviceId, std::shared_ptr<T>& notify)
{
    HcclSignalInfo signalInfo;
    CHK_PTR_NULL(notify);
    CHK_RET(notify->GetNotifyData(signalInfo));
    CHK_RET(GetNotifyAddr(deviceId, signalInfo, signalInfo.addr));
    CHK_RET(notify->SetNotifyData(signalInfo));
    return HCCL_SUCCESS;
}

HcclResult TransportDeviceP2p::CheckRelationship(u32 relationship)
{
    constexpr u32 sameChip = HCCL_TRANSPORT_RELATIONSHIP_SAME_SUPERPOD | HCCL_TRANSPORT_RELATIONSHIP_SAME_SERVER
                             | HCCL_TRANSPORT_RELATIONSHIP_SAME_CHIP;
    constexpr u32 sameServer = HCCL_TRANSPORT_RELATIONSHIP_SAME_SUPERPOD | HCCL_TRANSPORT_RELATIONSHIP_SAME_SERVER;
    constexpr u32 sameSuperpod = HCCL_TRANSPORT_RELATIONSHIP_SAME_SUPERPOD;

    if ((relationship != sameChip) && (relationship != sameServer) && (relationship != sameSuperpod)) {
        HCCL_ERROR("[TransportDeviceP2p] relationship is not supported, relationship:%d", relationship);
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

HcclResult TransportDeviceP2p::ConfigUseSdmaCopyToSignalRecord()
{
    CHK_RET(CheckRelationship(transportAttr_.relationship));

    // AICPU展开时，在节点间使用SDMA进行notify record操作，STARS可检出节点间链路异常，触发HCCL重执行
    if (useSdmaToSignalRecord_) {
        HCCL_DEBUG("[TransportDeviceP2p] use sdma to signal record");
        // NOTE: DRV只支持跨节点的 notify VA 映射，不支持节点内和本地的 notify VA 映射
        CHK_RET(ModifySignalAddrToVA(machinePara_.localDeviceId, remoteSendReadyNotify_));
        CHK_RET(ModifySignalAddrToVA(machinePara_.localDeviceId, remoteSendDoneNotify_));
        for (u32 i = 0; i < notifyNum_; i++) {
            CHK_RET(ModifySignalAddrToVA(machinePara_.localDeviceId, userRemoteNotify_[i]));
        }
        signalMem_ = DeviceMem::create(
            reinterpret_cast<void*>(transportAttr_.signalRecordBuff.address), transportAttr_.signalRecordBuff.length);
        CHK_SMART_PTR_NULL(signalMem_);
    }
    return HCCL_SUCCESS;
}

HcclResult TransportDeviceP2p::SignalRecord(
    std::shared_ptr<RemoteNotify>& remoteSignal, u64 remoteSignalAddr, u64 remoteSignalOffset, Stream& stream)
{
    HcclSignalInfo notifyInfo;
    CHK_RET(remoteSignal->GetNotifyData(notifyInfo));
    if (useSdmaToSignalRecord_) {
        DeviceMem dstDevMem
            = DeviceMem::create(reinterpret_cast<void*>(remoteSignalAddr), transportAttr_.signalRecordBuff.length);
        CHK_SMART_PTR_NULL(dstDevMem);
        return dispatcher_->SignalRecord(
            dstDevMem, signalMem_, stream, machinePara_.remoteWorldRank, transportAttr_.linkType, notifyInfo.resId);
    } else {
        return dispatcher_->SignalRecord(
            remoteSignal->ptr(), stream, machinePara_.remoteWorldRank, remoteSignalOffset, INVALID_VALUE_STAGE, false,
            remoteSignalAddr, notifyInfo.resId);
    }
}
} // namespace hccl
