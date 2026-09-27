/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "transport_p2p.h"
#include <securec.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <arpa/inet.h>
#include <unistd.h>

#include "mem_name_repository_pub.h"
#include "adapter_rts.h"
#include "mem_host_pub.h"

namespace hccl {
std::array<DeviceMem, MAX_MODULE_DEVICE_NUM> TransportP2p::notifyValueMem_;
std::array<std::mutex, MAX_MODULE_DEVICE_NUM> TransportP2p::notifyValueMutex_;
std::array<Referenced, MAX_MODULE_DEVICE_NUM> TransportP2p::instanceRef_;
TransportP2p::TransportP2p(
    DispatcherPub* dispatcher, const std::unique_ptr<NotifyPool>& notifyPool, MachinePara& machinePara,
    std::chrono::milliseconds timeout)
    : TransportBase(dispatcher, notifyPool, machinePara, timeout),
      remoteInputPtr_(nullptr),
      remoteOutputPtr_(nullptr),
      remoteOutputOffsetValue_(0),
      remoteInputOffsetValue_(0),
      remoteOutputMemName_(),
      remoteInputMemName_()
{
    if (machinePara_.deviceLogicId >= 0 && (static_cast<u32>(machinePara_.deviceLogicId) < MAX_MODULE_DEVICE_NUM)) {
        instanceRef_[machinePara_.deviceLogicId].Ref();
    }
    userLocalNotify_.resize(notifyNum_);
    userRemoteNotify_.resize(notifyNum_);
    userRemoteNotifyAddr_.resize(notifyNum_);
    userRemoteNotifyOffset_.resize(notifyNum_);
    remoteIpcMemPtrVector_.resize(machinePara.mem.size());
    remoteIpcMemOffsetValueVector_.resize(machinePara.mem.size());
    remoteIpcMemSizeVector_.resize(machinePara.mem.size());
    remoteIpcMemNameVector_.resize(machinePara.mem.size());
}

TransportP2p::~TransportP2p()
{
    HCCL_DEBUG("~TransportP2p Enter!");

    // 关闭rtIpcOpenMemory打开的对端共享内存和内存名称映射
    if (!isMemInclude_) {
        MemNameRepository::GetInstance(machinePara_.deviceLogicId)
            ->CloseIpcMem(static_cast<const u8*>(remoteOutputMemName_.ipcName));
        HCCL_DEBUG("remoteOutputMemName_.ipcName[%d]", remoteOutputMemName_.ipcName);
        MemNameRepository::GetInstance(machinePara_.deviceLogicId)
            ->CloseIpcMem(static_cast<const u8*>(remoteInputMemName_.ipcName));
        HCCL_DEBUG("remoteInputMemName_.ipcName[%d]", remoteInputMemName_.ipcName);
    }
    for (u32 i = 0; i < machinePara_.mem.size(); i++) {
        MemNameRepository::GetInstance(machinePara_.deviceLogicId)
            ->CloseIpcMem(static_cast<const u8*>(remoteIpcMemNameVector_[i].ipcName));
        HCCL_DEBUG("remoteIpcMemNameVector_[%u].ipcName[%s]", i, remoteIpcMemNameVector_[i].ipcName);
    }

    // 关闭rtIpcSetMemoryName 设置的内存名
    if (!isMemInclude_) {
        MemNameRepository::GetInstance(machinePara_.deviceLogicId)
            ->DestroyIpcMem(machinePara_.outputMem.ptr(), machinePara_.outputMem.size(), isSioToHccs_);
        HCCL_DEBUG(
            "machinePara_.outputMem addr:[%p], size:[%llu]", machinePara_.outputMem.ptr(),
            machinePara_.outputMem.size());
        MemNameRepository::GetInstance(machinePara_.deviceLogicId)
            ->DestroyIpcMem(machinePara_.inputMem.ptr(), machinePara_.inputMem.size(), isSioToHccs_);
        HCCL_DEBUG(
            "machinePara_.inputMem addr:[%p], size:[%llu]", machinePara_.inputMem.ptr(), machinePara_.inputMem.size());
    }
    for (u32 i = 0; i < machinePara_.mem.size(); i++) {
        MemNameRepository::GetInstance(machinePara_.deviceLogicId)
            ->DestroyIpcMem(machinePara_.mem[i].ptr(), machinePara_.mem[i].size(), isSioToHccs_);
        HCCL_DEBUG(
            "machinePara_.mem[%u] addr:[%p], size:[%llu]", machinePara_.mem[i].ptr(), machinePara_.mem[i].size());
    }

    SignalDestroy();

    if (machinePara_.deviceLogicId >= 0 && (static_cast<u32>(machinePara_.deviceLogicId) < MAX_MODULE_DEVICE_NUM)) {
        if (instanceRef_[machinePara_.deviceLogicId].Unref() == 0) {
            std::unique_lock<std::mutex> lock(notifyValueMutex_[machinePara_.deviceLogicId]);
            notifyValueMem_[machinePara_.deviceLogicId].free();
        }
    }
    HCCL_DEBUG("~TransportP2p Success!");
}

HcclResult TransportP2p::Init()
{
    HCCL_INFO(
        "machineType=[%d], serverId=[%s], localDeviceId=[%d], remoteDeviceId=[%d], "
        "localRank=[%u], localUserRank=[%u], remoteRank=[%u], remoteUserRank=[%u], "
        "deviceType=[%d], input_ptr=[%p], output_ptr=[%p], linkAttribute=[0x%x], linkMode=[%d], "
        "notifyNum[%u], isIndOp[%d], custom exchange data size [%llu], specifyLink[%d].",
        machinePara_.machineType, machinePara_.serverId.c_str(), machinePara_.localDeviceId,
        machinePara_.remoteDeviceId, machinePara_.localUserrank, machinePara_.localWorldRank,
        machinePara_.remoteUserrank, machinePara_.remoteWorldRank, machinePara_.deviceType, machinePara_.inputMem.ptr(),
        machinePara_.outputMem.ptr(), machinePara_.linkAttribute, machinePara_.linkMode, machinePara_.notifyNum,
        machinePara_.isIndOp, machinePara_.exchangeInfo.size(), machinePara_.specifyLink);
    HcclUs startut = TIME_NOW();

    /* make input memory shared interprocess and assigned a name */
    if (!machinePara_.isNewOneSide) {
        CHK_SMART_PTR_NULL(machinePara_.inputMem);
        CHK_SMART_PTR_NULL(machinePara_.outputMem);
    }

    CHK_PTR_NULL(dispatcher_);
    CHK_SMART_PTR_NULL(notifyPool_);
    CHK_RET(CheckDeviceId());
    CHK_RET(CheckExchangeData());
    SetMemIncludeFlag();
    // 上层初始化时保证 machinePara_.sockets 非空
    if (machinePara_.sockets.size() == 0) {
        HCCL_ERROR("machinePara sockets is empty.");
        return HCCL_E_INTERNAL;
    }
    defaultSocket_ = machinePara_.sockets[0];
    CHK_PTR_NULL(defaultSocket_);

    CHK_RET(CheckLinkMode());

    /* 本端与远端交换tgid 信息 */
    CHK_RET(ExchangeTgidMesg()); // tgid 无法合并交换，因为依赖对端的tgid判定是同一个进程还是跨进程

    CHK_RET(SetLinkType()); // 需要在交换sdid之后调用，确定是否超节点内节点间HCCS场景

    CHK_RET(FillExchangeDataTotalSize());

    CHK_RET(ConstructExchangeForSend());

    HcclResult ret = defaultSocket_->Send(exchangeDataForSend_.data(), exchangeDataTotalSize_);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[TransportP2p][Init] failed to send exchangeData exchangeDataTotalSize[%llu], custom exchange data "
            "size [%llu].",
            exchangeDataTotalSize_, machinePara_.exchangeInfo.size()),
        ret);

    exchangeDataForRecv_.resize(exchangeDataTotalSize_);
    ret = defaultSocket_->Recv(exchangeDataForRecv_.data(), exchangeDataTotalSize_);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[TransportP2p][Init] failed to recv exchangeData exchangeDataTotalSize[%llu], custom exchange data "
            "size [%llu].",
            exchangeDataTotalSize_, machinePara_.exchangeInfo.size()),
        ret);

    HCCL_DEBUG("[TransportP2p][Init] Socket Data Received");

    CHK_RET(ParseReceivedExchangeData());

    SetTransportRelationship();
    SetUseSdmaToSignalRecord();
    CHK_RET(CreateNotifyValueBuffer());

    HcclUs endut = TIME_NOW();
    HCCL_INFO("Time:%lld us", DURATION_US(endut - startut));

    HCCL_USER_CRITICAL_LOG(
        "create hccl transport:communicator[%s], local rank[%u], remote rank[%u], "
        "transporttype[%s]",
        machinePara_.tag.c_str(), machinePara_.localUserrank, machinePara_.remoteUserrank,
        GetLinkTypeEnumStr(GetLinkType()).c_str());

    return HCCL_SUCCESS;
}

void TransportP2p::SetUseSdmaToSignalRecord()
{
    // AICPU展开时，在节点间使用SDMA进行notify record操作，STARS可检出节点间链路异常，触发HCCL重执行
    useSdmaToSignalRecord_
        = ((transportAttr_.relationship & HCCL_TRANSPORT_RELATIONSHIP_SAME_SERVER) == 0)
          && ((transportAttr_.linkType == LinkType::LINK_HCCS_SW) || (transportAttr_.linkType == LinkType::LINK_HCCS));
}

HcclResult TransportP2p::ParseSpecifyLink(LinkTypeInServer& linkType)
{
    if (machinePara_.specifyLink == LinkTypeInServer::RESERVED_LINK_TYPE || machinePara_.specifyLink == linkType) {
        return HCCL_SUCCESS; // 未指定切换链路，保持默认
    } else if (machinePara_.specifyLink == LinkTypeInServer::HCCS_SW_TYPE && linkType == LinkTypeInServer::SIO_TYPE) {
        // 切换链路基于ipc实现, 多线程场景暂不支持
        s32 sendPid = 0;
        CHK_RET(SalGetBareTgid(&sendPid));
        CHK_PRT_RET(
            sendPid == recvPid_, HCCL_WARNING("%s specifyLink is not supported in multi-thread", __func__),
            HCCL_SUCCESS);

        // A3 DIE间通信场景, 将链路从SIO切换到HCCS
        linkType = LinkTypeInServer::HCCS_SW_TYPE;
        isSioToHccs_ = true;
        HCCL_INFO("%s specifyLink change to HCCS_SW_TYPE", __func__);
    } else {
        HCCL_ERROR(
            "%s fail, linkType:%d, specifyLink:%d is not supported", __func__, linkType, machinePara_.specifyLink);
        return HCCL_E_NOT_SUPPORT;
    }
    return HCCL_SUCCESS;
}

HcclResult TransportP2p::SetLinkType()
{
    // 计算linkType
    LinkTypeInServer linkType = LinkTypeInServer::HCCS_TYPE;
    if (recvSdid_ != INVALID_INT) { // 超节点内节点间走p2p通信时，链路类型为LINK_HCCS_SW
        linkType = LinkTypeInServer::HCCS_SW_TYPE;
    } else {
        CHK_RET(hrtGetPairDeviceLinkType(
            static_cast<u32>(machinePara_.localDeviceId), static_cast<u32>(machinePara_.remoteDeviceId), linkType));
    }

    CHK_RET(ParseSpecifyLink(linkType));

    switch (linkType) {
        case LinkTypeInServer::HCCS_TYPE:
            transportAttr_.linkType = hccl::LinkType::LINK_HCCS;
            break;
        case LinkTypeInServer::HCCS_SW_TYPE:
            transportAttr_.linkType = hccl::LinkType::LINK_HCCS_SW;
            break;
        case LinkTypeInServer::SIO_TYPE:
            transportAttr_.linkType = hccl::LinkType::LINK_SIO;
            break;
        default:
            transportAttr_.linkType = hccl::LinkType::LINK_PCIE;
            break;
    }

    HCCL_DEBUG("[TransportP2p] transportattr linktype: 0x%x", transportAttr_.linkType);
    return HCCL_SUCCESS;
}

HcclResult TransportP2p::CreateNotifyValueBuffer()
{
    if (!useSdmaToSignalRecord_) {
        return HCCL_SUCCESS;
    }

    u32 notifySize = 0;
    CHK_RET(hrtGetNotifySize(notifySize));
    std::unique_lock<std::mutex> lock(notifyValueMutex_[machinePara_.deviceLogicId]);
    if (notifyValueMem_[machinePara_.deviceLogicId].ptr() == nullptr) {
        u64 notifyVaule = 1; // notify值写1表示record
        CHK_RET(DeviceMem::alloc(notifyValueMem_[machinePara_.deviceLogicId], notifyValueSize_));
        HCCL_DEBUG(
            "create notify value buffer[%p], size[%u]", notifyValueMem_[machinePara_.deviceLogicId].ptr(), notifySize);

        CHK_RET(hrtMemSyncCopy(
            notifyValueMem_[machinePara_.deviceLogicId].ptr(), notifyValueMem_[machinePara_.deviceLogicId].size(),
            &notifyVaule, notifySize, HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));
    }
    transportAttr_.signalRecordBuff.address = reinterpret_cast<u64>(notifyValueMem_[machinePara_.deviceLogicId].ptr());
    transportAttr_.signalRecordBuff.length = notifySize;

    HCCL_DEBUG(
        "[TransportP2p] transportattr signalRecordBuff.address[%p], signalRecordBuff.length[%llu]",
        transportAttr_.signalRecordBuff.address, transportAttr_.signalRecordBuff.length);
    return HCCL_SUCCESS;
}

void TransportP2p::SetTransportRelationship()
{
    if (transportAttr_.linkType == hccl::LinkType::LINK_SIO) {
        // 芯片内
        transportAttr_.relationship |= HCCL_TRANSPORT_RELATIONSHIP_SAME_CHIP;
        transportAttr_.relationship |= HCCL_TRANSPORT_RELATIONSHIP_SAME_SERVER;
        transportAttr_.relationship |= HCCL_TRANSPORT_RELATIONSHIP_SAME_SUPERPOD;
    } else if (recvSdid_ == INVALID_INT) {
        // 节点内
        transportAttr_.relationship |= HCCL_TRANSPORT_RELATIONSHIP_SAME_SERVER;
        transportAttr_.relationship |= HCCL_TRANSPORT_RELATIONSHIP_SAME_SUPERPOD;
    } else {
        // 节点间
        transportAttr_.relationship |= HCCL_TRANSPORT_RELATIONSHIP_SAME_SUPERPOD;
    }

    HCCL_DEBUG("[TransportP2p] transportattr relationship: 0x%x", transportAttr_.relationship);
    return;
}

HcclResult TransportP2p::FillExchangeDataTotalSize()
{
    exchangeDataTotalSize_ = 0;
    s32 sendPid = 0;
    CHK_RET(SalGetBareTgid(&sendPid));
    u64 ipcMemDataSize = 0;
    if (sendPid != recvPid_ || recvSdid_ != INVALID_INT) {
        // 输入输出内存
        HCCL_DEBUG("[TransportP2p][FillExchangeDataTotalSize] Inter Proc");
        ipcMemDataSize = HCCL_IPC_MEM_NAME_LEN + sizeof(u64) + sizeof(u64); // size + offset
        if (!isMemInclude_) {
            exchangeInfoSize_.ipcMenSize = ipcMemDataSize * (2 + machinePara_.mem.size());
        } else {
            // in和out包含在整块CCLbuf的时候，不需要传ipcName,但是size和offset不能少
            exchangeInfoSize_.ipcMenSize = ipcMemDataSize * machinePara_.mem.size() + 2 * (sizeof(u64) + sizeof(u64));
        }
    } else {
        HCCL_DEBUG("[TransportP2p][FillExchangeDataTotalSize] intra Proc");
        ipcMemDataSize = sizeof(u64) + sizeof(u64); // addr + length
        exchangeInfoSize_.ipcMenSize
            = ipcMemDataSize * (2 + machinePara_.mem.size()); // 2: input  & output + mem.size()
    }

    if (!machinePara_.isNewOneSide) {
        // notify 信息
        if (machinePara_.linkMode != LinkMode::LINK_SIMPLEX_MODE
            || machinePara_.machineType == MachineType::MACHINE_CLIENT_TYPE) {
            exchangeInfoSize_.notifySize = NOTIFY_INFO_LENGTH;
        }
        if (machinePara_.linkMode != LinkMode::LINK_SIMPLEX_MODE
            || machinePara_.machineType == MachineType::MACHINE_SERVER_TYPE) {
            exchangeInfoSize_.notifySize += NOTIFY_INFO_LENGTH;
        }
        // 3.新增notify资源
        exchangeInfoSize_.notifySize += NOTIFY_INFO_LENGTH * notifyNum_;
    }

    // 自定义信息
    exchangeInfoSize_.exDataSize = machinePara_.exchangeInfo.size();

    // 独立算子内存
    if (machinePara_.isIndOp) {
        // userDeviceMem数量\userDeviceMem\userHostMem数量\userHostMem
        const int kMemCountItems = 2;
        exchangeInfoSize_.indOpMemSize
            = ipcMemDataSize * (machinePara_.userDeviceMem.size() + machinePara_.userHostMem.size());
        exchangeInfoSize_.indOpMemSize += sizeof(u64) * kMemCountItems;
    }

    exchangeDataTotalSize_ = exchangeInfoSize_.ipcMenSize + exchangeInfoSize_.notifySize + exchangeInfoSize_.exDataSize
                             + exchangeInfoSize_.indOpMemSize + sizeof(ExchangeInfoSize);
    HCCL_INFO(
        "[TransportP2p][FillExchangeDataTotalSize] exchangeDataTotalSize[%llu] memSize[%d]", exchangeDataTotalSize_,
        machinePara_.mem.size());
    return HCCL_SUCCESS;
}

HcclResult TransportP2p::ConstructExchangeForSend()
{
    exchangeDataForSend_.resize(exchangeDataTotalSize_);
    u8* exchangeDataPtr = exchangeDataForSend_.data();
    u64 exchangeDataBlankSize = exchangeDataTotalSize_;
    CHK_RET(ConstructDataLenForSend(exchangeDataPtr, exchangeDataBlankSize));
    u64 blankSizeRecord = exchangeDataBlankSize;

    s32 sendPid = 0;
    CHK_RET(SalGetBareTgid(&sendPid));
    HCCL_DEBUG("%s sendPid %d, recvPid %d, recvSdid %d", __func__, sendPid, recvPid_, recvSdid_);
    if (sendPid != recvPid_ || recvSdid_ != INVALID_INT) { // 跨进程方式交换
        // 构造IPC内存地址交换数据结构
        for (auto ipcMem : machinePara_.mem) {
            CHK_RET(ConstructIpcMemInfoForSend(ipcMem.ptr(), ipcMem.size(), exchangeDataPtr, exchangeDataBlankSize));
        }
        if (!isMemInclude_) {
            CHK_RET(ConstructIpcMemInfoForSend(
                machinePara_.outputMem.ptr(), machinePara_.outputMem.size(), exchangeDataPtr, exchangeDataBlankSize));
            CHK_RET(ConstructIpcMemInfoForSend(
                machinePara_.inputMem.ptr(), machinePara_.inputMem.size(), exchangeDataPtr, exchangeDataBlankSize));
        } else {
            CHK_RET(ConstructMemIncludeInfoForSend(exchangeDataPtr, exchangeDataBlankSize));
        }
    } else {
        // 构造进程内内存地址交换数据结构
        CHK_RET(ConstructIntraProcMemInfoForSend(
            machinePara_.outputMem.ptr(), machinePara_.outputMem.size(), exchangeDataPtr, exchangeDataBlankSize));
        CHK_RET(ConstructIntraProcMemInfoForSend(
            machinePara_.inputMem.ptr(), machinePara_.inputMem.size(), exchangeDataPtr, exchangeDataBlankSize));
        for (auto ipcMem : machinePara_.mem) {
            CHK_RET(
                ConstructIntraProcMemInfoForSend(ipcMem.ptr(), ipcMem.size(), exchangeDataPtr, exchangeDataBlankSize));
        }
    }
    CHK_RET(SumCheckSizeAndConsistent(
        ExInfoType::EX_IPCMEN_SIZE, exchangeInfoSize_.ipcMenSize, blankSizeRecord, exchangeDataBlankSize));

    CHK_RET(ConstructNotifyInfoForSend(exchangeDataPtr, exchangeDataBlankSize));
    CHK_RET(ConstructNotifyVectorInfoForSend(exchangeDataPtr, exchangeDataBlankSize)); // 新增notify资源的创建
    CHK_RET(SumCheckSizeAndConsistent(
        ExInfoType::EX_NOTIFY_SIZE, exchangeInfoSize_.notifySize, blankSizeRecord, exchangeDataBlankSize));

    CHK_RET(ConstructExchangeDataForSend(exchangeDataPtr, exchangeDataBlankSize));
    CHK_RET(SumCheckSizeAndConsistent(
        ExInfoType::EX_EXDATA_SIZE, exchangeInfoSize_.exDataSize, blankSizeRecord, exchangeDataBlankSize));

    // 独立算子内存资源，无需检查大小
    if (machinePara_.isIndOp) {
        if (sendPid != recvPid_ || recvSdid_ != INVALID_INT) { // 跨进程方式交换
            CHK_RET(ConstructNumInfoForSend(machinePara_.userDeviceMem.size(), exchangeDataPtr, exchangeDataBlankSize));
            for (auto ipcMem : machinePara_.userDeviceMem) {
                CHK_RET(
                    ConstructIpcMemInfoForSend(ipcMem.ptr(), ipcMem.size(), exchangeDataPtr, exchangeDataBlankSize));
            }
            CHK_RET(ConstructNumInfoForSend(machinePara_.userHostMem.size(), exchangeDataPtr, exchangeDataBlankSize));
            for (auto ipcMem : machinePara_.userHostMem) {
                CHK_RET(
                    ConstructIpcMemInfoForSend(ipcMem.ptr(), ipcMem.size(), exchangeDataPtr, exchangeDataBlankSize));
            }
        } else {
            CHK_RET(ConstructNumInfoForSend(machinePara_.userDeviceMem.size(), exchangeDataPtr, exchangeDataBlankSize));
            for (auto ipcMem : machinePara_.userDeviceMem) {
                CHK_RET(ConstructIntraProcMemInfoForSend(
                    ipcMem.ptr(), ipcMem.size(), exchangeDataPtr, exchangeDataBlankSize));
            }
            CHK_RET(ConstructNumInfoForSend(machinePara_.userHostMem.size(), exchangeDataPtr, exchangeDataBlankSize));
            for (auto ipcMem : machinePara_.userHostMem) {
                CHK_RET(ConstructIntraProcMemInfoForSend(
                    ipcMem.ptr(), ipcMem.size(), exchangeDataPtr, exchangeDataBlankSize));
            }
        }
    }
    if (exchangeDataBlankSize != 0) {
        HCCL_ERROR(
            "[TransportP2p][ConstructExchangeForSend] failed to construct exchange Data "
            "exchangeDataBlankSize[%llu]",
            exchangeDataBlankSize);
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS; // this function should not be called in normal process
}

// exchangeDataPtr对指针进行了引用，因为需要改变exchangeDataPtr的值
HcclResult
TransportP2p::ConstructIpcMemInfoForSend(void* ptr, u64 size, u8*& exchangeDataPtr, u64& exchangeDataBlankSize)
{
    HcclResult ret;
    u64 memOffset;
    SecIpcName_t memName;

    if (!machinePara_.isNewOneSide) {
        ret = MemNameRepository::GetInstance(machinePara_.deviceLogicId)
                  ->SetIpcMem(
                      ptr, size, memName.ipcName, HCCL_IPC_MEM_NAME_LEN, memOffset, recvPid_, recvSdid_, isSioToHccs_);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[Send][IpcMemMesg]errNo[0x%016llx], In send ipc mesg, get para mem name failed. "
                "mem addr[%p] local rank[%u]",
                HCCL_ERROR_CODE(ret), machinePara_.outputMem.ptr(), machinePara_.localUserrank),
            ret);
    }

    // 设置ipc mem属性，指定通信链路从sio切换至hccs
    if (isSioToHccs_) {
        u32 ipcAttr = 1; // 0: SIO(默认), 1: HCCS
        CHK_RET(hrtIpcSetMemoryAttr(memName.ipcName, ACL_RT_IPC_MEM_ATTR_ACCESS_LINK, ipcAttr));
    }

    CHK_SAFETY_FUNC_RET(memcpy_s(exchangeDataPtr, exchangeDataBlankSize, memName.ipcName, HCCL_IPC_MEM_NAME_LEN));
    exchangeDataPtr += HCCL_IPC_MEM_NAME_LEN;
    exchangeDataBlankSize -= HCCL_IPC_MEM_NAME_LEN;
    CHK_SAFETY_FUNC_RET(memcpy_s(exchangeDataPtr, exchangeDataBlankSize, &size, sizeof(u64)));
    exchangeDataPtr += sizeof(u64);
    exchangeDataBlankSize -= sizeof(u64);
    CHK_SAFETY_FUNC_RET(memcpy_s(exchangeDataPtr, exchangeDataBlankSize, &memOffset, sizeof(u64)));
    exchangeDataPtr += sizeof(u64);
    exchangeDataBlankSize -= sizeof(u64);

    return HCCL_SUCCESS;
}

HcclResult
TransportP2p::ConstructIntraProcMemInfoForSend(void* ptr, u64 size, u8*& exchangeDataPtr, u64& exchangeDataBlankSize)
{
    if (!machinePara_.isNewOneSide) {
        CHK_SAFETY_FUNC_RET(memcpy_s(exchangeDataPtr, exchangeDataBlankSize, &ptr, sizeof(u64)));
    }
    exchangeDataPtr += sizeof(u64);
    exchangeDataBlankSize -= sizeof(u64);
    CHK_SAFETY_FUNC_RET(memcpy_s(exchangeDataPtr, exchangeDataBlankSize, &size, sizeof(u64)));
    exchangeDataPtr += sizeof(u64);
    exchangeDataBlankSize -= sizeof(u64);

    return HCCL_SUCCESS;
}

HcclResult TransportP2p::ConstructNumInfoForSend(u64 num, u8*& exchangeDataPtr, u64& exchangeDataBlankSize)
{
    CHK_SAFETY_FUNC_RET(memcpy_s(exchangeDataPtr, exchangeDataBlankSize, &num, sizeof(u64)));
    exchangeDataPtr += sizeof(u64);
    exchangeDataBlankSize -= sizeof(u64);
    return HCCL_SUCCESS;
}

HcclResult TransportP2p::ParseMemNumInfo(u64& memNum, u8*& exchangeDataPtr, u64& exchangeDataBlankSize)
{
    CHK_SAFETY_FUNC_RET(memcpy_s(&memNum, sizeof(u64), exchangeDataPtr, sizeof(u64)));
    exchangeDataPtr += sizeof(u64);
    exchangeDataBlankSize -= sizeof(u64);
    return HCCL_SUCCESS;
}

HcclResult TransportP2p::ParseIpcMemInfo(
    void** memPtr, u64& size, u8* memName, u64& offset, u8*& exchangeDataPtr, u64& exchangeDataBlankSize)
{
    CHK_SAFETY_FUNC_RET(memcpy_s(memName, HCCL_IPC_MEM_NAME_LEN, exchangeDataPtr, HCCL_IPC_MEM_NAME_LEN));
    exchangeDataPtr += HCCL_IPC_MEM_NAME_LEN;
    exchangeDataBlankSize -= HCCL_IPC_MEM_NAME_LEN;

    CHK_SAFETY_FUNC_RET(memcpy_s(&size, sizeof(u64), exchangeDataPtr, sizeof(u64)));
    exchangeDataPtr += sizeof(u64);
    exchangeDataBlankSize -= sizeof(u64);

    CHK_SAFETY_FUNC_RET(memcpy_s(&offset, sizeof(u64), exchangeDataPtr, sizeof(u64)));
    exchangeDataPtr += sizeof(u64);
    exchangeDataBlankSize -= sizeof(u64);

    if (!machinePara_.isNewOneSide) {
        /* 根据名字，获取对端IPC 内存 */
        HcclResult ret = WaitPeerMemConfig(memPtr, const_cast<u8*>(memName), size, offset);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[Recv][IpcMemMesg]errNo[0x%016llx] In recv ipc mem mesg, wait peer mem config "
                "failed. local rank[%u]",
                HCCL_ERROR_CODE(ret), machinePara_.localUserrank),
            ret);

        CHK_PTR_NULL(*memPtr);
    }

    HCCL_DEBUG(
        "localUserrank[%u] receive from remoteUserrank[%u]", machinePara_.localUserrank, machinePara_.remoteUserrank);

    return HCCL_SUCCESS;
}

HcclResult TransportP2p::ParseIntraProcMemInfo(u64* addr, u64* size, u8*& exchangeDataPtr, u64& exchangeDataBlankSize)
{
    if (!machinePara_.isNewOneSide) {
        CHK_SAFETY_FUNC_RET(memcpy_s(addr, sizeof(u64), exchangeDataPtr, sizeof(u64)));
        CHK_PTR_NULL(reinterpret_cast<void*>(*addr));
    }

    exchangeDataPtr += sizeof(u64);
    exchangeDataBlankSize -= sizeof(u64);
    CHK_SAFETY_FUNC_RET(memcpy_s(size, sizeof(u64), exchangeDataPtr, sizeof(u64)));
    exchangeDataPtr += sizeof(u64);
    exchangeDataBlankSize -= sizeof(u64);
    return HCCL_SUCCESS;
}

HcclResult TransportP2p::ParseNotifyInfo(u8*& exchangeDataPtr, u64& exchangeDataBlankSize)
{
    s32 sendPid = 0;
    CHK_RET(SalGetBareTgid(&sendPid)); // 当前进程id
    HCCL_INFO("LinkRecvNotifyMesg, sendPid[%d], recvPid[%d]", sendPid, recvPid_);

    if (machinePara_.isAicpuModeEn) {
        if ((machinePara_.linkMode != LinkMode::LINK_SIMPLEX_MODE
             || machinePara_.machineType == MachineType::MACHINE_SERVER_TYPE)
            && machinePara_.isAicpuModeEn == true) {
            std::vector<u8> data(NOTIFY_INFO_LENGTH, 0);
            CHK_SAFETY_FUNC_RET(memcpy_s(&data[0], data.size(), exchangeDataPtr, NOTIFY_INFO_LENGTH));
            exchangeDataPtr += NOTIFY_INFO_LENGTH;
            exchangeDataBlankSize -= NOTIFY_INFO_LENGTH;
            CHK_RET(OpenRemoteNotify(data, remoteSendReadyDeviceNotify_));
        }

        if ((machinePara_.linkMode != LinkMode::LINK_SIMPLEX_MODE
             || machinePara_.machineType == MachineType::MACHINE_CLIENT_TYPE)
            && machinePara_.isAicpuModeEn == true) {
            std::vector<u8> data(NOTIFY_INFO_LENGTH, 0);
            CHK_SAFETY_FUNC_RET(memcpy_s(&data[0], data.size(), exchangeDataPtr, NOTIFY_INFO_LENGTH));
            exchangeDataPtr += NOTIFY_INFO_LENGTH;
            exchangeDataBlankSize -= NOTIFY_INFO_LENGTH;
            CHK_RET(OpenRemoteNotify(data, remoteSendDoneDeviceNotify_));
        }
    } else {
        if (machinePara_.linkMode != LinkMode::LINK_SIMPLEX_MODE
            || machinePara_.machineType == MachineType::MACHINE_SERVER_TYPE) {
            std::vector<u8> data(NOTIFY_INFO_LENGTH, 0);
            CHK_SAFETY_FUNC_RET(memcpy_s(&data[0], data.size(), exchangeDataPtr, NOTIFY_INFO_LENGTH));
            exchangeDataPtr += NOTIFY_INFO_LENGTH;
            exchangeDataBlankSize -= NOTIFY_INFO_LENGTH;
            CHK_RET(OpenRemoteNotify(data, remoteSendReadyNotify_));

            HcclSignalInfo notifyInfo;
            CHK_RET(remoteSendReadyNotify_->GetNotifyData(notifyInfo));
            CHK_RET(remoteSendReadyNotify_->GetNotifyOffset(remoteSendReadyOffset_));

            remoteSendReadyAddress_ = notifyInfo.addr;
        }

        if (machinePara_.linkMode != LinkMode::LINK_SIMPLEX_MODE
            || machinePara_.machineType == MachineType::MACHINE_CLIENT_TYPE) {
            std::vector<u8> data(NOTIFY_INFO_LENGTH, 0);
            CHK_SAFETY_FUNC_RET(memcpy_s(&data[0], data.size(), exchangeDataPtr, NOTIFY_INFO_LENGTH));
            exchangeDataPtr += NOTIFY_INFO_LENGTH;
            exchangeDataBlankSize -= NOTIFY_INFO_LENGTH;
            CHK_RET(OpenRemoteNotify(data, remoteSendDoneNotify_));
            HcclSignalInfo notifyInfo;
            CHK_RET(remoteSendDoneNotify_->GetNotifyData(notifyInfo));
            CHK_RET(remoteSendDoneNotify_->GetNotifyOffset(remoteSendDoneOffset_));

            remoteSendDoneAddress_ = notifyInfo.addr;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult TransportP2p::ParseNotifyInfoEx(u8*& exchangeDataPtr, u64& exchangeDataBlankSize)
{
    if (machinePara_.isNewOneSide) {
        return HCCL_SUCCESS;
    }
    return ParseNotifyInfo(exchangeDataPtr, exchangeDataBlankSize);
}

HcclResult TransportP2p::ParseNotifyVectorInfo(u8*& exchangeDataPtr, u64& exchangeDataBlankSize)
{
    if (machinePara_.isNewOneSide) {
        return HCCL_SUCCESS;
    }
    for (u32 i = 0; i < notifyNum_; i++) {
        std::vector<u8> data(NOTIFY_INFO_LENGTH, 0);
        CHK_SAFETY_FUNC_RET(memcpy_s(&data[0], data.size(), exchangeDataPtr, NOTIFY_INFO_LENGTH));
        exchangeDataPtr += NOTIFY_INFO_LENGTH;
        exchangeDataBlankSize -= NOTIFY_INFO_LENGTH;
        CHK_RET(OpenRemoteNotify(data, userRemoteNotify_[i]));

        if (!machinePara_.isAicpuModeEn) {
            HcclSignalInfo notifyInfo;
            CHK_RET(userRemoteNotify_[i]->GetNotifyData(notifyInfo));
            CHK_RET(userRemoteNotify_[i]->GetNotifyOffset(userRemoteNotifyOffset_[i]));
            userRemoteNotifyAddr_[i] = notifyInfo.addr;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult
TransportP2p::ParseCheckDataLen(ExchangeInfoSize& remoteInfoSize, u8*& exchangeDataPtr, u64& exchangeDataBlankSize)
{
    CHK_SAFETY_FUNC_RET(memcpy_s(&remoteInfoSize, sizeof(remoteInfoSize), exchangeDataPtr, sizeof(ExchangeInfoSize)));
    exchangeDataPtr += sizeof(ExchangeInfoSize);
    exchangeDataBlankSize -= sizeof(ExchangeInfoSize);
    return HCCL_SUCCESS;
}

HcclResult TransportP2p::ConstructNotifyInfoForSend(u8*& exchangeDataPtr, u64& exchangeDataBlankSize)
{
    if (machinePara_.isNewOneSide) {
        return HCCL_SUCCESS;
    }
    if (machinePara_.isAicpuModeEn) {
        if ((machinePara_.linkMode != LinkMode::LINK_SIMPLEX_MODE
             || machinePara_.machineType == MachineType::MACHINE_CLIENT_TYPE)) {
            RemoteRankInfo info(machinePara_.remoteDeviceId, machinePara_.remoteWorldRank, recvPid_, recvSdid_);
            CHK_RET(
                notifyPool_->Alloc(machinePara_.tag, info, localSendReadyDeviceNotify_, NotifyLoadType::DEVICE_NOTIFY));
            std::vector<u8> data(NOTIFY_INFO_LENGTH, 0);
            CHK_RET(localSendReadyDeviceNotify_->Serialize(data));
            CHK_SAFETY_FUNC_RET(memcpy_s(exchangeDataPtr, exchangeDataBlankSize, &data[0], data.size()));
            exchangeDataPtr += data.size();
            exchangeDataBlankSize -= data.size();
        }

        if ((machinePara_.linkMode != LinkMode::LINK_SIMPLEX_MODE
             || machinePara_.machineType == MachineType::MACHINE_SERVER_TYPE)) {
            RemoteRankInfo info(machinePara_.remoteDeviceId, machinePara_.remoteWorldRank, recvPid_, recvSdid_);
            CHK_RET(
                notifyPool_->Alloc(machinePara_.tag, info, localSendDoneDeviceNotify_, NotifyLoadType::DEVICE_NOTIFY));
            std::vector<u8> data(NOTIFY_INFO_LENGTH, 0);
            CHK_RET(localSendDoneDeviceNotify_->Serialize(data));
            CHK_SAFETY_FUNC_RET(memcpy_s(exchangeDataPtr, exchangeDataBlankSize, &data[0], data.size()));
            exchangeDataPtr += data.size();
            exchangeDataBlankSize -= data.size();
        }
    } else {
        if (machinePara_.linkMode != LinkMode::LINK_SIMPLEX_MODE
            || machinePara_.machineType == MachineType::MACHINE_CLIENT_TYPE) {
            RemoteRankInfo info(machinePara_.remoteDeviceId, machinePara_.remoteWorldRank, recvPid_, recvSdid_);
            CHK_RET(notifyPool_->Alloc(machinePara_.tag, info, localSendReadyNotify_));
            std::vector<u8> data(NOTIFY_INFO_LENGTH, 0);
            CHK_RET(localSendReadyNotify_->Serialize(data));
            CHK_SAFETY_FUNC_RET(memcpy_s(exchangeDataPtr, exchangeDataBlankSize, &data[0], data.size()));
            exchangeDataPtr += data.size();
            exchangeDataBlankSize -= data.size();
        }

        if (machinePara_.linkMode != LinkMode::LINK_SIMPLEX_MODE
            || machinePara_.machineType == MachineType::MACHINE_SERVER_TYPE) {
            RemoteRankInfo info(machinePara_.remoteDeviceId, machinePara_.remoteWorldRank, recvPid_, recvSdid_);
            CHK_RET(notifyPool_->Alloc(machinePara_.tag, info, localSendDoneNotify_));
            std::vector<u8> data(NOTIFY_INFO_LENGTH, 0);
            CHK_RET(localSendDoneNotify_->Serialize(data));
            CHK_SAFETY_FUNC_RET(memcpy_s(exchangeDataPtr, exchangeDataBlankSize, &data[0], data.size()));
            exchangeDataPtr += data.size();
            exchangeDataBlankSize -= data.size();
        }
    }
    return HCCL_SUCCESS;
}

HcclResult TransportP2p::ConstructNotifyVectorInfoForSend(u8*& exchangeDataPtr, u64& exchangeDataBlankSize)
{
    if (machinePara_.isNewOneSide) {
        return HCCL_SUCCESS;
    }
    NotifyLoadType notifyLoadType
        = machinePara_.isAicpuModeEn ? NotifyLoadType::DEVICE_NOTIFY : NotifyLoadType::HOST_NOTIFY;
    for (u32 i = 0; i < notifyNum_; i++) {
        RemoteRankInfo info(machinePara_.remoteDeviceId, machinePara_.remoteWorldRank, recvPid_, recvSdid_);
        CHK_RET(notifyPool_->Alloc(machinePara_.tag, info, userLocalNotify_[i], notifyLoadType));
        std::vector<u8> data(NOTIFY_INFO_LENGTH, 0);
        CHK_RET(userLocalNotify_[i]->Serialize(data));
        CHK_SAFETY_FUNC_RET(memcpy_s(exchangeDataPtr, exchangeDataBlankSize, &data[0], data.size()));
        exchangeDataPtr += data.size();
        exchangeDataBlankSize -= data.size();
    }
    return HCCL_SUCCESS;
}

HcclResult TransportP2p::ConstructDataLenForSend(u8*& exchangeDataPtr, u64& exchangeDataBlankSize)
{
    CHK_SAFETY_FUNC_RET(
        memcpy_s(exchangeDataPtr, exchangeDataBlankSize, &exchangeInfoSize_, sizeof(exchangeInfoSize_)));
    exchangeDataPtr += sizeof(exchangeInfoSize_);
    exchangeDataBlankSize -= sizeof(exchangeInfoSize_);
    return HCCL_SUCCESS;
}

HcclResult TransportP2p::ParseReceivedExchangeData()
{
    s32 sendPid = 0;
    CHK_RET(SalGetBareTgid(&sendPid)); // 当前进程id
    HCCL_INFO("ParseReceivedExchangeData, sendPid[%d], recvPid[%d]", sendPid, recvPid_);
    u8* exchangeDataPtr = exchangeDataForRecv_.data();
    u64 exchangeDataBlankSize = exchangeDataTotalSize_;
    ExchangeInfoSize remoteInfoSize;
    CHK_RET(ParseCheckDataLen(remoteInfoSize, exchangeDataPtr, exchangeDataBlankSize));
    if (!exchangeInfoSize_.compare(remoteInfoSize)) {
        HCCL_ERROR(
            "remoteExchangeDataSize check fail, localIpcMemSize[%u] localNotifySize[%u] localExDataSize[%u] "
            "remoteIpcMemSize[%u] remoteNotifySize[%u] remoteExDataSize[%u]",
            exchangeInfoSize_.ipcMenSize, exchangeInfoSize_.notifySize, exchangeInfoSize_.exDataSize,
            remoteInfoSize.ipcMenSize, remoteInfoSize.notifySize, remoteInfoSize.exDataSize);
        return HCCL_E_INTERNAL;
    }

    if (sendPid != recvPid_ || recvSdid_ != INVALID_INT) {
        for (u32 i = 0; i < remoteIpcMemPtrVector_.size(); ++i) {
            CHK_RET(ParseIpcMemInfo(
                &remoteIpcMemPtrVector_[i], remoteIpcMemSizeVector_[i], remoteIpcMemNameVector_[i].ipcName,
                remoteIpcMemOffsetValueVector_[i], exchangeDataPtr, exchangeDataBlankSize));
            HCCL_INFO(
                "[TransportP2p][ParseReceivedExchangeData]index[%d]: remoteIpcMemPtr:[%p], "
                "remoteIpcMemSize:[%llu]",
                i, remoteIpcMemPtrVector_[i], remoteIpcMemSizeVector_[i]);
        }
        if (!isMemInclude_) {
            CHK_RET(ParseIpcMemInfo(
                &remoteOutputPtr_, remoteOutputSize_, remoteOutputMemName_.ipcName, remoteOutputOffsetValue_,
                exchangeDataPtr, exchangeDataBlankSize));
            CHK_RET(ParseIpcMemInfo(
                &remoteInputPtr_, remoteInputSize_, remoteInputMemName_.ipcName, remoteInputOffsetValue_,
                exchangeDataPtr, exchangeDataBlankSize));
        } else {
            CHK_RET(ParseMemIncludeInfo(&remoteOutputPtr_, remoteOutputSize_, exchangeDataPtr, exchangeDataBlankSize));
            CHK_RET(ParseMemIncludeInfo(&remoteInputPtr_, remoteInputSize_, exchangeDataPtr, exchangeDataBlankSize));
        }
    } else {
        u64 memAddr;
        CHK_RET(ParseIntraProcMemInfo(&memAddr, &remoteOutputSize_, exchangeDataPtr, exchangeDataBlankSize));
        remoteOutputPtr_ = reinterpret_cast<void*>(memAddr);
        CHK_RET(ParseIntraProcMemInfo(&memAddr, &remoteInputSize_, exchangeDataPtr, exchangeDataBlankSize));
        remoteInputPtr_ = reinterpret_cast<void*>(memAddr);
        for (u32 i = 0; i < remoteIpcMemPtrVector_.size(); ++i) {
            CHK_RET(
                ParseIntraProcMemInfo(&memAddr, &remoteIpcMemSizeVector_[i], exchangeDataPtr, exchangeDataBlankSize));
            remoteIpcMemPtrVector_[i] = reinterpret_cast<void*>(memAddr);
            HCCL_INFO(
                "[TransportP2p][ParseReceivedExchangeData]index[%d]: remoteIpcMemPtr:[%p], "
                "remoteIpcMemSize:[%llu]",
                i, remoteIpcMemPtrVector_[i], remoteIpcMemSizeVector_[i]);
        }
    }
    // 将本端和远端的Mem都打印。
    HCCL_INFO(
        "[TransportP2p][ParseReceivedExchangeData]remoteOutputPtr_[%p], remoteOutputSize_[%llu], "
        "remoteInputPtr_[%p], remoteInputSize_[%llu]",
        remoteOutputPtr_, remoteOutputSize_, remoteInputPtr_, remoteInputSize_);

    CHK_RET(ParseNotifyInfoEx(exchangeDataPtr, exchangeDataBlankSize));
    CHK_RET(ParseNotifyVectorInfo(exchangeDataPtr, exchangeDataBlankSize));
    CHK_RET(ParseExchangeData(exchangeDataPtr, exchangeDataBlankSize));

    if (machinePara_.isIndOp) {
        if (sendPid != recvPid_ || recvSdid_ != INVALID_INT) {
            u64 deviceMemNum;
            CHK_RET(ParseMemNumInfo(deviceMemNum, exchangeDataPtr, exchangeDataBlankSize));
            remoteIndOpDeviceMemPtrVector_.resize(deviceMemNum);
            remoteIndOpDeviceMemSizeVector_.resize(deviceMemNum);
            remoteIndOpDeviceMemOffsetValueVector_.resize(deviceMemNum);
            remoteIndOpDeviceMemNameVector_.resize(deviceMemNum);
            for (u64 i = 0; i < deviceMemNum; ++i) {
                CHK_RET(ParseIpcMemInfo(
                    &remoteIndOpDeviceMemPtrVector_[i], remoteIndOpDeviceMemSizeVector_[i],
                    remoteIndOpDeviceMemNameVector_[i].ipcName, remoteIndOpDeviceMemOffsetValueVector_[i],
                    exchangeDataPtr, exchangeDataBlankSize));
                HCCL_INFO(
                    "[TransportP2p][ParseReceivedExchangeData]independent operator device mem index[%d]: "
                    "remoteIndOpDeviceMemPtr:[%p], remoteIndOpDeviceMemSize:[%llu]",
                    i, remoteIndOpDeviceMemPtrVector_[i], remoteIndOpDeviceMemSizeVector_[i]);
            }
            u64 hostMemNum;
            CHK_RET(ParseMemNumInfo(hostMemNum, exchangeDataPtr, exchangeDataBlankSize));
            remoteIndOpHostMemPtrVector_.resize(hostMemNum);
            remoteIndOpHostMemSizeVector_.resize(hostMemNum);
            remoteIndOpHostMemOffsetValueVector_.resize(hostMemNum);
            remoteIndOpHostMemNameVector_.resize(hostMemNum);
            for (u64 i = 0; i < hostMemNum; ++i) {
                CHK_RET(ParseIpcMemInfo(
                    &remoteIndOpHostMemPtrVector_[i], remoteIndOpHostMemSizeVector_[i],
                    remoteIndOpHostMemNameVector_[i].ipcName, remoteIndOpHostMemOffsetValueVector_[i], exchangeDataPtr,
                    exchangeDataBlankSize));
                HCCL_INFO(
                    "[TransportP2p][ParseReceivedExchangeData]independent operator host mem index[%d]: "
                    "remoteIndOpHostMemPtr:[%p], remoteIndOpHostMemSize:[%llu]",
                    i, remoteIndOpHostMemPtrVector_[i], remoteIndOpHostMemSizeVector_[i]);
            }
        } else {
            u64 deviceMemNum;
            u64 memAddr;
            CHK_RET(ParseMemNumInfo(deviceMemNum, exchangeDataPtr, exchangeDataBlankSize));
            remoteIndOpDeviceMemPtrVector_.resize(deviceMemNum);
            remoteIndOpDeviceMemSizeVector_.resize(deviceMemNum);
            for (u32 i = 0; i < deviceMemNum; ++i) {
                CHK_RET(ParseIntraProcMemInfo(
                    &memAddr, &remoteIndOpDeviceMemSizeVector_[i], exchangeDataPtr, exchangeDataBlankSize));
                remoteIndOpDeviceMemPtrVector_[i] = reinterpret_cast<void*>(memAddr);
                HCCL_INFO(
                    "[TransportP2p][ParseReceivedExchangeData]independent operator device mem index[%d]: "
                    "remoteIndOpDeviceMemPtr:[%p], remoteIndOpDeviceMemSize:[%llu]",
                    i, remoteIndOpDeviceMemPtrVector_[i], remoteIndOpDeviceMemSizeVector_[i]);
            }
            u64 hostMemNum;
            CHK_RET(ParseMemNumInfo(hostMemNum, exchangeDataPtr, exchangeDataBlankSize));
            remoteIndOpHostMemPtrVector_.resize(hostMemNum);
            remoteIndOpHostMemSizeVector_.resize(hostMemNum);
            for (u32 i = 0; i < hostMemNum; ++i) {
                CHK_RET(ParseIntraProcMemInfo(
                    &memAddr, &remoteIndOpHostMemSizeVector_[i], exchangeDataPtr, exchangeDataBlankSize));
                remoteIndOpHostMemPtrVector_[i] = reinterpret_cast<void*>(memAddr);
                HCCL_INFO(
                    "[TransportP2p][ParseReceivedExchangeData]independent operator host mem index[%d]: "
                    "remoteIndOpHostMemPtr:[%p], remoteIndOpHostMemSize:[%llu]",
                    i, remoteIndOpHostMemPtrVector_[i], remoteIndOpHostMemSizeVector_[i]);
            }
        }
    }

    if (exchangeDataBlankSize != 0) {
        HCCL_ERROR(
            "[TransportP2p][ParseReceivedExchangeData] failed to Parse exchange Data "
            "exchangeDataBlankSize[%llu]",
            exchangeDataBlankSize);
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS; // this function should not be called in normal process
}

HcclResult TransportP2p::SignalRecord(
    std::shared_ptr<RemoteNotify>& remoteSignal, u64 remoteSignalAddr, u64 remoteSignalOffset, Stream& stream)
{
    return dispatcher_->SignalRecord(
        remoteSignal->ptr(), stream, machinePara_.remoteWorldRank, remoteSignalOffset, INVALID_VALUE_STAGE, false,
        remoteSignalAddr);
}

HcclResult TransportP2p::TxDataSignal(Stream& stream)
{
    HcclResult ret;
    /* 发起send_ready_event事件 */
    ret = SignalRecord(remoteSendReadyNotify_, remoteSendReadyAddress_, remoteSendReadyOffset_, stream);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[TransportP2p][TxDataSignal]errNo[0x%016llx] In tx data signal, signal record failed.",
            HCCL_ERROR_CODE(ret)),
        ret);
    return HCCL_SUCCESS;
}

HcclResult TransportP2p::RxDataSignal(Stream& stream)
{
    /* 等待send_ready_event事件 */
    CHK_RET(dispatcher_->SignalWait(
        localSendReadyNotify_->ptr(), stream, machinePara_.localUserrank, machinePara_.remoteWorldRank,
        INVALID_VALUE_STAGE, false, localSendReadyNotify_->notifyId_));
    return HCCL_SUCCESS;
}

HcclResult TransportP2p::TxAck(Stream& stream)
{
    /* 发起send_done_signal事件 */
    CHK_RET(SignalRecord(remoteSendDoneNotify_, remoteSendDoneAddress_, remoteSendDoneOffset_, stream));
    return HCCL_SUCCESS;
}

HcclResult TransportP2p::RxAck(Stream& stream)
{
    /* 等待send_done_signal事件 */
    CHK_RET(dispatcher_->SignalWait(
        localSendDoneNotify_->ptr(), stream, machinePara_.localUserrank, machinePara_.remoteWorldRank,
        INVALID_VALUE_STAGE, false, localSendDoneNotify_->notifyId_));
    return HCCL_SUCCESS;
}

HcclResult TransportP2p::TxPrepare(Stream& stream)
{
    CHK_RET(TxAck(stream));

    return HCCL_SUCCESS;
}

HcclResult TransportP2p::RxPrepare(Stream& stream)
{
    CHK_RET(RxAck(stream));

    return HCCL_SUCCESS;
}

HcclResult TransportP2p::TxDone(Stream& stream)
{
    HcclResult ret = RxDataSignal(stream);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[TransportP2p][TxDone]RxDataSignal failed"), ret);
    return HCCL_SUCCESS;
}

HcclResult TransportP2p::RxDone(Stream& stream)
{
    HcclResult ret = TxDataSignal(stream);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[TransportP2p][RxDone]TxDataSignal failed"), ret);
    return HCCL_SUCCESS;
}

HcclResult TransportP2p::Post(u32 notifyIdx, Stream& stream)
{
    // 校验notifyIdx有效性
    bool bRet = (notifyIdx >= notifyNum_);
    CHK_PRT_RET(
        bRet,
        HCCL_ERROR(
            "[TransportP2p][Post]notifyNum[%u], notifyIdx[%u] out of range[0, %u]", notifyNum_, notifyIdx,
            notifyNum_ - 1),
        HCCL_E_INTERNAL);

    // 发起send_done_signal事件
    CHK_RET(SignalRecord(
        userRemoteNotify_[notifyIdx], userRemoteNotifyAddr_[notifyIdx], userRemoteNotifyOffset_[notifyIdx], stream));
    return HCCL_SUCCESS;
}

HcclResult TransportP2p::Wait(u32 notifyIdx, Stream& stream, const u32 timeOut)
{
    // 校验notifyIdx有效性
    bool bRet = (notifyIdx >= notifyNum_);
    CHK_PRT_RET(
        bRet,
        HCCL_ERROR(
            "[TransportP2p][Wait]notifyNum[%u], notifyIdx[%u] out of range[0, %u]", notifyNum_, notifyIdx,
            notifyNum_ - 1),
        HCCL_E_INTERNAL);

    // 等待send_done_signal事件
    CHK_RET(dispatcher_->SignalWait(
        userLocalNotify_[notifyIdx]->ptr(), stream, machinePara_.localUserrank, machinePara_.remoteWorldRank,
        INVALID_VALUE_STAGE, false, userLocalNotify_[notifyIdx]->notifyId_, timeOut));
    return HCCL_SUCCESS;
}

HcclResult TransportP2p::ExchangeMemAndNotifyWithoutIpc()
{
    HcclResult ret;
    /* 发送 output 内存 */
    ret = SendMemMesgWithoutIpc(machinePara_.outputMem.ptr(), machinePara_.outputMem.size());
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[Exchange][IpcMesg] In exchange ipc mesg, send ipc mem output mesg fail. ret[%d], "
            "ptr[%p], size[%llu]",
            ret, machinePara_.outputMem.ptr(), machinePara_.outputMem.size()),
        ret);

    /* 发送 input 内存 */
    ret = SendMemMesgWithoutIpc(machinePara_.inputMem.ptr(), machinePara_.inputMem.size());
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[Exchange][IpcMesg] In exchange ipc mesg, send ipc mem input mesg fail. ret[%d], "
            "ptr[%p], size[%llu]",
            ret, machinePara_.inputMem.ptr(), machinePara_.inputMem.size()),
        ret);

    /* 发送 notify 信息 */
    CHK_RET(LinkSendNotifyMesg());

    /* 接收 output 内存 */
    u64 memAddr;
    ret = RecvMemMesgWithoutIpc(memAddr, remoteOutputMemName_.ipcName, remoteOutputOffsetValue_);
    remoteOutputPtr_ = reinterpret_cast<void*>(memAddr);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[Exchange][IpcMesg]In exchange ipc mesg, receive ipc output mem mesg fail. ret[%d], "
            "ptr[%p], memptr[%p], offset[%llu]",
            ret, remoteOutputPtr_, remoteOutputMemName_.ipcName, remoteOutputOffsetValue_),
        ret);

    /* 接收 input 内存 */
    ret = RecvMemMesgWithoutIpc(memAddr, remoteInputMemName_.ipcName, remoteInputOffsetValue_);
    remoteInputPtr_ = reinterpret_cast<void*>(memAddr);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[Exchange][IpcMesg]In exchange ipc mesg, receive ipc input mem mesg fail. ret[%d], "
            "ptr[%p], memptr[%p], offset[%llu]",
            ret, remoteInputPtr_, remoteInputMemName_.ipcName, remoteInputOffsetValue_),
        ret);

    /* 接收 notify 信息 */
    CHK_RET(LinkRecvNotifyMesg());
    return HCCL_SUCCESS;
}

HcclResult TransportP2p::ExchangeMemAndNotifyWithIpc()
{
    HcclResult ret;

    /* 发送IPC output 内存 */
    ret = SendIpcMemMesg(machinePara_.outputMem.ptr(), machinePara_.outputMem.size());
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[Exchange][IpcMesg] In exchange ipc mesg, send ipc mem output mesg fail. ret[%d], "
            "ptr[%p], size[%llu]",
            ret, machinePara_.outputMem.ptr(), machinePara_.outputMem.size()),
        ret);

    /* 发送IPC input 内存 */
    ret = SendIpcMemMesg(machinePara_.inputMem.ptr(), machinePara_.inputMem.size());
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[Exchange][IpcMesg] In exchange ipc mesg, send ipc mem input mesg fail. ret[%d], "
            "ptr[%p], size[%llu]",
            ret, machinePara_.inputMem.ptr(), machinePara_.inputMem.size()),
        ret);

    /* 发送IPC notify 信息 */
    CHK_RET(LinkSendNotifyMesg());

    /* 接收IPC output 内存 */
    ret = RecvIpcMemMesg(&remoteOutputPtr_, remoteOutputMemName_.ipcName, remoteOutputOffsetValue_);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[Exchange][IpcMesg]In exchange ipc mesg, receive ipc output mem mesg fail. ret[%d], "
            "ptr[%p], memptr[%p], offset[%llu]",
            ret, remoteOutputPtr_, remoteOutputMemName_.ipcName, remoteOutputOffsetValue_),
        ret);

    /* 接收IPC input 内存 */
    ret = RecvIpcMemMesg(&remoteInputPtr_, remoteInputMemName_.ipcName, remoteInputOffsetValue_);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[Exchange][IpcMesg]In exchange ipc mesg, receive ipc input mem mesg fail. ret[%d], "
            "ptr[%p], memptr[%p], offset[%llu]",
            ret, remoteInputPtr_, remoteInputMemName_.ipcName, remoteInputOffsetValue_),
        ret);

    /* 发送IPC notify 信息 */
    CHK_RET(LinkRecvNotifyMesg());
    return HCCL_SUCCESS;
}

HcclResult TransportP2p::ExchangeMemAndNotifyMesg()
{
    s32 sendPid = 0;
    CHK_RET(SalGetBareTgid(&sendPid)); // 当前进程id
    HCCL_INFO("ExchangeMemAndNotifyMesg, sendPid[%d], recvPid[%d]", sendPid, recvPid_);
    if (sendPid != recvPid_) {
        CHK_RET(ExchangeMemAndNotifyWithIpc()); // 跨进程时处于安全考虑，交换的是IPC Memory Name
    } else {
        CHK_RET(ExchangeMemAndNotifyWithoutIpc()); // 不跨进程时，仍然使用vnic来交换，直接交换VA，不需要转成Name
    }
    return HCCL_SUCCESS;
}

HcclResult TransportP2p::SendMemMesgWithoutIpc(void* ptr, u64 size) const
{
    HcclResult ret;
    /* send memaddr to remote rank */
    std::stringstream ss;
    ss << ptr;
    std::string memAddr = ss.str();
    ret = defaultSocket_->Send(memAddr);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[Send]errNo[0x%016llx], In send ipc mesg, send name failed.remote "
            "userrank[%u] local rank[%u]",
            HCCL_ERROR_CODE(ret), machinePara_.remoteUserrank, machinePara_.localUserrank),
        ret);

    /* send memsize to remote rank */
    std::string memSize = std::to_string(size);
    ret = defaultSocket_->Send(memSize);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[Send]errNo[0x%016llx] In send ipc mesg, send size failed. remote rank[%u] "
            "size[%s] local rank[%u]",
            HCCL_ERROR_CODE(ret), machinePara_.remoteUserrank, memSize.c_str(), machinePara_.localUserrank),
        ret);
    return HCCL_SUCCESS;
}

HcclResult TransportP2p::RecvMemMesgWithoutIpc(u64& addr, [[maybe_unused]] u8* memName, u64& offset)
{
    HcclResult ret;
    std::string memAddr;

    /* 获取对端地址 */
    ret = defaultSocket_->Recv(memAddr);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[Recv]errNo[0x%016llx] In recv ipc mem mesg, receive mem name failed."
            "remote userrank[%u] local rank[%u]",
            HCCL_ERROR_CODE(ret), machinePara_.remoteUserrank, machinePara_.localUserrank),
        ret);

    CHK_RET(SalStrToULonglong(memAddr, HCCL_BASE_HEX, addr));
    /* 获取对端内存的大小 */
    std::string remoteMemSize;
    u64 size = 0;
    ret = defaultSocket_->Recv(remoteMemSize);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[Recv]errNo[0x%016llx] In recv ipc mem mesg, receive offset name failed."
            "remote userrank[%u] local rank[%u], remoteMemSize[%s]",
            HCCL_ERROR_CODE(ret), machinePara_.remoteUserrank, machinePara_.localUserrank, remoteMemSize.c_str()),
        ret);

    CHK_RET(SalStrToULonglong(remoteMemSize, HCCL_BASE_DECIMAL, size));
    /* 获取对端内存的偏移值 */
    offset = 0;
    return ret;
}

HcclResult TransportP2p::SendIpcMemMesg(void* ptr, u64 size) const
{
    HcclResult ret;
    /* make memory shared interprocess and assigned a name */
    u64 offset;
    SecIpcName_t memName;
    ret = MemNameRepository::GetInstance(machinePara_.deviceLogicId)
              ->SetIpcMem(ptr, size, memName.ipcName, HCCL_IPC_MEM_NAME_LEN, offset, recvPid_, recvSdid_, isSioToHccs_);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[Send][IpcMemMesg]errNo[0x%016llx], In send ipc mesg, get para mem name failed. "
            "mem addr[%p] local rank[%u]",
            HCCL_ERROR_CODE(ret), ptr, machinePara_.localUserrank),
        ret);

    std::string memOffset = std::to_string(offset);
    /* send memName to remote rank */
    ret = defaultSocket_->Send(memName.ipcName, HCCL_IPC_MEM_NAME_LEN);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[Send][IpcMemMesg]errNo[0x%016llx], In send ipc mesg, send name failed.remote "
            "userrank[%u] local rank[%u]",
            HCCL_ERROR_CODE(ret), machinePara_.remoteUserrank, machinePara_.localUserrank),
        ret);
    HCCL_INFO(
        "localUserrank=%u, ptr=%p, remoteUserrank=%u, mem_offset=%s", machinePara_.localUserrank, ptr,
        machinePara_.remoteUserrank, memOffset.c_str());

    /* send memsize to remote rank */
    std::string memSize = std::to_string(size);
    ret = defaultSocket_->Send(memSize);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[Send][IpcMemMesg]errNo[0x%016llx] In send ipc mesg, send size failed. remote rank[%u] "
            "size[%s] local rank[%u]",
            HCCL_ERROR_CODE(ret), machinePara_.remoteUserrank, memSize.c_str(), machinePara_.localUserrank),
        ret);

    /* send memOffset to remote rank */
    ret = defaultSocket_->Send(memOffset);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[Send][IpcMemMesg]errNo[0x%016llx] In send ipc mesg, send offset failed. remote rank[%u] "
            "offset[%s] local rank[%u]",
            HCCL_ERROR_CODE(ret), machinePara_.remoteUserrank, memOffset.c_str(), machinePara_.localUserrank),
        ret);

    HCCL_DEBUG(
        "localUserrank=%u, ptr=%p, remoteUserrank=%u, offset=%s", machinePara_.localUserrank, ptr,
        machinePara_.remoteUserrank, memOffset.c_str());
    return HCCL_SUCCESS;
}

HcclResult TransportP2p::RecvIpcMemMesg(void** memPtr, u8* memName, u64& offset)
{
    HcclResult ret;
    /* 获取对端内存名字 */
    ret = defaultSocket_->Recv(memName, HCCL_IPC_MEM_NAME_LEN);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[Recv][IpcMemMesg]errNo[0x%016llx] In recv ipc mem mesg, receive mem name failed."
            "remote userrank[%u] local rank[%u]",
            HCCL_ERROR_CODE(ret), machinePara_.remoteUserrank, machinePara_.localUserrank),
        ret);
    /* 获取对端内存的大小 */
    std::string remoteMemSize;
    u64 size = 0;
    ret = defaultSocket_->Recv(remoteMemSize);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[Recv][IpcMemMesg]errNo[0x%016llx] In recv ipc mem mesg, receive offset name failed."
            "remote userrank[%u] local rank[%u], remoteMemSize[%s]",
            HCCL_ERROR_CODE(ret), machinePara_.remoteUserrank, machinePara_.localUserrank, remoteMemSize.c_str()),
        ret);

    CHK_RET(SalStrToULonglong(remoteMemSize, HCCL_BASE_DECIMAL, size));

    /* 获取对端内存的偏移值 */
    std::string remoteOffsetName;
    ret = defaultSocket_->Recv(remoteOffsetName);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[Recv][IpcMemMesg]errNo[0x%016llx] In recv ipc mem mesg, receive offset name failed."
            "remote userrank[%u] local rank[%u], remoteOffsetName[%s]",
            HCCL_ERROR_CODE(ret), machinePara_.remoteUserrank, machinePara_.localUserrank, remoteOffsetName.c_str()),
        ret);

    CHK_RET(SalStrToULonglong(remoteOffsetName, HCCL_BASE_DECIMAL, offset));

    /* 根据名字，获取对端IPC 内存 */
    ret = WaitPeerMemConfig(memPtr, const_cast<u8*>(memName), size, offset);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[Recv][IpcMemMesg]errNo[0x%016llx] In recv ipc mem mesg, wait peer mem config "
            "failed. local rank[%u]",
            HCCL_ERROR_CODE(ret), machinePara_.localUserrank),
        ret);

    HCCL_DEBUG(
        "localUserrank[%u] receive from remoteUserrank[%u]", machinePara_.localUserrank, machinePara_.remoteUserrank);

    return HCCL_SUCCESS;
}

HcclResult TransportP2p::TxAsync(UserMemType dstMemType, u64 dstOffset, const void* src, u64 len, Stream& stream)
{
    HcclResult ret;
    /* 源端发起数据传输 */
    if (((machinePara_.linkAttribute & 0x2) == 0) && (src != nullptr)) { // 不支持目的端发起
        void* dstMemPtr = nullptr;
        CHK_RET(GetRemoteMem(dstMemType, &dstMemPtr));

        DeviceMem dstDevMem(static_cast<s8*>(dstMemPtr) + dstOffset, len);
        DeviceMem srcDevMem(const_cast<void*>(src), len);
        /* 增加hccl 数据传输时数据地址和size记录 */
        HCCL_INFO(
            "HCCL_KEY_INFO: srcAddr=[%p],srcSize=[%llu],dstAddr=[%p],dstSize=[%llu]", srcDevMem.ptr(), srcDevMem.size(),
            dstDevMem.ptr(), dstDevMem.size());
        CHK_RET(HcclD2DMemcpyAsync(
            dispatcher_, dstDevMem, srcDevMem, stream, machinePara_.remoteWorldRank, transportAttr_.linkType));
    }

    /* 发起send_ready_signal事件 */
    ret = SignalRecord(remoteSendReadyNotify_, remoteSendReadyAddress_, remoteSendReadyOffset_, stream);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[TransportP2p][TxAsync]errNo[0x%016llx] In tx async, signal record failed.", HCCL_ERROR_CODE(ret)),
        ret);

    return HCCL_SUCCESS;
}

HcclResult TransportP2p::TxData(UserMemType dstMemType, u64 dstOffset, const void* src, u64 len, Stream& stream)
{
    /* 源端发起数据传输 */
    if (((machinePara_.linkAttribute & 0x2) == 0) && (src != nullptr)) { // 不支持目的端发起
        void* dstMemPtr = nullptr;
        CHK_RET(GetRemoteMem(dstMemType, &dstMemPtr));

        DeviceMem dstDevMem(static_cast<s8*>(dstMemPtr) + dstOffset, len);
        DeviceMem srcDevMem(const_cast<void*>(src), len);
        /* 增加hccl 数据传输时数据地址和size记录 */
        HCCL_INFO(
            "HCCL_KEY_INFO: srcAddr=[%p],srcSize=[%llu],dstAddr=[%p],dstSize=[%llu]", srcDevMem.ptr(), srcDevMem.size(),
            dstDevMem.ptr(), dstDevMem.size());
        CHK_RET(HcclD2DMemcpyAsync(
            dispatcher_, dstDevMem, srcDevMem, stream, machinePara_.remoteWorldRank, transportAttr_.linkType));
    }

    return HCCL_SUCCESS;
}

HcclResult TransportP2p::RxData(UserMemType srcMemType, u64 srcOffset, void* dst, u64 len, Stream& stream)
{
    /* 目的端发起数据传输 */
    if ((machinePara_.linkAttribute & 0x2) && (dst != nullptr)) { // 支持目的端发起
        void* srcMemPtr = nullptr;
        CHK_RET(GetRemoteMem(srcMemType, &srcMemPtr));

        DeviceMem srcDevMem(static_cast<s8*>(srcMemPtr) + srcOffset, len);
        DeviceMem dstDevMem(static_cast<s8*>(dst), len);
        CHK_RET(HcclD2DMemcpyAsync(
            dispatcher_, dstDevMem, srcDevMem, stream, machinePara_.remoteWorldRank, transportAttr_.linkType));
    }

    return HCCL_SUCCESS;
}

HcclResult TransportP2p::TxAsync(std::vector<TxMemoryInfo>& txMems, Stream& stream)
{
    HcclResult ret;
    /* 源端发起数据传输 */
    if ((machinePara_.linkAttribute & 0x2) == 0) { // 不支持目的端发起
        for (auto& mem : txMems) {
            CHK_PTR_NULL(mem.src);
            void* dstMemPtr = nullptr;
            CHK_RET(GetRemoteMem(mem.dstMemType, &dstMemPtr));

            DeviceMem dstDevMem(static_cast<s8*>(dstMemPtr) + mem.dstOffset, mem.len);
            DeviceMem srcDevMem(const_cast<void*>(mem.src), mem.len);
            /* 增加hccl 数据传输时数据地址和size记录 */
            HCCL_INFO(
                "HCCL_KEY_INFO: srcAddr=[%p],srcSize=[%llu],dstAddr=[%p],dstSize=[%llu]", srcDevMem.ptr(),
                srcDevMem.size(), dstDevMem.ptr(), dstDevMem.size());
            CHK_RET(HcclD2DMemcpyAsync(
                dispatcher_, dstDevMem, srcDevMem, stream, machinePara_.remoteWorldRank, transportAttr_.linkType));
        }
    }

    /* 发起send_ready_signal事件 */
    ret = SignalRecord(remoteSendReadyNotify_, remoteSendReadyAddress_, remoteSendReadyOffset_, stream);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[TransportP2p][TxAsync]errNo[0x%016llx] In tx async, signal record failed.", HCCL_ERROR_CODE(ret)),
        ret);

    return HCCL_SUCCESS;
}

HcclResult TransportP2p::RxAsync(UserMemType srcMemType, u64 srcOffset, void* dst, u64 len, Stream& stream)
{
    /* 等待send_ready_signal事件 */
    CHK_RET(dispatcher_->SignalWait(
        localSendReadyNotify_->ptr(), stream, machinePara_.localUserrank, machinePara_.remoteWorldRank,
        INVALID_VALUE_STAGE, false, localSendReadyNotify_->notifyId_));

    /* 目的端发起数据传输 */
    if ((machinePara_.linkAttribute & 0x2) && (dst != nullptr)) { // 支持目的端发起
        void* srcMemPtr = nullptr;
        CHK_RET(GetRemoteMem(srcMemType, &srcMemPtr));

        DeviceMem srcDevMem(static_cast<s8*>(srcMemPtr) + srcOffset, len);
        DeviceMem dstDevMem(static_cast<s8*>(dst), len);
        CHK_RET(HcclD2DMemcpyAsync(
            dispatcher_, dstDevMem, srcDevMem, stream, machinePara_.remoteWorldRank, transportAttr_.linkType));
    }

    return HCCL_SUCCESS;
}

HcclResult TransportP2p::RxAsync(std::vector<RxMemoryInfo>& rxMems, Stream& stream)
{
    /* 等待send_ready_signal事件 */
    CHK_RET(dispatcher_->SignalWait(
        localSendReadyNotify_->ptr(), stream, machinePara_.localUserrank, machinePara_.remoteWorldRank,
        INVALID_VALUE_STAGE, false, localSendReadyNotify_->notifyId_));

    /* 目的端发起数据传输 */
    if ((machinePara_.linkAttribute & 0x2) != 0) { // 支持目的端发起
        for (auto& mem : rxMems) {
            CHK_PTR_NULL(mem.dst);
            void* srcMemPtr = nullptr;
            CHK_RET(GetRemoteMem(mem.srcMemType, &srcMemPtr));

            DeviceMem srcDevMem(static_cast<s8*>(srcMemPtr) + mem.srcOffset, mem.len);
            DeviceMem dstDevMem(static_cast<s8*>(mem.dst), mem.len);
            CHK_RET(HcclD2DMemcpyAsync(
                dispatcher_, dstDevMem, srcDevMem, stream, machinePara_.remoteWorldRank, transportAttr_.linkType));
        }
    }

    return HCCL_SUCCESS;
}

HcclResult TransportP2p::DataReceivedAck(Stream& stream)
{
    CHK_RET(TxAck(stream));
    CHK_RET(RxAck(stream));
    CHK_RET(TxDataSignal(stream));
    CHK_RET(RxDataSignal(stream));

    return HCCL_SUCCESS;
}

HcclResult TransportP2p::GetLocalNotify(std::vector<HcclSignalInfo>& localNotify)
{
    if (machinePara_.isNewOneSide) {
        return HCCL_SUCCESS;
    }
    HcclSignalInfo notifyInfo;

    if ((machinePara_.linkMode != LinkMode::LINK_SIMPLEX_MODE
         || machinePara_.machineType == MachineType::MACHINE_CLIENT_TYPE)) {
        if (machinePara_.isAicpuModeEn) {
            CHK_SMART_PTR_NULL(localSendReadyDeviceNotify_);
            CHK_RET(localSendReadyDeviceNotify_->GetNotifyData(notifyInfo));
            localNotify.push_back(notifyInfo);
        } else {
            CHK_SMART_PTR_NULL(localSendReadyNotify_);
            CHK_RET(localSendReadyNotify_->GetNotifyData(notifyInfo));
            localNotify.push_back(notifyInfo);
        }
    }

    if (machinePara_.linkMode != LinkMode::LINK_SIMPLEX_MODE
        || machinePara_.machineType == MachineType::MACHINE_SERVER_TYPE) {
        if (machinePara_.isAicpuModeEn) {
            CHK_SMART_PTR_NULL(localSendDoneDeviceNotify_);
            CHK_RET(localSendDoneDeviceNotify_->GetNotifyData(notifyInfo));
            localNotify.push_back(notifyInfo);
        } else {
            CHK_SMART_PTR_NULL(localSendDoneNotify_);
            CHK_RET(localSendDoneNotify_->GetNotifyData(notifyInfo));
            localNotify.push_back(notifyInfo);
        }
    }

    bool bRet = !(notifyNum_ == userLocalNotify_.size());
    CHK_PRT_RET(
        bRet,
        HCCL_ERROR(
            "[TransportP2p][GetLocalNotify]size of userLocalNotify_ doesn't equal to notifyNum_[%u]", notifyNum_),
        HCCL_E_INTERNAL);

    // 提取新增的notify资源
    for (u32 i = 0; i < notifyNum_; i++) {
        CHK_SMART_PTR_NULL(userLocalNotify_[i]);
        CHK_RET(userLocalNotify_[i]->GetNotifyData(notifyInfo));
        localNotify.push_back(notifyInfo);
    }
    return HCCL_SUCCESS;
}

HcclResult TransportP2p::GetRemoteNotify(std::vector<HcclSignalInfo>& localNotify)
{
    if (machinePara_.isNewOneSide) {
        return HCCL_SUCCESS;
    }
    HcclSignalInfo notifyInfo;
    if ((machinePara_.linkMode != LinkMode::LINK_SIMPLEX_MODE
         || machinePara_.machineType == MachineType::MACHINE_SERVER_TYPE)) {
        if (machinePara_.isAicpuModeEn) {
            CHK_SMART_PTR_NULL(remoteSendReadyDeviceNotify_);
            CHK_RET(remoteSendReadyDeviceNotify_->GetNotifyData(notifyInfo));
            localNotify.push_back(notifyInfo);
        } else {
            CHK_SMART_PTR_NULL(remoteSendReadyNotify_);
            CHK_RET(remoteSendReadyNotify_->GetNotifyData(notifyInfo));
            localNotify.push_back(notifyInfo);
        }
    }

    if (machinePara_.linkMode != LinkMode::LINK_SIMPLEX_MODE
        || machinePara_.machineType == MachineType::MACHINE_CLIENT_TYPE) {
        if (machinePara_.isAicpuModeEn) {
            CHK_SMART_PTR_NULL(remoteSendDoneDeviceNotify_);
            CHK_RET(remoteSendDoneDeviceNotify_->GetNotifyData(notifyInfo));
            localNotify.push_back(notifyInfo);
        } else {
            CHK_SMART_PTR_NULL(remoteSendDoneNotify_);
            CHK_RET(remoteSendDoneNotify_->GetNotifyData(notifyInfo));
            localNotify.push_back(notifyInfo);
        }
    }

    bool bRet = !(notifyNum_ == userRemoteNotify_.size());
    CHK_PRT_RET(
        bRet,
        HCCL_ERROR(
            "[TransportP2p][GetRemoteNotify]size of userRemoteNotify_ doesn't equal to notifyNum_[%u]", notifyNum_),
        HCCL_E_INTERNAL);

    // 新增notify的提取
    for (u32 i = 0; i < notifyNum_; i++) {
        CHK_SMART_PTR_NULL(userRemoteNotify_[i]);
        CHK_RET(userRemoteNotify_[i]->GetNotifyData(notifyInfo));
        localNotify.push_back(notifyInfo);
    }
    return HCCL_SUCCESS;
}

HcclResult TransportP2p::GetIndOpRemoteMem(HcclMem** remoteMem, uint32_t* memNum)
{
    CHK_PRT_RET(remoteMem == nullptr, HCCL_ERROR("[%s] remoteMem is nullptr", __func__), HCCL_E_PARA);
    CHK_PRT_RET(memNum == nullptr, HCCL_ERROR("[%s] memNum is nullptr", __func__), HCCL_E_PARA);

    *remoteMem = nullptr;
    *memNum = 0;
    uint32_t totalCount = remoteIndOpHostMemPtrVector_.size() + remoteIndOpDeviceMemPtrVector_.size();
    if (totalCount == 0) {
        HCCL_DEBUG("[%s] No remote memory regions available", __func__);
        return HCCL_SUCCESS;
    }
    // 检查向量大小是否匹配
    if (remoteIndOpHostMemPtrVector_.size() != remoteIndOpHostMemSizeVector_.size()
        || remoteIndOpDeviceMemPtrVector_.size() != remoteIndOpDeviceMemSizeVector_.size()) {
        HCCL_ERROR("[%s] Memory pointer and size vectors size mismatch", __func__);
        return HCCL_E_INTERNAL;
    }
    // 外部需要手动释放内存
    HcclMem* resultArray = static_cast<HcclMem*>(malloc(totalCount * sizeof(HcclMem)));
    CHK_PTR_NULL(resultArray);
    uint32_t index = 0;
    for (size_t i = 0; i < remoteIndOpDeviceMemPtrVector_.size(); ++i) {
        resultArray[index].type = HcclMemType::HCCL_MEM_TYPE_DEVICE;
        resultArray[index].addr = remoteIndOpDeviceMemPtrVector_[i];
        resultArray[index].size = remoteIndOpDeviceMemSizeVector_[i];
        index++;
    }
    for (size_t i = 0; i < remoteIndOpHostMemPtrVector_.size(); ++i) {
        resultArray[index].type = HcclMemType::HCCL_MEM_TYPE_HOST;
        resultArray[index].addr = remoteIndOpHostMemPtrVector_[i];
        resultArray[index].size = remoteIndOpHostMemSizeVector_[i];
        index++;
    }
    *remoteMem = resultArray;
    *memNum = index;

    HCCL_DEBUG("[%s] Successfully returned %u remote memory regions", __func__, index);

    return HCCL_SUCCESS;
}

HcclResult TransportP2p::GetRemoteMem(UserMemType memType, void** remotePtr)
{
    switch (memType) {
        case UserMemType::INPUT_MEM: {
            *remotePtr = remoteInputPtr_;
            break;
        }

        case UserMemType::OUTPUT_MEM: {
            *remotePtr = remoteOutputPtr_;
            break;
        }

        default: {
            HCCL_ERROR("[Get][RemoteMem]not support dst_mem_type=%d", memType);
            return HCCL_E_NOT_SUPPORT;
        }
    }

    return HCCL_SUCCESS;
}

HcclResult TransportP2p::GetRemoteMem(std::vector<void*>* remotePtr)
{
    *remotePtr = remoteIpcMemPtrVector_;
    return HCCL_SUCCESS;
}

HcclResult TransportP2p::GetRemoteMemSize(UserMemType memType, u64& size)
{
    switch (memType) {
        case UserMemType::INPUT_MEM: {
            size = remoteInputSize_;
            break;
        }

        case UserMemType::OUTPUT_MEM: {
            size = remoteOutputSize_;
            break;
        }

        default: {
            HCCL_ERROR("[Get][RemoteMem]not support dst_mem_type=%d", memType);
            return HCCL_E_NOT_SUPPORT;
        }
    }

    return HCCL_SUCCESS;
}

HcclResult TransportP2p::WaitPeerMemConfig(void** memPtr, const u8* memName, uint64_t size, u64 offset)
{
    CHK_PTR_NULL(memPtr);
    CHK_PTR_NULL(memName);

    bool firstOpened = false;
    // 支持进程间、进程内都可以通过name获取对端内存
    HcclResult ret = MemNameRepository::GetInstance(machinePara_.deviceLogicId)
                         ->OpenIpcMem(memPtr, size, memName, HCCL_IPC_MEM_NAME_LEN, offset, firstOpened, isSioToHccs_);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[Wait][WaitPeerMemConfig]errNo[0x%016llx] In link pcie, open mem failed. "
            "offset[%llu], size[%llu Byte], linkType[%d]",
            HCCL_ERROR_CODE(ret), offset, size, transportAttr_.linkType),
        ret);
    return HCCL_SUCCESS;
}

HcclResult TransportP2p::PostReady(Stream& stream)
{
    CHK_RET(SignalRecord(remoteSendReadyNotify_, remoteSendReadyAddress_, remoteSendReadyOffset_, stream));
    return HCCL_SUCCESS;
}

HcclResult TransportP2p::WaitReady(Stream& stream)
{
    CHK_RET(dispatcher_->SignalWait(
        localSendReadyNotify_->ptr(), stream, machinePara_.localUserrank, machinePara_.remoteWorldRank,
        INVALID_VALUE_STAGE, false, localSendReadyNotify_->notifyId_));
    return HCCL_SUCCESS;
}

HcclResult TransportP2p::PostFin(Stream& stream)
{
    CHK_RET(SignalRecord(remoteSendDoneNotify_, remoteSendDoneAddress_, remoteSendDoneOffset_, stream));
    return HCCL_SUCCESS;
}

HcclResult TransportP2p::WaitFin(Stream& stream)
{
    CHK_RET(dispatcher_->SignalWait(
        localSendDoneNotify_->ptr(), stream, machinePara_.localUserrank, machinePara_.remoteWorldRank,
        INVALID_VALUE_STAGE, false, localSendDoneNotify_->notifyId_));
    return HCCL_SUCCESS;
}

HcclResult
TransportP2p::WriteSync(struct Transport::Buffer& remoteBuf, struct Transport::Buffer& localBuf, Stream& stream)
{
    DeviceMem remoteDevMem(const_cast<void*>(remoteBuf.addr), remoteBuf.size);
    DeviceMem localDevMem(const_cast<void*>(localBuf.addr), localBuf.size);
    HCCL_INFO(
        "HCCL_KEY_INFO: localAddr=[%p],localSize=[%llu],remoteAddr=[%p],remoteSize=[%llu]", localDevMem.ptr(),
        localDevMem.size(), remoteDevMem.ptr(), remoteDevMem.size());
    CHK_RET(HcclD2DMemcpyAsync(
        dispatcher_, remoteDevMem, localDevMem, stream, machinePara_.remoteWorldRank, transportAttr_.linkType));
    return HCCL_SUCCESS;
}

HcclResult
TransportP2p::WriteAsyncEx(struct Transport::Buffer& remoteBuf, struct Transport::Buffer& localBuf, Stream& stream)
{
    bool isLocalHostAddr = false;
    bool isRemoteHostAddr = false;
    struct Transport::Buffer newLocalBuf {};
    struct Transport::Buffer newRemoteBuf {};
    CHK_RET(ReplaceMemAddr(localBuf, remoteBuf, newLocalBuf, newRemoteBuf, isLocalHostAddr, isRemoteHostAddr));
    DeviceMem dstDevMem(const_cast<void*>(newRemoteBuf.addr), newRemoteBuf.size);
    DeviceMem srcDevMem(const_cast<void*>(newLocalBuf.addr), newLocalBuf.size);
    CHK_RET(reinterpret_cast<DispatcherPub*>(dispatcher_)
                ->MemcpyAsync(dstDevMem, srcDevMem, stream, machinePara_.remoteWorldRank, transportAttr_.linkType));
    return HCCL_SUCCESS;
}

HcclResult
TransportP2p::WriteAsync(struct Transport::Buffer& remoteBuf, struct Transport::Buffer& localBuf, Stream& stream)
{
    if (machinePara_.isNewOneSide) {
        return WriteAsyncEx(remoteBuf, localBuf, stream);
    }

    DeviceMem remoteDevMem(const_cast<void*>(remoteBuf.addr), remoteBuf.size);
    DeviceMem localDevMem(const_cast<void*>(localBuf.addr), localBuf.size);
    HCCL_INFO(
        "HCCL_KEY_INFO: localAddr=[%p],localSize=[%llu],remoteAddr=[%p],remoteSize=[%llu]", localDevMem.ptr(),
        localDevMem.size(), remoteDevMem.ptr(), remoteDevMem.size());
    CHK_RET(HcclD2DMemcpyAsync(
        dispatcher_, remoteDevMem, localDevMem, stream, machinePara_.remoteWorldRank, transportAttr_.linkType));
    return HCCL_SUCCESS;
}

HcclResult TransportP2p::WriteReduceAsync(
    struct Transport::Buffer& remoteBuf, struct Transport::Buffer& localBuf, const HcclDataType datatype,
    HcclReduceOp redOp, Stream& stream)
{
    HCCL_INFO(
        "HCCL_KEY_INFO: localAddr=[%p],localSize=[%llu],remoteAddr=[%p],remoteSize=[%llu]", localBuf.addr,
        localBuf.size, remoteBuf.addr, remoteBuf.size);

    u64 reduceAttr = 0;
    if (IsSpInlineReduce()) {
        reduceAttr = INLINE_REDUCE_BIT;
    }
    CHK_RET(HcclReduceAsync(
        dispatcher_, const_cast<void*>(localBuf.addr), remoteBuf.size / SIZE_TABLE[datatype], datatype, redOp, stream,
        const_cast<void*>(remoteBuf.addr), GetRemoteRank(), GetLinkType(), reduceAttr));
    return HCCL_SUCCESS;
}

HcclResult
TransportP2p::ReadSync(struct Transport::Buffer& localBuf, struct Transport::Buffer& remoteBuf, Stream& stream)
{
    DeviceMem remoteDevMem(const_cast<void*>(remoteBuf.addr), remoteBuf.size);
    DeviceMem localDevMem(const_cast<void*>(localBuf.addr), localBuf.size);
    HCCL_INFO(
        "HCCL_KEY_INFO: localAddr=[%p],localSize=[%llu],remoteAddr=[%p],remoteSize=[%llu]", localDevMem.ptr(),
        localDevMem.size(), remoteDevMem.ptr(), remoteDevMem.size());
    CHK_RET(HcclD2DMemcpyAsync(
        dispatcher_, localDevMem, remoteDevMem, stream, machinePara_.remoteWorldRank, transportAttr_.linkType));
    return HCCL_SUCCESS;
}

HcclResult TransportP2p::ReadReduceSync(
    struct Transport::Buffer& localBuf, struct Transport::Buffer& remoteBuf, const HcclDataType datatype,
    HcclReduceOp redOp, Stream& stream)
{
    HCCL_INFO(
        "HCCL_KEY_INFO: localAddr=[%p],localSize=[%llu],remoteAddr=[%p],remoteSize=[%llu]", localBuf.addr,
        localBuf.size, remoteBuf.addr, remoteBuf.size);

    u64 reduceAttr = 0;
    if (IsSpInlineReduce()) {
        reduceAttr = INLINE_REDUCE_BIT;
    }
    CHK_RET(HcclReduceAsync(
        dispatcher_, const_cast<void*>(remoteBuf.addr), remoteBuf.size / SIZE_TABLE[datatype], datatype, redOp, stream,
        const_cast<void*>(localBuf.addr), GetRemoteRank(), GetLinkType(), reduceAttr));
    return HCCL_SUCCESS;
}

HcclResult
TransportP2p::ReadAsyncEx(struct Transport::Buffer& localBuf, struct Transport::Buffer& remoteBuf, Stream& stream)
{
    bool isLocalHostAddr = false;
    bool isRemoteHostAddr = false;
    struct Transport::Buffer newLocalBuf {};
    struct Transport::Buffer newRemoteBuf {};
    CHK_RET(ReplaceMemAddr(localBuf, remoteBuf, newLocalBuf, newRemoteBuf, isLocalHostAddr, isRemoteHostAddr));
    DeviceMem dstDevMem(const_cast<void*>(newLocalBuf.addr), newLocalBuf.size);
    DeviceMem srcDevMem(const_cast<void*>(newRemoteBuf.addr), newRemoteBuf.size);
    CHK_RET(reinterpret_cast<DispatcherPub*>(dispatcher_)
                ->MemcpyAsync(dstDevMem, srcDevMem, stream, machinePara_.remoteWorldRank, transportAttr_.linkType));
    return HCCL_SUCCESS;
}

HcclResult
TransportP2p::ReadAsync(struct Transport::Buffer& localBuf, struct Transport::Buffer& remoteBuf, Stream& stream)
{
    if (machinePara_.isNewOneSide) {
        return ReadAsyncEx(localBuf, remoteBuf, stream);
    }
    DeviceMem dstDevMem(const_cast<void*>(localBuf.addr), localBuf.size);
    DeviceMem srcDevMem(const_cast<void*>(remoteBuf.addr), remoteBuf.size);
    return HcclD2DMemcpyAsync(
        dispatcher_, dstDevMem, srcDevMem, stream, machinePara_.remoteWorldRank, transportAttr_.linkType);
}

HcclResult TransportP2p::SumCheckSizeAndConsistent(
    ExInfoType exInfoType, u32 rightInfoSize, u64& blankSizeRecord, u64 exchangeDataBlankSize)
{
    u32 checkInfoSize = blankSizeRecord - exchangeDataBlankSize;
    if (checkInfoSize != rightInfoSize) {
        HCCL_ERROR(
            "[SumCheckSizeAndConsistent] ExInfoType[%d] check size failed, checkInfoSize[%u] rightInfoSize[%u]",
            exInfoType, checkInfoSize, rightInfoSize);
        return HCCL_E_INTERNAL;
    }
    blankSizeRecord = exchangeDataBlankSize;
    return HCCL_SUCCESS;
}

HcclResult TransportP2p::ConstructMemIncludeInfoForSend(u8*& exchangeDataPtr, u64& exchangeDataBlankSize)
{
    u64 outputSize = machinePara_.outputMem.size();
    u64 outputOffset
        = reinterpret_cast<u64>(machinePara_.outputMem.ptr()) - reinterpret_cast<u64>(machinePara_.mem[0].ptr());
    CHK_SAFETY_FUNC_RET(memcpy_s(exchangeDataPtr, exchangeDataBlankSize, &outputSize, sizeof(u64)));
    exchangeDataPtr += sizeof(u64);
    exchangeDataBlankSize -= sizeof(u64);
    CHK_SAFETY_FUNC_RET(memcpy_s(exchangeDataPtr, exchangeDataBlankSize, &outputOffset, sizeof(u64)));
    exchangeDataPtr += sizeof(u64);
    exchangeDataBlankSize -= sizeof(u64);

    u64 inputSize = machinePara_.inputMem.size();
    u64 inputOffset
        = reinterpret_cast<u64>(machinePara_.inputMem.ptr()) - reinterpret_cast<u64>(machinePara_.mem[0].ptr());
    CHK_SAFETY_FUNC_RET(memcpy_s(exchangeDataPtr, exchangeDataBlankSize, &inputSize, sizeof(u64)));
    exchangeDataPtr += sizeof(u64);
    exchangeDataBlankSize -= sizeof(u64);
    CHK_SAFETY_FUNC_RET(memcpy_s(exchangeDataPtr, exchangeDataBlankSize, &inputOffset, sizeof(u64)));
    exchangeDataPtr += sizeof(u64);
    exchangeDataBlankSize -= sizeof(u64);

    return HCCL_SUCCESS;
}

HcclResult TransportP2p::ParseMemIncludeInfo(void** memPtr, u64& size, u8*& exchangeDataPtr, u64& exchangeDataBlankSize)
{
    u64 memOffset = 0;
    CHK_SAFETY_FUNC_RET(memcpy_s(&size, sizeof(u64), exchangeDataPtr, sizeof(u64)));
    exchangeDataPtr += sizeof(u64);
    exchangeDataBlankSize -= sizeof(u64);
    CHK_SAFETY_FUNC_RET(memcpy_s(&memOffset, sizeof(u64), exchangeDataPtr, sizeof(u64)));
    exchangeDataPtr += sizeof(u64);
    exchangeDataBlankSize -= sizeof(u64);
    if (!machinePara_.isNewOneSide) {
        *memPtr = reinterpret_cast<void*>(reinterpret_cast<u64>(remoteIpcMemPtrVector_[0]) + memOffset);
    }
    return HCCL_SUCCESS;
}

void TransportP2p::SetMemIncludeFlag()
{
    if (machinePara_.mem.empty()) {
        return;
    }
    // 当前只取mem[0]  ->expMem
    u64 memPtr = reinterpret_cast<u64>(machinePara_.mem[0].ptr());
    u64 memEndPtr = memPtr + machinePara_.mem[0].size();
    u64 inputMemPtr = reinterpret_cast<u64>(machinePara_.inputMem.ptr());
    u64 inputMemEndPtr = inputMemPtr + machinePara_.inputMem.size();
    u64 outputMemPtr = reinterpret_cast<u64>(machinePara_.outputMem.ptr());
    u64 outputMemEndPtr = outputMemPtr + machinePara_.outputMem.size();
    HCCL_DEBUG(
        "[SetMemIncludeFlag] memPtr[%u] memEndPtr[%u], inputMemPtr[%u] inputMemEndPtr[%u], "
        "outputMemPtr[%u] outputMemEndPtr[%u]",
        memPtr, memEndPtr, inputMemPtr, inputMemEndPtr, outputMemPtr, outputMemEndPtr);
    if ((memPtr <= inputMemPtr && inputMemEndPtr <= memEndPtr)
        && (memPtr <= outputMemPtr && outputMemEndPtr <= memEndPtr)) {
        isMemInclude_ = true;
    }
    return;
}

HcclResult TransportP2p::ReplaceMemAddr(
    Transport::Buffer& localMem, Transport::Buffer& remoteMem, Transport::Buffer& newLocalMem,
    Transport::Buffer& newRemoteMem, bool& isLocalHostAddr, bool& isRemoteHostAddr)
{
    HCCL_DEBUG(
        "[TransportP2p][ReplaceMemAddr]old localAddr=[%p],localSize=[%llu],remoteAddr=[%p],remoteSize=[%llu]",
        localMem.addr, localMem.size, remoteMem.addr, remoteMem.size);

    isLocalHostAddr = false;
    isRemoteHostAddr = false;
    void* localAddr = const_cast<void*>(localMem.addr);
    u64 localSize = localMem.size;
    auto localKey = BufferKey<uintptr_t, u64>(reinterpret_cast<uintptr_t>(localAddr), localSize);
    auto localBufferPair = localHcclMemExMgr_.Find(localKey);
    if (localBufferPair.first) {
        std::shared_ptr<HcclMemEx>& localBufMemPtr = localBufferPair.second;
        u64 localDataOffSet = static_cast<u8*>(localAddr) - static_cast<u8*>(localBufMemPtr->addr);
        newLocalMem.addr = static_cast<void*>(static_cast<u8*>(localBufMemPtr->devAddr) + localDataOffSet);
        newLocalMem.size = localMem.size;
        if (localBufMemPtr->type == HcclMemType::HCCL_MEM_TYPE_HOST) {
            isLocalHostAddr = true;
        }
    } else {
        HCCL_DEBUG("[TransportP2p][ReplaceMemAddr] Can't find localBufferPair by key {%p, %llu}", localAddr, localSize);
        newLocalMem.addr = localAddr;
        newLocalMem.size = localSize;
    }

    void* remoteAddr = const_cast<void*>(remoteMem.addr);
    auto remoteKey = BufferKey<uintptr_t, u64>(reinterpret_cast<uintptr_t>(remoteAddr), remoteMem.size);
    auto remoteBufferPair = remoteHcclMemExMgr_.Find(remoteKey);
    if (remoteBufferPair.first) {
        std::shared_ptr<HcclMemEx>& remoteBufMemPtr = remoteBufferPair.second;
        u64 remoteDataOffSet = static_cast<u8*>(remoteAddr) - static_cast<u8*>(remoteBufMemPtr->addr);
        newRemoteMem.addr = static_cast<void*>(static_cast<u8*>(remoteBufMemPtr->devAddr) + remoteDataOffSet);
        if (remoteBufMemPtr->type == HcclMemType::HCCL_MEM_TYPE_HOST) {
            isRemoteHostAddr = true;
        }
    } else {
        HCCL_DEBUG(
            "[TransportP2p][ReplaceMemAddr] Can't find remoteBuffer by key {%p, %llu}", remoteAddr, remoteMem.size);
        newRemoteMem.addr = remoteMem.addr;
    }
    newRemoteMem.size = remoteMem.size;

    HCCL_DEBUG(
        "[TransportP2p][ReplaceMemAddr]old localAddr=[%p],localSize=[%llu],remoteAddr=[%p],remoteSize=[%llu], "
        "isLocalHostAddr[%u] isRemoteHostAddr[%u]",
        newLocalMem.addr, newLocalMem.size, newRemoteMem.addr, newRemoteMem.size,
        static_cast<uint32_t>(isLocalHostAddr), static_cast<uint32_t>(isRemoteHostAddr));
    return HCCL_SUCCESS;
}

HcclResult TransportP2p::InitHcclMemExMgrWithMem(HcclMemEx* bufMem, u32 bufSize, HcclMemExMgr& hcommMemExMgr)
{
    for (u32 i = 0; i < bufSize; i++) {
        HcclMemEx& bufMemTmp = bufMem[i];

        std::shared_ptr<HcclMemEx> hcclMemEx = nullptr;
        hcclMemEx = std::make_shared<HcclMemEx>();
        CHK_PTR_NULL(hcclMemEx);

        HcclMemEx* hcclMemExPtr = reinterpret_cast<HcclMemEx*>(hcclMemEx.get());
        *hcclMemExPtr = bufMemTmp;

        hccl::BufferKey<uintptr_t, u64> tempKey(reinterpret_cast<uintptr_t>(bufMemTmp.addr), bufMemTmp.size);
        auto resultPair = hcommMemExMgr.Add(tempKey, hcclMemEx);
        if (!resultPair.second) {
            HCCL_ERROR(
                "[TransportP2p][InitHcclMemExMgrWithMem]add addr:%p, size[%lu], type[%u], devAddr[%p] fail",
                bufMemTmp.addr, bufMemTmp.size, static_cast<u32>(bufMemTmp.type), bufMemTmp.devAddr);
            return HCCL_E_INTERNAL;
        } else {
            HCCL_INFO(
                "[TransportP2p][InitHcclMemExMgrWithMem]add addr:%p, size[%lu], type[%u], devAddr[%p] done",
                bufMemTmp.addr, bufMemTmp.size, static_cast<u32>(bufMemTmp.type), bufMemTmp.devAddr);
        }
    }

    HCCL_INFO("[TransportP2p][InitHcclMemExMgrWithMem] done");
    return HCCL_SUCCESS;
}

HcclResult TransportP2p::InitHcclMemExMgr(MachinePara& machinePara)
{
    HCCL_INFO("[TransportP2p][InitHcclMemExMgr] start");
    CHK_RET(InitHcclMemExMgrWithMem(machinePara.localBufMem, machinePara.localBufSize, localHcclMemExMgr_));
    machinePara.localBufMem = nullptr;
    machinePara.localBufSize = 0;
    HCCL_INFO("[TransportP2p][InitHcclMemExMgr] local done");
    CHK_RET(InitHcclMemExMgrWithMem(machinePara.remoteBufMem, machinePara.remoteBufSize, remoteHcclMemExMgr_));
    machinePara.remoteBufMem = nullptr;
    machinePara.remoteBufSize = 0;
    HCCL_INFO("[TransportP2p][InitHcclMemExMgr] remote done");
    return HCCL_SUCCESS;
}
} // namespace hccl
