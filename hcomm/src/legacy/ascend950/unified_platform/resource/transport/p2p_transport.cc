/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "p2p_transport.h"

#include "orion_adapter_rts.h"
#include "exchange_ipc_notify_dto.h"
#include "exchange_ipc_buffer_dto.h"
#include "user_remote_mem_getter.h"

namespace Hccl {

P2PTransport::P2PTransport(
    CommonLocRes& commonLocRes, Attribution& attr, const LinkData& linkData, const Socket& socket)
    : BaseMemTransport(commonLocRes, attr, linkData, socket, TransportType::P2P)
{}

P2PTransport::P2PTransport(
    CommonLocRes& commonLocRes, Attribution& attr, const LinkData& linkData, const Socket& socket,
    std::function<void(u32 streamId, u32 taskId, TaskParam taskParam)> callback)
    : BaseMemTransport(commonLocRes, attr, linkData, socket, TransportType::P2P, callback)
{}

std::string P2PTransport::Describe() const
{
    string msg = StringFormat(
        "P2PTransport:commonLocRes=%s, pidMsgSize=%u, myPid=%u, rmtPid=%u, rmtPidValid=%d,",
        commonLocRes.Describe().c_str(), pidMsgSize, myPid, rmtPid, rmtPidValid);
    msg += StringFormat("exchangeDataSize=%u, ", exchangeDataSize);
    u32 pos = 0;
    for (auto& it : rmtNotifyVec) {
        msg += StringFormat("rmtNotify[%u]=%s, ", pos, it->Describe().c_str());
        pos++;
    }

    pos = 0;
    for (auto& it : rmtNotifyVec) {
        if (it != nullptr) {
            msg += StringFormat("rmtBuffer[%u]=%s, ", pos, it->Describe().c_str());
        } else {
            msg += StringFormat("rmtBuffer[%u]=nullptr, ", pos);
        }
        pos++;
    }
    return msg;
}

static void SubmitTask(const TaskP2pMemcpy& p2pMemcpy, const Stream& stream)
{
    HCCL_INFO(
        "[P2PTransport::%s]not support, p2p dst addr[%llu], stream[%p]", __func__, p2pMemcpy.GetDstAddr(),
        stream.GetPtr());
}

static void SubmitTask(const TaskSdmaReduce& sdmaReduce, const Stream& stream)
{
    HCCL_INFO(
        "[P2PTransport::%s]not support, sdmaReduce dst addr[%llu], stream[%p]", __func__, sdmaReduce.GetDstAddr(),
        stream.GetPtr());
}

template <typename TaskType>
std::function<void(const BaseTask&, const Stream&)> GetSubmitP2PTaskFunction()
{
    return [](const BaseTask& task, const Stream& stream) {
        SubmitTask(static_cast<const TaskType&>(task), stream);
    };
}

std::map<TaskType, std::function<void(const BaseTask&, const Stream&)>> g_p2pTaskSubmitRuleMap
    = {{TaskType::P2P_MEMCPY, GetSubmitP2PTaskFunction<TaskP2pMemcpy>()},
       {TaskType::SDMA_REDUCE, GetSubmitP2PTaskFunction<TaskSdmaReduce>()}};

static void SubmitP2PTask(unique_ptr<BaseTask> task, const Stream& stream)
{
    if (task != nullptr) { // task为空的情况下，不需要提交task
        g_p2pTaskSubmitRuleMap.at(task->GetType())(*task.get(), stream);
    }
}

MemoryBuffer P2PTransport::GetLocMemBuffer(const RmaBufferSlice& locSlice) const
{
    return MemoryBuffer(locSlice.addr, locSlice.size, 0);
}

MemoryBuffer P2PTransport::GetRmtMemBuffer(const RmtRmaBufferSlice& rmtSlice) const
{
    return MemoryBuffer(rmtSlice.addr, rmtSlice.size, 0);
}

void P2PTransport::Post(u32 index, const Stream& stream) { rmtNotifyVec[index]->Post(stream); }

void P2PTransport::Read(const RmaBufferSlice& locSlice, const RmtRmaBufferSlice& rmtSlice, const Stream& stream)
{
    SqeConfig config;
    SubmitP2PTask(
        commonLocRes.connVec[0]->PrepareRead(GetRmtMemBuffer(rmtSlice), GetLocMemBuffer(locSlice), config), stream);
}

void P2PTransport::ReadReduce(
    const RmaBufferSlice& locSlice, const RmtRmaBufferSlice& rmtSlice, const ReduceIn& reduceIn, const Stream& stream)
{
    SqeConfig config;
    SubmitP2PTask(
        commonLocRes.connVec[0]->PrepareReadReduce(
            GetRmtMemBuffer(rmtSlice), GetLocMemBuffer(locSlice), reduceIn.dataType, reduceIn.reduceOp, config),
        stream);
}

void P2PTransport::Write(const RmaBufferSlice& locSlice, const RmtRmaBufferSlice& rmtSlice, const Stream& stream)
{
    SqeConfig config;
    SubmitP2PTask(
        commonLocRes.connVec[0]->PrepareWrite(GetRmtMemBuffer(rmtSlice), GetLocMemBuffer(locSlice), config), stream);
}

void P2PTransport::WriteReduce(
    const RmaBufferSlice& locSlice, const RmtRmaBufferSlice& rmtSlice, const ReduceIn& reduceIn, const Stream& stream)
{
    SqeConfig config;
    SubmitP2PTask(
        commonLocRes.connVec[0]->PrepareWriteReduce(
            GetRmtMemBuffer(rmtSlice), GetLocMemBuffer(locSlice), reduceIn.dataType, reduceIn.reduceOp, config),
        stream);
}

TransportStatus P2PTransport::GetStatus()
{
    if (baseStatus == TransportStatus::READY) {
        return baseStatus;
    } else if (baseStatus == TransportStatus::INIT) {
        p2pStatus = P2PStatus::INIT;
    }

    if (!IsSocketReady()) {
        return baseStatus;
    }

    switch (p2pStatus) {
        case P2PStatus::INIT:
            p2pStatus = P2PStatus::SOCKET_OK;
            baseStatus = TransportStatus::SOCKET_OK;
            break;
        case P2PStatus::SOCKET_OK:
            PrepareSendData();
            p2pStatus = P2PStatus::SEND_DATA_SIZE;
            break;
        case P2PStatus::SEND_DATA_SIZE:
            RecvDataSize();
            p2pStatus = P2PStatus::RECV_DATA_SIZE;
            break;
        case P2PStatus::RECV_DATA_SIZE:
            SendExchangeData();
            p2pStatus = P2PStatus::SEND_DATA;
            break;
        case P2PStatus::SEND_DATA:
            RecvExchangeData();
            p2pStatus = P2PStatus::RECV_DATA;
            break;
        case P2PStatus::RECV_DATA:
            ProcessRecvData();
            SetBaseStatusReady();
            break;
        default:
            break;
    }
    HCCL_INFO(
        "%s, baseStatus=%s, p2pStatus = %s", GetLinkDescInfo().c_str(), baseStatus.Describe().c_str(),
        p2pStatus.Describe().c_str());
    return baseStatus;
}

bool P2PTransport::IsRmtPidValid() const
{
    // 优化方向：基于remoteAddr保存PID的单例，这样可以减少 PID交换
    return rmtPidValid;
}

void P2PTransport::SendPid()
{
    myPid = HrtDeviceGetBareTgid();
    HCCL_INFO("P2PTransport: send pid %u", myPid);

    BinaryStream binaryStream;
    binaryStream << myPid;

    std::vector<char> data;
    binaryStream.Dump(data);
    pidMsgSize = data.size();
    socket->SendAsync(&data[0], data.size());

    HCCL_INFO(
        "send pid %s, size=%zu, data=0x%s", GetLinkDescInfo().c_str(), data.size(),
        Bytes2hex(data.data(), data.size()).c_str());
}

void P2PTransport::RecvPid()
{
    std::vector<char> data(pidMsgSize);
    socket->RecvAsync(reinterpret_cast<u8*>(&data[0]), data.size());
    HCCL_INFO(
        "recv pid %s, size=%zu, data=%s", GetLinkDescInfo().c_str(), data.size(),
        Bytes2hex(data.data(), data.size()).c_str());

    BinaryStream binaryStream(data);
    binaryStream >> rmtPid;
    HCCL_INFO("P2PTransport: recv pid %u", rmtPid);
}

void P2PTransport::Grant() const
{
    // 暂时不做Grant处理
    return;
}

void P2PTransport::PrepareSendData()
{
    notifyNum = commonLocRes.notifyVec.size(); // 需要交换的notify数量
    bufferNum = commonLocRes.bufferVec.size(); // 需要交换的buffer数量

    HCCL_INFO(
        "%s commLocResExchange %s, notifyNum=%u, bufferNum=%u", GetLinkDescInfo().c_str(),
        commonLocRes.Describe().c_str(), notifyNum, bufferNum);

    BinaryStream binaryStream;

    HandshakeMsgPack(binaryStream);
    NotifyVecPack(binaryStream);
    BufferVecPack(binaryStream);

    sendData.clear();
    binaryStream.Dump(sendData);
    exchangeDataSize = sendData.size();
    socket->SendAsync(&exchangeDataSize, sizeof(exchangeDataSize));

    HCCL_INFO("send datasize %s, size=%u", GetLinkDescInfo().c_str(), exchangeDataSize);
}

void P2PTransport::RecvDataSize()
{
    socket->RecvAsync(reinterpret_cast<u8*>(&exchangeDataSize), sizeof(exchangeDataSize));
    HCCL_INFO("P2PTransport: recv datasize %u", exchangeDataSize);
}

void P2PTransport::SendExchangeData()
{
    socket->SendAsync(&sendData[0], sendData.size());
    HCCL_INFO(
        "send data %s, size=%zu, data=0x%s", GetLinkDescInfo().c_str(), sendData.size(),
        Bytes2hex(sendData.data(), sendData.size()).c_str());
}

void P2PTransport::RecvExchangeData()
{
    recvData.resize(exchangeDataSize);
    socket->RecvAsync(reinterpret_cast<u8*>(&recvData[0]), recvData.size());
    HCCL_INFO(
        "RecvExchangeData recv data %s, size=%zu, data=%s", GetLinkDescInfo().c_str(), recvData.size(),
        Bytes2hex(recvData.data(), recvData.size()).c_str());
}

void P2PTransport::ProcessRecvData()
{
    HCCL_INFO(
        "ProcessRecvData recv data %s, size=%zu, data=%s", GetLinkDescInfo().c_str(), recvData.size(),
        Bytes2hex(recvData.data(), recvData.size()).c_str());

    BinaryStream binaryStream(recvData);
    HandshakeMsgUnpack(binaryStream);
    RmtNotifyVecUnpackProc(binaryStream);
    RmtBufferVecUnpackProc(binaryStream);

    HCCL_INFO("%s unpack success", GetLinkDescInfo().c_str());
}

void P2PTransport::RmtNotifyVecUnpackProc(BinaryStream& binaryStream)
{
    u32 rmtNotifyNum;
    binaryStream >> rmtNotifyNum;
    HCCL_INFO("unpack notify %s locNum=%u, rmtNum=%u", GetLinkDescInfo().c_str(), notifyNum, rmtNotifyNum);
    if (rmtNotifyNum != notifyNum) {
        MACRO_THROW(
            InvalidParamsException,
            StringFormat("notifyNum=%u is not equal to rmtNotifyNum=%u", notifyNum, rmtNotifyNum));
    }

    rmtNotifyVec.clear(); // 清空remote资源
    for (u32 i = 0; i < rmtNotifyNum; i++) {
        u32 pos;
        binaryStream >> pos;
        ExchangeIpcNotifyDto dto;
        dto.Deserialize(binaryStream);
        HCCL_INFO("unpack notify pos=%u dto %s", pos, dto.Describe().c_str());
        rmtNotifyVec.push_back(make_unique<IpcRemoteNotify>(dto));
        HCCL_INFO("unpack notify pos=%u, rmtNotify=%s", pos, rmtNotifyVec[i]->Describe().c_str());
    }
}

void P2PTransport::BufferVecPack(BinaryStream& binaryStream)
{
    binaryStream << bufferNum;
    HCCL_INFO("start pack %s bufferVec", transportType.Describe().c_str());
    u32 pos = 0;
    for (auto& it : commonLocRes.bufferVec) {
        binaryStream << pos;
        if (it != nullptr) { // 非空的buffer，从buffer中获取 dto
            std::unique_ptr<Serializable> dto = it->GetExchangeDto();
            dto->Serialize(binaryStream);
            HCCL_INFO("pack buffer pos=%u dto %s", pos, dto->Describe().c_str());
        } else { // 空的buffer，dto所有字段为0(size=0)
            ExchangeIpcBufferDto exchangeDto;
            exchangeDto.Serialize(binaryStream);
            HCCL_INFO("pack buffer pos=%u, dto is null %s", pos, exchangeDto.Describe().c_str());
        }
        pos++;
    }
}

void P2PTransport::RmtBufferVecUnpackProc(BinaryStream& binaryStream)
{
    u32 rmtBufferNum;
    binaryStream >> rmtBufferNum;
    HCCL_INFO("unpack buffer %s locNum=%u rmtNum=%u", GetLinkDescInfo().c_str(), bufferNum, rmtBufferNum);
    if (rmtBufferNum != bufferNum) {
        MACRO_THROW(
            InvalidParamsException,
            StringFormat("bufferNum=%u is not equal to rmtBufferNum=%u", bufferNum, rmtBufferNum));
    }

    rmtBufferVec.clear();
    rmtRmaBufferVec.clear();
    for (u32 i = 0; i < rmtBufferNum; i++) {
        u32 pos;
        binaryStream >> pos;
        ExchangeIpcBufferDto dto;
        dto.Deserialize(binaryStream);
        HCCL_INFO("unpack buffer pos=%u, dto %s", pos, dto.Describe().c_str());

        if (dto.size == 0) { // size为0，则为 remote 空buffer
            HCCL_INFO("unpack nullptr, pos=%u", pos);
            rmtBufferVec.push_back(nullptr);
            rmtRmaBufferVec.push_back((nullptr));
        } else { // size非0，则构造一个remote buffer
            rmtBufferVec.push_back(make_unique<RemoteIpcRmaBuffer>(dto));
            rmtRmaBufferVec.push_back(rmtBufferVec.back().get());
            HCCL_INFO("unpack buffer pos=%u, rmtRmaBuffer=%s", pos, rmtBufferVec.back()->Describe().c_str());
        }
    }
}

std::vector<char> P2PTransport::GetUniqueId()
{
    if (baseStatus != TransportStatus::READY) {
        MACRO_THROW(InternalException, StringFormat("transport status is not ready, please check"));
    }
    u32 type = static_cast<u32>(transportType);
    BinaryStream binaryStream;
    binaryStream << type;
    binaryStream << notifyNum;
    binaryStream << bufferNum;

    // [header...][notifyUniqueId...][rmtNotifyUniqueId...][rmtBufferUniqueIds...]
    auto notifyUniqueIds = GetNotifyUniqueIds();
    binaryStream << notifyUniqueIds;

    auto rmtNotifyUniqueIds = GetRmtNotifyUniqueIds();
    binaryStream << rmtNotifyUniqueIds;

    auto rmtBufferUniqueIds = GetRmtBufferUniqueIds();
    binaryStream << rmtBufferUniqueIds;

    std::vector<char> result;
    binaryStream.Dump(result);
    return result;
}

std::vector<char> P2PTransport::GetUniqueIdV2()
{
    if (baseStatus != TransportStatus::READY) {
        MACRO_THROW(InternalException, StringFormat("transport status is not ready, please check"));
    }
    u32 type = static_cast<u32>(transportType);
    BinaryStream binaryStream;
    binaryStream << type;
    binaryStream << notifyNum;
    binaryStream << bufferNum;

    // [header...][notifyUniqueId...][rmtNotifyUniqueId...][rmtBufferUniqueIds...]
    auto notifyUniqueIds = GetNotifyUniqueIds();
    binaryStream << notifyUniqueIds;

    auto rmtNotifyUniqueIds = GetRmtNotifyUniqueIds();
    binaryStream << rmtNotifyUniqueIds;

    auto locBufferUniqueIds = GetLocBufferUniqueIds();
    binaryStream << locBufferUniqueIds;

    auto rmtBufferUniqueIds = GetRmtBufferUniqueIds();
    binaryStream << rmtBufferUniqueIds;

    std::vector<char> result;
    binaryStream.Dump(result);
    return result;
}

std::vector<char> P2PTransport::GetNotifyUniqueIds()
{
    HCCL_INFO("start packing all notify uniqueIds");
    std::vector<char> result(0);
    for (auto& it : commonLocRes.notifyVec) {
        HCCL_INFO("p2pMemTransport Notify %s", it->Describe().c_str());
        auto uniqueId = it->GetUniqueId();
        result.insert(result.end(), uniqueId.begin(), uniqueId.end());
    }
    return result;
}

std::vector<char> P2PTransport::GetSingleRmtNotifyUniqueId(u64 addr, u64 size, u32 notifyId) const
{
    BinaryStream binaryStream;
    binaryStream << addr;
    binaryStream << size;
    binaryStream << notifyId;
    HCCL_INFO("P2PTransport RmtNotifyAddr[addr=0x%llx, size=0x%llx, notifyId=%u]", addr, size, notifyId);
    std::vector<char> result;
    binaryStream.Dump(result);
    return result;
}

std::vector<char> P2PTransport::GetRmtNotifyUniqueIds() const
{
    HCCL_INFO("start packing all remote notify uniqueIds");
    std::vector<char> result(0);
    for (auto& it : rmtNotifyVec) {
        std::vector<char> uniqueId;
        if (it != nullptr) {
            uniqueId = GetSingleRmtNotifyUniqueId(it->GetAddr(), it->GetSize(), it->GetId());
            HCCL_INFO("P2PTransport::GetRmtNotifyUniqueIds, %s", it->Describe().c_str());
        } else {
            uniqueId = GetSingleRmtNotifyUniqueId(0, 0, 0); // 填充一个空的buffer
            HCCL_INFO("P2PTransport::GetRmtNotifyUniqueIds, null buffer");
        }
        result.insert(result.end(), uniqueId.begin(), uniqueId.end());
    }
    return result;
}

std::vector<char> P2PTransport::GetSingleBufferUniqueId(u64 addr, u64 size) const
{
    BinaryStream binaryStream;
    binaryStream << addr;
    binaryStream << size;
    HCCL_INFO("P2PTransport BufferAddr[addr=0x%llx, size=0x%llx]", addr, size);
    std::vector<char> result;
    binaryStream.Dump(result);
    return result;
}

std::vector<char> P2PTransport::GetLocBufferUniqueIds() const
{
    HCCL_INFO("start packing all local buffer uniqueIds");
    std::vector<char> result(0);
    for (auto& it : commonLocRes.bufferVec) {
        std::vector<char> uniqueId;
        if (it != nullptr) {
            uniqueId = GetSingleBufferUniqueId(it->GetAddr(), it->GetSize());
            HCCL_INFO("P2PTransport::GetLocBufferUniqueIds, %s", it->Describe().c_str());
        } else {
            uniqueId = GetSingleBufferUniqueId(0, 0); // 填充一个空的buffer
            HCCL_INFO("P2PTransport::GetLocBufferUniqueIds, null buffer");
        }
        result.insert(result.end(), uniqueId.begin(), uniqueId.end());
    }
    return result;
}

std::vector<char> P2PTransport::GetRmtBufferUniqueIds() const
{
    HCCL_INFO("start packing all remote buffer uniqueIds");
    std::vector<char> result(0);
    for (auto& it : rmtBufferVec) {
        std::vector<char> uniqueId;
        if (it != nullptr) {
            uniqueId = GetSingleBufferUniqueId(it->GetAddr(), it->GetSize());
            HCCL_INFO("P2PTransport::GetRmtBufferUniqueIds, %s", it->Describe().c_str());
        } else {
            uniqueId = GetSingleBufferUniqueId(0, 0); // 填充一个空的buffer
            HCCL_INFO("P2PTransport::GetRmtBufferUniqueIds, null buffer");
        }
        result.insert(result.end(), uniqueId.begin(), uniqueId.end());
    }
    return result;
}

HcclResult P2PTransport::GetRemoteMems(uint32_t* memNum, CommMem** remoteMem, char*** memInfos)
{
    std::lock_guard<std::mutex> lock(remoteMemsMutex_);
    Hccl::RemoteMemCtx<std::unique_ptr<RemoteIpcRmaBuffer>> remoteMemCtx{
        cacheValid_, rmtBufferVec, remoteUserMems_, memInfoCopies_, memInfoPointers_, remoteMem, memInfos, memNum};
    CHK_RET(GetRemoteUserMems(remoteMemCtx));
    return HCCL_SUCCESS;
}
} // namespace Hccl
