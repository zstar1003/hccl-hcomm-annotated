/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "prim_rules.h"
#include "not_support_exception.h"
#include "dev_capability.h"
namespace Hccl {
constexpr u32 INSTRUCTION_PRI_LOCAL_POST_TO = 90;
constexpr u32 INSTRUCTION_PRI_LOCAL_WAIT_FROM = 85;
constexpr u32 INSTRUCTION_PRI_LOCAL_COPY = 100;
constexpr u32 INSTRUCTION_PRI_LOCAL_REDUCE = 30;
constexpr u32 INSTRUCTION_PRI_POST_READY = 80;
constexpr u32 INSTRUCTION_PRI_WAIT_READY = 70;
constexpr u32 INSTRUCTION_PRI_READ = 60;
constexpr u32 INSTRUCTION_PRI_READ_REDUCE = 60;
constexpr u32 INSTRUCTION_PRI_WRITE = 60;
constexpr u32 INSTRUCTION_PRI_WRITE_REDUCE = 60;
constexpr u32 INSTRUCTION_PRI_WRITE_WITH_FIN = 60;
constexpr u32 INSTRUCTION_PRI_WRITE_REDUCE_WITH_FIN = 60;
constexpr u32 INSTRUCTION_PRI_POST_FIN = 60;
constexpr u32 INSTRUCTION_PRI_WAIT_FIN = 50;
constexpr u32 INSTRUCTION_PRI_POST_FIN_ACK = 50;
constexpr u32 INSTRUCTION_PRI_WAIT_FIN_ACK = 40;

const std::map<InstructionType, u32> INSTRUCTION_PRI_MAP
    = {{InstructionType::LOCAL_COPY, INSTRUCTION_PRI_LOCAL_COPY},
       {InstructionType::LOCAL_REDUCE, INSTRUCTION_PRI_LOCAL_REDUCE},
       {InstructionType::LOCAL_POST_TO, INSTRUCTION_PRI_LOCAL_POST_TO},
       {InstructionType::LOCAL_WAIT_FROM, INSTRUCTION_PRI_LOCAL_WAIT_FROM},
       {InstructionType::POST_READY, INSTRUCTION_PRI_POST_READY},
       {InstructionType::WAIT_READY, INSTRUCTION_PRI_WAIT_READY},
       {InstructionType::POST_FIN, INSTRUCTION_PRI_POST_FIN},
       {InstructionType::WAIT_FIN, INSTRUCTION_PRI_WAIT_FIN},
       {InstructionType::POST_FIN_ACK, INSTRUCTION_PRI_POST_FIN_ACK},
       {InstructionType::WAIT_FIN_ACK, INSTRUCTION_PRI_WAIT_FIN_ACK},
       {InstructionType::READ, INSTRUCTION_PRI_READ},
       {InstructionType::READ_REDUCE, INSTRUCTION_PRI_READ_REDUCE},
       {InstructionType::WRITE, INSTRUCTION_PRI_WRITE},
       {InstructionType::WRITE_REDUCE, INSTRUCTION_PRI_WRITE_REDUCE},
       {InstructionType::WRITE_WITH_FIN, INSTRUCTION_PRI_WRITE_WITH_FIN},
       {InstructionType::WRITE_REDUCE_WITH_FIN, INSTRUCTION_PRI_WRITE_REDUCE_WITH_FIN}};

inline void CheckLinkIsValid(const LinkData& link, const string& desc)
{
    // only support P2P, dev_net+RDMA  now
    if (link.GetType() == PortDeploymentType::P2P) {
        return;
    } else if (link.GetType() == PortDeploymentType::DEV_NET) {
        auto linkProtocol = link.GetLinkProtocol();
        HCCL_INFO("[CheckLinkIsValid] linkProtocol is[%s]", linkProtocol.Describe().c_str());
        if (linkProtocol == LinkProtocol::ROCE || linkProtocol == LinkProtocol::UB_CTP
            || linkProtocol == LinkProtocol::UB_TP || linkProtocol == LinkProtocol::UBOE
            || linkProtocol == LinkProtocol::UB_RTP) {
            return;
        }
    }
    string msg = StringFormat("type=%s is not support in %s", link.Describe().c_str(), desc.c_str());
    throw NotSupportException(msg);
}

inline bool IsSupportInlineReduce(const DataType& datatype, const ReduceOp& reduceOp, const LinkData& link)
{
    bool isDataType = DevCapability::GetInstance().GetInlineReduceDataTypeMap().at(datatype);
    bool isReduceOp = DevCapability::GetInstance().GetInlineReduceOpMap().at(reduceOp);

    bool result = isDataType && isReduceOp;
    if (link.GetType() == PortDeploymentType::P2P) {
        return result;
    } else {
        // here is DevNet
        bool isSupportDevNetInlineReduce = DevCapability::GetInstance().IsSupportDevNetInlineReduce();
        return result && isSupportDevNetInlineReduce;
    }
}

inline void AppendInsPostFinAck(RankId remote, const LinkData& link, vector<unique_ptr<Instruction>>& instructions)
{
    if (link.GetType() == PortDeploymentType::DEV_NET) {
        if (!DevCapability::GetInstance().IsSupportStarsPollNetCq()) {
            instructions.push_back(make_unique<InsPostFinAck>(remote, link));
        }
        return;
    }
    if (link.GetType() == PortDeploymentType::P2P) {
        // do nothing
        return;
    }

    // not support, throw exception
    string msg = StringFormat("link=%s does not need or not support AppendInsPostFinAck", link.Describe().c_str());
    THROW<NotSupportException>(msg);
}

inline vector<unique_ptr<Instruction>> PrimSendInReadMode(const PrimSend& send)
{
    vector<unique_ptr<Instruction>> instructions(InsArraySize::TWO);
    RankId remote = send.GetRemoteRank();
    const LinkData link = send.GetLink();
    u32 index = 0;

    instructions[index++] = make_unique<InsPostReady>(remote, link);
    instructions[index++] = make_unique<InsWaitFin>(remote, link);
    AppendInsPostFinAck(remote, link, instructions);

    return instructions;
}

inline void AppendInsWaitFinAck(RankId remote, const LinkData& link, vector<unique_ptr<Instruction>>& instructions)
{
    if (link.GetType() == PortDeploymentType::DEV_NET) {
        if (!DevCapability::GetInstance().IsSupportStarsPollNetCq()) {
            instructions.push_back(make_unique<InsWaitFinAck>(remote, link));
        }
        return;
    }
    if (link.GetType() == PortDeploymentType::P2P) {
        // do nothing
        return;
    }
    // not support, throw exception
    string msg = StringFormat("link=%s does not need or not support AppendInsWaitFinAck", link.Describe().c_str());
    THROW<NotSupportException>(msg);
}

inline vector<unique_ptr<Instruction>> PrimSendInWriteWithNotifyMode(const PrimSend& send)
{
    vector<unique_ptr<Instruction>> instructions(InsArraySize::ONE + send.Size());
    RankId remote = send.GetRemoteRank();
    const LinkData link = send.GetLink();
    u32 index = 0;

    instructions[index++] = make_unique<InsWaitReady>(remote, link);
    for (u32 pos = 0; pos < send.Size() - 1; pos++) {
        instructions[index++] = make_unique<InsWrite>(remote, link, send.GetLocalSlice(pos), send.GetRemoteSlice(pos));
    }
    instructions[index++] = make_unique<InsWriteWithFin>(
        remote, link, send.GetLocalSlice(send.Size() - 1), send.GetRemoteSlice(send.Size() - 1), NotifyType::NORMAL);
    AppendInsWaitFinAck(remote, link, instructions);

    return instructions;
}

inline vector<unique_ptr<Instruction>> PrimSendInNormalWriteMode(const PrimSend& send)
{
    vector<unique_ptr<Instruction>> instructions(InsArraySize::TWO + send.Size());
    RankId remote = send.GetRemoteRank();
    const LinkData link = send.GetLink();
    u32 index = 0;

    instructions[index++] = make_unique<InsWaitReady>(remote, link);
    for (u32 pos = 0; pos < send.Size(); pos++) {
        instructions[index++] = make_unique<InsWrite>(remote, link, send.GetLocalSlice(pos), send.GetRemoteSlice(pos));
    }
    instructions[index++] = make_unique<InsPostFin>(remote, link);
    AppendInsWaitFinAck(remote, link, instructions);

    return instructions;
}

inline vector<unique_ptr<Instruction>> PrimSendInWriteMode(const PrimSend& send)
{
    if (send.GetLink().GetType() == PortDeploymentType::P2P) {
        return PrimSendInNormalWriteMode(send);
    } else if (send.GetLink().GetType() == PortDeploymentType::DEV_NET) {
        if (DevCapability::GetInstance().IsSupportWriteWithNotify()) {
            return PrimSendInWriteWithNotifyMode(send);
        }
        return PrimSendInNormalWriteMode(send);
    }
    // not support, throw exception
    string msg = StringFormat("link=%s does not support PrimSendInWriteMode", send.GetLink().Describe().c_str());
    MACRO_THROW(NotSupportException, msg);
}

inline vector<unique_ptr<Instruction>> PrimRecvInReadMode(const PrimRecv& recv)
{
    vector<unique_ptr<Instruction>> instructions(InsArraySize::TWO + recv.Size());
    RankId remote = recv.GetRemoteRank();
    const LinkData link = recv.GetLink();
    u32 index = 0;

    instructions[index++] = make_unique<InsWaitReady>(remote, link);
    for (u32 pos = 0; pos < recv.Size(); pos++) {
        instructions[index++] = make_unique<InsRead>(remote, link, recv.GetLocalSlice(pos), recv.GetRemoteSlice(pos));
    }
    instructions[index++] = make_unique<InsPostFin>(remote, link);
    AppendInsWaitFinAck(remote, link, instructions);

    return instructions;
}

inline vector<unique_ptr<Instruction>> PrimRecvInWriteMode(const PrimRecv& recv)
{
    vector<unique_ptr<Instruction>> instructions(InsArraySize::TWO);
    RankId remote = recv.GetRemoteRank();
    const LinkData link = recv.GetLink();
    u32 index = 0;

    instructions[index++] = make_unique<InsPostReady>(remote, link);
    instructions[index++] = make_unique<InsWaitFin>(remote, link);
    AppendInsPostFinAck(remote, link, instructions);

    return instructions;
}

inline vector<unique_ptr<Instruction>> PrimSendReduceInReadModeWithInlineReduce(const PrimSendReduce& sendReduce)
{
    vector<unique_ptr<Instruction>> instructions(InsArraySize::TWO);
    RankId remote = sendReduce.GetRemoteRank();
    const LinkData link = sendReduce.GetLink();
    u32 index = 0;

    instructions[index++] = make_unique<InsPostReady>(remote, link);
    instructions[index++] = make_unique<InsWaitFin>(remote, link);
    AppendInsPostFinAck(remote, link, instructions);

    return instructions;
}

inline vector<unique_ptr<Instruction>>
PrimSendReduceInWriteWithNotifyModeWithInlineReduce(const PrimSendReduce& sendReduce)
{
    vector<unique_ptr<Instruction>> instructions(InsArraySize::ONE + sendReduce.Size());
    RankId remote = sendReduce.GetRemoteRank();
    const LinkData link = sendReduce.GetLink();
    u32 index = 0;

    instructions[index++] = make_unique<InsWaitReady>(remote, link);
    for (u32 pos = 0; pos < sendReduce.Size() - 1; pos++) {
        instructions[index++] = make_unique<InsWriteReduce>(
            remote, link, sendReduce.GetLocalSlice(pos), sendReduce.GetRemoteDstSlice(pos), sendReduce.GetDataType(),
            sendReduce.GetReduceOp());
    }
    instructions[index++] = make_unique<InsWriteReduceWithFin>(
        remote, link, sendReduce.GetLocalSlice(sendReduce.Size() - 1),
        sendReduce.GetRemoteDstSlice(sendReduce.Size() - 1), sendReduce.GetDataType(), sendReduce.GetReduceOp(),
        NotifyType::NORMAL);
    AppendInsWaitFinAck(remote, link, instructions);

    return instructions;
}

inline vector<unique_ptr<Instruction>> PrimSendReduceInNormalWriteModeWithInlineReduce(const PrimSendReduce& sendReduce)
{
    vector<unique_ptr<Instruction>> instructions(InsArraySize::TWO + sendReduce.Size());
    RankId remote = sendReduce.GetRemoteRank();
    const LinkData link = sendReduce.GetLink();
    u32 index = 0;

    instructions[index++] = make_unique<InsWaitReady>(remote, link);
    for (u32 pos = 0; pos < sendReduce.Size(); pos++) {
        instructions[index++] = make_unique<InsWriteReduce>(
            remote, link, sendReduce.GetLocalSlice(pos), sendReduce.GetRemoteDstSlice(pos), sendReduce.GetDataType(),
            sendReduce.GetReduceOp());
    }
    instructions[index++] = make_unique<InsPostFin>(remote, link);
    AppendInsWaitFinAck(remote, link, instructions);

    return instructions;
}

inline vector<unique_ptr<Instruction>> PrimSendReduceInWriteModeWithInlineReduce(const PrimSendReduce& sendReduce)
{
    if (sendReduce.GetLink().GetType() == PortDeploymentType::P2P) {
        return PrimSendReduceInNormalWriteModeWithInlineReduce(sendReduce);
    } else if (sendReduce.GetLink().GetType() == PortDeploymentType::DEV_NET) {
        if (DevCapability::GetInstance().IsSupportWriteWithNotify()) {
            return PrimSendReduceInWriteWithNotifyModeWithInlineReduce(sendReduce);
        }
        return PrimSendReduceInNormalWriteModeWithInlineReduce(sendReduce);
    }

    string msg = StringFormat(
        "link=%s does not support PrimSendReduceInWriteModeWithInlineReduce", sendReduce.GetLink().Describe().c_str());
    MACRO_THROW(NotSupportException, msg);
}

inline vector<unique_ptr<Instruction>> PrimSendReduceInReadModeWithoutInlineReduce(const PrimSendReduce& sendReduce)
{
    vector<unique_ptr<Instruction>> instructions(InsArraySize::TWO);
    RankId remote = sendReduce.GetRemoteRank();
    const LinkData link = sendReduce.GetLink();
    u32 index = 0;

    instructions[index++] = make_unique<InsPostReady>(remote, link);
    instructions[index++] = make_unique<InsWaitFin>(remote, link);
    AppendInsPostFinAck(remote, link, instructions);

    return instructions;
}

inline vector<unique_ptr<Instruction>> PrimSendReduceInWriteModeWithoutInlineReduce(const PrimSendReduce& sendReduce)
{
    vector<unique_ptr<Instruction>> instructions(InsArraySize::TWO + sendReduce.Size());
    RankId remote = sendReduce.GetRemoteRank();
    const LinkData link = sendReduce.GetLink();
    u32 index = 0;

    instructions[index++] = make_unique<InsWaitReady>(remote, link);
    for (u32 pos = 0; pos < sendReduce.Size(); pos++) {
        instructions[index++]
            = make_unique<InsWrite>(remote, link, sendReduce.GetLocalSlice(pos), sendReduce.GetRemoteSrcSlice(pos));
    }
    instructions[index++] = make_unique<InsPostFin>(remote, link);
    AppendInsWaitFinAck(remote, link, instructions);

    return instructions;
}

inline vector<unique_ptr<Instruction>> PrimRecvReduceInReadModeWithInlineReduce(const PrimRecvReduce& recvReduce)
{
    vector<unique_ptr<Instruction>> instructions(InsArraySize::TWO + recvReduce.Size());
    RankId remote = recvReduce.GetRemoteRank();
    const LinkData link = recvReduce.GetLink();
    u32 index = 0;

    instructions[index++] = make_unique<InsWaitReady>(remote, link);
    for (u32 pos = 0; pos < recvReduce.Size(); pos++) {
        instructions[index++] = make_unique<InsReadReduce>(
            remote, link, recvReduce.GetLocalDstSlice(pos), recvReduce.GetRemoteSlice(pos), recvReduce.GetDataType(),
            recvReduce.GetReduceOp());
    }
    instructions[index++] = make_unique<InsPostFin>(remote, link);
    AppendInsWaitFinAck(remote, link, instructions);

    return instructions;
}

inline vector<unique_ptr<Instruction>> PrimRecvReduceInWriteModeWithInlineReduce(const PrimRecvReduce& recvReduce)
{
    vector<unique_ptr<Instruction>> instructions(InsArraySize::TWO);
    RankId remote = recvReduce.GetRemoteRank();
    const LinkData link = recvReduce.GetLink();
    u32 index = 0;

    instructions[index++] = make_unique<InsPostReady>(remote, link);
    instructions[index++] = make_unique<InsWaitFin>(remote, link);
    AppendInsPostFinAck(remote, link, instructions);

    return instructions;
}

inline vector<unique_ptr<Instruction>> PrimRecvReduceInReadModeWithoutInlineReduce(const PrimRecvReduce& recvReduce)
{
    vector<unique_ptr<Instruction>> instructions(0);
    RankId remote = recvReduce.GetRemoteRank();
    const LinkData link = recvReduce.GetLink();

    instructions.push_back(make_unique<InsWaitReady>(remote, link));

    for (u32 pos = 0; pos < recvReduce.Size(); pos++) {
        instructions.push_back(
            make_unique<InsRead>(remote, link, recvReduce.GetLocalSrcSlice(pos), recvReduce.GetRemoteSlice(pos)));
    }

    instructions.push_back(make_unique<InsPostFin>(remote, link));
    AppendInsWaitFinAck(remote, link, instructions);

    for (u32 pos = 0; pos < recvReduce.Size(); pos++) {
        instructions.push_back(make_unique<InsLocalReduce>(
            recvReduce.GetLocalSrcSlice(pos), recvReduce.GetLocalDstSlice(pos), recvReduce.GetDataType(),
            recvReduce.GetReduceOp()));
    }

    return instructions;
}

inline vector<unique_ptr<Instruction>> PrimRecvReduceInWriteModeWithoutInlineReduce(const PrimRecvReduce& recvReduce)
{
    vector<unique_ptr<Instruction>> instructions(0);
    RankId remote = recvReduce.GetRemoteRank();
    const LinkData link = recvReduce.GetLink();

    instructions.push_back(make_unique<InsPostReady>(remote, link));
    instructions.push_back(make_unique<InsWaitFin>(remote, link));
    AppendInsPostFinAck(remote, link, instructions);

    for (u32 pos = 0; pos < recvReduce.Size(); pos++) {
        instructions.push_back(make_unique<InsLocalReduce>(
            recvReduce.GetLocalSrcSlice(pos), recvReduce.GetLocalDstSlice(pos), recvReduce.GetDataType(),
            recvReduce.GetReduceOp()));
    }
    return instructions;
}

vector<unique_ptr<Instruction>> Translate(const PrimPostTo& postTo)
{
    vector<unique_ptr<Instruction>> instructions(InsArraySize::ONE);
    u32 waitQid = postTo.GetQid();

    instructions[InsArrayIndex::ZERO]
        = make_unique<InsLocalPostTo>(waitQid, postTo.GetNotifyType(), postTo.GetTopicId());
    return instructions;
}

vector<unique_ptr<Instruction>> Translate(const PrimWaitFrom& waitFrom)
{
    vector<unique_ptr<Instruction>> instructions(InsArraySize::ONE);
    u32 postQid = waitFrom.GetQid();

    instructions[InsArrayIndex::ZERO]
        = make_unique<InsLocalWaitFrom>(postQid, NotifyType::NORMAL, waitFrom.GetTopicId());
    return instructions;
}

vector<unique_ptr<Instruction>> Translate(const PrimWaitGroup& waitGroup)
{
    vector<unique_ptr<Instruction>> instructions(InsArraySize::ONE);
    auto insLocalWaitGroup = make_unique<InsLocalWaitGroup>(waitGroup.GetTopicId());
    for (auto iter = waitGroup.Iter(); iter.HasNext(); ++iter) {
        insLocalWaitGroup->Append(*iter);
    }
    instructions[InsArrayIndex::ZERO] = std::move(insLocalWaitGroup);

    return instructions;
}

vector<unique_ptr<Instruction>> Translate(const PrimLocalReduce& localReduce)
{
    vector<unique_ptr<Instruction>> instructions(InsArraySize::ONE);
    instructions[InsArrayIndex::ZERO] = make_unique<InsLocalReduce>(
        localReduce.GetSrcSlice(), localReduce.GetDstSlice(), localReduce.GetDataType(), localReduce.GetReduceOp());
    return instructions;
}

vector<unique_ptr<Instruction>> Translate(const PrimLocalCopy& localCopy)
{
    vector<unique_ptr<Instruction>> instructions(InsArraySize::ONE);

    instructions[InsArrayIndex::ZERO] = make_unique<InsLocalCopy>(localCopy.GetSrcSlice(), localCopy.GetDstSlice());
    return instructions;
}

vector<unique_ptr<Instruction>> Translate(const PrimSend& send)
{
    if (send.Size() == 0) {
        vector<unique_ptr<Instruction>> instructions(0);
        return instructions;
    }
    CheckLinkIsValid(send.GetLink(), send.Describe());
    auto dmaMode = send.GetDmaMode();
    if (dmaMode == DmaMode::PUT) {
        return PrimSendInWriteMode(send);
    } else if (dmaMode == DmaMode::GET) {
        return PrimSendInReadMode(send);
    } else {
        if (send.GetLink().GetType() == PortDeploymentType::P2P) {
            return PrimSendInReadMode(send);
        } else {
            return PrimSendInWriteMode(send);
        }
    }
}

vector<unique_ptr<Instruction>> Translate(const PrimRecv& recv)
{
    if (recv.Size() == 0) {
        vector<unique_ptr<Instruction>> instructions(0);
        return instructions;
    }
    CheckLinkIsValid(recv.GetLink(), recv.Describe());
    auto dmaMode = recv.GetDmaMode();
    if (dmaMode == DmaMode::PUT) {
        return PrimRecvInWriteMode(recv);
    } else if (dmaMode == DmaMode::GET) {
        return PrimRecvInReadMode(recv);
    } else {
        if (recv.GetLink().GetType() == PortDeploymentType::P2P) {
            return PrimRecvInReadMode(recv);
        } else {
            return PrimRecvInWriteMode(recv);
        }
    }
}

vector<unique_ptr<Instruction>> TranslateWithInlineReduce(const PrimSendReduce& sendReduce)
{
    auto dmaMode = sendReduce.GetDmaMode();
    if (dmaMode == DmaMode::PUT) {
        return PrimSendReduceInWriteModeWithInlineReduce(sendReduce);
    } else if (dmaMode == DmaMode::GET) {
        return PrimSendReduceInReadModeWithInlineReduce(sendReduce);
    } else {
        if (sendReduce.GetLink().GetType() == PortDeploymentType::P2P) {
            return PrimSendReduceInReadModeWithInlineReduce(sendReduce);
        } else {
            return PrimSendReduceInWriteModeWithInlineReduce(sendReduce);
        }
    }
}

vector<unique_ptr<Instruction>> TranslateWithoutInlineReduce(const PrimSendReduce& sendReduce)
{
    auto dmaMode = sendReduce.GetDmaMode();
    if (dmaMode == DmaMode::PUT) {
        return PrimSendReduceInWriteModeWithoutInlineReduce(sendReduce);
    } else if (dmaMode == DmaMode::GET) {
        return PrimSendReduceInReadModeWithoutInlineReduce(sendReduce);
    } else {
        if (sendReduce.GetLink().GetType() == PortDeploymentType::P2P) {
            return PrimSendReduceInReadModeWithoutInlineReduce(sendReduce);
        } else {
            return PrimSendReduceInWriteModeWithoutInlineReduce(sendReduce);
        }
    }
}

vector<unique_ptr<Instruction>> Translate(const PrimSendReduce& sendReduce)
{
    if (sendReduce.Size() == 0) {
        vector<unique_ptr<Instruction>> instructions(0);
        return instructions;
    }
    CheckLinkIsValid(sendReduce.GetLink(), sendReduce.Describe());
    if (IsSupportInlineReduce(sendReduce.GetDataType(), sendReduce.GetReduceOp(), sendReduce.GetLink())) {
        return TranslateWithInlineReduce(sendReduce);
    } else {
        return TranslateWithoutInlineReduce(sendReduce);
    }
}

vector<unique_ptr<Instruction>> TranslateWithInlineReduce(const PrimRecvReduce& recvReduce)
{
    auto dmaMode = recvReduce.GetDmaMode();
    if (dmaMode == DmaMode::PUT) {
        return PrimRecvReduceInWriteModeWithInlineReduce(recvReduce);
    } else if (dmaMode == DmaMode::GET) {
        return PrimRecvReduceInReadModeWithInlineReduce(recvReduce);
    } else {
        if (recvReduce.GetLink().GetType() == PortDeploymentType::P2P) {
            return PrimRecvReduceInReadModeWithInlineReduce(recvReduce);
        } else {
            return PrimRecvReduceInWriteModeWithInlineReduce(recvReduce);
        }
    }
}

vector<unique_ptr<Instruction>> TranslateWithoutInlineReduce(const PrimRecvReduce& recvReduce)
{
    auto dmaMode = recvReduce.GetDmaMode();
    if (dmaMode == DmaMode::PUT) {
        return PrimRecvReduceInWriteModeWithoutInlineReduce(recvReduce);
    } else if (dmaMode == DmaMode::GET) {
        return PrimRecvReduceInReadModeWithoutInlineReduce(recvReduce);
    } else {
        if (recvReduce.GetLink().GetType() == PortDeploymentType::P2P) {
            return PrimRecvReduceInReadModeWithoutInlineReduce(recvReduce);
        } else {
            return PrimRecvReduceInWriteModeWithoutInlineReduce(recvReduce);
        }
    }
}

vector<unique_ptr<Instruction>> Translate(const PrimRecvReduce& recvReduce)
{
    if (recvReduce.Size() == 0) {
        vector<unique_ptr<Instruction>> instructions(0);
        return instructions;
    }
    CheckLinkIsValid(recvReduce.GetLink(), recvReduce.Describe());
    if (IsSupportInlineReduce(recvReduce.GetDataType(), recvReduce.GetReduceOp(), recvReduce.GetLink())) {
        return TranslateWithInlineReduce(recvReduce);
    } else {
        return TranslateWithoutInlineReduce(recvReduce);
    }
}

bool CompareInsRule(pair<unique_ptr<Instruction>, int>& insA, pair<unique_ptr<Instruction>, int>& insB)
{
    if (INSTRUCTION_PRI_MAP.at(insA.first->GetType()) == INSTRUCTION_PRI_MAP.at(insB.first->GetType())) {
        return insA.second < insB.second;
    } else {
        return INSTRUCTION_PRI_MAP.at(insA.first->GetType()) > INSTRUCTION_PRI_MAP.at(insB.first->GetType());
    }
}

vector<unique_ptr<Instruction>> GenerateTempInstruction(const PrimGroup& group)
{
    vector<unique_ptr<Instruction>> instructions;
    vector<unique_ptr<Instruction>> generateVec;
    group.CheckValid();
    for (auto iter = group.Iter(); iter.HasNext(); ++iter) {
        if (iter->GetType() == PrimType::SEND) {
            generateVec = Translate(static_cast<const PrimSend&>(*iter));
            instructions.insert(
                instructions.end(), make_move_iterator(generateVec.begin()), make_move_iterator(generateVec.end()));
        } else if (iter->GetType() == PrimType::RECV) {
            generateVec = Translate(static_cast<const PrimRecv&>(*iter));
            instructions.insert(
                instructions.end(), make_move_iterator(generateVec.begin()), make_move_iterator(generateVec.end()));
        } else if (iter->GetType() == PrimType::SEND_REDUCE) {
            generateVec = Translate(static_cast<const PrimSendReduce&>(*iter));
            instructions.insert(
                instructions.end(), make_move_iterator(generateVec.begin()), make_move_iterator(generateVec.end()));
        } else if (iter->GetType() == PrimType::RECV_REDUCE) {
            generateVec = Translate(static_cast<const PrimRecvReduce&>(*iter));
            instructions.insert(
                instructions.end(), make_move_iterator(generateVec.begin()), make_move_iterator(generateVec.end()));
        }
    }
    return instructions;
}

vector<unique_ptr<Instruction>> Translate(const PrimGroup& group)
{
    vector<unique_ptr<Instruction>> tempInstruction = GenerateTempInstruction(group);
    vector<pair<unique_ptr<Instruction>, int>> pairInstructions;
    for (size_t i = 0; i < tempInstruction.size(); i++) {
        pairInstructions.emplace_back(std::move(tempInstruction[i]), i);
    }
    sort(pairInstructions.begin(), pairInstructions.end(), CompareInsRule);
    vector<unique_ptr<Instruction>> instructions;
    for (auto& pairInstruction : pairInstructions) {
        instructions.push_back(std::move(pairInstruction.first));
    }
    return instructions;
}
} // namespace Hccl
