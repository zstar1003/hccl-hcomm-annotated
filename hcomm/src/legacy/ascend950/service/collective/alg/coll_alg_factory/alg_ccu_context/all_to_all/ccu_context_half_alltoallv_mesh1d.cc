/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <random>
#include <algorithm>
#include "ccu_context_half_alltoallv_mesh1d.h"
#include "ccu_instruction_half_alltoallv_mesh1d.h"

namespace Hccl {

constexpr uint16_t RANK_NUM_PER_CKE = 16; // 本rank给远端置位时应当写的CKE，16个对端一个CKE

void CcuContextHalfAllToAllVMesh1D::ExchangeCtxResource()
{
    if (missionId_ == 1) {
        ExportVariable(userInAddr_, ctxName_ + "_UserInAddr_" + std::to_string(missionId_));
        ExportVariable(sendSizeAddr_, ctxName_ + "_SendSizeAddr_" + std::to_string(missionId_));
        ExportVariable(sendOffsetAddr_, ctxName_ + "_SendOffsetAddr__" + std::to_string(missionId_));
        ExportVariable(recvOffset_, ctxName_ + "_RecvOffset_" + std::to_string(missionId_));
    } else {
        anoUserInAddr_ = ImportVariable(ctxName_ + "_UserInAddr_" + std::to_string(1 - missionId_));
        anoSendSizeAddr_ = ImportVariable(ctxName_ + "_SendSizeAddr_" + std::to_string(1 - missionId_));
        anoSendOffsetAddr_ = ImportVariable(ctxName_ + "_SendOffsetAddr__" + std::to_string(1 - missionId_));
        anoRecvOffset_ = ImportVariable(ctxName_ + "_RecvOffset_" + std::to_string(1 - missionId_));
    }
    ExportMaskSignal(locMiSignal0_, ctxName_ + "_MiSync0_" + std::to_string(missionId_));
    anoMiSignal0_ = ImportMaskSignal(ctxName_ + "_MiSync0_" + std::to_string(1 - missionId_));
    ExportMaskSignal(locMiSignal1_, ctxName_ + "_MiSync1_" + std::to_string(missionId_));
    anoMiSignal1_ = ImportMaskSignal(ctxName_ + "_MiSync1_" + std::to_string(1 - missionId_));

    return;
}

CcuContextHalfAllToAllVMesh1D::CcuContextHalfAllToAllVMesh1D(
    const CcuCtxArg& arg, const std::vector<CcuTransport*>& transports, const CcuTransportGroup& group)
    : CcuContextAlgBase(arg, transports, group)
{
    const CcuCtxArgHalfAllToAllVMesh1D* ctxArg = dynamic_cast<const CcuCtxArgHalfAllToAllVMesh1D*>(&arg);
    if (ctxArg == nullptr) {
        THROW<NullPtrException>(StringFormat("CcuContextHalfAllToAllVMesh1D::ctxArg ptr is null"));
    }
    ctxName_ = ctxArg->GetCtxSignature().Describe();
    rankId_ = ctxArg->rankId;
    if (ctxArg->dimSize.size() > 0) {
        rankSize_ = ctxArg->dimSize[0];
    }
    missionId_ = ctxArg->missionId;
    signalNum_ = (rankSize_ + RANK_NUM_PER_CKE - 1) / RANK_NUM_PER_CKE; // 每个CKE有16个bit
    myCclBufferAddr_ = ctxArg->cclBufferAddr;

    userInAddr_ = CreateVariable();
    sendSizeAddr_ = CreateVariable();
    sendOffsetAddr_ = CreateVariable();
    recvOffset_ = CreateVariable();
    goSize_ = CreateGroupOpSize();

    curSrc_ = CreateMemory();
    if (transports.size() == 0 || transports.size() < rankSize_ - 1) {
        THROW<NullPtrException>(StringFormat("CcuContextHalfAllToAllVMesh1D transports is empty or size is less"));
    }
    for (uint32_t i = 0; i < rankSize_; i++) {
        // curDst在初始化时即赋值各个rank的cclBuffer地址
        if (i == rankId_) {
            curDst_.emplace_back(CreateMemory());
        } else {
            uint16_t transIdx = (i < rankId_) ? i : i - 1;
            curDst_.emplace_back(GetRmtBuffer(*transports[transIdx], 0));
        }
        token_.emplace_back(CreateVariable());
        sendSizeA_.emplace_back(CreateVariable());
        sendSizeB_.emplace_back(CreateVariable());
        sendOffsetA_.emplace_back(CreateVariable());
        sendOffsetB_.emplace_back(CreateVariable());
    }
    ccuStartSignal_ = CreateMaskSignal();
    ccuEndSignal_ = CreateMaskSignal();
    for (uint32_t i = 0; i < signalNum_; i++) {
        writeDoneSignal_.emplace_back(CreateMaskSignal());
    }

    // mission间交互的资源
    locMiSignal0_ = CreateMaskSignal();
    locMiSignal1_ = CreateMaskSignal();
    ExchangeCtxResource();

    return;
}

void CcuContextHalfAllToAllVMesh1D::LoadArgs()
{
    if (missionId_ == 0) {
        Load(userInAddr_);
        Load(sendSizeAddr_);
        Load(token_[rankId_]);
        Load(sendOffsetAddr_);
        Load(goSize_);
        Load(recvOffset_);

        // Mi0将参数同步给Mi1
        LocalCtxPostVar(userInAddr_, anoUserInAddr_, anoMiSignal0_, 1 << 0);         // 用第1个bit标记
        LocalCtxPostVar(sendSizeAddr_, anoSendSizeAddr_, anoMiSignal0_, 1 << 1);     // 用第2个
        LocalCtxPostVar(sendOffsetAddr_, anoSendOffsetAddr_, anoMiSignal0_, 1 << 2); // 用第3个
        LocalCtxPostVar(recvOffset_, anoRecvOffset_, anoMiSignal0_, 1 << 3);         // 用第4个
    } else {
        LocalWait(locMiSignal0_, (1 << 4) - 1); // 共同步4个参数
    }
    return;
}

void CcuContextHalfAllToAllVMesh1D::LoadArgsFromMem()
{
    // 暂不支持用单条指令加载多个参数
    CcuRep::Variable dataLength = CreateVariable();
    CcuRep::Variable tempAddr = CreateVariable();
    LoadArgs();

    // 加载本端的cclbuffer地址
    curDst_[rankId_].addr = myCclBufferAddr_;

    // 连续加载rankSize * 2个sendSize
    dataLength = 8; // 每个Xn占8个byte

    u32 argsCount = sendSizeA_.size() + sendSizeB_.size() + sendOffsetA_.size() + sendOffsetB_.size();
    std::vector<CcuRep::Variable> tempArgs(argsCount);
    HCCL_INFO("CcuContextHalfAllToAllVMesh1D LoadArgsFromMem, argsCount:[%u]", argsCount);

    for (uint32_t i = 0; i < tempArgs.size(); ++i) {
        tempArgs[i] = CreateContinuousVariable();
    }
    LoadVariable(sendSizeAddr_, tempArgs[0], argsCount);

    u32 argIdx = 0;
    for (uint32_t i = 0; i < sendSizeA_.size(); i++) {
        sendSizeA_[i] = tempArgs[argIdx];
        argIdx++;
    }
    for (uint32_t i = 0; i < sendSizeB_.size(); i++) {
        sendSizeB_[i] = tempArgs[argIdx];
        argIdx++;
    }
    for (uint32_t i = 0; i < sendOffsetA_.size(); i++) {
        sendOffsetA_[i] = tempArgs[argIdx];
        argIdx++;
    }
    for (uint32_t i = 0; i < sendOffsetB_.size(); i++) {
        sendOffsetB_[i] = tempArgs[argIdx];
        argIdx++;
    }
    return;
}

void CcuContextHalfAllToAllVMesh1D::MissionSync(uint32_t signalIndex)
{
    const uint32_t MISSION_NUM = 2;
    if (signalIndex > 1) {
        THROW<InvalidParamsException>(
            StringFormat("[CcuContextHalfAllToAllVMesh1D] Unexpected SignalIndex[%u]", signalIndex));
    }
    LocalCtxPost(anoMiSignal1_, 1 << (missionId_ + signalIndex * MISSION_NUM));
    LocalWait(locMiSignal1_, 1 << (1 - missionId_ + signalIndex * MISSION_NUM));
    return;
}

void CcuContextHalfAllToAllVMesh1D::PostSync()
{
    if (missionId_ == 0) {
        uint16_t signalId = rankId_ / RANK_NUM_PER_CKE;
        uint16_t selfBit = 1 << (rankId_ % RANK_NUM_PER_CKE);
        for (auto t : transports) {
            if (t == nullptr) {
                THROW<NullPtrException>(StringFormat("CcuContextHalfAllToAllVMesh1D::Algorithm transport ptr is null"));
            }
            RemotePost(*t, signalId, selfBit);
        }

        for (uint16_t sId = 0; sId < signalNum_; sId++) {
            uint32_t waitBit;
            if (sId != signalNum_ - 1) {
                waitBit = (1 << RANK_NUM_PER_CKE) - 1; // 等待全部16个peer
            } else {
                waitBit = ((1 << (rankSize_ - (signalNum_ - 1) * RANK_NUM_PER_CKE)) - 1);
            }
            if (sId == signalId) {
                waitBit &= ~selfBit; // 如果这个CKE上有自己对应的bit，设为0
            }
            GroupWait(*transportGroup, sId, waitBit);
        }
    }
    return;
}

void CcuContextHalfAllToAllVMesh1D::Algorithm()
{
    HCCL_INFO("[CcuContextHalfAllToAllVMesh1D] AllgatherMesh1D Algorithm Begins.");
    LoadArgsFromMem();

    // 向每个对端发送数据
    CcuRep::Memory lgSrc = CreateMemory();
    CcuRep::Memory lgDst = CreateMemory();
    CcuRep::Variable tempCount = CreateVariable();
    lgSrc.token = token_[rankId_];
    lgDst.token = token_[rankId_];
    curSrc_.token = token_[rankId_];

    for (uint32_t peerId = 0; peerId < rankSize_; peerId++) {
        CcuRep::Variable& curCount = missionId_ == 0 ? sendSizeA_[peerId] : sendSizeB_[peerId];
        CcuRep::Variable& curOffset = missionId_ == 0 ? sendOffsetA_[peerId] : sendOffsetB_[peerId];
        uint16_t peerSignalId = peerId / RANK_NUM_PER_CKE;
        uint16_t peerBit = 1 << (peerId % RANK_NUM_PER_CKE);

        tempCount = curCount;
        curSrc_.addr = userInAddr_;
        curSrc_.addr += curOffset;
        curDst_[peerId].addr += recvOffset_;
        if (missionId_ == 1) {
            curDst_[peerId].addr += sendSizeA_[peerId]; // Mi1的dst需要加chunkOffset
        }
        if (peerId == rankId_) {
            lgSrc.addr = curSrc_.addr;
            lgDst.addr = curDst_[peerId].addr;
            LocalPost(writeDoneSignal_[peerSignalId], peerBit);
        } else {
            uint16_t transIdx = (peerId < rankId_) ? peerId : peerId - 1;
            CCU_IF(tempCount != 0)
            {
                Write(
                    *(transports[transIdx]), curDst_[peerId], curSrc_, tempCount, writeDoneSignal_[peerSignalId],
                    peerBit);
            }
            CCU_IF(tempCount == 0) { LocalPost(writeDoneSignal_[peerSignalId], peerBit); }
        }
    }

    if (missionId_ == 0) {
        GroupCopy(lgDst, lgSrc, goSize_);
    }
    for (uint16_t sId = 0; sId < signalNum_; sId++) {
        uint32_t waitBit;
        if (sId != signalNum_ - 1) {
            waitBit = (1 << RANK_NUM_PER_CKE) - 1; // 等待全部16个peer
        } else {
            waitBit = ((1 << (rankSize_ - (signalNum_ - 1) * RANK_NUM_PER_CKE)) - 1);
        }
        LocalWait(writeDoneSignal_[sId], waitBit);
    }

    MissionSync(0);
    PostSync();
    HCCL_INFO("[CcuContextHalfA2AUnions] Algorithm Ends.");
    return;
}

std::vector<uint64_t> CcuContextHalfAllToAllVMesh1D::GeneArgs(const CcuTaskArg& arg)
{
    (void)arg;
    std::vector<uint64_t> args = {};
    uint64_t argNum = missionId_ == 0 ? 9 : 0; // Mi0有9个Load，Mi1不做Load
    for (uint32_t i = 0; i < argNum; i++) {
        args.emplace_back(0);
    }
    return args;
}
} // namespace Hccl
