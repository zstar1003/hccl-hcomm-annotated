/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_context_all_to_all_v_mesh1d.h"
#include "ccu_instruction_all_to_all_v_mesh1d.h"

namespace Hccl {
constexpr int OUTPUT_XN_ID = 0;
constexpr int TOKEN_XN_ID = 1;
constexpr int CKE_IDX_0 = 0;
constexpr int CKE_IDX_1 = 1;
constexpr int CKE_IDX_2 = 2;

CcuContextAllToAllVMesh1D::CcuContextAllToAllVMesh1D(
    const CcuCtxArg& arg, const std::vector<CcuTransport*>& transports, const CcuTransportGroup& group)
    : CcuContextAlgBase(arg, transports, group)
{
    const CcuCtxArgAllToAllVMesh1D* ctxArg = dynamic_cast<const CcuCtxArgAllToAllVMesh1D*>(&arg);
    if (ctxArg == nullptr) {
        THROW<NullPtrException>(StringFormat("CcuContextAllToAllVMesh1D::ctxArg ptr is null"));
    }
    rankId_ = ctxArg->rankId;
    if (ctxArg->dimSize.size() > 0) {
        rankSize_ = ctxArg->dimSize[0];
    }
    loadFromMem = ctxArg->loadFromMem;

    if (transports.size() == 0) {
        THROW<NullPtrException>(StringFormat("CcuContextAllToAllVMesh1D transports is empty"));
    }
}

void CcuContextAllToAllVMesh1D::PreSync()
{
    CcuRep::Variable tempDst = CreateVariable();
    u32 transportId = 0;
    for (u32 id = 0; id < rankSize_; id++) {
        if (id == rankId_) {
            continue;
        }
        tempDst = output_[rankId_];
        tempDst += sendRecvInfo_[id].recvOffset;
        // index = 0，传递output信息
        WriteVariableWithSignal(*transports[transportId], tempDst, OUTPUT_XN_ID, CKE_IDX_1, selfBit_);
        // index = 1，传递token信息
        WriteVariableWithSignal(*transports[transportId], token_[rankId_], TOKEN_XN_ID, CKE_IDX_2, selfBit_);
        transportId++;
    }

    GroupWait(*transportGroup, CKE_IDX_1, allOtherBit_); // index = 1，传递output信息
    GroupWait(*transportGroup, CKE_IDX_2, allOtherBit_); // index = 2，传递token信息
}

void CcuContextAllToAllVMesh1D::PostSync()
{
    for (auto t : transports) {
        if (t == nullptr) {
            THROW<NullPtrException>(StringFormat("CcuContextAllToAllVMesh1D::Algorithm transport ptr is null"));
        }
        RemotePost(*t, CKE_IDX_0, selfBit_);
    }
    GroupWait(*transportGroup, CKE_IDX_0, allOtherBit_);
}

void CcuContextAllToAllVMesh1D::CreateVariables()
{
    u32 transportId = 0;
    input_.push_back(CreateVariable());
    output_.reserve(rankSize_);
    token_.reserve(rankSize_);
    for (u32 id = 0; id < rankSize_; id++) {
        if (id == rankId_) {
            output_.push_back(CreateVariable());
            token_.push_back(CreateVariable());
        } else { // 非本地，使用远端Variable
            CHK_PRT_RET(
                transports[transportId] == nullptr || transportId >= transports.size(),
                HCCL_ERROR(
                    "[CcuContextAllToAllVMesh1D] Algorithm transport ptr is null or transportIdx is out of bounds"), );
            output_.push_back(CreateVariable((*transports[transportId]), OUTPUT_XN_ID)); // 与远端交换本卡的接收地址
            token_.push_back(CreateVariable((*transports[transportId]), TOKEN_XN_ID));
            transportId++;
        }
    }

    src_.reserve(rankSize_);
    dst_.reserve(rankSize_);
    for (uint32_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        src_.push_back(CreateMemory());
        dst_.push_back(CreateMemory());
    }

    srcOffset_ = CreateVariable();
    dstOffset_ = CreateVariable();
    a2avXnAddr_ = CreateVariable();

    // 前同步。交换信息，将本Rank load的in\out等地址信息写到所有对端的对应Variable中，并同步
    selfBit_ = 1 << rankId_;                                   // 本rank的mask
    allBit_ = (1 << rankSize_) - 1;                            // 等待包含自身的全部对端
    allOtherBit_ = ((1 << rankSize_) - 1) & (~(1 << rankId_)); // 等待其他所有对端

    locMask_ = CreateMaskSignal();
    //  all2allv 数据搬运
    completedRankCount_ = CreateVariable();
    xnMaxTransportSize_ = CreateVariable();
    xnMaxTransportGoSize_ = CreateGroupOpSize();
    localTailGoSize_ = CreateGroupOpSize();
    xnConst1_ = CreateVariable();

    xnLength_ = CreateVariable();
    xnLength_ = 8; // xn长度为8byte
}

void CcuContextAllToAllVMesh1D::LoadArgs()
{
    // 从SQE load args，本rank需要的input、output地址等信息
    // inputAddr, outputAddr, tokenInfo, srcStride, dstStride, srcOffset, dstOffset
    Load(input_[0]);
    Load(output_[rankId_]); // load的目的存放寄存器
    Load(token_[rankId_]);
    Load(srcOffset_);
    Load(dstOffset_);
    Load(localTailGoSize_);
    if (loadFromMem) {
        Load(a2avXnAddr_);
    } else {
        Load(xnMaxTransportGoSize_);
    }

    // 恢复当前卡对所有卡的收发信息
    sendRecvInfo_.resize(rankSize_);
    for (uint32_t i = 0; i < rankSize_; i++) {
        sendRecvInfo_[i].tailSize = CreateVariable();
        sendRecvInfo_[i].loopNum = CreateVariable();
        sendRecvInfo_[i].sendOffset = CreateVariable();
        sendRecvInfo_[i].recvOffset = CreateVariable();
    }
    LoadAll2allSendRecvInfo(sendRecvInfo_);
}

void CcuContextAllToAllVMesh1D::CalcGroupSrcDst()
{
    for (uint32_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        src_[rankIdx].token = token_[rankIdx];
        dst_[rankIdx].token = token_[rankIdx];

        // src_[rankIdx] = usrInAddr + sendoffset + srcOffset_
        src_[rankIdx].addr = input_[0];
        src_[rankIdx].addr += sendRecvInfo_[rankIdx].sendOffset;
        src_[rankIdx].addr += srcOffset_;

        // dst_[r] = recvBuf[r] + recvOffset + dstOffset_
        if (rankIdx == rankId_) {
            // 写目的端为本端时需要特殊处理：使用接收基地址 + 块地址offset + 已发送数据量
            dst_[rankIdx].addr = output_[rankId_];
            dst_[rankIdx].addr += sendRecvInfo_[rankIdx].recvOffset;
            dst_[rankIdx].addr += dstOffset_;
        } else {
            // 对端交换的接收块起始地址 + 已接收的数据偏移
            dst_[rankIdx].addr = output_[rankIdx];
            dst_[rankIdx].addr += dstOffset_;
        }
    }
}

void CcuContextAllToAllVMesh1D::DoAll2AllVMultiLoop()
{
    HCCL_DEBUG("[CcuContextAllToAllVMesh1D] alltoallv mesh 1d use GroupCopy start");
    xnMaxTransportSize_ = UB_MAX_TRANS_SIZE;
    completedRankCount_ = 0;
    xnConst1_ = 1;
    u32 transportId = 0;
    CCU_WHILE(completedRankCount_ != rankSize_)
    { // 循环发送数据，直到所有对端数据都发送完成
        for (uint32_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) { // 循环发送所有对端数据
            if (rankIdx == rankId_) {
                continue;
            }
            CCU_IF(sendRecvInfo_[rankIdx].loopNum == UINT64_MAX)
            { // 已经完成，直接置位完成信号
                LocalPost(locMask_, (1 << rankIdx));
            }
            CCU_IF(sendRecvInfo_[rankIdx].loopNum != UINT64_MAX)
            { // 还没有完成，则继续循环
                CCU_IF(sendRecvInfo_[rankIdx].loopNum == UINT64_MAX - 1)
                { // 最后一轮循环, 发送尾块数据
                    CCU_IF(sendRecvInfo_[rankIdx].tailSize == 0)
                    { // 尾块数据量为 0，则不需要发送尾块数据
                        LocalPost(locMask_, (1 << rankIdx));
                    }
                    CCU_IF(sendRecvInfo_[rankIdx].tailSize != 0)
                    { // 尾块数据量不为 0，则需要发送尾块数据
                        Write(
                            *transports[transportId], dst_[rankIdx], src_[rankIdx], sendRecvInfo_[rankIdx].tailSize,
                            locMask_, 1 << rankIdx);
                    }
                    completedRankCount_ += xnConst1_; // 之后一轮循环完成，更新已完成的rank数
                }
                CCU_IF(sendRecvInfo_[rankIdx].loopNum != UINT64_MAX - 1)
                { // 未完成，则继续循环，发送整块数据
                    Write(
                        *transports[transportId], dst_[rankIdx], src_[rankIdx], xnMaxTransportSize_, locMask_,
                        1 << rankIdx);
                    // 更新偏移
                    src_[rankIdx].addr += xnMaxTransportSize_;
                    dst_[rankIdx].addr += xnMaxTransportSize_;
                }
                sendRecvInfo_[rankIdx].loopNum += xnConst1_;
            }
            transportId++;
        }
        CCU_IF(sendRecvInfo_[rankId_].loopNum == UINT64_MAX)
        { // 已经完成，直接置位完成信号
            LocalPost(locMask_, (1 << rankId_));
        }

        CCU_IF(sendRecvInfo_[rankId_].loopNum != UINT64_MAX)
        { // 还没有完成，则继续循环
            CCU_IF(sendRecvInfo_[rankId_].loopNum == UINT64_MAX - 1)
            { // 最后一轮循环, 发送尾块数据
                CCU_IF(sendRecvInfo_[rankId_].tailSize == 0)
                { // 尾块数据量为 0，则不需要发送尾块数据
                    LocalPost(locMask_, (1 << rankId_));
                }
                CCU_IF(sendRecvInfo_[rankId_].tailSize != 0)
                { // 尾块数据量不为 0，则需要发送尾块数据
                    GroupCopy(dst_[rankId_], src_[rankId_], localTailGoSize_);
                    LocalPost(locMask_, 1 << rankId_);
                }
                completedRankCount_ += xnConst1_; // 之后一轮循环完成，更新已完成的rank数
            }
            CCU_IF(sendRecvInfo_[rankId_].loopNum != UINT64_MAX - 1)
            { // 未完成，则继续循环，发送整块数据
                GroupCopy(dst_[rankId_], src_[rankId_], xnMaxTransportGoSize_);
                LocalPost(locMask_, 1 << rankId_);
                // 更新偏移
                src_[rankId_].addr += xnMaxTransportSize_;
                dst_[rankId_].addr += xnMaxTransportSize_;
            }
            sendRecvInfo_[rankId_].loopNum += xnConst1_;
        }
        // 等待本轮发送完成
        LocalWait(locMask_, allBit_);
    }
}

void CcuContextAllToAllVMesh1D::Algorithm()
{
    HCCL_INFO("[ccuAllToAllVMesh1D_context] AllToAllVMesh1D run");
    CreateVariables();
    LoadArgs();
    PreSync();
    // 创建GSA， src为本地的各片HBM地址GSA列表，dst为所有对端的HBM地址GSA列表
    CalcGroupSrcDst();
    DoAll2AllVMultiLoop();
    //  后同步
    PostSync();
    HCCL_INFO("[AllToAllAlgo] AllToAllMesh1D end");
    return;
}

std::vector<uint64_t> CcuContextAllToAllVMesh1D::GeneArgs(const CcuTaskArg& arg)
{
    const CcuTaskArgAllToAllVMesh1D* taskArg = dynamic_cast<const CcuTaskArgAllToAllVMesh1D*>(&arg);
    if (taskArg == nullptr) {
        THROW<NullPtrException>(StringFormat("CcuContextAllToAllVMesh1D::taskArg ptr is null"));
    }
    uint64_t inputAddr = taskArg->inputAddr_;
    uint64_t outputAddr = taskArg->outputAddr_;
    uint64_t tokenInfo = taskArg->token_;

    uint64_t srcOffset = taskArg->srcOffset_;
    uint64_t dstOffset = taskArg->dstOffset_;

    HCCL_INFO(
        "[AllToAllVAlgo] inputAddr[%llu], outputAddr[%llu],"
        "srcOffset[%llu], dstOffset[%llu]",
        inputAddr, outputAddr, srcOffset, dstOffset);
    std::vector<uint64_t> processReturn = {inputAddr, outputAddr, tokenInfo, srcOffset, dstOffset};

    u64 localTailSize = taskArg->localSendRecvInfo_.sendLength[rankId_] % UB_MAX_TRANS_SIZE;
    auto localTailGoSize = CalGoSize(localTailSize);
    for (auto val : localTailGoSize) {
        processReturn.push_back(val);
    }

    if (loadFromMem) {
        processReturn.push_back(0); // 空地址占位，保证参数个数与load个数一致
        return processReturn;
    }

    uint64_t xnMaxTransportSize = UB_MAX_TRANS_SIZE;
    HCCL_INFO("[CcuContextAllToAllVMesh1D][GeneArgs] CalGoSize size[%llu]", xnMaxTransportSize);
    auto xnMaxTransportGoSize = CalGoSize(xnMaxTransportSize);
    for (auto val : xnMaxTransportGoSize) {
        processReturn.push_back(val);
    }

    uint64_t rankSize = taskArg->sliceSize_.size();
    for (uint64_t i = 0; i < rankSize; i++) {
        uint64_t tailSize = taskArg->localSendRecvInfo_.sendLength[i] % UB_MAX_TRANS_SIZE;
        uint64_t loopNum = UINT64_MAX - 1 - (taskArg->localSendRecvInfo_.sendLength[i] / UB_MAX_TRANS_SIZE);
        uint64_t sendOffset = taskArg->localSendRecvInfo_.sendOffset[i];
        uint64_t recvOffset = taskArg->localSendRecvInfo_.recvOffset[i];
        HCCL_INFO("[CcuContextAllToAllVMesh1D][GeneArgs] CalGoSize size[%llu]", tailSize);
        processReturn.push_back(tailSize);
        processReturn.push_back(loopNum);
        processReturn.push_back(sendOffset);
        processReturn.push_back(recvOffset);
        HCCL_INFO(
            "[AllToAllVAlgo] rankIdx[i] taskArg->sliceSize[%llu],"
            "loopNum[%llu],"
            "taskArg->localSendRecvInfo.sendOffset[%llu],"
            "taskArg->localSendRecvInfo.recvOffset[%llu]",
            taskArg->sliceSize_[i], loopNum, taskArg->localSendRecvInfo_.sendOffset[i],
            taskArg->localSendRecvInfo_.recvOffset[i]);
    }
    return processReturn;
}

void CcuContextAllToAllVMesh1D::LoadAll2allSendRecvInfo(std::vector<A2AsingleSendRecvInfo>& sendRecvInfo)
{
    if (loadFromMem) {
        // 连续加载ranksize个sendSize,loopNum,sendOffset,receiveOffset
        u32 argsCount = sendRecvInfo.size() * 4;
        std::vector<CcuRep::Variable> tempArgs(argsCount);
        HCCL_INFO("AllToAllVAlgo LoadArgsFromMem, argsCount: [%u]", argsCount);
        for (uint32_t i = 0; i < tempArgs.size(); ++i) {
            tempArgs[i] = CreateContinuousVariable();
        }
        LoadVariable(a2avXnAddr_, tempArgs[0], argsCount);

        // 赋值给对应的 XN
        u32 argIdx = 0;
        for (uint32_t i = 0; i < sendRecvInfo.size(); i++) {
            sendRecvInfo[i].tailSize = tempArgs[argIdx];
            argIdx++;
            sendRecvInfo[i].loopNum = UINT64_MAX - 1;
            argIdx++;
            sendRecvInfo[i].sendOffset = tempArgs[argIdx];
            argIdx++;
            sendRecvInfo[i].recvOffset = tempArgs[argIdx];
            argIdx++;
        }
    } else {
        for (uint32_t i = 0; i < rankSize_; i++) {
            Load(sendRecvInfo[i].tailSize);
            Load(sendRecvInfo[i].loopNum);
            Load(sendRecvInfo[i].sendOffset);
            Load(sendRecvInfo[i].recvOffset);
        }
    }
}

void CcuContextAllToAllVMesh1D::RefreshArgs(
    CollOpParams opParams, u32 rankSize, std::vector<uint64_t>& args, const u32 myRank)
{
    uint64_t inputAddr;
    uint64_t outputAddr;
    uint64_t token = 0;
    uint64_t srcOffset = 0;
    uint64_t dstOffset = 0;

    inputAddr = reinterpret_cast<uint64_t>(opParams.sendBuf);
    outputAddr = reinterpret_cast<uint64_t>(opParams.recvBuf);

    args.push_back(inputAddr);
    args.push_back(outputAddr);
    args.push_back(token);
    args.push_back(srcOffset);
    args.push_back(dstOffset);

    // 配置本地拷贝的moConfig参数
    u32 loopCount = LOCAL_COPY_MS_PER_LOOP;
    u32 memSlice = CCU_MS_LOCAL_COPY_LOOP_COUNT * CcuRep::CCU_MS_SIZE;
    GroupOpConfig moConfig{CcuRep::CCU_MS_INTERLEAVE, loopCount, memSlice};

    u64 mySendCounts = *(static_cast<const u64*>(opParams.all2AllVDataDes.sendCounts) + myRank);
    u64 mySendLength = mySendCounts * DataTypeSizeGet(opParams.all2AllVDataDes.sendType);
    uint64_t localTailSize = mySendLength % UB_MAX_TRANS_SIZE;
    auto localTailGoSize = CcuContext::CalGoSizeStatic(localTailSize, moConfig);
    for (auto val : localTailGoSize) {
        args.push_back(val);
    }

    uint64_t xnMaxTransportSize = UB_MAX_TRANS_SIZE;
    HCCL_INFO("[CcuContextAllToAllVMesh1D][RefreshArgs] CalGoSizeStatic size [%llu]", xnMaxTransportSize);
    auto xnMaxTransportGoSize = CcuContext::CalGoSizeStatic(xnMaxTransportSize, moConfig);
    for (auto val : xnMaxTransportGoSize) {
        args.push_back(val);
    }

    for (u32 i = 0; i < rankSize; i++) {
        u64 curSendCounts = *(static_cast<const u64*>(opParams.all2AllVDataDes.sendCounts) + i);
        u64 curSendDispls = *(static_cast<const u64*>(opParams.all2AllVDataDes.sdispls) + i);
        u64 sendLength = curSendCounts * DataTypeSizeGet(opParams.all2AllVDataDes.sendType);
        u64 sendOffset = curSendDispls * DataTypeSizeGet(opParams.all2AllVDataDes.sendType);

        u64 curRecvDispls = *(static_cast<const u64*>(opParams.all2AllVDataDes.rdispls) + i);
        u64 recvOffset = curRecvDispls * DataTypeSizeGet(opParams.all2AllVDataDes.recvType);

        uint64_t tailSize = sendLength % UB_MAX_TRANS_SIZE;
        uint64_t loopNum = UINT64_MAX - 1 - (sendLength / UB_MAX_TRANS_SIZE);
        HCCL_INFO("[CcuContextAllToAllVMesh1D][RefreshArgs] CalGoSizeStatic size [%llu]", tailSize);

        args.push_back(tailSize);
        args.push_back(loopNum);
        args.push_back(sendOffset);
        args.push_back(recvOffset);
    }

    for (u32 i = 0; i < args.size(); i++) {
        HCCL_INFO("[CcuContextAllToAllVMesh1D][RefreshArgs] SFL args[%u] is [%llu]", i, args[i]);
    }
}
} // namespace Hccl
