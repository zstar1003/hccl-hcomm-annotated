/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_context_all_to_all_mesh2d.h"
#include "ccu_instruction_all_to_all_mesh2d.h"

namespace Hccl {

constexpr uint16_t CKE_ID_0 = 0;
constexpr uint16_t CKE_ID_1 = 1;
constexpr uint16_t CKE_ID_2 = 2;
constexpr uint16_t CKE_ID_3 = 3;
constexpr uint16_t FST_AXIS_ID = 0;
constexpr uint16_t SEC_AXIS_ID = 1;

CcuContextAlltoAllMesh2D::CcuContextAlltoAllMesh2D(
    const CcuCtxArg& arg, const std::vector<CcuTransport*>& transports, const CcuTransportGroup& group)
    : CcuContextAlgBase(arg, transports, group)
{
    goSize_ = CreateGroupOpSize();
    input = CreateVariable();
    bufferB = CreateVariable();
    sliceSize_ = CreateVariable();
    baseOffset = CreateVariable();
    firstTransportSize = CreateVariable();
    firstChunkOffset = CreateVariable();
    firstInputStrideLocal = CreateVariable();
    firstInputStrideAnother = CreateVariable();
    firstBufferOffset = CreateVariable();
    firstBufferStride = CreateVariable();
    firstOutputOffset = CreateVariable();
    secondTransportSize = CreateVariable();
    secondChunkOffset = CreateVariable();
    secondInputOffset = CreateVariable();
    secondInputStride = CreateVariable();
    secondBufferStrideLocal = CreateVariable();
    secondBufferStrideAnother = CreateVariable();
    secondOutputOffset = CreateVariable();
    secondOutputStride = CreateVariable();
    localAxisSignal = CreateMaskSignal();

    const CcuCtxArgAlltoAllMesh2D* ctxArg = dynamic_cast<const CcuCtxArgAlltoAllMesh2D*>(&arg);
    if (ctxArg == nullptr) {
        THROW<NullPtrException>(StringFormat("CcuContextAlltoAllMesh2D::ctxArg ptr is null"));
    }
    if (transports.size() == 0) {
        THROW<NullPtrException>(StringFormat("CcuContextAlltoAllMesh2D transports is empty"));
    }
    rankId_ = ctxArg->rankId;
    dimSize = ctxArg->dimSize;
    axisId = ctxArg->axisId;
    uint32_t max_dimSize = 2;
    if (dimSize.size() != max_dimSize or axisId > 1) { // dimSize不为2，或axisId超过1，则不为2D场景
        THROW<NullPtrException>(
            StringFormat("[CcuContextAlltoAllMesh2D] dimSize[%u] or axisId[%u] is invalid", dimSize.size(), axisId));
    }
    CHK_PRT_THROW(
        dimSize[0] == 0 || dimSize[1] == 0,
        HCCL_ERROR("[CcuContextAlltoAllMesh2D] dimSize0[%llu] or dimSize1[%llu] is zero", dimSize[0], dimSize[1]),
        InvalidParamsException, "dimSize[0] or dimSize[1] is invalid");
    dimId.emplace_back(rankId_ % dimSize[0]);
    dimId.emplace_back(rankId_ / dimSize[0]);
    localId = dimId[axisId];
    localSize = dimSize[axisId];
    anotherId = dimId[1 - axisId]; // 本rank在另一个轴上的Id
    anotherSize = dimSize[1 - axisId];
    HCCL_INFO(
        "[CcuContextAlltoAllMesh2D] RankId[%u], DimSize: D0[%u]--D1[%u], localId[%u], localSize[%u]", rankId_,
        dimSize[0], dimSize[1], localId, localSize);

    AllocGoResource(LOC_CPY_LOOP_NUM); // 只用8个loop做本地搬运，每个loop搬4K

    localAxisSignalName = "CcuContextAlltoAllMesh2DAxisSync_" + std::to_string(axisId);
    anotherAxisSignalName = "CcuContextAlltoAllMesh2DAxisSync_" + std::to_string(1 - axisId);
}

void CcuContextAlltoAllMesh2D::InitResources()
{
    // 用write语义，input只有本地的1个，scratch和output需要交换
    ExportMaskSignal(localAxisSignal, localAxisSignalName);
    anotherAxisSignal = ImportMaskSignal(anotherAxisSignalName);

    uint32_t transportIdx = 0;
    for (uint32_t peerId = 0; peerId < localSize; peerId++) {
        if (peerId == localId) {
            bufferA.emplace_back(CreateVariable());
            output.emplace_back(CreateVariable());
            token.emplace_back(CreateVariable());
        } else {
            HCCL_INFO("[CcuContextAlltoAllMesh2D]Rank[%u], PeerId[%u], TransportId[%u]", rankId_, peerId, transportIdx);
            bufferA.emplace_back(
                CreateVariable(*(transports[transportIdx]), 0)); // 获取transport中id=1的Var来传递bufferA
            output.emplace_back(CreateVariable(*(transports[transportIdx]), 1)); // 1 for output
            token.emplace_back(CreateVariable(*(transports[transportIdx]), 2));  // 2 for token
            transportIdx++;
        }
    }

    for (uint16_t i = 0; i < localSize; i++) {
        inputAddrs.emplace_back(CreateMemory());
        bufferAddrs.emplace_back(CreateMemory());
        outputAddrs.emplace_back(CreateMemory());
    }

    for (uint16_t sliceId = 0; sliceId < anotherSize; sliceId++) {
        firstSignal.emplace_back(
            CreateMaskSignal()); // 每个对端发anotherSize个分片，localSize个分片共用一个信号，共anotherSize个
        secondSignal.emplace_back(CreateMaskSignal());
    }

    return;
}

void CcuContextAlltoAllMesh2D::LoadArgs()
{
    Load(input);
    Load(output[localId]);
    Load(token[localId]);
    Load(bufferA[localId]);
    Load(bufferB);
    Load(sliceSize_);
    Load(goSize_);

    Load(baseOffset); // 10号
    Load(firstTransportSize);
    Load(firstChunkOffset);
    Load(firstInputStrideLocal);
    Load(firstInputStrideAnother);
    Load(firstBufferOffset); // 15号
    Load(firstBufferStride); // 16号
    Load(firstOutputOffset);

    Load(secondTransportSize);
    Load(secondChunkOffset);
    Load(secondInputOffset);
    Load(secondInputStride);
    Load(secondBufferStrideLocal);
    Load(secondBufferStrideAnother);
    Load(secondOutputOffset);
    Load(secondOutputStride);

    return;
}

void CcuContextAlltoAllMesh2D::ExchangeInfoAndSync()
{
    // 交换信息并做同步，前同步固定用1,2,3号信号
    uint16_t selfBit = 1 << localId;
    uint16_t allBit = ((1 << localSize) - 1) & (~(1 << localId));

    for (auto t : transports) {
        if (t == nullptr) {
            THROW<NullPtrException>(StringFormat("CcuContextAlltoAllMesh2D::Algorithm transport ptr is null"));
        }
        WriteVariableWithSignal(*t, bufferA[localId], 0, CKE_ID_1, selfBit); // index = 0，传递第一轮output信息
        WriteVariableWithSignal(*t, output[localId], 1, CKE_ID_2, selfBit);  // index = 1，传递第二轮output信息
        WriteVariableWithSignal(*t, token[localId], 2, CKE_ID_3, selfBit);   // index = 2，传递token信息
    }
    GroupWait(*transportGroup, CKE_ID_1, allBit);
    GroupWait(*transportGroup, CKE_ID_2, allBit);
    GroupWait(*transportGroup, CKE_ID_3, allBit);

    return;
}

void CcuContextAlltoAllMesh2D::RankSync(uint32_t signalIndex)
{
    // 与远端做同步
    uint16_t selfBit = 1 << localId;
    uint16_t allBit = ((1 << localSize) - 1) & (~(1 << localId));

    for (auto t : transports) {
        if (t == nullptr) {
            THROW<NullPtrException>(StringFormat("CcuContextAlltoAllMesh2D::Algorithm transport ptr is null"));
        }
        RemotePost(*t, signalIndex, selfBit);
    }
    GroupWait(*transportGroup, signalIndex, allBit);

    return;
}

void CcuContextAlltoAllMesh2D::AxisSync(uint32_t signalIndex)
{
    const uint32_t DIE_NUM = 2; // 2个die
    if (signalIndex > 1) {
        THROW<InvalidParamsException>(
            StringFormat("[CcuContextAlltoAllMesh2D] Unexpected SignalIndex[%u]", signalIndex));
    }
    LocalCtxPost(anotherAxisSignal, 1 << (axisId + signalIndex * DIE_NUM));
    LocalWait(localAxisSignal, 1 << (1 - axisId + signalIndex * DIE_NUM));
    return;
}

void CcuContextAlltoAllMesh2D::FirstStepOneSlice(uint16_t sliceId)
{
    if (sliceId == anotherId) {
        // 当前分片属于对端，直接写到对端output
        uint32_t transIdx = 0; // 约定transport中的link按照rankId从小到大的顺序排列
        for (uint32_t peerId = 0; peerId < localSize; peerId++) {
            if (peerId == localId) {
                LocalPost(firstSignal[sliceId], (1 << peerId));
            } else {
                Write(
                    *(transports[transIdx]), outputAddrs[peerId], inputAddrs[peerId], firstTransportSize,
                    firstSignal[sliceId], (1 << peerId));
                transIdx++;
            }
            inputAddrs[peerId].addr += firstInputStrideAnother; // 给每个对端的下一片slice的input地址，增加对应偏移
            bufferAddrs[peerId].addr += firstBufferStride; // 跳过对端buffer中不需要转发的那一片
        }
    } else {
        // 当前分片需要对端转发，写到对端的bufferX/bufferY
        uint32_t transIdx = 0;
        for (uint32_t peerId = 0; peerId < localSize; peerId++) {
            if (peerId == localId) {
                LocalPost(firstSignal[sliceId], (1 << peerId)); // 对于本die经过转发无法到达的对端，只设置标记不发送数据
                continue;
            }
            Write(
                *(transports[transIdx]), bufferAddrs[peerId], inputAddrs[peerId], firstTransportSize,
                firstSignal[sliceId], (1 << peerId));
            inputAddrs[peerId].addr += firstInputStrideAnother;
            bufferAddrs[peerId].addr += firstBufferStride; // 给对端用于转发的分片，每片相对前片加localSize*sliceSize
            transIdx++;
        }
    }

    return;
}

void CcuContextAlltoAllMesh2D::FirstStep()
{
    CcuRep::Memory lgSrc = CreateMemory();
    CcuRep::Memory lgDst = CreateMemory();

    // 统一处理token，访问第i个对端需要使用对应的token
    for (uint16_t i = 0; i < localSize; i++) {
        inputAddrs[i].token = token[i];
        bufferAddrs[i].token = token[i];
        outputAddrs[i].token = token[i];
    }
    lgSrc.token = token[localId];
    lgDst.token = token[localId];
    // 本rank的input内存块用一组mem地址来分割
    inputAddrs[0].addr = input;
    inputAddrs[0].addr += baseOffset;
    inputAddrs[0].addr += firstChunkOffset;
    for (uint16_t i = 1; i < localSize; i++) {
        // 准备发送给rank0对应分片的地址即为input首地址，后续rank的偏移依次递增
        inputAddrs[i].addr = inputAddrs[i - 1].addr + firstInputStrideLocal;
    }
    for (uint16_t i = 0; i < localSize; i++) {
        // output offset
        outputAddrs[i].addr = output[i];
        outputAddrs[i].addr += baseOffset;
        outputAddrs[i].addr += firstChunkOffset;
        outputAddrs[i].addr += firstOutputOffset; // 第一轮直接发送给对端的slice的偏移
        // buffer offset
        bufferAddrs[i].addr = bufferA[i];
        bufferAddrs[i].addr += firstBufferOffset; // 发送给对端用于转发的分片，第一片的起始偏移，后续每片步进相同长度
    }
    // 准备LG搬运的地址
    lgSrc.addr = inputAddrs[localId].addr;
    lgDst.addr = outputAddrs[localId].addr;
    for (uint16_t sliceId = 0; sliceId < anotherSize; sliceId++) {
        if (sliceId == anotherId) {
            break;
        }
        lgSrc.addr += firstInputStrideAnother; // 在input中找到自身对应的那个分片，跳出循环
    }

    {
        // 当第一轮的搬运量为零时，跳过搬运
        CcuRep::Condition cond(this, firstTransportSize != 0);

        for (uint16_t sliceId = 0; sliceId < anotherSize; sliceId++) { // sliceId等于dstRank在另一个维度上的id
            FirstStepOneSlice(sliceId);
        }
    }
    // Loopgroup做本地搬运，必然不为零
    if (axisId == 0) {
        LocalCopyByLoopGroup(lgDst, lgSrc, goSize_);
    }

    // 检查第一轮的数据是否已发完
    {
        // 当第一轮的搬运量非零时，检查相应完成标记
        CcuRep::Condition cond(this, firstTransportSize != 0);
        for (uint16_t sliceId = 0; sliceId < anotherSize; sliceId++) {
            LocalWait(firstSignal[sliceId], (1 << localSize) - 1); // 等待第一轮所有分片都发完
        }
    }

    return;
}

void CcuContextAlltoAllMesh2D::SecondStep()
{
    {
        // 当第二轮的搬运量为零时，跳过搬运
        CcuRep::Condition cond(this, secondTransportSize != 0);

        // 地址计算
        // input offset，本rank的input内存块用一组GSA来分割
        inputAddrs[0].addr = input;
        inputAddrs[0].addr += baseOffset;
        bufferAddrs[0].addr = bufferB;
        inputAddrs[0].addr += secondChunkOffset;
        inputAddrs[0].addr += secondInputOffset; // 一共从input发送localSize-1个分片（跳过自己），用localSize个input地址
        for (uint16_t i = 1; i < localSize; i++) {
            inputAddrs[i].addr = inputAddrs[i - 1].addr + secondInputStride;
            // 每轮给每个对端从buffer发送1个分片，共anotherSize-1轮
            bufferAddrs[i].addr = bufferAddrs[i - 1].addr + secondBufferStrideLocal;
        }
        // output offset
        for (uint16_t i = 0; i < localSize; i++) {
            // 给每个对端的output写anotherSize个分片，这些分片的src的rankId从offset开始，以stride步进
            outputAddrs[i].addr = output[i];
            outputAddrs[i].addr += baseOffset;
            outputAddrs[i].addr += secondChunkOffset;
            outputAddrs[i].addr += secondOutputOffset;
        }

        // 从input与buffer中给每个对端发anotherSize个分片
        for (uint16_t sliceId = 0; sliceId < anotherSize; sliceId++) {
            uint32_t transIdx = 0;
            for (uint32_t peerId = 0; peerId < localSize; peerId++) {
                if (peerId == localId) {
                    LocalPost(secondSignal[sliceId], (1 << peerId)); // 给自己的分片在第一轮已经发过，第二轮只设置标记
                    continue;
                }
                if (sliceId == anotherId) {
                    // 从input发出
                    Write(
                        *(transports[transIdx]), outputAddrs[peerId], inputAddrs[peerId], secondTransportSize,
                        secondSignal[sliceId], (1 << peerId));
                } else {
                    // 从buffer发出
                    Write(
                        *(transports[transIdx]), outputAddrs[peerId], bufferAddrs[peerId], secondTransportSize,
                        secondSignal[sliceId], (1 << peerId));
                }
                transIdx++;
                outputAddrs[peerId].addr += secondOutputStride;
                bufferAddrs[peerId].addr += secondBufferStrideAnother;
            }
        }
        for (uint16_t sliceId = 0; sliceId < anotherSize; sliceId++) {
            LocalWait(secondSignal[sliceId], (1 << localSize) - 1); // 等待第二轮所有分片都发完
        }
    }

    return;
}

void CcuContextAlltoAllMesh2D::CreateLocalCopyLoop()
{
    std::string opStr = "a2a_localcpy_loopgroup";
    for (uint32_t index = 0; index < 2; index++) { // 需要2个Loop
        CcuRep::LoopBlock lb(this, "a2a_localcpy_loop_" + std::to_string(index));
        CcuRep::Memory src = CreateMemory();
        CcuRep::Variable len = CreateVariable();
        CcuRep::Memory dst = CreateMemory();
        lb(src, dst, len);

        CcuRep::CcuBuffer buf = moRes.ccuBuffer[index * moConfig.msInterleave];
        CcuRep::MaskSignal sem = moRes.maskSignal[index];

        LocalCopy(buf, src, len, sem);
        LocalWait(sem);
        LocalCopy(dst, buf, len, sem);
        LocalWait(sem);
    }
    return;
}

void CcuContextAlltoAllMesh2D::LocalCopyByLoopGroup(CcuRep::Memory dst, CcuRep::Memory src, GroupOpSize& goPara)
{
    std::string opStr = "a2a_localcpy_loopgroup";
    CreateLocalCopyLoop();

    {
        CcuRep::Condition cond(this, goPara.loopParam != 0);

        CcuRep::Variable loopParam = CreateVariable();
        loopParam = CcuRep::GetLoopParam(0, moConfig.memSlice * moConfig.loopCount, 0);
        loopParam += goPara.loopParam;

        CcuRep::Variable sliceSize = CreateVariable();
        sliceSize = moConfig.memSlice;
        auto lc = Loop("a2a_localcpy_loop_0")(src, dst, sliceSize);

        CcuRep::Variable paraCfg = CreateVariable();
        paraCfg = CcuRep::GetParallelParam(moConfig.loopCount - 1, 0, 1);
        CcuRep::Variable offsetCfg = CreateVariable();
        offsetCfg = CcuRep::GetOffsetParam(moConfig.memSlice, moConfig.msInterleave, 1);
        LoopGroup({lc}, {loopParam}, paraCfg, offsetCfg);
    }

    {
        CcuRep::Condition cond(this, goPara.parallelParam != 0);

        dst.addr += goPara.addrOffset;
        src.addr += goPara.addrOffset;
        auto lc0 = Loop("a2a_localcpy_loop_0")(src, dst, goPara.residual);

        src.addr += goPara.residual;
        dst.addr += goPara.residual;
        CcuRep::Variable sliceSize = CreateVariable();
        sliceSize = moConfig.memSlice;
        auto lc1 = Loop("a2a_localcpy_loop_1")(src, dst, sliceSize);

        CcuRep::Variable loopCfg0 = CreateVariable();
        loopCfg0 = CcuRep::GetLoopParam(0, 0, 1);
        CcuRep::Variable loopCfg1 = CreateVariable();
        loopCfg1 = CcuRep::GetLoopParam(0, 0, 1);
        CcuRep::Variable offsetCfg = CreateVariable();
        offsetCfg = CcuRep::GetOffsetParam(moConfig.memSlice, moConfig.msInterleave, 1);
        LoopGroup({lc0, lc1}, {loopCfg0, loopCfg1}, goPara.parallelParam, offsetCfg);
    }
}

void CcuContextAlltoAllMesh2D::Algorithm()
{
    // 初始化寄存器资源 & 加载外部输入参数
    HCCL_INFO("[CcuContextAlltoAllMesh2D] AllgatherMesh1D Algorithm Init Begins.");
    InitResources();
    LoadArgs();

    // 第一轮，X方向发a，Y方向发后b，到对端的块均放在output，要沿X转发的b块放在对端的bufferX，根据转发目的、自身locId两级偏移
    HCCL_INFO("[CcuContextAlltoAllMesh2D] Algorithm first step begins.");
    ExchangeInfoAndSync();
    FirstStep();
    RankSync(CKE_ID_0);
    AxisSync(FST_AXIS_ID);

    // 第二轮，从input和buffer中将剩余的本端分片以及待转发分片发给对端；其中给每个对端发1个本端分片，localSize-1个转发分片
    HCCL_INFO("[CcuContextAlltoAllMesh2D] Algorithm second step begins.");
    RankSync(CKE_ID_1);
    SecondStep();
    RankSync(CKE_ID_0);
    AxisSync(SEC_AXIS_ID);

    HCCL_INFO("[CcuContextAlltoAllMesh2D] Algorithm Ends.");
    return;
}

void CcuContextAlltoAllMesh2D::CalculateArgs(const CcuTaskArgAlltoAllMesh2D* taskArg)
{
    if (taskArg == nullptr) {
        THROW<NullPtrException>(StringFormat("CcuContextAlltoAllMesh2D::taskArg ptr is null"));
    }

    uint64_t sendStride = taskArg->sendStride;
    uint64_t recvStride = taskArg->recvStride;
    uint64_t aSize = taskArg->aSize;
    uint64_t bSize = taskArg->bSize;
    uint64_t sendLength = taskArg->sendLength;

    uint64_t sliceSize = aSize + bSize;
    uint64_t srcStride = sendLength + sendStride;
    uint64_t dstStride = sendLength + recvStride;

    // 根据axisId决定bufferA与bufferB的地址，暂定a与b的大小相等
    if (axisId == 0) {
        firstTransportSizeValue = aSize;
        firstChunkOffsetValue = 0;
        firstInputStrideLocalValue = srcStride;
        firstInputStrideAnotherValue = dimSize[0] * srcStride;

        secondTransportSizeValue = bSize;
        secondChunkOffsetValue = aSize;
        secondInputOffsetValue = dimId[1] * dimSize[0] * srcStride;
        secondInputStrideValue = srcStride;
        secondOutputOffsetValue = dimId[0] * dstStride;
        secondOutputStrideValue = dimSize[0] * dstStride;
    } else {
        firstTransportSizeValue = bSize;
        firstChunkOffsetValue = aSize;
        firstInputStrideLocalValue = dimSize[0] * srcStride;
        firstInputStrideAnotherValue = srcStride;

        secondTransportSizeValue = aSize;
        secondChunkOffsetValue = 0;
        secondInputOffsetValue = dimId[0] * srcStride;
        secondInputStrideValue = dimSize[0] * srcStride;
        secondOutputOffsetValue = dimId[1] * dimSize[0] * dstStride;
        secondOutputStrideValue = dstStride;
    }

    firstBufferOffsetValue = dimId[axisId] * sliceSize;
    firstBufferStrideValue = dimSize[axisId] * sliceSize;
    firstOutputOffsetValue = rankId_ * dstStride;

    secondBufferStrideLocalValue = dimSize[1 - axisId] * sliceSize;
    secondBufferStrideAnotherValue = sliceSize;

    return;
}

std::vector<uint64_t> CcuContextAlltoAllMesh2D::GeneArgs(const CcuTaskArg& arg)
{
    const CcuTaskArgAlltoAllMesh2D* taskArg = dynamic_cast<const CcuTaskArgAlltoAllMesh2D*>(&arg);
    if (taskArg == nullptr) {
        THROW<NullPtrException>(StringFormat("CcuContextAlltoAllMesh2D::taskArg ptr is null"));
    }

    // input&output&buffer地址
    uint64_t inputAddr = taskArg->inputAddr;
    uint64_t outputAddr = taskArg->outputAddr;
    uint64_t scratchAddr = taskArg->scratchAddr;
    uint64_t tokenInfo = taskArg->token;
    uint64_t sliceSizeValue = taskArg->aSize + taskArg->bSize;

    // scratch的前rankSize*sliceSize大小为bufferY，后一块为bufferX
    // die0第一轮写到对端的bufferY，第二轮从本端bufferX发送；die1第一轮写到对端的bufferX，第二轮从本端bufferY发送
    uint64_t bufferAAddr = 0;
    uint64_t bufferBAddr = 0;
    if (axisId == 0) {
        bufferAAddr = scratchAddr;                                            // 需要交换给对端，是bufferY
        bufferBAddr = scratchAddr + dimSize[0] * dimSize[1] * sliceSizeValue; // bufferX rankSize * sliceSize
    } else {
        bufferAAddr = scratchAddr + dimSize[0] * dimSize[1] * sliceSizeValue; // bufferX
        bufferBAddr = scratchAddr;                                            // 不需要交换给对端，是bufferY
    }

    // loopgroup按照sliceSize大小做本地搬运，只die0执行
    auto goSize = CalGoSize(taskArg->aSize + taskArg->bSize);
    CalculateArgs(taskArg);

    HCCL_INFO(
        "[CcuContextAlltoAllMesh2D][GeneArgs] RankId[%u]--AxisId[%u], inputAddr[%llu], outputAddr[%llu], \
bufferA[%llu], bufferB[%llu], goSize--[%llu][%llu][%llu][%llu], sendStride[%llu], recvStride[%llu], \
sendRecvSize[%llu], sendLength[%llu], aSize[%llu], bSize[%llu], baseOffset[%llu]",
        rankId_, axisId, inputAddr, outputAddr, bufferAAddr, bufferBAddr, goSize[0], goSize[1], goSize[2], goSize[3],
        taskArg->sendStride, taskArg->recvStride, sliceSizeValue, taskArg->sendLength, taskArg->aSize, taskArg->bSize,
        taskArg->baseOffset);

    HCCL_INFO(
        "[CcuContextAlltoAllMesh2D][CalculateArgs] firstTransportSize[%llu], firstChunkOffset[%llu], \
firstInputStrideLocal[%llu], firstInputStrideAnother[%llu], firstBufferOffset[%llu], firstBufferStride[%llu], \
firstOutputOffset[%llu], secondTransportSize[%llu], secondChunkOffset[%llu], secondInputOffset[%llu], \
secondInputStride[%llu], secondBufferStrideLocal[%llu], secondBufferStrideAnother[%llu], \
secondOutputOffset[%llu], secondOutputStride[%llu]",
        firstTransportSizeValue, firstChunkOffsetValue, firstInputStrideLocalValue, firstInputStrideAnotherValue,
        firstBufferOffsetValue, firstBufferStrideValue, firstOutputOffsetValue, secondTransportSizeValue,
        secondChunkOffsetValue, secondInputOffsetValue, secondInputStrideValue, secondBufferStrideLocalValue,
        secondBufferStrideAnotherValue, secondOutputOffsetValue, secondOutputStrideValue);

    return {
        inputAddr,
        outputAddr,
        tokenInfo,
        bufferAAddr,
        bufferBAddr,
        sliceSizeValue,
        goSize[0],
        goSize[1],
        goSize[2],
        goSize[3],
        taskArg->baseOffset,
        firstTransportSizeValue,
        firstChunkOffsetValue,
        firstInputStrideLocalValue,
        firstInputStrideAnotherValue,
        firstBufferOffsetValue,
        firstBufferStrideValue,
        firstOutputOffsetValue,
        secondTransportSizeValue,
        secondChunkOffsetValue,
        secondInputOffsetValue,
        secondInputStrideValue,
        secondBufferStrideLocalValue,
        secondBufferStrideAnotherValue,
        secondOutputOffsetValue,
        secondOutputStrideValue};
}
} // namespace Hccl
