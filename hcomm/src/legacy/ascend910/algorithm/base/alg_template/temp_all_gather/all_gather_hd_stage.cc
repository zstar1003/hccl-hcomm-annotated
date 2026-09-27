/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include "all_gather_hd_stage_pub.h"

namespace hccl {
namespace {
    static u32 GetStepNumInterServer(u32 rankSize)
    {
        u32 nSteps = 0;
        for (u32 tmp = rankSize - 1; tmp != 0; tmp >>= 1, nSteps++) {
        }
        return nSteps;
    }

    static void ReorderSequence(u32 start, u32 end, u32 len, std::vector<u32>& tree, std::vector<u32>& tmp)
    {
        const u32 divideTwo = 2;

        for (u32 i = start; i < end; i++) {
            u32 offset = i - start;
            if ((offset & 1) == 0) {
                tmp[start + offset / divideTwo] = tree[i];
            } else {
                tmp[start + (offset + len) / divideTwo] = tree[i];
            }
        }
    }

    // 参考NHRBase::GetRankMapping的实现
    static void GetRankMapping(const u32 rankSize, std::vector<u32>& sliceMap)
    {
        std::vector<u32> tree;
        for (u32 i = 0; i < rankSize; i++) {
            tree.push_back(i);
        }

        // 其他的再进行计算
        std::vector<u32> tmp(rankSize);
        u32 nSteps = GetStepNumInterServer(rankSize);
        u32 len = rankSize;

        for (u32 step = 0; step < nSteps; step++) {
            u32 nSlices = (rankSize - 1 + (1 << step)) / (1 << (step + 1));
            if (nSlices <= 1) {
                break;
            }

            bool endFlag = false;
            for (u32 part = 0; part * len < rankSize; part++) {
                u32 start = part * len;
                u32 end = std::min(start + len, rankSize);
                ReorderSequence(start, end, len, tree, tmp);

                if (((end - start) & 1) == 1) {
                    endFlag = true;
                }
            }

            for (u32 i = 0; i < rankSize; i++) {
                tree[i] = tmp[i];
            }

            if (endFlag) {
                break;
            }

            len >>= 1;
        }

        // 因为取的是tree中rank的idx，所以直接返回反向的映射
        sliceMap.resize(rankSize);
        for (u32 i = 0; i < rankSize; i++) {
            sliceMap[tree[i]] = i;
        }
        return;
    }
} // namespace

AllGatherHDStage::AllGatherHDStage(const HcclDispatcher dispatcher) : AlgTemplateBase(dispatcher) {}

AllGatherHDStage::~AllGatherHDStage() {}

HcclResult AllGatherHDStage::Prepare(PrepareData& param)
{
    userRank_ = param.userRank;
    opInfo_ = param.opInfo;

    meshStreams_ = *param.subStreamsPtr;
    meshSignalPtr_ = param.signalPtr;
    meshSignalAuxPtr_ = param.signalAuxPtr;

    return AlgTemplateBase::Prepare(
        param.inputMem, param.outputMem, param.scratchMem, param.count, param.dataType, param.stream,
        HCCL_REDUCE_RESERVED, INVALID_VALUE_RANKID);
}

HcclResult AllGatherHDStage::MainRecordSub(u32 streamNum)
{
    const std::vector<std::shared_ptr<LocalNotify>>& meshSignalAux = *meshSignalAuxPtr_;
    for (u32 signalIndex = 0; signalIndex < streamNum; signalIndex++) {
        CHK_RET(LocalNotify::Post(stream_, dispatcher_, meshSignalAux[signalIndex], profilerInput_.stage));
    }
    return HCCL_SUCCESS;
}

HcclResult AllGatherHDStage::SubWaitMain(u32 streamNum)
{
    const std::vector<std::shared_ptr<LocalNotify>>& meshSignalAux = *meshSignalAuxPtr_;
    for (u32 streamIndex = 0; streamIndex < streamNum; streamIndex++) {
        CHK_RET(LocalNotify::Wait(
            meshStreams_[streamIndex], dispatcher_, meshSignalAux[streamIndex], profilerInput_.stage));
    }
    return HCCL_SUCCESS;
}

HcclResult AllGatherHDStage::MainWaitSub(u32 streamNum)
{
    const std::vector<std::shared_ptr<LocalNotify>>& meshSignal = *meshSignalPtr_;
    for (u32 signalIndex = 0; signalIndex < streamNum; signalIndex++) {
        CHK_RET(LocalNotify::Wait(stream_, dispatcher_, meshSignal[signalIndex], profilerInput_.stage));
    }
    return HCCL_SUCCESS;
}

HcclResult AllGatherHDStage::SubRecordMain(u32 streamNum)
{
    const std::vector<std::shared_ptr<LocalNotify>>& meshSignal = *meshSignalPtr_;
    for (u32 streamIndex = 0; streamIndex < streamNum; streamIndex++) {
        CHK_RET(
            LocalNotify::Post(meshStreams_[streamIndex], dispatcher_, meshSignal[streamIndex], profilerInput_.stage));
    }
    return HCCL_SUCCESS;
}

// ring allreduce算法的函数入口
HcclResult AllGatherHDStage::RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK>& links)
{
    HcclResult ret = HCCL_SUCCESS;
    CHK_SMART_PTR_NULL(dispatcher_);
    CHK_PTR_NULL(stream_.ptr());
    HCCL_INFO(
        "AllGatherHDStage run: rank[%u] ranksize[%u] inputMem[%p] outputMem[%p] count[%llu]", rank, rankSize,
        inputMem_.ptr(), outputMem_.ptr(), count_);

    if (links.size() < rankSize) {
        HCCL_ERROR(
            "[AllGatherHDStage][RunAsync]rank[%u] linksize[%llu] is less than rankSize[%u]", rank, links.size(),
            rankSize);
        return HCCL_E_INTERNAL;
    }

    ret = RunAllGatherStage(rank, rankSize, links);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[AllGatherHDStage][RunAsync]rank[%u] count[%llu] failed in step", rank, count_), ret);

    HCCL_INFO("AllGatherHDStage finished: rank[%u] ranksize[%u]", rank, rankSize);
    return HCCL_SUCCESS;
}

HcclResult AllGatherHDStage::ReverseId(u32 oriIdx, u32& revIdx)
{
    revIdx = 0;
    u32 powerBase = 0;
    for (u32 i = 0; i < powerSteps_; i++) {
        powerBase = static_cast<u32>(pow(base, i));
        revIdx += (oriIdx / powerBase % base) * static_cast<u32>(pow(base, powerSteps_ - i - 1));
    }
    return HCCL_SUCCESS;
}

HcclResult AllGatherHDStage::PrepareSliceData(
    u32 subRank, u32 subRankSize, u32 size, u32 batchSize, std::vector<Slice>& slices) const
{
    Slice temp;
    u32 power = static_cast<u32>(log2(subRankSize));
    slices.clear();
    slices.reserve(power);
    for (u32 step = 0; step < power; step++) {
        u32 sliceNum = pow(base, step);
        u32 offset = static_cast<u32>(subRank ^ (1 << step)) / sliceNum * sliceNum;
        temp.offset = (offset * size) % batchSize;
        temp.size = sliceNum * size;
        slices.push_back(temp);
    }
    return HCCL_SUCCESS;
}

HcclResult AllGatherHDStage::RunAllGatherStage(u32 rank, u32 rankSize, const std::vector<LINK>& links)
{
    HCCL_INFO(
        "RunAllGatherStage run: rank[%u] totalrank[%u] outputMem[%p] count[%llu]", rank, rankSize, outputMem_.ptr(),
        count_);
    u32 unitSize = SIZE_TABLE[dataType_];
    totalSize_ = unitSize * count_;
    // 对应因式分解中的2的幂次部分
    powerSteps_ = static_cast<u32>(log2(rankSize & (-rankSize)));
    if (outputMem_.ptr() != opInfo_->outputAddr) {
        if (powerSteps_ >= base) {
            finalSteps_ = base;
        } else if (powerSteps_ >= 1) {
            finalSteps_ = 1;
        }
    }
    // 对应因式分解中的奇数部分
    noPower_ = rankSize / (rankSize & (-rankSize));
    CHK_RET(RunPreCopy(rank, rankSize, links));
    if (noPower_ > 1) {
        CHK_RET(RunAllGatherNoPower(rank, rankSize, links));
    }
    if ((powerSteps_ - finalSteps_) >= 1) {
        CHK_RET(RunAllGatherPower(rank, rankSize, links));
    }
    if (finalSteps_ == base) {
        CHK_RET(RunAllGatherLastTwo(rank, rankSize, links));
    } else if (finalSteps_ == 1) {
        CHK_RET(RunAllGatherLastOne(rank, rankSize, links));
    } else {
        CHK_RET(RunAllGatherLast(rank, rankSize, links));
    }
    return HCCL_SUCCESS;
}

HcclResult AllGatherHDStage::RunPreCopy(u32 rank, u32 rankSize, const std::vector<LINK>& links)
{
    // 交换数据
    HCCL_INFO(
        "RunPreCopy run: rank[%u] totalrank[%u] outputMem[%p] count[%llu]", rank, rankSize, outputMem_.ptr(), count_);
    std::vector<u32> noPowerMap;
    GetRankMapping(noPower_, noPowerMap);
    std::vector<u32> noPowerRevMap(noPower_);
    CHK_PRT_RET(
        noPowerMap.size() != noPower_,
        HCCL_ERROR("[AllGatherHDStage][RunPreCopy]rank[%u] count[%llu] failed", rank, count_), HCCL_E_RESERVED);
    for (u32 i = 0; i < noPowerMap.size(); i++) {
        noPowerRevMap[noPowerMap[i]] = i;
    }
    u32 groupIdx = rank % static_cast<u32>(pow(base, powerSteps_));
    u32 group = rank / static_cast<u32>(pow(base, powerSteps_));
    u32 revRank = 0;
    u32 revIdx = 0;
    CHK_RET(ReverseId(groupIdx, revIdx));
    // 将revRank的数据写到本卡
    revRank = revIdx * noPower_ + noPowerMap[group];
    // 将本卡数据写到revRankrev卡
    groupIdx = rank % noPower_;
    group = rank / noPower_;
    CHK_RET(ReverseId(group, revIdx));
    u32 revRankrev = noPowerRevMap[groupIdx] * static_cast<u32>(pow(base, powerSteps_)) + revIdx;
    DeviceMem UserMemIn = DeviceMem::create(opInfo_->inputAddr, totalSize_);
    if (revRank != rank && revRankrev != rank) {
        CHK_RET(links[revRank]->TxAck(stream_));
        CHK_RET(links[revRankrev]->RxAck(stream_));
        void* remMemPtr = nullptr;
        CHK_RET(links[revRankrev]->GetRemoteMem(UserMemType::OUTPUT_MEM, &remMemPtr));
        DeviceMem dst = DeviceMem::create(
            static_cast<u8*>(remMemPtr) + (rank % (rankSize / static_cast<u32>(pow(base, finalSteps_)))) * totalSize_,
            totalSize_);
        DeviceMem src = UserMemIn;
        CHK_RET(HcclD2DMemcpyAsync(
            dispatcher_, dst, src, stream_, links[revRankrev]->GetRemoteRank(), links[revRankrev]->GetLinkType()));
        CHK_RET(links[revRankrev]->TxDataSignal(stream_));
        CHK_RET(links[revRank]->RxDataSignal(stream_));
    } else {
        DeviceMem dst
            = outputMem_.range((rank % (rankSize / static_cast<u32>(pow(base, finalSteps_)))) * totalSize_, totalSize_);
        DeviceMem src = UserMemIn;
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dst, src, stream_));
    }
    return HCCL_SUCCESS;
}

HcclResult AllGatherHDStage::RunAllGatherNoPower(u32 rank, u32 rankSize, const std::vector<LINK>& links)
{
    std::unique_ptr<AlgTemplateBase> tempAlg
        = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_GATHER_NHR, dispatcher_);
    CHK_SMART_PTR_NULL(tempAlg);
    CHK_RET(tempAlg->Prepare(true));
    HCCL_INFO(
        "[AllGatherHDStage][RunAllGather] rank[%u] tempAlg AllGatherNHR inputMem[%p] outputMem[%p] mem_size[%llu] "
        "count[%llu] planeID:[%d]",
        rank, inputMem_.ptr(), outputMem_.ptr(), outputMem_.size(), count_, profilerInput_.planeID);
    u32 groupIdx = rank % static_cast<u32>(pow(base, powerSteps_));
    u32 group = rank / static_cast<u32>(pow(base, powerSteps_));
    u32 revIdx = 0;
    CHK_RET(ReverseId(groupIdx, revIdx));
    u64 baseOffset = ((revIdx * noPower_) % (rankSize / static_cast<u32>(pow(base, finalSteps_)))) * totalSize_;
    std::vector<Slice> slices;
    for (u32 i = 0; i < noPower_; i++) {
        Slice temp;
        temp.offset = i * totalSize_;
        temp.size = totalSize_;
        slices.push_back(temp);
    }
    DeviceMem nhrOutput = outputMem_.range(baseOffset, outputMem_.size() - baseOffset);
    CHK_RET(tempAlg->Prepare(
        nhrOutput, nhrOutput, nhrOutput, count_, dataType_, stream_, reductionOp_, 0, slices, baseOffset));

    CHK_RET(tempAlg->RegisterProfiler(profilerInput_.planeID, profilerInput_.stage, profilerInput_.step, stream_));

    std::vector<LINK> nhrLinks;
    for (u32 i = 0; i < noPower_; i++) {
        u32 remote = i * static_cast<u32>(pow(base, powerSteps_)) + groupIdx;
        nhrLinks.push_back(links[remote]);
    }
    return tempAlg->RunAsync(group, noPower_, nhrLinks);
}

HcclResult AllGatherHDStage::RunBetweenStep(u32 rank, u32 neighCur, u32 neighNext, const std::vector<LINK>& links)
{
    (void)rank;
    CHK_RET(MainRecordSub(1));
    CHK_RET(SubWaitMain(1));

    CHK_RET(links[neighCur]->TxDataSignal(meshStreams_[0]));
    CHK_RET(links[neighCur]->RxDataSignal(meshStreams_[0]));

    CHK_RET(links[neighNext]->TxAck(stream_));
    CHK_RET(links[neighNext]->RxAck(stream_));

    CHK_RET(SubRecordMain(1));
    CHK_RET(MainWaitSub(1));

    return HCCL_SUCCESS;
}

HcclResult AllGatherHDStage::RunAllGatherPower(u32 rank, u32 rankSize, const std::vector<LINK>& links)
{
    DeviceMem userMemOut = DeviceMem::create(opInfo_->outputAddr, totalSize_ * rankSize);

    void* remMemPtr = nullptr;
    DeviceMem dst;
    DeviceMem src;
    u32 dstRank;
    u32 dstGroupIdx;
    u32 group = rank / static_cast<u32>(pow(base, powerSteps_));
    u32 groupIdx = rank % static_cast<u32>(pow(base, powerSteps_));
    u32 revGroup = 0;
    CHK_RET(ReverseId(groupIdx, revGroup));
    CHK_RET(PrepareSliceData(
        revGroup, pow(base, powerSteps_), noPower_ * totalSize_,
        totalSize_ * rankSize / static_cast<u32>(pow(base, finalSteps_)), slicePower_));
    for (u32 step = 0; step < powerSteps_ - finalSteps_; step++) {
        dstGroupIdx = groupIdx ^ (1 << (powerSteps_ - 1 - step));
        dstRank = group * pow(base, powerSteps_) + dstGroupIdx;
        if (step == 0) {
            CHK_RET(links[dstRank]->TxAck(stream_));
            CHK_RET(links[dstRank]->RxAck(stream_));
        }
        CHK_RET(links[dstRank]->GetRemoteMem(UserMemType::OUTPUT_MEM, &remMemPtr));
        dst = outputMem_.range(slicePower_[step].offset, slicePower_[step].size);
        src = DeviceMem::create(static_cast<u8*>(remMemPtr) + slicePower_[step].offset, slicePower_[step].size);
        CHK_RET(HcclD2DMemcpyAsync(
            dispatcher_, dst, src, stream_, links[dstRank]->GetRemoteRank(), links[dstRank]->GetLinkType()));
        if (step != (powerSteps_ - finalSteps_ - 1)) {
            CHK_RET(RunBetweenStep(
                rank, dstRank, group * pow(base, powerSteps_) + (groupIdx ^ (1 << (powerSteps_ - 1 - step - 1))),
                links));
        } else {
            CHK_RET(links[dstRank]->TxDataSignal(stream_));
            CHK_RET(links[dstRank]->RxDataSignal(stream_));
        }
    }
    return HCCL_SUCCESS;
}
HcclResult AllGatherHDStage::RunAllGatherLastTwo(u32 rank, u32 rankSize, const std::vector<LINK>& links)
{
    // mesh
    DeviceMem userMemOut = DeviceMem::create(opInfo_->outputAddr, totalSize_ * rankSize);
    DeviceMem emptyMem = outputMem_.range(0, 0);
    CHK_RET(MainRecordSub(base));
    CHK_RET(SubWaitMain(base));

    u32 subGroupSize = static_cast<u32>(pow(base, base));
    CHK_PRT_RET(
        meshStreams_.size() < (subGroupSize - 1),
        HCCL_ERROR("[AllGatherHDStage][RunAllGatherLastTwo]rank[%u] count[%llu] failed", rank, count_),
        HCCL_E_RESERVED);
    for (u32 round = 1; round < subGroupSize; round++) {
        u32 dstRank = rank / subGroupSize * subGroupSize + BackwardRank(rank % subGroupSize, subGroupSize, round);
        Stream& subStream = round == (subGroupSize - 1) ? stream_ : meshStreams_[round - 1];
        CHK_RET(links[dstRank]->TxAck(subStream));
        CHK_RET(links[dstRank]->RxAck(subStream));
    }

    CHK_RET(SubRecordMain(base));
    CHK_RET(MainWaitSub(base));

    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, emptyMem, emptyMem, stream_));

    CHK_RET(SubWaitMain(meshStreams_.size()));
    CHK_RET(MainRecordSub(meshStreams_.size()));

    if (outputMem_.ptr() != opInfo_->outputAddr) {
        DeviceMem src = outputMem_.range(0, totalSize_ * rankSize / subGroupSize);
        DeviceMem dst = userMemOut.range(
            totalSize_ * rankSize / subGroupSize * (resMap[rank % subGroupSize]), totalSize_ * rankSize / subGroupSize);
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dst, src, stream_));
    }

    for (u32 round = 1; round < subGroupSize; round++) {
        u32 dstRank = rank / subGroupSize * subGroupSize + BackwardRank(rank % subGroupSize, subGroupSize, round);
        Stream& subStream = meshStreams_[round - 1];
        void* remMemPtr = nullptr;
        CHK_RET(links[dstRank]->GetRemoteMem(UserMemType::OUTPUT_MEM, &remMemPtr));
        DeviceMem src = DeviceMem::create(static_cast<u8*>(remMemPtr), totalSize_ * rankSize / subGroupSize);
        DeviceMem dst = userMemOut.range(
            totalSize_ * rankSize / subGroupSize * (resMap[dstRank % subGroupSize]),
            totalSize_ * rankSize / subGroupSize);
        CHK_RET(HcclD2DMemcpyAsync(
            dispatcher_, dst, src, subStream, links[dstRank]->GetRemoteRank(), links[dstRank]->GetLinkType()));
        CHK_RET(links[dstRank]->TxDataSignal(subStream));
        CHK_RET(links[dstRank]->RxDataSignal(subStream));
    }
    CHK_RET(SubRecordMain(meshStreams_.size()));
    CHK_RET(MainWaitSub(meshStreams_.size()));
    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, emptyMem, emptyMem, stream_));
    return HCCL_SUCCESS;
}

HcclResult AllGatherHDStage::RunAllGatherLastOne(u32 rank, u32 rankSize, const std::vector<LINK>& links)
{
    DeviceMem userMemOut = DeviceMem::create(opInfo_->outputAddr, totalSize_ * rankSize);
    u32 group = rank / static_cast<u32>(pow(base, powerSteps_));
    u32 groupIdx = rank % static_cast<u32>(pow(base, powerSteps_));
    DeviceMem emptyMem = outputMem_.range(0, 0);
    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, emptyMem, emptyMem, stream_));

    u32 dstRank = group * pow(base, powerSteps_) + (groupIdx ^ (1 << 0));
    CHK_RET(links[dstRank]->TxAck(stream_));
    CHK_RET(links[dstRank]->RxAck(stream_));

    CHK_RET(MainRecordSub(1));
    CHK_RET(SubWaitMain(1));
    // 本地拷贝
    if (outputMem_.ptr() != opInfo_->outputAddr) {
        DeviceMem src = outputMem_.range(0, totalSize_ * rankSize / base);
        DeviceMem dst = userMemOut.range(totalSize_ * rankSize / base * (rank % base), totalSize_ * rankSize / base);
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dst, src, stream_));
    }
    // 对端拷到usrout上
    void* remMemPtr = nullptr;
    CHK_RET(links[dstRank]->GetRemoteMem(UserMemType::OUTPUT_MEM, &remMemPtr));
    DeviceMem dst = userMemOut.range(totalSize_ * rankSize / base * (1 - (rank % base)), totalSize_ * rankSize / base);
    DeviceMem src = DeviceMem::create(static_cast<u8*>(remMemPtr) + 0, totalSize_ * rankSize / base);
    CHK_RET(HcclD2DMemcpyAsync(
        dispatcher_, dst, src, meshStreams_[0], links[dstRank]->GetRemoteRank(), links[dstRank]->GetLinkType()));

    CHK_RET(SubRecordMain(1));
    CHK_RET(MainWaitSub(1));

    CHK_RET(links[dstRank]->TxDataSignal(stream_));
    CHK_RET(links[dstRank]->RxDataSignal(stream_));

    return HCCL_SUCCESS;
}

HcclResult AllGatherHDStage::RunAllGatherLast(
    [[maybe_unused]] u32 rank, u32 rankSize, [[maybe_unused]] const std::vector<LINK>& links)
{
    if (outputMem_.ptr() != opInfo_->outputAddr) {
        DeviceMem userMemOut = DeviceMem::create(opInfo_->outputAddr, totalSize_ * rankSize);
        DeviceMem src = outputMem_.range(0, totalSize_ * rankSize);
        DeviceMem dst = userMemOut.range(0, totalSize_ * rankSize);
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dst, src, stream_));
    }
    return HCCL_SUCCESS;
}

REGISTER_TEMPLATE(TemplateType::TEMPLATE_ALL_GATHER_HD_STAGE, AllGatherHDStage);
} // namespace hccl
