/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "all_gather_mesh_mix.h"
#include "alg_template_register.h"

namespace hccl {
AllgatherMeshMix::AllgatherMeshMix(const HcclDispatcher dispatcher) : AlgTemplateBase(dispatcher) {}

AllgatherMeshMix::~AllgatherMeshMix() {}

HcclResult AllgatherMeshMix::Prepare(
    std::vector<Stream>& meshStreams, std::vector<std::shared_ptr<LocalNotify>>& meshSignal,
    std::vector<std::shared_ptr<LocalNotify>>& meshSignalAux, [[maybe_unused]] u32 userRank, HcomCollOpInfo* opInfo,
    u32 interRank, u32 interRankSize)
{
    meshStreams_ = meshStreams;
    meshSignal_ = &meshSignal;
    meshSignalAux_ = &meshSignalAux;
    interRank_ = interRank;
    interRankSize_ = interRankSize;
    opInfo_ = opInfo;
    return HCCL_SUCCESS;
}

HcclResult AllgatherMeshMix::MainRecordSub()
{
    for (u32 signalIndex = 0; signalIndex < (*meshSignalAux_).size(); signalIndex++) {
        CHK_RET(LocalNotify::Post(stream_, dispatcher_, (*meshSignalAux_)[signalIndex], profilerInput_.stage));
    }
    return HCCL_SUCCESS;
}

HcclResult AllgatherMeshMix::SubWaitMain()
{
    for (u32 streamIndex = 0; streamIndex < (*meshSignalAux_).size(); streamIndex++) {
        CHK_RET(LocalNotify::Wait(
            meshStreams_[streamIndex], dispatcher_, (*meshSignalAux_)[streamIndex], profilerInput_.stage));
    }
    return HCCL_SUCCESS;
}

HcclResult AllgatherMeshMix::MainWaitSub()
{
    for (u32 signalIndex = 0; signalIndex < (*meshSignal_).size(); signalIndex++) {
        CHK_RET(LocalNotify::Wait(stream_, dispatcher_, (*meshSignal_)[signalIndex], profilerInput_.stage));
    }
    return HCCL_SUCCESS;
}

HcclResult AllgatherMeshMix::SubRecordMain()
{
    for (u32 streamIndex = 0; streamIndex < (*meshSignal_).size(); streamIndex++) {
        CHK_RET(LocalNotify::Post(
            meshStreams_[streamIndex], dispatcher_, (*meshSignal_)[streamIndex], profilerInput_.stage));
    }
    return HCCL_SUCCESS;
}

// allgather的入口函数
HcclResult AllgatherMeshMix::RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK>& links)
{
    HCCL_INFO(
        "AllGatherMesh run: rank[%u] totalrank[%u] inputMem[%p] outputMem[%p] count[%llu]", rank, rankSize,
        inputMem_.ptr(), outputMem_.ptr(), count_);
    u32 unitSize = DataUnitSize(dataType_);
    u64 sliceSize = count_ * unitSize;         // 当前count
    u64 totalSize = opInfo_->count * unitSize; // 总输入count

    u8* curUerMemOutPtr = static_cast<u8*>(opInfo_->outputAddr);
    u8* curCommMemOutPtr = static_cast<u8*>(outputMem_.ptr());

    CHK_RET(MainRecordSub());
    CHK_RET(SubWaitMain());

    for (u32 round = 1; round < rankSize; round++) {
        u32 dstRank = BackwardRank(rank, rankSize, round);
        Stream& subStream = meshStreams_[round - 1];
        CHK_RET(links[dstRank]->TxAck(subStream));
        CHK_RET(links[dstRank]->RxAck(subStream));
    }

    CHK_RET(SubRecordMain());
    CHK_RET(MainWaitSub());

    CHK_RET(AlgTemplateBase::ExecEmptyTask(inputMem_, outputMem_, stream_, dispatcher_));

    CHK_RET(SubWaitMain());
    CHK_RET(MainRecordSub());

    DeviceMem src;
    DeviceMem dst;
    for (u32 i = 0; i < interRankSize_; i++) {
        src = DeviceMem::create(curCommMemOutPtr + (i * rankSize + rank) * sliceSize, sliceSize);
        dst = DeviceMem::create(curUerMemOutPtr + (i * rankSize + rank) * totalSize, sliceSize);
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dst, src, stream_));
    }

    for (u32 round = 1; round < rankSize; round++) {
        u32 dstRank = BackwardRank(rank, rankSize, round);
        Stream& subStream = meshStreams_[round - 1];
        // 本rank要收数据
        void* remMemPtr = nullptr;
        // 从对端的input内存拿数据，input==output也没有关系
        CHK_RET(links[dstRank]->GetRemoteMem(UserMemType::OUTPUT_MEM, &remMemPtr));

        for (u32 i = 0; i < interRankSize_; i++) {
            src = DeviceMem::create(static_cast<u8*>(remMemPtr) + (i * rankSize + dstRank) * sliceSize, sliceSize);
            dst = DeviceMem::create(curUerMemOutPtr + (i * rankSize + dstRank) * totalSize, sliceSize);
            CHK_RET(HcclD2DMemcpyAsync(
                dispatcher_, dst, src, subStream, links[dstRank]->GetRemoteRank(), links[dstRank]->GetLinkType()));
        }

        CHK_RET(links[dstRank]->TxDataSignal(subStream));
        CHK_RET(links[dstRank]->RxDataSignal(subStream));
    }
    CHK_RET(SubRecordMain());
    CHK_RET(MainWaitSub());
    CHK_RET(AlgTemplateBase::ExecEmptyTask(inputMem_, outputMem_, stream_, dispatcher_));

    HCCL_INFO("AllGatherMesh finished: rank[%u]", rank);
    return HCCL_SUCCESS;
}
REGISTER_TEMPLATE(TemplateType::TEMPLATE_ALL_GATHER_MESH_MIX, AllgatherMeshMix);
} // namespace hccl
